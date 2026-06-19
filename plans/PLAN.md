# Plan: Cross-Platform (Linux/Windows) Support with Modern Architecture

## Context

Scribyte is a Windows-first local dictation tool using OpenVINO Whisper on Intel NPU. The current codebase is tightly coupled to Windows WASAPI device discovery and hardcodes NPU device priority. We need to:

1. **Add Linux support** with GPU -> CPU fallback (Windows keeps NPU -> GPU -> CPU).
2. **Unify device discovery** under a single `sounddevice` path — no WASAPI-specific filtering needed since `sd.query_devices()` works on both platforms.
3. **Improve startup logging** to clearly report which device was selected and loaded.
4. **Apply modern best practices** — proper test organization, test markers for slow/integration tests, clean architecture that won't conflict with future network-based transcription endpoints.

## Approach

### 1. Device Discovery — Unified sounddevice

Replace the WASAPI-specific `pick_wasapi_input_device()` with a single `pick_input_device()` that uses generic `sounddevice` on **both** platforms. There is no reason to keep WASAPI-specific filtering — `sounddevice` (via PortAudio) already reports WASAPI devices correctly on Windows, and `sd.default.input` gives the system default on all platforms.

```python
def pick_input_device(fallback_sample_rate: int, device_index: int | None = None) -> InputDeviceSelection:
    """Unified device selection using sounddevice. Works on Windows and Linux."""
    devices = list_input_devices()  # calls sd.query_devices(), filters for inputs
    if device_index is not None:
        for d in devices:
            if d.index == device_index:
                return d.to_selection()
        raise RecorderStateError(f"Device {device_index} not found")
    # Prefer system default input, fall back to first available
    default = sd.default.input
    for d in devices:
        if isinstance(default, int) and d.index == default:
            return d.to_selection()
    return devices[0].to_selection()
```

This eliminates all platform-specific device discovery branches. The `sd.default.input` is populated by the OS audio subsystem on both Windows (WASAPI) and Linux (ALSA/PulseAudio/PipeWire).

### 2. Recorder Refactoring

- `RecorderState` in `recorder_sounddevice.py` will call `pick_input_device()` from `recorder_devices.py`.
- `recorder_devices.py` will be simplified to a single `list_input_devices()` and `pick_input_device()` using `sd.query_devices()` directly.
- The old WASAPI-specific functions (`WasapiInputDevice`, `build_wasapi_input_devices`, `list_wasapi_input_devices`, `choose_wasapi_input_device`) are removed from production code (they were over-engineering).
- `recorder.py` (the re-export layer) will export the unified `pick_input_device`.

### 3. Startup Logging Improvements

In `app/main.py`'s `lifespan` function, the startup log already exists but is minimal. We'll enhance it with:

- Explicit logging of **each fallback attempt** with device name, success/failure, and error message.
- A final **"Selected device: X (runtime: Y)"** summary line.
- The `startup_log` list already gets stored in `app.state.startup_log` and exposed via `/status` — we keep this and enrich it.
- Python `logging` module usage with `logging.getLogger("scribyte.startup")` — already present, just enrich the messages.

### 4. Test Architecture

Current tests are all in two files (`test_api.py` and `test_npu_transcription.py`). We'll reorganize following FastAPI best practices with proper separation of concerns:

| Test File | Coverage | Mark | Speed |
|-----------|----------|------|-------|
| `tests/test_api.py` | Fast API/HTTP-level tests: status endpoint, start/stop recording flow, error cases (409, 400, 503), double-start protection | `pytest` (default) | <1s |
| `tests/test_recorder.py` | Recorder state management, chunk collection, stop-and-return behavior | `pytest` (default) | <1s |
| `tests/test_recorder_devices.py` | Platform device selection, `pick_input_device()` fallback logic, audio preprocessing | `pytest` (default) | <1s |
| `tests/test_transcriber.py` | Transcriber fallback chain (NPU->GPU->CPU), silence-aware chunking, warmup behavior, short audio rejection | `pytest` (default) | <1s |
| `tests/test_integration_npu.py` | Full NPU-backed transcription on real hardware fixture audio | `@pytest.mark.integration` | 30s-60s |
| `tests/test_integration_gpu.py` | Full GPU-backed transcription on real hardware fixture audio | `@pytest.mark.integration` | 30s-60s |
| `tests/test_integration_cpu.py` | Full CPU-backed transcription on real hardware fixture audio | `@pytest.mark.integration` | 30s-60s |

Integration tests are **not run by default**. They are only triggered explicitly with:
```bash
uv run pytest -m integration
```
This keeps local iteration fast (<1s) while still allowing full hardware validation when needed.

### 5. Scripts

- **`scripts/audio_device_debug.py`**: Cross-platform audio debugging script (replaces `wasapi_debug.py`). Uses `sd.query_devices()` directly — works identically on both Linux and Windows.

## Files to Modify

| File | Change |
|------|--------|
| `app/services/recorder_devices.py` | Simplified: single `list_input_devices()` + `pick_input_device()` using `sd.query_devices()` |
| `app/services/recorder_sounddevice.py` | Update `RecorderState._resolve_input_device()` to call `pick_input_device()` |
| `app/services/recorder.py` | Export `pick_input_device` alongside existing re-exports |
| `app/main.py` | Enhance startup logging with richer device selection info |
| `app/core/config.py` | Minor updates for cross-platform constants |
| `tests/test_api.py` | Keep, but improve coverage and add more edge cases |
| `tests/test_recorder.py` | **NEW** — recorder state and stop/start tests |
| `tests/test_recorder_devices.py` | **NEW** — device selection and audio preprocessing tests |
| `tests/test_transcriber.py` | **NEW** — fallback chain and chunking tests |
| `tests/test_integration_npu.py` | Renamed from `test_npu_transcription.py`, marked with `@pytest.mark.integration` |
| `tests/test_integration_gpu.py` | **NEW** — GPU integration test (same fixture, device="GPU") |
| `tests/test_integration_cpu.py` | **NEW** — CPU integration test (same fixture, device="CPU") |
| `scripts/audio_device_debug.py` | **NEW** — cross-platform audio debug script (replaces `wasapi_debug.py`) |

## Reuse

- `WhisperTranscriber` already has internal NPU->GPU->CPU fallback in `__init__` — no changes needed here.
- `_determine_device_order()` in `main.py` already supports device limit via CLI args or env var.
- `InputDeviceSelection` dataclass in `recorder_devices.py` — reuse as-is.
- `RecorderStateError` — reuse as-is.
- `prepare_audio()` — reuse as-is (works on any platform).
- `debug_audio.py` — reuse as-is (uses `soundfile` which is cross-platform).
- `silence_aware_chunks()` — reuse as-is.
- `Transcriber` Protocol — reuse as-is.
- All Pydantic schemas in `schemas/dictation.py` — reuse as-is.
- `dependencies.py` — reuse as-is (no changes needed).
- `scripts/check_device.py` — reuse as-is (OpenVINO Core works on Linux too).

## Steps

- [ ] **Step 1**: Simplify `recorder_devices.py` — remove WASAPI-specific filtering, add single `list_input_devices()` (wraps `sd.query_devices()`) and `pick_input_device()` (uses `sd.default.input` + first-available fallback).
- [ ] **Step 2**: Update `RecorderState._resolve_input_device()` in `recorder_sounddevice.py` to call `pick_input_device()`.
- [ ] **Step 3**: Update `recorder.py` to export `pick_input_device`.
- [ ] **Step 4**: Enhance startup logging in `main.py` — enrich messages with device index, name, and runtime device info.
- [ ] **Step 5**: Create new test files: `tests/test_recorder.py`, `tests/test_recorder_devices.py`, `tests/test_transcriber.py`.
- [ ] **Step 6**: Move `test_npu_transcription.py` to `tests/test_integration_npu.py` and mark with `@pytest.mark.integration`.
- [ ] **Step 7**: Create `tests/test_integration_gpu.py` and `tests/test_integration_cpu.py` with `@pytest.mark.integration` markers.
- [ ] **Step 8**: Update `test_api.py` with more edge-case coverage (short recording, empty audio, concurrent access).
- [ ] **Step 9**: Create `scripts/audio_device_debug.py` — cross-platform audio debug.
- [ ] **Step 10**: Run `uv run pyright` and `uv run pytest` to verify no regressions.
- [ ] **Step 11**: Update `README.md` and `AGENTS.md` with Linux setup instructions.

## Verification

### Functional tests (no hardware needed)
```bash
uv run pyright
uv run pytest
uv run pytest tests/test_api.py
```
- These tests use `FakeTranscriber` and `FakeRecorder`, so they pass regardless of platform.
- The `FakeRecorder` uses `"Test Microphone [Windows WASAPI]"` — this is fine since the test doesn't actually open audio hardware.

### Integration tests (requires hardware)
```bash
uv run pytest -m integration
```
This runs all three integration tests (NPU, GPU, CPU) against fixture audio. Each takes 30s-60s.

### Linux-specific verification
```bash
# On Linux, start the server and check startup logs show GPU->CPU fallback path
uv run fastapi run --host 127.0.0.1
# Check /status for the selected device
curl http://127.0.0.1:8000/status
# Test recording flow
curl -X POST http://127.0.0.1:8000/start_recording
curl -X POST http://127.0.0.1:8000/stop_recording_and_transcribe
```
