# Scribyte Agent Guide

## Purpose

Scribyte is a local dictation project built around this flow:

```text
Hold hotkey -> record microphone -> release -> transcribe locally -> paste text
```

Current architecture split:

- FastAPI in Python is the runtime and compute plane.
- AutoHotkey v2 is the desktop control plane (Windows only).
- OpenVINO Whisper is the transcription engine.

The runtime is optimized for repeated low-latency dictation, so the Whisper model must remain loaded in memory and must not be recreated per request.

## Supported Platforms

- **Windows**: `NPU -> GPU -> CPU` fallback. WASAPI-first microphone selection.
- **Linux**: `GPU -> CPU` fallback. ALSA/PulseAudio/PipeWire microphone selection via `sounddevice`.

## Current State

What is implemented now:

1. UV-managed Python project setup with Python pinned to 3.11.
2. A package-style FastAPI backend under `app/`.
3. Startup-time Whisper initialization through the FastAPI lifespan hook.
4. Cross-platform microphone capture using `sounddevice` (generic device selection).
5. Unified input device selection: WASAPI-aware on Windows, first-available on Linux.
6. Audio preprocessing that resamples captured input to 16 kHz.
7. Silence-aware chunking in the transcription path.
8. Debug WAV capture to `%TEMP%\scribyte-debug-recordings` (Windows) or `$TMPDIR` (Linux).
9. A local AutoHotkey hold-to-talk client in `scribyte.ahk` (Windows).
10. Hardware-free API tests plus NPU/GPU/CPU-backed fixture tests with integration markers.

## Validated Checks

These commands have succeeded in this repo state:

1. `uv run pyright`
2. `uv run pytest` (63 passed, 3 deselected integration tests)
3. `uv run pytest tests/test_api.py`
4. `uv run pytest -m integration` (when hardware is available)

Implication:

- If the NPU fixture test passes but live dictation quality is poor, the likely fault boundary is microphone capture or device selection, not the basic Whisper path.

## Repository Layout

```text
app/
  api/
  core/
  schemas/
  services/
docs/
  reference/
scripts/
tests/
typings/
scribyte.ahk
README.md
AGENTS.md
pyproject.toml
```

Layout rules:

- Keep FastAPI runtime code in `app/`.
- Keep developer or hardware-debug scripts in `scripts/`.
- Keep prototype or historical reference code in `docs/reference/`.
- Keep `scribyte.ahk` at the repo root for operator convenience.

## Important Files

### Runtime entrypoints

- `app/main.py`
  - App factory and FastAPI lifespan wiring.
  - Initializes the transcriber once on startup with platform-specific device preference.

- `app/api/routes/dictation.py`
  - Owns `GET /status`, `POST /start_recording`, and `POST /stop_recording_and_transcribe`.

- `app/dependencies.py`
  - Owns state access and `503` behavior when the transcriber is not ready.

### Recorder path

- `app/services/recorder_sounddevice.py`
  - Live `sounddevice` recorder state and stream lifecycle.

- `app/services/recorder_devices.py`
  - Cross-platform device discovery using `sd.query_devices()`, `pick_input_device()` for selection.

- `app/services/recorder_audio.py`
  - Audio cleanup and resampling helpers.

- `app/services/debug_audio.py`
  - Writes captured microphone audio to a temp WAV file for diagnosis.

### Transcription path

- `app/services/transcriber.py`
  - Persistent Whisper wrapper and silence-aware chunking.
  - Handles `NPU -> GPU -> CPU` fallback (Windows) or `GPU -> CPU` fallback (Linux).

- `docs/reference/silence_chunked_whisper.py`
  - Original prototype/reference implementation.
  - Use it for behavioral comparison, not as live runtime code.

### Operator utilities

- `scripts/check_device.py`
  - Prints the OpenVINO device list.

- `scripts/wasapi_debug.py`
  - Lists WASAPI inputs and records debug WAV samples (Windows-specific).

- `scripts/audio_device_debug.py`
  - Cross-platform device listing and audio recording for debugging.

### Client

- `scribyte.ahk`
  - Hotkey-driven local desktop client for the API (Windows).

## Runtime Behavior

### Startup

On startup the app:

1. Creates the FastAPI app.
2. Determines the device order based on platform:
   - Windows: `NPU -> GPU -> CPU`
   - Linux: `GPU -> CPU`
3. Attempts to initialize `WhisperTranscriber` with the preferred device.
4. Warms the transcriber once.
5. Creates a recorder state object using the transcriber sample rate when available.

If transcriber initialization fails, the server can still start, but transcription-dependent endpoints will return `503` and `/status` will expose `startup_error`.

Startup logs report:

- Device selection order
- Runtime device name (index and description)
- Model path
- Any fallback failures with reasons

### API contract

The live API surface is:

- `GET /status`
- `POST /start_recording`
- `POST /stop_recording_and_transcribe`

`GET /status` returns readiness, runtime device, model path, recording state, sample rate, startup error, startup log, and debug recordings directory.

`POST /start_recording`:

- starts microphone capture
- returns `409` if already recording
- returns `503` if the transcriber is not ready
- returns the selected input device name

`POST /stop_recording_and_transcribe`:

- stops capture
- rejects inactive recording state with `409`
- rejects very short captures with `400`
- saves a debug WAV before transcription
- returns text, chunk count, duration, latency, and debug WAV path

## Testing Strategy

Use these checks after code changes:

```bash
uv run pyright
uv run pytest
uv run pytest tests/test_api.py
uv run pytest -m integration
```

What they mean:

- `tests/test_api.py` covers API behavior and recorder-device selection logic without hardware.
- `tests/test_recorder.py` covers RecorderState lifecycle and thread safety.
- `tests/test_recorder_devices.py` covers device listing and selection.
- `tests/test_transcriber.py` covers silence-aware chunking and transcriber fallback logic.
- `tests/test_integration_npu.py`, `tests/test_integration_gpu.py`, `tests/test_integration_cpu.py` validate real transcription quality against committed WAV fixtures on each hardware tier.
- All integration tests are marked with `@pytest.mark.integration` and are excluded from default `pytest` runs.

Validation rule:

- After code changes, run at least `uv run pyright` and `uv run pytest` before considering the work done.
- If you changed transcription behavior, also run the integration tests when the environment supports it.
- Integration tests require the `whisper_base_ov` model to be present.

## Debugging Guidance

When live transcription quality is poor:

1. Inspect the `debug_audio_path` returned by the API.
2. Listen to the saved WAV under `%TEMP%\scribyte-debug-recordings` (Windows) or `$TMPDIR` (Linux).
3. If the audio sounds wrong, debug capture or device selection before changing model logic.
4. Use `scripts/audio_device_debug.py --list` to inspect available input devices.
5. Use `scripts/audio_device_debug.py --device-index 0 --seconds 5` to record and compare input quality.

## Constraints and Invariants

1. Preserve the persistent model lifecycle.
2. Keep audio capture in Python, not in AutoHotkey.
3. Reuse the current silence-aware chunking behavior unless there is a deliberate tested reason to change it.
4. Prefer adding runtime code under `app/` instead of adding new root-level Python modules.
5. Keep reference-only code out of the live import path.
6. Do not hardcode WASAPI-specific logic; use generic `sounddevice` primitives.
7. Design the API cleanly for future network-based transcription endpoints and API key authentication.

## Near-Term Gaps

These are still open and should be documented as present limitations, not hidden assumptions:

1. AutoHotkey client is Windows-only.
2. The runtime hardcodes device preference per platform (`NPU` on Windows, `GPU` on Linux).
3. Optional microphone device selection is not yet exposed through the API.
4. No API key authentication or network-based transcription yet.

## Documentation Ownership

When the runtime behavior changes, keep these files aligned:

1. `README.md` for user setup and operations.
2. `AGENTS.md` for repository conventions and maintenance guidance.
3. `docs/reference/README.md` for the role of archived reference code.
