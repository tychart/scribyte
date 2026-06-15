# Scribyte Agent Guide

## Purpose

Scribyte is a Windows-first local dictation project built around this workflow:

```text
Hold hotkey -> record microphone -> release -> transcribe on Intel NPU -> paste text
```

The intended split is:

- FastAPI in Python is the compute plane.
- AutoHotkey v2 is the control plane.
- OpenVINO Whisper on Intel NPU is the transcription engine.

The project is optimized for low-latency repeated dictation, so the Whisper model must stay loaded in memory and must not be recreated per request.

## Current State

### Implemented

The repo already contains a functioning backend foundation.

1. UV-managed Python project setup is in place.
2. Python is pinned to 3.11.
3. Runtime, model-export, and dev dependencies are declared in `pyproject.toml`.
4. The backend follows a package-style FastAPI structure under `app/`.
5. The FastAPI app uses a lifespan handler to initialize the persistent Whisper pipeline once at startup.
6. A Python-side microphone recorder exists using `sounddevice`.
7. The API exposes:
   - `GET /status`
   - `POST /start_recording`
   - `POST /stop_recording_and_transcribe`
8. Silence-aware chunking from the original working script has been preserved in the service layer.
9. Hardware-free API tests exist and currently pass.
10. A real NPU integration test exists and currently passes against committed fixture audio.
11. The API now saves each captured microphone recording to a debug WAV file under the Windows temp directory and returns that path in the transcription response.

### Validated

The following checks have already succeeded during implementation:

1. `uv lock`
2. Python compile checks for the FastAPI package and tests
3. `uv run pytest tests/test_api.py`
4. `uv run pytest -m npu tests/test_npu_transcription.py -rs`

That means the NPU transcription path works on at least one known-good fixture and the current likely fault boundary for bad live dictation is the microphone capture path, not the Whisper NPU path itself.

## Project Structure

```text
app/
  __init__.py
  main.py
  dependencies.py
  api/
    __init__.py
    routes/
      __init__.py
      dictation.py
  core/
    __init__.py
    config.py
  schemas/
    __init__.py
    dictation.py
  services/
    __init__.py
    debug_audio.py
    recorder.py
    transcriber.py
tests/
  test_api.py
  test_npu_transcription.py
  test-audio/
    sample1.wav
    sample1.txt
main.py
check_device.py
silence_chunked_whisper.py
README.md
pyproject.toml
AGENTS.md
```

## Important Files

### Source of Truth

- `silence_chunked_whisper.py`
  - Original working script.
  - Use this as the behavioral source of truth for chunking and OpenVINO Whisper usage.

### FastAPI App

- `app/main.py`
  - App factory and lifespan wiring.
  - Loads the persistent transcriber once at startup.

- `app/api/routes/dictation.py`
  - HTTP routes for status, start recording, and stop/transcribe.

- `app/dependencies.py`
  - Shared dependency helpers for app state access.

### Services

- `app/services/transcriber.py`
  - Persistent Whisper wrapper.
  - Silence-aware chunking.
  - NPU transcription logic.

- `app/services/recorder.py`
  - Microphone capture and recorder state.

- `app/services/debug_audio.py`
  - Writes captured microphone audio to a local debug WAV file.

### Schemas and Config

- `app/schemas/dictation.py`
  - Response models.

- `app/core/config.py`
  - Central constants such as sample rate, model path, silence thresholds, and debug recording directory.

### Tests

- `tests/test_api.py`
  - Hardware-free API behavior tests.

- `tests/test_npu_transcription.py`
  - Real NPU integration test using committed fixture audio.

- `tests/test-audio/`
  - Repo-local audio fixtures for NPU transcription validation.

## Runtime and Setup Decisions

### Python Version

Use Python 3.11.

Reason:

- This stack was validated with 3.11.
- Newer Python versions can break parts of the OpenVINO and Whisper toolchain.
- Only downgrade to 3.10 if actual compatibility issues force it.

### Package Management

Use UV for everything.

Do not rely on ad hoc `pip install` commands if the dependency belongs in project metadata.

### Model Workflow

The current documented workflow is:

```powershell
uv run optimum-cli export openvino --model openai/whisper-base whisper_base_ov
```

The project assumes a local exported OpenVINO model directory named `whisper_base_ov`.

### FFmpeg

FFmpeg is a Windows prerequisite and is intentionally documented outside Python dependencies.

## API Behavior

### `GET /status`

Returns readiness and runtime metadata, including:

- whether the transcriber is ready
- selected device
- model path
- recording state
- sample rate
- startup error if initialization failed
- debug recordings directory

### `POST /start_recording`

Starts microphone capture.

Expected behavior:

- returns `409` if recording is already in progress
- returns `503` if the transcriber failed to initialize

### `POST /stop_recording_and_transcribe`

Stops capture, writes a debug WAV file, runs silence-aware chunking, and transcribes through the persistent Whisper model.

Expected behavior:

- returns `409` if recording is not active
- returns `400` if the recording is too short
- returns transcription payload including:
  - `text`
  - `chunk_count`
  - `duration_seconds`
  - `latency_seconds`
  - `debug_audio_path`

## Debug Audio Behavior

Captured microphone audio is written to an app-specific Windows temp directory using Python's tempdir:

```text
%TEMP%\scribyte-debug-recordings
```

This is the preferred location over a repo-local debug directory because:

- it avoids cluttering the repo
- it behaves like a machine-local `/tmp`
- the files are purely diagnostic artifacts

If live dictation quality is poor, inspect the returned `debug_audio_path` first. That is the fastest way to determine whether the problem is in capture or transcription.

## Testing Strategy

### Hardware-Free Tests

Use `tests/test_api.py` for:

- status response shape
- start/stop flow
- invalid state transitions
- short recording rejection

### Real NPU Tests

Use `tests/test_npu_transcription.py` for:

- real transcription quality against trusted fixture WAV files
- validation that the actual OpenVINO NPU path produces usable text

Fixture convention:

- each `*.wav` in `tests/test-audio/`
- must have a matching `*.txt`
- example: `sample1.wav` and `sample1.txt`

Recommended commands:

```powershell
uv run pytest tests/test_api.py
uv run pytest -m npu tests/test_npu_transcription.py -rs
```

## Known Findings

1. The NPU transcription path passed the real fixture test.
2. Therefore, a live API result like `"you"` from a long utterance is more likely caused by microphone capture quality or input-device behavior than by Whisper on NPU.
3. The next debugging step for bad live transcription should start with listening to the saved debug WAV file.

## Big Picture Plan

### Phase 0: Foundation

Completed or largely completed:

1. Pin Python 3.11.
2. Manage dependencies in `pyproject.toml`.
3. Document the validated UV-based Windows setup in `README.md`.
4. Preserve the original working script as a reference implementation.

### Phase 1: Backend MVP

Completed or largely completed:

1. Extract reusable transcription logic into modules.
2. Build a persistent FastAPI transcriber service using lifespan.
3. Add Python-side microphone recording.
4. Implement the MVP API contract.
5. Add hardware-free tests.
6. Add NPU-backed fixture tests.
7. Add debug WAV dumping to inspect captured audio.

### Phase 1 Remaining

Still to do:

1. Real-world manual validation across more than one utterance length.
2. Investigate microphone capture quality if debug WAVs sound bad.
3. Verify behavior across different input devices if needed.

### Phase 2: AutoHotkey Integration

Planned:

1. Add an AutoHotkey v2 script.
2. Bind a hold-to-talk hotkey.
3. Call `POST /start_recording` on key down.
4. Call `POST /stop_recording_and_transcribe` on key up.
5. Show toasts for recording, transcribing, success, and failure.
6. Copy the returned text to the clipboard and paste with `Ctrl+V`.

Important rule:

- AHK should never handle audio capture directly.

### Phase 2: Hardening

Planned:

1. Improve diagnostics and error payloads.
2. Add clearer logging around startup, device readiness, and capture failures.
3. Add optional microphone device selection.
4. Add fallback device strategy if NPU is unavailable, likely `NPU -> GPU -> CPU`.
5. Improve README operator guidance for daily use.

### Phase 2: Configuration and Packaging

Planned:

1. Keep central settings in the config layer.
2. Document expected model layout and run commands.
3. Optionally add launcher tasks or helper scripts.

### Phase 3: Explicitly Out of MVP

Do not prioritize these until the hold-to-talk path is stable:

1. Streaming partial transcription
2. LLM cleanup passes
3. Tray app packaging
4. Startup-on-login automation
5. Alternate IPC layers beyond local HTTP

## Guidance for Future Changes

1. Preserve the persistent model lifecycle. Do not move `WhisperPipeline` creation into request handlers.
2. Keep audio capture in Python, not in AHK.
3. Reuse the silence-aware chunking behavior from the original working script unless there is a deliberate, tested reason to change it.
4. Validate narrow behavior after edits:
   - compile checks for touched modules
   - hardware-free tests for API behavior
   - NPU fixture test when touching transcription behavior
5. When debugging live transcription quality, inspect the debug WAV before changing model logic.
6. Prefer adding settings and structure inside the `app/` package instead of growing flat root modules.

## Recommended Immediate Next Steps

1. Use the returned `debug_audio_path` from a bad dictation run and listen to the file.
2. If the file sounds wrong, debug the recorder or microphone device path next.
3. If the file sounds clean but transcription is still poor, inspect preprocessing and chunking around live-recorded audio.
4. After backend behavior is stable, implement the AutoHotkey hold-to-talk script.
