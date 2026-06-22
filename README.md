# Scribyte

Scribyte is a local dictation tool built around a simple loop:

```text
Hold hotkey -> record microphone -> release -> transcribe locally -> paste text
```

Scribyte runs on Windows and Linux. On Windows it prefers Intel NPU with `NPU -> GPU -> CPU` fallback. On Linux it uses `GPU -> CPU` fallback. The transcription engine is OpenVINO Whisper, kept loaded in memory at startup for low-latency responses.

## What the repo does today

- Runs a local FastAPI service that keeps the Whisper model loaded in memory.
- Records microphone audio cross-platform with `sounddevice`.
- Selects the first available input device on Linux, prefers matching WASAPI on Windows.
- Resamples captured audio to 16 kHz before transcription.
- Splits long recordings on silence before sending chunks into Whisper.
- Saves each captured recording to a debug WAV under `%TEMP%\scribyte-debug-recordings` (Windows) or `$TMPDIR` (Linux).
- Provides an AutoHotkey client in `scribyte.ahk` for hold-to-talk dictation on Windows.
- Supports `NPU -> GPU -> CPU` fallback on Windows, `GPU -> CPU` fallback on Linux.

## Architecture

- `app/`: FastAPI application, routes, dependencies, schemas, services, and centralized logging config.
- `scripts/`: local utility scripts for hardware checks and recorder debugging.
- `tests/`: API tests, NPU-backed transcription fixture tests, and integration test markers.
- `docs/reference/`: reference code kept for comparison, not production runtime.
- `scribyte.ahk`: Windows desktop hotkey client for the local API.

The server is started exclusively via the FastAPI CLI (`fastapi run` or `fastapi dev`). The Whisper pipeline is created once at startup and reused for every request — do not move model initialization into request handlers unless you are intentionally changing the latency model.

## Getting started on Windows

This section is the shortest path from a clean Windows machine to a working local dictation setup.

### 1. Install prerequisites

Install these first:

1. Python package manager `uv`: https://docs.astral.sh/uv/
2. AutoHotkey v2 if you want the desktop hold-to-talk workflow
3. Intel NPU drivers if you want to run on `NPU`

Notes:

- The project is pinned to Python 3.11.
- Newer Python versions can break parts of the OpenVINO and Whisper toolchain.
- If you only want to inspect the code or run some non-hardware tests, you can do that without AutoHotkey.

### 2. Clone the repo

```powershell
git clone <your-repo-url>
cd scribyte
```

### 3. Sync the project environment

```powershell
uv sync
uv sync --group dev
uv sync --group model-export
```

What those commands mean:

- `uv sync`: installs the runtime environment for this project and uses the repo's pinned Python version.
- `uv sync --group dev`: also installs the development tools, including `pytest`, `pyright`, and `httpx`.
- `uv sync --group model-export`: also installs the heavier model-export toolchain used to generate `whisper_base_ov` locally if it doesn't already exist.

In most cases, users do not need to install Python separately first. UV can download and manage the pinned interpreter automatically.

You would only need an explicit Python install first if UV-managed Python downloads are disabled in your environment, or if you are working offline.

If you want to install the interpreter explicitly anyway, you can still do:

```powershell
uv python install 3.11
uv venv --python 3.11
uv sync
```

### 4. Make sure the Whisper OpenVINO model exists

This repo expects a local exported model directory named `whisper_base_ov`.

If you do not already have it (likely), generate it by running the following command on the project's root directory:

```powershell
uv run --group model-export optimum-cli export openvino --model openai/whisper-base whisper_base_ov
```

This will pull down the openai whisper model from hugging face and convert the format to something that can be run by an NPU.

Expected output is a folder named `whisper_base_ov` containing files such as:

```text
config.json
generation_config.json
openvino_encoder_model.xml
openvino_decoder_model.xml
openvino_tokenizer.xml
openvino_detokenizer.xml
```

### 5. Verify that OpenVINO can see your devices

Run:

```powershell
uv run python scripts/check_device.py
```

If your machine is configured correctly for Intel NPU, you should see `NPU` in the printed device list.

### 6. Start the local API

Use the FastAPI CLI:

```powershell
uv run fastapi run --host 127.0.0.1
```

Development (auto-reload):

```powershell
uv run fastapi dev
```

By default `fastapi run` binds to `0.0.0.0`. Use `--host 127.0.0.1` when you want the server exposed only locally.

The app listens on `http://127.0.0.1:8000`.

### 7. Confirm the backend is healthy

Open these in your browser after the server starts:

1. `http://127.0.0.1:8000/status`
2. `http://127.0.0.1:8000/docs`

`/status` should report whether the transcriber is ready, whether a recording is active, the sample rate, the selected runtime device, and the debug recordings directory.

### 8. Launch the AutoHotkey client

Run `scribyte.ahk` with AutoHotkey v2.

Default script settings:

```ahk
global SCRIBYTE_API_URL := "http://127.0.0.1:8000"
global SCRIBYTE_HOLD_KEY := "F8"
global SCRIBYTE_PASTE_SHORTCUT := "^v"
```

Typical use:

1. Start the API.
2. Launch `scribyte.ahk`.
3. Focus the app where you want text pasted.
4. Hold `F8` while speaking.
5. Release `F8` to transcribe and paste.

The tray menu also includes `Check Scribyte Status` for a quick readiness check.

### 9. If dictation quality is poor, inspect the captured audio first

Every successful transcription request saves a debug WAV file under:

```text
%TEMP%\scribyte-debug-recordings
```

If the transcription quality is bad, listen to the saved WAV before changing model code. If the audio sounds wrong there, the problem is likely capture or device selection rather than Whisper itself.

## Getting started on Linux

### 1. Install system dependencies

Install ALSA development headers and PulseAudio/PipeWire libraries:

```bash
# Debian/Ubuntu
sudo apt update && sudo apt install -y python3-dev libasound2-dev libpulse-dev

# Fedora/RHEL
sudo dnf install -y python3-devel alsa-lib-devel pulseaudio-libs-devel

# Arch
sudo pacman -S --needed python-devel alsa-lib pulseaudio
```

### 2. Install prerequisites

1. Python package manager `uv`: https://docs.astral.sh/uv/
2. Intel GPU drivers (if you want GPU acceleration)

Notes:

- The project is pinned to Python 3.11.
- Newer Python versions can break parts of the OpenVINO and Whisper toolchain.
- On Linux the app uses `GPU -> CPU` fallback (no NPU support yet).

### 3. Clone the repo

```bash
git clone <your-repo-url>
cd scribyte
```

### 4. Sync the project environment

```bash
uv sync
uv sync --group dev
uv sync --group model-export
```

### 5. Make sure the Whisper OpenVINO model exists

```bash
uv run --group model-export optimum-cli export openvino --model openai/whisper-base whisper_base_ov
```

### 6. Verify that OpenVINO can see your devices

```bash
uv run python scripts/check_device.py
```

On Linux, you should see `GPU` and/or `CPU` in the printed device list.

### 7. Start the local API

```bash
uv run fastapi run --host 127.0.0.1
```

Development (auto-reload):

```bash
uv run fastapi dev
```

By default `fastapi run` binds to `0.0.0.0`. Use `--host 127.0.0.1` for localhost-only access.

### 8. Confirm the backend is healthy

Open in your browser:

1. `http://127.0.0.1:8000/status`
2. `http://127.0.0.1:8000/docs`

### 9. Use the audio debug script

```bash
uv run python scripts/audio_device_debug.py --list
uv run python scripts/audio_device_debug.py --device-index 0 --seconds 5
```

## API summary

The current API surface is:

- `GET /status`
- `POST /start_recording`
- `POST /stop_recording_and_transcribe`

### `GET /status`

Returns runtime metadata including:

- `ready`
- `device`
- `model_path`
- `recording`
- `sample_rate`
- `startup_error`
- `debug_recordings_dir`

### `POST /start_recording`

Starts microphone capture.

Current behavior:

- Returns `200` on success
- Returns `409` if a recording is already in progress
- Returns `503` if the transcriber failed to initialize at startup
- Returns the selected input device name in `input_device`

### `POST /stop_recording_and_transcribe`

Stops microphone capture, saves a debug WAV, and transcribes the captured audio.

Current behavior:

- Returns `200` on success
- Returns `409` if no recording is active
- Returns `400` if the captured audio is too short to transcribe
- Returns `500` if Whisper transcription fails at runtime

The response includes:

- `text`
- `chunk_count`
- `duration_seconds`
- `latency_seconds`
- `debug_audio_path`

## Utility scripts

Use these when debugging local setup instead of changing app code blindly.

### Check OpenVINO devices

```powershell
uv run python scripts/check_device.py
```

### Inspect WASAPI microphone selection (Windows)

List detected WASAPI inputs:

```powershell
uv run python scripts/wasapi_debug.py --list
```

Record a short sample from the default-matching WASAPI microphone:

```powershell
uv run python scripts/wasapi_debug.py
```

Record from a specific WASAPI device index:

```powershell
uv run python scripts/wasapi_debug.py --device-index 14 --seconds 8
```

The script writes both a raw capture and a prepared 16 kHz WAV into a temp debug directory so you can compare input quality before and after preprocessing.

### Cross-platform audio device listing and recording

```bash
uv run python scripts/audio_device_debug.py --list
uv run python scripts/audio_device_debug.py --device-index 0 --seconds 5
```

## Testing

Recommended checks:

```bash
uv run pyright
uv run pytest
uv run pytest tests/test_api.py
uv run pytest -m integration
```

What they cover:

- `uv run pyright`: strict type checking
- `uv run pytest`: fast unit tests only (default, excludes integration markers)
- `tests/test_api.py`: hardware-free API and recorder-selection behavior
- `tests/test_integration_npu.py`: real NPU-backed transcription against committed fixtures (`@pytest.mark.integration`)
- `tests/test_integration_gpu.py`: real GPU-backed transcription (`@pytest.mark.integration`)
- `tests/test_integration_cpu.py`: real CPU-backed transcription (`@pytest.mark.integration`)

The integration tests do not require exact punctuation matching. They normalize and compare the expected phrase or token overlap instead.

## Project layout

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
pyproject.toml
```

## Reference code

`docs/reference/silence_chunked_whisper.py` is kept as a reference implementation for chunking and OpenVINO Whisper behavior. It is useful when comparing service behavior against the original working prototype, but it is not part of the live FastAPI runtime.

## Current limitations

- The runtime initializes the transcriber on `NPU` (Windows) or `GPU` (Linux); falls back to `CPU` when a device is unavailable.
- Set `SCRIBYTE_LIMIT=cpu` or `SCRIBYTE_LIMIT=gpu` to control device selection.
- Microphone capture debugging still starts with listening to saved WAV output.
- No API key authentication or network-based transcription yet (design is prepared for it).

## Development notes

- Keep the API code under `app/`.
- Keep operator or hardware utilities under `scripts/`.
- Keep prototypes and historical reference code under `docs/reference/`.
- Keep audio capture in Python, not AutoHotkey.
- Start the server with `fastapi run` or `fastapi dev` — there is no separate entrypoint script.
