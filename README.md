# Scribyte

Scribyte is a Windows-first (Purely becuse my machine's NPU dosn't work through WSL) local dictation tool built around a simple loop:

```text
Hold hotkey -> record microphone -> release -> transcribe locally -> paste text
```

Today the repo is optimized for Windows, FastAPI, AutoHotkey v2, and OpenVINO Whisper on Intel NPU. Linux compatibility and whisper processing fallback (from NPU -> GPU -> CPU is a future goal), but the current setup and support guidance in this document are Windows-specific.

## What the repo does today

- Runs a local FastAPI service that keeps the Whisper model loaded in memory.
- Records microphone audio in Python with `sounddevice`.
- Prefers a matching Windows WASAPI input device when choosing the microphone.
- Resamples captured audio to 16 kHz before transcription.
- Splits long recordings on silence before sending chunks into Whisper.
- Saves each captured recording to a debug WAV under `%TEMP%\scribyte-debug-recordings`.
- Provides an AutoHotkey client in `scribyte.ahk` for hold-to-talk dictation and paste.

## Architecture

- `app/`: FastAPI application, routes, dependencies, schemas, and services.
- `scripts/`: local utility scripts for hardware checks and recorder debugging.
- `tests/`: API tests and NPU-backed transcription fixture tests.
- `docs/reference/`: reference code kept for comparison, not production runtime.
- `scribyte.ahk`: Windows desktop hotkey client for the local API.

The important design rule is that the Whisper pipeline is created once at startup and reused for every request. Do not move model initialization into request handlers unless you are intentionally changing the latency model.

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
- `uv sync --group model-export`: also installs the heavier model-export toolchain used to generate `whisper_base_ov` locally if it dosn't already exist.

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

Use the configured FastAPI entrypoint:

Production (localhost only):
```powershell
uv run fastapi run --host 127.0.0.1
```

Development

```powershell
uv run fastapi dev
```

Alternative:

```powershell
uv run python -m app.main
```

The plain `fastapi run` command binds to `0.0.0.0` by default, so use the explicit `--host 127.0.0.1` flag above when you want the production server exposed only to the local machine.

With the commands above, the app listens on `http://127.0.0.1:8000`.

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

### Inspect WASAPI microphone selection

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

## Testing

Recommended checks:

```powershell
uv run pyright
uv run pytest
uv run pytest tests/test_api.py
uv run pytest -m npu tests/test_npu_transcription.py -rs
```

What they cover:

- `uv run pyright`: strict type checking
- `uv run pytest`: full test suite
- `tests/test_api.py`: hardware-free API and recorder-selection behavior
- `tests/test_npu_transcription.py`: real NPU-backed transcription against committed fixtures

The NPU fixture test does not require exact punctuation matching. It normalizes and compares the expected phrase or token overlap instead.

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

- Windows is the only documented and supported workflow right now.
- The runtime currently initializes the transcriber on `NPU` in `app/main.py`.
- There is no device fallback chain yet.
- Microphone capture debugging still starts with listening to saved WAV output.

## Development notes

- Keep the API code under `app/`.
- Keep operator or hardware utilities under `scripts/`.
- Keep prototypes and historical reference code under `docs/reference/`.
- Keep audio capture in Python, not AutoHotkey.
