# Scribyte

Scribyte is a local dictation tool built around a simple loop:

```text
Hold hotkey -> record microphone -> release -> transcribe locally -> paste text
```

Scribyte runs on Linux and Windows. On Windows it prefers Intel NPU with `NPU -> GPU -> CPU` fallback. On Linux it uses `GPU -> CPU` fallback. The transcription engine is OpenVINO Whisper, kept loaded in memory at startup for low-latency responses.

Model selection is runtime-configurable through `SCRIBYTE_MODEL`. By default Scribyte uses `base`, which resolves to the root-level folder `whisper_base_ov`.

## What This Repo Does Today

- Runs a local FastAPI service that keeps the Whisper model loaded in memory.
- Records microphone audio cross-platform with `sounddevice`.
- Selects the first available input device on Linux, prefers matching WASAPI on Windows.
- Resamples captured audio to 16 kHz before transcription.
- Splits long recordings on silence before sending chunks into Whisper.
- Saves each captured recording to a debug WAV under `%TEMP%\scribyte-debug-recordings` on Windows or `$TMPDIR` on Linux.
- Provides an AutoHotkey client for hold-to-talk dictation on Windows.
- Supports `NPU -> GPU -> CPU` fallback on Windows and `GPU -> CPU` fallback on Linux.

## At A Glance

| Area | Windows | Linux |
| --- | --- | --- |
| Runtime acceleration | `NPU -> GPU -> CPU` | `GPU -> CPU` |
| Desktop hotkey client | AutoHotkey v2 | Python client under `client/linux/` |
| Audio backend | WASAPI-aware selection | ALSA/PulseAudio/PipeWire via `sounddevice` |
| Recommended server URL | `http://127.0.0.1:8000` | `http://127.0.0.1:8000` |

## Architecture

| Path | Purpose |
| --- | --- |
| `app/` | FastAPI application, routes, schemas, services, and logging |
| `scripts/` | Local hardware and audio debugging utilities |
| `tests/` | Unit tests, API tests, and integration tests |
| `docs/reference/` | Reference code for comparison, not production runtime |
| `client/` | Desktop-side helpers and hotkey clients |

The server is started via the FastAPI CLI with `fastapi run` or `fastapi dev`. The Whisper pipeline is created once at startup and reused for every request.

## Quick Start

### 1. Install prerequisites

#### Windows

Install these first:

1. `uv`: https://docs.astral.sh/uv/
2. AutoHotkey v2 if you want the Windows hold-to-talk workflow
3. Intel NPU drivers if you want to run on `NPU`

Notes:

- The project is pinned to Python 3.11.
- Newer Python versions can break parts of the OpenVINO and Whisper toolchain.
- You do not need AutoHotkey just to run the API or tests.

#### Linux

Install system audio dependencies first:

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install -y python3-dev libasound2-dev libpulse-dev

# Fedora/RHEL
sudo dnf install -y python3-devel alsa-lib-devel pulseaudio-libs-devel

# Arch
sudo pacman -S --needed python-devel alsa-lib pulseaudio
```

Then install:

1. `uv`: https://docs.astral.sh/uv/
2. Intel GPU drivers if you want GPU acceleration

### 2. Clone the repo

| Bash | PowerShell |
| --- | --- |
| `git clone <your-repo-url>`<br>`cd scribyte` | `git clone <your-repo-url>`<br>`Set-Location scribyte` |

### 3. Sync the environment

```bash
uv sync
uv sync --group dev
uv sync --group model-export
```

What they do:

| Command | Purpose |
| --- | --- |
| `uv sync` | Installs the runtime environment using the repo's pinned Python version |
| `uv sync --group model-export` | Adds the export toolchain for generating local OpenVINO Whisper models |
| `uv sync --group dev` | Adds developer tooling such as `pytest`, `pyright`, and `httpx` (Optional unless developing)|

### 4. Make sure the OpenVINO Whisper model exists

This repo expects one or more exported model directories at the repo root, such as `whisper_base_ov`, `whisper_small_ov`, or `whisper_medium_ov`.

Generate the base model:

```bash
uv run --group model-export optimum-cli export openvino --model openai/whisper-base whisper_base_ov
```

This downloads the upstream Whisper model from Hugging Face and exports it into an OpenVINO format that Scribyte can load locally.

Expected output includes a `whisper_base_ov` folder containing files such as:

```text
config.json
generation_config.json
openvino_encoder_model.xml
openvino_decoder_model.xml
openvino_tokenizer.xml
openvino_detokenizer.xml
```

Optional exports for other quality and latency tradeoffs:

```bash
uv run --group model-export optimum-cli export openvino --model openai/whisper-small whisper_small_ov
uv run --group model-export optimum-cli export openvino --model openai/whisper-medium whisper_medium_ov
```

> Note: There will be a lot of TracerWarning messages when converting the models, this is normal

### 5. Choose the model at runtime

Accepted `SCRIBYTE_MODEL` values:

- simple names such as `base`, `small`, or `medium`, which resolve to `whisper_<name>_ov`
- a full repo-root folder name such as `whisper_small_ov`
- a relative or absolute path to a custom exported model directory

| Bash | PowerShell |
| --- | --- |
| `export SCRIBYTE_MODEL=base` | `$env:SCRIBYTE_MODEL = "base"` |
| `export SCRIBYTE_MODEL=small` | `$env:SCRIBYTE_MODEL = "small"` |
| `export SCRIBYTE_MODEL=medium` | `$env:SCRIBYTE_MODEL = "medium"` |

### 6. Verify device availability

This command is the same in Bash and PowerShell:

```bash
uv run python scripts/check_device.py
```

Expected result:

- If your machine has an NPU and the required drivers are installed correctly, you should see `NPU` in the available device list.
- You should also see other available execution devices on the machine, such as `GPU` or `CPU`, depending on the hardware and runtime setup.

### 7. Start the API

Production-style local run:

```bash
uv run fastapi run --host 127.0.0.1
```

Development with auto-reload:

```bash
uv run fastapi dev
```

By default, `fastapi run` binds to `0.0.0.0`. Use `--host 127.0.0.1` when you want localhost-only access. This is important for security because this program has access to the microphone on your local machine, so you wouldn't want anyone else activating that over the network.

The app listens on `http://127.0.0.1:8000`.

### 8. Confirm the backend is healthy

After the server starts, open:

1. `http://127.0.0.1:8000/status`
2. `http://127.0.0.1:8000/docs`

`/status` reports whether the transcriber is ready, whether recording is active, the sample rate, the selected runtime device, and the debug recordings directory.

### 9. Launch a client

#### Windows AutoHotkey client

Run `client/windows/scribyte.ahk` with AutoHotkey v2.

Default script settings:

```ahk
global SCRIBYTE_API_URL := "http://127.0.0.1:8000"
global SCRIBYTE_HOLD_KEY := "F8"
global SCRIBYTE_PASTE_SHORTCUT := "^v"
```

Typical use:

1. Start the API.
2. Launch `client/windows/scribyte.ahk`.
3. Focus the app where you want text pasted.
4. Hold `F8` while speaking.
5. Release `F8` to transcribe and paste.

#### Linux hotkey client

The Linux client lives under `client/linux/scribyte_hotkey_linux.py`.

### 10. If dictation quality is poor, inspect the audio first

Every successful transcription request saves a debug WAV.

| Platform | Default location |
| --- | --- |
| Windows | `%TEMP%\scribyte-debug-recordings` |
| Linux | `$TMPDIR` or the system temp directory |

If the transcription quality is poor, listen to the saved WAV before changing model code. If the WAV sounds wrong, the likely fault boundary is microphone capture or device selection rather than Whisper itself.

## Shell Reference

This section is the fastest way to copy the right command for your shell.

| Task | Bash | PowerShell |
| --- | --- | --- |
| Clone repo | `git clone <your-repo-url> && cd scribyte` | `git clone <your-repo-url>`<br>`Set-Location scribyte` |
| Install runtime deps | `uv sync` | `uv sync` |
| Install dev deps | `uv sync --group dev` | `uv sync --group dev` |
| Install model-export deps | `uv sync --group model-export` | `uv sync --group model-export` |
| Export base model | `uv run --group model-export optimum-cli export openvino --model openai/whisper-base whisper_base_ov` | `uv run --group model-export optimum-cli export openvino --model openai/whisper-base whisper_base_ov` |
| Set model to small | `export SCRIBYTE_MODEL=small` | `$env:SCRIBYTE_MODEL = "small"` |
| Force CPU | `export SCRIBYTE_LIMIT=cpu` | `$env:SCRIBYTE_LIMIT = "cpu"` |
| Start API | `uv run fastapi run --host 127.0.0.1` | `uv run fastapi run --host 127.0.0.1` |
| Start dev server | `uv run fastapi dev` | `uv run fastapi dev` |
| Check devices | `uv run python scripts/check_device.py` | `uv run python scripts/check_device.py` |
| Run tests | `uv run pytest` | `uv run pytest` |
| Run type checks | `uv run pyright` | `uv run pyright` |

## API Summary

The current API surface is:

- `GET /status`
- `POST /start_recording`
- `POST /stop_recording_and_transcribe`

### `GET /status`

Returns runtime metadata including:

| Field | Meaning |
| --- | --- |
| `ready` | Whether the transcriber initialized successfully |
| `device` | Runtime device currently in use |
| `model_path` | Resolved model directory |
| `recording` | Whether a recording is currently active |
| `sample_rate` | Recorder sample rate |
| `startup_error` | Startup failure detail, if initialization failed |
| `debug_recordings_dir` | Folder used for saved debug WAV files |

### `POST /start_recording`

Starts microphone capture.

| Result | Meaning |
| --- | --- |
| `200` | Recording started successfully |
| `409` | A recording is already in progress |
| `503` | The transcriber failed to initialize at startup |

Successful responses include the selected input device name in `input_device`.

### `POST /stop_recording_and_transcribe`

Stops microphone capture, saves a debug WAV, and transcribes the captured audio.

| Result | Meaning |
| --- | --- |
| `200` | Recording stopped and transcription succeeded |
| `400` | The captured audio was too short to transcribe |
| `409` | No recording is active |
| `500` | Whisper transcription failed at runtime |

Successful responses include:

- `text`
- `chunk_count`
- `duration_seconds`
- `latency_seconds`
- `debug_audio_path`

## Utility Scripts

Use these when debugging setup instead of changing runtime code blindly.

### Check OpenVINO devices

```bash
uv run python scripts/check_device.py
```

### Inspect WASAPI microphone selection on Windows

| Task | Bash | PowerShell |
| --- | --- | --- |
| List WASAPI inputs | `uv run python scripts/wasapi_debug.py --list` | `uv run python scripts/wasapi_debug.py --list` |
| Record default-matching microphone | `uv run python scripts/wasapi_debug.py` | `uv run python scripts/wasapi_debug.py` |
| Record specific device index | `uv run python scripts/wasapi_debug.py --device-index 14 --seconds 8` | `uv run python scripts/wasapi_debug.py --device-index 14 --seconds 8` |

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

| Command or test | Coverage |
| --- | --- |
| `uv run pyright` | Strict type checking |
| `uv run pytest` | Fast unit tests only; integration markers excluded by default |
| `tests/test_api.py` | Hardware-free API and recorder-selection behavior |
| `tests/test_integration_npu.py` | Real NPU-backed transcription against committed fixtures |
| `tests/test_integration_gpu.py` | Real GPU-backed transcription against committed fixtures |
| `tests/test_integration_cpu.py` | Real CPU-backed transcription against committed fixtures |

The integration tests do not require exact punctuation matching. They normalize and compare the expected phrase or token overlap instead.

## Project Layout

```text
app/
  api/
  core/
  schemas/
  services/
client/
  linux/
  windows/
docs/
  reference/
scripts/
tests/
typings/
README.md
pyproject.toml
```

## Reference Code

`docs/reference/silence_chunked_whisper.py` is kept as a reference implementation for chunking and OpenVINO Whisper behavior. It is useful for comparing service behavior against the original prototype, but it is not part of the live FastAPI runtime.

## Current Limitations

- The runtime initializes the transcriber on `NPU` on Windows or `GPU` on Linux, then falls back to `CPU` when needed.
- Set `SCRIBYTE_LIMIT=cpu` or `SCRIBYTE_LIMIT=gpu` to control device selection.
- Set `SCRIBYTE_MODEL=base`, `SCRIBYTE_MODEL=small`, or another exported folder or path to control which Whisper model is loaded at startup.
- Microphone capture debugging still starts with listening to saved WAV output.
- There is no API key authentication or network-based transcription yet.

## Development Notes

- Keep API code under `app/`.
- Keep operator and hardware utilities under `scripts/`.
- Keep prototypes and historical reference code under `docs/reference/`.
- Keep audio capture in Python, not AutoHotkey.
- Start the server with `fastapi run` or `fastapi dev`.

## Current limitations

- The runtime initializes the transcriber on `NPU` (Windows) or `GPU` (Linux); falls back to `CPU` when a device is unavailable.
- Set `SCRIBYTE_LIMIT=cpu` or `SCRIBYTE_LIMIT=gpu` to control device selection.
- Set `SCRIBYTE_MODEL=base`, `SCRIBYTE_MODEL=small`, or another exported folder/path to control which Whisper model is loaded at startup.
- Microphone capture debugging still starts with listening to saved WAV output.
- No API key authentication or network-based transcription yet (design is prepared for it).

## Development notes

- Keep the API code under `app/`.
- Keep operator or hardware utilities under `scripts/`.
- Keep prototypes and historical reference code under `docs/reference/`.
- Keep audio capture in Python, not AutoHotkey.
- Start the server with `fastapi run` or `fastapi dev` — there is no separate entrypoint script.
