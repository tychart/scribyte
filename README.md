# Scribyte

Windows-first local dictation using OpenVINO Whisper with Intel NPU acceleration.

## Python version

Use Python 3.11 for this project.

Newer Python versions can break parts of the OpenVINO and Whisper tooling stack. This repo is pinned to 3.11 on purpose.

## Prerequisites

1. Install `uv`.
2. Install FFmpeg on Windows and add its `bin` directory to `PATH`.
3. Make sure Intel NPU drivers are installed if you want to run on `NPU`.

## Fresh install

Create or sync the environment with UV:

```powershell
uv python install 3.11
uv sync
uv sync --group model-export
uv sync --group dev
```

If you prefer to recreate the venv explicitly:

```powershell
uv venv --python 3.11
uv sync
uv sync --group model-export
uv sync --group dev
```

## Convert Whisper to OpenVINO IR

Export the base Whisper model into the local `whisper_base_ov` directory:

```powershell
uv run --group model-export optimum-cli export openvino --model openai/whisper-base whisper_base_ov
```

`optimum-cli` is exposed by the `optimum` dependency that comes in through `optimum-intel` in the `model-export` group. It is not a package you add directly with `uv add optimum-cli`.

Expected output is a folder like:

```text
whisper_base_ov/
	encoder.xml
	decoder.xml
	config.json
	generation_config.json
	openvino_tokenizer.xml
	openvino_detokenizer.xml
```

## Verify OpenVINO devices

Create a quick check script or use the one in this repo once it exists:

```python
from openvino import Core

core = Core()
print(core.available_devices)
```

Run it with:

```powershell
uv run python check_device.py
```

On a machine with the Intel NPU exposed correctly, you should see `NPU` in the output.

## Audio notes

- Whisper input should be 16 kHz mono audio.
- FFmpeg is required for some audio decoding workflows.
- The dictation service will record in Python; AutoHotkey should not handle audio capture.

## Current reference script

The current working transcription reference is `silence_chunked_whisper.py`.

It proves:

- OpenVINO Whisper runs on this machine
- silence-aware chunking works
- the NPU call path is valid

That file is the source of truth while the FastAPI service is being built.

## Planned app workflow

The target workflow is:

```text
Hold hotkey -> record microphone -> release -> transcribe on NPU -> paste text
```

Architecture split:

- Python / FastAPI: audio capture, persistent Whisper model, transcription
- AutoHotkey v2: hotkeys, toasts, API calls, paste

## FastAPI project structure

The backend follows FastAPI's recommended bigger-application layout, with a package entrypoint, routers, and service modules:

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
		recorder.py
		transcriber.py
tests/
	test_api.py
```

Why this shape:

- `app/main.py` keeps the actual `FastAPI` application factory and lifespan wiring small.
- `app/api/routes/` holds HTTP routes through `APIRouter`.
- `app/services/` holds recorder and transcription logic outside of HTTP handlers.
- `app/schemas/` holds request and response models.
- `app/dependencies.py` centralizes shared FastAPI dependency helpers.

## Run the API

With the FastAPI entrypoint configured in `pyproject.toml`, you can run the service with:

```powershell
uv run fastapi dev
```

Or directly with Uvicorn:

```powershell
uv run python -m app.main
```

## NPU transcription integration test

To separate recorder problems from transcription problems, there is an integration test that loads real audio fixtures and runs them through the actual `WhisperTranscriber` on `NPU`.

Fixture convention:

- put WAV fixtures in `tests/test-audio/`
- add a matching `.txt` file with the expected transcription
- example: `sample1.wav` and `sample1.txt`

The expected text does not need to match punctuation exactly; the test normalizes casing and punctuation before comparing.

Run just the NPU integration test with:

```powershell
uv run pytest -m npu tests/test_npu_transcription.py -rs
```

If this test passes but the API still returns output like `"you"` for long dictation, the problem is likely in the recording path or the captured microphone audio rather than the NPU transcription pipeline itself.

## Next implementation steps

1. Build the FastAPI app with a persistent `WhisperPipeline` loaded once at startup.
2. Add microphone recording endpoints.
3. Reuse the silence-aware chunking logic from the reference script.
4. Add the AutoHotkey hold-to-talk script.
