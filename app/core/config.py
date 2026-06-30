import tempfile
from pathlib import Path

API_TITLE = "Scribyte API"
API_DESCRIPTION = "Local dictation API powered by OpenVINO Whisper on Intel NPU"
API_VERSION = "0.1.0"

MODEL_ENV_VAR = "SCRIBYTE_MODEL"
DEFAULT_MODEL_NAME = "base"
MODEL_PATH = f"whisper_{DEFAULT_MODEL_NAME}_ov"
SAMPLE_RATE = 16000
MAX_CHUNK_SECONDS = 30
TOP_DB = 40
MIN_SILENCE_SECONDS = 0.3
MIN_CHUNK_SECONDS = 0.5
LANGUAGE = "<|en|>"

DEBUG_RECORDINGS_DIR = Path(tempfile.gettempdir()) / "scribyte-debug-recordings"
