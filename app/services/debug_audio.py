from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from app.core.config import DEBUG_RECORDINGS_DIR


def save_debug_recording(audio: np.ndarray, sample_rate: int) -> Path:
    DEBUG_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    file_path = DEBUG_RECORDINGS_DIR / f"recording-{timestamp}-{uuid4().hex[:8]}.wav"
    sf.write(file_path, audio, sample_rate)
    return file_path