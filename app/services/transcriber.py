import time
from typing import Iterator, Protocol, SupportsFloat, TypedDict

import librosa
import numpy as np
from numpy.typing import NDArray
import openvino_genai as ov_genai

from app.core.config import (
    LANGUAGE,
    MAX_CHUNK_SECONDS,
    MIN_CHUNK_SECONDS,
    MIN_SILENCE_SECONDS,
    MODEL_PATH,
    SAMPLE_RATE,
    TOP_DB,
)


class WhisperTranscriberError(RuntimeError):
    pass


class TranscriptionResult(TypedDict):
    text: str
    chunk_count: int
    duration_seconds: float
    latency_seconds: float


class Transcriber(Protocol):
    device: str
    model_path: str
    sample_rate: int

    def warmup(self) -> None: ...

    def transcribe(self, audio: NDArray[np.float32]) -> TranscriptionResult: ...


def _to_audio_sequence(audio: NDArray[np.float32]) -> list[SupportsFloat]:
    return audio.astype(np.float32, copy=False).tolist()


def silence_aware_chunks(
    audio: NDArray[np.float32],
    max_chunk_seconds: int = MAX_CHUNK_SECONDS,
) -> Iterator[NDArray[np.float32]]:
    max_chunk_samples = max_chunk_seconds * SAMPLE_RATE
    min_silence_samples = int(MIN_SILENCE_SECONDS * SAMPLE_RATE)
    intervals = librosa.effects.split(audio, top_db=TOP_DB, frame_length=512, hop_length=128)

    if len(intervals) == 0:
        yield audio
        return

    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        gap = start - merged[-1][1]
        if gap < min_silence_samples:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    chunk_start = merged[0][0]
    chunk_end = merged[0][1]

    for interval_start, interval_end in merged[1:]:
        if interval_end - chunk_start > max_chunk_samples:
            yield audio[chunk_start:chunk_end]
            chunk_start = interval_start
            chunk_end = interval_end
        else:
            chunk_end = interval_end

    yield audio[chunk_start:chunk_end]


class WhisperTranscriber:
    def __init__(self, model_path: str = MODEL_PATH, device: str = "NPU", allow_fallback: bool = True):
        self.model_path = model_path
        self.device = device
        self.sample_rate = SAMPLE_RATE

        # Normalize device order and apply fallback if enabled.
        def _order_from_requested(req: str) -> list[str]:
            req_up = (req or "").upper()
            if req_up == "NPU":
                return ["NPU", "GPU", "CPU"]
            if req_up == "GPU":
                return ["GPU", "CPU"]
            return ["CPU"]

        tried: list[tuple[str, Exception | None]] = []
        last_error: Exception | None = None

        devices_to_try = _order_from_requested(device) if allow_fallback else [device]

        for dev in devices_to_try:
            try:
                self.pipeline = ov_genai.WhisperPipeline(model_path, dev)
                self.device = dev
                # Determine the runtime-visible device name when possible.
                try:
                    from openvino import Core

                    core = Core()
                    available = list(core.available_devices)
                    # prefer a device name that contains the requested device token
                    token = dev.upper()
                    match = None
                    for dname in available:
                        if token in dname.upper():
                            match = dname
                            break
                    if match is None and available:
                        match = available[0]
                    self.runtime_device_name = match
                except Exception:
                    # best-effort only; don't fail if openvino Core isn't available
                    self.runtime_device_name = None
                break
            except Exception as error:  # pragma: no cover - hardware/runtime dependent
                last_error = error
                tried.append((dev, error))

        if not hasattr(self, "pipeline"):
            # Build a helpful message including attempts
            attempts = ", ".join(f"{d}: {e}" for d, e in tried)
            raise WhisperTranscriberError(
                f"Failed to initialize WhisperPipeline for requested device(s) {devices_to_try}. Attempts: {attempts}"
            ) from last_error

    def warmup(self) -> None:
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        try:
            self.pipeline.generate(_to_audio_sequence(silence), language=LANGUAGE)
        except Exception as error:  # pragma: no cover - hardware/runtime dependent
            raise WhisperTranscriberError(f"Whisper warmup failed: {error}") from error

    def transcribe(self, audio: NDArray[np.float32]) -> TranscriptionResult:
        normalized_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        start = time.time()
        full_text: list[str] = []
        chunk_count = 0

        for chunk in silence_aware_chunks(normalized_audio):
            duration = len(chunk) / self.sample_rate
            if duration < MIN_CHUNK_SECONDS:
                continue

            try:
                result = self.pipeline.generate(_to_audio_sequence(chunk), language=LANGUAGE)
            except Exception as error:  # pragma: no cover - hardware/runtime dependent
                raise WhisperTranscriberError(f"Chunk transcription failed: {error}") from error

            text = result.texts[0].strip()
            if text:
                full_text.append(text)
            chunk_count += 1

        latency = time.time() - start
        return {
            "text": " ".join(full_text).strip(),
            "chunk_count": chunk_count,
            "duration_seconds": len(normalized_audio) / self.sample_rate,
            "latency_seconds": latency,
        }