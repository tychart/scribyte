import logging
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


_logger = logging.getLogger("scribyte.transcriber")


def _detect_npu_error(msg: str) -> str | None:
    """Check if an error is an NPU availability failure and return a user-friendly message.

    Returns the friendly message if detected, otherwise None.
    """
    upper = msg.upper()
    # NPU compiler/runtime libraries missing — the most common cause.
    if "VCL COMPILER LOADING FAILED" in upper:
        return "NPU not available — the Intel NPU compiler/runtime libraries are missing on this system."
    # NPU library not found (generic "cannot load library" with NPU lib names)
    if "INTEL_NPU" in upper and "CANNOT LOAD LIBRARY" in upper:
        return "NPU not available — the Intel NPU compiler/runtime libraries are missing on this system."
    return None


def _detect_gpu_error(msg: str) -> str | None:
    """Check if an error is a GPU inference failure and return a user-friendly message.

    Returns the friendly message if detected, otherwise None.
    Covers VRAM exhaustion, context creation failures, and other GPU runtime issues.
    """
    upper = msg.upper()
    # clEnqueueMapBuffer, error code: -4 CL_MEM_OBJECT_ALLOCATION_FAILURE
    if "CL_MEM_OBJECT_ALLOCATION_FAILURE" in upper:
        return "GPU VRAM is full — the model could not allocate memory on the GPU. Falling back to CPU."
    # clEnqueueMapBuffer with CL_INVALID_VALUE (-30) — OpenVINO reports this when the GPU
    # cannot satisfy buffer mapping requests (VRAM exhaustion or driver-level limits).
    if "CLENQUEUEMAPBUFFER" in upper and "CL_INVALID_VALUE" in upper:
        return "GPU VRAM is full — the model could not allocate memory on the GPU. Falling back to CPU."
    # clCreateContext failure — GPU device unavailable (driver, VRAM, or permission issue).
    if "CLENQUEUEMAPBUFFER" not in upper and (
        "CL_CREATECONTEXT" in upper or "CLCREATECONTEXT" in upper
    ):
        return "GPU inference failed — the GPU device is unavailable or its OpenCL runtime is not working. Falling back to CPU."
    # General OCL / GPU out-of-memory patterns
    if "OUT OF HOST MEMORY" in upper or "OUT OF DEVICE MEMORY" in upper:
        return "GPU memory allocation failed — not enough VRAM available. Falling back to CPU."
    return None


def format_ov_error(error: Exception) -> str:
    """Extract a concise, human-readable error from OpenVINO exceptions.

    OpenVINO errors are multi-line with raw cpp/ocl paths. This strips those
    internal traces and keeps only the meaningful user-facing information.
    NPU and GPU failures are rewritten to plain-English messages.
    """
    import re

    msg = str(error)

    # Check for hardware/library-specific errors before stripping traces.
    npu_msg = _detect_npu_error(msg)
    if npu_msg is not None:
        return npu_msg
    gpu_msg = _detect_gpu_error(msg)
    if gpu_msg is not None:
        return gpu_msg

    lines = msg.splitlines()
    # Patterns that indicate OpenVINO internal trace lines (skip these entirely).
    internal_patterns = ("src/plugins/intel_", "src/inference/src/")
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in internal_patterns):
            continue
        # Lines like "Exception from src/.../file.cpp:NNN: detail" — extract just the detail.
        if stripped.startswith("Exception from"):
            idx = stripped.find("Exception from")
            after = stripped[idx + len("Exception from"):].strip()
            # Strip path:line_number: and keep only the message part.
            # e.g. "src/inference/src/cpp/infer_request.cpp:224: some detail"
            # -> "some detail"
            m = re.match(r"[^:]+:\s*\d+\s*:\s*(.*)", after)
            if m and m.group(1):
                filtered.append(m.group(1))
            continue
        if stripped:
            filtered.append(stripped)
    if filtered:
        return " ".join(filtered)
    return msg


class WhisperTranscriber:
    def __init__(self, model_path: str = MODEL_PATH, device: str = "NPU"):
        self.model_path = model_path
        self.device = device
        self.sample_rate = SAMPLE_RATE

        try:
            self.pipeline = ov_genai.WhisperPipeline(model_path, device)
        except Exception as error:  # pragma: no cover - hardware/runtime dependent
            concise = format_ov_error(error)
            _logger.debug("WhisperPipeline init failed on %s (full trace below)", device, exc_info=True)
            raise WhisperTranscriberError(
                f"WhisperPipeline failed on {device}: {concise}"
            ) from error

        # Determine the runtime-visible device name when possible.
        try:
            from openvino import Core

            core = Core()
            available = list(core.available_devices)
            # prefer a device name that contains the requested device token
            token = device.upper()
            match = None
            for dname in available:
                if token in dname.upper():
                    match = dname
                    break
            if match is None and available:
                match = available[0]
            self.runtime_device_name = match
        except Exception:  # best-effort only; don't fail if openvino Core isn't available
            self.runtime_device_name = None

    def warmup(self) -> None:
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        list(silence_aware_chunks(silence))
        try:
            self.pipeline.generate(_to_audio_sequence(silence), language=LANGUAGE)
        except Exception as error:  # pragma: no cover - hardware/runtime dependent
            concise = format_ov_error(error)
            raise WhisperTranscriberError(f"Whisper warmup failed: {concise}") from error

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