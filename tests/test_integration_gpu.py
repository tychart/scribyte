"""Integration tests for GPU-backed transcription.

These tests require actual GPU hardware (NVIDIA/AMD) and the exported whisper model.
Run explicitly with: uv run pytest -m integration
"""

import re
from pathlib import Path
import sys
import wave

import librosa
import numpy as np
from numpy.typing import NDArray
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.transcriber import WhisperTranscriber


FIXTURE_DIR = Path(__file__).resolve().parent / "test-audio"
SAMPLE_RATE = 16000


def _load_pcm_wav(audio_path: Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channel_count = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_bytes = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported fixture sample width: {sample_width}")

    audio = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if channel_count > 1:
        audio = audio.reshape(-1, channel_count).mean(axis=1, dtype=np.float32)

    return audio, sample_rate


def _load_fixture_audio(audio_path: Path) -> NDArray[np.float32]:
    audio, sample_rate = _load_pcm_wav(audio_path)

    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        audio = np.asarray(audio, dtype=np.float32)

    return audio


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    collapsed = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(collapsed.split())


def _expected_tokens(value: str) -> set[str]:
    normalized = _normalize_text(value)
    return {token for token in normalized.split() if len(token) >= 3}


def _fixture_cases() -> list[tuple[Path, Path]]:
    wav_files = sorted(FIXTURE_DIR.glob("*.wav"))
    cases: list[tuple[Path, Path]] = []

    for wav_path in wav_files:
        transcript_path = wav_path.with_suffix(".txt")
        if transcript_path.exists():
            cases.append((wav_path, transcript_path))

    return cases


def _case_id(value: Path | object) -> str:
    return value.name if isinstance(value, Path) else str(value)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("audio_path", "transcript_path"),
    _fixture_cases(),
    ids=_case_id,
)
def test_gpu_transcription_matches_reference_audio(audio_path: Path, transcript_path: Path) -> None:
    expected_text = transcript_path.read_text(encoding="utf-8").strip()
    if not expected_text:
        pytest.fail(f"Expected transcript file is empty: {transcript_path}")

    audio = _load_fixture_audio(audio_path)
    transcriber = WhisperTranscriber(device="GPU")
    result = transcriber.transcribe(audio)

    actual_text = result["text"]
    normalized_actual = _normalize_text(actual_text)
    normalized_expected = _normalize_text(expected_text)

    assert normalized_actual, "The GPU transcription came back empty"

    expected_tokens = _expected_tokens(expected_text)
    actual_tokens = set(normalized_actual.split())
    matched_tokens = expected_tokens & actual_tokens

    if normalized_expected not in normalized_actual:
        assert matched_tokens, (
            "The GPU transcription did not contain the expected phrase or any expected keyword. "
            f"Expected={expected_text!r}, Actual={actual_text!r}"
        )

        overlap = len(matched_tokens) / max(len(expected_tokens), 1)
        assert overlap >= 0.5, (
            "The GPU transcription quality was too far from the expected text. "
            f"Fixture={audio_path.name!r}, Expected={expected_text!r}, Actual={actual_text!r}, Overlap={overlap:.2f}"
        )
