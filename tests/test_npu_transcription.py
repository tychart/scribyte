import re
from pathlib import Path
import sys

import librosa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.transcriber import WhisperTranscriber


FIXTURE_DIR = Path(__file__).resolve().parent / "test-audio"


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


@pytest.mark.npu
@pytest.mark.parametrize(
    ("audio_path", "transcript_path"),
    _fixture_cases(),
    ids=lambda value: value.name if isinstance(value, Path) else str(value),
)
def test_npu_transcription_matches_reference_audio(audio_path: Path, transcript_path: Path) -> None:
    expected_text = transcript_path.read_text(encoding="utf-8").strip()
    if not expected_text:
        pytest.fail(f"Expected transcript file is empty: {transcript_path}")

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    transcriber = WhisperTranscriber(device="NPU")
    result = transcriber.transcribe(audio)

    actual_text = result["text"]
    normalized_actual = _normalize_text(actual_text)
    normalized_expected = _normalize_text(expected_text)

    assert normalized_actual, "The NPU transcription came back empty"

    expected_tokens = _expected_tokens(expected_text)
    actual_tokens = set(normalized_actual.split())
    matched_tokens = expected_tokens & actual_tokens

    if normalized_expected not in normalized_actual:
        assert matched_tokens, (
            "The NPU transcription did not contain the expected phrase or any expected keyword. "
            f"Expected={expected_text!r}, Actual={actual_text!r}"
        )

        overlap = len(matched_tokens) / max(len(expected_tokens), 1)
        assert overlap >= 0.5, (
            "The NPU transcription quality was too far from the expected text. "
            f"Fixture={audio_path.name!r}, Expected={expected_text!r}, Actual={actual_text!r}, Overlap={overlap:.2f}"
        )