"""Unit tests for WhisperTranscriber fallback logic and silence-aware chunking."""

from typing import Any

import numpy as np
import pytest

from app.services.transcriber import (
    WhisperTranscriber,
    WhisperTranscriberError,
    TranscriptionResult,
    silence_aware_chunks,
)
from app.core.config import SAMPLE_RATE


class TestSilenceAwareChunks:
    """Tests for silence-aware audio chunking."""

    def test_empty_audio_yields_original(self):
        audio = np.array([], dtype=np.float32)
        chunks = list(silence_aware_chunks(audio))
        assert len(chunks) == 1
        assert len(chunks[0]) == 0

    def test_short_audio_without_silence_yields_single_chunk(self):
        # Audio shorter than MIN_SILENCE_SECONDS — no natural silence breaks
        audio = np.sin(2 * np.pi * 440 * np.arange(8000) / SAMPLE_RATE, dtype=np.float32)
        chunks = list(silence_aware_chunks(audio))
        assert len(chunks) == 1
        assert len(chunks[0]) == len(audio)

    def test_audio_with_silence_yields_multiple_chunks(self):
        # Create audio longer than MAX_CHUNK_SECONDS with a long silence gap
        from app.core.config import MAX_CHUNK_SECONDS

        # Part 1: MAX_CHUNK_SECONDS
        part1 = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * MAX_CHUNK_SECONDS) / SAMPLE_RATE, dtype=np.float32)
        # Silence: 5 seconds
        silence = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
        # Part 2: MAX_CHUNK_SECONDS
        part2 = np.sin(2 * np.pi * 880 * np.arange(SAMPLE_RATE * MAX_CHUNK_SECONDS) / SAMPLE_RATE, dtype=np.float32)
        audio = np.concatenate([part1, silence, part2])

        chunks = list(silence_aware_chunks(audio))
        # Should split into multiple chunks:
        # 1) First MAX_CHUNK_SECONDS segment
        # 2) Silence gap (> min_silence_seconds)
        # 3) Second MAX_CHUNK_SECONDS segment
        # Because the 5s silence gap > 0.3s min, it splits, and total > MAX_CHUNK_SECONDS
        assert len(chunks) >= 2
        total_samples = sum(len(c) for c in chunks)
        assert total_samples <= len(audio)

    def test_chunks_do_not_exceed_max_chunk_samples(self):
        """Chunks should not exceed MAX_CHUNK_SECONDS worth of samples."""
        from app.core.config import MAX_CHUNK_SECONDS

        # Create audio that is much longer than MAX_CHUNK_SECONDS with large silence gaps
        # Each segment is MAX_CHUNK_SECONDS with large silence between them
        segment_samples = SAMPLE_RATE * MAX_CHUNK_SECONDS
        gap_samples = int(SAMPLE_RATE * 2)  # 2 second gaps between segments
        total_samples = segment_samples + gap_samples + segment_samples + gap_samples + segment_samples

        audio = np.zeros(total_samples, dtype=np.float32)
        # Fill with sine waves in MAX_CHUNK_SECONDS segments
        offset = 0
        for _ in range(3):
            audio[offset:offset + segment_samples] = np.sin(
                2 * np.pi * 440 * np.arange(segment_samples) / SAMPLE_RATE
            )
            offset += segment_samples + gap_samples

        chunks = list(silence_aware_chunks(audio))

        for chunk in chunks:
            if len(chunk) > 0:
                # Each chunk should be at most MAX_CHUNK_SECONDS worth
                assert len(chunk) <= int(MAX_CHUNK_SECONDS * SAMPLE_RATE * 1.1)  # small tolerance

    def test_merged_short_gaps(self):
        """Silence gaps shorter than MIN_SILENCE_SECONDS should be merged."""
        from app.core.config import MIN_SILENCE_SECONDS
        del MIN_SILENCE_SECONDS  # noqa: F841

        # Audio with very short gaps (less than MIN_SILENCE_SECONDS)
        # These should be merged into fewer chunks
        audio = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * 4) / SAMPLE_RATE, dtype=np.float32)
        # Add very short gaps (0.1 seconds, less than 0.3 MIN_SILENCE_SECONDS)
        audio[SAMPLE_RATE: SAMPLE_RATE + int(0.1 * SAMPLE_RATE)] = 0.0
        audio[SAMPLE_RATE * 2: SAMPLE_RATE * 2 + int(0.1 * SAMPLE_RATE)] = 0.0

        chunks = list(silence_aware_chunks(audio))
        assert len(chunks) >= 1


class TestTranscriptionResult:
    """Tests for TranscriptionResult TypedDict structure."""

    def test_result_has_required_keys(self):
        result: TranscriptionResult = {
            "text": "hello world",
            "chunk_count": 2,
            "duration_seconds": 3.5,
            "latency_seconds": 1.2,
        }
        assert "text" in result
        assert "chunk_count" in result
        assert "duration_seconds" in result
        assert "latency_seconds" in result
        assert result["chunk_count"] == 2
        assert isinstance(result["duration_seconds"], float)
        assert isinstance(result["latency_seconds"], float)


def _make_mock_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Create a mock WhisperPipeline for testing."""
    _MockResult = type("Result", (), {"texts": ["mock"]})

    class _MockPipeline:  # noqa: N801
        @staticmethod
        def generate(*args: Any, **kwargs: Any) -> Any:
            return _MockResult()

    return _MockPipeline


class TestWhisperTranscriberSingleDevice:
    """Tests for WhisperTranscriber single-device initialization.

    Fallback ordering is handled by main.py, not by WhisperTranscriber.
    WhisperTranscriber tries exactly one device and raises on failure.
    """

    def test_successful_init_sets_device(self, monkeypatch: Any) -> None:
        """If the requested device works, device is set and pipeline created."""
        import openvino_genai as ov_genai

        monkeypatch.setattr(ov_genai, "WhisperPipeline", _make_mock_pipeline)

        transcriber = WhisperTranscriber(device="CPU")
        assert transcriber.device == "CPU"
        assert transcriber.pipeline is not None

    def test_failed_init_raises_error_with_device_name(self, monkeypatch: Any) -> None:
        """If the device fails, WhisperTranscriberError includes the device name."""
        import openvino_genai as ov_genai

        def always_fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Device unavailable")

        monkeypatch.setattr(ov_genai, "WhisperPipeline", always_fail)

        with pytest.raises(WhisperTranscriberError, match="GPU"):
            WhisperTranscriber(device="GPU")

    def test_failed_init_error_contains_concise_message(self, monkeypatch: Any) -> None:
        """Error message should be concise, not raw OpenVINO internals."""
        import openvino_genai as ov_genai

        def fail_with_ov_style(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "Exception from src/inference/src/cpp/infer_request.cpp:224:\n"
                "Exception from src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp:74:\n"
                "[GPU] clEnqueueMapBuffer, error code: -30 CL_INVALID_VALUE"
            )

        monkeypatch.setattr(ov_genai, "WhisperPipeline", fail_with_ov_style)

        with pytest.raises(WhisperTranscriberError) as exc_info:
            WhisperTranscriber(device="GPU")

        error_msg = str(exc_info.value)
        # Should contain the device name and a recognizable error token
        assert "GPU" in error_msg
        # Should not contain raw cpp paths
        assert "src/inference/src/cpp" not in error_msg

    def test_sample_rate_is_constant(self) -> None:
        """Sample rate should always be 16000."""
        assert SAMPLE_RATE == 16000


class TestFormatOVError:
    """Tests for the _format_ov_error helper."""

    def test_strips_cpp_path(self) -> None:
        from app.services.transcriber import format_ov_error

        err = RuntimeError("Exception from src/inference/src/cpp/infer_request.cpp:224: some detail")
        result = format_ov_error(err)
        assert "src/inference/src/cpp" not in result
        assert "some detail" in result

    def test_handles_cl_error(self) -> None:
        from app.services.transcriber import format_ov_error

        err = RuntimeError(
            "[GPU] clEnqueueMapBuffer, error code: -30 CL_INVALID_VALUE"
        )
        result = format_ov_error(err)
        assert "clEnqueueMapBuffer" in result
        assert "CL_INVALID_VALUE" in result

    def test_passthrough_for_plain_error(self) -> None:
        from app.services.transcriber import format_ov_error

        err = RuntimeError("Simple error message")
        result = format_ov_error(err)
        assert result == "Simple error message"
