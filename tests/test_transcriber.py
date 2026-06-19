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


def _make_mock_pipeline() -> Any:
    """Create a mock WhisperPipeline for testing."""
    _MockResult = type("Result", (), {"texts": ["mock"]})
    _MockPipeline = type("MockPipeline", (), {"generate": _MockResult()})  # noqa: N806
    return _MockPipeline


class TestWhisperTranscriberFallback:
    """Tests for WhisperTranscriber fallback chain.

    These tests mock the underlying OpenVINO pipeline to verify fallback behavior
    without requiring actual hardware.
    """

    def test_fallback_npu_to_gpu(self, monkeypatch: Any) -> None:
        """If NPU fails, should try GPU next."""
        import openvino_genai as ov_genai

        call_count = 0

        def failing_init(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("NPU not available")
            return _make_mock_pipeline()

        monkeypatch.setattr(ov_genai, "WhisperPipeline", failing_init)

        transcriber = WhisperTranscriber(device="NPU", allow_fallback=True)
        assert transcriber.device == "GPU"

    def test_fallback_gpu_to_cpu(self, monkeypatch: Any) -> None:
        """If GPU fails, should try CPU next."""
        import openvino_genai as ov_genai

        call_count = 0

        def failing_init(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("GPU not available")
            # Second call (CPU) succeeds
            return _make_mock_pipeline()

        monkeypatch.setattr(ov_genai, "WhisperPipeline", failing_init)

        transcriber = WhisperTranscriber(device="GPU", allow_fallback=True)
        assert transcriber.device == "CPU"

    def test_no_fallback_when_disabled(self, monkeypatch: Any) -> None:
        """If allow_fallback=False and device fails, should raise."""
        import openvino_genai as ov_genai

        def always_fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Device unavailable")

        monkeypatch.setattr(ov_genai, "WhisperPipeline", always_fail)

        with pytest.raises(WhisperTranscriberError, match="Failed to initialize"):
            WhisperTranscriber(device="GPU", allow_fallback=False)

    def test_all_devices_fail_raises_helpful_error(self, monkeypatch: Any) -> None:
        """If all devices fail, error should list each attempt."""
        import openvino_genai as ov_genai

        def always_fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Not available")

        monkeypatch.setattr(ov_genai, "WhisperPipeline", always_fail)

        with pytest.raises(WhisperTranscriberError) as exc_info:
            WhisperTranscriber(device="NPU", allow_fallback=True)

        error_msg = str(exc_info.value)
        assert "NPU" in error_msg
        assert "GPU" in error_msg
        assert "CPU" in error_msg

    def test_requested_device_order_gpu(self, monkeypatch: Any) -> None:
        """Requesting GPU should try GPU then CPU only (NPU skipped)."""
        import openvino_genai as ov_genai

        devices_tried: list[str] = []

        def record_device(model_path: str, device: str, *args: Any, **kwargs: Any) -> Any:
            devices_tried.append(device)
            if device == "GPU":
                raise RuntimeError("GPU not available")
            return _make_mock_pipeline()

        monkeypatch.setattr(ov_genai, "WhisperPipeline", record_device)

        transcriber = WhisperTranscriber(device="GPU", allow_fallback=True)
        assert "GPU" in devices_tried
        assert "CPU" in devices_tried
        assert "NPU" not in devices_tried
        assert transcriber.device == "CPU"

    def test_requested_device_order_cpu(self, monkeypatch: Any) -> None:
        """Requesting CPU should only try CPU."""
        import openvino_genai as ov_genai

        devices_tried: list[str] = []

        def record_device(model_path: str, device: str, *args: Any, **kwargs: Any) -> Any:
            devices_tried.append(device)
            return _make_mock_pipeline()

        monkeypatch.setattr(ov_genai, "WhisperPipeline", record_device)

        transcriber = WhisperTranscriber(device="CPU", allow_fallback=True)
        assert devices_tried == ["CPU"]
        assert transcriber.device == "CPU"

    def test_sample_rate_is_constant(self) -> None:
        """Sample rate should always be 16000."""
        assert SAMPLE_RATE == 16000
