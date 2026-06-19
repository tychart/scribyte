"""Unit tests for RecorderState — recording lifecycle and chunk collection."""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import sounddevice as sd

from app.services.recorder_sounddevice import RecorderState
from app.services.recorder_contract import RecorderStateError


class TestRecorderStateLifecycle:
    """Tests for RecorderState start/stop lifecycle."""

    def test_initial_state_not_recording(self):
        recorder = RecorderState(sample_rate=16000)
        assert recorder.is_recording is False
        assert recorder.input_device is None
        assert recorder.sample_rate == 16000

    def test_start_raises_if_already_recording(self):
        recorder = RecorderState(sample_rate=16000)
        # Set _stream to a fake stream to trigger "already recording" error
        fake_stream = MagicMock()
        recorder._stream = fake_stream  # type: ignore[attr-defined]
        recorder._chunks = []  # type: ignore[attr-defined]
        with pytest.raises(RecorderStateError, match="already in progress"):
            recorder.start()

    def test_stop_raises_if_not_recording(self):
        recorder = RecorderState(sample_rate=16000)
        with pytest.raises(RecorderStateError, match="not currently running"):
            recorder.stop()


class TestRecorderStateAudioCollection:
    """Tests for audio chunk collection during recording."""

    def test_callback_appends_chunks(self):
        recorder = RecorderState(sample_rate=16000)
        chunk1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        chunk2 = np.array([0.4, 0.5], dtype=np.float32)

        recorder._callback(chunk1, 3, None, sd.CallbackFlags(0))  # type: ignore[attr-defined]
        recorder._callback(chunk2, 2, None, sd.CallbackFlags(0))  # type: ignore[attr-defined]

        assert len(recorder._chunks) == 2  # type: ignore[attr-defined]
        np.testing.assert_array_equal(recorder._chunks[0], chunk1)  # type: ignore[attr-defined]
        np.testing.assert_array_equal(recorder._chunks[1], chunk2)  # type: ignore[attr-defined]

    def test_callback_with_stereo_input_flattens(self):
        recorder = RecorderState(sample_rate=16000)
        stereo_chunk = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        recorder._callback(stereo_chunk, 2, None, sd.CallbackFlags(0))  # type: ignore[attr-defined]
        assert recorder._chunks[0].shape == (4,)  # type: ignore[attr-defined]

    def test_stop_concatenates_chunks(self):
        recorder = RecorderState(sample_rate=16000)
        # Use zero-mean values so prepare_audio doesn't change them
        recorder._chunks = [  # type: ignore[attr-defined]
            np.array([-0.15, -0.05], dtype=np.float32),
            np.array([0.05, 0.15], dtype=np.float32),
        ]
        # Simulate being recording by setting _stream to a mock with stop/close
        fake_stream = MagicMock()
        recorder._stream = fake_stream  # type: ignore[attr-defined]

        audio = recorder.stop()
        fake_stream.stop.assert_called_once()
        fake_stream.close.assert_called_once()
        assert recorder.is_recording is False
        assert recorder._chunks == []  # type: ignore[attr-defined]
        np.testing.assert_array_equal(audio, np.array([-0.15, -0.05, 0.05, 0.15], dtype=np.float32))

    def test_stop_with_empty_chunks_returns_empty(self):
        recorder = RecorderState(sample_rate=16000)
        fake_stream = MagicMock()
        recorder._stream = fake_stream  # type: ignore[attr-defined]
        recorder._chunks = []  # type: ignore[attr-defined]

        audio = recorder.stop()
        assert len(audio) == 0

    def test_stop_resets_state(self):
        recorder = RecorderState(sample_rate=16000, channels=2)
        fake_stream = MagicMock()
        recorder._stream = fake_stream  # type: ignore[attr-defined]
        recorder._input_device = "Test Mic"  # type: ignore[attr-defined]
        recorder._capture_sample_rate = 48000  # type: ignore[attr-defined]
        recorder._started_at = time.time()  # type: ignore[attr-defined]
        recorder._chunks = [np.array([1.0], dtype=np.float32)]  # type: ignore[attr-defined]

        recorder.stop()

        assert recorder._stream is None  # type: ignore[attr-defined]
        assert recorder._input_device is None  # type: ignore[attr-defined]
        assert recorder._capture_sample_rate == 16000  # type: ignore[attr-defined]
        assert recorder._started_at is None  # type: ignore[attr-defined]
        assert recorder._chunks == []  # type: ignore[attr-defined]


class TestRecorderStateConcurrency:
    """Tests for thread safety of RecorderState."""

    def test_callback_is_thread_safe(self) -> None:
        recorder = RecorderState(sample_rate=16000)
        thread_results: list[bool] = []
        errors: list[Exception] = []

        def add_chunk(chunk_data: np.ndarray) -> None:
            try:
                recorder._callback(chunk_data, len(chunk_data), None, sd.CallbackFlags(0))  # type: ignore[attr-defined]
                thread_results.append(True)
            except Exception as e:
                errors.append(e)

        threads: list[threading.Thread] = []
        for _ in range(4):
            t = threading.Thread(target=add_chunk, args=(np.array([0.5], dtype=np.float32),))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(thread_results)
        assert len(recorder._chunks) == 4  # type: ignore[attr-defined]

    def test_stop_interrupts_callback_gracefully(self) -> None:
        recorder = RecorderState(sample_rate=16000)
        fake_stream = MagicMock()
        recorder._stream = fake_stream  # type: ignore[attr-defined]
        recorder._chunks = [np.array([0.1, 0.2], dtype=np.float32)]  # type: ignore[attr-defined]

        # Stop should not raise even with concurrent callback access
        recorder.stop()
        assert recorder._stream is None  # type: ignore[attr-defined]
        assert recorder.is_recording is False
