"""Fast API / HTTP-level tests for Scribyte endpoints.

Tests API behavior without requiring hardware (FakeTranscriber + FakeRecorder).
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import tempfile

from fastapi import FastAPI
import httpx
import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import create_app
from app.services.recorder import RecorderStateError, pick_wasapi_input_device, pick_wasapi_input_devices, prepare_audio
from app.services.recorder_devices import pick_input_device
from app.services.transcriber import TranscriptionResult


@asynccontextmanager
async def make_test_client(app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    lifespan_context = app.router.lifespan_context
    typed_lifespan = lifespan_context
    async with typed_lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


class FakeRecorder:
    def __init__(self, audio: NDArray[np.float32] | None = None):
        self.sample_rate = 16000
        self.is_recording = False
        self._input_device = "Test Microphone"
        self.audio = audio if audio is not None else np.ones(16000, dtype=np.float32)

    @property
    def input_device(self) -> str | None:
        return self._input_device

    def start(self) -> None:
        if self.is_recording:
            raise RecorderStateError("Recording is already in progress")
        self.is_recording = True

    def stop(self) -> NDArray[np.float32]:
        if not self.is_recording:
            raise RecorderStateError("Recording is not currently running")
        self.is_recording = False
        return self.audio


class FakeTranscriber:
    def __init__(self):
        self.device = "NPU"
        self.model_path = "whisper_base_ov"
        self.sample_rate = 16000

    def warmup(self) -> None:
        return None

    def transcribe(self, audio: NDArray[np.float32]) -> TranscriptionResult:
        return {
            "text": f"samples={len(audio)}",
            "chunk_count": 1,
            "duration_seconds": len(audio) / self.sample_rate,
            "latency_seconds": 0.01,
        }


class TestStatusEndpoint:
    """Tests for GET /status."""

    def test_status_reports_ready(self):
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True
        assert data["device"] == "NPU"
        assert data["model_path"] == "whisper_base_ov"
        assert data["recording"] is False
        assert data["sample_rate"] == 16000
        assert data["startup_error"] is None
        assert data["debug_recordings_dir"] == str(Path(tempfile.gettempdir()) / "scribyte-debug-recordings")

    def test_status_reports_not_ready_when_no_transcriber(self):
        app = create_app(transcriber=None, recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is False
        assert data["startup_error"] is not None

    def test_status_exposes_startup_error(self):
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                # Set startup error after the lifespan opens
                # (lifespan sets it to None, so we override after)
                app.state.startup_error = "Model not found"
                app.state.startup_log = ["Startup failed"]
                return await client.get("/status")

        response = asyncio.run(run_test())
        data = response.json()
        assert data["startup_error"] == "Model not found"
        assert data["startup_log"] == ["Startup failed"]
        # ready should be False because startup_error is set
        assert data["ready"] is False

    def test_status_reports_recording_state(self):
        recorder = FakeRecorder()
        recorder.is_recording = True
        app = create_app(transcriber=FakeTranscriber(), recorder=recorder)

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        assert response.json()["recording"] is True


class TestStartRecordingEndpoint:
    """Tests for POST /start_recording."""

    def test_start_recording_success(self):
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.post("/start_recording")

        response = asyncio.run(run_test())
        assert response.status_code == 200
        data = response.json()
        assert data["recording"] is True
        assert data["sample_rate"] == 16000
        assert data["input_device"] == "Test Microphone"

    def test_start_recording_conflict_when_already_recording(self):
        recorder = FakeRecorder()
        recorder.is_recording = True
        app = create_app(transcriber=FakeTranscriber(), recorder=recorder)

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.post("/start_recording")

        response = asyncio.run(run_test())
        assert response.status_code == 409

    def test_start_recording_503_when_no_transcriber(self):
        app = create_app(transcriber=None, recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.post("/start_recording")

        response = asyncio.run(run_test())
        assert response.status_code == 503


class TestStopRecordingAndTranscribeEndpoint:
    """Tests for POST /stop_recording_and_transcribe."""

    def test_transcription_success(self):
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

        async def run_test() -> tuple[httpx.Response, httpx.Response]:
            async with make_test_client(app) as client:
                start = await client.post("/start_recording")
                stop = await client.post("/stop_recording_and_transcribe")
                return start, stop

        start_response, stop_response = asyncio.run(run_test())
        assert start_response.status_code == 200
        assert stop_response.status_code == 200
        data = stop_response.json()
        assert data["text"] == "samples=16000"
        assert data["chunk_count"] == 1
        assert data["duration_seconds"] == 1.0
        assert data["latency_seconds"] == 0.01
        assert data["debug_audio_path"].endswith(".wav")

    def test_stop_without_start_returns_conflict(self):
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.post("/stop_recording_and_transcribe")

        response = asyncio.run(run_test())
        assert response.status_code == 409

    def test_short_recording_returns_bad_request(self):
        short_audio = np.ones(1000, dtype=np.float32)
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder(audio=short_audio))

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                await client.post("/start_recording")
                return await client.post("/stop_recording_and_transcribe")

        response = asyncio.run(run_test())
        assert response.status_code == 400

    def test_empty_audio_returns_bad_request(self):
        empty_audio = np.array([], dtype=np.float32)
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder(audio=empty_audio))

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                await client.post("/start_recording")
                return await client.post("/stop_recording_and_transcribe")

        response = asyncio.run(run_test())
        assert response.status_code == 400

    def test_transcription_failure_returns_500(self):
        from app.services.transcriber import WhisperTranscriberError

        class FailingTranscriber(FakeTranscriber):
            def transcribe(self, audio: NDArray[np.float32]) -> TranscriptionResult:
                raise WhisperTranscriberError("Whisper pipeline error")

        app = create_app(transcriber=FailingTranscriber(), recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                await client.post("/start_recording")
                return await client.post("/stop_recording_and_transcribe")

        response = asyncio.run(run_test())
        assert response.status_code == 500
        assert "Whisper pipeline error" in response.json()["detail"]


class TestDoubleStartProtection:
    """Tests for double-start prevention."""

    def test_double_start_returns_conflict(self):
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

        async def run_test() -> tuple[httpx.Response, httpx.Response]:
            async with make_test_client(app) as client:
                first = await client.post("/start_recording")
                second = await client.post("/start_recording")
                return first, second

        first, second = asyncio.run(run_test())
        assert first.status_code == 200
        assert second.status_code == 409


class TestWasapiLegacyFunctions:
    """Tests for legacy WASAPI-specific device selection (still supported for backward compat)."""

    def test_pick_wasapi_input_device_prefers_matching_wasapi(self):
        default_device = {
            "name": "Surface Stereo Microphones (5- ",
            "index": 1,
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        }
        all_devices = [
            default_device,
            {
                "name": "Surface Stereo Microphones (5- SoundWire Audio)",
                "index": 14,
                "hostapi": 2,
                "max_input_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]
        hostapi_names = {0: "MME", 2: "Windows WASAPI"}

        selection = pick_wasapi_input_device(default_device, all_devices, hostapi_names, 16000)

        assert selection is not None
        assert selection.index == 14
        assert selection.name == "Surface Stereo Microphones (5- SoundWire Audio) [Windows WASAPI]"
        assert selection.sample_rate == 48000

    def test_pick_wasapi_input_devices_filters_non_wasapi_backends(self):
        default_device = {
            "name": "Surface Stereo Microphones (5- ",
            "index": 1,
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        }
        all_devices = [
            default_device,
            {
                "name": "Surface Stereo Microphones (5- SoundWire Audio)",
                "index": 7,
                "hostapi": 1,
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "name": "Surface Stereo Microphones (5- SoundWire Audio)",
                "index": 14,
                "hostapi": 2,
                "max_input_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]
        hostapi_names = {0: "MME", 1: "Windows DirectSound", 2: "Windows WASAPI"}

        selections = pick_wasapi_input_devices(default_device, all_devices, hostapi_names, 16000)

        assert [selection.index for selection in selections] == [14]

    def test_pick_wasapi_input_device_returns_none_without_wasapi(self):
        default_device = {
            "name": "Surface Stereo Microphones (5- ",
            "index": 1,
            "hostapi": 0,
            "max_input_channels": 2,
            "default_samplerate": 44100.0,
        }
        all_devices = [
            default_device,
            {
                "name": "Surface Stereo Microphones (5- SoundWire Audio)",
                "index": 7,
                "hostapi": 1,
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            },
        ]
        hostapi_names = {0: "MME", 1: "Windows DirectSound"}

        selection = pick_wasapi_input_device(default_device, all_devices, hostapi_names, 16000)

        assert selection is None


class TestAudioPreprocessing:
    """Tests for audio preprocessing functions."""

    def test_prepare_audio_resamples_and_removes_dc_offset(self):
        duration_seconds = 0.25
        source_sample_rate = 48000
        target_sample_rate = 16000
        samples = int(duration_seconds * source_sample_rate)
        time_axis = np.arange(samples, dtype=np.float32) / np.float32(source_sample_rate)
        audio = 0.2 * np.sin(2 * np.pi * 220 * time_axis).astype(np.float32) + np.float32(0.05)

        prepared = prepare_audio(audio, source_sample_rate, target_sample_rate)

        assert abs(float(np.mean(prepared))) < 1e-3
        assert abs(len(prepared) - int(duration_seconds * target_sample_rate)) <= 2

    def test_prepare_audio_no_resample_when_same_rate(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = prepare_audio(audio, 16000, 16000)
        assert abs(np.mean(result)) < 1e-3
        assert len(result) == 1000

    def test_prepare_audio_empty_input(self):
        audio = np.array([], dtype=np.float32)
        result = prepare_audio(audio, 16000, 16000)
        assert len(result) == 0


class TestUnifiedPickInputDevice:
    """Tests for the new unified pick_input_device function."""

    def test_pick_input_device_returns_valid_selection(self):
        selection = pick_input_device(fallback_sample_rate=16000)
        assert selection is not None
        assert selection.index is not None
        assert selection.sample_rate > 0
