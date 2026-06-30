"""Fast API / HTTP-level tests for Scribyte endpoints.

Tests API behavior without requiring hardware (FakeTranscriber + FakeRecorder).
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import tempfile
from typing import Any

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

    def start(self, device_index: int | None = None) -> None:
        del device_index
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
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                # Simulate startup failure after the lifespan opens
                app.state.transcriber = None
                app.state.startup_error = "Model not found"
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

    def test_status_reports_configured_model_when_startup_fails(self):
        app = create_app(recorder=FakeRecorder(), model_selection="small")

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                app.state.transcriber = None
                app.state.startup_error = "Model not found"
                return await client.get("/status")

        response = asyncio.run(run_test())
        data = response.json()
        assert data["model_path"].endswith("whisper_small_ov")

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
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder(), model_selection="small")

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                # Simulate startup failure after the lifespan opens
                app.state.transcriber = None
                app.state.startup_error = "Model not found"
                return await client.post("/start_recording")

        response = asyncio.run(run_test())
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not found"

    def test_start_recording_503_mentions_configured_model_without_startup_error(self):
        app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder(), model_selection="small")

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                app.state.transcriber = None
                app.state.startup_error = None
                return await client.post("/start_recording")

        response = asyncio.run(run_test())
        assert response.status_code == 503
        assert response.json()["detail"].endswith("whisper_small_ov is not ready")


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


class TestDeviceFallback:
    """Tests for device fallback ordering in create_app."""

    def test_determine_device_order_default(self) -> None:
        from app.main import determine_device_order

        assert determine_device_order(None) == ["NPU", "GPU", "CPU"]

    def test_determine_device_order_gpu(self) -> None:
        from app.main import determine_device_order

        assert determine_device_order("GPU") == ["GPU", "CPU"]

    def test_determine_device_order_cpu(self) -> None:
        from app.main import determine_device_order

        assert determine_device_order("CPU") == ["CPU"]

    def test_determine_model_path_defaults_to_base_model_folder(self) -> None:
        from app.main import PROJECT_ROOT, determine_model_path

        assert determine_model_path(None) == PROJECT_ROOT / "whisper_base_ov"

    def test_determine_model_path_expands_simple_model_name(self) -> None:
        from app.main import PROJECT_ROOT, determine_model_path

        assert determine_model_path("small") == PROJECT_ROOT / "whisper_small_ov"

    def test_determine_model_path_accepts_full_folder_name(self) -> None:
        from app.main import PROJECT_ROOT, determine_model_path

        assert determine_model_path("whisper_large_v3_ov") == PROJECT_ROOT / "whisper_large_v3_ov"

    def test_determine_model_path_preserves_custom_relative_path(self) -> None:
        from app.main import determine_model_path

        assert determine_model_path("models/custom") == Path("models/custom")

    def test_fallback_npu_to_gpu_in_app(self, monkeypatch: Any) -> None:
        """When NPU fails, main.py should try GPU next and succeed."""
        from app.main import create_app
        import openvino_genai as ov_genai

        def failing_init(model_path: str, device: str, *args: Any, **kwargs: Any) -> Any:
            if device == "NPU":
                raise RuntimeError("NPU not available")
            return _make_mock_pipeline()

        monkeypatch.setattr(ov_genai, "WhisperPipeline", failing_init)

        app = create_app()

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        data = response.json()
        assert response.status_code == 200
        assert data["device"] == "GPU"
        assert data["ready"] is True
        fallback_lines = [line for line in data["startup_log"] if "NPU not available" in line]
        assert len(fallback_lines) == 1
        assert "falling back" in fallback_lines[0].lower()

    def test_successful_init_does_not_log_fallback(self, monkeypatch: Any) -> None:
        """A successful initialization should not claim a fallback happened."""
        from app.main import create_app
        import openvino_genai as ov_genai

        monkeypatch.setattr(ov_genai, "WhisperPipeline", _make_mock_pipeline)

        app = create_app()

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        data = response.json()

        assert response.status_code == 200
        initialized_lines = [line for line in data["startup_log"] if line.startswith("Initialized transcriber on")]
        assert len(initialized_lines) == 1
        assert "falling back" not in initialized_lines[0].lower()

    def test_fallback_all_devices_fail_in_app(self, monkeypatch: Any) -> None:
        """When all devices fail, app should report startup error."""
        from app.main import create_app
        import openvino_genai as ov_genai

        def always_fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Device unavailable")

        monkeypatch.setattr(ov_genai, "WhisperPipeline", always_fail)

        app = create_app()

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        data = response.json()
        assert response.status_code == 200
        assert data["ready"] is False
        assert data["startup_error"] is not None
        assert any(line == "Initializing transcriber on NPU" for line in data["startup_log"])
        assert any("NPU not available" in line for line in data["startup_log"])
        assert any(line == "Initializing transcriber on GPU" for line in data["startup_log"])
        assert any(line == "Initializing transcriber on CPU" for line in data["startup_log"])

    def test_no_fallback_when_device_limit_gpu(self, monkeypatch: Any) -> None:
        """When device_limit=GPU, NPU should not be tried."""
        from app.main import create_app
        import openvino_genai as ov_genai

        call_order: list[str] = []

        def record_device(model_path: str, device: str, *args: Any, **kwargs: Any) -> Any:
            call_order.append(device)
            if device == "GPU":
                raise RuntimeError("GPU not available")
            return _make_mock_pipeline()

        monkeypatch.setattr(ov_genai, "WhisperPipeline", record_device)

        app = create_app(device_limit="gpu")

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        data = response.json()
        assert response.status_code == 200
        assert data["device"] == "CPU"
        assert "NPU" not in call_order  # NPU never tried
        # CPU should be in log as fallback
        assert any("falling back" in line.lower() for line in data["startup_log"])

    def test_model_selection_uses_requested_model_folder(self, monkeypatch: Any) -> None:
        from app.main import create_app
        import openvino_genai as ov_genai

        captured_model_paths: list[str] = []

        def record_model_path(model_path: str, device: str, *args: Any, **kwargs: Any) -> Any:
            del device, args, kwargs
            captured_model_paths.append(model_path)
            return _make_mock_pipeline()

        monkeypatch.setattr(ov_genai, "WhisperPipeline", record_model_path)

        app = create_app(model_selection="small")

        async def run_test() -> httpx.Response:
            async with make_test_client(app) as client:
                return await client.get("/status")

        response = asyncio.run(run_test())
        data = response.json()

        assert response.status_code == 200
        assert captured_model_paths[0].endswith("whisper_small_ov")
        assert data["model_path"].endswith("whisper_small_ov")
        assert "Configured model path:" in data["startup_log"][0]


def _make_mock_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Create a mock WhisperPipeline for testing."""
    _MockResult = type("Result", (), {"texts": ["mock"]})

    class _MockPipeline:  # noqa: N801
        @staticmethod
        def generate(*args: Any, **kwargs: Any) -> Any:
            return _MockResult()

    return _MockPipeline
