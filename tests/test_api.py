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
        self._input_device = "Test Microphone [Windows WASAPI]"
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


def test_status_reports_ready() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    async def run_test() -> httpx.Response:
        async with make_test_client(app) as client:
            return await client.get("/status")

    response = asyncio.run(run_test())

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["debug_recordings_dir"] == str(Path(tempfile.gettempdir()) / "scribyte-debug-recordings")


def test_recording_flow_round_trip() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    async def run_test() -> tuple[httpx.Response, httpx.Response]:
        async with make_test_client(app) as client:
            start_response = await client.post("/start_recording")
            stop_response = await client.post("/stop_recording_and_transcribe")
            return start_response, stop_response

    start_response, stop_response = asyncio.run(run_test())

    assert start_response.status_code == 200
    assert start_response.json()["input_device"] == "Test Microphone [Windows WASAPI]"
    assert stop_response.status_code == 200
    assert stop_response.json()["text"] == "samples=16000"
    assert stop_response.json()["debug_audio_path"].endswith(".wav")


def test_pick_wasapi_input_device_prefers_matching_wasapi() -> None:
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


def test_pick_wasapi_input_devices_filters_non_wasapi_backends() -> None:
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


def test_pick_wasapi_input_device_returns_none_without_wasapi() -> None:
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


def test_prepare_audio_resamples_and_removes_dc_offset() -> None:
    duration_seconds = 0.25
    source_sample_rate = 48000
    target_sample_rate = 16000
    samples = int(duration_seconds * source_sample_rate)
    time_axis = np.arange(samples, dtype=np.float32) / np.float32(source_sample_rate)
    audio = 0.2 * np.sin(2 * np.pi * 220 * time_axis).astype(np.float32) + np.float32(0.05)

    prepared = prepare_audio(audio, source_sample_rate, target_sample_rate)

    assert abs(float(np.mean(prepared))) < 1e-3
    assert abs(len(prepared) - int(duration_seconds * target_sample_rate)) <= 2


def test_double_start_returns_conflict() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    async def run_test() -> tuple[httpx.Response, httpx.Response]:
        async with make_test_client(app) as client:
            first = await client.post("/start_recording")
            second = await client.post("/start_recording")
            return first, second

    first, second = asyncio.run(run_test())

    assert first.status_code == 200
    assert second.status_code == 409


def test_stop_without_start_returns_conflict() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    async def run_test() -> httpx.Response:
        async with make_test_client(app) as client:
            return await client.post("/stop_recording_and_transcribe")

    response = asyncio.run(run_test())

    assert response.status_code == 409


def test_short_recording_returns_bad_request() -> None:
    short_audio = np.ones(1000, dtype=np.float32)
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder(audio=short_audio))

    async def run_test() -> httpx.Response:
        async with make_test_client(app) as client:
            await client.post("/start_recording")
            return await client.post("/stop_recording_and_transcribe")

    response = asyncio.run(run_test())

    assert response.status_code == 400