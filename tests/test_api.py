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
from app.services.recorder import RecorderStateError
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
        self.input_device = "Test Microphone"
        self.audio = audio if audio is not None else np.ones(16000, dtype=np.float32)

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
    assert start_response.json()["input_device"] == "Test Microphone"
    assert stop_response.status_code == 200
    assert stop_response.json()["text"] == "samples=16000"
    assert stop_response.json()["debug_audio_path"].endswith(".wav")


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