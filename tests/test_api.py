from pathlib import Path
import sys

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import create_app
from app.services.recorder import RecorderStateError


class FakeRecorder:
    def __init__(self, audio: np.ndarray | None = None):
        self.sample_rate = 16000
        self.is_recording = False
        self.audio = audio if audio is not None else np.ones(16000, dtype=np.float32)

    def start(self) -> None:
        if self.is_recording:
            raise RecorderStateError("Recording is already in progress")
        self.is_recording = True

    def stop(self) -> np.ndarray:
        if not self.is_recording:
            raise RecorderStateError("Recording is not currently running")
        self.is_recording = False
        return self.audio


class FakeTranscriber:
    def __init__(self):
        self.device = "NPU"
        self.model_path = "whisper_base_ov"
        self.sample_rate = 16000

    def transcribe(self, audio: np.ndarray) -> dict[str, object]:
        return {
            "text": f"samples={len(audio)}",
            "chunk_count": 1,
            "duration_seconds": len(audio) / self.sample_rate,
            "latency_seconds": 0.01,
        }


def test_status_reports_ready() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    with TestClient(app) as client:
        response = client.get("/status")

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_recording_flow_round_trip() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    with TestClient(app) as client:
        start_response = client.post("/start_recording")
        stop_response = client.post("/stop_recording_and_transcribe")

    assert start_response.status_code == 200
    assert stop_response.status_code == 200
    assert stop_response.json()["text"] == "samples=16000"


def test_double_start_returns_conflict() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    with TestClient(app) as client:
        first = client.post("/start_recording")
        second = client.post("/start_recording")

    assert first.status_code == 200
    assert second.status_code == 409


def test_stop_without_start_returns_conflict() -> None:
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder())

    with TestClient(app) as client:
        response = client.post("/stop_recording_and_transcribe")

    assert response.status_code == 409


def test_short_recording_returns_bad_request() -> None:
    short_audio = np.ones(1000, dtype=np.float32)
    app = create_app(transcriber=FakeTranscriber(), recorder=FakeRecorder(audio=short_audio))

    with TestClient(app) as client:
        client.post("/start_recording")
        response = client.post("/stop_recording_and_transcribe")

    assert response.status_code == 400