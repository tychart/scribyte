from pydantic import BaseModel


class StatusResponse(BaseModel):
    ready: bool
    device: str | None = None
    model_path: str | None = None
    recording: bool
    sample_rate: int
    startup_error: str | None = None
    debug_recordings_dir: str | None = None


class StartRecordingResponse(BaseModel):
    recording: bool
    sample_rate: int


class TranscriptionResponse(BaseModel):
    text: str
    chunk_count: int
    duration_seconds: float
    latency_seconds: float
    debug_audio_path: str | None = None