from fastapi import HTTPException, Request

from app.core.config import MODEL_PATH
from app.services.recorder import RecorderState
from app.services.transcriber import WhisperTranscriber


def get_recorder_state(request: Request) -> RecorderState:
    return request.app.state.recorder


def get_optional_transcriber(request: Request) -> WhisperTranscriber | None:
    return getattr(request.app.state, "transcriber", None)


def get_startup_error(request: Request) -> str | None:
    return getattr(request.app.state, "startup_error", None)


def get_transcriber(request: Request) -> WhisperTranscriber:
    transcriber = get_optional_transcriber(request)
    if transcriber is None:
        raise HTTPException(
            status_code=503,
            detail=get_startup_error(request) or f"Transcriber for {MODEL_PATH} is not ready",
        )
    return transcriber