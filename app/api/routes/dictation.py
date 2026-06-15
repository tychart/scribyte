from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_optional_transcriber, get_recorder_state, get_startup_error, get_transcriber
from app.schemas.dictation import StartRecordingResponse, StatusResponse, TranscriptionResponse
from app.services.recorder import RecorderState, RecorderStateError
from app.services.transcriber import WhisperTranscriber, WhisperTranscriberError

router = APIRouter(tags=["dictation"])


@router.get("/status", response_model=StatusResponse)
def get_status(
    recorder_state: Annotated[RecorderState, Depends(get_recorder_state)],
    active_transcriber: Annotated[WhisperTranscriber | None, Depends(get_optional_transcriber)],
    startup_error: Annotated[str | None, Depends(get_startup_error)],
) -> StatusResponse:
    return StatusResponse(
        ready=active_transcriber is not None and startup_error is None,
        device=getattr(active_transcriber, "device", None),
        model_path=getattr(active_transcriber, "model_path", None),
        recording=recorder_state.is_recording,
        sample_rate=recorder_state.sample_rate,
        startup_error=startup_error,
    )


@router.post("/start_recording", response_model=StartRecordingResponse)
def start_recording(
    recorder_state: Annotated[RecorderState, Depends(get_recorder_state)],
    active_transcriber: Annotated[WhisperTranscriber, Depends(get_transcriber)],
) -> StartRecordingResponse:
    del active_transcriber
    try:
        recorder_state.start()
    except RecorderStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    return StartRecordingResponse(recording=True, sample_rate=recorder_state.sample_rate)


@router.post("/stop_recording_and_transcribe", response_model=TranscriptionResponse)
def stop_recording_and_transcribe(
    recorder_state: Annotated[RecorderState, Depends(get_recorder_state)],
    active_transcriber: Annotated[WhisperTranscriber, Depends(get_transcriber)],
) -> TranscriptionResponse:
    try:
        audio = recorder_state.stop()
    except RecorderStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    duration_seconds = len(audio) / active_transcriber.sample_rate
    if duration_seconds < 0.2:
        raise HTTPException(status_code=400, detail="Recording too short to transcribe")

    try:
        result = active_transcriber.transcribe(audio)
    except WhisperTranscriberError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return TranscriptionResponse.model_validate(result)