from typing import Annotated

from anyio import to_thread
from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import DEBUG_RECORDINGS_DIR
from app.dependencies import get_optional_transcriber, get_recorder_state, get_startup_error, get_transcriber, get_startup_log
from app.schemas.dictation import (
    InputDeviceSchema,
    ListDevicesResponse,
    StartRecordingResponse,
    StatusResponse,
    TranscriptionResponse,
)
from app.services.recorder_devices import list_input_devices
from app.services.debug_audio import save_debug_recording
from app.services.recorder import RecorderState, RecorderStateError
from app.services.transcriber import WhisperTranscriber, WhisperTranscriberError

router = APIRouter(tags=["dictation"])


@router.get("/devices", response_model=ListDevicesResponse)
def get_devices() -> ListDevicesResponse:
    """List all available input devices."""
    devices = list_input_devices(fallback_sample_rate=16000)
    return ListDevicesResponse(
        devices=[
            InputDeviceSchema(
                index=d.index,
                name=d.name,
                sample_rate=d.sample_rate,
                hostapi=d.hostapi_name,
            )
            for d in devices
        ]
    )


@router.get("/status", response_model=StatusResponse)
def get_status(
    request: Request,
    recorder_state: Annotated[RecorderState, Depends(get_recorder_state)],
    active_transcriber: Annotated[WhisperTranscriber | None, Depends(get_optional_transcriber)],
    startup_error: Annotated[str | None, Depends(get_startup_error)],
    startup_log: Annotated[list[str] | None, Depends(get_startup_log)],
) -> StatusResponse:
    devices = list_input_devices(fallback_sample_rate=recorder_state.sample_rate)
    return StatusResponse(
        ready=active_transcriber is not None and startup_error is None,
        device=getattr(active_transcriber, "device", None),
        model_path=getattr(active_transcriber, "model_path", getattr(request.app.state, "model_path", None)),
        recording=recorder_state.is_recording,
        sample_rate=recorder_state.sample_rate,
        startup_error=startup_error,
        startup_log=startup_log,
        debug_recordings_dir=str(DEBUG_RECORDINGS_DIR),
        input_devices=[
            InputDeviceSchema(
                index=d.index,
                name=d.name,
                sample_rate=d.sample_rate,
                hostapi=d.hostapi_name,
            )
            for d in devices
        ],
    )


@router.post("/start_recording", response_model=StartRecordingResponse)
async def start_recording(
    recorder_state: Annotated[RecorderState, Depends(get_recorder_state)],
    active_transcriber: Annotated[WhisperTranscriber, Depends(get_transcriber)],
    device_index: int | None = None,
) -> StartRecordingResponse:
    """Start recording from microphone. Optionally specify device_index to select a specific input device."""
    del active_transcriber
    try:
        recorder_state.start(device_index=device_index)
    except RecorderStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    return StartRecordingResponse(
        recording=True,
        sample_rate=recorder_state.sample_rate,
        input_device=recorder_state.input_device,
    )


@router.post("/stop_recording_and_transcribe", response_model=TranscriptionResponse)
async def stop_recording_and_transcribe(
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

    debug_audio_path = save_debug_recording(audio, active_transcriber.sample_rate)

    try:
        result = await to_thread.run_sync(active_transcriber.transcribe, audio)
    except WhisperTranscriberError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return TranscriptionResponse.model_validate({**result, "debug_audio_path": str(debug_audio_path)})