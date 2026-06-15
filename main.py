from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import uvicorn

from recorder import RecorderState, RecorderStateError
from transcriber import MODEL_PATH, SAMPLE_RATE, WhisperTranscriber, WhisperTranscriberError


def create_app(
    transcriber: WhisperTranscriber | None = None,
    recorder: RecorderState | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_error = None
        app.state.transcriber = transcriber
        app.state.recorder = recorder

        if app.state.transcriber is None:
            try:
                app.state.transcriber = WhisperTranscriber(model_path=MODEL_PATH, device="NPU")
                app.state.transcriber.warmup()
            except WhisperTranscriberError as error:
                app.state.startup_error = str(error)

        if app.state.recorder is None:
            sample_rate = SAMPLE_RATE
            if app.state.transcriber is not None:
                sample_rate = app.state.transcriber.sample_rate
            app.state.recorder = RecorderState(sample_rate=sample_rate)

        yield

    app = FastAPI(title="Scribyte API", lifespan=lifespan)

    @app.get("/status")
    def get_status() -> dict[str, object]:
        recorder_state: RecorderState = app.state.recorder
        active_transcriber: WhisperTranscriber | None = app.state.transcriber
        startup_error: str | None = app.state.startup_error
        return {
            "ready": active_transcriber is not None and startup_error is None,
            "device": getattr(active_transcriber, "device", None),
            "model_path": getattr(active_transcriber, "model_path", MODEL_PATH),
            "recording": recorder_state.is_recording,
            "sample_rate": recorder_state.sample_rate,
            "startup_error": startup_error,
        }

    @app.post("/start_recording")
    def start_recording() -> dict[str, object]:
        recorder_state: RecorderState = app.state.recorder
        if app.state.transcriber is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.startup_error or "Transcriber is not ready",
            )

        try:
            recorder_state.start()
        except RecorderStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

        return {
            "recording": True,
            "sample_rate": recorder_state.sample_rate,
        }

    @app.post("/stop_recording_and_transcribe")
    def stop_recording_and_transcribe() -> dict[str, object]:
        recorder_state: RecorderState = app.state.recorder
        active_transcriber: WhisperTranscriber | None = app.state.transcriber

        if active_transcriber is None:
            raise HTTPException(
                status_code=503,
                detail=app.state.startup_error or "Transcriber is not ready",
            )

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

        return result

    return app


app = create_app()


def main() -> None:
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
