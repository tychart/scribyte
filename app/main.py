from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from app.api.routes.dictation import router as dictation_router
from app.core.config import API_DESCRIPTION, API_TITLE, API_VERSION, MODEL_PATH, SAMPLE_RATE
from app.services.recorder import RecorderState, RecorderStateError
from app.services.transcriber import WhisperTranscriber, WhisperTranscriberError


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

        active_recorder = app.state.recorder
        if active_recorder.is_recording:
            try:
                active_recorder.stop()
            except RecorderStateError:
                pass

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.include_router(dictation_router)
    return app


app = create_app()


def main() -> None:
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()