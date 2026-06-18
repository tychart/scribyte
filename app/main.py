from contextlib import asynccontextmanager
import logging
import os
import sys

from fastapi import FastAPI
import uvicorn

from app.api.routes.dictation import router as dictation_router
from app.core.config import API_DESCRIPTION, API_TITLE, API_VERSION, MODEL_PATH, SAMPLE_RATE
from app.services.recorder import Recorder, RecorderState, RecorderStateError
from app.services.transcriber import Transcriber, WhisperTranscriber, WhisperTranscriberError


def _determine_device_order(limit: str | None) -> list[str]:
    if limit is None:
        return ["NPU", "GPU", "CPU"]
    limit = limit.lower()
    if limit == "gpu":
        return ["GPU", "CPU"]
    if limit == "cpu":
        return ["CPU"]
    return ["NPU", "GPU", "CPU"]


def create_app(
    transcriber: Transcriber | None = None,
    recorder: Recorder | None = None,
    device_limit: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger = logging.getLogger("scribyte.startup")
        startup_log: list[str] = []
        app.state.startup_error = None
        app.state.startup_log = startup_log
        app.state.transcriber = transcriber
        app.state.recorder = recorder

        # determine device preference order
        device_order = _determine_device_order(device_limit or os.environ.get("SCRIBYTE_LIMIT"))
        logger.info("Device selection order: %s", ",".join(device_order))
        startup_log.append(f"Device selection order: {', '.join(device_order)}")

        if app.state.transcriber is None:
            last_exc: Exception | None = None
            for device in device_order:
                startup_log.append(f"Attempting to initialize transcriber on {device}")
                try:
                    app.state.transcriber = WhisperTranscriber(model_path=MODEL_PATH, device=device)
                    app.state.transcriber.warmup()
                    runtime_name = getattr(app.state.transcriber, "runtime_device_name", None)
                    startup_log.append(
                        f"Initialized transcriber on {device} (runtime device: {runtime_name})"
                    )
                    logger.info("Initialized transcriber on %s (%s)", device, runtime_name)
                    break
                except WhisperTranscriberError as error:
                    last_exc = error
                    msg = f"Failed to initialize on {device}: {error}"
                    startup_log.append(msg)
                    logger.warning(msg)

            if getattr(app.state, "transcriber", None) is None:
                app.state.startup_error = str(last_exc) if last_exc is not None else "Unknown transcriber error"

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


_device_limit: str | None = None
if "--gpu" in sys.argv:
    _device_limit = "gpu"
elif "--cpu" in sys.argv:
    _device_limit = "cpu"
else:
    _device_limit = os.environ.get("SCRIBYTE_LIMIT")

app = create_app(device_limit=_device_limit)


def main() -> None:
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()