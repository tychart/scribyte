from contextlib import asynccontextmanager
import logging
import logging.config
import os
from pathlib import Path

from fastapi import FastAPI

from app.api.routes.dictation import router as dictation_router
from app.core.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DEFAULT_MODEL_NAME,
    MODEL_ENV_VAR,
    SAMPLE_RATE,
)
from app.logging_config import LOGGING_CONFIG
from app.services.recorder import Recorder, RecorderState, RecorderStateError
from app.services.transcriber import Transcriber, WhisperTranscriber, WhisperTranscriberError

# Apply the logging configuration at module import time so it takes effect
# for all loggers (uvicorn + scribyte) regardless of how the server is
# started. This must happen before any module-level logging calls.
logging.config.dictConfig(LOGGING_CONFIG)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_FALLBACK_CHAIN = ["NPU", "GPU", "CPU"]


def determine_device_order(limit: str | None) -> list[str]:
    if limit is None:
        return DEVICE_FALLBACK_CHAIN.copy()

    normalized = limit.upper()
    if normalized not in DEVICE_FALLBACK_CHAIN:
        return DEVICE_FALLBACK_CHAIN.copy()

    start_index = DEVICE_FALLBACK_CHAIN.index(normalized)
    return DEVICE_FALLBACK_CHAIN[start_index:]


def determine_model_path(selection: str | None) -> Path:
    normalized = (selection or DEFAULT_MODEL_NAME).strip()
    if not normalized:
        normalized = DEFAULT_MODEL_NAME

    candidate = Path(normalized)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate

    model_name = normalized.lower()
    if model_name.startswith("whisper_") and model_name.endswith("_ov"):
        folder_name = model_name
    else:
        folder_name = f"whisper_{model_name}_ov"

    return PROJECT_ROOT / folder_name


def create_app(
    transcriber: Transcriber | None = None,
    recorder: Recorder | None = None,
    device_limit: str | None = None,
    model_selection: str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger = logging.getLogger("scribyte.startup")
        startup_log: list[str] = []
        app.state.startup_error = None
        app.state.startup_log = startup_log
        app.state.transcriber = transcriber
        app.state.recorder = recorder
        selected_model_path = determine_model_path(model_selection or os.environ.get(MODEL_ENV_VAR))
        app.state.model_path = str(selected_model_path)
        logger.info("Configured model path: %s", selected_model_path)
        startup_log.append(f"Configured model path: {selected_model_path}")

        # determine device preference order
        device_order = determine_device_order(device_limit or os.environ.get("SCRIBYTE_LIMIT"))
        logger.info("Device selection order: %s", ",".join(device_order))
        startup_log.append(f"Device selection order: {', '.join(device_order)}")

        if app.state.transcriber is None:
            last_exc: Exception | None = None
            selected_device: str | None = None
            runtime_name: str | None = None

            for i, device in enumerate(device_order):
                next_device = device_order[i + 1] if i + 1 < len(device_order) else None

                logger.info("Initializing transcriber on %s", device)
                startup_log.append(f"Initializing transcriber on {device}")
                try:
                    app.state.transcriber = WhisperTranscriber(model_path=str(selected_model_path), device=device)
                    app.state.transcriber.warmup()
                    runtime_name = getattr(app.state.transcriber, "runtime_device_name", None)
                    selected_device = device
                    logger.info(
                        "Initialized transcriber on %s (runtime device: %s, model: %s)",
                        device,
                        runtime_name,
                        selected_model_path,
                    )
                    startup_log.append(
                        f"Initialized transcriber on {device} "
                        f"(runtime device: {runtime_name}, model: {selected_model_path})"
                    )
                    break
                except WhisperTranscriberError as error:
                    last_exc = error
                    fallback_to = f", falling back to {next_device}" if next_device else ""
                    concise_msg = f"{device} not available{fallback_to}: {error}"
                    logger.warning(concise_msg)
                    logger.debug("%s — full trace:", error, exc_info=True)
                    startup_log.append(f"{device} not available{fallback_to}: {error}")

            if selected_device is not None:
                logger.info(
                    "Selected transcriber device: %s (runtime: %s)",
                    selected_device,
                    runtime_name,
                )
                startup_log.append(f"Selected device: {selected_device} (runtime: {runtime_name})")
            else:
                app.state.startup_error = str(last_exc) if last_exc is not None else "Unknown transcriber error"
                logger.error("All device attempts failed. Startup error: %s", app.state.startup_error)
                startup_log.append(f"Startup error: {app.state.startup_error}")

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


_device_limit: str | None = os.environ.get("SCRIBYTE_LIMIT")

app = create_app(device_limit=_device_limit)