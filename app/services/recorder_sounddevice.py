import threading
import time

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from app.services.recorder_audio import prepare_audio
from app.services.recorder_contract import RecorderStateError
from app.services.recorder_devices import pick_input_device


class RecorderState:
    def __init__(self, sample_rate: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._lock = threading.RLock()
        self._chunks: list[NDArray[np.float32]] = []
        self._stream: sd.InputStream | None = None
        self._started_at: float | None = None
        self._input_device: str | None = None
        self._capture_sample_rate = sample_rate

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def input_device(self) -> str | None:
        return self._input_device

    def _resolve_input_device(self):
        return pick_input_device(fallback_sample_rate=self.sample_rate)

    def _callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info, status
        with self._lock:
            chunk = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
            self._chunks.append(chunk)

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                raise RecorderStateError("Recording is already in progress")

            input_device = self._resolve_input_device()

            self._chunks = []
            self._started_at = time.time()

        stream = None
        try:
            stream = sd.InputStream(
                samplerate=input_device.sample_rate,
                device=input_device.index,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            stream.start()
        except Exception as error:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            with self._lock:
                self._input_device = None
                self._capture_sample_rate = self.sample_rate
                self._started_at = None
            raise RecorderStateError(
                f"Failed to start recording stream for {input_device.name or input_device.index}: {error}"
            ) from error

        with self._lock:
            self._input_device = input_device.name
            self._capture_sample_rate = input_device.sample_rate
            self._stream = stream

    def stop(self) -> NDArray[np.float32]:
        with self._lock:
            if self._stream is None:
                raise RecorderStateError("Recording is not currently running")

            stream = self._stream
            self._stream = None

        stream.stop()
        stream.close()

        with self._lock:
            if not self._chunks:
                self._started_at = None
                self._input_device = None
                self._capture_sample_rate = self.sample_rate
                return np.array([], dtype=np.float32)

            audio = np.concatenate(self._chunks).astype(np.float32, copy=False)
            self._chunks = []
            self._started_at = None
            self._input_device = None
            capture_sample_rate = self._capture_sample_rate
            self._capture_sample_rate = self.sample_rate
            return prepare_audio(audio, capture_sample_rate, self.sample_rate)