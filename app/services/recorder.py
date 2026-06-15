import threading
import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd


class RecorderStateError(RuntimeError):
    pass


class Recorder(Protocol):
    sample_rate: int
    is_recording: bool

    def start(self) -> None: ...

    def stop(self) -> NDArray[np.float32]: ...


class RecorderState:
    def __init__(self, sample_rate: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._lock = threading.RLock()
        self._chunks: list[NDArray[np.float32]] = []
        self._stream: sd.InputStream | None = None
        self._started_at: float | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            return
        with self._lock:
            chunk = np.asarray(indata, dtype=np.float32).reshape(-1)
            self._chunks.append(chunk)

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                raise RecorderStateError("Recording is already in progress")

            self._chunks = []
            self._started_at = time.time()
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()

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
                return np.array([], dtype=np.float32)

            audio = np.concatenate(self._chunks).astype(np.float32, copy=False)
            self._chunks = []
            self._started_at = None
            return audio