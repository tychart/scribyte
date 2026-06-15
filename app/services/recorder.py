import threading
import time

import numpy as np
import sounddevice as sd


class RecorderStateError(RuntimeError):
    pass


class RecorderState:
    def __init__(self, sample_rate: int, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._lock = threading.RLock()
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._started_at: float | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            return
        with self._lock:
            self._chunks.append(indata.copy().reshape(-1))

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

    def stop(self) -> np.ndarray:
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