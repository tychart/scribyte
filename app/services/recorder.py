import threading
import time
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd


class RecorderStateError(RuntimeError):
    pass


class Recorder(Protocol):
    sample_rate: int
    @property
    def is_recording(self) -> bool: ...

    @property
    def input_device(self) -> str | None: ...

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
        self._input_device: str | None = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def input_device(self) -> str | None:
        return self._input_device

    def _resolve_input_device(self) -> str | None:
        query_devices = cast(Any, getattr(sd, "query_devices", None))
        if query_devices is None:
            return None

        try:
            device_info = cast(object, query_devices(kind="input"))
        except Exception:
            return None

        if isinstance(device_info, dict):
            typed_device_info = cast(dict[str, object], device_info)
            device_name = typed_device_info.get("name")
            return device_name if isinstance(device_name, str) else None

        return None

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
            self._input_device = self._resolve_input_device()
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
                self._input_device = None
                return np.array([], dtype=np.float32)

            audio = np.concatenate(self._chunks).astype(np.float32, copy=False)
            self._chunks = []
            self._started_at = None
            self._input_device = None
            return audio