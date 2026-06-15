from typing import Callable

import numpy as np
from numpy.typing import NDArray


class CallbackFlags:
    def __bool__(self) -> bool: ...


class InputStream:
    def __init__(
        self,
        *,
        samplerate: int,
        device: int | None = ...,
        channels: int,
        dtype: str,
        callback: Callable[[NDArray[np.float32], int, object, CallbackFlags], None],
    ) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...