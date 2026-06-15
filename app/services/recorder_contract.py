from dataclasses import dataclass
from typing import Protocol

from numpy.typing import NDArray
import numpy as np


class RecorderStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputDeviceSelection:
    index: int | None
    name: str | None
    sample_rate: int


class Recorder(Protocol):
    sample_rate: int

    @property
    def is_recording(self) -> bool: ...

    @property
    def input_device(self) -> str | None: ...

    def start(self) -> None: ...

    def stop(self) -> NDArray[np.float32]: ...
