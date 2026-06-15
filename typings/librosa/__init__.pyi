import numpy as np
from numpy.typing import NDArray

from . import effects as effects


def load(path: str, *, sr: int, mono: bool = ...) -> tuple[NDArray[np.float32], int]: ...


def resample(y: NDArray[np.float32], *, orig_sr: int, target_sr: int) -> NDArray[np.float32]: ...