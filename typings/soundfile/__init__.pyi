from os import PathLike
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

PathLikeStr: TypeAlias = str | PathLike[str]


def write(file: PathLikeStr, data: NDArray[np.float32], samplerate: int) -> None: ...