import numpy as np
from numpy.typing import NDArray


def split(
    y: NDArray[np.float32],
    *,
    top_db: float = ...,
    ref: float | object = ...,
    frame_length: int = ...,
    hop_length: int = ...,
    aggregate: object = ...,
) -> NDArray[np.int_]: ...