import librosa
import numpy as np
from numpy.typing import NDArray


def prepare_audio(
    audio: NDArray[np.float32],
    capture_sample_rate: int,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    normalized_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if normalized_audio.size == 0:
        return normalized_audio

    centered_audio = normalized_audio - np.float32(np.mean(normalized_audio))
    if capture_sample_rate == target_sample_rate:
        return centered_audio.astype(np.float32, copy=False)

    resampled_audio = librosa.resample(
        centered_audio,
        orig_sr=capture_sample_rate,
        target_sr=target_sample_rate,
    )
    return np.asarray(resampled_audio, dtype=np.float32)