import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, cast

import librosa
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd


class RecorderStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputDeviceSelection:
    index: int | None
    name: str | None
    sample_rate: int


def hostapi_priority(hostapi_name_value: str | None) -> int:
    if hostapi_name_value is None:
        return 99
    normalized_name = hostapi_name_value.upper()
    if "WASAPI" in normalized_name:
        return 0
    if "DIRECTSOUND" in normalized_name:
        return 1
    if "MME" in normalized_name:
        return 2
    return 3


def coerce_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def coerce_sample_rate(value: object, fallback: int) -> int:
    return int(round(value)) if isinstance(value, int | float) and value > 0 else fallback


def hostapi_name(hostapi_names: Mapping[int, str], device_info: Mapping[str, object]) -> str | None:
    hostapi_index = coerce_int(device_info.get("hostapi"))
    return hostapi_names.get(hostapi_index) if hostapi_index is not None else None


def format_input_device_name(device_name: str | None, hostapi_name_value: str | None) -> str | None:
    if device_name is None:
        return None
    if hostapi_name_value is None:
        return device_name
    return f"{device_name} [{hostapi_name_value}]"


def device_names_match(default_device_name: str, candidate_device_name: str) -> bool:
    normalized_default_name = default_device_name.strip().lower()
    normalized_candidate_name = candidate_device_name.strip().lower()
    return normalized_default_name.startswith(normalized_candidate_name) or normalized_candidate_name.startswith(
        normalized_default_name
    )


def pick_input_devices(
    default_device: Mapping[str, object] | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> list[InputDeviceSelection]:
    if default_device is None:
        return [InputDeviceSelection(index=None, name=None, sample_rate=fallback_sample_rate)]

    default_name = default_device.get("name")
    default_device_name = default_name if isinstance(default_name, str) else None
    default_hostapi_name = hostapi_name(hostapi_names, default_device)
    default_index = coerce_int(default_device.get("index"))
    default_sample_rate = coerce_sample_rate(
        default_device.get("default_samplerate"),
        fallback_sample_rate,
    )

    candidates: list[tuple[int, InputDeviceSelection]] = []

    if default_device_name is not None:
        for device_mapping in all_devices:
            device = dict(device_mapping)
            device_name = device.get("name")
            if not isinstance(device_name, str) or not device_names_match(default_device_name, device_name):
                continue

            max_input_channels = coerce_int(device.get("max_input_channels")) or 0
            if max_input_channels < 1:
                continue

            device_hostapi_name = hostapi_name(hostapi_names, device)
            if device_hostapi_name is None:
                continue

            device_index = coerce_int(device.get("index"))
            device_sample_rate = coerce_sample_rate(
                device.get("default_samplerate"),
                default_sample_rate,
            )
            candidates.append(
                (
                    hostapi_priority(device_hostapi_name),
                    InputDeviceSelection(
                        index=device_index,
                        name=format_input_device_name(device_name, device_hostapi_name),
                        sample_rate=device_sample_rate,
                    ),
                )
            )

    default_selection = InputDeviceSelection(
        index=default_index,
        name=format_input_device_name(default_device_name, default_hostapi_name),
        sample_rate=default_sample_rate,
    )
    ordered_candidates = [selection for _, selection in sorted(candidates, key=lambda item: item[0])]
    if not ordered_candidates:
        return [default_selection]

    if all(candidate.index != default_selection.index for candidate in ordered_candidates):
        ordered_candidates.append(default_selection)

    return ordered_candidates


def pick_input_device(
    default_device: Mapping[str, object] | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> InputDeviceSelection:
    return pick_input_devices(default_device, all_devices, hostapi_names, fallback_sample_rate)[0]


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
        self._capture_sample_rate = sample_rate

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def input_device(self) -> str | None:
        return self._input_device

    def _resolve_input_device(self) -> InputDeviceSelection:
        query_devices = cast(Any, getattr(sd, "query_devices", None))
        query_hostapis = cast(Any, getattr(sd, "query_hostapis", None))
        if query_devices is None:
            return InputDeviceSelection(index=None, name=None, sample_rate=self.sample_rate)

        try:
            device_info = cast(object, query_devices(kind="input"))
        except Exception:
            return InputDeviceSelection(index=None, name=None, sample_rate=self.sample_rate)

        hostapi_names: dict[int, str] = {}
        if query_hostapis is not None:
            try:
                hostapis = list(query_hostapis())
            except Exception:
                hostapis = None
            if isinstance(hostapis, list):
                typed_hostapis = cast(list[object], hostapis)
                for index, hostapi in enumerate(typed_hostapis):
                    if isinstance(hostapi, dict):
                        typed_hostapi = cast(dict[str, object], hostapi)
                        hostapi_name = typed_hostapi.get("name")
                        if isinstance(hostapi_name, str):
                            hostapi_names[index] = hostapi_name

        if isinstance(device_info, dict):
            typed_device_info = cast(dict[str, object], device_info)
            all_devices: list[dict[str, object]] = []
            try:
                devices = list(query_devices())
            except Exception:
                devices = None
            if isinstance(devices, list):
                typed_devices = cast(list[object], devices)
                all_devices = [cast(dict[str, object], device) for device in typed_devices if isinstance(device, dict)]
            return pick_input_device(
                typed_device_info,
                all_devices,
                hostapi_names,
                self.sample_rate,
            )

        return InputDeviceSelection(index=None, name=None, sample_rate=self.sample_rate)

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
            query_devices = cast(Any, getattr(sd, "query_devices", None))
            query_hostapis = cast(Any, getattr(sd, "query_hostapis", None))

            input_candidates = [self._resolve_input_device()]
            if query_devices is not None and query_hostapis is not None:
                try:
                    default_device_info = cast(object, query_devices(kind="input"))
                    hostapis = list(query_hostapis())
                    devices = list(query_devices())
                except Exception:
                    default_device_info = None
                    hostapis = []
                    devices = []
                if isinstance(default_device_info, dict):
                    hostapi_names = {
                        index: typed_hostapi.get("name")
                        for index, hostapi in enumerate(hostapis)
                        if isinstance(hostapi, dict)
                        for typed_hostapi in [cast(dict[str, object], hostapi)]
                        if isinstance(typed_hostapi.get("name"), str)
                    }
                    input_candidates = pick_input_devices(
                        cast(dict[str, object], default_device_info),
                        [cast(dict[str, object], device) for device in devices if isinstance(device, dict)],
                        cast(dict[int, str], hostapi_names),
                        self.sample_rate,
                    )

            errors: list[str] = []
            for input_device in input_candidates:
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
                    self._input_device = input_device.name
                    self._capture_sample_rate = input_device.sample_rate
                    self._stream = stream
                    return
                except Exception as error:
                    errors.append(f"{input_device.name or input_device.index}: {error}")
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass

            self._input_device = None
            self._capture_sample_rate = self.sample_rate
            raise RecorderStateError("Failed to start recording stream. Tried: " + " | ".join(errors))

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
            capture_sample_rate = self._capture_sample_rate
            self._capture_sample_rate = self.sample_rate
            return prepare_audio(audio, capture_sample_rate, self.sample_rate)