from typing import Mapping, Sequence

from app.services.recorder_audio import prepare_audio
from app.services.recorder_contract import InputDeviceSelection, Recorder, RecorderStateError
from app.services.recorder_devices import build_wasapi_input_devices
from app.services.recorder_devices import choose_wasapi_input_device
from app.services.recorder_sounddevice import RecorderState
from app.services.recorder_devices import device_names_match


def pick_wasapi_input_devices(
    default_device: Mapping[str, object] | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> list[InputDeviceSelection]:
    default_name_value = default_device.get("name") if default_device is not None else None
    default_input_name = default_name_value if isinstance(default_name_value, str) else None
    return [
        device.to_selection()
        for device in build_wasapi_input_devices(default_input_name, all_devices, hostapi_names, fallback_sample_rate)
    ]


def pick_wasapi_input_device(
    default_device: Mapping[str, object] | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> InputDeviceSelection | None:
    selections = pick_wasapi_input_devices(default_device, all_devices, hostapi_names, fallback_sample_rate)
    return selections[0] if selections else None


def pick_input_devices(
    default_device: Mapping[str, object] | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> list[InputDeviceSelection]:
    return pick_wasapi_input_devices(default_device, all_devices, hostapi_names, fallback_sample_rate)


def pick_input_device(
    default_device: Mapping[str, object] | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> InputDeviceSelection | None:
    return pick_wasapi_input_device(default_device, all_devices, hostapi_names, fallback_sample_rate)

__all__ = [
    "InputDeviceSelection",
    "Recorder",
    "RecorderState",
    "RecorderStateError",
    "choose_wasapi_input_device",
    "device_names_match",
    "pick_input_device",
    "pick_input_devices",
    "pick_wasapi_input_device",
    "pick_wasapi_input_devices",
    "prepare_audio",
]