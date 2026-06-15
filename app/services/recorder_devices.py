from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import sounddevice as sd

from app.services.recorder_contract import InputDeviceSelection, RecorderStateError


@dataclass(frozen=True)
class WasapiInputDevice:
    index: int
    name: str
    sample_rate: int
    hostapi_name: str
    is_default_match: bool

    @property
    def display_name(self) -> str:
        return format_input_device_name(self.name, self.hostapi_name) or self.name

    def to_selection(self) -> InputDeviceSelection:
        return InputDeviceSelection(
            index=self.index,
            name=self.display_name,
            sample_rate=self.sample_rate,
        )


def coerce_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def coerce_sample_rate(value: object, fallback: int) -> int:
    return int(round(value)) if isinstance(value, int | float) and value > 0 else fallback


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


def _query_devices() -> list[dict[str, object]]:
    query_devices = cast(Any, getattr(sd, "query_devices", None))
    if query_devices is None:
        raise RecorderStateError("sounddevice.query_devices is unavailable")

    devices = list(query_devices())
    return [cast(dict[str, object], device) for device in devices if isinstance(device, dict)]


def _query_hostapi_names() -> dict[int, str]:
    query_hostapis = cast(Any, getattr(sd, "query_hostapis", None))
    if query_hostapis is None:
        raise RecorderStateError("sounddevice.query_hostapis is unavailable")

    hostapis = list(query_hostapis())
    hostapi_names: dict[int, str] = {}
    for index, hostapi in enumerate(hostapis):
        if not isinstance(hostapi, dict):
            continue
        typed_hostapi = cast(dict[str, object], hostapi)
        hostapi_name = typed_hostapi.get("name")
        if isinstance(hostapi_name, str):
            hostapi_names[index] = hostapi_name
    return hostapi_names


def _query_default_input_name() -> str | None:
    query_devices = cast(Any, getattr(sd, "query_devices", None))
    if query_devices is None:
        return None

    try:
        default_input = query_devices(kind="input")
    except Exception:
        return None

    if not isinstance(default_input, dict):
        return None

    typed_default_input = cast(dict[str, object], default_input)
    device_name = typed_default_input.get("name")
    return device_name if isinstance(device_name, str) else None


def build_wasapi_input_devices(
    default_input_name: str | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> list[WasapiInputDevice]:
    wasapi_inputs: list[WasapiInputDevice] = []

    for device_mapping in all_devices:
        device = dict(device_mapping)
        hostapi_index = coerce_int(device.get("hostapi"))
        hostapi_name = hostapi_names.get(hostapi_index) if hostapi_index is not None else None
        if hostapi_name is None or "WASAPI" not in hostapi_name.upper():
            continue

        max_input_channels = coerce_int(device.get("max_input_channels")) or 0
        if max_input_channels < 1:
            continue

        device_index = coerce_int(device.get("index"))
        device_name = device.get("name")
        if device_index is None or not isinstance(device_name, str):
            continue

        device_sample_rate = coerce_sample_rate(device.get("default_samplerate"), fallback_sample_rate)
        is_default_match = default_input_name is not None and device_names_match(default_input_name, device_name)
        wasapi_inputs.append(
            WasapiInputDevice(
                index=device_index,
                name=device_name,
                sample_rate=device_sample_rate,
                hostapi_name=hostapi_name,
                is_default_match=is_default_match,
            )
        )

    return sorted(wasapi_inputs, key=lambda device: (not device.is_default_match, device.index))


def list_wasapi_input_devices(fallback_sample_rate: int) -> list[WasapiInputDevice]:
    return build_wasapi_input_devices(
        default_input_name=_query_default_input_name(),
        all_devices=_query_devices(),
        hostapi_names=_query_hostapi_names(),
        fallback_sample_rate=fallback_sample_rate,
    )


def choose_wasapi_input_device(
    fallback_sample_rate: int,
    device_index: int | None = None,
) -> InputDeviceSelection:
    devices = list_wasapi_input_devices(fallback_sample_rate)
    if not devices:
        raise RecorderStateError("No WASAPI input devices were found")

    if device_index is None:
        return devices[0].to_selection()

    for device in devices:
        if device.index == device_index:
            return device.to_selection()

    raise RecorderStateError(f"WASAPI input device {device_index} was not found")
