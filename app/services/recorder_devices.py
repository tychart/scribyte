from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import sounddevice as sd

from app.services.recorder_contract import InputDeviceSelection, RecorderStateError

# sounddevice query_devices/query_hostapis may not be in stubs
_SdQueryDevices = Any  # type: ignore
_SdQueryHostApis = Any  # type: ignore


def _query_devices() -> list[dict[str, Any]]:
    """Query all sounddevice devices. Cross-platform."""
    query_func = getattr(sd, "query_devices", None)
    if query_func is None:
        return []
    devices = list(query_func())  # type: ignore[arg-type]
    return [dict(d) for d in devices if isinstance(d, dict)]  # type: ignore[arg-type]


def _query_hostapi_names() -> dict[int, str]:
    """Query host API names. Cross-platform."""
    query_func = getattr(sd, "query_hostapis", None)
    if query_func is None:
        return {}
    hostapis = list(query_func())  # type: ignore[arg-type]
    hostapi_names: dict[int, str] = {}
    for idx, hostapi in enumerate(hostapis):  # type: ignore[type-arg]
        if not isinstance(hostapi, dict):
            continue
        hostapi_typed: dict[str, Any] = hostapi  # type: ignore[reportUnknownVariableType]
        name = hostapi_typed.get("name")  # type: ignore[union-attr, arg-type]
        if isinstance(name, str):
            hostapi_names[idx] = name
    return hostapi_names


def _coerce_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _coerce_sample_rate(value: object, fallback: int) -> int:
    if isinstance(value, (int, float)) and value > 0:
        return int(round(value))
    return fallback


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    sample_rate: int
    hostapi_name: str | None

    @property
    def display_name(self) -> str:
        if self.hostapi_name:
            return f"{self.name} [{self.hostapi_name}]"
        return self.name

    def to_selection(self) -> InputDeviceSelection:
        return InputDeviceSelection(
            index=self.index,
            name=self.display_name,
            sample_rate=self.sample_rate,
        )


def list_input_devices(fallback_sample_rate: int) -> list[InputDevice]:
    """List all available input devices across all host APIs. Cross-platform."""
    all_devices = _query_devices()
    hostapi_names = _query_hostapi_names()

    inputs: list[InputDevice] = []
    for device in all_devices:
        hostapi_idx = _coerce_int(device.get("hostapi"))
        hostapi_name = hostapi_names.get(hostapi_idx) if hostapi_idx is not None else None

        max_input_channels = _coerce_int(device.get("max_input_channels")) or 0
        if max_input_channels < 1:
            continue

        device_index = _coerce_int(device.get("index"))
        device_name = device.get("name")
        if device_index is None or not isinstance(device_name, str):
            continue

        device_sample_rate = _coerce_sample_rate(device.get("default_samplerate"), fallback_sample_rate)

        inputs.append(
            InputDevice(
                index=device_index,
                name=device_name,
                sample_rate=device_sample_rate,
                hostapi_name=hostapi_name,
            )
        )

    return sorted(inputs, key=lambda d: d.index)


def pick_input_device(fallback_sample_rate: int, device_index: int | None = None) -> InputDeviceSelection:
    """
    Unified device selection using sounddevice. Works on Windows and Linux.

    - If device_index is specified, returns that exact device.
    - Otherwise, prefers sd.default.input (the OS default) and falls back to
      the first available input device.
    """
    devices = list_input_devices(fallback_sample_rate)

    if not devices:
        raise RecorderStateError("No input devices were found")

    if device_index is not None:
        for device in devices:
            if device.index == device_index:
                return device.to_selection()
        raise RecorderStateError(f"Device index {device_index} was not found")

    # Prefer system default input
    _sd_default = getattr(sd, "default", None)
    default_input: object = None
    if _sd_default is not None:
        _default_attr: object = getattr(_sd_default, "input", None)  # type: ignore[arg-type]
        default_input = _default_attr
    if isinstance(default_input, int):
        for device in devices:
            if device.index == default_input:
                return device.to_selection()

    # Fall back to first available
    return devices[0].to_selection()


# Keep old WASAPI-specific functions for backward compatibility with scripts/tests
# They are deprecated but still available for the old wasapi_debug.py script.

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


def build_wasapi_input_devices(
    default_input_name: str | None,
    all_devices: Sequence[Mapping[str, object]],
    hostapi_names: Mapping[int, str],
    fallback_sample_rate: int,
) -> list[WasapiInputDevice]:
    wasapi_inputs: list[WasapiInputDevice] = []

    for device_mapping in all_devices:
        device = dict(device_mapping)
        hostapi_index = _coerce_int(device.get("hostapi"))
        hostapi_name = hostapi_names.get(hostapi_index) if hostapi_index is not None else None
        if hostapi_name is None or "WASAPI" not in hostapi_name.upper():
            continue

        max_input_channels = _coerce_int(device.get("max_input_channels")) or 0
        if max_input_channels < 1:
            continue

        device_index = _coerce_int(device.get("index"))
        device_name = device.get("name")
        if device_index is None or not isinstance(device_name, str):
            continue

        device_sample_rate = _coerce_sample_rate(device.get("default_samplerate"), fallback_sample_rate)
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

    return sorted(wasapi_inputs, key=lambda d: (not d.is_default_match, d.index))


def list_wasapi_input_devices(fallback_sample_rate: int) -> list[WasapiInputDevice]:
    default_input_name = None
    query_func = getattr(sd, "query_devices", None)
    if query_func is not None:
        try:
            default_input = query_func(kind="input")  # type: ignore[arg-type]
            if isinstance(default_input, dict):
                name = default_input.get("name")  # type: ignore[union-attr, arg-type]
                if isinstance(name, str):
                    default_input_name = name
        except Exception:
            pass

    _all_devs = _query_devices()  # type: ignore[return-value, assignment]
    return build_wasapi_input_devices(
        default_input_name=default_input_name,
        all_devices=_all_devs,
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


__all__ = [
    "InputDeviceSelection",
    "InputDevice",
    "RecorderStateError",
    "WasapiInputDevice",
    "list_input_devices",
    "pick_input_device",
    "build_wasapi_input_devices",
    "list_wasapi_input_devices",
    "choose_wasapi_input_device",
    "format_input_device_name",
    "device_names_match",
]
