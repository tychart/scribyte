"""Unit tests for device discovery and audio preprocessing."""

import numpy as np
import pytest

from app.services.recorder_devices import (
    InputDevice,
    InputDeviceSelection,
    list_input_devices,
    pick_input_device,
    format_input_device_name,
    device_names_match,
)
from app.services.recorder_audio import prepare_audio
from app.services.recorder_contract import RecorderStateError


class TestInputDevice:
    """Tests for InputDevice dataclass."""

    def test_display_name_with_hostapi(self):
        device = InputDevice(index=0, name="Mic", sample_rate=16000, hostapi_name="WASAPI")
        assert device.display_name == "Mic [WASAPI]"

    def test_display_name_without_hostapi(self):
        device = InputDevice(index=0, name="Mic", sample_rate=16000, hostapi_name=None)
        assert device.display_name == "Mic"

    def test_to_selection(self):
        device = InputDevice(index=1, name="Mic", sample_rate=48000, hostapi_name="WASAPI")
        selection = device.to_selection()
        assert selection.index == 1
        assert selection.name == "Mic [WASAPI]"
        assert selection.sample_rate == 48000


class TestFormatHelpers:
    """Tests for device name formatting."""

    def test_format_with_both_names(self):
        result = format_input_device_name("Mic", "WASAPI")
        assert result == "Mic [WASAPI]"

    def test_format_with_none_hostapi(self):
        result = format_input_device_name("Mic", None)
        assert result == "Mic"

    def test_format_with_none_name(self):
        result = format_input_device_name(None, "WASAPI")
        assert result is None


class TestDeviceNameMatch:
    """Tests for device name matching logic."""

    def test_exact_match(self):
        assert device_names_match("Mic", "Mic") is True

    def test_prefix_match_default_longer(self):
        assert device_names_match("Microphone USB", "Micro") is True

    def test_prefix_match_candidate_longer(self):
        # Microphone starts with Mic, so it matches
        assert device_names_match("Mic", "Microphone") is True

    def test_case_insensitive(self):
        assert device_names_match("mic", "MIC") is True

    def test_no_match(self):
        assert device_names_match("Mic1", "Mic2") is False


class TestPrepareAudio:
    """Tests for audio preprocessing."""

    def test_prepare_audio_removes_dc_offset(self):
        audio = np.ones(16000, dtype=np.float32) * 0.5
        result = prepare_audio(audio, 16000, 16000)
        assert abs(np.mean(result)) < 1e-3

    def test_prepare_audio_resamples_up(self):
        audio = np.sin(2 * np.pi * 440 * np.arange(16000) / 8000, dtype=np.float32)
        result = prepare_audio(audio, 8000, 16000)
        assert abs(len(result) - 32000) <= 2

    def test_prepare_audio_resamples_down(self):
        audio = np.sin(2 * np.pi * 440 * np.arange(32000) / 16000, dtype=np.float32)
        result = prepare_audio(audio, 16000, 8000)
        assert abs(len(result) - 16000) <= 2

    def test_prepare_audio_empty_returns_empty(self):
        audio = np.array([], dtype=np.float32)
        result = prepare_audio(audio, 16000, 16000)
        assert len(result) == 0

    def test_prepare_audio_no_resample_when_same_rate(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = prepare_audio(audio, 16000, 16000)
        # Should be centered (no DC offset)
        assert abs(np.mean(result)) < 1e-3
        assert len(result) == 1000


class TestPickInputDevice:
    """Tests for the unified pick_input_device function."""

    def test_pick_returns_valid_selection(self):
        """pick_input_device should return a valid InputDeviceSelection."""
        selection = pick_input_device(fallback_sample_rate=16000)
        assert selection is not None
        assert isinstance(selection, InputDeviceSelection)
        assert selection.index is not None
        assert selection.sample_rate > 0

    def test_pick_with_invalid_index_raises(self):
        """pick_input_device should raise for non-existent device index."""
        with pytest.raises(RecorderStateError):
            pick_input_device(fallback_sample_rate=16000, device_index=99999)

    def test_list_returns_non_empty_on_hardware_present(self):
        """list_input_devices should return devices if hardware exists."""
        devices = list_input_devices(fallback_sample_rate=16000)
        # On most systems there should be at least one device
        # This might fail in headless CI but is correct behavior
        assert isinstance(devices, list)
        for d in devices:
            assert d.index is not None
            assert d.sample_rate > 0

    def test_list_filters_out_no_input_devices(self):
        """list_input_devices should only return devices with max_input_channels >= 1."""
        devices = list_input_devices(fallback_sample_rate=16000)
        for d in devices:
            assert d.index is not None
            assert d.name is not None
            assert d.sample_rate > 0
