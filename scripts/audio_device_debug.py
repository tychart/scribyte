#!/usr/bin/env python3
"""Cross-platform audio device debug tool.

List and record from audio input devices using sounddevice.
Works identically on Windows (WASAPI), Linux (ALSA/PulseAudio/PipeWire), and macOS.

Examples:
    # List all input devices:
    uv run python scripts/audio_device_debug.py --list

    # Record 5 seconds from the default input device:
    uv run python scripts/audio_device_debug.py

    # Record from a specific device index:
    uv run python scripts/audio_device_debug.py --device-index 3 --seconds 8
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import tempfile
import time
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd
import soundfile as sf

from app.services.recorder_devices import (
    InputDeviceSelection,
    list_input_devices,
    pick_input_device,
)
from app.services.recorder_audio import prepare_audio

TARGET_SAMPLE_RATE = 16000
OUTPUT_DIR = Path(tempfile.gettempdir()) / "scribyte-audio-debug"


def describe_audio(audio: NDArray[np.float32]) -> str:
    if audio.size == 0:
        return "samples=0"

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    mean = float(np.mean(audio))
    duration_seconds = audio.size / TARGET_SAMPLE_RATE
    return (
        f"samples={audio.size} duration_seconds={duration_seconds:.2f} "
        f"peak={peak:.4f} rms={rms:.4f} mean={mean:.4f}"
    )


def _timestamp_prefix() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:8]}"


def record_audio(device: InputDeviceSelection, seconds: float) -> tuple[Path, Path]:
    """Record audio from the given device and return raw + prepared WAV paths."""
    if seconds <= 0:
        raise RuntimeError("Recording duration must be greater than zero")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _timestamp_prefix()
    raw_output = OUTPUT_DIR / f"{prefix}-raw.wav"
    prepared_output = OUTPUT_DIR / f"{prefix}-prepared-16k.wav"

    chunks: list[NDArray[np.float32]] = []

    def callback(
        indata: NDArray[np.float32],
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            print(f"  callback_status={status}")
        chunks.append(np.asarray(indata, dtype=np.float32).reshape(-1).copy())

    stream = sd.InputStream(
        device=device.index,
        samplerate=device.sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    )

    print(
        f"recording_from_index={device.index} name={device.name!r} "
        f"native_sample_rate={device.sample_rate} duration_seconds={seconds:.2f}"
    )
    stream.start()
    try:
        time.sleep(seconds)
    finally:
        stream.stop()
        stream.close()

    raw_audio = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.array([], dtype=np.float32)
    prepared_audio = prepare_audio(raw_audio, device.sample_rate, TARGET_SAMPLE_RATE)

    sf.write(raw_output, raw_audio, device.sample_rate)
    sf.write(prepared_output, prepared_audio, TARGET_SAMPLE_RATE)

    print(f"raw_output={raw_output}")
    print(f"prepared_output={prepared_output}")
    print(f"prepared_audio_stats {describe_audio(prepared_audio)}")
    return raw_output, prepared_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-platform audio device debug tool")
    parser.add_argument("--list", action="store_true", help="List detected input devices and exit")
    parser.add_argument("--device-index", type=int, help="Input device index to record from")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording length in seconds")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    devices = list_input_devices(fallback_sample_rate=TARGET_SAMPLE_RATE)

    if not devices:
        print("No input devices found. Is a microphone connected?")
        return

    print("Detected input devices:")
    for device in devices:
        print(
            f"  index={device.index} sample_rate={device.sample_rate} name={device.display_name}"
        )

    if args.list:
        return

    device = pick_input_device(fallback_sample_rate=TARGET_SAMPLE_RATE, device_index=args.device_index)
    record_audio(device, args.seconds)


if __name__ == "__main__":
    main()
