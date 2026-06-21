#!/usr/bin/env python3

import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from evdev import InputDevice, categorize, ecodes, list_devices


SCRIBYTE_API_URL = "http://127.0.0.1:8000"
SCRIBYTE_HOLD_KEY = ecodes.KEY_F8

scribyte_is_recording = False


def api_request(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    data = None
    headers = {"Accept": "application/json"}

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        SCRIBYTE_API_URL + path,
        data=data,
        method=method,
        headers=headers,
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw or "{}")
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw or "{}")
        except json.JSONDecodeError:
            parsed = {"detail": raw}
        return err.code, parsed


def notify(message: str, timeout_ms: int = 3000) -> None:
    subprocess.run(
        ["notify-send", "-t", str(timeout_ms), "Scribyte", message],
        check=False,
    )


def paste_text_wayland(text: str) -> None:
    # Put text on Wayland clipboard.
    subprocess.run(
        ["wl-copy"],
        input=text.encode("utf-8"),
        check=True,
    )

    # Give clipboard ownership a tiny moment to settle.
    time.sleep(0.05)

    # Send Ctrl+V via ydotool.
    # Linux input key codes: LEFTCTRL = 29, V = 47.
    subprocess.run(
        ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
        check=True,
    )


def get_error_message(body: dict[str, Any]) -> str:
    detail = body.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    return "Unexpected API response."


def refresh_recording_state() -> None:
    global scribyte_is_recording

    try:
        status, body = api_request("GET", "/status")
    except Exception:
        scribyte_is_recording = False
        return

    if status != 200:
        scribyte_is_recording = False
        return

    scribyte_is_recording = bool(body.get("recording", False))


def start_dictation() -> None:
    global scribyte_is_recording

    if scribyte_is_recording:
        return

    try:
        status, body = api_request("POST", "/start_recording")
    except Exception as err:
        notify(f"Could not reach Scribyte API.\n{err}", 5000)
        return

    if status != 200:
        notify(f"Start failed.\n{get_error_message(body)}", 5000)
        refresh_recording_state()
        return

    scribyte_is_recording = bool(body.get("recording", True))
    sample_rate = body.get("sample_rate", 0)

    message = "Recording..."
    if sample_rate:
        message += f"\n{sample_rate} Hz"

    notify(message, 1800)


def stop_dictation() -> None:
    global scribyte_is_recording

    if not scribyte_is_recording:
        return

    try:
        status, body = api_request("POST", "/stop_recording_and_transcribe")
    except Exception as err:
        refresh_recording_state()
        notify(f"Could not reach Scribyte API.\n{err}", 5000)
        return

    if status != 200:
        refresh_recording_state()
        notify(f"Transcription failed.\n{get_error_message(body)}", 5000)
        return

    scribyte_is_recording = False

    text = str(body.get("text", "")).strip()
    if not text:
        notify("Transcription was empty.")
        return

    paste_text_wayland(text)

    chunk_count = body.get("chunk_count", 0)
    latency_seconds = body.get("latency_seconds", 0)

    summary = "Pasted dictation"
    if chunk_count:
        summary += f"\nChunks: {chunk_count}"
    if latency_seconds:
        summary += f" | Latency: {latency_seconds:.2f}s"

    notify(summary)


def pick_keyboard() -> InputDevice:
    candidates: list[InputDevice] = []

    for path in list_devices():
        dev = InputDevice(path)
        caps = dev.capabilities().get(ecodes.EV_KEY, [])
        if SCRIBYTE_HOLD_KEY in caps:
            candidates.append(dev)

    if not candidates:
        raise RuntimeError("No keyboard device exposing F8 was found.")

    print("Keyboard candidates:")
    for index, dev in enumerate(candidates):
        print(f"{index}: {dev.path} | {dev.name}")

    return candidates[0]


def main() -> None:
    refresh_recording_state()

    dev = pick_keyboard()
    print(f"Listening for F8 on {dev.path} | {dev.name}")
    notify("Scribyte hotkey loaded. Hold F8 to dictate.")

    for event in dev.read_loop():
        if event.type != ecodes.EV_KEY:
            continue

        key_event = categorize(event)

        if key_event.scancode != SCRIBYTE_HOLD_KEY:
            continue

        # value: 1 = key down, 0 = key up, 2 = repeat
        if key_event.keystate == key_event.key_down:
            start_dictation()
        elif key_event.keystate == key_event.key_up:
            stop_dictation()


if __name__ == "__main__":
    main()