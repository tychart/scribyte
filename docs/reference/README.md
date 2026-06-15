# Reference Code

This folder holds prototype and comparison code that is useful for understanding how the current service evolved, but is not part of the live FastAPI runtime.

## What belongs here

- Original one-off scripts that proved a behavior before it was moved into `app/`
- Historical reference implementations worth preserving for debugging or comparison
- Small code artifacts that explain why a production behavior exists today

## What does not belong here

- Active FastAPI runtime code
- Utility scripts that operators or developers should run regularly
- Tests

## Current file

- `silence_chunked_whisper.py`
  - Original working Whisper prototype
  - Useful as the behavioral reference for silence-aware chunking and direct OpenVINO Whisper usage

If the behavior of the production transcriber changes, this folder should only be updated when the reference artifact itself is still worth keeping. Do not treat it as a second production code path.