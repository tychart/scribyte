from typing import Sequence, SupportsFloat


class WhisperDecodedResults:
    texts: list[str]


class WhisperPipeline:
    def __init__(self, model_path: str, device: str) -> None: ...

    def generate(
        self,
        raw_speech_input: Sequence[SupportsFloat],
        generation_config: object | None = ...,
        streamer: object | None = ...,
        **kwargs: object,
    ) -> WhisperDecodedResults: ...