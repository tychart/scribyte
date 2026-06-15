import numpy as np
import librosa
import time
import openvino_genai as ov_genai

MODEL_PATH = "whisper_base_ov"
SAMPLE_RATE = 16000

# Maximum chunk duration — chunks always break at silence, but never exceed this
MAX_CHUNK_SECONDS = 30

# Silence detection tuning:
# top_db: how many dB below peak counts as silence (lower = more sensitive)
# min_silence_seconds: ignore silence gaps shorter than this (avoids splitting mid-word)
TOP_DB = 40
MIN_SILENCE_SECONDS = 0.3

audio, _ = librosa.load("sample2.wav", sr=SAMPLE_RATE)


def silence_aware_chunks(audio, max_chunk_seconds=MAX_CHUNK_SECONDS):
    """
    Split audio into chunks at silence boundaries, never exceeding max_chunk_seconds.
    Uses librosa to detect non-silent intervals, then groups them greedily.
    No overlap needed since every cut is at a natural pause.
    """
    max_chunk_samples = max_chunk_seconds * SAMPLE_RATE
    min_silence_samples = int(MIN_SILENCE_SECONDS * SAMPLE_RATE)

    # Get intervals of non-silent audio: shape (N, 2) of [start, end] sample indices
    intervals = librosa.effects.split(audio, top_db=TOP_DB, frame_length=512, hop_length=128)

    if len(intervals) == 0:
        yield audio
        return

    # Merge intervals separated by very short silences (less than min_silence_samples)
    # This avoids splitting in the middle of a word
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        gap = start - merged[-1][1]
        if gap < min_silence_samples:
            merged[-1][1] = end  # extend current interval
        else:
            merged.append([start, end])

    # Greedily group merged intervals into chunks up to max_chunk_samples.
    # Each chunk spans from the start of its first interval to the end of its last.
    chunk_start = merged[0][0]
    chunk_end = merged[0][1]

    for i in range(1, len(merged)):
        interval_start, interval_end = merged[i]

        # Would adding this interval exceed the max chunk size?
        if interval_end - chunk_start > max_chunk_samples:
            yield audio[chunk_start:chunk_end]
            chunk_start = interval_start
            chunk_end = interval_end
        else:
            chunk_end = interval_end  # extend chunk to include this interval

    # Yield the final chunk
    yield audio[chunk_start:chunk_end]


def run(device):
    print(f"\n--- Running on {device} ---")
    pipe = ov_genai.WhisperPipeline(MODEL_PATH, device)

    start = time.time()
    full_text = []

    chunk_count = 0
    for chunk in silence_aware_chunks(audio):
        duration = len(chunk) / SAMPLE_RATE
        if duration < 0.5:  # skip very short fragments
            continue

        chunk_count += 1
        result = pipe.generate(chunk, language="<|en|>")
        text = result.texts[0].strip()
        full_text.append(text)
        print(f"  Chunk {chunk_count} ({duration:.1f}s): {text[:60]}...")

    end = time.time()

    print("\nFull Text:", " ".join(full_text))
    print(f"Processed {chunk_count} chunks")
    print(f"Latency: {end - start:.2f}s")


# run("CPU")
run("NPU")
