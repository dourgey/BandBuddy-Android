#!/usr/bin/env python3
"""Stream interleaved stereo float32 PCM into a standard PCM16 WAV."""

from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    args = parser.parse_args()

    if args.input.stat().st_size % (2 * np.dtype("<f4").itemsize):
        raise RuntimeError("Input must be interleaved stereo float32 PCM")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values = np.memmap(args.input, mode="r", dtype="<f4")
    frames = values.size // 2
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(args.sample_rate)
        for start in range(0, values.size, 2 * 44_100):
            block = np.asarray(values[start : start + 2 * 44_100], dtype=np.float32)
            pcm16 = np.rint(np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2")
            output.writeframesraw(pcm16.tobytes())
    print(f"{args.output}: {frames} frames, {frames / args.sample_rate:.3f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
