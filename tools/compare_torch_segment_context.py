#!/usr/bin/env python3
"""Measure how fixed Demucs segment length changes the official Torch result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

from validate_e2e_stems_against_torch import (
    EXPECTED_WEIGHT_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    SAMPLE_RATE,
    load_poc_module,
    metrics,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input-pcm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_model(torch, apply_model, model, mix: np.ndarray, segment: float, shifts: int) -> np.ndarray:
    waveform = torch.from_numpy(mix)
    reference = waveform.mean(0)
    mean = reference.mean()
    std = reference.std() + 1e-8
    random.seed(0)
    torch.manual_seed(0)
    with torch.inference_mode():
        separated = apply_model(
            model,
            ((waveform - mean) / std)[None],
            segment=segment,
            shifts=shifts,
            split=True,
            overlap=0.25,
            device="cpu",
        )[0]
        separated = separated * std + mean
    return np.ascontiguousarray(separated.cpu().numpy(), dtype=np.float32)


def run_app_overlap(torch, model, core, poc, mix: np.ndarray, seconds: float) -> np.ndarray:
    window = int(round(seconds * SAMPLE_RATE))
    stride = window * 3 // 4
    frames = mix.shape[1]
    output = np.zeros((len(model.sources), 2, frames), dtype=np.float32)
    accumulated_weight = np.zeros(frames, dtype=np.float32)
    half = window // 2
    weights = np.asarray(
        [(index + 1 if index < half else window - index) / half for index in range(window)],
        dtype=np.float32,
    )
    offset = 0
    while offset < frames:
        actual = min(window, frames - offset)
        pad_left = (window - actual) // 2
        window_mix = np.zeros((1, 2, window), dtype=np.float32)
        window_mix[0, :, pad_left : pad_left + actual] = mix[:, offset : offset + actual]
        window_tensor = torch.from_numpy(window_mix)
        with torch.inference_mode():
            spectrum = poc.spec_to_channels(torch, model._spec(window_tensor))
            frequency, time = core(window_tensor, spectrum)
            separated = (
                model._ispec(poc.channels_to_spec(torch, frequency), length=window) + time
            )[0].cpu().numpy()
        active_weights = weights[pad_left : pad_left + actual]
        output[:, :, offset : offset + actual] += (
            separated[:, :, pad_left : pad_left + actual] * active_weights[None, None]
        )
        accumulated_weight[offset : offset + actual] += active_weights
        offset += min(stride, frames - offset)
    output /= np.maximum(accumulated_weight, 1e-8)[None, None]
    return output


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.poc_root / "python" / "worker"))

    import torch
    from demucs.api import Separator
    from demucs.apply import apply_model
    from model_download import verify_model

    poc = load_poc_module(args.poc_root)
    repository = verify_model(MODEL_NAME, args.model_root)
    weight = repository / f"{MODEL_REVISION}.th"
    weight_hash = sha256(weight)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected official Torch weight hash: {weight_hash}")

    separator = Separator(
        model=MODEL_NAME,
        repo=repository,
        device="cpu",
        shifts=0,
        overlap=0.25,
        split=True,
    )
    model = poc.unwrap_single_model(separator.model).cpu().eval()
    poc.install_deterministic_pos_embedding(model)
    core = poc.HTDemucsCore.build(torch, model).eval()

    interleaved = np.fromfile(args.input_pcm, dtype="<f4")
    if interleaved.size % 2:
        raise RuntimeError("Input PCM is not interleaved stereo float32")
    mix = np.ascontiguousarray(interleaved.reshape(-1, 2).T)

    desktop_no_shift = run_model(torch, apply_model, model, mix, float(model.segment), 0)
    desktop_default = run_model(torch, apply_model, model, mix, float(model.segment), 1)
    candidates = {}
    for seconds in (2.0, 3.0, 4.0, 5.0, 6.0):
        result = run_model(torch, apply_model, model, mix, seconds, 0)
        app_result = run_app_overlap(torch, model, core, poc, mix, seconds)
        candidates[f"{seconds:.1f}"] = {
            "vsDesktopNoShift": metrics(desktop_no_shift, result),
            "vsDesktopDefaultOneShift": metrics(desktop_default, result),
            "appStyleCenteredFinalVsDesktopNoShift": metrics(desktop_no_shift, app_result),
            "appStyleCenteredFinalVsDesktopDefaultOneShift": metrics(desktop_default, app_result),
        }

    report = {
        "model": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "officialTorchWeightSha256": weight_hash,
        "inputPcm": str(args.input_pcm),
        "inputPcmSha256": sha256(args.input_pcm),
        "frames": int(mix.shape[1]),
        "desktopSegmentSeconds": float(model.segment),
        "desktopDefaultOneShiftVsNoShift": metrics(desktop_no_shift, desktop_default),
        "fixedSegmentNoShiftCandidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "desktopSegmentSeconds": float(model.segment),
        "candidates": {
            seconds: {
                "demucsSplitCorrelation": values["vsDesktopNoShift"]["correlation"],
                "demucsSplitSignalToNoiseDb": values["vsDesktopNoShift"]["signalToNoiseDb"],
                "appStyleCorrelation": values["appStyleCenteredFinalVsDesktopNoShift"]["correlation"],
                "appStyleSignalToNoiseDb": values["appStyleCenteredFinalVsDesktopNoShift"]["signalToNoiseDb"],
            }
            for seconds, values in candidates.items()
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
