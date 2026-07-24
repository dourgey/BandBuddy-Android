#!/usr/bin/env python3
"""Validate Android's six M4A stems against mobile-window and standard Torch Demucs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import av
import numpy as np


MODEL_NAME = "htdemucs_6s"
MODEL_REVISION = "5c90dfd2-34c22ccb"
EXPECTED_WEIGHT_SHA256 = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
WINDOW = 343_980
STRIDE = WINDOW * 3 // 4
SAMPLE_RATE = 44_100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--input-pcm", type=Path, required=True)
    parser.add_argument("--device-stems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_poc_module(poc_root: Path) -> Any:
    source = poc_root / "poc" / "android-demucs" / "scripts" / "export_core_model.py"
    spec = importlib.util.spec_from_file_location("bandbuddy_poc_export", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load POC core wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64, copy=False).reshape(-1)
    candidate64 = candidate.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(candidate64).all():
        raise RuntimeError("Decoded device stem contains NaN or infinity")
    delta = candidate64 - reference64
    signal_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    error_rms = float(np.sqrt(np.mean(delta * delta)))
    reference_centered = reference64 - reference64.mean()
    candidate_centered = candidate64 - candidate64.mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    correlation = float(np.dot(reference_centered, candidate_centered) / denominator) if denominator else 1.0
    return {
        "maxAbsoluteError": float(np.max(np.abs(delta))),
        "meanAbsoluteError": float(np.mean(np.abs(delta))),
        "rootMeanSquareError": error_rms,
        "normalizedRootMeanSquareError": error_rms / max(signal_rms, 1e-12),
        "signalToNoiseDb": 20.0 * math.log10(max(signal_rms, 1e-12) / max(error_rms, 1e-12)),
        "correlation": correlation,
        "referenceRms": signal_rms,
        "candidateRms": float(np.sqrt(np.mean(candidate64 * candidate64))),
        "candidatePeak": float(np.max(np.abs(candidate64))),
        "clippedFraction": float(np.mean(np.abs(candidate64) >= .999)),
    }


def decode_m4a(path: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo", rate=SAMPLE_RATE
        )
        for packet in container.demux(audio=0):
            for frame in packet.decode():
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray())
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray())
    if not chunks:
        raise RuntimeError(f"No audio decoded from {path}")
    value = np.ascontiguousarray(np.concatenate(chunks, axis=1), dtype=np.float32)
    if value.shape[0] != 2:
        raise RuntimeError(f"Expected stereo output from {path}, got {value.shape}")
    return value


def torch_overlap_add(torch: Any, model: Any, core: Any, poc: Any, mix: np.ndarray) -> np.ndarray:
    frames = mix.shape[1]
    sources = len(model.sources)
    output = np.zeros((sources, 2, frames), dtype=np.float32)
    accumulated_weight = np.zeros(frames, dtype=np.float32)
    half = WINDOW // 2
    weights = np.asarray(
        [(index + 1 if index < half else WINDOW - index) / half for index in range(WINDOW)],
        dtype=np.float32,
    )
    offset = 0
    while offset < frames:
        actual = min(WINDOW, frames - offset)
        pad_left = (WINDOW - actual) // 2
        window = np.zeros((1, 2, WINDOW), dtype=np.float32)
        window[0, :, pad_left : pad_left + actual] = mix[:, offset : offset + actual]
        window_torch = torch.from_numpy(window)
        with torch.inference_mode():
            spectrum = poc.spec_to_channels(torch, model._spec(window_torch))
            frequency, time = core(window_torch, spectrum)
            separated = (
                model._ispec(poc.channels_to_spec(torch, frequency), length=WINDOW) + time
            )[0].cpu().numpy()
        active_weights = weights[pad_left : pad_left + actual]
        output[:, :, offset : offset + actual] += separated[
            :, :, pad_left : pad_left + actual
        ] * active_weights[None, None]
        accumulated_weight[offset : offset + actual] += active_weights
        offset += min(STRIDE, frames - offset)
    output /= np.maximum(accumulated_weight, 1e-8)[None, None]
    return output


def torch_standard_demucs(torch: Any, separator: Any, source_names: tuple[str, ...], mix: np.ndarray) -> np.ndarray:
    """Run the unmodified Demucs API path used by a normal desktop separation."""
    torch.manual_seed(0)
    random.seed(0)
    with torch.inference_mode():
        _normalized_mix, separated = separator.separate_tensor(
            torch.from_numpy(mix), sr=SAMPLE_RATE
        )
    return np.ascontiguousarray(
        np.stack([separated[name].cpu().numpy() for name in source_names]),
        dtype=np.float32,
    )


def common_aac_lead(reference: np.ndarray, decoded: np.ndarray) -> int:
    reference_signature = reference.sum(axis=(0, 1)).astype(np.float64)
    decoded_signature = decoded.sum(axis=(0, 1)).astype(np.float64)
    maximum = decoded_signature.size - reference_signature.size
    if maximum < 0:
        raise RuntimeError(
            f"Decoded AAC is shorter than expected: {decoded_signature.size} < {reference_signature.size}"
        )
    convolution_size = decoded_signature.size + reference_signature.size - 1
    fft_size = 1 << (convolution_size - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(decoded_signature, fft_size)
        * np.fft.rfft(reference_signature[::-1], fft_size),
        fft_size,
    )
    candidates = correlation[reference_signature.size - 1 : reference_signature.size + maximum]
    return int(np.argmax(candidates))


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.poc_root / "python" / "worker"))

    import torch
    from demucs.api import Separator
    from model_download import verify_model

    poc = load_poc_module(args.poc_root)
    repository = verify_model(MODEL_NAME, args.model_root)
    weight = repository / f"{MODEL_REVISION}.th"
    weight_hash = sha256(weight)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected official Torch weight hash: {weight_hash}")
    separator = Separator(model=MODEL_NAME, repo=repository, device="cpu", shifts=1, overlap=.25, split=True)
    model = poc.unwrap_single_model(separator.model).cpu().eval()
    poc.install_deterministic_pos_embedding(model)
    core = poc.HTDemucsCore.build(torch, model).eval()
    source_names = tuple(model.sources)

    interleaved = np.fromfile(args.input_pcm, dtype="<f4")
    if interleaved.size % 2:
        raise RuntimeError("Input PCM is not interleaved stereo float32")
    mix = np.ascontiguousarray(interleaved.reshape(-1, 2).T)
    torch_stems = torch_overlap_add(torch, model, core, poc, mix)
    reference_channel_mean = mix.mean(axis=0)
    global_mean = float(reference_channel_mean.mean())
    global_std = float(reference_channel_mean.std(ddof=1) + 1e-8)
    normalized_mix = np.ascontiguousarray((mix - global_mean) / global_std, dtype=np.float32)
    globally_normalized_torch_stems = (
        torch_overlap_add(torch, model, core, poc, normalized_mix) * global_std + global_mean
    )
    standard_torch_stems = torch_standard_demucs(
        torch, separator, source_names, mix
    )

    decoded_items = [decode_m4a(args.device_stems / f"{name}.m4a") for name in source_names]
    decoded_lengths = {item.shape[1] for item in decoded_items}
    if len(decoded_lengths) != 1:
        raise RuntimeError(f"Six AAC stems have different lengths: {sorted(decoded_lengths)}")
    decoded = np.stack(decoded_items)
    lead = common_aac_lead(torch_stems, decoded)
    aligned = decoded[:, :, lead : lead + mix.shape[1]]
    if aligned.shape[-1] != mix.shape[1]:
        raise RuntimeError(f"AAC alignment produced a short result: {aligned.shape}")

    all_metrics = metrics(torch_stems, aligned)
    per_stem = {
        name: metrics(torch_stems[index], aligned[index])
        for index, name in enumerate(source_names)
    }
    reconstruction = metrics(mix, aligned.sum(axis=0))
    standard_all_metrics = metrics(standard_torch_stems, aligned)
    standard_per_stem = {
        name: metrics(standard_torch_stems[index], aligned[index])
        for index, name in enumerate(source_names)
    }
    mobile_window_vs_standard = metrics(standard_torch_stems, torch_stems)
    globally_normalized_mobile_window_vs_standard = metrics(
        standard_torch_stems, globally_normalized_torch_stems
    )
    accepted = (
        all_metrics["correlation"] >= .99
        and all_metrics["signalToNoiseDb"] >= 20.0
        and all(value["clippedFraction"] < 1e-4 for value in per_stem.values())
    )
    report = {
        "model": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "officialTorchWeightSha256": weight_hash,
        "inputPcm": str(args.input_pcm),
        "inputPcmSha256": sha256(args.input_pcm),
        "frames": int(mix.shape[1]),
        "window": WINDOW,
        "stride": STRIDE,
        "windowCount": int(math.ceil(mix.shape[1] / STRIDE)),
        "sourceOrder": list(source_names),
        "aac": {
            "decodedFrames": int(decoded.shape[-1]),
            "encoderLeadFrames": lead,
            "trailingPaddingFrames": int(decoded.shape[-1] - lead - mix.shape[1]),
        },
        "decodedPhoneM4aVsOfficialTorchOverlapAdd": {
            "allStems": all_metrics,
            "perStem": per_stem,
        },
        "decodedPhoneM4aVsStandardDesktopTorchDemucs": {
            "settings": {
                "segmentSeconds": float(model.segment),
                "split": True,
                "overlap": 0.25,
                "shifts": 1,
                "randomSeed": 0,
            },
            "allStems": standard_all_metrics,
            "perStem": standard_per_stem,
        },
        "officialTorchMobileWindowVsStandardDesktop": mobile_window_vs_standard,
        "officialTorchGloballyNormalizedMobileWindowVsStandardDesktop": {
            "globalMean": global_mean,
            "globalStandardDeviation": global_std,
            **globally_normalized_mobile_window_vs_standard,
        },
        "decodedStemSumVsInputMix": reconstruction,
        "qualityGate": {
            "minimumAllStemCorrelation": .99,
            "minimumAllStemSignalToNoiseDb": 20.0,
            "maximumClippedFractionPerStem": 1e-4,
            "accepted": bool(accepted),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "accepted": bool(accepted),
        "aacLeadFrames": lead,
        "correlation": all_metrics["correlation"],
        "signalToNoiseDb": all_metrics["signalToNoiseDb"],
        "standardDesktopCorrelation": standard_all_metrics["correlation"],
        "standardDesktopSignalToNoiseDb": standard_all_metrics["signalToNoiseDb"],
    }, ensure_ascii=False, indent=2))
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
