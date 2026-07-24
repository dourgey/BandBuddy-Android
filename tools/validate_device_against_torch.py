#!/usr/bin/env python3
"""Compare exported Android QNN tensors directly with the official Torch model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


MODEL_NAME = "htdemucs_6s"
MODEL_REVISION = "5c90dfd2-34c22ccb"
EXPECTED_WEIGHT_SHA256 = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
DEFAULT_SAMPLES = 88_200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="Exact waveform window length exported by Android",
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="NAME=EXPORT_DIRECTORY",
    )
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
        raise RuntimeError("Device output contains NaN or infinity")
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
    }


def read_exact(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = math.prod(shape)
    value = np.fromfile(path, dtype="<f4")
    if value.size != expected:
        raise RuntimeError(f"Unexpected float count in {path}: {value.size}, expected {expected}")
    return np.ascontiguousarray(value.reshape(shape))


def parse_cases(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected NAME=EXPORT_DIRECTORY, got {value!r}")
        result.append((name, Path(path)))
    return result


def main() -> int:
    args = parse_args()
    samples = args.samples
    if samples <= 0:
        raise ValueError("--samples must be positive")
    spectrum_frames = (samples + 1_023) // 1_024
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
    reports: list[dict[str, Any]] = []
    accepted = True

    for name, directory in parse_cases(args.case):
        mix = read_exact(directory / "mix.f32le", (1, 2, samples))
        device_spec = read_exact(directory / "spec.f32le", (1, 4, 2_048, spectrum_frames))
        device_frequency = read_exact(
            directory / "frequency.f32le", (1, len(source_names), 4, 2_048, spectrum_frames)
        )
        device_time = read_exact(directory / "time.f32le", (1, len(source_names), 2, samples))
        device_final = read_exact(directory / "final.f32le", (1, len(source_names), 2, samples))

        mix_torch = torch.from_numpy(mix)
        spec_torch = torch.from_numpy(device_spec)
        with torch.inference_mode():
            original_use_train_segment = model.use_train_segment
            model.use_train_segment = False
            original_torch_final = model(mix_torch).cpu().numpy()
            model.use_train_segment = original_use_train_segment

            torch_spec = poc.spec_to_channels(torch, model._spec(mix_torch)).cpu().numpy()
            torch_frequency, torch_time = core(mix_torch, spec_torch)
            torch_boundary_final = (
                model._ispec(poc.channels_to_spec(torch, torch_frequency), length=samples)
                + torch_time
            ).cpu().numpy()
            torch_frequency_np = torch_frequency.cpu().numpy()
            torch_time_np = torch_time.cpu().numpy()

        phone_vs_original = metrics(original_torch_final, device_final)
        case_passes = (
            phone_vs_original["correlation"] >= .99999
            and phone_vs_original["signalToNoiseDb"] >= 50.0
        )
        accepted &= case_passes
        reports.append({
            "name": name,
            "directory": str(directory),
            "nativeSpectrumVsTorch": metrics(torch_spec, device_spec),
            "deviceCoreVsTorchSameBoundary": {
                "frequency": metrics(torch_frequency_np, device_frequency),
                "time": metrics(torch_time_np, device_time),
            },
            "torchDeviceBoundaryVsOriginalTorch": metrics(original_torch_final, torch_boundary_final),
            "phoneFinalVsOriginalTorch": {
                "allStems": phone_vs_original,
                "perStem": {
                    source: metrics(original_torch_final[:, index], device_final[:, index])
                    for index, source in enumerate(source_names)
                },
            },
            "accepted": case_passes,
        })

    report = {
        "model": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "officialTorchWeightSha256": weight_hash,
        "sourceOrder": list(source_names),
        "contract": {
            "samples": samples,
            "sampleRate": model.samplerate,
            "spectrumFrames": spectrum_frames,
        },
        "cases": reports,
        "qualityGate": {
            "minimumPhoneFinalCorrelation": .99999,
            "minimumPhoneFinalSignalToNoiseDb": 50.0,
            "accepted": bool(accepted),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "accepted": bool(accepted),
        "cases": [
            {
                "name": item["name"],
                "correlation": item["phoneFinalVsOriginalTorch"]["allStems"]["correlation"],
                "signalToNoiseDb": item["phoneFinalVsOriginalTorch"]["allStems"]["signalToNoiseDb"],
            }
            for item in reports
        ],
    }, ensure_ascii=False, indent=2))
    return 0 if accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
