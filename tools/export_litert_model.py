#!/usr/bin/env python3
"""Export the pinned HTDemucs six-stem core to LiteRT and validate three runtimes.

Run this script in the Linux conversion environment.  It deliberately imports
the core wrapper from the Android POC so the PyTorch reference and the deployed
model cannot silently drift apart.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


MODEL_NAME = "htdemucs_6s"
MODEL_REVISION = "5c90dfd2-34c22ccb"
EXPECTED_WEIGHT_SHA256 = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
SOURCE_NAMES = ("drums", "bass", "other", "vocals", "guitar", "piano")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_poc_module(poc_root: Path) -> Any:
    source = poc_root / "poc" / "android-demucs" / "scripts" / "export_core_model.py"
    spec = importlib.util.spec_from_file_location("bandbuddy_poc_export", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load POC core wrapper: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64, copy=False).reshape(-1)
    candidate64 = candidate.astype(np.float64, copy=False).reshape(-1)
    delta = candidate64 - reference64
    reference_centered = reference64 - reference64.mean()
    candidate_centered = candidate64 - candidate64.mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    correlation = float(np.dot(reference_centered, candidate_centered) / denominator) if denominator else 1.0
    return {
        "maxAbsoluteError": float(np.max(np.abs(delta))),
        "meanAbsoluteError": float(np.mean(np.abs(delta))),
        "rootMeanSquareError": float(np.sqrt(np.mean(delta * delta))),
        "correlation": correlation,
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    worker_root = args.poc_root / "python" / "worker"
    sys.path.insert(0, str(worker_root))

    import onnxruntime as ort
    import torch
    import litert_torch
    from demucs.api import Separator
    from model_download import verify_model

    poc = load_poc_module(args.poc_root)
    repository = verify_model(MODEL_NAME, args.model_root)
    weight_path = repository / f"{MODEL_REVISION}.th"
    weight_sha256 = sha256(weight_path)
    if weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected PyTorch checkpoint: {weight_sha256}")

    separator = Separator(
        model=MODEL_NAME,
        repo=repository,
        device="cpu",
        shifts=1,
        overlap=0.25,
        split=True,
    )
    loaded_model = separator.model.cpu().eval()
    model = poc.unwrap_single_model(loaded_model).cpu().eval()
    poc.install_deterministic_pos_embedding(model)
    frames = int(round(float(model.segment) * model.samplerate))
    mix = poc.deterministic_mix(torch, frames, model.samplerate)

    with torch.inference_mode():
        reference_started = time.perf_counter()
        torch_final = model(mix)
        torch_reference_seconds = time.perf_counter() - reference_started
        spectrum = model._spec(mix)
        spec_channels = poc.spec_to_channels(torch, spectrum)
        core = poc.HTDemucsCore.build(torch, model).eval()
        core_started = time.perf_counter()
        torch_frequency, torch_time = core(mix, spec_channels)
        torch_core_seconds = time.perf_counter() - core_started
        torch_reconstructed = (
            model._ispec(poc.channels_to_spec(torch, torch_frequency), length=frames)
            + torch_time
        )

    mix_np = mix.detach().cpu().numpy()
    spec_np = spec_channels.detach().cpu().numpy()
    torch_frequency_np = torch_frequency.detach().cpu().numpy()
    torch_time_np = torch_time.detach().cpu().numpy()
    torch_final_np = torch_final.detach().cpu().numpy()

    onnx_started = time.perf_counter()
    onnx_session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_frequency, onnx_time = onnx_session.run(
        ["frequency_channels", "time_waveform"],
        {"mix": mix_np, "spec_channels": spec_np},
    )
    onnx_seconds = time.perf_counter() - onnx_started
    with torch.inference_mode():
        onnx_final = (
            model._ispec(
                poc.channels_to_spec(torch, torch.from_numpy(onnx_frequency)),
                length=frames,
            )
            + torch.from_numpy(onnx_time)
        ).numpy()

    conversion_started = time.perf_counter()
    lite_model = litert_torch.convert(
        core,
        (mix, spec_channels),
        strict_export=True,
        lightweight_conversion=True,
        enable_x64=False,
    )
    conversion_seconds = time.perf_counter() - conversion_started
    lite_model.export(str(args.output))

    lite_started = time.perf_counter()
    lite_outputs = lite_model(mix_np, spec_np)
    lite_seconds = time.perf_counter() - lite_started
    if not isinstance(lite_outputs, tuple) or len(lite_outputs) != 2:
        raise RuntimeError(f"Unexpected LiteRT outputs: {type(lite_outputs)}")
    lite_frequency, lite_time = (np.asarray(value) for value in lite_outputs)
    with torch.inference_mode():
        lite_final = (
            model._ispec(
                poc.channels_to_spec(torch, torch.from_numpy(lite_frequency)),
                length=frames,
            )
            + torch.from_numpy(lite_time)
        ).numpy()

    report = {
        "model": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "weightSha256": weight_sha256,
        "versions": {
            "torch": torch.__version__,
            "onnxRuntime": ort.__version__,
            "liteRtTorch": getattr(litert_torch, "__version__", "unknown"),
        },
        "shapes": {
            "mix": list(mix_np.shape),
            "specChannels": list(spec_np.shape),
            "frequencyChannels": list(torch_frequency_np.shape),
            "timeWaveform": list(torch_time_np.shape),
            "finalStems": list(torch_final_np.shape),
        },
        "seconds": {
            "torchFull": torch_reference_seconds,
            "torchCore": torch_core_seconds,
            "onnxCore": onnx_seconds,
            "liteRtConversion": conversion_seconds,
            "liteRtCore": lite_seconds,
        },
        "pytorchCoreReconstruction": metrics(torch_final_np, torch_reconstructed.numpy()),
        "onnxVsPyTorch": {
            "frequencyChannels": metrics(torch_frequency_np, onnx_frequency),
            "timeWaveform": metrics(torch_time_np, onnx_time),
            "finalStems": metrics(torch_final_np, onnx_final),
        },
        "liteRtVsPyTorch": {
            "frequencyChannels": metrics(torch_frequency_np, lite_frequency),
            "timeWaveform": metrics(torch_time_np, lite_time),
            "finalStems": metrics(torch_final_np, lite_final),
            "perStemFinal": {
                name: metrics(torch_final_np[:, index], lite_final[:, index])
                for index, name in enumerate(SOURCE_NAMES)
            },
        },
        "liteRtVsOnnx": {
            "frequencyChannels": metrics(onnx_frequency, lite_frequency),
            "timeWaveform": metrics(onnx_time, lite_time),
            "finalStems": metrics(onnx_final, lite_final),
        },
        "artifacts": {
            "liteRt": str(args.output),
            "liteRtBytes": args.output.stat().st_size,
            "liteRtSha256": sha256(args.output),
            "onnx": str(args.onnx),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
