#!/usr/bin/env python3
"""Export a shorter-context FP32 HTDemucs core and validate it against Torch.

The original checkpoint is fully convolutional and accepts shorter waveforms.
Reducing the fixed deployment window lowers the largest HTP activation below
the device's VTCM limit without changing weights or numerical precision. This
script distinguishes conversion fidelity (LiteRT vs the original Torch model
at the same length) from the audible context trade-off (short Torch window vs
the center of the original 7.8-second Torch window).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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
    parser.add_argument("--samples", type=int, default=88_200)
    parser.add_argument("--validation-pcm", type=Path, action="append", default=[])
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
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
    }


def pcm_window(path: Path, offset: int, frames: int) -> np.ndarray:
    if path.stat().st_size % 8:
        raise RuntimeError(f"PCM byte count is not stereo float32: {path}")
    values = np.memmap(path, mode="r", dtype="<f4").reshape(-1, 2)
    if offset < 0 or offset + frames > len(values):
        raise RuntimeError(f"PCM window {offset}:{offset + frames} exceeds {len(values)}")
    return np.ascontiguousarray(values[offset : offset + frames].T[None], dtype=np.float32)


def reconstruct(torch: Any, model: Any, poc: Any, frequency: np.ndarray, waveform: np.ndarray, length: int) -> np.ndarray:
    with torch.inference_mode():
        return (
            model._ispec(
                poc.channels_to_spec(torch, torch.from_numpy(frequency)),
                length=length,
            )
            + torch.from_numpy(waveform)
        ).cpu().numpy()


def main() -> int:
    args = parse_args()
    if args.samples < 44_100:
        raise RuntimeError("Deployment window below one second is not supported")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing:
        if not args.output.is_file():
            raise RuntimeError("--reuse-existing requires an existing output model")
    else:
        args.output.unlink(missing_ok=True)

    sys.path.insert(0, str(args.poc_root / "python" / "worker"))
    import ai_edge_litert.interpreter as litert
    import litert_torch
    import torch
    from demucs.api import Separator
    from model_download import verify_model

    poc = load_poc_module(args.poc_root)
    repository = verify_model(MODEL_NAME, args.model_root)
    weight_path = repository / f"{MODEL_REVISION}.th"
    weight_hash = sha256(weight_path)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected PyTorch checkpoint: {weight_hash}")

    separator = Separator(model=MODEL_NAME, repo=repository, device="cpu", shifts=1, overlap=.25, split=True)
    model = poc.unwrap_single_model(separator.model).cpu().eval()
    poc.install_deterministic_pos_embedding(model)
    if tuple(model.sources) != SOURCE_NAMES:
        raise RuntimeError(f"Unexpected source order: {model.sources}")
    original_samples = int(round(float(model.segment) * model.samplerate))

    mix = poc.deterministic_mix(torch, args.samples, model.samplerate)
    core = poc.HTDemucsCore.build(torch, model).eval()
    with torch.inference_mode():
        torch_started = time.perf_counter()
        original_use_train_segment = model.use_train_segment
        model.use_train_segment = False
        torch_final = model(mix)
        model.use_train_segment = original_use_train_segment
        torch_default_padded = model(mix)
        torch_seconds = time.perf_counter() - torch_started
        spec = poc.spec_to_channels(torch, model._spec(mix))
        torch_frequency, torch_waveform = core(mix, spec)
        torch_reconstructed = (
            model._ispec(poc.channels_to_spec(torch, torch_frequency), length=args.samples)
            + torch_waveform
        )

    conversion_started = time.perf_counter()
    if not args.reuse_existing:
        lite_model = litert_torch.convert(
            core,
            (mix, spec),
            strict_export=True,
            lightweight_conversion=True,
            enable_x64=False,
        )
        lite_model.export(str(args.output))
        conversion_seconds = time.perf_counter() - conversion_started

        def run_lite(mix_value: np.ndarray, spec_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            outputs = lite_model(mix_value, spec_value)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise RuntimeError(f"Unexpected LiteRT outputs: {type(outputs)}")
            return tuple(np.asarray(value) for value in outputs)  # type: ignore[return-value]
    else:
        conversion_seconds = 0.0
        interpreter = litert.Interpreter(model_path=str(args.output), num_threads=8)
        interpreter.allocate_tensors()
        runner = interpreter.get_signature_runner("serving_default")

        def run_lite(mix_value: np.ndarray, spec_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            outputs = [np.asarray(value) for value in runner(args_0=mix_value, args_1=spec_value).values()]
            frequency = next((value for value in outputs if value.ndim == 5), None)
            waveform = next((value for value in outputs if value.ndim == 4), None)
            if frequency is None or waveform is None:
                raise RuntimeError(f"Unexpected LiteRT output shapes: {[value.shape for value in outputs]}")
            return frequency, waveform

    lite_started = time.perf_counter()
    lite_frequency, lite_waveform = run_lite(mix.numpy(), spec.numpy())
    lite_seconds = time.perf_counter() - lite_started
    lite_final = reconstruct(
        torch, model, poc, lite_frequency, lite_waveform, args.samples
    )

    real_reports: list[dict[str, Any]] = []
    crop_start = (original_samples - args.samples) // 2
    for path in args.validation_pcm:
        total_frames = path.stat().st_size // 8
        if total_frames < original_samples:
            continue
        long_offset = (total_frames - original_samples) // 2
        long_mix_np = pcm_window(path, long_offset, original_samples)
        short_mix_np = np.ascontiguousarray(
            long_mix_np[:, :, crop_start : crop_start + args.samples]
        )
        with torch.inference_mode():
            long_final = model(torch.from_numpy(long_mix_np)).cpu().numpy()
            default_padded_short = model(torch.from_numpy(short_mix_np)).cpu().numpy()
            original_use_train_segment = model.use_train_segment
            model.use_train_segment = False
            short_final = model(torch.from_numpy(short_mix_np)).cpu().numpy()
            model.use_train_segment = original_use_train_segment
            short_spec = poc.spec_to_channels(
                torch, model._spec(torch.from_numpy(short_mix_np))
            ).cpu().numpy()
        short_lite_frequency, short_lite_waveform = run_lite(short_mix_np, short_spec)
        short_lite_final = reconstruct(
            torch,
            model,
            poc,
            np.asarray(short_lite_frequency),
            np.asarray(short_lite_waveform),
            args.samples,
        )
        centered_long = long_final[
            :, :, :, crop_start : crop_start + args.samples
        ]
        real_reports.append({
            "pcm": str(path),
            "pcmSha256": sha256(path),
            "longWindowOffsetFrames": long_offset,
            "liteRtVsPyTorchSameShortWindow": {
                "allStems": metrics(short_final, short_lite_final),
                "perStem": {
                    name: metrics(short_final[:, index], short_lite_final[:, index])
                    for index, name in enumerate(SOURCE_NAMES)
                },
            },
            "shortTorchVsCenteredOriginalContextTorch": {
                "allStems": metrics(centered_long, short_final),
                "perStem": {
                    name: metrics(centered_long[:, index], short_final[:, index])
                    for index, name in enumerate(SOURCE_NAMES)
                },
            },
            "shortUnpaddedTorchVsOriginalDefaultPaddedTorch": {
                "allStems": metrics(default_padded_short, short_final),
                "perStem": {
                    name: metrics(default_padded_short[:, index], short_final[:, index])
                    for index, name in enumerate(SOURCE_NAMES)
                },
            },
        })

    deterministic_metrics = metrics(torch_final.numpy(), lite_final)
    conversion_passes = deterministic_metrics["correlation"] >= .99999 and deterministic_metrics["maxAbsoluteError"] <= .001
    for item in real_reports:
        value = item["liteRtVsPyTorchSameShortWindow"]["allStems"]
        conversion_passes &= value["correlation"] >= .99999 and value["maxAbsoluteError"] <= .001

    report = {
        "model": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "weightSha256": weight_hash,
        "sourceOrder": list(SOURCE_NAMES),
        "contract": {
            "sampleRate": model.samplerate,
            "samples": args.samples,
            "seconds": args.samples / model.samplerate,
            "spectrumFrames": int(spec.shape[-1]),
            "mixShape": list(mix.shape),
            "specShape": list(spec.shape),
            "frequencyShape": list(torch_frequency.shape),
            "waveformShape": list(torch_waveform.shape),
        },
        "seconds": {
            "torchFull": torch_seconds,
            "liteRtConversion": conversion_seconds,
            "liteRtCore": lite_seconds,
        },
        "torchCoreReconstruction": metrics(torch_final.numpy(), torch_reconstructed.numpy()),
        "shortUnpaddedTorchVsOriginalDefaultPaddedTorch": metrics(
            torch_default_padded.numpy(), torch_final.numpy()
        ),
        "deterministicLiteRtVsPyTorch": {
            "allStems": deterministic_metrics,
            "perStem": {
                name: metrics(torch_final.numpy()[:, index], lite_final[:, index])
                for index, name in enumerate(SOURCE_NAMES)
            },
        },
        "realAudioValidation": real_reports,
        "qualityGate": {
            "scope": "conversion fidelity at the same short context",
            "minimumCorrelation": .99999,
            "maximumAbsoluteError": .001,
            "acceptedForDeploymentTesting": bool(conversion_passes),
        },
        "artifacts": {
            "liteRt": str(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256(args.output),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.report),
        "model": str(args.output),
        "acceptedForDeploymentTesting": bool(conversion_passes),
    }, ensure_ascii=False, indent=2))
    return 0 if conversion_passes else 3


if __name__ == "__main__":
    raise SystemExit(main())
