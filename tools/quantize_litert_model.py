#!/usr/bin/env python3
"""Calibrate an INT8 LiteRT HTDemucs core and validate it against PyTorch.

The Android model is only replaced when this script's quality gate passes. The
reference is the pinned original ``.th`` checkpoint, not ONNX or the FP32
LiteRT conversion. Calibration and validation windows are disjoint portions of
real stereo float32 PCM files decoded by the Android application. Quantization
observers run in the PyTorch graph so calibration never retains the model's
roughly 20 GiB of intermediate LiteRT tensors at once.
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
from typing import Any, Iterable

import numpy as np


MODEL_NAME = "htdemucs_6s"
MODEL_REVISION = "5c90dfd2-34c22ccb"
EXPECTED_WEIGHT_SHA256 = "34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd"
EXPECTED_FP32_SHA256 = "ad8223d5769dbf85b12b6b9496787b49815153aa7ae597ea90b26a05ff6cc0ce"
SOURCE_NAMES = ("drums", "bass", "other", "vocals", "guitar", "piano")
SIGNATURE = "serving_default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument(
        "--calibration-pcm",
        type=Path,
        action="append",
        required=True,
        help="Real interleaved stereo float32 PCM; repeat for multiple songs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration-windows", type=int, default=6)
    parser.add_argument("--validation-windows", type=int, default=2)
    parser.add_argument("--calibration-threads", type=int, default=8)
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


class StereoPcm:
    """Memory-mapped interleaved little-endian float32 stereo PCM."""

    def __init__(self, path: Path) -> None:
        if path.stat().st_size % 8:
            raise RuntimeError(f"PCM byte count is not stereo float32: {path.stat().st_size}")
        self.path = path
        self.values = np.memmap(path, mode="r", dtype="<f4").reshape(-1, 2)
        self.frames = int(self.values.shape[0])

    def window(self, offset: int, frames: int) -> np.ndarray:
        if not 0 <= offset <= self.frames - frames:
            raise ValueError(f"Window {offset}:{offset + frames} exceeds {self.frames} frames")
        # LiteRT and PyTorch both consume planar [1, 2, samples]. A real copy is
        # intentional so the native calibration code receives contiguous data.
        return np.ascontiguousarray(self.values[offset : offset + frames].T[None], dtype=np.float32)


def choose_calibration_offsets(total_frames: int, frames: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("Window count must be positive")
    maximum = total_frames - frames
    if maximum < 0:
        raise RuntimeError(f"Calibration PCM has {total_frames} frames; model requires {frames}")
    offsets = [int(round(maximum * (index + 1) / (count + 1))) for index in range(count)]
    if len(set(offsets)) != count:
        raise RuntimeError("PCM is too short to provide distinct representative windows")
    return offsets


def choose_validation_offsets(
    total_frames: int,
    frames: int,
    count: int,
    calibration_offsets: list[int],
) -> list[int]:
    if count < 1:
        raise ValueError("Window count must be positive")
    maximum = total_frames - frames
    # Search a dense, deterministic grid. Each selected offset maximizes its
    # distance from calibration and prior holdouts, while a full model window
    # of separation prevents even one overlapping audio sample.
    candidates = [int(round(maximum * index / 1000)) for index in range(1001)]
    valid = [
        candidate
        for candidate in dict.fromkeys(candidates)
        if all(abs(candidate - forbidden) >= frames for forbidden in calibration_offsets)
    ]
    selected: list[int] = []
    while len(selected) < count:
        remaining = [
            candidate
            for candidate in valid
            if all(abs(candidate - existing) >= frames for existing in selected)
        ]
        if not remaining:
            raise RuntimeError("PCM is too short to provide non-overlapping holdout windows")
        anchors = calibration_offsets + selected
        best = max(remaining, key=lambda candidate: (min(abs(candidate - anchor) for anchor in anchors), -candidate))
        selected.append(best)
    return sorted(selected)


def spectrum_input(torch: Any, model: Any, poc: Any, mix: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(mix)
    with torch.inference_mode():
        spectrum = model._spec(tensor)
        channels = poc.spec_to_channels(torch, spectrum)
    return np.ascontiguousarray(channels.cpu().numpy(), dtype=np.float32)


def calibration_samples(
    torch: Any,
    model: Any,
    poc: Any,
    pcm: StereoPcm,
    offsets: Iterable[int],
    frames: int,
) -> list[dict[str, np.ndarray]]:
    samples: list[dict[str, np.ndarray]] = []
    for index, offset in enumerate(offsets, start=1):
        mix = pcm.window(offset, frames)
        spec = spectrum_input(torch, model, poc, mix)
        samples.append({"args_0": mix, "args_1": spec})
        print(f"Prepared calibration window {index} at frame {offset}", flush=True)
    return samples


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
        "referenceRootMeanSquare": signal_rms,
        "normalizedRootMeanSquareError": error_rms / max(signal_rms, 1e-12),
        "signalToNoiseDb": 20.0 * math.log10(max(signal_rms, 1e-12) / max(error_rms, 1e-12)),
        "correlation": correlation,
    }


def make_runner(litert: Any, path: Path) -> tuple[Any, Any]:
    interpreter = litert.Interpreter(model_path=str(path), num_threads=8)
    signatures = interpreter.get_signature_list()
    if SIGNATURE not in signatures:
        raise RuntimeError(f"Missing {SIGNATURE} signature: {signatures}")
    expected = signatures[SIGNATURE]
    if expected["inputs"] != ["args_0", "args_1"]:
        raise RuntimeError(f"Unexpected inputs: {expected}")
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_signature_runner(SIGNATURE)


def run_core(runner: Any, mix: np.ndarray, spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outputs = runner(args_0=mix, args_1=spec)
    values = [np.asarray(value) for value in outputs.values()]
    frequency = next((value for value in values if value.ndim == 5), None)
    waveform = next((value for value in values if value.ndim == 4), None)
    if frequency is None or waveform is None:
        raise RuntimeError(f"Unexpected LiteRT output shapes: {[value.shape for value in values]}")
    return frequency, waveform


def reconstruct(torch: Any, model: Any, poc: Any, frequency: np.ndarray, waveform: np.ndarray, frames: int) -> np.ndarray:
    with torch.inference_mode():
        final = (
            model._ispec(
                poc.channels_to_spec(torch, torch.from_numpy(frequency.astype(np.float32, copy=False))),
                length=frames,
            )
            + torch.from_numpy(waveform.astype(np.float32, copy=False))
        )
    return final.cpu().numpy()


def aggregate_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    return {
        f"worst{key[0].upper()}{key[1:]}": (
            min(item[key] for item in items)
            if key in {"signalToNoiseDb", "correlation"}
            else max(item[key] for item in items)
        )
        for key in keys
    }


def main() -> int:
    args = parse_args()
    if sha256(args.fp32_model) != EXPECTED_FP32_SHA256:
        raise RuntimeError("The FP32 LiteRT input is not the validated pinned conversion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    worker_root = args.poc_root / "python" / "worker"
    sys.path.insert(0, str(worker_root))

    import ai_edge_litert.interpreter as litert
    import litert_torch
    from litert_torch.quantize import pt2e_quantizer
    import torch
    from demucs.api import Separator
    from model_download import verify_model
    from torchao.quantization.pt2e import quantize_pt2e

    poc = load_poc_module(args.poc_root)
    repository = verify_model(MODEL_NAME, args.model_root)
    weight_path = repository / f"{MODEL_REVISION}.th"
    weight_hash = sha256(weight_path)
    if weight_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"Unexpected PyTorch checkpoint: {weight_hash}")

    separator = Separator(model=MODEL_NAME, repo=repository, device="cpu", shifts=1, overlap=0.25, split=True)
    loaded_model = separator.model.cpu().eval()
    model = poc.unwrap_single_model(loaded_model).cpu().eval()
    poc.install_deterministic_pos_embedding(model)
    frames = int(round(float(model.segment) * model.samplerate))
    if tuple(model.sources) != SOURCE_NAMES:
        raise RuntimeError(f"Unexpected source order: {model.sources}")

    pcms = [StereoPcm(path) for path in args.calibration_pcm]
    calibration_sources: list[tuple[StereoPcm, list[int]]] = []
    validation_sources: list[tuple[StereoPcm, list[int]]] = []
    samples: list[dict[str, np.ndarray]] = []
    for pcm in pcms:
        calibration_offsets = choose_calibration_offsets(
            pcm.frames, frames, args.calibration_windows
        )
        validation_offsets = choose_validation_offsets(
            pcm.frames, frames, args.validation_windows, calibration_offsets
        )
        calibration_sources.append((pcm, calibration_offsets))
        validation_sources.append((pcm, validation_offsets))
        print(f"Preparing real-song calibration source: {pcm.path}", flush=True)
        samples.extend(
            calibration_samples(torch, model, poc, pcm, calibration_offsets, frames)
        )

    print("Exporting the pinned PyTorch core for PT2E quantization...", flush=True)
    quantization_started = time.perf_counter()
    core = poc.HTDemucsCore.build(torch, model).eval()
    example_mix = torch.from_numpy(samples[0]["args_0"])
    example_spec = torch.from_numpy(samples[0]["args_1"])
    exported_core = torch.export.export(
        core, (example_mix, example_spec), strict=True
    ).module()

    # Broadly annotating adds/muls also catches integer-only shape arithmetic in
    # this graph. Limit PT2E to the neural compute kernels that QNN can execute
    # efficiently and that dominate both VTCM use and model size.
    class CorePt2eQuantizer(pt2e_quantizer.PT2EQuantizer):
        STATIC_OPS = ["linear", "addmm", "conv_relu", "conv"]

    pt2e = CorePt2eQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared_core = quantize_pt2e.prepare_pt2e(exported_core, pt2e)
    observer_count = sum(
        1
        for _name, module in prepared_core.named_modules()
        if "Observer" in type(module).__name__
    )
    print(
        f"Calibrating {observer_count} observers with {len(samples)} real-song windows...",
        flush=True,
    )
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            prepared_core(
                torch.from_numpy(sample["args_0"]),
                torch.from_numpy(sample["args_1"]),
            )
            print(f"Calibrated window {index}/{len(samples)}", flush=True)

    print("Rewriting supported neural kernels for INT8 compute...", flush=True)
    converted_core = quantize_pt2e.convert_pt2e(
        prepared_core, fold_quantize=False
    )
    graph_targets = [str(node.target) for node in converted_core.graph.nodes]
    quantize_nodes = sum(
        "quantized_decomposed.quantize_per_" in target
        and "dequantize" not in target
        for target in graph_targets
    )
    per_channel_nodes = sum(
        "quantized_decomposed.quantize_per_channel" in target
        and "dequantize" not in target
        for target in graph_targets
    )
    print(
        f"Lowering {quantize_nodes} INT8 boundaries ({per_channel_nodes} per-channel) to LiteRT...",
        flush=True,
    )
    quantized_litert = litert_torch.convert(
        converted_core,
        (example_mix, example_spec),
        lightweight_conversion=True,
        enable_x64=False,
    )
    quantized_litert.export(str(args.output))
    quantization_seconds = time.perf_counter() - quantization_started
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("Quantizer did not produce a model")

    fp32_interpreter, fp32_runner = make_runner(litert, args.fp32_model)
    int8_interpreter, int8_runner = make_runner(litert, args.output)
    window_reports: list[dict[str, Any]] = []
    per_stem_int8: dict[str, list[dict[str, float]]] = {name: [] for name in SOURCE_NAMES}
    per_stem_fp32: dict[str, list[dict[str, float]]] = {name: [] for name in SOURCE_NAMES}

    try:
        total_validation_windows = sum(len(offsets) for _, offsets in validation_sources)
        validation_index = 0
        for pcm, validation_offsets in validation_sources:
            for offset in validation_offsets:
                validation_index += 1
                print(
                    f"Validating holdout window {validation_index}/{total_validation_windows} "
                    f"from {pcm.path.name} at frame {offset}...",
                    flush=True,
                )
                mix = pcm.window(offset, frames)
                spec = spectrum_input(torch, model, poc, mix)

                with torch.inference_mode():
                    torch_started = time.perf_counter()
                    reference = model(torch.from_numpy(mix)).cpu().numpy()
                    torch_seconds = time.perf_counter() - torch_started

                fp32_started = time.perf_counter()
                fp32_frequency, fp32_waveform = run_core(fp32_runner, mix, spec)
                fp32_seconds = time.perf_counter() - fp32_started
                fp32_final = reconstruct(torch, model, poc, fp32_frequency, fp32_waveform, frames)

                int8_started = time.perf_counter()
                int8_frequency, int8_waveform = run_core(int8_runner, mix, spec)
                int8_seconds = time.perf_counter() - int8_started
                int8_final = reconstruct(torch, model, poc, int8_frequency, int8_waveform, frames)

                fp32_metrics = metrics(reference, fp32_final)
                int8_metrics = metrics(reference, int8_final)
                stem_report: dict[str, Any] = {}
                for stem_index, name in enumerate(SOURCE_NAMES):
                    fp32_stem = metrics(reference[:, stem_index], fp32_final[:, stem_index])
                    int8_stem = metrics(reference[:, stem_index], int8_final[:, stem_index])
                    per_stem_fp32[name].append(fp32_stem)
                    per_stem_int8[name].append(int8_stem)
                    stem_report[name] = {"fp32VsPyTorch": fp32_stem, "int8VsPyTorch": int8_stem}

                window_reports.append({
                    "pcm": str(pcm.path),
                    "pcmSha256": sha256(pcm.path),
                    "offsetFrames": offset,
                    "offsetSeconds": offset / float(model.samplerate),
                    "seconds": {"torchFull": torch_seconds, "fp32LiteRtCore": fp32_seconds, "int8LiteRtCore": int8_seconds},
                    "fp32VsPyTorch": fp32_metrics,
                    "int8VsPyTorch": int8_metrics,
                    "perStem": stem_report,
                })
    finally:
        # The Python binding releases native arenas when the interpreter is
        # collected; explicit deletion keeps peak memory bounded before JSON.
        del fp32_runner, int8_runner, fp32_interpreter, int8_interpreter

    aggregate_int8 = {name: aggregate_metrics(items) for name, items in per_stem_int8.items()}
    aggregate_fp32 = {name: aggregate_metrics(items) for name, items in per_stem_fp32.items()}
    # A stem passes only if every unseen real-song window remains extremely
    # correlated and the quantization error stays at least 25 dB below signal.
    gate_failures = [
        name
        for name, item in aggregate_int8.items()
        if item["worstCorrelation"] < 0.99 or item["worstSignalToNoiseDb"] < 25.0
    ]

    report = {
        "model": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "sourceOrder": list(SOURCE_NAMES),
        "weights": {"path": str(weight_path), "sha256": weight_hash},
        "calibration": {
            "sampleRate": model.samplerate,
            "sources": [
                {
                    "pcm": str(pcm.path),
                    "pcmSha256": sha256(pcm.path),
                    "frames": pcm.frames,
                    "offsetFrames": offsets,
                    "windowCount": len(offsets),
                }
                for pcm, offsets in calibration_sources
            ],
            "totalWindowCount": sum(len(offsets) for _, offsets in calibration_sources),
        },
        "validation": {
            "holdouts": [
                {"pcm": str(pcm.path), "offsetFrames": offsets}
                for pcm, offsets in validation_sources
            ],
            "windowCount": sum(len(offsets) for _, offsets in validation_sources),
            "windows": window_reports,
            "aggregatePerStem": {
                name: {"fp32VsPyTorch": aggregate_fp32[name], "int8VsPyTorch": aggregate_int8[name]}
                for name in SOURCE_NAMES
            },
        },
        "qualityGate": {
            "criteria": {"minimumPerStemCorrelation": 0.99, "minimumPerStemSignalToNoiseDb": 25.0},
            "acceptedForDeployment": not gate_failures,
            "failedStems": gate_failures,
        },
        "artifacts": {
            "fp32": {"path": str(args.fp32_model), "bytes": args.fp32_model.stat().st_size, "sha256": sha256(args.fp32_model)},
            "int8": {"path": str(args.output), "bytes": args.output.stat().st_size, "sha256": sha256(args.output)},
            "sizeReductionPercent": 100.0 * (1.0 - args.output.stat().st_size / args.fp32_model.stat().st_size),
        },
        "quantization": {
            "recipe": "PyTorch PT2E static INT8 for conv/linear neural kernels",
            "weights": "int8 symmetric per-channel",
            "activations": "int8 affine per-tensor",
            "compute": "integer",
            "observerCount": observer_count,
            "quantizeNodeCount": quantize_nodes,
            "perChannelQuantizeNodeCount": per_channel_nodes,
            "seconds": quantization_seconds,
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.report),
        "int8Model": str(args.output),
        "acceptedForDeployment": not gate_failures,
        "failedStems": gate_failures,
    }, ensure_ascii=False, indent=2))
    return 0 if not gate_failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
