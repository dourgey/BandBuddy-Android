#!/usr/bin/env python3
"""Gate Android QNN backend candidates against the pinned official Torch model.

Each candidate directory is produced by
``InferenceBackendCandidateInstrumentedTest`` and contains one subdirectory per
real calibration window. Only the planar model input and final six-stem PCM are
required, so multiple hardware backends can be compared without pulling every
large intermediate tensor from the phone.
"""

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
SAMPLES = 343_980
SOURCE_COUNT = 6
MINIMUM_CORRELATION = 0.99999
MINIMUM_SNR_DB = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poc-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
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


def parse_candidates(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory:
            raise ValueError(f"Expected NAME=EXPORT_DIRECTORY, got {value!r}")
        result.append((name, Path(directory)))
    return result


def read_exact(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    expected = math.prod(shape)
    value = np.fromfile(path, dtype="<f4")
    if value.size != expected:
        raise RuntimeError(f"Unexpected float count in {path}: {value.size}, expected {expected}")
    return np.ascontiguousarray(value.reshape(shape))


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64, copy=False).reshape(-1)
    candidate64 = candidate.astype(np.float64, copy=False).reshape(-1)
    if not np.isfinite(candidate64).all():
        return {
            "maxAbsoluteError": float("inf"),
            "rootMeanSquareError": float("inf"),
            "signalToNoiseDb": float("-inf"),
            "correlation": float("nan"),
            "candidatePeak": float("inf"),
        }
    delta = candidate64 - reference64
    signal_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    error_rms = float(np.sqrt(np.mean(delta * delta)))
    reference_centered = reference64 - reference64.mean()
    candidate_centered = candidate64 - candidate64.mean()
    denominator = np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    correlation = (
        float(np.dot(reference_centered, candidate_centered) / denominator)
        if denominator
        else 1.0
    )
    return {
        "maxAbsoluteError": float(np.max(np.abs(delta))),
        "meanAbsoluteError": float(np.mean(np.abs(delta))),
        "rootMeanSquareError": error_rms,
        "normalizedRootMeanSquareError": error_rms / max(signal_rms, 1e-12),
        "signalToNoiseDb": 20.0
        * math.log10(max(signal_rms, 1e-12) / max(error_rms, 1e-12)),
        "correlation": correlation,
        "referenceRms": signal_rms,
        "candidateRms": float(np.sqrt(np.mean(candidate64 * candidate64))),
        "candidatePeak": float(np.max(np.abs(candidate64))),
    }


def main() -> int:
    args = parse_args()
    candidates = parse_candidates(args.candidate)
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

    separator = Separator(
        model=MODEL_NAME,
        repo=repository,
        device="cpu",
        shifts=1,
        overlap=0.25,
        split=True,
    )
    model = poc.unwrap_single_model(separator.model).cpu().eval()
    poc.install_deterministic_pos_embedding(model)
    source_names = tuple(model.sources)
    if len(source_names) != SOURCE_COUNT:
        raise RuntimeError(f"Unexpected source count: {source_names}")

    reference_by_case: dict[str, np.ndarray] = {}
    mix_by_case: dict[str, np.ndarray] = {}
    first_directory = candidates[0][1]
    for case_name in ("f1", "wuyun"):
        mix = read_exact(first_directory / case_name / "mix.f32le", (1, 2, SAMPLES))
        mix_by_case[case_name] = mix
        with torch.inference_mode():
            original_use_train_segment = model.use_train_segment
            model.use_train_segment = False
            reference_by_case[case_name] = model(torch.from_numpy(mix)).cpu().numpy()
            model.use_train_segment = original_use_train_segment

    candidate_reports: list[dict[str, Any]] = []
    all_accepted = True
    for candidate_name, directory in candidates:
        device_summary_path = directory / "device-summary.json"
        device_summary = (
            json.loads(device_summary_path.read_text(encoding="utf-8"))
            if device_summary_path.is_file()
            else None
        )
        case_reports: list[dict[str, Any]] = []
        candidate_accepted = True
        for case_name in ("f1", "wuyun"):
            mix = read_exact(directory / case_name / "mix.f32le", (1, 2, SAMPLES))
            if not np.array_equal(mix, mix_by_case[case_name]):
                raise RuntimeError(
                    f"{candidate_name}/{case_name} does not use the common calibration input"
                )
            output = read_exact(
                directory / case_name / "final.f32le",
                (1, SOURCE_COUNT, 2, SAMPLES),
            )
            overall = metrics(reference_by_case[case_name], output)
            accepted = (
                math.isfinite(overall["correlation"])
                and overall["correlation"] >= MINIMUM_CORRELATION
                and overall["signalToNoiseDb"] >= MINIMUM_SNR_DB
            )
            candidate_accepted &= accepted
            case_reports.append(
                {
                    "name": case_name,
                    "allStems": overall,
                    "perStem": {
                        source: metrics(reference_by_case[case_name][:, index], output[:, index])
                        for index, source in enumerate(source_names)
                    },
                    "accepted": accepted,
                }
            )
        all_accepted &= candidate_accepted
        candidate_reports.append(
            {
                "name": candidate_name,
                "directory": str(directory),
                "deviceSummary": device_summary,
                "cases": case_reports,
                "accepted": candidate_accepted,
            }
        )

    report = {
        "model": MODEL_NAME,
        "modelRevision": MODEL_REVISION,
        "officialTorchWeightSha256": weight_hash,
        "sourceOrder": list(source_names),
        "qualityGate": {
            "minimumCorrelation": MINIMUM_CORRELATION,
            "minimumSignalToNoiseDb": MINIMUM_SNR_DB,
        },
        "candidates": candidate_reports,
        "allAccepted": all_accepted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output),
                "candidates": [
                    {
                        "name": candidate["name"],
                        "accepted": candidate["accepted"],
                        "cases": [
                            {
                                "name": case["name"],
                                "correlation": case["allStems"]["correlation"],
                                "signalToNoiseDb": case["allStems"]["signalToNoiseDb"],
                            }
                            for case in candidate["cases"]
                        ],
                    }
                    for candidate in candidate_reports
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all_accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
