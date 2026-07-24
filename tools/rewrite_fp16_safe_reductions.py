#!/usr/bin/env python3
"""Rewrite variance reductions so FP16 accumulation cannot overflow.

LiteRT lowers the variance used by HTDemucs GroupNorm to this sequence::

    centered * centered -> reduce_sum -> multiply(1 / N)

QNN's HTP FP16 backend can overflow in ``reduce_sum`` before the reciprocal is
applied.  The mathematically equivalent sequence below keeps every partial sum
bounded while preserving the FP32 model contract::

    centered * sqrt(1 / N) -> square -> reduce_sum

The script recognizes the complete mean/variance data-flow pattern rather than
editing hard-coded tensor numbers.  Transformer nodes may be excluded when that
single block is intentionally kept on the LiteRT FP32 CPU path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema


PATTERN = ("SUM", "MUL", "SUB", "MUL", "SUM", "MUL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--exclude-nodes",
        action="append",
        default=[],
        metavar="FIRST:LAST",
        help="inclusive operator range not to rewrite",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def excluded_ranges(values: list[str]) -> list[range]:
    result: list[range] = []
    for value in values:
        first_text, separator, last_text = value.partition(":")
        if not separator:
            raise ValueError(f"Expected FIRST:LAST, got {value!r}")
        first, last = int(first_text), int(last_text)
        if first < 0 or last < first:
            raise ValueError(f"Invalid node range: {value!r}")
        result.append(range(first, last + 1))
    return result


def operator_names(model: Any, subgraph: Any) -> list[str]:
    inverse = {
        value: name
        for name, value in vars(schema.BuiltinOperator).items()
        if isinstance(value, int)
    }
    return [
        inverse.get(model.operatorCodes[operator.opcodeIndex].builtinCode, "UNKNOWN")
        for operator in subgraph.operators
    ]


def is_variance_pattern(operators: list[Any], start: int) -> bool:
    first_sum, mean_mul, subtract, square, variance_sum, variance_mul = operators[start : start + 6]
    source = int(first_sum.inputs[0])
    mean = int(mean_mul.outputs[0])
    centered = int(subtract.outputs[0])
    squared = int(square.outputs[0])
    return (
        int(mean_mul.inputs[0]) == int(first_sum.outputs[0])
        and list(map(int, subtract.inputs[:2])) == [source, mean]
        and list(map(int, square.inputs[:2])) == [centered, centered]
        and int(variance_sum.inputs[0]) == squared
        and int(variance_mul.inputs[0]) == int(variance_sum.outputs[0])
    )


def constant_float(model: Any, subgraph: Any, tensor_index: int) -> float:
    tensor = subgraph.tensors[tensor_index]
    if tensor.type != schema.TensorType.FLOAT32:
        raise RuntimeError(f"Expected FLOAT32 constant tensor {tensor_index}")
    data = model.buffers[tensor.buffer].data
    values = np.asarray(data, dtype=np.uint8).tobytes()
    floats = np.frombuffer(values, dtype="<f4")
    if len(floats) != 1:
        raise RuntimeError(f"Expected scalar reciprocal tensor {tensor_index}")
    return float(floats[0])


def append_sqrt_constant(model: Any, subgraph: Any, source_tensor_index: int, value: float, node: int) -> int:
    source_tensor = subgraph.tensors[source_tensor_index]
    tensor = copy.deepcopy(source_tensor)
    tensor.name = f"bandbuddy.fp16_safe_variance_scale.node_{node}".encode()

    buffer = schema.BufferT()
    buffer.data = np.frombuffer(np.asarray([value], dtype="<f4").tobytes(), dtype=np.uint8).copy()
    buffer.offset = 0
    buffer.size = 0
    tensor.buffer = len(model.buffers)
    model.buffers.append(buffer)
    subgraph.tensors.append(tensor)
    return len(subgraph.tensors) - 1


def copy_operator_kind(target: Any, source: Any) -> None:
    target.opcodeIndex = source.opcodeIndex
    target.builtinOptionsType = source.builtinOptionsType
    target.builtinOptions = copy.deepcopy(source.builtinOptions)
    target.builtinOptions2Type = source.builtinOptions2Type
    target.builtinOptions2 = copy.deepcopy(source.builtinOptions2)


def rewrite(model: Any, excluded: list[range]) -> list[dict[str, Any]]:
    if len(model.subgraphs) != 1:
        raise RuntimeError(f"Expected one subgraph, got {len(model.subgraphs)}")
    subgraph = model.subgraphs[0]
    names = operator_names(model, subgraph)
    rewrites: list[dict[str, Any]] = []

    for start in range(len(names) - len(PATTERN) + 1):
        if tuple(names[start : start + len(PATTERN)]) != PATTERN:
            continue
        if any(start in item for item in excluded):
            continue
        if not is_variance_pattern(subgraph.operators, start):
            continue

        square = subgraph.operators[start + 3]
        variance_sum = subgraph.operators[start + 4]
        variance_mul = subgraph.operators[start + 5]
        centered = int(square.inputs[0])
        axes = int(variance_sum.inputs[1])
        reciprocal_tensor = int(variance_mul.inputs[1])
        reciprocal = constant_float(model, subgraph, reciprocal_tensor)
        if not math.isfinite(reciprocal) or reciprocal <= 0.0:
            raise RuntimeError(f"Invalid variance reciprocal at node {start}: {reciprocal}")
        scale = math.sqrt(reciprocal)
        scale_tensor = append_sqrt_constant(model, subgraph, reciprocal_tensor, scale, start)

        # centered * scale
        square.inputs = np.asarray([centered, scale_tensor], dtype=np.int32)

        # scaled * scaled (replace reduce_sum while retaining its output tensor)
        scaled = int(square.outputs[0])
        copy_operator_kind(variance_sum, square)
        variance_sum.inputs = np.asarray([scaled, scaled], dtype=np.int32)
        intermediate = int(variance_sum.outputs[0])
        subgraph.tensors[intermediate].shape = copy.deepcopy(subgraph.tensors[scaled].shape)
        subgraph.tensors[intermediate].shapeSignature = copy.deepcopy(
            subgraph.tensors[scaled].shapeSignature
        )

        # reduce_sum(scaled squared), replacing the final reciprocal multiply.
        original_sum = subgraph.operators[start + 4]
        # ``original_sum`` has just been mutated, so copy SUM kind/options from
        # the first reduction in the recognized pattern instead.
        copy_operator_kind(variance_mul, subgraph.operators[start])
        variance_mul.inputs = np.asarray([intermediate, axes], dtype=np.int32)

        output_tensor = int(variance_mul.outputs[0])
        name = (subgraph.tensors[output_tensor].name or b"").decode(errors="replace")
        rewrites.append({
            "startNode": start,
            "outputTensor": output_tensor,
            "outputName": name,
            "reciprocal": reciprocal,
            "preScale": scale,
        })
    return rewrites


def main() -> int:
    args = parse_args()
    source_bytes = args.input.read_bytes()
    packed = schema.Model.GetRootAsModel(source_bytes, 0)
    model = schema.ModelT.InitFromObj(packed)
    rewrites = rewrite(model, excluded_ranges(args.exclude_nodes))
    if not rewrites:
        raise RuntimeError("No recognized variance reductions were rewritten")

    builder = flatbuffers.Builder(len(source_bytes) + len(rewrites) * 128)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(builder.Output())

    report = {
        "input": str(args.input),
        "inputSha256": sha256(args.input),
        "output": str(args.output),
        "outputSha256": sha256(args.output),
        "outputBytes": args.output.stat().st_size,
        "rewriteCount": len(rewrites),
        "rewrites": rewrites,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
