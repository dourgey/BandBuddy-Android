#!/usr/bin/env python3
"""Map selected LiteRT node ids between fixed-shape exports by tensor identity."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow.lite.python import schema_py_generated as schema


def load(path: Path):
    return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(), 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("nodes", nargs="+", type=int)
    args = parser.parse_args()

    source = load(args.source)
    target = load(args.target)
    source_graph = source.subgraphs[0]
    target_graph = target.subgraphs[0]

    target_by_key: dict[tuple[int, str], list[int]] = {}
    for index, operator in enumerate(target_graph.operators):
        kind = int(target.operatorCodes[operator.opcodeIndex].builtinCode)
        output_names = tuple(
            (target_graph.tensors[int(tensor)].name or b"").decode(errors="replace")
            for tensor in operator.outputs
        )
        for name in output_names:
            target_by_key.setdefault((kind, name), []).append(index)

    for node in args.nodes:
        operator = source_graph.operators[node]
        kind = int(source.operatorCodes[operator.opcodeIndex].builtinCode)
        names = [
            (source_graph.tensors[int(tensor)].name or b"").decode(errors="replace")
            for tensor in operator.outputs
        ]
        matches = sorted({candidate for name in names for candidate in target_by_key.get((kind, name), [])})
        print(f"{node} -> {matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
