#!/usr/bin/env python3
"""Print LiteRT operator/tensor details for selected node ids."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow.lite.python import schema_py_generated as schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("nodes", nargs="+", type=int)
    args = parser.parse_args()

    packed = schema.Model.GetRootAsModel(args.model.read_bytes(), 0)
    model = schema.ModelT.InitFromObj(packed)
    graph = model.subgraphs[0]
    inverse = {
        value: name
        for name, value in vars(schema.BuiltinOperator).items()
        if isinstance(value, int)
    }
    print(f"operators={len(graph.operators)} tensors={len(graph.tensors)}")
    for node in args.nodes:
        operator = graph.operators[node]
        kind = inverse.get(model.operatorCodes[operator.opcodeIndex].builtinCode, "UNKNOWN")

        def describe(tensor_id: int) -> str:
            tensor = graph.tensors[tensor_id]
            name = (tensor.name or b"").decode(errors="replace")
            return f"{tensor_id}:{list(tensor.shape)}:{name}"

        inputs = [describe(int(value)) for value in operator.inputs]
        outputs = [describe(int(value)) for value in operator.outputs]
        print(f"\nnode={node} op={kind}\n  in=" + "\n     ".join(inputs) + "\n  out=" + "\n      ".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
