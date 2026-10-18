#!/usr/bin/env python3
"""Apply a narrow, reversible GDN reshape patch to an installed vLLM source."""

from __future__ import annotations

import ast
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


OLD = "            z = z.reshape(z.size(0), -1, self.head_v_dim)"
NEW = """            z = z.reshape(
                z.size(0), self.num_v_heads // self.tp_size, self.head_v_dim
            )"""
ALREADY_PATCHED = NEW


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/qwen_gdn_linear_attn.py")
        return 2

    path = Path(sys.argv[1])
    source = path.read_text()
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "QwenGatedDeltaNetAttention"
    ]
    if len(classes) != 1:
        print(f"STOP: expected one QwenGatedDeltaNetAttention class, found {len(classes)}")
        return 1

    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "forward_cuda"
    ]
    if len(methods) != 1:
        print(f"STOP: expected one forward_cuda method, found {len(methods)}")
        return 1

    lines = source.splitlines(keepends=True)
    method = methods[0]
    start, end = method.lineno - 1, method.end_lineno
    method_source = "".join(lines[start:end])

    if ALREADY_PATCHED in method_source:
        print("Patch already present in QwenGatedDeltaNetAttention.forward_cuda; no change made.")
        for number, line in enumerate(method_source.splitlines(), method.lineno):
            if "z = z.reshape" in line or "z.size(0), self.num_v_heads" in line:
                print(f"{number}: {line}")
        return 0

    if method_source.count(OLD) != 1:
        print(
            "STOP: expected exactly one original reshape in forward_cuda, "
            f"found {method_source.count(OLD)}; no change made."
        )
        return 1

    # This method already allocates core_attn_out with the same local-head formula.
    expected_shape = "self.num_v_heads // self.tp_size, self.head_v_dim"
    if expected_shape not in method_source:
        print("STOP: matching local-head shape not found in forward_cuda; no change made.")
        return 1

    updated_method = method_source.replace(OLD, NEW, 1)
    updated_source = "".join(lines[:start]) + updated_method + "".join(lines[end:])
    backup = path.with_name(
        path.name + ".pre-explicit-gdn-heads." + datetime.now().strftime("%Y%m%d-%H%M%S") + ".bak"
    )
    shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".explicit-heads.tmp")
    temporary.write_text(updated_source)
    temporary.chmod(path.stat().st_mode)
    os.replace(temporary, path)

    print(f"Backup: {backup}")
    print("Patched only QwenGatedDeltaNetAttention.forward_cuda.")
    for number, line in enumerate(updated_method.splitlines(), method.lineno):
        if "z = z.reshape" in line or "z.size(0), self.num_v_heads" in line:
            print(f"{number}: {line}")
    print("No model weights were loaded by this patch operation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
