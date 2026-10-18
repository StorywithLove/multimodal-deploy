#!/usr/bin/env python3
"""Remove dynamic-dimension markers for the fixed-size AMD PLE input buffers."""

from __future__ import annotations

import ast
import difflib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET_DIMS = ("query_start_loc", "ngram_context")


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _compile_dims(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpModel"
    ]
    if len(classes) != 1:
        raise ValueError(f"expected one Qwen4ExpModel class, found {len(classes)}")

    matches: list[ast.Dict] = []
    for decorator in classes[0].decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if _call_name(decorator.func) != "support_torch_compile":
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "dynamic_arg_dims" and isinstance(keyword.value, ast.Dict):
                matches.append(keyword.value)
    if len(matches) != 1:
        raise ValueError(
            "expected one support_torch_compile dynamic_arg_dims mapping, "
            f"found {len(matches)}"
        )

    result: dict[str, int] = {}
    for key, value in zip(matches[0].keys, matches[0].values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            if isinstance(value, ast.Constant) and isinstance(value.value, int):
                result[key.value] = value.value
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/amd/model.py")
        return 2

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"STOP: file not found: {path}; no change made.")
        return 1

    source = path.read_text()
    try:
        dims = _compile_dims(source)
    except (SyntaxError, ValueError) as exc:
        print(f"STOP: {exc}; no change made.")
        return 1

    present = [name for name in TARGET_DIMS if name in dims]
    if not present:
        print("PLE input dimensions are already absent from dynamic_arg_dims; no change made.")
        return 0
    if len(present) != len(TARGET_DIMS):
        print(
            "STOP: only part of the expected PLE metadata is present: "
            f"{present}; inspect the installed source manually. No change made."
        )
        return 1
    wrong_values = {name: dims[name] for name in TARGET_DIMS if dims[name] != 0}
    if wrong_values:
        print(f"STOP: unexpected dynamic dimensions {wrong_values}; no change made.")
        return 1

    updated = source
    for name in TARGET_DIMS:
        line = f'        "{name}": 0,\n'
        if updated.count(line) != 1:
            print(
                f"STOP: expected exactly one source line for {name}; "
                "no change made."
            )
            return 1
        updated = updated.replace(line, "", 1)

    try:
        updated_dims = _compile_dims(updated)
    except (SyntaxError, ValueError) as exc:
        print(f"STOP: transformed source is invalid ({exc}); no change made.")
        return 1
    if any(name in updated_dims for name in TARGET_DIMS):
        print("STOP: PLE dynamic markers remain after transformation; no change made.")
        return 1

    backup = path.with_name(
        path.name
        + ".pre-static-ple-dims."
        + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        + ".bak"
    )
    shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".static-ple-dims.tmp")
    temporary.write_text(updated)
    temporary.chmod(path.stat().st_mode)
    os.replace(temporary, path)

    print(f"Backup: {backup}")
    print("Removed only query_start_loc/ngram_context from dynamic_arg_dims.")
    print(f"Retained dynamic dimensions: {updated_dims}")
    print("No model weights were loaded and no server was started.")
    diff = difflib.unified_diff(
        source.splitlines(),
        updated.splitlines(),
        fromfile="installed model.py (before)",
        tofile="installed model.py (after)",
        lineterm="",
    )
    print("\n".join(diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
