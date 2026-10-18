#!/usr/bin/env python3
"""Apply fixed PLE buffer and query_start_loc tail handling to installed vLLM."""

from __future__ import annotations

import ast
import difflib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# This is a method-scoped adaptation of the behavior in:
# devtyagi3909/vllm commit bb5dc2b40a1e0a98dfcd7216611168808ef40356
# (linked to vllm-project/vllm issue #61486). It does not change the GDN reshape.
TRANSFORMS = {
    "_prepare_ngram_context": [
        (
            """        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]""",
            """        num_reqs = input_batch.num_reqs
        context = self.ngram_context""",
        ),
    ],
    "prepare_inputs": [
        (
            """        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])""",
            """        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(
            input_batch.query_start_loc[: num_reqs_padded + 1]
        )
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])""",
        ),
    ],
    "prepare_dummy_inputs": [
        (
            "        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]",
            "        query_start_loc = self.ple_query_start_loc",
        ),
        (
            "        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])",
            """        torch.cumsum(
            query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1]
        )
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)""",
        ),
        (
            "        ngram_context = self.ngram_context[:num_reqs]",
            "        ngram_context = self.ngram_context",
        ),
    ],
}


def get_methods(source: str) -> tuple[dict[str, ast.AST], list[str]]:
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpModelState"
    ]
    if len(classes) != 1:
        raise ValueError(
            f"expected one Qwen4ExpModelState class, found {len(classes)}"
        )
    methods = {
        node.name: node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = set(TRANSFORMS) - methods.keys()
    if missing:
        raise ValueError(f"expected methods not found: {sorted(missing)}")
    return methods, source.splitlines(keepends=True)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/amd/model_state.py")
        return 2

    path = Path(sys.argv[1])
    source = path.read_text()
    try:
        methods, lines = get_methods(source)
    except (SyntaxError, ValueError) as exc:
        print(f"STOP: {exc}; no change made.")
        return 1

    edits: list[tuple[int, int, str]] = []
    all_already_applied = True
    for name, transforms in TRANSFORMS.items():
        node = methods[name]
        start, end = node.lineno - 1, node.end_lineno
        block = "".join(lines[start:end])
        updated = block
        for old, new in transforms:
            old_count = updated.count(old)
            if old_count == 1:
                all_already_applied = False
                updated = updated.replace(old, new, 1)
            elif old_count == 0 and updated.count(new) == 1:
                continue
            else:
                print(
                    f"STOP: {name}: expected one old snippet or one patched snippet; "
                    f"old_count={old_count}. No change made."
                )
                return 1
        edits.append((node.lineno - 1, node.end_lineno, updated))

    if all_already_applied:
        print("Fixed PLE buffer handling is already present; no change made.")
        return 0

    # Rebuild from original, non-overlapping source spans. Replacing methods
    # in-place using AST line numbers would shift later spans after an earlier
    # method grows.
    updated_parts: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(edits):
        if start < cursor:
            print("STOP: overlapping method spans; no change made.")
            return 1
        updated_parts.extend(("".join(lines[cursor:start]), replacement))
        cursor = end
    updated_parts.append("".join(lines[cursor:]))
    updated_source = "".join(updated_parts)
    try:
        ast.parse(updated_source)
    except SyntaxError as exc:
        print(f"STOP: generated source is invalid ({exc}); no change made.")
        return 1

    backup = path.with_name(
        path.name
        + ".pre-fixed-ple-buffers."
        + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        + ".bak"
    )
    shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".fixed-ple-buffers.tmp")
    temporary.write_text(updated_source)
    temporary.chmod(path.stat().st_mode)
    os.replace(temporary, path)

    print(f"Backup: {backup}")
    print("Patched only Qwen4ExpModelState._prepare_ngram_context/prepare_inputs/prepare_dummy_inputs.")
    print("PLE buffers now retain fixed shapes and unused query_start_loc tail entries are padded.")
    print("This operation did not load model weights or start a server.")
    diff = difflib.unified_diff(
        source.splitlines(),
        updated_source.splitlines(),
        fromfile="installed model_state.py (before)",
        tofile="installed model_state.py (after)",
        lineterm="",
    )
    print("\n".join(diff))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
