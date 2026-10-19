#!/usr/bin/env python3
"""Probe the PLE request-capacity guard with CPU tensors; load no vLLM weights.

This deliberately uses the eager *backend* of torch.compile to isolate Dynamo
shape guards. It does not test Inductor kernels, the full PLE forward, or graphs.
"""

import json

import torch


class RequestCapacityGuard(torch.nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.register_buffer("padded_buffer", torch.zeros(capacity, 4))

    def forward(self, query_start_loc):
        num_reqs = query_start_loc.numel() - 1
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError("request workspace capacity exceeded")
        return query_start_loc + 1


def probe(capacity):
    torch._dynamo.reset()
    result = {"request_capacity": capacity, "query_lengths_checked": []}
    query_start_loc = torch.arange(2, dtype=torch.int64)
    torch._dynamo.mark_dynamic(query_start_loc, 0)
    compiled = torch.compile(
        RequestCapacityGuard(capacity), backend="eager", fullgraph=True
    )
    try:
        torch.testing.assert_close(compiled(query_start_loc), query_start_loc + 1)
        result["query_lengths_checked"].append(2)
        if capacity == 2:
            query_start_loc = torch.arange(3, dtype=torch.int64)
            torch._dynamo.mark_dynamic(query_start_loc, 0)
            torch.testing.assert_close(
                compiled(query_start_loc), query_start_loc + 1
            )
            result["query_lengths_checked"].append(3)
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def main():
    cases = [probe(1), probe(2)]
    baseline_error = cases[0].get("error", "")
    supported = (
        cases[0]["status"] == "failed"
        and "specialized" in baseline_error
        and cases[1]["status"] == "passed"
    )
    print(json.dumps({
        "torch_version": torch.__version__,
        "torch_path": torch.__file__,
        "device": "cpu",
        "backend": "eager",
        "fullgraph": True,
        "cases": cases,
        "capacity_guard_hypothesis_supported": supported,
        "scope": "isolated Dynamo guard only; full O1 validation still required",
    }, indent=2))
    return 0 if supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
