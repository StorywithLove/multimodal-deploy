#!/usr/bin/env python3
"""Run a small, backend-neutral serving matrix against an OpenAI-compatible API.

The server is intentionally managed outside this script. This makes the same
client usable with llama.cpp, vLLM and SGLang while keeping backend flags in
the corresponding deployment record.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import ipaddress
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any


SCRIPT_VERSION = "0.2.2"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value <= 0:
                raise ValueError("matrix values must be positive")
            values.append(value)
    if not values:
        raise ValueError("matrix list is empty")
    return values


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def is_loopback_url(url: str) -> bool:
    """Return whether a request must stay local rather than use a proxy."""
    hostname = urlparse(url).hostname
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def open_request(request: urllib.request.Request, timeout_s: float):
    """Open requests while preserving local API traffic from download proxies."""
    if is_loopback_url(request.full_url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=timeout_s
        )
    return urllib.request.urlopen(request, timeout=timeout_s)


def json_request(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with open_request(request, timeout_s) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            detail = {"error": raw[:1000]}
        return exc.code, detail


def discover_model(endpoint: str, timeout_s: float) -> str:
    marker = "/v1/chat/completions"
    if marker in endpoint:
        models_endpoint = endpoint.split(marker, 1)[0] + "/v1/models"
    else:
        models_endpoint = endpoint.rstrip("/") + "/v1/models"
    request = urllib.request.Request(models_endpoint, headers={"Accept": "application/json"})
    with open_request(request, timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    candidates = data.get("data") or data.get("models") or []
    if candidates:
        first = candidates[0]
        model = first.get("id") or first.get("name") or first.get("model")
        if model:
            return str(model)
    raise RuntimeError("could not discover a model from " + models_endpoint)


def build_prompt(target_tokens: int) -> str:
    # This is a deterministic approximate input. The result records the
    # server-reported prompt_tokens instead of pretending this is exact.
    unit = (
        "This is a deterministic serving benchmark sentence. "
        "Keep the input context and answer the final instruction. "
    )
    body = (unit * max(1, (target_tokens * 4) // len(unit) + 1))[: target_tokens * 4]
    # A short answer would make decode throughput meaningless.  This asks the
    # model to consume the completion budget while remaining deterministic at
    # temperature 0; the server-reported completion count remains the source
    # of truth for every result.
    return (
        body
        + "\nFinal instruction: repeatedly output the exact word 'benchmark' "
        "separated by one space until the generation limit stops you. "
        "Do not add an explanation, punctuation, or closing text."
    )


def make_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a serving benchmark assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }


def usage_value(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def run_one(
    request_id: int,
    target_context: int,
    endpoint: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "request_id": request_id,
        "target_context": target_context,
        "started_at_utc": utc_now(),
    }
    try:
        status, response = json_request(
            endpoint,
            make_payload(
                model,
                build_prompt(target_context),
                max_tokens,
                temperature,
            ),
            timeout_s,
        )
        usage = response.get("usage", {})
        record.update(
            {
                "http_status": status,
                "ok": 200 <= status < 300 and "error" not in response,
                "prompt_tokens": usage_value(usage, "prompt_tokens"),
                "completion_tokens": usage_value(usage, "completion_tokens"),
            }
        )
        if "error" in response:
            record["error"] = response["error"]
    except Exception as exc:  # A failed request is evidence, not a script crash.
        record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    record["latency_s"] = round(time.perf_counter() - started, 6)
    return record


def stream_has_output(event: dict[str, Any]) -> bool:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    delta = choices[0].get("delta") or {}
    if not isinstance(delta, dict):
        return False
    return bool(
        delta.get("content")
        or delta.get("reasoning_content")
        or delta.get("tool_calls")
    )


def run_ttft_probe(
    endpoint: str,
    model: str,
    target_context: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> dict[str, Any]:
    payload = make_payload(
        model,
        build_prompt(target_context),
        max_tokens,
        temperature,
        stream=True,
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_event_s: float | None = None
    first_output_s: float | None = None
    try:
        with open_request(request, timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                elapsed = time.perf_counter() - started
                first_event_s = first_event_s or elapsed
                if first_output_s is None and stream_has_output(event):
                    first_output_s = elapsed
                    break
        return {
            "ok": True,
            "first_event_s": first_event_s,
            "ttft_s": first_output_s if first_output_s is not None else first_event_s,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def value_to_gib(value: Any, key: str) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            number = float(value.strip().replace(",", ""))
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        return None
    lower = key.lower()
    if "(b)" in lower or "bytes" in lower or lower.endswith("_b"):
        return number / (1024**3)
    if "mb" in lower:
        return number / 1024
    if "gb" in lower:
        return number
    # rocm-smi's JSON normally includes units in the key. For a bare large
    # number, MiB is the least surprising fallback for a VRAM metric.
    return number / 1024 if number > 1024 else number


def find_vram_used(value: Any) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            lower = str(key).lower()
            if "vram" in lower and "used" in lower:
                parsed = value_to_gib(item, lower)
                if parsed is not None:
                    return parsed
            nested = find_vram_used(item)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = find_vram_used(item)
            if nested is not None:
                return nested
    return None


class GpuSampler:
    def __init__(self, command: str, interval_s: float, enabled: bool) -> None:
        self.command = command
        self.interval_s = interval_s
        self.enabled = enabled and bool(command)
        self.samples_gib: list[float] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def sample_once(self) -> None:
        try:
            completed = subprocess.run(
                shlex.split(self.command),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or "GPU command failed")
            parsed = json.loads(completed.stdout)
            used_gib = find_vram_used(parsed)
            if used_gib is None:
                raise RuntimeError(
                    "GPU command JSON contains no recognized VRAM-used value"
                )
            self.samples_gib.append(round(used_gib, 4))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if message not in self.errors:
                self.errors.append(message)

    def loop(self) -> None:
        while not self.stop_event.is_set():
            self.sample_once()
            self.stop_event.wait(self.interval_s)

    def __enter__(self) -> "GpuSampler":
        if self.enabled:
            self.sample_once()
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        if self.enabled:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=max(1.0, self.interval_s * 2))
            self.sample_once()

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "command": self.command if self.enabled else None,
            "samples": len(self.samples_gib),
            "baseline_gib": min(self.samples_gib) if self.samples_gib else None,
            "peak_gib": max(self.samples_gib) if self.samples_gib else None,
            "delta_gib": (
                round(max(self.samples_gib) - min(self.samples_gib), 4)
                if self.samples_gib
                else None
            ),
        }
        if self.errors:
            result["errors"] = self.errors
        return result


def aggregate_cell(
    requests: list[dict[str, Any]],
    cell_started: float,
    cell_finished: float,
    ttft: dict[str, Any] | None,
    gpu: dict[str, Any],
) -> dict[str, Any]:
    successes = [item for item in requests if item.get("ok")]
    latencies = [float(item["latency_s"]) for item in successes]
    prompt_tokens = [
        int(item["prompt_tokens"])
        for item in successes
        if item.get("prompt_tokens") is not None
    ]
    completion_tokens = [
        int(item["completion_tokens"])
        for item in successes
        if item.get("completion_tokens") is not None
    ]
    wall_s = max(cell_finished - cell_started, 0.000001)
    result: dict[str, Any] = {
        "requests": len(requests),
        "completed": len(successes),
        "failed": len(requests) - len(successes),
        "failure_rate": round((len(requests) - len(successes)) / len(requests), 6)
        if requests
        else None,
        "wall_time_s": round(wall_s, 6),
        "throughput": {
            "requests_per_s": round(len(successes) / wall_s, 6),
            "prompt_tokens_per_s": round(sum(prompt_tokens) / wall_s, 6)
            if prompt_tokens
            else None,
            "completion_tokens_per_s": round(sum(completion_tokens) / wall_s, 6)
            if completion_tokens
            else None,
        },
        "latency_s": {
            "p50": round(percentile(latencies, 0.50), 6)
            if latencies
            else None,
            "p95": round(percentile(latencies, 0.95), 6)
            if latencies
            else None,
        },
        "actual_tokens": {
            "prompt_total": sum(prompt_tokens) if prompt_tokens else None,
            "completion_total": sum(completion_tokens)
            if completion_tokens
            else None,
        },
        "ttft_probe": ttft,
        "gpu": gpu,
        "per_request": requests,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON workload contract")
    parser.add_argument("--endpoint", help="OpenAI-compatible /v1/chat/completions URL")
    parser.add_argument("--model", help="Model ID; discovered from /v1/models if omitted")
    parser.add_argument("--label", default="openai-server-matrix")
    parser.add_argument("--output", default="results/openai-server-matrix.json")
    parser.add_argument("--contexts", help="Override context targets, e.g. 512,4096,8192")
    parser.add_argument("--concurrencies", help="Override concurrency levels, e.g. 1,2,4")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--requests-per-cell", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--gpu-command")
    parser.add_argument("--gpu-sample-interval", type=float)
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--ttft-probe", dest="ttft_probe", action="store_true")
    parser.add_argument("--no-ttft-probe", dest="ttft_probe", action="store_false")
    parser.set_defaults(ttft_probe=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    matrix = config.get("matrix", {})
    measurement = config.get("measurement", {})

    try:
        contexts = parse_int_list(args.contexts) if args.contexts else list(matrix["contexts"])
        concurrencies = (
            parse_int_list(args.concurrencies)
            if args.concurrencies
            else list(matrix["concurrencies"])
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"invalid matrix: {exc}") from exc

    max_tokens = args.max_tokens or int(matrix.get("max_tokens", 256))
    requests_per_cell = args.requests_per_cell or int(matrix.get("requests_per_cell", 8))
    warmup_requests = int(matrix.get("warmup_requests", 0))
    temperature = (
        args.temperature if args.temperature is not None else float(matrix.get("temperature", 0.0))
    )
    timeout_s = args.timeout or float(measurement.get("timeout_s", 600))
    gpu_command = (
        args.gpu_command
        if args.gpu_command is not None
        else str(measurement.get("gpu_command", ""))
    )
    gpu_interval = args.gpu_sample_interval or float(
        measurement.get("gpu_sample_interval_s", 0.5)
    )
    ttft_probe = (
        args.ttft_probe
        if args.ttft_probe is not None
        else bool(measurement.get("ttft_probe", False))
    )

    if max_tokens <= 0 or requests_per_cell <= 0 or timeout_s <= 0:
        raise SystemExit("max tokens, requests per cell and timeout must be positive")

    plan = {
        "label": args.label,
        "contexts": contexts,
        "concurrencies": concurrencies,
        "max_tokens": max_tokens,
        "requests_per_cell": requests_per_cell,
        "warmup_requests": warmup_requests,
        "temperature": temperature,
        "ttft_probe": ttft_probe,
        "gpu_sampling": not args.no_gpu_sampling,
        "cells": len(contexts) * len(concurrencies),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not args.endpoint:
        raise SystemExit("--endpoint is required unless --dry-run is used")

    model = args.model or discover_model(args.endpoint, timeout_s)
    all_cells: list[dict[str, Any]] = []
    for context in contexts:
        for concurrency in concurrencies:
            print(
                f"[matrix] context={context} concurrency={concurrency} "
                f"requests={requests_per_cell}",
                file=sys.stderr,
            )
            for warmup_id in range(warmup_requests):
                run_one(
                    warmup_id,
                    context,
                    args.endpoint,
                    model,
                    max_tokens,
                    temperature,
                    timeout_s,
                )

            cell_started = time.perf_counter()
            with GpuSampler(
                gpu_command,
                gpu_interval,
                enabled=not args.no_gpu_sampling,
            ) as sampler:
                ttft = (
                    run_ttft_probe(
                        args.endpoint,
                        model,
                        context,
                        max_tokens,
                        temperature,
                        timeout_s,
                    )
                    if ttft_probe
                    else None
                )
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as executor:
                    futures = [
                        executor.submit(
                            run_one,
                            request_id,
                            context,
                            args.endpoint,
                            model,
                            max_tokens,
                            temperature,
                            timeout_s,
                        )
                        for request_id in range(requests_per_cell)
                    ]
                    requests = [future.result() for future in futures]
                cell_finished = time.perf_counter()
            cell = aggregate_cell(
                requests,
                cell_started,
                cell_finished,
                ttft,
                sampler.summary(),
            )
            cell.update({"target_context": context, "concurrency": concurrency})
            all_cells.append(cell)

    result = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "label": args.label,
        "endpoint": args.endpoint,
        "model": model,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        },
        "config_file": os.path.abspath(args.config),
        "workload": plan,
        "cells": all_cells,
    }
    output_parent = os.path.dirname(args.output)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
