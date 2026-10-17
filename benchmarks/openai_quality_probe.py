#!/usr/bin/env python3
"""Run a fixed small quality set against an OpenAI-compatible chat endpoint.

The server is managed outside this script.  Run the same questions and
settings once for each model / quantization, then compare the archived JSON
responses manually or with a separate judge.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SCRIPT_VERSION = "0.1.1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def is_loopback_url(url: str) -> bool:
    hostname = urlparse(url).hostname
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def open_request(request: urllib.request.Request, timeout_s: float):
    # The model server is normally local while HTTP(S)_PROXY is used for
    # downloading public prompt files.  Never proxy local API traffic.
    if is_loopback_url(request.full_url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout_s)
    return urllib.request.urlopen(request, timeout=timeout_s)


def read_questions(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path)
    raw = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        metadata: dict[str, Any] = {"source_file": str(source)}
    else:
        data = json.loads(raw)
        if isinstance(data, list):
            rows = data
            metadata = {"source_file": str(source)}
        elif isinstance(data, dict):
            rows = data.get("items") or data.get("questions") or []
            metadata = {
                key: value
                for key, value in data.items()
                if key not in {"items", "questions"}
            }
            metadata.setdefault("source_file", str(source))
        else:
            raise ValueError("question file must contain a JSON object, array or JSONL")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"question {index} is not an object")
        prompt = row.get("prompt")
        if prompt is None:
            turns = row.get("turns")
            if isinstance(turns, list) and turns:
                prompt = turns[0]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"question {index} has no usable prompt")
        normalized.append(
            {
                "question_id": row.get("question_id", index),
                "category": row.get("category", "unknown"),
                "prompt": prompt,
            }
        )
    if not normalized:
        raise ValueError("question file contains no questions")
    return metadata, normalized


def models_endpoint(endpoint: str) -> str:
    marker = "/v1/chat/completions"
    if marker in endpoint:
        return endpoint.split(marker, 1)[0] + "/v1/models"
    return endpoint.rstrip("/") + "/v1/models"


def discover_model(endpoint: str, timeout_s: float) -> str:
    request = urllib.request.Request(
        models_endpoint(endpoint),
        headers={"Accept": "application/json"},
        method="GET",
    )
    with open_request(request, timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    candidates = data.get("data") or data.get("models") or []
    if candidates:
        first = candidates[0]
        model = first.get("id") or first.get("name") or first.get("model")
        if model:
            return str(model)
    raise RuntimeError("could not discover a model from " + models_endpoint(endpoint))


def usage_value(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = choices[0].get("text")
    return text if isinstance(text, str) else ""


def json_post(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> tuple[int | None, dict[str, Any], str | None]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with open_request(request, timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"raw": raw[:4000]}
            return response.status, data, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"raw": raw[:4000]}
        return exc.code, data, None
    except Exception as exc:  # noqa: BLE001 - preserve per-question failures
        return None, {}, f"{type(exc).__name__}: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="JSON or JSONL quality prompt file")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible chat endpoint")
    parser.add_argument("--label", required=True, help="Result label, e.g. q4 or q5")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", help="Model ID; discovered from /v1/models when omitted")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="Fixed system prompt sent for every question",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tokens <= 0 or args.timeout <= 0:
        raise SystemExit("--max-tokens and --timeout must be positive")

    question_metadata, questions = read_questions(args.questions)
    model = args.model or discover_model(args.endpoint, args.timeout)
    started_at = utc_now()
    results: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": question["prompt"]},
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        request_started = time.perf_counter()
        status, data, error = json_post(args.endpoint, payload, args.timeout)
        elapsed_s = time.perf_counter() - request_started
        usage = data.get("usage") if isinstance(data, dict) else {}
        completion_tokens = usage_value(usage, "completion_tokens")
        timings = data.get("timings") if isinstance(data, dict) else None
        server_generation_tps = None
        if isinstance(timings, dict):
            value = timings.get("predicted_per_second")
            if isinstance(value, (int, float)):
                server_generation_tps = float(value)
        observed_tps = (
            completion_tokens / elapsed_s
            if completion_tokens is not None and elapsed_s > 0
            else None
        )
        success = error is None and status is not None and 200 <= status < 300
        result = {
            "index": index,
            "question_id": question["question_id"],
            "category": question["category"],
            "prompt": question["prompt"],
            "http_status": status,
            "success": success,
            "error": error,
            "elapsed_s": round(elapsed_s, 6),
            "prompt_tokens": usage_value(usage, "prompt_tokens"),
            "completion_tokens": completion_tokens,
            "total_tokens": usage_value(usage, "total_tokens"),
            "observed_output_tokens_per_s": (
                round(observed_tps, 6) if observed_tps is not None else None
            ),
            "server_generation_tokens_per_s": server_generation_tps,
            "response_text": extract_content(data) if isinstance(data, dict) else "",
            "response_choices": data.get("choices", []),
            "timings": timings,
        }
        if not success and isinstance(data, dict) and data:
            result["error_response"] = data
        results.append(result)
        print(
            f"[quality] {index}/{len(questions)} category={question['category']} "
            f"status={status} elapsed={elapsed_s:.2f}s",
            file=sys.stderr,
        )

    successful = [item for item in results if item["success"]]
    elapsed_values = [item["elapsed_s"] for item in successful]
    observed_tps_values = [
        item["observed_output_tokens_per_s"]
        for item in successful
        if item["observed_output_tokens_per_s"] is not None
    ]
    summary = {
        "empty_answer_count": sum(not item["response_text"].strip() for item in successful),
        "length_finish_count": sum(
            any(choice.get("finish_reason") == "length" for choice in item["response_choices"])
            for item in successful
        ),
        "question_count": len(results),
        "success_count": len(successful),
        "failure_count": len(results) - len(successful),
        "mean_elapsed_s": statistics.fmean(elapsed_values) if elapsed_values else None,
        "mean_observed_output_tokens_per_s": (
            statistics.fmean(observed_tps_values) if observed_tps_values else None
        ),
        "completion_tokens": sum(
            item["completion_tokens"] or 0 for item in successful
        ),
    }
    output = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": utc_now(),
        "started_at_utc": started_at,
        "label": args.label,
        "endpoint": args.endpoint,
        "model": model,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        },
        "workload": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "system_prompt": args.system_prompt,
            "question_file": os.path.abspath(args.questions),
            "question_metadata": question_metadata,
        },
        "summary": summary,
        "items": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {output_path}")
    return 0 if summary["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
