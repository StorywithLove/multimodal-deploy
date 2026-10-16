# Shared serving benchmark

`openai_server_matrix.py` is a client-side benchmark for an already running OpenAI-compatible chat endpoint. It does not download models or start / stop servers. That keeps server lifecycle and workload measurement separate.

The same client is used for llama.cpp, vLLM and SGLang. The backend-specific command is recorded in the model's `record.md`; the workload contract is [mi300x-serving-matrix.json](../configs/benchmark/mi300x-serving-matrix.json).

## Dry run

```powershell
python benchmarks/openai_server_matrix.py --config configs/benchmark/mi300x-serving-matrix.json --endpoint http://127.0.0.1:8000/v1/chat/completions --dry-run
```

## First llama.cpp baseline

Run this while `llama-server` is already listening:

```powershell
python benchmarks/openai_server_matrix.py --config configs/benchmark/mi300x-serving-matrix.json --endpoint http://127.0.0.1:8000/v1/chat/completions --label qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e1 --contexts 512,4096,8192 --concurrencies 1 --output results/qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e1.json
```

The script obtains a model ID from `/v1/models` when `--model` is omitted. Use `--model` when a server exposes multiple model aliases.

## What is recorded

Each JSON contains the endpoint label, host metadata, matrix settings, per-request status and latency, actual token usage returned by the server, aggregate prompt / completion throughput, P50/P95 latency, optional TTFT probe, and optional HBM samples from `rocm-smi`.

The target context is a workload input, not a claim about exact tokenizer length. Aggregates use the actual `usage.prompt_tokens` and `usage.completion_tokens` when the server returns them; missing usage is reported instead of guessed.

The deterministic final instruction deliberately runs until the configured generation limit, so completion throughput measures decoding rather than a one-token greeting. It is a serving workload, not a quality evaluation.

For `localhost`, `127.0.0.1`, and `::1` endpoints, the client bypasses inherited HTTP(S) proxy settings. This keeps a temporary download proxy from intercepting the local serving API.

ROCm-SMI may encode VRAM counters as numeric strings; the sampler accepts both strings and numeric JSON values. If a configured GPU command yields no recognized counter, the result records an explicit sampling error instead of silently reporting zero samples.

`record.md` is the interpretation layer: it explains server flags, anomalies and conclusions, and links to the JSON. The script is the measuring instrument; the JSON is the evidence.
