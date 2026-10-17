# Results

Store small, sanitized JSON summaries produced by the shared benchmark here. Do not commit model weights, tokens, cookies, temporary gateway URLs, or raw private logs.

The first planned artifact is the llama.cpp E1 baseline for Qwen3.8-Flash-Next / UD-Q4_K_XL.

For ephemeral benchmark instances, a raw JSON capture may also be retained when it contains only benchmark metadata and measurements. Intermediate captures without complete HBM data remain local and match `*-pre-hbm.json` in `.gitignore`.

## Directory layout

New artifacts use the following layout so language-model serving results stay separate from future vision and video work:

```text
results/
  llm/
    qwen3.8-flash-next/
      qwen3.8-flash-next-llamacpp-e1.json
      qwen3.8-flash-next-llamacpp-e2.json
      qwen3.8-flash-next-llamacpp-e3.json
      qwen3.8-flash-next-llamacpp-e4-serving.json
      qwen3.8-flash-next-llamacpp-e4-quality.json
  vision/
  video/
```

The GitHub repository keeps reviewed evidence in the model directory; filenames carry the model, backend and experiment name, while the merged JSON records preserve quantization and original source filenames inside `runs[].result`. E1–E3 use one JSON each; E4 uses two because serving/startup and quality probes have different schemas. An ephemeral remote benchmark instance may write temporary captures directly under its flat `results/` directory; merge reviewed JSON files into `results/llm/<model>/` before syncing them back to the repository.

For this model, `*-llamacpp-*.json` refers to the GGUF Q4/Q5 route, while future `*-vllm-fp8-*.json` and `*-sglang-fp8-*.json` refer to the official FP8 Safetensors checkpoint. Those are deployment-route comparisons across different weight formats.
