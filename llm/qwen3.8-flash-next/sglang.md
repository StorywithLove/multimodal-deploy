# Qwen3.8-Flash-Next / SGLang / FP8

- 后端：SGLang
- 权重：[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- 格式：Hugging Face Transformers / Safetensors，细粒度 FP8，block size 128
- 硬件目标：单张 AMD MI300X 192GB，ROCm，`gfx942`
- 对照基线：[llama.cpp / GGUF](llama.cpp.md)
- 平行路线：[vLLM / FP8](vllm.md)
- 结果目录：`results/llm/qwen3.8-flash-next/`

## 口径

SGLang 与 vLLM 使用同一个 `Qwen/Qwen3.8-Flash-Next-FP8` 目录；不重新找另一套模型，也不复用 GGUF。SGLang 官方 Qwen 页面给出 FP8 部署入口，但其当前精度/硬件表没有把 MI300X 列为已验证平台，因此先做单请求启动预检，再决定是否进入基准矩阵。

## 模型目录与环境

复用 vLLM 文档中已经下载并校验的 `MODEL_DIR`，但使用独立 Python 环境，避免 vLLM 与 SGLang 依赖互相覆盖。当前实例的 `/mnt/data/lcx` 只有约 100 GB，因此沿用 vLLM 在 `/root/tmp` 中下载的模型；不要再次复制一份权重：

```bash
export MODEL_ROOT=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$MODEL_ROOT/Qwen3.8-Flash-Next-FP8"
export SGLANG_RESULT_ROOT=results
mkdir -p "$SGLANG_RESULT_ROOT"

uv venv --python 3.12 --seed "$MODEL_ROOT/.venv-sglang"
source "$MODEL_ROOT/.venv-sglang/bin/activate"
pip install -U 'sglang>=0.5.20'
python -c 'import sglang, torch; print("sglang", getattr(sglang, "__version__", "unknown")); print("torch", torch.__version__, torch.version.hip); print(torch.cuda.is_available())'
```

## 单请求启动预检

先使用官方最小入口，固定本机端口 30000；暂不添加 MTP、长上下文或其他后端优化参数：

```bash
python3 -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host 127.0.0.1 --port 30000 \
  2>&1 | tee "$SGLANG_RESULT_ROOT/qwen3.8-flash-next-sglang-fp8-startup.log"
```

另一终端验证：

```bash
curl -fsS http://127.0.0.1:30000/health && echo
curl -fsS http://127.0.0.1:30000/v1/models | tee "$SGLANG_RESULT_ROOT/qwen3.8-flash-next-sglang-fp8-models.json"
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-Flash-Next-FP8","messages":[{"role":"user","content":"用一句话说明你是什么模型。"}],"max_tokens":256,"temperature":0}'
```

如果架构、权重或 ROCm kernel 不兼容，保留完整启动日志后停止，不要立即扩展并发或上下文。

## 测评顺序

1. 单请求冒烟：`max_tokens=256`；
2. 复用同一 `configs/benchmark/mi300x-serving-matrix.json` 做 SGLang E1；
3. E1 参数确认无误后，再决定是否做 E2/E3；
4. 质量抽样统一用 `max_tokens=4096`，并同时保存 `content`、`reasoning_content` 和 `finish_reason`。

结果文件使用带后端和格式的名称，例如：

```text
results/llm/qwen3.8-flash-next/
  qwen3.8-flash-next-sglang-fp8-e1.json
  qwen3.8-flash-next-sglang-fp8-e2.json
  qwen3.8-flash-next-sglang-fp8-e3.json
  qwen3.8-flash-next-sglang-fp8-e4-quality.json
```

