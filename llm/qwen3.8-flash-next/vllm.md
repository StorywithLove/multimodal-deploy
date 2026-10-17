# Qwen3.8-Flash-Next / vLLM / FP8

- 后端：vLLM
- 权重：[Qwen/Qwen3.8-Flash-Next-FP8（Hugging Face）](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- 镜像：[Qwen/Qwen3.8-Flash-Next-FP8（ModelScope）](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next-FP8)
- 格式：Hugging Face Safetensors，细粒度 FP8，block size 128
- 硬件：AMD MI300X 192GB，ROCm 7.2.3，gfx942
- 模型目录：`/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8`
- 持久目录：`/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm0310-20261017/`
- 当前结果：[vLLM 0.31.0 eager 启动结果](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-e0-eager-startup.json)

## 部署

### 遇到的 bug

1. **预编译 wheel 与 glibc 不匹配**：实例为 glibc 2.35，而可用的 ROCm wheel 要求 `manylinux_2_39_x86_64`。因此改用 ROCm 7.2.3、`gfx942` 的源码构建路线。
2. **vLLM 0.30.0 启动超时**：原生扩展可以导入，但 EngineCore 在 600 秒内没有 ready，HTTP 端口未监听，最终出现 `Timed out waiting for engine core processes to start`。
3. **vLLM 0.31.0 默认编译路径失败**：权重可以加载，随后 Torch Dynamo 在动态形状 `query_start_loc.size()[0]` 处触发约束冲突。当前仍需单独排查，不能把它判断为模型下载或基础扩展缺失。
4. **非阻断警告**：启动期间出现部分可选后端或扩展未找到的 warning；在 eager 模式下服务仍能完成启动和请求处理。

### 部署成功的具体要点

- 使用 Python 3.12.13、Torch `2.12.0+git6bbd260`、HIP `7.2.53211`、Triton `3.7.1`。
- 使用 vLLM `0.31.0+rocm723` 源码构建；Qwen3.8-Flash-Next 的 131 个 Safetensors 分片能够完成加载，FP8 权重实际使用 `torch.float8_e4m3fn`。
- 使用 `--enforce-eager` 时，服务完成启动，`/health` 返回 200，`/v1/models` 可查询，聊天请求返回 `finish_reason=stop`。
- 当前 eager 结果是可复现的功能基线，不是最终性能基线；默认非 eager 路径仍待处理。
- 冒烟响应中出现可见的 `</think>` 和前置提示词痕迹，接口可用性已通过，但后续质量测评需要统一处理 reasoning/chat template 的输出格式。
- 持久目录只保存 vLLM 的虚拟环境、wheel、源码归档、元数据和日志；约 173 GiB 的模型仍留在临时模型目录，不复制到工作区。

#### 1. 一次性复制 vLLM 环境到持久目录

实例重启后，如果 `/root/tmp` 中的构建环境已经消失，先重新完成源码构建；如果构建环境仍存在，执行下面代码复制运行环境。模型目录不参与复制。

```bash
set -euo pipefail

export BUILD_ROOT=/root/tmp/qwen3.8-flash-next-vllm0310
export ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm
export PERSIST_ROOT="$ARTIFACT_ROOT/vllm0310-20261017"
export SOURCE_VENV="$BUILD_ROOT/.venv-vllm-0310-base2"
export PERSIST_VENV="$PERSIST_ROOT/venv-vllm-0310"

test -x "$SOURCE_VENV/bin/python"
mkdir -p "$PERSIST_ROOT/wheels" "$PERSIST_ROOT/source" "$PERSIST_ROOT/meta" "$PERSIST_ROOT/logs"

if [ ! -x "$PERSIST_VENV/bin/python" ]; then
  mkdir -p "$PERSIST_VENV"
  cp -a "$SOURCE_VENV/." "$PERSIST_VENV/"
  for f in "$PERSIST_VENV/bin/activate" \
           "$PERSIST_VENV/bin/activate.csh" \
           "$PERSIST_VENV/bin/activate.fish"; do
    [ -f "$f" ] && sed -i "s#$SOURCE_VENV#$PERSIST_VENV#g" "$f"
  done
fi
```

#### 2. 验证 Python、vLLM 和原生扩展来自持久目录

```bash
set -euo pipefail

export ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm
export PERSIST_ROOT="$ARTIFACT_ROOT/vllm0310-20261017"
export PERSIST_VENV="$PERSIST_ROOT/venv-vllm-0310"

export VLLM_PY="$PERSIST_VENV/bin/python"
test -x "$VLLM_PY"

"$VLLM_PY" - <<'PY'
import sys
import torch
import triton
import vllm

print("python:", sys.executable)
print("sys.prefix:", sys.prefix)
print("vllm:", vllm.__version__)
print("vllm path:", vllm.__file__)
print("torch:", torch.__version__, torch.version.hip)
print("triton:", triton.__version__)

import vllm._C
import vllm._C_stable_libtorch
import vllm._rocm_C
print("vllm native extensions: OK")

assert sys.prefix.startswith("/mnt/workspace/artifacts/")
assert vllm.__file__.startswith(sys.prefix)
PY
```

#### 3. 从持久目录启动 eager 基线

这段代码不依赖 `/root/tmp` 中的 vLLM 源码或虚拟环境，只依赖持久目录中的绝对 Python 路径；模型仍使用临时模型目录。

```bash
set +e
set +u
set +o pipefail

unset BUILD_ROOT SOURCE_VENV
export ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm
export PERSIST_ROOT="$ARTIFACT_ROOT/vllm0310-20261017"
export PERSIST_VENV="$PERSIST_ROOT/venv-vllm-0310"
export VLLM_PY="$PERSIST_VENV/bin/python"
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export LOG="$ARTIFACT_ROOT/logs/vllm0310-persistent-eager.log"

export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ROCM_ARCH=gfx942
export VLLM_ENGINE_READY_TIMEOUT_S=1200

mkdir -p "$ARTIFACT_ROOT/logs"

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next-fp8 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 8 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --enforce-eager \
  2>&1 | tee "$LOG"

server_rc=${PIPESTATUS[0]}
echo "server exit=$server_rc"
```

#### 4. 下一步：持久目录中的非 eager 对照

先复现上面的 eager 基线，再在同一持久环境、同一模型和 `max_model_len=1024` 下，仅移除 `--enforce-eager`，尝试关闭编译：

```bash
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next-fp8 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --dtype auto \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 8 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --compilation-config '{"level":0}' \
  2>&1 | tee "$ARTIFACT_ROOT/logs/vllm0310-persistent-no-eager.log"
```

该步骤只用于定位 Torch Dynamo / ROCm 后端问题，不重新构建或安装环境。失败时保留日志并回到 eager 基线。

## 测评

