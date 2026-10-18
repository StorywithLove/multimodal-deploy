# Qwen3.8-Flash-Next / vLLM / FP8

- 后端：vLLM
- 权重：[Qwen/Qwen3.8-Flash-Next-FP8（Hugging Face）](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- 镜像：[Qwen/Qwen3.8-Flash-Next-FP8（ModelScope）](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next-FP8)
- 格式：Hugging Face Safetensors，细粒度 FP8，block size 128
- 硬件：AMD MI300X 192GB，ROCm 7.2.3，gfx942
- 模型目录：`/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8`
- AMD 实例项目目录：`/mnt/workspace/multimodal-deploy`；凡是在项目目录中执行的 Python、基准或探针命令，先执行 `cd /mnt/workspace/multimodal-deploy`
- 持久目录：`/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm0310-20261017/`
- 当前结果：[vLLM 启动汇总](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-startup.json)；[O1/O2 E1 汇总](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-e1.json)；[E2 原始结果](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-o2-e2-context512-4096-8192-concurrency1-2-4-8-raw.json)；[E3 合并结果](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-o2-e3-context8192-32768-65536-concurrency1-4-raw.json)；[probe 汇总](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-probe.json)。

## 模型下载

模型文件约 173 GiB，下载到临时模型目录；不要复制到 vLLM 持久目录。当前实例优先使用 ModelScope：

```bash
export MODEL_ROOT=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$MODEL_ROOT/Qwen3.8-Flash-Next-FP8"

mkdir -p "$MODEL_ROOT"

modelscope download \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --local_dir "$MODEL_DIR"
```

如果 ModelScope 链路不可用，再使用 Hugging Face 下载到同一个目录：

```bash
export MODEL_ROOT=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$MODEL_ROOT/Qwen3.8-Flash-Next-FP8"

mkdir -p "$MODEL_ROOT"

hf download Qwen/Qwen3.8-Flash-Next-FP8 \
  --local-dir "$MODEL_DIR"
```

## 部署

### 部署历程（目标 → 尝试 → 结果）

部署基线是 Ubuntu 22.04 + ROCm 7.2.3（HIP 7.2.53211）+ glibc 2.35，GPU 架构为 `gfx942`。下面按实际因果顺序记录，避免把“尝试过的路线”和“最终采用的路线”混在一起：

1. **目标：在官方支持的 ROCm 运行环境中部署 vLLM。** 尝试按照官方指导使用 ROCm/vLLM Docker 镜像；结果是当前实例只有 Docker CLI，没有可连接的 Docker daemon，容器无法启动，因此不能依赖该镜像在容器内构建或运行 vLLM wheel。
2. **目标：直接使用官方预编译 ROCm wheel。** 尝试在宿主机安装可用的 `vllm==0.30.0+rocm723` / `vllm==0.31.0+rocm723` wheel；结果是这些 wheel 的平台标签要求 `manylinux_2_39_x86_64`（glibc ≥ 2.39），而系统是 glibc 2.35，无法安装。这不是把“ROCm 7.2.3 + glibc 2.39”迁移成“ROCm 7.2.3 + glibc 2.35”，而是原有 ROCm 7.2.3 / glibc 2.35 环境与预编译 wheel 的兼容性不满足。
3. **目标：在当前实例上获得可用的 vLLM 安装。** 尝试沿用当前 ROCm 7.2.3、`gfx942` 环境，从源码构建 vLLM 0.30.0 并生成 wheel；结果是原生扩展可以导入，但 EngineCore 在 600 秒内没有 ready，HTTP 端口未监听，启动超时并出现 `Timed out waiting for engine core processes to start`。
4. **目标：解决 0.30.0 的启动超时并完成模型服务启动。** 升级并重新编译 vLLM 0.31.0 后，eager 基线可以启动；默认 compiled 路径最初在 Torch Dynamo 的 PLE/GDN 动态形状处理中失败。后续通过固定 PLE 输入 buffer 形状、移除 `query_start_loc` / `ngram_context` 的动态维声明、并将 GDN reshape 的 `-1` 改为显式 head 数，O1 与 O2 compiled 服务均启动并通过健康检查和请求测试。

因此，当前应将部署结果理解为：**vLLM 0.31.0 + ROCm 7.2.3 的源码构建和补丁后的 O1/O2 compiled 部署均已验证；eager 仅保留为早期功能回退基线，当前性能测评使用 O2。**

### 其他已知问题

- **非阻断警告**：环境曾出现可选后端/扩展及依赖检查 warning；补丁后的 O1/O2 compiled 服务仍能启动和处理请求。保留 warning 供复现时核对，不把它们与 Dynamo 动态形状修复混为一谈。

### 部署成功的具体要点

- 使用 Python 3.12.13、Torch `2.12.0+git6bbd260`、HIP `7.2.53211`、Triton `3.7.1`。
- 使用 vLLM `0.31.0+rocm723` 源码构建；Qwen3.8-Flash-Next 的 131 个 Safetensors 分片能够完成加载，FP8 权重实际使用 `torch.float8_e4m3fn`。
- eager 与补丁后的 O1/O2 compiled 路径均完成启动；O1/O2 未使用 `--enforce-eager`，`/health` 返回 200，`/v1/models` 可查询，聊天请求及 E1–E3 已完成的请求矩阵成功。
- eager 是早期故障隔离/功能回退基线；当前性能主线使用 O2。O1 留作已验证的编译档位对照。
- 最近一次关键配置修复：从 Qwen4Exp AMD `support_torch_compile` 的 `dynamic_arg_dims` 移除 `query_start_loc` 和 `ngram_context`，令固定容量的 PLE 输入不再被错误声明为动态维。
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

cd /mnt/workspace/multimodal-deploy

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

#### 3. 从持久目录启动 eager 功能回退（历史路径）

这段代码不依赖 `/root/tmp` 中的 vLLM 源码或虚拟环境，只依赖持久目录中的绝对 Python 路径；模型仍使用临时模型目录。

```bash
set +e
set +u
set +o pipefail

cd /mnt/workspace/multimodal-deploy

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

服务启动后，在另一个终端执行下面的验证命令。`health_http=200`、能够返回模型列表，并且聊天请求返回 `finish_reason=stop`，才算本轮启动冒烟测试通过：

```bash
curl -sS -o /dev/null \
  -w 'health_http=%{http_code}\n' \
  http://127.0.0.1:8000/health

curl -sS http://127.0.0.1:8000/v1/models
echo

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-fp8",
    "messages": [
      {"role": "user", "content": "请用一句话说明什么是量化模型。"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
echo
```

#### 4. O1 compiled 启动

O1 已在持久 venv 中通过启动和请求验证，未启用 `--enforce-eager`。原始启动结果：[O1 E1 单请求基线](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-e1.json)；O1 实际解析配置曾由启动日志核实为 Dynamo/Inductor、`PIECEWISE` graph、`dynamic_shapes_config=BACKED`。O1 启动参数沿用 eager 命令的模型、ROCm 环境、端口、dtype、显存和 offload 设置，只使用 compiled 档位：

```bash
cd /mnt/workspace/multimodal-deploy

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next-fp8 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --dtype auto \
  --gpu-memory-utilization 0.92 --cpu-offload-gb 8 \
  --max-model-len 1024 --max-num-seqs 2 \
  --optimization-level 1
```

#### 5. O2 compiled 启动（当前性能测试配置）

当前性能测评使用 O2，服务从持久 venv 启动，模型权重仍从临时目录读取；不加 `--enforce-eager`。E1 的长上下文配置是 `max_model_len=8192, max_num_seqs=2`；E2 是 `8192, 8`；E3 是 `65536, 4`。以下示例为 E2：

```bash
cd /mnt/workspace/multimodal-deploy

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next-fp8 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --dtype auto \
  --gpu-memory-utilization 0.92 --cpu-offload-gb 8 \
  --max-model-len 8192 --max-num-seqs 8 \
  --optimization-level 2
```

O1、O2 均为编译服务；`--compilation-config '{"mode":0}'` 是早期诊断配置，不等同于 O1/O2 或严格 eager。vLLM 从持久目录启动已验证，但模型仍位于 `/root/tmp`，实例重建后若该临时目录丢失需重新下载模型。性能结果及启动耗时说明见下方“测评”和附录。

## 测评

### 口径

- E1：单并发，每个矩阵单元 8 个请求、1 个 warmup，每请求最多生成 256 tokens。
- E2：目标上下文 512/4096/8192，concurrency 1/2/4/8；每单元 8 个请求、1 个 warmup。
- E3：目标上下文 8192/32768/65536，concurrency 1；每单元 8 个请求、1 个 warmup。
- 目标上下文不是实际 prompt 长度；实际输入 tokens 以 JSON 的 `actual_tokens` 为准。
- `prompt_tokens_per_s` 为矩阵单元实际 prompt token 总数除以 wall time，不是独立 prefill kernel 吞吐；`completion_tokens_per_s` 为 completion token 总数除以 wall time。
- 当前结果为单轮数据，适合阶段性比较，不代表多轮统计或置信区间。显存 `delta_gib=0` 仅表示采样期未观察到相对基线增量，不代表没有 KV cache 占用。

### E1：O1/O2 对照及 O2 长上下文

结果：[E1 汇总 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-e1.json)。各单元 8/8 成功。

| 配置 / 目标上下文 | 实际 prompt tokens | prompt tok/s | completion tok/s | TTFT (s) | latency p50 (s) | p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| O1 / 512 | 433 | 48.06 | 28.42 | 0.324 | 8.956 | 9.015 |
| O2 / 512（初始） | 433 | 61.00 | 36.07 | 0.310 | 7.043 | 7.069 |
| O2 / 512（扩展矩阵） | 433 | 60.84 | 35.97 | 0.311 | 7.061 | 7.089 |
| O2 / 4096 | 2801 | 377.66 | 34.52 | 0.548 | 7.331 | 7.355 |
| O2 / 8192 | 5507 | 757.57 | 35.22 | 0.417 | 7.203 | 7.222 |

目标 8192 单元实际输入为 5507 tokens。O1/O2 的单轮数据只作为观测，不据此推导稳定性能差异。

### E2：O2 并发扩展

[原始 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-o2-e2-context512-4096-8192-concurrency1-2-4-8-raw.json)。12 个矩阵单元、96/96 请求成功。单元格格式为 completion tok/s / 请求 latency p50 秒：

| 目标上下文（实际 prompt tokens） | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| 512（433） | 35.91 / 7.07 | 55.44 / 8.92 | 87.33 / 11.51 | 116.77 / 17.13 |
| 4096（2801） | 34.50 / 7.34 | 53.66 / 9.40 | 79.07 / 12.62 | 101.77 / 19.48 |
| 8192（5507） | 35.20 / 7.21 | 55.46 / 9.12 | 82.25 / 12.17 | 108.58 / 18.36 |

并发增加提升聚合吞吐，同时提高单请求延迟。与 llama.cpp（UD-Q4_K_XL）数据为异精度、异框架观测，不是纯框架对比。

### E3：O2 长上下文单并发与 C4 补充

[合并 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-o2-e3-context8192-32768-65536-concurrency1-4-raw.json)包含 3 个 C1 单元及 1 个 C4 补充单元，共 32/32 请求成功；[C4 原始单元](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-o2-e3-context32768-concurrency4-raw.json)单独保留用于追溯。

| 目标上下文（实际 prompt tokens） | prompt tok/s | completion tok/s | TTFT (s) | latency p50 (s) | p95 (s) |
|---|---:|---:|---:|---:|---:|
| 8192（5507） | 729.52 | 33.91 | 0.457 | 7.48 | 7.50 |
| 32768（21740） | 2762.22 | 32.53 | 0.734 | 7.77 | 7.78 |
| 65536（43385） | 5587.96 | 32.97 | 0.625 | 7.67 | 7.70 |

实际最长 prompt 为 43385 tokens，不代表已测到 65536 个实际输入 tokens。E2/E3 启动日志显示进程启动至 `Application startup complete` 约 346 秒、335 秒；属于日志时间差近似值，不是独立 health probe 的精确计时。

E3 补充单元（目标上下文 32768、并发 4，实际 prompt 21740 tokens）：8/8 成功，prompt 吞吐 6138.96 tok/s，completion 吞吐 72.29 tok/s，TTFT 0.730 s，latency p50/p95 为 13.735/14.219 s。

### 当前进度

E1、E2、E3 上述单元已完成；E3 的 32768/C4 补充单元也已归档。vLLM E4 尚未完成。

## 附录：Torch Dynamo 的 `query_start_loc.size()[0]` 动态维约束

### 问题定位（O1 早期排查）

O1 初次编译时，`query_start_loc.size()[0]` 被标记为动态维，却在 tracing 中被特化为常量 2，触发 `ConstraintViolationError`；首个 user stack 指向 AMD PLE 容量检查。仅将 `padded_buffer` 首维扩为至少 2 不足以修复完整编译。CPU 探针发现 ngram 与 packed buffer 行数不一致时仍会失败；`unbacked` 路径还暴露 GDN 中 `z.reshape(z.size(0), -1, self.head_v_dim)` 的 `Ne(128*u0, 0)` 数据依赖 guard。以上均为定位阶段的中间结果，已被最终补丁与成功复测 supersede。

### 最终补丁组合

补丁脚本留在仓库 `llm/qwen3.8-flash-next/patches/`，实际修改应用于持久 venv 的安装源码：

1. [PLE 固定 buffer 补丁](patches/apply_qwen4exp_amd_fixed_ple_buffers.py)：常规与 dummy 路径传递全尺寸固定 `query_start_loc` / `ngram_context` buffer，并填充 padding 尾部，避免输入 shape 随请求数变化。
2. [PLE 静态维声明补丁](patches/apply_qwen4exp_amd_static_ple_dims.py)：从 `Qwen4ExpModel` 的 `dynamic_arg_dims` 移除 `query_start_loc` 与 `ngram_context`；其余动态维声明保留。
3. [GDN 显式 head reshape 补丁](patches/apply_gdn_forward_cuda_explicit_heads.py)：仅在 `QwenGatedDeltaNetAttention.forward_cuda` 中，将 reshape 的 `-1` 改成 `self.num_v_heads // self.tp_size`。CPU Dynamo 探针中，原式失败，显式 head 数在 token 数 3、5 时均通过。

仅扩大 `padded_buffer` 的实验补丁 [vllm0310-amd-ple-min-requests-2.patch](patches/vllm0310-amd-ple-min-requests-2.patch) 单独应用后仍失败，不是最终解法。`mode=0` 是关闭 Torch Dynamo、仍保留部分 decode graph 的诊断路径，不等同 O1/O2 或 `--enforce-eager`。

### 最终验证与边界

最终补丁组合后，O1 与 O2 均在未启用 `--enforce-eager` 的 compiled 路径启动；health、模型列表、chat 请求通过。O1 日志确认 Dynamo/Inductor、`PIECEWISE`、`BACKED`；O2 用于后续 E1–E3 性能矩阵。结论适用于本次记录的 Python 3.12.13、Torch `2.12.0+git6bbd260`、ROCm/HIP 7.2.3、vLLM `0.31.0+rocm723`、Triton 3.7.1 环境。补丁尚不能视为已合入上游；更新或重建 venv 后需重新检查源码并应用。

