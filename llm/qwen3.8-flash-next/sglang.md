# Qwen3.8-Flash-Next / SGLang / FP8

- 硬件：AMD Instinct MI300X 192GB，ROCm 7.2.3，`gfx942`
- 权重：`Qwen/Qwen3.8-Flash-Next-FP8`，复用 vLLM 已下载目录
- 端口：`127.0.0.1:30000`
- 状态：MI300X 上 FP8 服务启动、健康检查与单请求冒烟通过；E1–E4 待测

官方 Qwen 配方要求使用包含该模型支持的 SGLang 构建（至少 `v0.5.20`），并建议使用 `--reasoning-parser auto`；该模型始终保留 reasoning。[官方配方](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx)

## 部署与检查（AMD/ROCm 专用）

不要使用通用的 `uv pip install 'sglang>=0.5.20'`，它会选择 CUDA 依赖并拉取 `nvidia-*`。以下步骤复用已验证的 vLLM ROCm/Torch 环境，复制出独立的 SGLang 环境。先执行第 0 步一次，再在同一个窗口依次运行后续步骤；失败恢复时保留当前 Shell 与变量。

参考：[AMD GPU 安装说明](https://sgl-project.github.io/platforms/amd_gpu.html)、[v0.5.21 HIP 依赖定义](https://github.com/sgl-project/sglang/blob/v0.5.21/python/pyproject_other.toml)、[v0.5.21 ROCm 构建流程](https://github.com/sgl-project/sglang/blob/v0.5.21/docker/rocm.Dockerfile)、[ROCm 7.2 PyTorch wheel 索引](https://download.pytorch.org/whl/rocm7.2/)、[ModelScope 下载命令](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)。

### 0. 公共变量、日志与窗口保活

本段只需执行一次，并且要在可交互的 Bash 窗口运行。后续步骤都在同一窗口执行。公共段先清除旧版 ERR/EXIT trap，并关闭 `errexit`；命令失败或按 `Ctrl+C` 后，Bash 会直接回到提示符，已导出的变量和日志设置仍保留，直接粘贴下一步即可。这里不再设置“按回车关闭窗口”的等待逻辑。

```bash
# 清掉旧版本残留的失败/退出处理器；关闭 errexit，避免失败状态结束当前交互 Shell。
trap - ERR EXIT
set +e
set -uo pipefail
cd /mnt/workspace/multimodal-deploy

export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export VLLM_ROCM_VENV=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm0310-20261017/venv-vllm-0310
export SGLANG_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export SGLANG_SRC=/root/tmp/qwen3.8-flash-next/sglang-v0.5.21
export SGLANG_REF=v0.5.21
export RESULT_DIR=/mnt/workspace/results/llm/qwen3.8-flash-next
export INSTALL_LOG="$RESULT_DIR/sglang-install-rocm723.log"
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ROCM_ARCH=gfx942
export GIT_TERMINAL_PROMPT=0
mkdir -p "$RESULT_DIR"

exec > >(tee -a "$INSTALL_LOG") 2>&1
echo '[OK] 公共变量与日志已就绪；命令失败后会返回当前 Shell 提示符。'
```

### 1. 检查 AMD/ROCm 基础环境

```bash
export VLLM_ROCM_VENV=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm0310-20261017/venv-vllm-0310
echo '[1/6] 检查 vLLM ROCm/Torch 环境和 GPU。'
test -x "$VLLM_ROCM_VENV/bin/python"
"$VLLM_ROCM_VENV/bin/python" -c 'import torch; assert torch.version.hip, torch.__version__; print("base torch", torch.__version__, "hip", torch.version.hip)'
rocm-smi
```

### 2. 创建独立 SGLang 环境

不重新下载 CUDA Torch，直接复制已验证的 ROCm venv；旧的 `.venv-sglang` 不使用、不删除。

```bash
export VLLM_ROCM_VENV=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm0310-20261017/venv-vllm-0310
export SGLANG_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export PYTHON_BIN="$SGLANG_VENV/bin/python"
echo '[2/6] 创建独立 SGLang venv。'
if [ ! -x "$PYTHON_BIN" ]; then
  cp -a "$VLLM_ROCM_VENV" "$SGLANG_VENV"
  for f in "$SGLANG_VENV/bin/activate" "$SGLANG_VENV/bin/activate.csh" "$SGLANG_VENV/bin/activate.fish"; do
    [ -f "$f" ] && sed -i "s#$VLLM_ROCM_VENV#$SGLANG_VENV#g" "$f"
  done
fi
"$PYTHON_BIN" -c 'import torch; print("sglang base torch", torch.__version__, "hip", torch.version.hip); assert torch.version.hip'
```

### 3. 下载并检查 SGLang 源码

```bash
export SGLANG_SRC=/root/tmp/qwen3.8-flash-next/sglang-v0.5.21
export SGLANG_REF=v0.5.21
echo '[3/6] 准备 SGLang v0.5.21 源码。'
if [ -e "$SGLANG_SRC" ] && [ ! -d "$SGLANG_SRC/.git" ]; then
  echo "[ERROR] $SGLANG_SRC 已存在但不是完整 Git checkout；未自动删除。"
else
  if [ ! -d "$SGLANG_SRC/.git" ]; then
    git clone --depth 1 --branch "$SGLANG_REF" https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
  fi
  git -C "$SGLANG_SRC" rev-parse --short HEAD
  test -f "$SGLANG_SRC/python/sglang/kernels/aot/setup_rocm.py"
fi
```

### 4. 编译并无依赖安装 gfx942 ROCm kernel

必须先把 AOT kernel 的 CUDA 元数据切换为 `pyproject_rocm.toml`。否则即使 wheel 用 `--no-deps` 安装，其元数据仍可能要求 `torch==2.13.0`，第 5 步就会拉取 CUDA Torch 及 `nvidia-*`。这里先保存原文件、切换 ROCm 元数据，再生成 wheel 并无依赖安装。

```bash
export SGLANG_SRC=/root/tmp/qwen3.8-flash-next/sglang-v0.5.21
export SGLANG_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export RESULT_DIR=/mnt/workspace/results/llm/qwen3.8-flash-next
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ROCM_ARCH=gfx942
mkdir -p "$RESULT_DIR/sglang-kernel-wheel-rocm723"

echo '[4/6] 编译 SGLang ROCm kernel。'
cd "$SGLANG_SRC/python/sglang/kernels/aot"
[ -f pyproject.cuda.toml ] || cp pyproject.toml pyproject.cuda.toml
cp pyproject_rocm.toml pyproject.toml
"$PYTHON_BIN" -m pip uninstall -y sglang-kernel || true
if AMDGPU_TARGET=gfx942 PYTORCH_ROCM_ARCH=gfx942 "$PYTHON_BIN" setup_rocm.py bdist_wheel --dist-dir "$RESULT_DIR/sglang-kernel-wheel-rocm723"; then
  KERNEL_WHEEL=$(find "$RESULT_DIR/sglang-kernel-wheel-rocm723" -maxdepth 1 -type f -name 'sglang_kernel-*.whl' -print -quit)
  if [ -n "$KERNEL_WHEEL" ] && "$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$KERNEL_WHEEL"; then
    "$PYTHON_BIN" -c 'from importlib.metadata import distribution; reqs=distribution("sglang-kernel").requires or []; print("sglang-kernel requirements:", reqs); assert not any(r.lower().startswith("torch") for r in reqs), reqs'
  else
    echo '[ERROR] ROCm kernel wheel 缺失或安装失败；Shell 保持打开，请检查日志。'
  fi
else
  echo '[ERROR] ROCm kernel 编译失败；Shell 保持打开，请检查日志。'
fi
```

### 5. 切换 HIP 依赖并安装 SGLang

这次日志中的 `modelscope` 是 SGLang `runtime_base` 明确列出的普通依赖，单独下载它是正常的。CUDA 包的解析链则与 `timm` 有关：`runtime_common` 固定 `timm==1.0.16`，而该版本又无版本限制地依赖 `torch` 和 `torchvision`；如果从默认 PyPI 安装 `torchvision`，版本配对会把解析推向 PyPI 的 Torch（此前日志已出现 `torch==2.13.0`），继而下载整套 `nvidia-*` CUDA 运行库。ROCm 环境的 Torch 是 `2.12.0+git6bbd260`，不应被这次依赖解析替换。旧第 4 步的 CUDA kernel 元数据也可能单独要求 `torch==2.13.0`；第 4 步改为 ROCm 元数据 wheel 并 `--no-deps` 后应消除此项。

SGLang v0.5.21 的官方 ROCm Dockerfile 用 pip 约束 Torch 栈后安装 SGLang；AMD 的 ROCm 7.2 wheel 索引提供 Python 3.12 的 `torchvision 0.27.0+rocm7.2`，与 Torch 2.12 配套。先以 `--no-deps` 安装这个 ROCm wheel并检查导入，再把当前 Torch/torchvision/torchaudio 写进约束。若检查失败，停在这里，不要运行 SGLang 安装。本机这份 Torch 是 vLLM ROCm 镜像构建的 `2.12.0+git6bbd260`，不在 PyPI 索引中；此次 uv dry-run 明确报该精确版本无候选，即使环境中已有安装也未通过解析。因此这里改用官方 ROCm Dockerfile 同样使用的 pip。dry-run 若失败、出现 `nvidia-*` 或要替换当前 Torch 栈，也不要继续正式安装。

SGLang 依赖安装成功后，不要用全环境 `pip check` 作为 SGLang 是否可用的判据：复制来的 vLLM 包仍留在 SGLang venv 中，而其 grpcio、openai、outlines-core、timm 约束与 SGLang 的版本不同，会让 `pip check` 报冲突；这些冲突不代表原 vLLM venv 被修改。用 SGLang/Torch/HIP 导入及 GPU 检查验证当前环境。

若已运行过旧第 4/5 步，先核实当前环境和 kernel 元数据；应看到 `torch.version.hip` 有值，且 `sglang-kernel` 不再要求 `torch==2.13.0`：

```bash
"$PYTHON_BIN" - <<'PY'
import torch
from importlib.metadata import PackageNotFoundError, distribution

print("torch:", torch.__version__, "HIP:", torch.version.hip)
try:
    kernel = distribution("sglang-kernel")
    print("sglang-kernel:", kernel.version, "requires:", kernel.requires)
except PackageNotFoundError:
    print("sglang-kernel: not installed")
PY
```

这次安装日志还显示缺少 `cargo`。当前只部署 Qwen3.8 文本模型，设置 `SGLANG_BUILD_RUST_EXTS=none` 跳过 Rust 多模态扩展；以后需要多模态功能时再安装 Rust toolchain 并移除此设置。

```bash
export SGLANG_SRC=/root/tmp/qwen3.8-flash-next/sglang-v0.5.21
export SGLANG_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export RESULT_DIR=/mnt/workspace/results/llm/qwen3.8-flash-next
mkdir -p "$RESULT_DIR"
echo '[5/6] 使用 pyproject_other.toml 安装 SGLang HIP 运行时。'
cd "$SGLANG_SRC"
[ -f python/pyproject.cuda.toml ] || cp python/pyproject.toml python/pyproject.cuda.toml
cp python/pyproject_other.toml python/pyproject.toml
export SGLANG_BUILD_RUST_EXTS=none
uv pip install --python "$PYTHON_BIN" --no-deps --index-url https://download.pytorch.org/whl/rocm7.2 'torchvision==0.27.0+rocm7.2'
if "$PYTHON_BIN" -c 'import torch, torchvision; print("torch", torch.__version__, "HIP", torch.version.hip); print("torchvision", torchvision.__version__); assert torch.version.hip'; then
  # 从已安装 distribution 元数据生成精确版本约束；不依赖 pip freeze 的输出格式。
  "$PYTHON_BIN" - <<'PY' > "$RESULT_DIR/sglang-rocm-constraints.txt"
from importlib.metadata import PackageNotFoundError, version

for package in ("torch", "torchvision"):
    print(f"{package}=={version(package)}")
for package in ("torchaudio",):
    try:
        print(f"{package}=={version(package)}")
    except PackageNotFoundError:
        pass
PY
  if grep -q '^torch==' "$RESULT_DIR/sglang-rocm-constraints.txt" && grep -q '^torchvision==' "$RESULT_DIR/sglang-rocm-constraints.txt"; then
    cat "$RESULT_DIR/sglang-rocm-constraints.txt"
    if "$PYTHON_BIN" -m pip install --dry-run --constraint "$RESULT_DIR/sglang-rocm-constraints.txt" --pre -e "python[srt_hip]" 2>&1 | tee "$RESULT_DIR/sglang-hip-install-plan.txt"; then
      if grep -Eiq 'nvidia-' "$RESULT_DIR/sglang-hip-install-plan.txt"; then
        echo '[ERROR] 计划中仍有 CUDA/NVIDIA runtime；已中止安装，请保留 plan 日志。'
      elif "$PYTHON_BIN" -m pip install --constraint "$RESULT_DIR/sglang-rocm-constraints.txt" --pre -e "python[srt_hip]"; then
        "$PYTHON_BIN" -c 'import sglang, torch, torchvision; print("sglang", getattr(sglang, "__version__", "unknown")); print("torch", torch.__version__, "HIP", torch.version.hip); print("torchvision", torchvision.__version__); print("cuda_available", torch.cuda.is_available()); assert torch.version.hip and torch.cuda.is_available()'
      else
        echo '[ERROR] SGLang 安装失败；Shell 保持打开，请查看上方输出和安装日志。'
      fi
    else
      echo '[ERROR] 依赖 dry-run 失败；未启动正式安装，Shell 保持打开，请查看计划日志。'
    fi
  else
    echo '[ERROR] Torch/torchvision 约束文件不完整；未继续安装。'
  fi
else
  echo '[ERROR] ROCm Torch/torchvision 导入检查失败；未继续安装。'
fi
```

### 6. 最终导入与设备检查

```bash
export SGLANG_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export PYTHON_BIN="$SGLANG_VENV/bin/python"
echo '[6/6] 检查 SGLang、Torch、HIP 和 GPU。'
"$PYTHON_BIN" -c 'import sglang, torch; print("sglang", getattr(sglang, "__version__", "unknown")); print("torch", torch.__version__, "hip", torch.version.hip); print("cuda_available", torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); assert torch.version.hip and torch.cuda.is_available()'
```

## 启动服务

```bash
export SGLANG_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export RESULT_DIR=/mnt/workspace/results/llm/qwen3.8-flash-next
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ROCM_ARCH=gfx942

"$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next-fp8-sglang \
  --host 127.0.0.1 --port 30000 \
  --tp-size 1 \
  --context-length 8192 \
  --max-running-requests 1 \
  --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --reasoning-parser auto \
  2>&1 | tee "$RESULT_DIR/qwen3.8-flash-next-sglang-fp8-server.log"
```

旧参数加载 131/131 个 FP8 shard 后，因 `max_mamba_cache_size=4`、每请求需要 5 个 state slots（`4 // 5 = 0`）而退出。上面的 BF16 单并发命令已在 MI300X 启动成功；它改变了 recurrent state 精度，后续需用 E4 验证质量，再扩大并发。

另开终端做健康检查与冒烟：

```bash
export RESULT_DIR=/mnt/workspace/results/llm/qwen3.8-flash-next
mkdir -p "$RESULT_DIR"
curl -fsS http://127.0.0.1:30000/health | tee "$RESULT_DIR/sglang-health.json"
curl -fsS http://127.0.0.1:30000/v1/models | tee "$RESULT_DIR/sglang-models.json"
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp8-sglang","messages":[{"role":"user","content":"用一句话说明你是什么模型。"}],"max_tokens":256,"temperature":0}' \
  | tee "$RESULT_DIR/sglang-smoke.json"
```

## E1–E4 测试

```bash
export SGLANG_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export PYTHON_BIN="$SGLANG_VENV/bin/python"
cd /mnt/workspace/multimodal-deploy

# E1：单请求，512/4096/8192
"$PYTHON_BIN" -u benchmarks/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:30000/v1/chat/completions \
  --model qwen3.8-flash-next-fp8-sglang \
  --label qwen3.8-flash-next-sglang-fp8-e1 \
  --contexts 512,4096,8192 --concurrencies 1 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e1.json

# E2：两并发至少需要 10 个 state slots（每请求 5 个）；只有日志确认 cache 容量足够时，才以 --max-running-requests 2 重启并测 1/2。
# 不要直接按 8 并发重启；8 并发至少需要 40 个 state slots，且还要有足够 KV 显存。
"$PYTHON_BIN" -u benchmarks/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:30000/v1/chat/completions \
  --model qwen3.8-flash-next-fp8-sglang \
  --label qwen3.8-flash-next-sglang-fp8-e2 \
  --contexts 512,4096,8192 --concurrencies 1,2 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e2.json

# E3：先停服，保留 BF16 与单并发，仅将 --context-length 提到 65536；若 KV/state cache 不足，降低上下文后再测。
"$PYTHON_BIN" -u benchmarks/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:30000/v1/chat/completions \
  --model qwen3.8-flash-next-fp8-sglang \
  --label qwen3.8-flash-next-sglang-fp8-e3 \
  --contexts 8192,32768,65536 --concurrencies 1 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e3.json

# E4：Mini-8 质量抽样，保留 reasoning_content、content、finish_reason
"$PYTHON_BIN" -u benchmarks/openai_quality_probe.py \
  --questions benchmarks/quality/mtbench-mini-8.json \
  --endpoint http://127.0.0.1:30000/v1/chat/completions \
  --model qwen3.8-flash-next-fp8-sglang \
  --label qwen3.8-flash-next-sglang-fp8-e4-quality-mini8-max4096 \
  --max-tokens 4096 --temperature 0 --timeout 600 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e4-quality.json
```

## 结果

| 测试 | 结果 |
|---|---|
| SGLang HIP 依赖与设备检查 | ✅ SGLang 0.5.21 导入成功；Torch 2.12.0+git6bbd260 / HIP 7.2.53211，`cuda_available=True`。粘贴输出未见 GPU 名称行；全环境 `pip check` 的冲突来自 SGLang venv 遗留的 vLLM |
| 旧参数启动 | ❌ FP8 权重 131/131 加载成功；Mamba state cache 4 slots、每请求需 5，有效请求数 0 |
| BF16 单并发启动 | ✅ 131/131 shard，加载 59.01 秒；权重占用 170.55 GB，随后 Mamba cache 10 slots（SSM 0.58 GB）、KV cache 31488 tokens；CUDA graph 后可用显存 18.58 GB，`cpu_offload_gb=0` |
| `/health`、`/v1/models` 与冒烟 | ✅ 健康检查无报错；模型 ID 正确、`max_model_len=8192`；聊天请求 HTTP 200，返回一句话文本与非空 `reasoning_content`，`finish_reason=stop`（59 prompt / 67 completion tokens） |
| E1 单请求 | ⏳ 待执行，结果写入 `...-e1.json` |
| E2 并发 | ⏳ 待执行，结果写入 `...-e2.json` |
| E3 长上下文 | ⏳ 待执行，结果写入 `...-e3.json` |
| E4 Mini-8 | ⏳ 待执行，结果写入 `...-e4-quality.json` |

以上启动与冒烟数据来自 AMD 实例的输出；原始日志与 JSON 在远端 `$RESULT_DIR`。E1–E4 完成后再回填成功率、吞吐、TTFT、P50/P95 和显存峰值。

日志中的 `rope_parameters` 未识别警告及可选 `cutlass`/`torchcodec` 导入提示未阻止文本冒烟；缺少 `torchcodec` 会影响音频输入。E4 仍需确认输出质量。
