# Qwen3.8-Flash-Next / SGLang / FP8

- 硬件：AMD Instinct MI300X 192GB，ROCm 7.2.3，`gfx942`
- 权重：`Qwen/Qwen3.8-Flash-Next-FP8`，复用 vLLM 已下载目录
- 端口：`127.0.0.1:30000`
- 状态：MI300X 上 FP8 服务启动、健康检查与单请求冒烟通过；E1、E2 已完成；E3 的 Triton 8192/32768 单并发测评完成，65536 本轮未测（当前 KV pool 容量不足）；E4 Mini-8 与 3 轮启动就绪时间均已完成；SGLang venv 与源码已迁移至 artifacts，并从持久路径启动、完成 health/chat 冒烟；模型权重仍在 `/root/tmp`

## 模型下载

SGLang 与 vLLM 使用同一份 `Qwen3.8-Flash-Next-FP8` 权重，模型目录为 `/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8`。下载命令、ModelScope 首选方式及 Hugging Face 备用方式见 [vLLM 文档的模型下载章节](https://github.com/StorywithLove/multimodal-deploy/blob/main/llm/qwen3.8-flash-next/vllm.md)。模型文件约 173 GiB；若实例重启后临时目录丢失，按链接中的步骤重新下载。

官方 Qwen 配方要求使用包含该模型支持的 SGLang 构建（至少 `v0.5.20`），并建议使用 `--reasoning-parser auto`；该模型始终保留 reasoning。[官方配方](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next.mdx)

## 部署与检查（AMD/ROCm 专用）

部署摘要：

- **依赖坑点：**不要用通用 `uv pip install 'sglang>=0.5.20'`；默认依赖解析可能安装 CUDA Torch/`nvidia-*`。从已验证的 ROCm vLLM 环境复制 Torch，并用 HIP 依赖定义、ROCm `torchvision` 与 Torch 约束安装 SGLang。
- **kernel 坑点：**AOT kernel 默认元数据曾要求 `torch==2.13.0`，会替换 ROCm Torch；必须切换 `pyproject_rocm.toml`、为 `gfx942` 编译，并 `--no-deps` 安装。
- **构建坑点：**editable 安装曾因缺少 `cargo` 失败；当前仅测文本模型，设置 `SGLANG_BUILD_RUST_EXTS=none`。
- **检查坑点：**复制的 vLLM 包会让全环境 `pip check` 报依赖冲突；以 SGLang/Torch/HIP 导入、设备检查、`/health`、`/v1/models` 和 chat 冒烟为准。
- **推理坑点：**Mamba state 用 BF16；旧 dtype 只分配 4 个 slots、少于每请求所需的 5 个；BF16 配置有 10 slots。32768 下 AITER 会折减静态显存预算并导致缓存池初始化失败，显式使用 Triton 后启动成功，KV pool 为 31488 tokens。
- **成功基线：**SGLang 0.5.21、Python 3.12.13、Torch 2.12.0+git6bbd260、HIP 7.2.53211、`gfx942`；E1–E4 serving/质量结果及 E4 三轮启动就绪时间均已完成。

结果与日志统一写入项目目录：`/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next`。venv 与 SGLang 源码放在持久目录 `/mnt/workspace/artifacts/qwen3.8-flash-next-sglang`；模型权重仍在 `/root/tmp`，本次环境迁移不包含权重。安装细节见下方各步。部署时先执行第 0 步一次，再在同一个窗口依次运行后续步骤；失败恢复时保留当前 Shell 与变量。

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
export SGLANG_ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang
export SGLANG_VENV="$SGLANG_ARTIFACT_ROOT/venv-sglang-rocm723"
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export SGLANG_SRC="$SGLANG_ARTIFACT_ROOT/source/sglang-v0.5.21"
export SGLANG_REF=v0.5.21
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
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
export SGLANG_ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang
export SGLANG_VENV="$SGLANG_ARTIFACT_ROOT/venv-sglang-rocm723"
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
export SGLANG_SRC="$SGLANG_ARTIFACT_ROOT/source/sglang-v0.5.21"
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
export SGLANG_ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang
export SGLANG_SRC="$SGLANG_ARTIFACT_ROOT/source/sglang-v0.5.21"
export SGLANG_VENV="$SGLANG_ARTIFACT_ROOT/venv-sglang-rocm723"
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
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
export SGLANG_ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang
export SGLANG_SRC="$SGLANG_ARTIFACT_ROOT/source/sglang-v0.5.21"
export SGLANG_VENV="$SGLANG_ARTIFACT_ROOT/venv-sglang-rocm723"
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
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
export SGLANG_ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang
export SGLANG_VENV="$SGLANG_ARTIFACT_ROOT/venv-sglang-rocm723"
export PYTHON_BIN="$SGLANG_VENV/bin/python"
echo '[6/6] 检查 SGLang、Torch、HIP 和 GPU。'
"$PYTHON_BIN" -c 'import sglang, torch; print("sglang", getattr(sglang, "__version__", "unknown")); print("torch", torch.__version__, "hip", torch.version.hip); print("cuda_available", torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); assert torch.version.hip and torch.cuda.is_available()'
```

### 将当前临时环境迁移到持久目录（已安装环境执行一次）

持久化环境是有必要的：实例重启后可避免重新装包和编译。理想做法是在持久目录创建 venv；当前环境已装好，下面提供一次性的**快照复制**，同时复制 editable 安装所指向的源码，并重新绑定 SGLang 源码路径。Python 官方不保证 venv 可移动，复制后必须通过路径、Torch/HIP、SGLang 导入检查；这不保证所有旧 console-script 入口都可移植，所以部署统一用 `$PYTHON_BIN -m ...`，且验证成功前保留 `/root/tmp` 原件。若检查失败，回到第 0–6 步在 artifacts 路径重建环境。此操作仅迁移环境与源码，不迁移模型权重。[Python 3.12 venv 可移植性说明](https://docs.python.org/3.12/library/venv.html)。

```bash
export SOURCE_VENV=/root/tmp/qwen3.8-flash-next/.venv-sglang-rocm723
export SOURCE_SRC=/root/tmp/qwen3.8-flash-next/sglang-v0.5.21
export SGLANG_ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang
export SGLANG_VENV="$SGLANG_ARTIFACT_ROOT/venv-sglang-rocm723"
export SGLANG_SRC="$SGLANG_ARTIFACT_ROOT/source/sglang-v0.5.21"
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
mkdir -p "$RESULT_DIR" "$SGLANG_ARTIFACT_ROOT/source"
df -h "$SGLANG_ARTIFACT_ROOT"

if [ ! -x "$SOURCE_VENV/bin/python" ] || [ ! -d "$SOURCE_SRC/.git" ]; then
  echo '[ERROR] /root/tmp 中的已验证 venv 或源码不完整；没有复制。'
elif { [ -e "$SGLANG_VENV" ] && [ ! -x "$PYTHON_BIN" ]; } || { [ -e "$SGLANG_SRC" ] && [ ! -d "$SGLANG_SRC/.git" ]; }; then
  echo '[ERROR] 持久目标存在但不完整；没有覆盖，请先检查目标内容。'
else
  SOURCE_BYTES=$(du -sb "$SOURCE_VENV" "$SOURCE_SRC" | awk '{total += $1} END {printf "%.0f\n", total}')
  AVAILABLE_BYTES=$(df -PB1 "$SGLANG_ARTIFACT_ROOT" | awk 'NR == 2 {printf "%.0f\n", $4}')
  if ! [[ "$SOURCE_BYTES" =~ ^[0-9]+$ && "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] 字节数不是十进制整数（source=$SOURCE_BYTES, available=$AVAILABLE_BYTES）；未开始复制。"
  else
    REQUIRED_BYTES=$((SOURCE_BYTES + SOURCE_BYTES / 10))
    echo "source bytes=$SOURCE_BYTES; available bytes=$AVAILABLE_BYTES; required with 10% margin=$REQUIRED_BYTES"
    if [ "$AVAILABLE_BYTES" -lt "$REQUIRED_BYTES" ]; then
      echo '[ERROR] artifacts 可用空间不足（需留 10% 余量）；未开始复制。'
    else
      if [ ! -x "$PYTHON_BIN" ]; then
        mkdir -p "$SGLANG_VENV"
        cp -a "$SOURCE_VENV/." "$SGLANG_VENV/"
        for f in "$SGLANG_VENV/bin/activate" "$SGLANG_VENV/bin/activate.csh" "$SGLANG_VENV/bin/activate.fish"; do
          [ -f "$f" ] && sed -i "s#$SOURCE_VENV#$SGLANG_VENV#g" "$f"
        done
      fi
      if [ ! -d "$SGLANG_SRC/.git" ]; then
        cp -a "$SOURCE_SRC" "$SGLANG_SRC"
      fi
      export SGLANG_BUILD_RUST_EXTS=none
      if [ -x "$PYTHON_BIN" ] && [ -d "$SGLANG_SRC/.git" ] && \
         "$PYTHON_BIN" -m pip install --no-deps -e "$SGLANG_SRC/python"; then
        if "$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path
import sglang
import torch

venv = Path(os.environ["SGLANG_VENV"]).resolve()
source = Path(os.environ["SGLANG_SRC"]).resolve()
print("python:", sys.executable)
print("sys.prefix:", sys.prefix)
print("sglang:", getattr(sglang, "__version__", "unknown"), sglang.__file__)
print("torch:", torch.__version__, "HIP:", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
assert Path(sys.prefix).resolve() == venv
assert Path(sglang.__file__).resolve().is_relative_to(source)
assert torch.version.hip and torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
PY
        then
          "$PYTHON_BIN" -m pip freeze > "$RESULT_DIR/sglang-pip-freeze-20261019.txt"
          echo '[OK] 持久环境与 editable 源码路径验证通过；旧 /root/tmp 副本仍保留。'
        else
          echo '[ERROR] 持久环境检查失败；不要删除 /root/tmp 副本。'
        fi
      else
        echo '[ERROR] 持久目录复制或 editable 重装失败；保留 /root/tmp 副本并检查日志。'
      fi
    fi
  fi
fi
```

持久化复核已通过：使用 artifacts 下的 venv 与 editable 源码启动服务，并完成 `/health`、`/v1/models` 与 chat 冒烟。模型权重仍位于 `/root/tmp`，不在本次环境/源码迁移范围内；确认权重可持久保存或可重新获取前，不要清理权重目录。后续启动/测评均使用 `$SGLANG_VENV/bin/python` 和 `$SGLANG_SRC` 的 artifacts 路径；全新重建则从第 0–6 步开始，步骤已直接指向持久目录。

## 启动服务

```bash
export SGLANG_ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang
export SGLANG_VENV="$SGLANG_ARTIFACT_ROOT/venv-sglang-rocm723"
export PYTHON_BIN="$SGLANG_VENV/bin/python"
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
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
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
mkdir -p "$RESULT_DIR"
curl -fsS http://127.0.0.1:30000/health | tee "$RESULT_DIR/sglang-health.json"
curl -fsS http://127.0.0.1:30000/v1/models | tee "$RESULT_DIR/sglang-models.json"
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp8-sglang","messages":[{"role":"user","content":"用一句话说明你是什么模型。"}],"max_tokens":256,"temperature":0}' \
  | tee "$RESULT_DIR/sglang-smoke.json"
```

## 部署与冒烟结果

| 测试 | 结果 |
|---|---|
| SGLang HIP 依赖与设备检查 | ✅ SGLang 0.5.21 导入成功；Torch 2.12.0+git6bbd260 / HIP 7.2.53211，`cuda_available=True`。粘贴输出未见 GPU 名称行；全环境 `pip check` 的冲突来自 SGLang venv 遗留的 vLLM |
| 旧参数启动 | ❌ FP8 权重 131/131 加载成功；Mamba state cache 4 slots、每请求需 5，有效请求数 0 |
| BF16 单并发启动 | ✅ 131/131 shard，加载 59.01 秒；权重占用 170.55 GB，随后 Mamba cache 10 slots（SSM 0.58 GB）、KV cache 31488 tokens；CUDA graph 后可用显存 18.58 GB，`cpu_offload_gb=0` |
| `/health`、`/v1/models` 与冒烟 | ✅ 健康检查无报错；模型 ID 正确、`max_model_len=8192`；聊天请求 HTTP 200，返回一句话文本与非空 `reasoning_content`，`finish_reason=stop`（59 prompt / 67 completion tokens） |
| artifacts 持久环境启动复核（2026-10-19） | ✅ 日志确认 SGLang 源码与 venv 均从 `/mnt/workspace/artifacts/qwen3.8-flash-next-sglang` 加载；131/131 shard，Mamba 10 slots、KV cache 31488 tokens；`/v1/models` 返回模型 ID 与 `max_model_len=8192`，chat HTTP 200（59 prompt / 69 completion tokens），有 `reasoning_content` 且 `finish_reason=stop`。本次模型权重仍从 `/root/tmp` 加载。 |

以上启动与冒烟数据来自 AMD 实例的输出；原始日志与 JSON 在远端 `$RESULT_DIR`。日志中的 `rope_parameters` 未识别警告及可选 `cutlass`/`torchcodec` 导入提示未阻止文本冒烟；缺少 `torchcodec` 会影响音频输入。

## E1：单请求基线（已完成）

沿用上文已验证的 8192 上下文、BF16 Mamba state、单并发服务。结果文件：`results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e1.json`。

```bash
export PYTHON_BIN=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang/venv-sglang-rocm723/bin/python
cd /mnt/workspace/multimodal-deploy
"$PYTHON_BIN" -u configs/benchmark/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:30000/v1/chat/completions \
  --model qwen3.8-flash-next-fp8-sglang \
  --label qwen3.8-flash-next-sglang-fp8-e1 \
  --contexts 512,4096,8192 --concurrencies 1 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e1.json
```

| 目标上下文 | 实际输入 tokens/请求 | 完成/请求 | 输出 tok/s | TTFT s | 延迟 P50/P95 s | 已用显存峰值 GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 433 | 8/8 | 37.10 | 1.00 | 6.76 / 6.79 | 173.63 |
| 4096 | 2801 | 8/8 | 35.18 | 0.19 | 7.24 / 7.27 | 174.08 |
| 8192 | 5507 | 8/8 | 34.23 | 1.84 | 7.23 / 7.27 | 174.44 |

三组共 24/24 请求成功、HTTP 200；每请求均生成 256 tokens（本次 `max_tokens=256`）。表中上下文为生成提示词时的**目标值**，实际输入以服务返回的 `prompt_tokens` 为准。TTFT 是每组热身后的一次流式探针，不是 8 个请求的均值；同一提示词重复使用，可能受到前缀缓存影响。输出 tok/s 按该组完成 tokens / 整组墙钟时间计算，包含探针开销；显存列是采样到的整卡已用 VRAM 峰值，而非单组新增占用。

## E2：并发 1/2（已完成）

先停服，将上文启动命令中的 `--max-running-requests 1` 改为 `2` 后重启；保留 BF16 和 8192 上下文。日志须确认至少 10 个 Mamba state slots（每请求需 5 个）及足够 KV 容量，否则记录阻塞原因，不把排队的请求当作真正的两并发。不要直接升到 8 并发。结果文件：`results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e2.json`。

```bash
export PYTHON_BIN=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang/venv-sglang-rocm723/bin/python
cd /mnt/workspace/multimodal-deploy
"$PYTHON_BIN" -u configs/benchmark/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:30000/v1/chat/completions \
  --model qwen3.8-flash-next-fp8-sglang \
  --label qwen3.8-flash-next-sglang-fp8-e2 \
  --contexts 512,4096,8192 --concurrencies 1,2 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e2.json
```

| 目标上下文 | 实际输入 tokens/请求 | 并发 | 完成/请求 | 输出 tok/s | TTFT s | 延迟 P50/P95 s | 已用显存峰值 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 433 | 1 | 8/8 | 37.63 | 0.18 | 6.77 / 6.78 | 173.53 |
| 512 | 433 | 2 | 8/8 | 66.17 | 0.18 | 7.61 / 7.86 | 173.53 |
| 4096 | 2801 | 1 | 8/8 | 35.17 | 0.18 | 7.24 / 7.27 | 174.20 |
| 4096 | 2801 | 2 | 8/8 | 62.36 | 0.18 | 8.14 / 8.16 | 174.20 |
| 8192 | 5507 | 1 | 8/8 | 35.16 | 0.25 | 7.24 / 7.25 | 174.56 |
| 8192 | 5507 | 2 | 8/8 | 62.64 | 0.18 | 8.10 / 8.15 | 174.56 |

六组共 48/48 请求成功、HTTP 200；每请求均生成 256 tokens。相同目标上下文下，两并发的输出吞吐为单并发的约 **1.76–1.78 倍**，P50 延迟增加约 **0.84–0.90 秒**，符合有效并发的表现；但本地 JSON 不含服务启动参数和 Mamba slots 日志，服务端实际并发上限仍须以启动日志确认。TTFT 仍是各组热身后的单次探针（0.18–0.25 秒），重复提示词可能受前缀缓存影响，不应解读为冷启动首 token 延迟。显存为整卡已用量峰值，不是该组的增量。

## E3：长上下文（8192/32768 已完成；65536 未测）

32768 启动日志确认：FP8 权重 131/131 加载成功，耗时 55.19 秒、占用 170.55 GB，加载后可用 20.57 GB；之后在 Mamba/KV 缓存池初始化失败，`total_rest_memory=-24.47 GB`、`max_mamba_cache_size=-213`，没有发出请求。日志记录的最终参数是 `context_length=32768`、`attention_backend=aiter`、`chunked_prefill_size=16384`、`max_prefill_tokens=16384`、`max_running_requests=1`、`cpu_offload_gb=0`、`speculative_algorithm=None`，但 `mem_fraction_static=0.765`。这正好是命令设置的 `0.90 × 0.85`：SGLang 在 AITER + 上下文大于 8192 时会将静态显存比例下调 15%。因此 32768 与 65536 都越过同一阈值，单纯从 65536 降到 32768 并没有解除该预算折减；两次错误的剩余预算也几乎相同。此失败发生在缓存池 sizing 阶段，不是请求推理时的物理显存 OOM。8192 成功配置曾分配 31488 个 KV tokens；32768 测试的实际提示词约 21740 tokens，切换 backend 后若能保住相近 KV 容量，理论上足以跑该测试，但仍须以新日志确认。参考 [SGLang 上游 AITER 折减说明](https://github.com/sgl-project/sglang/pull/25199) 和 [AMD ROCm attention 后端说明](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx)。

随后仅将 standard attention backend 改为 `triton`，其他 32768 启动参数保持不变。该配置启动成功：最终参数为 `attention_backend=triton`、`mem_fraction_static=0.90`；Mamba cache 10 slots（conv 0.02 GB、SSM 0.58 GB），BF16 KV pool 31488 tokens（K/V 各 0.36 GB），`max_running_requests=1`，启动就绪后可用显存 18.58 GB。详见远端日志 `qwen3.8-flash-next-sglang-fp8-e3-context32768-triton-server.log`。这证明 backend 切换解除了启动预算失败；后续 E3 8192/32768 单并发请求均已成功完成。

共享 workload 在 32768 目标档的实际 prompt 为 21740 tokens，每请求生成 256 tokens；总需求约 21996，低于当前 31488-token pool。65536 目标档实际 prompt 约 43385 tokens，加 256 输出约需 43641 tokens，因此当前服务的 pool 不够，不能在当前服务上直接测 65536。E3 本轮已完成 8192/32768 两档、每档 8 个请求，结果见下表；除非需要与 vLLM 的 65536 档严格对齐，否则无需为完成当前 E3 额外启动 65536。

```bash
export PYTHON_BIN=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang/venv-sglang-rocm723/bin/python
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export ROCM_HOME=/opt/rocm ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx942
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$RESULT_DIR"
"$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name qwen3.8-flash-next-fp8-sglang \
  --host 127.0.0.1 --port 30000 --tp-size 1 \
  --context-length 32768 --max-running-requests 1 \
  --attention-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.90 --reasoning-parser auto \
  2>&1 | tee "$RESULT_DIR/qwen3.8-flash-next-sglang-fp8-e3-context32768-triton-server.log"
```

服务就绪后另开终端，检查启动日志中的 KV 容量；以下命令会核对 `/v1/models` 的 `max_model_len=32768`，不符则跳过基准，避免误测旧服务。只测 8192/32768，不把 65536 混入这轮：

```bash
export PYTHON_BIN=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang/venv-sglang-rocm723/bin/python
cd /mnt/workspace/multimodal-deploy
if "$PYTHON_BIN" -c 'import json, urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:30000/v1/models", timeout=5)); n=d["data"][0]["max_model_len"]; print("max_model_len:", n); assert n == 32768, n'; then
  "$PYTHON_BIN" -u configs/benchmark/openai_server_matrix.py \
    --config configs/benchmark/mi300x-serving-matrix.json \
    --endpoint http://127.0.0.1:30000/v1/chat/completions \
    --model qwen3.8-flash-next-fp8-sglang \
    --label qwen3.8-flash-next-sglang-fp8-e3-context32768 \
    --contexts 8192,32768 --concurrencies 1 \
    --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e3.json
else
  echo '[ERROR] 不是 32768 上下文服务；已跳过 E3 基准。'
fi
```

| 目标上下文 | 实际输入 tokens | 完成/请求或状态 | 输出 tok/s | TTFT | 延迟 P50/P95 | 显存峰值 GiB |
|---:|---:|---|---:|---:|---:|---:|
| 8192 | 5507 | 8/8 成功，0 失败 | 35.03 | 0.254 | 7.259 / 7.291 | 175.17 |
| 32768 / AITER | — | ❌ 缓存池初始化失败，未发请求（有效 `mem_fraction_static=0.765`） | — | — | — | — |
| 32768 / Triton | 21740 | 8/8 成功，0 失败 | 34.81 | 0.221 | 7.315 / 7.335 | 179.78 |
| 65536 / AITER | — | ❌ 缓存池初始化失败，未发请求 | — | — | — | — |
| 65536 / Triton | 约 43385 | 未启动；当前 KV pool 31488 tokens，不足以跑此档 | — | — | — | — |

两档各完成 8/8 请求，均按每请求 256 tokens 正常结束。长上下文档的输出吞吐为 34.81 tok/s，较 8192 档的 35.03 tok/s 低约 0.6%；P50 延迟相差约 0.06 秒。TTFT 是每档单次探针（0.254 / 0.221 秒），不宜据此判断长上下文使首 token 更快。JSON 中 GPU peak 是整卡已用显存（175.17 / 179.78 GiB），而请求期间增量仅 0.0001 / 0.0039 GiB，不应把整卡峰值解读成 E3 请求的显存增量。

如果 E3 需要与 vLLM 的 65536 档一一比较，可另启一个 `context-length=65536`、`attention-backend=triton` 的实例，并把 `mem-fraction-static` 从 0.90 小幅提高到 0.92；先确认服务最终 `max_total_num_tokens >= 43641`，否则不要发 65536 请求。提高比例后仍须看 `available_gpu_mem` 并做冒烟。若目标只是完成当前可行的长上下文测试，无需等待 65536，直接用当前服务跑 8192/32768，并将 65536 标注为本轮未测/容量受限。

## E4：Mini-8 质量抽样与启动就绪时间（已完成）

与其它框架的 E4 对照：llama.cpp 还测了 Q5 serving 以及 Q4/Q5 启动就绪时间；vLLM 复用 E1–E3 的 serving/HBM 结果，并补测 3 轮启动就绪时间和 Mini-8（默认 thinking 与关闭 thinking）。SGLang 的 E1–E3 serving/HBM 结果已齐，可直接作为性能证据复用，不必重跑；本路线只有 FP8，没有 Q4/Q5 量化对照。SGLang E4 已完成 Mini-8 与 3 轮启动到 `/health` 就绪时间测量。

E4 复用 E3 已启动的 **32768 / Triton / 单并发**服务，不需要为质量探针另改模型或启动参数。若 E3 服务仍在 `127.0.0.1:30000` 运行，直接做下方服务核验并运行 Mini-8；不要重复启动占用同一端口。若服务已停止或参数不同，先按此配置启动，再运行质量探针：

服务停止时，用以下完整命令恢复 E4 所需服务。它与 E3 的 32768 Triton 启动配置一致；服务日志单独记为 E4，便于和质量 JSON 对照：

```bash
export PYTHON_BIN=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang/venv-sglang-rocm723/bin/python
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export ROCM_HOME=/opt/rocm ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx942
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$RESULT_DIR"
"$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_DIR" --served-model-name qwen3.8-flash-next-fp8-sglang \
  --host 127.0.0.1 --port 30000 --tp-size 1 \
  --context-length 32768 --max-running-requests 1 \
  --attention-backend triton --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.90 --reasoning-parser auto \
  2>&1 | tee "$RESULT_DIR/qwen3.8-flash-next-sglang-fp8-e4-context32768-triton-server.log"
```

服务就绪后，在另一个终端核验健康状态和模型上下文；只有服务确为 `32768` 才运行 E4。探针串行发送 Mini-8 的 8 个问题，固定 `temperature=0`、每题 `max_tokens=4096`、超时 600 秒；不覆盖 thinking 设置。检查回答正文、`reasoning_content` 与 `finish_reason`，不能只凭 HTTP 成功判定质量。结果文件：`results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e4-quality.json`。

```bash
export PYTHON_BIN=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang/venv-sglang-rocm723/bin/python
cd /mnt/workspace/multimodal-deploy
if "$PYTHON_BIN" -c 'import json, urllib.request; urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=5); d=json.load(urllib.request.urlopen("http://127.0.0.1:30000/v1/models", timeout=5)); m=d["data"][0]; print("model:", m["id"], "max_model_len:", m["max_model_len"]); assert m["id"] == "qwen3.8-flash-next-fp8-sglang" and m["max_model_len"] == 32768'; then
  "$PYTHON_BIN" -u configs/benchmark/openai_quality_probe.py \
    --questions configs/benchmark/quality/mtbench-mini-8.json \
    --endpoint http://127.0.0.1:30000/v1/chat/completions \
    --model qwen3.8-flash-next-fp8-sglang \
    --label qwen3.8-flash-next-sglang-fp8-e4-quality-mini8-max4096 \
    --max-tokens 4096 --temperature 0 --timeout 600 \
    --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e4-quality.json
else
  echo '[ERROR] 服务未就绪或不是指定的 32768 SGLang 服务；已跳过 E4。'
fi
```

| 服务配置 | 成功/题数 | 空回答 | `finish_reason=length` | 总输出 tokens | 平均耗时 | 平均观测 tok/s | 结构检查 |
|---|---:|---:|---:|---:|---:|---:|---|
| 32768 / Triton / 单并发 / Mamba BF16 | 8/8 | 0 | 0 | 8980 | 29.95 s | 37.52 | 全部 `finish_reason=stop`；`content` 无 `</think>`，`reasoning_content` 单独返回；未做人工评分 |

结果：[SGLang E4 Mini-8 原始 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-sglang-fp8-e4-quality.json)。该轮未设置 `chat_template_kwargs`，即模型默认 thinking 行为；`mean_observed_output_tokens_per_s` 是端到端观测速率，不等同于服务端纯 decode 吞吐。Mini-8 是可用性与输出行为抽样，不等同于人工质量评分。

### E4 补充：启动到健康就绪（已完成）

测量为启动进程到 `/health` 返回 200，包含权重加载与服务初始化；`uncontrolled` 不清理 OS 文件缓存，因此不要称为冷启动。三轮均使用与 E4 质量测试相同的 **32768 / Triton / 单并发 / Mamba BF16** 配置，且均就绪。汇总如下；如需复测，先停止 30000 端口上现有服务，再运行后面的命令。探针每轮会自行启动并停止服务、保存 JSON 和 `.server.log`；失败时只停止后续轮次，Shell 回到提示符。

| 轮次 | 启动到 `/health` | 就绪 | 缓存状态 |
|---:|---:|---|---|
| 1 | 91.21 s | ✅ | uncontrolled |
| 2 | 93.14 s | ✅ | uncontrolled |
| 3 | 90.14 s | ✅ | uncontrolled |

均值 **91.50 s**，中位数 **91.21 s**，范围 **90.14–93.14 s**。原始 JSON：[`第 1 轮`](../../results/llm/qwen3.8-flash-next/sglang-fp8-e4-startup-1-raw.json)、[`第 2 轮`](../../results/llm/qwen3.8-flash-next/sglang-fp8-e4-startup-2-raw.json)、[`第 3 轮`](../../results/llm/qwen3.8-flash-next/sglang-fp8-e4-startup-3-raw.json)。这些同步的原始 JSON 内部记录的远端日志路径仍为历史 `/mnt/workspace/results/...`；文档中的后续运行命令均已统一输出到项目内 `results/`，原始测量文件不改写。

```bash
export PYTHON_BIN=/mnt/workspace/artifacts/qwen3.8-flash-next-sglang/venv-sglang-rocm723/bin/python
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export ROCM_HOME=/opt/rocm ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx942
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
cd /mnt/workspace/multimodal-deploy
mkdir -p "$RESULT_DIR/reproductions"
if curl -fsS http://127.0.0.1:30000/health >/dev/null; then
  echo '[ERROR] 30000 端口仍有健康服务；先停止它，再运行启动测量。'
else
  for run in 1 2 3; do
    echo "=== SGLang E4 startup $run/3 ==="
    if "$PYTHON_BIN" -u configs/benchmark/server_startup_probe.py \
      --health http://127.0.0.1:30000/health \
      --output "$RESULT_DIR/reproductions/sglang-fp8-e4-startup-${run}-raw.json" \
      --cache-state uncontrolled --timeout 1200 -- \
      "$PYTHON_BIN" -m sglang.launch_server \
      --model-path "$MODEL_DIR" --served-model-name qwen3.8-flash-next-fp8-sglang \
      --host 127.0.0.1 --port 30000 --tp-size 1 \
      --context-length 32768 --max-running-requests 1 \
      --attention-backend triton --mamba-ssm-dtype bfloat16 \
      --mem-fraction-static 0.90 --reasoning-parser auto; then
      echo "[OK] startup run $run/3 completed."
    else
      rc=$?
      echo "[ERROR] startup run $run/3 failed (exit=$rc); stopped further runs."
      break
    fi
  done
fi
```

E1–E3 的每行对应结果 JSON 中一个 `cells` 元素：`completed/requests`、`throughput.completion_tokens_per_s`、`ttft_probe`、`latency_s.p50/p95`、`gpu.peak_gib`；E3 的实际输入长度从 `per_request` 核对。E4 的汇总取 `summary`，具体质量判断结合各题回答核对。未执行或失败的项目保留状态与原因，不填猜测值。
