# Qwen3.8-Flash-Next / vLLM / FP8

- 后端：vLLM
- 权重：[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- 格式：Hugging Face Transformers / Safetensors，细粒度 FP8，block size 128
- 硬件目标：单张 AMD MI300X 192GB，ROCm，`gfx942`
- 对照基线：[llama.cpp / GGUF](llama.cpp.md)
- 结果目录：`results/llm/qwen3.8-flash-next/`

## 口径

这是原生 FP8 Safetensors 与 GGUF Q4/Q5 的部署路线对照。FP8 权重不是把现有 GGUF 转换得到的文件；vLLM 只使用这里声明的 FP8 模型目录。官方 Qwen 模型页明确给出 vLLM 用法，但当前 vLLM recipe 的 AMD 示例主要覆盖 MI355X，因此 MI300X 必须先做启动预检，再决定是否进入 E1–E3。

## 磁盘与模型目录

当前 `/mnt/data/lcx` 只有约 100 GB，不足以放下约 173 GiB 的 FP8 权重；`/etc/dsw/instance` 又是只读挂载。因此先清理已经完成 E4 且约 252 GB 的 GGUF 目录，再把 FP8 和 Hugging Face 缓存放在 `/root/tmp`，为下载和启动保留足够根分区余量。把路径固定下来：

```bash
export MODEL_ROOT=/root/tmp/qwen3.8-flash-next
export HF_HOME="$MODEL_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export MODEL_DIR="$MODEL_ROOT/Qwen3.8-Flash-Next-FP8"
mkdir -p "$MODEL_ROOT" "$HF_HOME"

# 关闭 Xet 分块缓存，避免下载时额外占用较大的缓存空间；需要更高吞吐时可删除这一行。

export GGUF_DIR="$MODEL_ROOT/Qwen3.8-Flash-Next-GGUF"
df -h /
du -sh "$GGUF_DIR" 2>/dev/null || true
# E4 已归档且不再需要 llama.cpp 重跑时，释放约 252 GiB 根分区空间。
rm -rf -- "$GGUF_DIR"
df -h /

hf download Qwen/Qwen3.8-Flash-Next-FP8 \
  --local-dir "$MODEL_DIR"

du -sh "$MODEL_DIR" "$HF_HUB_CACHE" 2>/dev/null || true
find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' | wc -l
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
```

上面的清理只针对已经完成 E4、且结果已归档的 GGUF；保留约 490 MiB 的 `llama.cpp` 源码即可。若还需要重跑 llama.cpp，应先跳过 `rm -rf`，但必须确认根分区仍有足够余量。

如果 Hugging Face 下载速度过慢，可改用同名的 ModelScope 模型
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next-FP8)。使用 `--local_dir` 直接写入同一个 `MODEL_DIR`，不要另外指定缓存目录，避免产生第二份完整权重：

```bash
command -v modelscope || python3 -m pip install -U modelscope
modelscope download \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --local_dir "$MODEL_DIR"
```

先用 `config.json` 和 `model.safetensors.index.json` 做小文件探测，确认 ModelScope 链路后再开始完整下载。ModelScope 官方 CLI 的参数名是 `--local_dir`；Qwen 官方也将 ModelScope 列为 Hugging Face 不便访问时的下载入口。

```bash
export MS_PROBE_DIR="$MODEL_ROOT/Qwen3.8-Flash-Next-FP8-ms-probe"
mkdir -p "$MS_PROBE_DIR"
modelscope download \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  config.json model.safetensors.index.json \
  --local_dir "$MS_PROBE_DIR"
```

## 安装预检

当前官方 vLLM ROCm wheel 要求 Python 3.12；当前最新稳定索引提供 `0.31.0+rocm723`，但该 wheel 的平台标签要求更高 glibc。当前实例是 glibc 2.35，因此先探测 Qwen3.8 所需的 `0.30.0+rocm723` wheel；若仍不兼容，再转官方 ROCm 容器或源码构建。若解析过程出现 `nvidia-*` CUDA 依赖，应立即停止，这表示选到了 PyPI CUDA wheel：

```bash
python3 --version
rocminfo 2>/dev/null | grep -m 3 -E 'gfx942|Name:' || true
cat /opt/rocm/.info/version 2>/dev/null || true
ldd --version | head -n 1
uv venv --python 3.12 --seed "$MODEL_ROOT/.venv-vllm"
source "$MODEL_ROOT/.venv-vllm/bin/activate"
# Qwen3.8-Flash-Next 从 vLLM 0.30.0 开始支持；先探测 glibc 2.35 是否可用
uv pip install 'vllm==0.30.0+rocm723' \
  --extra-index-url https://wheels.vllm.ai/rocm/0.30.0/rocm723 \
  --index-strategy unsafe-best-match
python -c 'import vllm, torch; print("vllm", vllm.__version__); print("torch", torch.__version__, torch.version.hip); print(torch.cuda.is_available())'
```

如果实例的 Python、ROCm 或 glibc 与 wheel 不匹配，先停在环境预检，不要改用 CUDA wheel。

### 2026-10-17 实例预检结果

- Python 3.12.13；ROCm 7.2.3；单 GPU `gfx942`，VRAM 约 191.7 GiB；`/dev/kfd` 与 `/dev/dri` 已暴露。
- 系统 glibc 为 2.35。
- `vllm==0.30.0+rocm723` 与 `vllm==0.31.0+rocm723` 当前可见 wheel 均为 `manylinux_2_39_x86_64`，因此在本实例上无法通过预编译 wheel 安装；这不是网络或代理问题。
- 预编译 ROCm wheel 受当前 glibc 2.35 平台标签限制，未采用；已改为 ROCm 7.2.3 + `gfx942` 源码构建并生成 vLLM wheel。Docker 支线检查显示当前实例只有 Docker CLI，没有可连接的 daemon，因此暂不采用官方 ROCm 容器路线。

## Docker 路线检查（2026-10-17）

本日同时检查了官方 ROCm 容器所需的 Docker 运行条件：

- Docker CLI 存在于 `/etc/dsw/runtime/export_bin/docker`，Client 版本为 `28.1.0`；
- `DOCKER_HOST` 未设置，默认端点为 `unix:///var/run/docker.sock`；
- `/var/run/docker.sock` 与 `/run/docker.sock` 均不存在；
- 当前没有 `dockerd` 进程，也没有 `dockerd` 可执行文件；
- `docker info` 只能返回 Client 信息，Server 报无法连接 daemon。

结论是“Docker 客户端已安装，但当前 DSW 实例没有可用的 Docker daemon”，不能据此启动 `vllm/vllm-openai-rocm` 容器。该结论与 vLLM 源码构建路线相互独立；今天先保留已生成的 vLLM wheel 与持久化虚拟环境包，AMD FP8 PLE 兼容性留到下一工作日处理。

## 单请求启动预检

### 2026-10-17 启动预检结果

当前源码构建的 `vllm==0.30.0+rocm723` 已能导入 `vllm._C`、`vllm._C_stable_libtorch` 和 `vllm._rocm_C`，但使用 `/root/tmp` 路径启动时，600 秒内没有等到 EngineCore ready，随后退出：

```text
TimeoutError: Timed out waiting for engine core processes to start.
health_http=000
```

Triton FP8 MoE 后端已选中，多模态 warmup 也已完成；因此该结果先记录为启动超时，不把它误判为基础组件缺失。模型权重不复制到持久目录，当前实例继续使用 `/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8`；持久目录只复用已经保存的 vLLM wheel、环境/源码、日志和结果。复测只验证已保存的 vLLM 构建物能否启动当前模型，AMD Qwen3.8 FP8 PLE 兼容性另行处理。证据归档见 [`qwen3.8-flash-next-vllm-fp8-e0-startup.json`](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-e0-startup.json)。

先用保守的单 GPU、单请求、10K 上下文启动；不启用 MTP、专家并行、FP8 KV cache 等额外优化。下面的参数用于判断“能否加载并回答”，不是最终性能配置：

```bash
export VLLM_RESULT_ROOT=results
mkdir -p "$VLLM_RESULT_ROOT"

vllm serve "$MODEL_DIR" \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --served-model-name qwen3.8-flash-next-fp8 \
  --max-model-len 10240 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  2>&1 | tee "$VLLM_RESULT_ROOT/qwen3.8-flash-next-vllm-fp8-startup.log"
```

另一终端先验证：

```bash
curl -fsS http://127.0.0.1:8000/health && echo
curl -fsS http://127.0.0.1:8000/v1/models | tee "$VLLM_RESULT_ROOT/qwen3.8-flash-next-vllm-fp8-models.json"
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp8","messages":[{"role":"user","content":"用一句话说明你是什么模型。"}],"max_tokens":256,"temperature":0}'
```

若启动 OOM 或模型架构不识别，保存完整启动日志并停止；这属于 MI300X 兼容性结果，不要直接调大上下文或打开更多优化来掩盖问题。

## 2026-10-17 vLLM 0.31.0 持久化复现结果

本轮已经完成了 vLLM 0.31.0 的功能性启动预检。与之前的 v0.30.0 超时不同，v0.31.0 已经完成 AMD Qwen4Exp PLE FP8 权重加载：

- `Qwen4ExpPLEFp8EmbeddingMethod` 初始化成功；
- 131 个 Safetensors 分片通过权重加载阶段；
- `weight_dtype=torch.float8_e4m3fn`，PLE 的 FP8 权重路径已实际执行；
- `Application startup complete`；
- `/health` 返回 HTTP 200，`/v1/models` 返回 `qwen3.8-flash-next-fp8`；
- 单请求聊天冒烟返回 `finish_reason=stop`。

冒烟响应中可见 `</think>` 标记和一小段元提示词式前缀；接口可用性已通过，但后续质量对照还要统一处理 reasoning/chat template 的可见性，不能只按 HTTP 200 判断输出质量。

本轮使用 `--enforce-eager`。不带该参数时，模型同样能进入权重加载阶段，但随后在 Torch Dynamo 动态形状编译阶段失败，错误集中在 `query_start_loc.size()[0]` 的约束冲突。因此当前结论是：**vLLM 0.31.0 + AMD FP8 PLE 已具备功能可用性，默认 V1 编译路径仍需单独排障；eager 模式暂作为可复现基线。**

成功结果见 [`qwen3.8-flash-next-vllm-fp8-e0-eager-startup.json`](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-vllm-fp8-e0-eager-startup.json)。

### 持久化边界

持久目录只保存 vLLM 运行所需的虚拟环境、vLLM wheel、源码归档、依赖清单、版本指纹和日志；不复制约 173 GiB 的 FP8 模型。模型仍由当前实例单独下载到临时模型目录，避免把模型文件重复写入工作区。

仓库中的 [`vllm-persistent-replay.sh`](vllm-persistent-replay.sh) 已同步到 AMD 实例的持久路径 `/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm-persistent-replay.sh`，提供 `persist`、`verify` 和 `replay` 三个动作：

```bash
# 当前实例仍保留 /root/tmp 构建物时执行一次
export BUILD_ROOT=/root/tmp/qwen3.8-flash-next-vllm0310
export PERSIST_SCRIPT=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm-persistent-replay.sh
bash "$PERSIST_SCRIPT" persist

# 检查环境是否真的已经迁移到持久目录
bash "$PERSIST_SCRIPT" verify

# 后续从持久目录启动；只需要把 MODEL_DIR 指向当前实例的模型目录
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
bash "$PERSIST_SCRIPT" replay
```

`replay` 使用 `/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm0310-20261017/venv-vllm-0310` 中的 Python 和 vLLM，不依赖 `/root/tmp` 中的 vLLM 源码或虚拟环境；`MODEL_DIR` 是唯一保留在临时空间的模型路径。
脚本不会为 `replay` 设置或读取 `BUILD_ROOT`；该变量只在第一次 `persist` 时指向当前临时构建目录。

### 下一步：持久目录无 eager 对照

先从持久环境复现上面的 eager 结果，再在同一个持久环境、同一个模型和 `max_model_len=1024` 下移除 `--enforce-eager`，只改变编译配置。建议先尝试：

```bash
export ARTIFACT_ROOT=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm
export PERSIST_ROOT="$ARTIFACT_ROOT/vllm0310-20261017"
export PERSIST_VENV="$PERSIST_ROOT/venv-vllm-0310"
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export LOG="$ARTIFACT_ROOT/logs/vllm0310-persistent-no-eager.log"

export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ROCM_ARCH=gfx942
export VLLM_ENGINE_READY_TIMEOUT_S=1200

mkdir -p "$ARTIFACT_ROOT/logs"

"$PERSIST_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
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
  2>&1 | tee "$LOG"
```

这一步用于判断“关闭编译但不打开 eager”是否能避开当前 `query_start_loc` 约束错误。若 vLLM 0.31.0 拒绝该 `compilation-config`，不要改动环境，保留错误日志并回到 `--enforce-eager` 基线；只有该控制变量明确后，才考虑 V0 引擎或 Triton 版本调整。

当前 vLLM 结果文件：

```text
results/llm/qwen3.8-flash-next/
  qwen3.8-flash-next-vllm-fp8-e0-startup.json          # v0.30.0 历史超时
  qwen3.8-flash-next-vllm-fp8-e0-eager-startup.json   # v0.31.0 eager 成功
```

## 测评顺序

1. 单请求冒烟：`max_tokens=256`；
2. 复用 `configs/benchmark/mi300x-serving-matrix.json` 做 vLLM E1；
3. 确认 E1 参数和结果结构后，再决定是否做 E2/E3；
4. 质量抽样统一用 `max_tokens=4096`，与 llama.cpp 的质量口径一致。

结果文件使用带后端和格式的名称，例如：

```text
results/llm/qwen3.8-flash-next/
  qwen3.8-flash-next-vllm-fp8-e1.json
  qwen3.8-flash-next-vllm-fp8-e2.json
  qwen3.8-flash-next-vllm-fp8-e3.json
  qwen3.8-flash-next-vllm-fp8-e4-quality.json
```


