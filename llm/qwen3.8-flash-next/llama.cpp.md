# Qwen3.8-Flash-Next / llama.cpp / GGUF

- 平台：ModelScope 单张 AMD MI300X 192GB，ROCm 7.2，`gfx942`
- 模型：`Qwen/Qwen3.8-Flash-Next`
- 量化：Unsloth `UD-Q4_K_XL` GGUF（约 104 GiB 下载体积）
- 首个后端：ROCm 版 `llama.cpp`
- 当前状态：**部署模块已完成；E1–E3 基准、E4-Q5 性能基准、Q4/Q5 启动时间及 2048/4096 上限质量抽样完成；4096 轮两组均无失败、空回答或长度截断，但 Mini-8 尚未证明 Q5 质量优于 Q4。**

本页只记录 `llama.cpp + GGUF` 路线：`unsloth/Qwen3.8-Flash-Next-GGUF`、`UD-Q4_K_XL` / `UD-Q5_K_XL`。后续原生 FP8 Safetensors 路线分开记录：[vLLM 部署](vllm.md)、[SGLang 部署](sglang.md)。两条路线使用同一硬件和统一基准脚本，但结果属于部署路线对照，不是同一权重格式下的纯框架对照。

本次 GGUF 在启动时报告 `modalities: text`，因此本记录只覆盖文本推理和服务接口；图片、视频与视觉微调属于独立路线。

参考：[Qwen 官方模型页](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)、[ROCm llama.cpp 文档](https://rocm.docs.amd.com/projects/llama-cpp/en/latest/)。

## 模块 A：模型部署 / Deployment

### 1. 环境检查

先确认 GPU、ROCm、可用磁盘和 PyTorch HIP 可用性，而不是立即下载模型：

```bash
rocm-smi
rocminfo 2>/dev/null | grep -m 5 -E 'Name:.*gfx|gfx942' || true
df -h / /root/tmp /tmp 2>/dev/null || df -h
free -h
python - <<'PY'
import torch
print('torch =', torch.__version__)
print('torch.version.hip =', torch.version.hip)
print('cuda_available =', torch.cuda.is_available())
print('device_count =', torch.cuda.device_count())
if torch.cuda.is_available():
    print('capability =', torch.cuda.get_device_capability(0))
PY
```

实测环境具备单 GPU、约 192GB HBM、200GiB 主机内存；PyTorch 可通过 `cuda:0` 使用 ROCm 设备，目标架构为 `gfx942`。

### 2. 工作目录与模型下载

模型、编译目录放在空间较大的 `/root/tmp`，不要放入容量较小的 `/tmp`：

```bash
export WORK=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$WORK/Qwen3.8-Flash-Next-GGUF"
export UI_DIR="$WORK/llama-ui"
mkdir -p "$WORK" "$MODEL_DIR" "$UI_DIR" "$WORK/outputs"

modelscope download unsloth/Qwen3.8-Flash-Next-GGUF \
  --repo-type model \
  --include 'UD-Q4_K_XL/*' \
  --include '*.json' '*.md' \
  --cache-dir "$MODEL_DIR"

find "$MODEL_DIR" -type f -name '*.gguf' -printf '%12s %p\n' | sort -n
du -sh "$MODEL_DIR"
```

只下载选中的 `UD-Q4_K_XL`，不要将整个 GGUF 仓库的量化版本一并下载。实际下载得到 4 个分片，总量约 104 GiB。

### 3. 编译 ROCm llama.cpp

```bash
cd "$WORK"
git -c http.version=HTTP/1.1 \
  -c http.lowSpeedLimit=1 \
  -c http.lowSpeedTime=600 \
  clone --depth 1 --single-branch \
  https://github.com/AMD-Ecosystem/llama.cpp.git "$WORK/llama.cpp"

cd "$WORK/llama.cpp"
export LLAMACPP_ROCM_ARCH=gfx942
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$LLAMACPP_ROCM_ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=ON
cmake --build build --config Release -j"$(nproc)"
```

处理过的异常：普通浅克隆遇到 `GnuTLS recv error (-110)`，改用 HTTP/1.1 浅克隆后成功。后续的 `cd`、CMake “source directory” 等报错是克隆失败的连锁反应，不应单独修复。

### 4. 文本冒烟测试

```bash
export MODEL_FILE="$(find "$MODEL_DIR" -type f -name '*UD-Q4_K_XL*00001*.gguf' | head -n 1)"
test -n "$MODEL_FILE" && test -f "$MODEL_FILE"

"$WORK/llama.cpp/build/bin/llama-cli" \
  -m "$MODEL_FILE" -ngl 999 -c 8192 -n 256 --jinja --temp 0.6 \
  -p '请用中文说明你是什么模型，并列出三项适合在 AMD MI300X 192GB 上验证的实验。'
```

结果：模型成功加载、完成中文回答；这是“模型与 HIP 路径可运行”的验证，不作为服务性能结论。

### 5. API、Web UI 与 DSW 网关

```bash
"$WORK/llama.cpp/build/bin/llama-server" \
  -m "$MODEL_FILE" -ngl 999 -c 8192 --jinja \
  --ui --path "$UI_DIR" \
  --host 127.0.0.1 --port 8000

curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/v1/models
```

- 服务仅绑定 `127.0.0.1:8000`；通过 DSW 的登录态端口网关访问，而非开放公网端口。实例对应入口形如 `https://<workspace-id>-proxy-8000.dsw-gateway-cn-hangzhou.data.aliyun.com/`，不要把真实临时 URL 写入公开仓库。
- `llama-server` 是 `llama.cpp` 项目内的 HTTP 服务二进制，复用其模型加载、采样和 GPU 后端；`llama.cpp` 是整个项目 / 推理运行时，`llama-cli`、`llama-server` 都是其应用程序。
- 本次二进制开启 `--ui` 后根路径仍返回 404，原因是 UI 静态资源未随构建产出。下载 `index.html` 及它引用的带哈希 `_app/immutable/...` JS/CSS，保留相对目录后，以 `--path "$UI_DIR"` 服务即可。
- 下载 UI 资源时网络受限，临时设置 `HTTP_PROXY` / `HTTPS_PROXY` 后成功；代理只用于下载，不是服务或浏览器访问的必需条件。

实测：`/health` 返回 `{"status":"ok"}`，`/v1/models` 能返回已加载模型；Web UI 经 DSW 网关完成实际对话。

## 模块 B：实验测评 / Evaluation

### 共同测量标准

这里的四个实验不是仅写在本页的“待办”。它们共享：

- 工作负载契约：[mi300x-serving-matrix.json](../../configs/benchmark/mi300x-serving-matrix.json)：上下文目标、并发、输出上限、请求数和采样频率；
- 测量工具：[openai_server_matrix.py](../../configs/benchmark/openai_server_matrix.py)：请求已启动的 OpenAI-compatible 服务，记录实际返回 token 用量、端到端延迟、吞吐、错误和 HBM 采样；
- E4 质量探针：[openai_quality_probe.py](../../configs/benchmark/openai_quality_probe.py)，固定提示集：[MT-Bench Mini-8](../../configs/benchmark/quality/mtbench-mini-8.json)；启动 ready 探针：[server_startup_probe.py](../../configs/benchmark/server_startup_probe.py)；
- 原始证据：[results/](../../results/)：每轮生成一个小型 JSON；
- 本页：记录服务器启动参数、实验目的、异常、图表 / 结果链接与解释。

换言之，`llama.cpp.md` 是本路线的实验笔记和结论，配置文件是“什么算公平负载”，脚本是量尺，JSON 是证据。后续 vLLM / SGLang 只要提供同样的 OpenAI-compatible endpoint，即可使用同一量尺；模型格式或精度不一致时，结果会明确标注为“部署路线比较”，不伪装成纯框架对比。

基准器验证：已在 AMD MI300X 实例完成 dry-run；配置成功展开为 12 个测试单元（上下文 512/4096/8192 × 并发 1/2/4/8）。E1 已完成单并发正式测量。

在新终端复测前，先设置模型和服务路径；后续代码块中的 `$SERVER`、`$MODEL_FILE`、`$Q4` 和 `$Q5` 均使用这些值：

```bash
export WORK=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$WORK/Qwen3.8-Flash-Next-GGUF"
export SERVER="$WORK/llama.cpp/build/bin/llama-server"
export Q4="$(find "$MODEL_DIR" -type f -name '*UD-Q4_K_XL*00001*.gguf' -print -quit)"
export Q5="$(find "$MODEL_DIR" -type f -name '*UD-Q5_K_XL*00001*.gguf' -print -quit)"
export MODEL_FILE="$Q4"
test -x "$SERVER" && test -f "$Q4" && test -f "$Q5"
```

### E1 — 单请求基线

- [x] 固定单槽 / 单并发，依次测目标上下文 512、4096、8192。
- [x] 记录 TTFT、实际 prompt / completion token、端到端延迟、生成吞吐和 HBM 基线 / 峰值。
- [x] 将结果归档为 `results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e1.json`。

首次正式跑数前，将服务重启为单槽：`-c 10240 --parallel 1`。其中 10240 是服务容量，给目标 8192 prompt、最多 256 个生成 token 和模板余量留出空间；它不改变本实验的三个工作负载目标。测量期间不通过 Web UI 发送其他请求。共同基准器的固定末尾指令会尽量消耗生成上限，实际 token 数仍以 API 返回的 `usage` 为准。

下载 UI 时设置过代理也不影响 v0.2.1+ 的本地测量：对 `localhost` / `127.0.0.1` / `::1` endpoint，基准器会自动绕过代理。

首次 E1 运行曾因 ROCm-SMI 数值字符串未被旧解析器识别而缺少 HBM 样本；v0.2.2 修复后，探针和正式 E1 均确认 `gpu.samples > 0`。

```bash
# 终端 A：先在旧 llama-server 终端按 Ctrl+C，再以单槽启动。
"$WORK/llama.cpp/build/bin/llama-server" \
  -m "$MODEL_FILE" -ngl 999 -c 10240 --parallel 1 --jinja \
  --host 127.0.0.1 --port 8000

# 终端 B：确认服务后生成正式 E1 证据文件。
cd /mnt/workspace/multimodal-deploy
curl -fsS http://127.0.0.1:8000/health && echo
python3 -u configs/benchmark/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --label qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e1 \
  --contexts 512,4096,8192 --concurrencies 1 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e1-raw.json
```

该命令生成单次原始测量；复核后归档为上面的 E1 bundle。

正式结果（服务参数：`-ngl 999 -c 10240 --parallel 1 --jinja`）：

| 目标上下文 | 实际 prompt / 请求 | TTFT | P50 / P95 | completion 吞吐 | HBM 基线 → 峰值 |
|---:|---:|---:|---:|---:|---:|
| 512 | 433 | 78.9 ms | 6.481 / 6.527 s | 39.31 tok/s | 77.5343 → 77.5566 GiB |
| 4096 | 2801 | 83.0 ms | 6.644 / 6.711 s | 38.32 tok/s | 77.5594 → 77.5803 GiB |
| 8192 | 5507 | 91.4 ms | 6.763 / 6.804 s | 37.67 tok/s | 77.5330 → 77.5555 GiB |

三组均为 8/8 成功、失败率 0%，每次实际生成 256 token。随着上下文目标增加，TTFT 从约 79 ms 增至 91 ms，单请求生成吞吐从 39.31 降至 37.67 token/s；HBM 峰值稳定在约 77.6 GiB，单个测试单元内增量约 0.02 GiB。这里的“目标上下文”是近似输入工作负载，实际 prompt token 已单独记录，不能把 8192 行直接解释为精确 8K prompt。

结果文件：[E1 脱敏结果（含 HBM）](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e1.json)

目的：建立单请求、单槽、客户端并发为 1 时的无排队参考工作点：在目标上下文 512/4096/8192 下同时记录 TTFT、端到端 P50/P95、实际 Prompt/Completion Token、生成吞吐和 HBM 基线/峰值。结果显示上下文从 512 增至 8192 时，TTFT 约 79→91 ms、吞吐约 39.31→37.67 tok/s，而 HBM 峰值稳定在约 77.6 GiB；因此 E1 可作为后续判断并发、长上下文和量化额外开销的基线。

### E2 — 并发 / 槽位

- [x] 保持 E1 的模型、量化、上下文和输出上限；逐级测试客户端并发 1、2、4、8。
- [x] 服务端固定为 8 个 Slot：build 支持 `--kv-unified-per-slot` 时使用 `--parallel 8 --kv-unified-per-slot 10240`，否则使用 `--parallel 8 -c 81920`；客户端并发逐级为 1、2、4、8。
- [x] 比较 aggregate throughput、P50 / P95 延迟、失败率和 HBM 增量。

E2 结果：12 个测试单元（3 个目标上下文 × 4 个客户端并发），共 96/96 请求成功，失败率 0%；脚本 v0.2.2，所有单元均有有效 HBM 采样。

可复跑命令：先将 Q4 服务重启为 8 个 Slot（build 支持统一 KV 时优先使用第一条；否则使用容量等价的 fallback），再运行共享矩阵测量器。该客户端并发矩阵为 1/2/4/8，工作负载参数从统一配置读取。

```bash
# 终端 A：选择当前 build 支持的一种 8-Slot 配置后启动服务。
"$SERVER" -m "$MODEL_FILE" -ngl 999 --parallel 8 \
  --kv-unified-per-slot 10240 --jinja --host 127.0.0.1 --port 8000
# 若不支持 --kv-unified-per-slot，则用：
# "$SERVER" -m "$MODEL_FILE" -ngl 999 -c 81920 --parallel 8 \
#   --jinja --host 127.0.0.1 --port 8000

# 终端 B：确认服务健康后测量。
cd /mnt/workspace/multimodal-deploy
curl -fsS http://127.0.0.1:8000/health && echo
python3 -u configs/benchmark/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --label qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e2 \
  --contexts 512,4096,8192 --concurrencies 1,2,4,8 \
  --output results/llm/qwen3.8-flash-next/reproductions/llamacpp-e2-context512-4096-8192-concurrency1-2-4-8-raw.json
```

| 客户端并发 | completion 吞吐（512 / 4096 / 8192，tok/s） | 各上下文 P95 最大值 | HBM 峰值范围 |
|---:|---:|---:|---:|
| 1 | 38.58 / 35.52 / 34.81 | 9.167 s | 80.392–80.414 GiB |
| 2 | 54.10 / 63.65 / 62.28 | 14.276 s | 80.397–80.418 GiB |
| 4 | 99.17 / 72.56 / 69.20 | 18.386 s | 80.373–80.396 GiB |
| 8 | 117.92 / 70.02 / 65.10 | 31.246 s | 80.383–80.393 GiB |

按三个上下文单元平均，客户端并发从 1 提升到 2、4、8 时，completion 吞吐约为 36.30、60.01、80.31、84.35 tok/s；并发 8 相比并发 4 吞吐仅小幅增加，但最差 P95 从 18.386 s 上升到 31.246 s，说明本配置的吞吐拐点约在并发 4。HBM 峰值稳定在 80.37–80.42 GiB，未出现 OOM；相比 E1 约 77.6 GiB 的单槽基线增加约 2.8 GiB，符合多 Slot / KV Cache 容量预留增加的现象，但不能仅凭占用差值断言全部来自 KV Cache。

结果文件：[E2 脱敏结果（含 HBM）](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e2.json)

目的：在服务端固定 8 个 Slot 后逐步提高客户端并发，定位吞吐与尾延迟的拐点，并结合 HBM 变化判断退化更像排队/调度竞争，还是显存/KV Cache 压力。结果显示平均吞吐从并发 1 的约 36.30 tok/s 提升到并发 4 的约 80.31 tok/s，但并发 8 仅增至约 84.35 tok/s，最差 P95 却从 18.386 s 升至 31.246 s，HBM 仍稳定在约 80.37–80.42 GiB；因此本配置的高效服务档位约在客户端并发 4，而不是盲目填满 8 个 Slot。

### E3 — 长上下文 / KV cache

- [x] 在低并发下测 8192、32768、65536 上下文目标；显存不足或模型限制应如实记录。
- [x] 选择 32768 上下文，比较低并发 1 与 E2 代表并发 4。
- [x] 标记 KV cache 是否统一、slot 数、实际 prompt token 和 HBM 峰值。

本轮服务配置：固定 `--parallel 1`；build 支持 `--kv-unified-per-slot` 时使用 `--kv-unified-per-slot 65536`，否则使用 `-c 65536`。因此本轮是单 Slot 的长上下文容量测试，不是高并发长上下文测试。

E3 结果：3 个目标上下文 × 单并发，共 24/24 请求成功，失败率 0%；脚本 v0.2.2，301 个 HBM 样本均有效。

可复跑 E3 基线：服务使用单 Slot、65K 容量；客户端并发固定为 1。统一矩阵配置仍提供 8 请求/单元、1 个 warmup 和 HBM/TTFT 采样。

```bash
# 终端 A：支持统一 KV 的 build 使用 --kv-unified-per-slot 65536；
# 否则将该选项替换为 -c 65536。
"$SERVER" -m "$MODEL_FILE" -ngl 999 --parallel 1 \
  --kv-unified-per-slot 65536 --jinja --host 127.0.0.1 --port 8000

# 终端 B：
cd /mnt/workspace/multimodal-deploy
curl -fsS http://127.0.0.1:8000/health && echo
python3 -u configs/benchmark/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --label qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e3 \
  --contexts 8192,32768,65536 --concurrencies 1 \
  --output results/llm/qwen3.8-flash-next/reproductions/llamacpp-e3-context8192-32768-65536-c1-raw.json
```

| 目标上下文 | 实际 Prompt | TTFT | P50 / P95 | completion 吞吐 | HBM 基线 → 峰值 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 5,507 | 89.0 ms | 6.757 / 6.780 s | 37.790 tok/s | 79.239 → 79.262 GiB |
| 32768 | 21,740 | 123.0 ms | 7.079 / 7.091 s | 36.048 tok/s | 79.239 → 79.262 GiB |
| 65536 | 43,385 | 186.3 ms | 7.513 / 7.527 s | 33.925 tok/s | 79.239 → 79.262 GiB |

随着实际 Prompt 从 5,507 增加到 43,385 Token，TTFT 从 89.0 ms 增至 186.3 ms，completion 吞吐从 37.790 降至 33.925 tok/s，P95 从 6.780 s 增至 7.527 s；说明长上下文带来了可观测但可控的 Prefill/调度开销。HBM 稳定在约 79.24 GiB、单元增量约 23 MiB，符合 65K Slot 的 KV Cache 可能在服务启动时预留的现象，不能据此判断 KV Cache 没有占用显存。目标上下文是工作负载目标，实际 Token 以 API 返回的 `usage` 为准，因此 65536 目标对应 43,385 个实际 Prompt Token 不构成异常。

结果文件：[E3 脱敏结果（含 HBM，含基础与补充实验）](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e3.json)

E3 补充实验：使用 Q4 模型、`--parallel 4`，build 支持 `--kv-unified-per-slot` 时设置每 Slot 32768，否则使用 `-c 131072`；在 32768 上下文下比较客户端并发 1 与 4。两组均为 8/8 成功，HBM 样本完整。

| 客户端并发 | TTFT | P50 / P95 | completion 吞吐 | HBM 基线 → 峰值 |
|---:|---:|---:|---:|---:|
| 1 | 135.2 ms | 7.068 / 7.095 s | 36.060 tok/s | 81.372 → 81.395 GiB |
| 4 | 123.4 ms | 49.922 / 89.116 s | 20.213 tok/s | 81.372 → 81.400 GiB |

与原 E3 的 32768 单并发基线（36.048 tok/s、P95 7.091 s）一致，说明补充实验的单并发结果稳定。并发 4 后，completion 吞吐降至 20.213 tok/s，P95 升至 89.116 s；在 32768 长上下文下，E2 短上下文中接近拐点的并发 4 不再适合作为高效服务档位。四 Slot 配置的 HBM 约 81.37–81.40 GiB，比单 Slot 基线约高 2.1 GiB，未出现 OOM。

补充实验已合并进上面的 E3 归档文件，`runs` 中以 `followup-32768-c1-c4` 标识。

32768/C1、C4 补充单元的复跑命令（服务端固定 4 个 Slot、每 Slot 32768 容量；若不支持统一 KV，用 `-c 131072` 替代）：

```bash
# 终端 A：
"$SERVER" -m "$MODEL_FILE" -ngl 999 --parallel 4 \
  --kv-unified-per-slot 32768 --jinja --host 127.0.0.1 --port 8000

# 终端 B：
cd /mnt/workspace/multimodal-deploy
curl -fsS http://127.0.0.1:8000/health && echo
python3 -u configs/benchmark/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --label qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e3-32768-c1-c4 \
  --contexts 32768 --concurrencies 1,4 \
  --output results/llm/qwen3.8-flash-next/reproductions/llamacpp-e3-context32768-c1-c4-raw.json
```

目的：分两步区分长上下文和并发的影响：先固定单槽/单请求，观察 8192、32768、65536 目标上下文的 Prefill、延迟和吞吐变化；再固定 32768 上下文，比较客户端并发 1 与 4。结果显示实际 Prompt 从 5,507 增至 43,385 Token 时，TTFT 约 89→186 ms、吞吐约 37.790→33.925 tok/s，仍全部成功；但 32768 上下文下并发 4 时吞吐降至 20.213 tok/s、P95 升至 89.116 s，而 HBM 仅小幅增加。因此 E3 证明模型不仅能加载长上下文，还能稳定处理单请求；同时表明长上下文下的主要风险是 Prefill/KV Cache 调度竞争，而不是本轮观测到的显存耗尽。HBM 采样只能提供方向性证据，不能精确拆分权重与 KV Cache。

### E4 — 量化权衡

- [x] 下载并测试同一模型族的相邻量化版本 `UD-Q5_K_XL`。
- [x] 复用 E1 工作负载完成 Q5 性能测量。
- [x] 补充固定小型质量提示集：MT-Bench Mini-8（8 个类别各取首轮 1 题）。
- [x] 在相同 API、`temperature=0`、`max_tokens=512` 下完成 Q4/Q5 质量可用性抽样。
- [x] 记录 Q4/Q5 启动到 `/health` 就绪时间；TTFT、吞吐与 HBM 已完成。
- [x] 使用 `max_tokens=4096` 完成 Q4/Q5 质量补测；两组均 8/8 成功、0 空回答、0 次长度截断。

E4-Q5 结果：3 个目标上下文 × 单并发，共 24/24 请求成功，失败率 0%；脚本 v0.2.2，279 个 HBM 样本均有效。

| 目标上下文 | 实际 Prompt | TTFT | P50 / P95 | completion 吞吐 | HBM 基线 → 峰值 |
|---:|---:|---:|---:|---:|---:|
| 512 | 433 | 94.8 ms | 6.448 / 6.486 s | 39.541 tok/s | 97.403 → 97.426 GiB |
| 4096 | 2801 | 112.9 ms | 6.613 / 6.643 s | 38.556 tok/s | 97.444 → 97.465 GiB |
| 8192 | 5507 | 107.5 ms | 6.727 / 6.771 s | 37.898 tok/s | 97.418 → 97.440 GiB |

与 E1 的 `UD-Q4_K_XL` 基线相比，Q5 在三个上下文上的生成速度约高 0.23 tok/s（约 0.6%），但 HBM 峰值从约 77.6 GiB 增至约 97.4 GiB，增加约 19.9 GiB；因此当前证据显示 Q5 的额外显存成本明显，而 serving 吞吐收益很小。

质量可用性抽样：固定使用 MT-Bench Mini-8（`writing`、`roleplay`、`reasoning`、`math`、`coding`、`extraction`、`stem`、`humanities` 各一题的首轮），经同一 `llama-server` OpenAI-compatible 端点分别测试 Q4 与 Q5。两组均为 8/8 HTTP 200、失败率 0%，且 completion token 总数同为 3,037。

E4-Q5 serving 的复测代码复用 E1–E3 的 `openai_server_matrix.py`；Q4 对照取 E1 的单并发基线。先将服务切换为 Q5、单 Slot（`-c 10240 --parallel 1`），再分终端启动服务并运行测量：

```bash
# 终端 A：
"$SERVER" -m "$Q5" -ngl 999 -c 10240 --parallel 1 --jinja \
  --host 127.0.0.1 --port 8000

# 终端 B：
cd /mnt/workspace/multimodal-deploy
curl -fsS http://127.0.0.1:8000/health && echo
python3 -u configs/benchmark/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --label qwen3.8-flash-next-ud-q5-k-xl-llamacpp-e4-serving \
  --contexts 512,4096,8192 --concurrencies 1 \
  --output results/llm/qwen3.8-flash-next/reproductions/llamacpp-e4-q5-serving-context512-4096-8192-raw.json
```

E4 的启动时间测量由 `server_startup_probe.py` 负责，Mini-8 质量测量由 `run_qwen_e4_quality.sh` 调用 `openai_quality_probe.py`；下方“E4 收尾”保留完整运行命令。

| 量化 | 平均端到端耗时 | 平均观测输出速率 | 达到 512 token 上限的题数 | 空最终回答 |
|---|---:|---:|---:|---|
| `UD-Q4_K_XL` | 9.961 s | 37.428 tok/s | 5 / 8 | `coding` |
| `UD-Q5_K_XL` | 9.877 s | 37.740 tok/s | 4 / 8 | `roleplay`、`coding` |

Q5 的平均观测输出速率仅高约 0.31 tok/s（约 0.8%），与 E4 serving 基准中的微小差异一致。两组均有多题碰到 512 token 上限，且空最终回答很可能与默认保留 reasoning 输出有关；因此本轮证明了端到端接口、固定提示集和响应归档均可用，但**不能据此断言 Q5 的文本质量优于 Q4**。后续若需严谨质量结论，应统一提高两边的输出上限，并保留 reasoning 字段和 finish_reason；`--no-reasoning-preserve` 控制历史推理保留，不是关闭单轮思考；本轮 GPU 时间耗尽，不补跑。

结果文件：[E4 serving 与启动归档](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e4-serving.json)、[E4 质量归档](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e4-quality.json)。其中 Q4 serving 使用 E1 单请求基线作为对照，Q5 使用 E4-Q5 serving 结果。

启动时间补测结果已合并进 [E4 serving 与启动归档](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e4-serving.json)，其中 `startup_runs` 保留 Q4/Q5 各 3 轮原始记录。

六轮均为 `ready=true`，启动参数完全一致（`-ngl 999 -c 10240 --parallel 1 --jinja`），指标为 `process_launch_to_health_ready`；缓存状态均为 `uncontrolled`。

| 量化 | 3 轮启动时间（s） | 中位数（s） | 平均值（s） |
|---|---:|---:|---:|
| `UD-Q4_K_XL` | 99.049 / 81.593 / 81.424 | 81.593 | 87.356 |
| `UD-Q5_K_XL` | 120.277 / 102.444 / 102.251 | 102.444 | 108.324 |

按中位数，Q5 比 Q4 多 20.851 s，约高 25.6%；首轮较慢且两种量化都出现，说明 OS 文件缓存等因素会影响绝对值。因此这里可作同机同参数下的相对启动开销证据，不宣称冷启动基准，也不把它解释为首 token 延迟。Q5 的额外启动时间与其更大的权重体积方向一致。

### E4 收尾：加载时间与质量补测

前提：Q4/Q5 GGUF、`llama-server` 和本仓库基准脚本已经准备好，且 8000 端口没有健康服务。本节不重复远程 Linux 环境、模型下载或 llama.cpp 编译步骤。

#### 启动时间

使用 `server_startup_probe.py` 测量进程启动到 `/health` 返回 HTTP 200。脚本只停止自己启动的子进程，并将同名 `.server.log` 写在 JSON 旁边。将 `SERVER`、`Q4`、`Q5` 分别设置为已有的服务程序和两种量化 GGUF 第一分片：

```bash
RESULT_ROOT="results"
for quant in q4 q5; do
  if [ "$quant" = q4 ]; then model="$Q4"; else model="$Q5"; fi
  for run in 1 2 3; do
    python configs/benchmark/server_startup_probe.py \
      --output "$RESULT_ROOT/qwen3.8-flash-next-ud-${quant}-k-xl-llamacpp-e4-startup-${run}.json" \
      --cache-state uncontrolled --timeout 900 -- \
      "$SERVER" -m "$model" -ngl 999 -c 10240 --parallel 1 --jinja \
      --host 127.0.0.1 --port 8000 || break
  done
done
```

指标是进程启动到健康检查成功，包含权重加载、上下文分配和服务初始化；不清理 OS 文件缓存，因此标记为 `uncontrolled`，不宣称冷启动，也不把下载或首请求延迟算入。首轮与后续轮次分开报告，后续轮次可报告中位数。

#### 质量补测

Q4/Q5 使用同一 MT-Bench Mini-8、`temperature=0`、`max_tokens=4096` 和服务模板。推荐直接运行仓库脚本；它会检查两种量化的第一分片、拒绝复用已占用的健康端口，分别启动服务并等待 `/health`，保存 JSON 与 `.server.log`，最后自动停服：

```bash
MAX_TOKENS=4096 bash configs/benchmark/run_qwen_e4_quality.sh
```

脚本运行时会产生 Q4/Q5 原始 JSON；本次复核后已合并归档为 [E4 质量归档](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e4-quality.json)。若远端实例的仓库路径或模型缓存不同，可在运行前覆盖 `WORK`、`MODEL_DIR`、`SERVER`、`RESULT_ROOT` 环境变量。终端保持在脚本末尾，便于查看失败日志而不会自动关闭。

先检查 `summary.empty_answer_count`、`length_finish_count`、推理字段和 `finish_reason`，再比较指令遵循、事实正确性和完整性。Mini-8 只能作为小样本可用性抽样，不能单独证明 Q5 普遍优于 Q4；不要用输出长度或 HTTP 200 代替质量判断。

#### 2048 上限补测结果

结果文件：[E4 质量归档](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e4-quality.json) 中的 `q4-quality-2048` 与 `q5-quality-2048` 记录。两组均使用相同 Mini-8、`temperature=0`、`max_tokens=2048` 和 `--ngl 999 -c 10240 --parallel 1 --jinja`，均为 8/8 HTTP 200、无空回答。

本轮结果文件由远端旧版探针生成，没有保留 `response_choices` 和 `finish_reason`；下面的“达到上限”按 `completion_tokens == 2048` 判定，因此是保守的截断信号：

| 量化 | 总 completion token | 平均观测输出速率 | 达到 2048 上限 | 空回答 |
|---|---:|---:|---:|---:|
| `UD-Q4_K_XL` | 6,689 | 37.414 tok/s | 2/8（`writing`、`humanities`） | 0/8 |
| `UD-Q5_K_XL` | 8,141 | 38.137 tok/s | 3/8（`writing`、`coding`、`humanities`） | 0/8 |

两组的 `reasoning`、`math`、`extraction` 都给出了可核对的正确结果，`roleplay` 和 `stem` 输出连贯；但写作、人文两题，以及 Q5 的 coding 题都在上限处结束，不能据此比较完整性。Q5 平均观测速率比 Q4 高约 0.723 tok/s（约 1.9%），同时生成 token 总数高约 21.7%，这更像是输出长度差异，不能作为质量优势证据。

#### 4096 上限补测结果

结果文件：[E4 质量归档](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-llamacpp-e4-quality.json) 中的 `q4-quality-4096` 与 `q5-quality-4096` 记录。两组均使用相同 Mini-8、`temperature=0`、`max_tokens=4096` 和 `-ngl 999 -c 10240 --parallel 1 --jinja`；本轮由探针 `0.1.1` 生成，保留了 `response_choices`、`finish_reason` 和长度统计。

| 量化 | 成功 | 空回答 | 长度截断 | 总 completion token | 平均观测输出速率 | 平均端到端耗时 |
|---|---:|---:|---:|---:|---:|---:|
| `UD-Q4_K_XL` | 8/8 | 0/8 | 0/8 | 8,565 | 38.036 tok/s | 27.395 s |
| `UD-Q5_K_XL` | 8/8 | 0/8 | 0/8 | 10,274 | 38.308 tok/s | 32.512 s |

两组 8 题的 `finish_reason` 全为 `stop`，因此本轮没有观察到 2048 轮中的上限截断。Q5 比 Q4 多生成约 19.9% 的 token，平均端到端耗时高约 18.7%，但平均观测输出速率只高约 0.7%；耗时差异主要来自生成内容更长，而不是单位 token 速度显著更快。Q5 的 `roleplay`、`coding` 输出更长，Q4 的 `humanities` 输出更长；在这组 Mini-8 上，两边都未显示足以支持“Q5 质量更优”的稳定证据，因此不扩展为完整 MT-Bench 质量结论。

质量实验是否继续：本轮目标是验证 2048 上限是否造成截断，而不是给 Q4/Q5 做完整质量排名。4096 轮已经满足这个目标：两组 8/8 成功、0 空回答、0 次长度截断，且 `finish_reason` 全为 `stop`，因此 E4 不再继续追加完整 MT-Bench。这里“样本不足以证明 Q5 质量稳定优于 Q4”只表示 Mini-8 只有 8 道题，且没有独立评分或多轮重复；它不表示 Q5 质量较差，也不表示当前实验失败。只有在后续必须做正式量化质量排名，或实际业务输出经常超过 4096 token 时，才值得另开质量实验，并预先固定评分集、评分方法、输出上限和停止条件。

本轮耗时较长的主要原因是先后使用了 512、2048、4096 三个输出上限，并在缺少 `finish_reason` 的旧版探针下重复确认截断。后续 vLLM / SGLang 测试应先固定模型、量化/精度、上下文、并发和 `max_tokens`，先做一轮 1–2 题冒烟；只有发现失败、截断或参数不一致时才扩展。对 Mini-8，`max_tokens=4096` 作为初始上限已经足够；若仍有 `length` 截断，再统一提高到 8192。`max_tokens` 是单次生成上限，不是上下文窗口大小。

目的：在同一张 192GB MI300X、同一套服务参数和相同工作负载下，从 serving 吞吐、HBM 占用、启动到 `/health` 就绪时间以及固定 Mini-8 质量抽样四个维度比较 Q4/Q5，形成这张卡上的量化选择建议。结果显示 Q5 serving 吞吐仅约高 0.6%，HBM 却多约 19.9 GiB；启动中位数约 102.444 s，相比 Q4 的 81.593 s 高约 25.6%；4096 质量轮虽无截断，但未证明 Q5 有稳定质量优势。因此当前 MI300X + llama.cpp 配置优先选 Q4，Q5 仅作为质量敏感场景的备选；该结论不泛化到其他硬件、后端或权重格式。

### 后续对照顺序

1. 完成 E1–E4 的 llama.cpp / GGUF 路线；
2. 使用同一份官方 FP8 Safetensors 权重，分别完成 vLLM 与 SGLang 的单请求启动预检；
3. 预检通过后，按相同工作负载契约先做各自 E1，再决定是否扩展 E2/E3；
4. 视觉模型部署、视频模型和微调另开实验卡，不与本次文本 serving 基准混合。

## 状态

- [x] 环境自检
- [x] 下载 UD-Q4_K_XL
- [x] 编译 ROCm llama.cpp
- [x] 文本冒烟
- [x] API、Web UI 与 DSW 登录态网关实际对话
- [x] E1 单请求基线
- [x] E2 并发 / 槽位
- [x] E3 长上下文 / KV cache
- [x] E4 量化权衡（性能、启动时间和 2048/4096 质量抽样已完成；4096 轮未观察到稳定质量优势）
- [ ] vLLM 对照
- [ ] SGLang 对照（按需）
