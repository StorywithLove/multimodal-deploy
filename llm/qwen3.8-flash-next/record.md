# Qwen3.8-Flash-Next / UD-Q4_K_XL

- 平台：ModelScope 单张 AMD MI300X 192GB，ROCm 7.2，`gfx942`
- 模型：`Qwen/Qwen3.8-Flash-Next`
- 量化：Unsloth `UD-Q4_K_XL` GGUF（约 104 GiB 下载体积）
- 首个后端：ROCm 版 `llama.cpp`
- 当前状态：**部署模块已完成；E1–E2 与 E3 低并发长上下文测评完成，E3 代表并发补充与 E4 待开始**

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
- 测量工具：[openai_server_matrix.py](../../benchmarks/openai_server_matrix.py)：请求已启动的 OpenAI-compatible 服务，记录实际返回 token 用量、端到端延迟、吞吐、错误和 HBM 采样；
- 原始证据：[results/](../../results/)：每轮生成一个小型 JSON；
- 本页：记录服务器启动参数、实验目的、异常、图表 / 结果链接与解释。

换言之，`record.md` 是实验笔记和结论，配置文件是“什么算公平负载”，脚本是量尺，JSON 是证据。后续 vLLM / SGLang 只要提供同样的 OpenAI-compatible endpoint，即可使用同一量尺；模型格式或精度不一致时，结果会明确标注为“部署路线比较”，不伪装成纯框架对比。

基准器验证：已在 AMD MI300X 实例完成 dry-run；配置成功展开为 12 个测试单元（上下文 512/4096/8192 × 并发 1/2/4/8）。E1 已完成单并发正式测量。

### E1 — 单请求基线

- [x] 固定单槽 / 单并发，依次测目标上下文 512、4096、8192。
- [x] 记录 TTFT、实际 prompt / completion token、端到端延迟、生成吞吐和 HBM 基线 / 峰值。
- [x] 将结果保存为 `results/qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e1.json`。

首次正式跑数前，将服务重启为单槽：`-c 10240 --parallel 1`。其中 10240 是服务容量，给目标 8192 prompt、最多 256 个生成 token 和模板余量留出空间；它不改变本实验的三个工作负载目标。测量期间不通过 Web UI 发送其他请求。共同基准器的固定末尾指令会尽量消耗生成上限，实际 token 数仍以 API 返回的 `usage` 为准。

下载 UI 时设置过代理也不影响 v0.2.1+ 的本地测量：对 `localhost` / `127.0.0.1` / `::1` endpoint，基准器会自动绕过代理。

首次 E1 运行曾因 ROCm-SMI 数值字符串未被旧解析器识别而缺少 HBM 样本；v0.2.2 修复后，探针和正式 E1 均确认 `gpu.samples > 0`。

```bash
# 终端 A：先在旧 llama-server 终端按 Ctrl+C，再以单槽启动。
"$WORK/llama.cpp/build/bin/llama-server" \
  -m "$MODEL_FILE" -ngl 999 -c 10240 --parallel 1 --jinja \
  --host 127.0.0.1 --port 8000

# 终端 B：确认服务后生成正式 E1 证据文件。
cd /mnt/workspace
curl -fsS http://127.0.0.1:8000/health && echo
python benchmarks/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --label qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e1 \
  --contexts 512,4096,8192 --concurrencies 1 \
  --output results/qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e1.json
```

正式结果（服务参数：`-ngl 999 -c 10240 --parallel 1 --jinja`）：

| 目标上下文 | 实际 prompt / 请求 | TTFT | P50 / P95 | completion 吞吐 | HBM 基线 → 峰值 |
|---:|---:|---:|---:|---:|---:|
| 512 | 433 | 78.9 ms | 6.481 / 6.527 s | 39.31 tok/s | 77.5343 → 77.5566 GiB |
| 4096 | 2801 | 83.0 ms | 6.644 / 6.711 s | 38.32 tok/s | 77.5594 → 77.5803 GiB |
| 8192 | 5507 | 91.4 ms | 6.763 / 6.804 s | 37.67 tok/s | 77.5330 → 77.5555 GiB |

三组均为 8/8 成功、失败率 0%，每次实际生成 256 token。随着上下文目标增加，TTFT 从约 79 ms 增至 91 ms，单请求生成吞吐从 39.31 降至 37.67 token/s；HBM 峰值稳定在约 77.6 GiB，单个测试单元内增量约 0.02 GiB。这里的“目标上下文”是近似输入工作负载，实际 prompt token 已单独记录，不能把 8192 行直接解释为精确 8K prompt。

结果文件：[E1 脱敏结果（含 HBM）](../../results/qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e1.json)

目的：先回答“在没有排队和批处理影响时，这条 GGUF + llama.cpp 路线的工作点是什么”。

### E2 — 并发 / 槽位

- [x] 保持 E1 的模型、量化、上下文和输出上限；逐级测试客户端并发 1、2、4、8。
- [x] 服务端固定为 8 个 Slot：build 支持 `--kv-unified-per-slot` 时使用 `--parallel 8 --kv-unified-per-slot 10240`，否则使用 `--parallel 8 -c 81920`；客户端并发逐级为 1、2、4、8。
- [x] 比较 aggregate throughput、P50 / P95 延迟、失败率和 HBM 增量。

E2 结果：12 个测试单元（3 个目标上下文 × 4 个客户端并发），共 96/96 请求成功，失败率 0%；脚本 v0.2.2，所有单元均有有效 HBM 采样。

| 客户端并发 | completion 吞吐（512 / 4096 / 8192，tok/s） | 各上下文 P95 最大值 | HBM 峰值范围 |
|---:|---:|---:|---:|
| 1 | 38.58 / 35.52 / 34.81 | 9.167 s | 80.392–80.414 GiB |
| 2 | 54.10 / 63.65 / 62.28 | 14.276 s | 80.397–80.418 GiB |
| 4 | 99.17 / 72.56 / 69.20 | 18.386 s | 80.373–80.396 GiB |
| 8 | 117.92 / 70.02 / 65.10 | 31.246 s | 80.383–80.393 GiB |

按三个上下文单元平均，客户端并发从 1 提升到 2、4、8 时，completion 吞吐约为 36.30、60.01、80.31、84.35 tok/s；并发 8 相比并发 4 吞吐仅小幅增加，但最差 P95 从 18.386 s 上升到 31.246 s，说明本配置的吞吐拐点约在并发 4。HBM 峰值稳定在 80.37–80.42 GiB，未出现 OOM；相比 E1 约 77.6 GiB 的单槽基线增加约 2.8 GiB，符合多 Slot / KV Cache 容量预留增加的现象，但不能仅凭占用差值断言全部来自 KV Cache。

结果文件：[E2 脱敏结果（含 HBM）](../../results/qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e2.json)

目的：找出“单请求很快”何时因排队、槽位或 KV cache 变成“整体服务变慢”。

### E3 — 长上下文 / KV cache

- [x] 在低并发下测 8192、32768、65536 上下文目标；显存不足或模型限制应如实记录。
- [ ] 再选一个可运行长上下文点，测试低并发与 E2 的代表并发；本轮暂未执行该补充项。
- [x] 标记 KV cache 是否统一、slot 数、实际 prompt token 和 HBM 峰值。

本轮服务配置：固定 `--parallel 1`；build 支持 `--kv-unified-per-slot` 时使用 `--kv-unified-per-slot 65536`，否则使用 `-c 65536`。因此本轮是单 Slot 的长上下文容量测试，不是高并发长上下文测试。

E3 结果：3 个目标上下文 × 单并发，共 24/24 请求成功，失败率 0%；脚本 v0.2.2，301 个 HBM 样本均有效。

| 目标上下文 | 实际 Prompt | TTFT | P50 / P95 | completion 吞吐 | HBM 基线 → 峰值 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 5,507 | 89.0 ms | 6.757 / 6.780 s | 37.790 tok/s | 79.239 → 79.262 GiB |
| 32768 | 21,740 | 123.0 ms | 7.079 / 7.091 s | 36.048 tok/s | 79.239 → 79.262 GiB |
| 65536 | 43,385 | 186.3 ms | 7.513 / 7.527 s | 33.925 tok/s | 79.239 → 79.262 GiB |

随着实际 Prompt 从 5,507 增加到 43,385 Token，TTFT 从 89.0 ms 增至 186.3 ms，completion 吞吐从 37.790 降至 33.925 tok/s，P95 从 6.780 s 增至 7.527 s；说明长上下文带来了可观测但可控的 Prefill/调度开销。HBM 稳定在约 79.24 GiB、单元增量约 23 MiB，符合 65K Slot 的 KV Cache 可能在服务启动时预留的现象，不能据此判断 KV Cache 没有占用显存。目标上下文是工作负载目标，实际 Token 以 API 返回的 `usage` 为准，因此 65536 目标对应 43,385 个实际 Prompt Token 不构成异常。

结果文件：[E3 脱敏结果（含 HBM）](../../results/qwen3.8-flash-next-ud-q4-k-xl-llamacpp-e3.json)

目的：区分“模型能加载”与“长上下文可以稳定服务”，并定位 HBM 是被权重、KV cache 还是并发挤占。

### E4 — 量化权衡

- [ ] 仅在下载额外量化版本后执行；首选同一模型族的相邻量化级别。
- [ ] 复用 E1 工作负载，补充一个固定小型质量提示集。
- [ ] 比较质量、TTFT、吞吐、HBM 与加载时间，不以单一 tokens/s 决定结论。

目的：得到针对单张 192GB MI300X 的量化选择建议，而不是泛化为所有硬件的结论。

### 后续对照顺序

1. 完成 E1–E4 的 llama.cpp 路线；
2. 准备模型版本、精度和上下文尽量可比的原生权重，接入 vLLM；
3. 当连续批处理、前缀缓存或高并发调度成为实际瓶颈时，再引入 SGLang；
4. 视觉模型部署、视频模型和微调另开实验卡，不与本次文本 serving 基准混合。

## 状态

- [x] 环境自检
- [x] 下载 UD-Q4_K_XL
- [x] 编译 ROCm llama.cpp
- [x] 文本冒烟
- [x] API、Web UI 与 DSW 登录态网关实际对话
- [x] E1 单请求基线
- [x] E2 并发 / 槽位
- [ ] E3 长上下文 / KV cache（低并发已完成，代表并发补充待做）
- [ ] E4 量化权衡
- [ ] vLLM 对照
- [ ] SGLang 对照（按需）
