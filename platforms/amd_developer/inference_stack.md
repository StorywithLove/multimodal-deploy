# AMD 实例上的虚拟环境与推理框架

记录 Radeon Cloud「一键实例」上推理框架（vLLM、Ollama）的安装与重建方式。
核心原则：**框架装在独立 venv 里，用本地 wheel 离线重建**——这台机器的出口只有平台白名单代理，`files.pythonhosted.org` 被限速到 43 KB/s，联网装 vLLM 基本等于装不上。

网络侧的问题见 [remote-ssh/README.md](remote-ssh/README.md)。

---

## 1. 环境基线

| 项 | 值 | 来源 |
|---|---|---|
| 系统 | Ubuntu 24.04.4 LTS（容器，非 VM） | 镜像 |
| GPU | gfx1100 / Navi31 XTW-C GDDR6 **48 GB** | 实例规格 |
| 系统内存 | 256 GB（`TOTAL_GTT: 257824 MB`） | 镜像 |
| Python | 3.12.3（`/usr/bin/python3.12`） | 镜像 |
| 包管理器 | uv 0.10.9（`/opt/venv/bin/uv`） | 镜像 |
| ROCm | 7.2.5（`torch.version.hip = 7.2.53211`） | wheel 自带，无需单独装驱动 |
| Jupyter | `PID 1`，`/opt/venv/bin/jupyter-lab` | 镜像 |

注意 ROCm 是**随 torch wheel 打包**的，不需要 `apt install rocm`，也不该装。

---

## 2. 目录约定：什么能挺过重启

这台机器是 k8s 单容器。`/` 是 overlay，**只有 `/workspace` 是挂载的 ext4 持久卷**（`/dev/loop5`，98 GB）。

| 路径 | 位置 | 重启后 |
|---|---|---|
| `/workspace/**` | 持久卷 | ✅ 保留 |
| `/opt/venv`（python + uv + jupyter） | overlay | ❌ 清空，但镜像会重建 |
| `/root/tmp/**`（模型、日志） | overlay | ❌ 丢失 |
| `/root/.cache/uv`、`/root/.cache/vllm` | overlay | ❌ 丢失（缓存，可重建） |
| `/root/.ssh/authorized_keys` | overlay | ⚠️ 依赖平台重新注入 |

结论：**能重建的东西（venv、日志、缓存）放 `/workspace`；大文件模型反而放 `/root/tmp`**——19 GB 模型丢了 5 分钟能从 hf-mirror 重下，而 2.2 GB 的 ROCm wheel 丢了很难补。

重启后核对：

```bash
ls -d /workspace/agents-201/{wheels,vllm-rocm-env}   # 期望两个都在
cd /workspace/agents-201/wheels && sha256sum -c --quiet SHA256SUMS.txt
ls -d /root/tmp/agents-201/models/Qwen3.5-9B         # 期望不在 → 跑 models 步骤重下
tmux ls                                              # 期望为空，tmux 会话不保留
```

---

## 3. vLLM（Qwen3.5-9B / ROCm 7.2）

### 3.1 wheel 清单

9 个 `.whl`，合计 2.2 GB。全部来自本地备份目录
`D:\用户资料\AI模型\vLLM\vLLM-rocm-7.2.1\wheels`，**不要试图联网重下**。

| 文件 | 大小 | 说明 |
|---|---|---|
| `torch-2.10.0+git8514f05-cp312-cp312-manylinux_2_35_x86_64.whl` | 1.3 G | ROCm 版 torch，含 HIP 运行时 |
| `triton-3.6.0-cp312-cp312-manylinux_2_35_x86_64.whl` | 327 M | |
| `vllm-0.20.2rc1.dev15+g321fa2d6d.rocm721-cp312-cp312-manylinux_2_34_x86_64.whl` | 293 M | 主包，源码构建的 ROCm 版 |
| `amd_aiter-0.1.10.post3-cp312-cp312-manylinux_2_34_x86_64.whl` | 251 M | AMD 算子库 |
| `flash_attn-2.8.3-cp312-cp312-manylinux_2_34_x86_64.whl` | 113 M | |
| `torchvision-0.24.1+d801a34-cp312-cp312-manylinux_2_34_x86_64.whl` | 1.2 M | |
| `amdsmi-26.2.2+e1a6bc5663-py3-none-any.whl` | 1.1 M | 显存监控 |
| `torchaudio-2.9.0+eaa9e4e-cp312-cp312-manylinux_2_34_x86_64.whl` | 481 K | |
| `triton_kernels-1.0.0-py3-none-any.whl` | 96 K | |
| `SHA256SUMS.txt` | 1.1 K | 校验清单，**必须是 LF 换行** |

其余依赖（`transformers 5.22.1` 等 200 多个包）由阿里云源补齐，这部分联网没问题。

### 3.2 从零重建

上传 `wheels/` 和 `start-agent-vllm.sh` 到实例后：

```bash
# 0. 换源——不设这个，装 transformers 会卡在 43 KB/s
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export PIP_INDEX_URL=$UV_DEFAULT_INDEX

# 1. 校验 wheel 完整性（2.2G 约 10 秒）
cd /workspace/agents-201/wheels && sha256sum -c SHA256SUMS.txt

# 2. 建 venv —— 原脚本缺的一步
#    home=/usr/bin、include-system-site-packages=true，与镜像原有环境保持一致
/usr/bin/python3.12 -m venv --system-site-packages /workspace/agents-201/vllm-rocm-env

# 3. 装 vLLM（离线 wheel，命中缓存约 6 秒）
/opt/venv/bin/uv pip install --python /workspace/agents-201/vllm-rocm-env/bin/python \
  --link-mode copy --find-links /workspace/agents-201/wheels \
  /workspace/agents-201/wheels/vllm-0.20.2rc1.dev15+g321fa2d6d.rocm721-cp312-cp312-manylinux_2_34_x86_64.whl \
  'transformers>=5.6,<6'
/opt/venv/bin/uv pip check --python /workspace/agents-201/vllm-rocm-env/bin/python

# 4. 下模型（19G，hf-mirror 全速，约 5 分钟）
HF_ENDPOINT=https://hf-mirror.com /opt/venv/bin/hf download Qwen/Qwen3.5-9B \
  --local-dir /root/tmp/agents-201/models/Qwen3.5-9B

# 5. 启动（前台；建议放 tmux）
bash /workspace/agents-201/start-agent-vllm.sh
```

`start-agent-vllm.sh` 已把这些步骤串起来，支持分步执行：

```bash
bash start-agent-vllm.sh models   # 只在模型缺失时下载
bash start-agent-vllm.sh env      # 校验 wheel → 建 venv → 装 vllm → 自检
bash start-agent-vllm.sh serve    # 启动服务
bash start-agent-vllm.sh all      # 默认
```

`env` 步骤是幂等的，重复跑只会重新校验和 `uv pip install`（已装则秒过）。
自检会断言 `torch.version.hip` 存在、`torch.version.cuda is None`、`gcnArchName` 以 `gfx1100` 开头——不满足直接退出，不会带着半残的环境去起服务。

### 3.3 服务参数

```bash
vllm serve /root/tmp/agents-201/models/Qwen3.5-9B \
  --served-model-name qwen35 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-model-len 8192 --port 8088 --gpu-memory-utilization 0.6
```

### 3.4 实测：启动耗时约 5 分钟

| 时刻 | 阶段 | 耗时 |
|---|---|---|
| 10:51:14 | 脚本启动，`uv pip install` 命中缓存 | ~6 s |
| 10:51:54 | `import torch + vllm` 完成 | ~40 s |
| 10:52:04 | 解析结构 `Qwen3_5ForConditionalGeneration` | ~10 s |
| 10:52:10 → 10:56:02 | **引擎初始化 198.6 s**（profile + KV cache + warmup），其中 `torch.compile` 70.97 s | 198.6 s |
| 10:56:11 | `Application startup complete` | **合计 297 s** |

大头是引擎初始化。`torch.compile` 即使命中 `/root/.cache/vllm/torch_compile_cache` 也要 71 秒；冷缓存会更久。

### 3.5 实测：接口

| 测试 | 结果 |
|---|---|
| `GET /health` | 200，0.8–1.1 ms |
| `GET /v1/models` | `qwen35`，`max_model_len: 8192` |
| 对话（默认思考） | 200，但 512 token 全被 thinking 吃光，`content: None` |
| 对话（`enable_thinking:false`） | 42 token / 0.96 s |
| Function Calling | `finish_reason: tool_calls` → `get_weather({"city":"北京"})`，1.58 s |
| 吞吐（单流） | 512 token / 11.86 s = **43.2 tok/s** |
| 显存 | 29.2 / 49.1 GB（59.5%，与 `--gpu-memory-utilization 0.6` 吻合） |

```bash
curl -s --noproxy '*' http://127.0.0.1:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen35","messages":[{"role":"user","content":"你好"}],"max_tokens":64,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

`--noproxy '*'` 不能漏：容器里注入了 `HTTP_PROXY`，不加的话 `curl` 会绕去平台代理。

### 3.6 能不能在线装？——实测：不能

换源只解决「下载速度」，不解决「源上有没有这个包」。这 9 个是 AMD / vLLM 的 ROCm 专用构建，只发布在两个源上，而这两个都被平台代理拦死：

| 源 | 实测结果 |
|---|---|
| `wheels.vllm.ai/rocm/` | ❌ `CONNECT tunnel failed, response 403` |
| `rocm.nightlies.amd.com/whl/` | ❌ `CONNECT tunnel failed, response 403` |
| `download.pytorch.org/whl/rocm7.2/` | ✅ 通，**11.9 MB/s** |

国内镜像上虽然有同名的包，但版本对不上，也没有 `+rocm` 本地版本标签：

| 包 | 本地（可用） | 阿里云源上 |
|---|---|---|
| vllm | 0.20.2rc1.dev15+**rocm721** | 0.20.2 通用版；`uv` 解析出它要求 **torch 2.11.0**（PyPI CUDA 版） |
| torch | 2.10.0+**git8514f05** | 2.10.0 / 2.11.0，无 `+rocm` 标签 |
| amd-aiter | 0.1.10.post3 | 0.1.7.post2.dev18 |
| amdsmi | 26.2.2 | 7.0.x |
| torchvision | 0.24.1+d801a34 | 0.29.2 |
| torchaudio | 2.9.0+eaa9e4e | 2.11.1 |
| triton-kernels | 1.0.0 | 0.1.0 |
| flash-attn | 2.8.3 | 无 wheel |

装上 PyPI 的 torch 2.11.0 之后 `torch.version.cuda` 不为 `None`，vLLM 的 ROCm kernel 找不到配套 ABI，服务起不来。**所以这 2.2 GB 必须本地留底。**

**唯一的例外**：`download.pytorch.org` 没被墙，能跑满 11.9 MB/s，`rocm7.2` 索引里有 torch **2.11.0 / 2.12.x / 2.13.0 / 2.14.x**。将来若升级到配套 torch ≥ 2.11 的 vLLM 版本，**torch / torchvision / torchaudio 三件套可以改为在线装**，只剩 vllm / triton / amd-aiter 需要本地 wheel——本地备份能从 2.2 GB 降到约 700 MB。

### 3.7 踩坑

1. **清单文件不能是 CRLF。** 从 Windows 传上去的 `SHA256SUMS.txt` 带 `\r`，`sha256sum -c` 会把它当文件名的一部分，9 个 wheel 全部 `FAILED open or read`。修法：`tr -d '\r' < SHA256SUMS.txt > tmp && mv tmp SHA256SUMS.txt`。
2. **venv 不会自己出现。** 原脚本直接往 `vllm-rocm-env/bin/python` 里装，默认它已存在。重建时必须先 `python3.12 -m venv`。
3. **`--link-mode copy` 不能省。** uv 默认 hardlink，而 uv 缓存在 `/root/.cache`、venv 在 `/workspace`，跨设备会直接报错。
4. **不要直连 PyPI。** 不设 `UV_DEFAULT_INDEX` 会挂在 `files.pythonhosted.org`，表现是「卡住」而不是报错，用 `ps -eo pid,etime,time,args | grep uv` 看 CPU 时间不涨即可确认是限速不是死锁。
5. **默认思考模式会吃光 token。** `--reasoning-parser qwen3` 生效后，模型默认长思考，`max_tokens` 给小了会出现「请求成功但 `content` 为空」，内容全在 `reasoning` 字段里。Agent 调用时显式传 `chat_template_kwargs: {"enable_thinking": false}`，或把 `max_tokens` 提到 1000 以上。
6. **`UPLOAD_VERIFIED` 标记已废弃。** 原来靠 `touch` 一个空文件告诉脚本「上传完了」，但文件传一半也能 `touch`。现在改为 `sha256sum -c` 校验，上传完成与否由哈希说了算。
7. **显存还能再放大。** 0.6 只用了 29 GB，48 GB 卡上可以提到 0.85 左右换取更长上下文和更多并发；代价是开别的进程容易 OOM。

---

## 4. Ollama

> **待补（2026-11-02）**：记录 ollama 的安装方式（出口限制下如何获取二进制）、模型导入（`Qwen3.5.Modelfile` 转 GGUF）、以及和 vLLM 在 8088/11434 上的共存方式。
> 现有半成品：`/root/tmp/agents-201/Qwen3.5.Modelfile`、`models/Qwen3.5-9B-GGUF/`。

---

## 5. 备份清单

留在本地的只有两样，合计 **2.2 GB**：

| 文件 | 大小 | 位置 |
|---|---|---|
| 9 个 `.whl` + `SHA256SUMS.txt` | 2.2 G | `D:\用户资料\AI模型\vLLM\vLLM-rocm-7.2.1\wheels\` |
| `start-agent-vllm.sh` | 1.6 K | `D:\用户资料\AI模型\vLLM\vLLM-rocm-7.2.1\` |

不用备份：模型（hf-mirror 5 分钟可重下）、venv（wheel 重建，6 秒）、`/opt/venv`（镜像自带）、各种缓存。

> 将来若 vLLM 升级到配套 torch ≥ 2.11，torch 三件套可改为走 `download.pytorch.org` 在线装，备份体积能从 2.2 GB 降到约 700 MB（见 [3.6](#36-能不能在线装实测不能)）。
