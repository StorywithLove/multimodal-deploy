# AMD 实例上的虚拟环境与推理框架

参考：[Power Google ADK on AMD platform and local LLMs](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.html) —— AI Developer Hub 原版教程，比 5061 模板页多一格 `pip install -r requirements/requirements.txt`，且用 `%pip` 才装进 notebook 的 kernel 环境。

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

这台机器是 k8s 单容器。`/` 是 overlay，**只有 `/workspace` 是挂载的持久卷**，但**规格不固定**：上一台是 `/dev/loop5` ext4 98 GB，当前这台是 NFS 卷 25 GB（`nfs-ws-25.amd-oneclick-storage.svc.cluster.local`）。规划时按 25 GB 算，实际以 `df -h /workspace` 为准。

| 路径 | 位置 | 重启后 |
|---|---|---|
| `/workspace/**` | 持久卷 | ✅ 保留 |
| `/opt/venv`（python + uv + jupyter） | overlay | ❌ 清空，但镜像会重建 |
| `/root/tmp/**`（模型、日志） | overlay | ❌ 丢失 |
| `/root/.cache/uv`、`/root/.cache/vllm` | overlay | ❌ 丢失（缓存，可重建） |
| `/root/.ssh/authorized_keys` | overlay | ⚠️ 依赖平台重新注入 |

结论：**能重建的东西（venv、日志、缓存）放 `/workspace`；大文件模型反而放 `/root/tmp`**——19 GB 模型丢了 5 分钟能从 hf-mirror 重下，而 2.2 GB 的 ROCm wheel 丢了很难补。

### 2.1 持久卷上的结构

**一个框架一个目录，各自闭环**；根目录只放框架目录，加第三个框架时就是再建一个目录，不往上堆散件。

```
/workspace/agents_201/
├── vllm/                  # 脚本、wheel、env、日志都在这里，互不越界
│   ├── start.sh           # 自定位：所有路径相对脚本自身，整个目录可随意搬走
│   ├── wheels/  2.2 G     # 不可重下（出口挡死了）
│   ├── env/     8.2 G     # 可重建（离线 wheel，冷缓存 3–5 分钟）
│   └── logs/agent-vllm.log
└── ollama/                # 2.7 G，同样自闭环
    ├── bin/ollama         # 42 M，不可重下（ollama.com 被出口代理拒绝）
    ├── lib/ollama/ 2.6 G  # ggml-hip 后端 + ROCm 7.2 运行时，随二进制走
    ├── start.sh           # 自定位；export OLLAMA_MODELS 后起 serve
    ├── Modelfile.qwen3.5  # 纯文本，几百字节
    └── logs/ollama.log
```

为什么不按角色分层（`bin/ wheels/ env/ models/`）：那样每加一个框架都要在每一层插东西，`wheels` 还容易撞名，脚本反而得猜自己该装哪个。按框架分，目录本身承担语义，脚本名可以从 `start-agent-vllm.sh` 简化成 `start.sh`。

**模型不进这里。** `Qwen3.5-9B` 19 GB 在 `/root/tmp/agents-201/models/`，GGUF 同理。持久卷只有 25 GB，装不下也不值得装。

重启后核对：

```bash
ls -1 /workspace/agents_201/                          # 期望只有 vllm/、ollama/
cd /workspace/agents_201/vllm/wheels && sha256sum -c --quiet SHA256SUMS.txt
ls /workspace/agents_201/vllm/env/bin/vllm            # 不在 → 跑 env 步骤重建
ls -l /workspace/agents_201/ollama/bin/ollama         # 不在 → 重传解压，见第 4 节
ls /root/tmp/agents-201/models/Qwen3.5-9B             # 期望不在 → 跑 models 步骤重下
ls /root/tmp/agents-201/ollama-models/gguf            # 期望不在 → 重下 GGUF + ollama create
tmux ls                                               # 期望为空，tmux 会话不保留
```

### 2.2 两种框架的安装哲学

vLLM 是推理引擎，靠 PyTorch/ROCm 生态和 `torch.compile` 换吞吐，所以「安装」必然会卷进依赖解析；Ollama 是本地运行时，强调单文件分发和零依赖，用 GGUF 量化换小体积和快启动。结果就是 vLLM 的环境可重建但路径锁死，Ollama 的目录不可重建但可以搬。

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

上传 `wheels/` 和 `start-agent-vllm.sh`（远端落地为 `vllm/start.sh`）后，下面这套是脚本内部做的事，手动跑一遍便于定位问题：

```bash
V=/workspace/agents_201/vllm          # 脚本根目录

# 0. 换源——不设这个，装 transformers 会卡在 43 KB/s
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
export PIP_INDEX_URL=$UV_DEFAULT_INDEX

# 1. 校验 wheel 完整性（2.2G 约 10 秒）
cd $V/wheels && sha256sum -c SHA256SUMS.txt

# 2. 建 venv —— 原脚本缺的一步
#    home=/usr/bin、include-system-site-packages=true
/usr/bin/python3.12 -m venv --system-site-packages $V/env

# 3. 装 vLLM（离线 wheel；热缓存约 6 秒，冷缓存 3–5 分钟）
/opt/venv/bin/uv pip install --python $V/env/bin/python \
  --link-mode copy --find-links $V/wheels \
  $V/wheels/vllm-0.20.2rc1.dev15+g321fa2d6d.rocm721-cp312-cp312-manylinux_2_34_x86_64.whl \
  'transformers>=5.6,<6'
/opt/venv/bin/uv pip check --python $V/env/bin/python

# 4. 下模型（19G，hf-mirror 全速，约 5 分钟）—— 注意在易失区
HF_ENDPOINT=https://hf-mirror.com /opt/venv/bin/hf download Qwen/Qwen3.5-9B \
  --local-dir /root/tmp/agents-201/models/Qwen3.5-9B

# 5. 启动（前台；放 tmux，别放 Jupyter 里）
bash $V/start.sh serve
```

`start.sh` 把这些步骤串好了，支持分步执行：

```bash
bash $V/start.sh models   # 只在模型缺失时下载
bash $V/start.sh env      # 校验 wheel → 建 venv → 装 vllm → 自检
bash $V/start.sh serve    # 启动服务
bash $V/start.sh all      # 默认
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
2. **venv 不会自己出现。** 原脚本直接往 `env/bin/python` 里装，默认它已存在。重建时必须先 `python3.12 -m venv`。
3. **`--link-mode copy` 不能省。** uv 默认 hardlink，而 uv 缓存在 `/root/.cache`、venv 在 `/workspace`，跨设备会直接报错。
4. **不要直连 PyPI。** 不设 `UV_DEFAULT_INDEX` 会挂在 `files.pythonhosted.org`，表现是「卡住」而不是报错，用 `ps -eo pid,etime,time,args | grep uv` 看 CPU 时间不涨即可确认是限速不是死锁。
5. **默认思考模式会吃光 token。** `--reasoning-parser qwen3` 生效后，模型默认长思考，`max_tokens` 给小了会出现「请求成功但 `content` 为空」，内容全在 `reasoning` 字段里。Agent 调用时显式传 `chat_template_kwargs: {"enable_thinking": false}`，或把 `max_tokens` 提到 1000 以上。
6. **`UPLOAD_VERIFIED` 标记已废弃。** 原来靠 `touch` 一个空文件告诉脚本「上传完了」，但文件传一半也能 `touch`。现在改为 `sha256sum -c` 校验，上传完成与否由哈希说了算。
7. **显存还能再放大。** 0.6 只用了 29 GB，48 GB 卡上可以提到 0.85 左右换取更长上下文和更多并发；代价是开别的进程容易 OOM。
8. **venv 不能改名，也不能搬目录。** `bin/` 下 79 个脚本的 shebang、`bin/activate` 里的 `VIRTUAL_ENV` 都写死绝对路径；`site-packages` 里 7134 个 `.pyc` 也记着旧路径（只影响 traceback 显示）。`.pth` 文件为 0，所以没有功能性坏路径，但也改不干净。**换位置就重建，别 `mv`**——离线 wheel 装一遍只要几分钟。`pyvenv.cfg` 里是 `home = /usr/bin`，与安装位置无关，不用管。
9. **续传脚本不能用来同步「改过内容」的文件。** `sync-to-amd.sh` 早先按远端已有字节数续传，隐含假设「远端已有的字节是新文件的合法前缀」——这对传输中断成立，对改过的文件不成立：6.5 KB 的新脚本被接到了 6.2 KB 的旧脚本屁股上，得到一个 `bash -n` 都通不过的混血文件。现在 <1 MiB 的文件一律整体重写，且所有文件传完都比对 sha256，不一致就清空重传。

---

## 4. Ollama（0.40.13 / ROCm 7.2）

> **状态（2026-11-02）：已装好并在跑。** 二进制和运行时在持久卷（2.7 GB），GGUF 与 blob 在 `/root/tmp`（重启即失，约 2 分钟可重建）。

`ollama.com` 被出口代理直接拒绝，所以二进制只能从本地备份传上去；实测上传速度比 [§5](#5-换实例后同步-wheels) 记的快一个量级。

### 4.1 实测结果

| 项 | 结果 |
|---|---|
| 版本 | `0.40.13`（client/server 同版本） |
| GPU | `ROCm0 / gfx1100`，`total=48.0 GiB`，启动时 `available=16.1 GiB`（vLLM 已占 30.9 GB） |
| 模型 | `qwen35` = Qwen3.5-9B Q4_K_M，5.68 GB，`ollama ps` 显示 **100% GPU**、ctx 8192 |
| 加载 | 首次 2.57 s（GGUF 在页缓存）；驻留后 0.8–0.9 s |
| 生成 | **44–51 tok/s**（36–128 token 的短请求，`think:false`） |
| Function Calling | ✅ `get_weather({"city":"北京"})`，26 token / 1.25 s |
| 显存 | vLLM 30.9 GB → 加载后 **37.0 / 51.5 GB**，Ollama 实占约 6.1 GB，余 14.5 GB |
| 端口 | 11434（vLLM 的 8088 不冲突） |

### 4.2 落位与体积

```
/workspace/agents_201/ollama/       # 2.7 GB
├── bin/ollama             42 M     # 不可重下（ollama.com 被出口代理拒绝）
├── lib/ollama/            2.6 G    # libggml-hip.so 838 M + ROCm 7.2 运行时
├── start.sh              1.5 K     # 自定位；export OLLAMA_MODELS 后起 serve
├── Modelfile.qwen3.5      306 B    # 纯文本
├── extract.py             777 B    # 解 .tar.zst（镜像里没有 zstd 命令）
└── logs/ollama.log
```

官方 base 包解出来有 5.0 GB，其中 `cuda_v12`（1.3 G）和 `cuda_v13`（1.1 G）在 AMD 机器上永远用不到，**已删**；重装时要再删一次，否则白占 2.4 GB 持久卷。GPU 后端 `lib/ollama/rocm_v7_2/libggml-hip.so` 和 ROCm 运行时放在同一目录，Ollama 启动时会自动把 `lib/ollama` 与 `lib/ollama/rocm_v7_2` 都加进 `OLLAMA_LIBRARY_PATH`，不用手配。

### 4.3 从零重建

本地备份在 `D:\用户资料\AI模型\Ollama\Ollama-rocm-0.40.13\`：

| 文件 | 大小 | 说明 |
|---|---|---|
| `ollama-linux-amd64.tar.zst` | 1.53 G | `bin/ollama` + CPU/Vulkan 后端 |
| `ollama-linux-amd64-rocm.tar.zst` | 1.05 G | `lib/ollama/rocm_v7_2/`，含 `libggml-hip.so` |
| `sync-to-amd.sh` | 2.8 K | 断点续传，逻辑同 [§5](#5-换实例后同步-wheels) |
| `extract.py` / `start.sh` / `Modelfile.qwen3.5` | 2.6 K | |

**① 上传**（2.58 GB，实测 **4 分 27 秒**）：

```bash
cd "D:/用户资料/AI模型/Ollama/Ollama-rocm-0.40.13"
bash sync-to-amd.sh          # 归档 → ollama/.archive；三个小文件 → ollama/ 根目录
```

脚本同时把 `extract.py`、`start.sh`、`Modelfile.qwen3.5` 传到框架根目录，所以下一步解压时它们已经在了。

分文件：base 1.53 G / 117 s = **13.0 MB/s**，rocm 1.05 G / 142 s = **7.4 MB/s**。这次走的是**办公室网络**；vLLM 那次是**家用手机热点**，只有 0.9–1.7 MB/s（见 [§5](#5-换实例后同步-wheels)）。差距来自出口网络，不是实例——**规划按手上网络的实际值算，开传 30 秒再估**。

**② 解压**——镜像里没有 `zstd`，GNU tar 的 `--zstd` 也只是调用外部程序的过滤器：

```bash
/opt/venv/bin/uv pip install --python /opt/venv/bin/python zstandard \
  -i https://mirrors.aliyun.com/pypi/simple/
O=/workspace/agents_201/ollama
/opt/venv/bin/python $O/extract.py $O/.archive/ollama-linux-amd64.tar.zst $O
/opt/venv/bin/python $O/extract.py $O/.archive/ollama-linux-amd64-rocm.tar.zst $O
```

`extract.py` 用 `tarfile` 流式读 `zstandard` 的解压流，`filter='tar'` 会拒绝绝对路径和指向目录外的链接。

**③ 补执行位（必做）**——Python 解出来的文件一律是 `644`，`llama-server` 没有 x 位，Ollama 的 GPU 探测直接 `permission denied`，然后**静默退化成 CPU**：服务照常起、API 照常回，日志里只有一行 INFO `failure during llama-server GPU discovery ... permission denied`，`inference compute` 只剩 `id=cpu`。

`start.sh check` 会在启动前验这三个文件的 x 位和 `libggml-hip.so` 是否存在，不合格直接退出（实测正常 `exit=0`，丢失 x 位 `exit=1`）。

```bash
chmod 755 $O/bin/ollama $O/lib/ollama/llama-server $O/lib/ollama/llama-quantize
rm -rf $O/.archive $O/lib/ollama/cuda_v12 $O/lib/ollama/cuda_v13
```

**④ 下 GGUF**（5.68 GB，hf-mirror 全速约 1 分钟）：

```bash
HF_ENDPOINT=https://hf-mirror.com /opt/venv/bin/hf download \
  unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf \
  --local-dir /root/tmp/agents-201/ollama-models/gguf
```

**⑤ 启动**（tmux，别放 Jupyter）：

```bash
tmux new -d -s ollama 'bash /workspace/agents_201/ollama/start.sh serve'
```

**⑥ 注册模型**（13 秒；会把 GGUF 复制成 blob，`/root/tmp` 因此占 11 GB）：

```bash
export OLLAMA_HOST=127.0.0.1:11434
$O/bin/ollama create qwen35 -f $O/Modelfile.qwen3.5
```

### 4.4 服务参数与共卡

`start.sh` 只做三件事：定位自身、设环境变量、`exec bin/ollama serve`。关键变量：

| 变量 | 值 | 原因 |
|---|---|---|
| `OLLAMA_MODELS` | `/root/tmp/agents-201/ollama-models` | 默认 `/root/.ollama` 在 overlay 上；11 GB 不该占 25 GB 持久卷 |
| `OLLAMA_HOST` | `127.0.0.1:11434` | 走 SSH 转发即可，不对外 |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | 与 vLLM 共卡，只常驻一个 |
| `OLLAMA_NUM_PARALLEL` | `2` | 每路都要一份 KV cache |
| `OLLAMA_KEEP_ALIVE` | `5m` | 空闲 5 分钟释放 6.1 GB，给 vLLM 让路 |

`Modelfile` 只有 `FROM` + 采样参数 + `num_ctx 8192`：**不需要写 `TEMPLATE`**，`ollama show` 显示 `template source = gguf_chat_template`，模板已经在 GGUF 元数据里；`num_ctx` 对齐 vLLM，避免两套上下文各吃一份显存。

### 4.5 接口实测

```bash
# 生成（think:false 才会直接给答案）
curl -s --noproxy '*' http://127.0.0.1:11434/api/generate \
  -d '{"model":"qwen35","prompt":"你好","stream":false,"think":false,
       "options":{"num_predict":64}}'

# 对话
curl -s --noproxy '*' http://127.0.0.1:11434/api/chat \
  -d '{"model":"qwen35","messages":[{"role":"user","content":"用一句话说明什么是 KV cache"}],
       "stream":false,"think":false,"options":{"num_predict":80}}'
```

| 测试 | 结果 |
|---|---|
| `GET /api/version` | `{"version":"0.40.13"}` |
| 生成（默认思考） | 128 token 全在 thinking 里，`done_reason: length`，正文为空 |
| 生成（`think:false`） | 43 token / 0.98 s = **43.9 tok/s** |
| 对话（`think:false`） | 36 token / 0.75 s = **47.8 tok/s** |
| Function Calling | `tool_calls: get_weather({"city":"北京"})`，1.25 s |
| `ollama ps` | `qwen35  5.7 GB  100% GPU  8192  llamacpp` |

`think` 是 Ollama 0.40 的顶层字段（`ollama show` 的 Capabilities 里有 `thinking`）；不给就按模板默认长思考，和 vLLM 侧 [3.7](#37-踩坑) 第 5 条是同一个坑。

### 4.6 踩坑

1. **`ollama.com` 被代理拒绝是预期内的。** 启动日志有两条 `WARN`（`model-recommendations` / `model show cloud cache hydration`，`Forbidden`），每 5 分钟重试一次；不影响本地推理，忽略即可。
2. **解压丢执行位会静默降级成 CPU。** 见 4.3 ③——服务能起、API 能回，只是 `inference compute` 里没有 GPU、`total_vram=0 B`。先看 `ollama ps` 的 PROCESSOR 列是不是 `100% GPU`。
3. **`/root/tmp` 里有两份 5.7 GB。** GGUF 原件 + `ollama create` 复制出的 blob = 11 GB。原件删掉也能跑（blob 才是模型本体），但改参数重建就得重下；hf-mirror 1 分钟，留着更省事。
4. **CUDA 后端要手动删。** base 包同时带 `cuda_v12` / `cuda_v13`，共 2.4 GB，AMD 机器上纯浪费持久卷。
5. **模型默认长思考。** 和 vLLM 一样，`num_predict` 给小了会出现「请求成功但正文为空」。Ollama 侧用顶层 `"think": false`。
6. **重启后 tmux 和模型都没了。** 二进制在持久卷，`tmux ls` 为空；`/root/tmp` 清空意味着 GGUF 和 blob 都没了。恢复顺序：`start.sh serve` → 重下 GGUF → `ollama create`，约 2 分钟。

---

## 5. 换实例后同步 wheels

实例重建（不是重启）会拿到一个**全新实例**：hostname、SSH 端口、主机密钥全变，`/workspace` 是空的新卷，旧卷不跟随。这时 wheels 得重新上传。

**本地上行取决于出口网络，不是这台机器**：

| 网络 | 方式 | 实测 |
|---|---|---|
| 家用手机热点 | 单路 `ssh` 传 30 MB | 34 s → **882 KB/s** |
| 家用手机热点 | 3 路并行传 30 MB | 33 s → **909 KB/s** |
| 办公室网络 | 传 Ollama 两个归档 2.58 GB | 267 s → **9.7 MB/s**（单文件 7.4–13.0，见 [4.3](#43-从零重建)） |

手机热点下多开连接没有意义（882 → 909 KB/s），瓶颈在热点上行；办公室网络快一个量级。**规划按你当下所在网络的值算，别把 882 KB/s 当成机器上限。** 手机热点下 2.2 GB 满速约 **42 分钟**，办公室网络约 4 分钟。

vLLM 的 wheels 是在手机热点下传的：实际 2.34 GB 用了 **23 分钟**（16:05→16:28 UTC，均速约 1.7 MB/s），比估算快，因为 `sync-to-amd.sh` 从断点 157 MB 处接着传，只补了剩余的 1.79 GB。

**所以不能用 `scp`**：它没有断点续传，一次 `Connection reset by peer` 就得从 0 重来（实测传到 152.7 MB 时被重置，前功尽弃）。改用本地备份目录里的 `sync-to-amd.sh`，按远端已有字节数续传：

```bash
cd "D:/用户资料/AI模型/vLLM/vLLM-rocm-7.2.1"
bash sync-to-amd.sh                        # 默认 → radeon-gpu:/workspace/agents_201/vllm
DEST=/workspace/other bash sync-to-amd.sh  # 换目标目录
```

wheels 用 `tail -c +N | ssh 'cat >>'` 按远端已有字节数续传；**小于 1 MiB 的脚本一律整体重写**（续传对改过内容的文件会拼出混血文件，见 [3.7](#37-踩坑) 第 9 条）。每个文件最多重试 20 次，传完逐个比对 sha256，不一致就清空重传，最后再跑一遍 `sha256sum -c`。中途断了直接重跑同一条命令。

本地文件名是 `start-agent-vllm.sh`，同步到远端叫 `$DEST/start.sh`——框架目录已经承担了语义，不必在文件名里再重复一遍。

**容量注意**：新实例的 `/workspace` 是 25 GB 的 NFS 卷（上一台是 98 GB 的 loop 卷，规格不固定）。wheels 2.2 GB + venv 8.2 GB + Ollama 2.7 GB = 13.1 GB 放得下（实测剩余约 12 GB）；19 GB 的 safetensors 模型和 11 GB 的 GGUF/blob 放不下，必须留在 `/root/tmp`——脚本的默认值正是这样分布的。

### 5.1 查询与关闭实例（rcc CLI）

`rcc` 装在本机 conda 的 `stock` 环境里，不在默认 PATH：

```powershell
conda activate stock
rcc instance list                 # 显示当前实例 ID，如 rgapi-one-click-25248-a539cf14
rcc instance status               # 就绪状态
rcc instance delete --yes         # 删除当前实例，不可恢复
```

实例 ID、启动时间可以从两个地方确认：`rcc instance list`，或实例内 `ps -o lstart= -p 1`（Jupyter 起于容器启动时，时间基本一致）。按小时计费时以这个时刻起算。

`/workspace` 由平台持久化：删除实例后重新启动，里面的 wheels 等文件仍在，不用重传。但 overlay 上的 venv、模型、`/root/.cache` 会随容器重置。即便如此，本地那份 wheels 备份仍要保留——它不依赖平台，也不占用实例费用。

## 6. 备份清单

留在本地的是两个框架的「不可重下件」，合计 **4.8 GB**：

| 文件 | 大小 | 位置 |
|---|---|---|
| 9 个 `.whl` + `SHA256SUMS.txt` | 2.2 G | `D:\用户资料\AI模型\vLLM\vLLM-rocm-7.2.1\wheels\` |
| `start-agent-vllm.sh` / `sync-to-amd.sh` | 6.4 K / 2.8 K | `D:\用户资料\AI模型\vLLM\vLLM-rocm-7.2.1\` |
| `ollama-linux-amd64.tar.zst` | 1.53 G | `D:\用户资料\AI模型\Ollama\Ollama-rocm-0.40.13\` |
| `ollama-linux-amd64-rocm.tar.zst` | 1.05 G | 同上 |
| `sync-to-amd.sh` / `extract.py` / `start.sh` / `Modelfile.qwen3.5` | 约 5 K | 同上 |

不用备份：模型（hf-mirror 5 分钟可重下，GGUF 5.7 GB 约 1 分钟）、venv（离线 wheel 重建，冷缓存 3–5 分钟）、`/opt/venv`（镜像自带）、各种缓存。

两份脚本都**没有硬编码根目录**：`start-agent-vllm.sh` 以自身所在目录为锚，`sync-to-amd.sh` 以本地目录为锚，远端路径由 `DEST` 决定。换机器、换实例、改目录结构都只改环境变量，不改代码。

> 将来若 vLLM 升级到配套 torch ≥ 2.11，torch 三件套可改为走 `download.pytorch.org` 在线装，备份体积能从 2.2 GB 降到约 700 MB（见 [3.6](#36-能不能在线装实测不能)）。
