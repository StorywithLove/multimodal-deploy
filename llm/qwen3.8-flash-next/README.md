# Qwen3.8-Flash-Next

本目录记录该模型在 AMD MI300X 192GB 上的部署路线、后端对比与实验结果。这里的结论只适用于本模型、当前硬件和记录的运行配置；语言模型分支的通用结构见 [llm/README.md](../README.md)。

## 模型文件存储规则

**大模型权重、GGUF 原始分片、Ollama 模型 blob 和权重缓存都不要写入 ModelScope Notebook 的 `/mnt/workspace` 持久化目录。** ModelScope Notebook 文档将该挂载说明为 100GB 持久化空间，并建议用 `du -sh /mnt/workspace` 查看已用空间；本项目的模型文件统一放在实例本地的 `/root/tmp` 下。该路径是实例本地临时空间，实例重建后需要重新准备模型文件。导入前先运行 `df -h /root/tmp /mnt/workspace`，确认 `/root/tmp` 有足够空间同时容纳源权重和后端副本；本路线的 Ollama 导入会额外复制约 104 GiB。文档链接：[ModelScope Notebook 存储说明](https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-04.md)。

`/mnt/workspace` 只用于项目代码、配置、日志和实验结果；不要把 `df` 在共享网络挂载上显示的池化容量当作个人配额。

## 部署路线

| 后端 | 权重格式 / 精度 | 状态 | 部署记录 |
|---|---|---|---|
| llama.cpp | Unsloth GGUF UD-Q4_K_XL、UD-Q5_K_XL | E1–E4 完成 | [llama.cpp.md](llama.cpp.md) |
| vLLM | 原生 FP8 Safetensors | E1–E4 完成 | [vllm.md](vllm.md) |
| SGLang | 同一份原生 FP8 Safetensors | E1–E4 完成；32K 使用 Triton attention backend | [sglang.md](sglang.md) |
| Ollama | UD-Q4_K_XL GGUF | 已迁移至 `/root/tmp`；E1–E4 完成（API ready 中位数 1.088 s，首次模型请求墙钟 82.471 s） | [ollama.md](ollama.md) |

已测路线使用单张 MI300X、ROCm 7.2.x、`gfx942`，并复用 `../../configs/benchmark/mi300x-serving-matrix.json` 和 `../../configs/benchmark/openai_server_matrix.py`。llama.cpp 使用 GGUF Q4/Q5，Ollama 使用同源 GGUF UD-Q4_K_XL，vLLM 与 SGLang 使用原生 FP8；跨格式结果代表**部署组合**，不能写成纯框架排名。

### Ollama 与 llama.cpp 的关系

本次 Ollama 路线导入与 llama.cpp 路线相同的 `UD-Q4_K_XL` GGUF；`ollama ps` 显示其运行时为 `llamacpp`。直接 llama.cpp 路线由 `llama-server` 读取 GGUF 并提供 API，Ollama 则在同一 runner 之上增加模型登记、blob 管理、服务生命周期、OpenAI-compatible API 和并发调度。因此两组结果应按“直接 llama.cpp 部署”与“Ollama 管理的 llama.cpp runner 部署”两条端到端路线解读，同时固定 GGUF、上下文、并发和请求配置。

## 短上下文吞吐

以下取同一工作负载矩阵中并发 1、每个测试单元 8 个请求的 completion 聚合吞吐。Ollama 数值来自 E1，其余为各路线 E1/E2 并发 1 单元。512/4096/8192 是目标上下文；实际 prompt 分别约 433/2,801/5,507 tokens。每请求最多生成 256 tokens。

| 后端 / 权重 | 512 tok/s | 4096 tok/s | 8192 tok/s |
|---|---:|---:|---:|
| llama.cpp / UD-Q4_K_XL | 38.58 | 35.52 | 34.81 |
| vLLM / FP8 | 35.91 | 34.50 | 35.20 |
| SGLang / FP8 | 37.63 | 35.17 | 35.16 |
| Ollama / UD-Q4_K_XL | 34.85 | 34.00 | 33.36 |

单并发时四条路线约在 33–39 tok/s，整体接近。Ollama E1 三档均 8/8 成功，吞吐随实际输入增长略降；这点差异不足以单独判断框架胜负。

## 并发扩展

吞吐单位为 tok/s。Ollama E2 在 max-parallel 8 配置下完成 12 个测试格，每格 8/8 成功。SGLang 本轮最高测到并发 2；其 C4/C8 保持“未测”，不推断框架并发上限。

### Ollama E2 并发矩阵

| 目标上下文 | C1 | C2 | C4 | C8 | C8 P95 延迟 |
|---|---:|---:|---:|---:|---:|
| 512 | 33.26 | 51.36 | 73.08 | 85.99 | 16.934 s |
| 4096 | 33.47 | 49.63 | 61.26 | 61.46 | 26.385 s |
| 8192 | 32.86 | 46.43 | 54.85 | 50.70 | 33.311 s |

Ollama 的 C8 在 512 档比 C4 快 17.7%；4096 档与 C4 基本持平（+0.3%）；8192 档比 C4 慢 7.6%，P95 从 19.370 s 增至 33.311 s。因此 C8 已稳定通过，但长上下文下吞吐/延迟更适合看 C4。12 格均为 8/8 成功；E2 采样到的最大整卡 HBM 为 81.168 GiB，基线包含已驻留模型。

| 目标上下文 | C2 llama.cpp Q4 | C2 vLLM FP8 | C2 SGLang FP8 | C2 Ollama UD-Q4_K_XL | C8 llama.cpp Q4 | C8 vLLM FP8 | C8 SGLang FP8 | C8 Ollama UD-Q4_K_XL |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 512 | 54.10 | 55.44 | 66.17 | 51.36 | 117.92 | 116.77 | 未测 | 85.99 |
| 4096 | 63.65 | 53.66 | 62.36 | 49.63 | 70.02 | 101.77 | 未测 | 61.46 |
| 8192 | 62.28 | 55.46 | 62.64 | 46.43 | 65.10 | 108.58 | 未测 | 50.70 |

C2 时 llama.cpp、vLLM、SGLang 的结果接近；Ollama GGUF 的 C2 吞吐略低。C8 时 vLLM 在 4096/8192 目标上下文达到 101.77/108.58 tok/s，高于 llama.cpp Q4 与 Ollama Q4；SGLang 没有 C4/C8 数据。跨权重格式的差异代表部署路线结果，不作为纯框架排名。

Ollama E2、E3 和 E4 已完成；E4 将 daemon API ready（中位数 1.088 s）与首次模型请求墙钟（82.471 s）分开记录，后者包含模型加载耗时。

## 长上下文

| 后端 / 权重 | 目标上下文 | 实际 prompt | 并发 | completion tok/s | P95 延迟 | 本轮状态 |
|---|---:|---:|---:|---:|---:|---|
| llama.cpp / Q4 | 32768 | 21740 | 1 | 36.05 | 7.091 s | 8/8 成功 |
| vLLM / FP8 | 32768 | 21740 | 1 | 32.53 | 7.78 s | 成功 |
| SGLang / FP8 + Triton | 32768 | 21740 | 1 | 34.81 | 7.335 s | 8/8 成功 |
| Ollama / UD-Q4_K_XL | 8192 | 5507 | 1 | 33.48 | 6.804 s | 8/8 成功 |
| Ollama / UD-Q4_K_XL | 32768 | 21740 | 1 | 31.76 | 7.167 s | 8/8 成功 |
| llama.cpp / Q4 | 65536 | 43385 | 1 | 33.93 | 7.527 s | 成功 |
| vLLM / FP8 | 65536 | 43385 | 1 | 32.97 | 7.70 s | 成功 |
| SGLang / FP8 | 65536 | 约 43385 | 1 | — | — | 当前 KV pool 不足，未测 |

单并发下，llama.cpp Q4 在 32K 和 65K 两档略快于 vLLM。vLLM 的长上下文优势出现在并发 4：32K 目标时为 72.29 tok/s、P95 14.219 s；llama.cpp Q4 同档为 20.213 tok/s、P95 89.116 s。此结果仍是 FP8 对 Q4 的部署路线比较。

## 显存与启动就绪

| 后端 / 权重 | 显存证据 | 启动到就绪 | 测量配置与限制 |
|---|---|---:|---|
| llama.cpp / Q4 | E1 峰值约 77.6 GiB；E2 约 80.4 GiB | 中位数 81.593 s | context 10240、单 Slot；3 轮，缓存状态 uncontrolled |
| llama.cpp / Q5 | E1 峰值约 97.4 GiB | 中位数 102.444 s | 与 Q4 同路线对照 |
| vLLM / FP8 | 模型文件约 173 GiB；基准记录没有同口径的绝对峰值 | 中位数 202.098 s | O2、max model len 8192、max seqs 2；3 轮，缓存状态 uncontrolled |
| SGLang / FP8 | E1/E2 整卡峰值约 173.5–174.6 GiB；32K/Triton 峰值 179.78 GiB | 中位数 91.21 s | 32K、max running requests 1、Triton；3 轮，缓存状态 uncontrolled |
| Ollama / UD-Q4_K_XL | E1 整卡峰值 77.76–77.82 GiB | API ready 中位数 1.088 s；首次模型请求 82.471 s | 首次请求 `load_duration` 80.950 s；缓存状态 uncontrolled |

Q4 的显存优势明显，但主要来自 Q4 权重和部署组合。同一 llama.cpp 配置从 Q4 换到 Q5，吞吐约提高 0.6%，显存多约 19.9 GiB，启动中位数高约 25.6%；Mini-8 未证明 Q5 有稳定质量优势，因此当前 MI300X 配置优先选 Q4 更划算。

SGLang 的 FP8 启动就绪实测快于 vLLM（中位数约 91 秒对 202 秒）。两者使用同一 FP8 权重和硬件；但上下文、并发和运行时初始化路径不同，不能把全部差异归因于框架。两边都是原生 FP8 Safetensors，因此也不能把差距归因于“原生 tensors”。启动包含权重加载与服务初始化，且未清理文件缓存，不是冷启动基准。

## 输出可用性与质量边界

- llama.cpp Q4/Q5 使用 4096 token 上限完成 Mini-8：均 8/8 成功、无空回答、无长度截断；没有独立人工评分，不能据此判断 Q5 质量更优。
- vLLM FP8 的默认 thinking 与关闭 thinking 两轮 Mini-8 均 8/8 成功、无空回答和长度截断。默认 thinking 输出出现过程性文本；关闭 thinking 后该格式问题消失。两轮输出长度不同，不能用耗时直接推断性能变化。
- SGLang FP8 Mini-8 为 8/8 成功、无空回答和长度截断，reasoning 内容单独返回；同样没有人工评分。
- Ollama UD-Q4_K_XL 的 E4 Mini-8 为 8/8 成功、无空回答和长度截断，平均观测输出速度 34.986 tok/s；同样没有人工评分，长题目的耗时受输出长度影响。

E4 验证了接口和输出流程可用，不构成回答质量排名。跨后端质量对比需要固定 thinking 设置、输出上限和评分方法。

## 后续优化方向：图片理解

本轮 E1–E4 覆盖的是文本请求，未验证图片理解。虽然 [GGUF 模型卡](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) 和 [FP8 模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)标注支持视觉输入，部署时仍需确认视觉组件、后端接口和图文请求格式均可用。后续可单独建立统一图片集进行接口探测与质量测试；该项不属于本轮文本性能结论。

- llama.cpp：GGUF 路线需配套视觉投影文件（mmproj）并通过 `--mmproj` 加载。
- vLLM / SGLang：用兼容的图文消息格式发送图片，验证当前 FP8 部署。
- Ollama：当前导入 tag 未报告 `vision` 能力，需先确认该 GGUF 路线的视觉导入方式。

## 当前选择边界

- **低显存部署**：llama.cpp + Q4 是当前数据最明确的选择。
- **高并发服务**：vLLM 当前测到 C8，并在 4K/8K 短上下文和 32K/C4 单元显示较好的吞吐扩展。SGLang 尚未测 C4/C8，不能据此断言其最大并发更低。
- **启动体验**：SGLang 在已测 FP8 配置下更快到健康就绪；统一上下文、并发和启动阶段后，才能形成严格结论。
- **上下文能力**：llama.cpp 与 vLLM 完成 65K 目标档；SGLang 当前配置下 32K 已通过 Triton，65K 受 KV pool 限制未测。

新增后端或量化时，沿用同一模型目录下的后端部署记录，并记录权重表示、精度、上下文、并发和请求设置。若无法统一权重表示，报告端到端部署取舍，不写成纯框架排名。
