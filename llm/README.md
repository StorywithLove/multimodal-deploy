# 语言大模型

本页汇总语言模型分支中的模型路线和实测概要，帮助快速查看部署取舍。每个模型各自保留一节，完整测试条件、数据和限制放在该模型的项目页；共用工作负载与测评规则见文末链接。

## 模型文件存储约定

ModelScope Notebook 的 `/mnt/workspace` 是持久化挂载空间；**不要把大型模型权重、GGUF 分片、Ollama blobs 或权重缓存写入该目录**，避免触及实例配额并影响后续启动。模型文件放在实例本地 `/root/tmp`，导入前检查其可用空间；Ollama 导入会额外复制一份权重。`/mnt/workspace` 用于项目代码、配置、日志和实验结果。检查命令：`df -h /root/tmp /mnt/workspace`、`du -sh /mnt/workspace`。ModelScope 的 Notebook 存储说明见[此处](https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-04.md)；`du` 显示目录已用量，挂载上的 `df` 容量不一定等于个人配额。

## 模型索引

| 模型 | 当前路线 | 状态 | 项目页 |
|---|---|---|---|
| Qwen3.8-Flash-Next | llama.cpp / GGUF Q4、Q5；vLLM / FP8；SGLang / FP8；Ollama / GGUF | 四条路线已完成当前 E1–E4；Ollama API ready 中位数 1.088 s，首次模型请求墙钟 82.471 s | [实测详情](qwen3.8-flash-next/README.md) |

## Qwen3.8-Flash-Next

已测路线运行于单张 AMD MI300X 192GB，权重精度并不相同：llama.cpp 使用 GGUF Q4/Q5，Ollama 使用同源 GGUF UD-Q4_K_XL，vLLM 与 SGLang 使用同一份 FP8 Safetensors。因此以下是该模型的部署路线比较，不构成纯框架排名。Ollama 的 GGUF 推理使用 `llamacpp` runner；它与直接 `llama-server` 的差别还包括模型登记、blob 管理、服务生命周期和 API 层。

部署操作记录：[llama.cpp](qwen3.8-flash-next/llama.cpp.md)、[vLLM](qwen3.8-flash-next/vllm.md)、[SGLang](qwen3.8-flash-next/sglang.md)、[Ollama](qwen3.8-flash-next/ollama.md)。

### 短上下文吞吐与并发

单并发的目标上下文为 512、4096、8192；下表是每个档位的 completion 聚合吞吐，单位 tok/s。

| 后端 / 权重 | 512 | 4096 | 8192 |
|---|---:|---:|---:|
| llama.cpp / Q4 | 38.58 | 35.52 | 34.81 |
| vLLM / FP8 | 35.91 | 34.50 | 35.20 |
| SGLang / FP8 | 37.63 | 35.17 | 35.16 |
| Ollama / UD-Q4_K_XL（E1） | 34.85 | 34.00 | 33.36 |

单并发时四条路线整体接近，约为 33–39 tok/s。Ollama E2 的 C1/C2/C4/C8 矩阵 12 格全部 8/8 成功；C8 吞吐在 512/4096/8192 档分别为 85.99/61.46/50.70 tok/s，8192 档 C4 更快（54.85 tok/s）。Ollama E3 已完成 8192/32768 长上下文单并发测评；其余路线并发 2 时，8192 档分别为 llama.cpp Q4 62.28、vLLM FP8 55.46、SGLang FP8 62.64 tok/s；并发 8 时，llama.cpp Q4 为 65.10、vLLM FP8 为 108.58 tok/s，SGLang 本轮未测到 C8。

因此，vLLM 在本轮已测的高并发项目中吞吐扩展更明显；SGLang 缺少 C4/C8 数据，不能据此推断它的并发上限。完整并发矩阵见 [模型项目页](qwen3.8-flash-next/README.md)。

### 长上下文

| 测试项 | llama.cpp / Q4 | vLLM / FP8 | SGLang / FP8 |
|---|---:|---:|---:|
| 32K 目标、并发 1：吞吐 / P95 | 36.05 tok/s / 7.091 s | 32.53 tok/s / 7.78 s | 34.81 tok/s / 7.335 s |
| 65K 目标、并发 1：吞吐 / P95 | 33.93 tok/s / 7.527 s | 32.97 tok/s / 7.70 s | 未测：当前 KV pool 不足 |
| 32K 目标、并发 4：吞吐 / P95 | 20.213 tok/s / 89.116 s | 72.29 tok/s / 14.219 s | 未测 |

单并发长上下文下 llama.cpp Q4 略快于 vLLM FP8；32K 并发 4 的已测单元中，vLLM 吞吐更高、P95 更低。该结论仍受权重精度、服务参数和测试覆盖范围影响。

### 显存与启动就绪

| 后端 / 权重 | 显存数据 | 启动到就绪中位数 | 说明 |
|---|---:|---:|---|
| llama.cpp / Q4 | E1 约 77.6 GiB；E2 约 80.4 GiB | 81.593 s | context 10240、单 Slot |
| llama.cpp / Q5 | E1 约 97.4 GiB | 102.444 s | 同一路线的量化对照 |
| vLLM / FP8 | 模型文件约 173 GiB；缺少同口径绝对峰值 | 202.098 s | O2、max model len 8192、max seqs 2 |
| SGLang / FP8 | E1/E2 约 173.5–174.6 GiB；32K/Triton 179.78 GiB | 91.21 s | 32K、max running requests 1、Triton |
| Ollama / UD-Q4_K_XL | E1 整卡峰值 77.76–77.82 GiB | API ready 中位数 1.088 s；首次模型请求 82.471 s | `load_duration` 80.950 s；不把 daemon ready 当模型 ready |

llama.cpp Q4 的显存占用最低，优势主要来自 Q4 权重。相较 Q4，同一路线 Q5 显存多约 19.9 GiB、启动中位数高约 25.6%，吞吐只高约 0.6%；现有 Mini-8 检查没有人工评分，尚不能证明 Q5 有稳定质量优势。

SGLang 在当前 FP8 配置下启动就绪快于 vLLM，但上下文、并发和初始化路径不完全一致，缓存状态也未控制，不能把全部差异归因于框架。SGLang 与 vLLM 都使用原生 FP8 Safetensors。

### 当前结论与边界

- **显存受限**：本轮数据支持优先考虑 llama.cpp + Q4；跨路线差异同时包含权重精度影响。
- **较高并发**：vLLM 在本轮 C8 与 32K/C4 已测项目中吞吐扩展较好；SGLang 还没有对应 C4/C8 数据。
- **启动体验**：SGLang 本轮到健康就绪较快；需要统一配置和缓存状态后再作严格比较。
- **输出检查**：四条已测路线的 E4 基本输出检查均通过；Ollama Mini-8 为 8/8 成功、无空回答和长度截断，未做人工质量评分，不构成回答质量排名。

以上概要只适用于当前模型、硬件和已记录配置。新增模型或后端时，应分别记录模型版本、权重表示与精度、上下文、并发、请求设置和测量条件；如果无法统一权重格式，报告完整部署方案的取舍。

## 共用比较方法

- [仓库总览](../README.md)：各模态和模型的入口。
- 历史测评工具与工作负载配置已移除，原命令不能直接复现。
- results/：结构化测试结果。

所有比较尽量复用相同工作负载。报告吞吐时说明聚合口径，报告显存与启动时注明测量配置和限制；未执行的测试项标记为“未测”。
