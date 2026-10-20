# Inference Engineering Portfolio / 推理工程作品集

面向 AMD MI300X / ROCm 的模型部署与推理测评作品集。这里记录已经验证的部署路径、关键实测结果、共用测评工具和工程经验。

项目按模态与任务组织，具体模型放在相应主题下；后续增加模型时沿用相同的阅读结构。这里展示项目概览，语言模型分支保留更多比较数据，各模型项目页记录完整测试细节。

## 推理引擎与部署流程

下表概括四种引擎在本项目中的典型模型接入与启动方式。模型格式和具体参数会随模型、框架版本与硬件路线变化。

| 推理引擎 | 运行方式 | 大致流程 |
|---|---|---|
| llama.cpp | 直接读取 GGUF；由 `llama-server` 加载模型并提供推理 API | `llama-server -m <模型.gguf>` → 加载权重与运行环境 → 服务就绪 |
| Ollama | 服务进程管理模型；本项目的 GGUF 路线通过 Modelfile 登记模型，并由 llama.cpp runner 推理 | `ollama serve` → `Modelfile` 指向 GGUF → `ollama create` 导入 → `ollama run` 或 API 首次请求时加载模型 |
| vLLM | 以模型仓库或本地检查点启动 serving engine；本项目使用 FP8 Safetensors | `vllm serve <模型仓库或路径>` → 加载权重并初始化引擎 → API 服务就绪 |
| SGLang | 以模型仓库或本地检查点启动 serving engine；本项目使用 FP8 Safetensors | `python -m sglang.launch_server --model-path <模型仓库或路径>` → 加载权重并初始化引擎 → API 服务就绪 |

llama.cpp 直接从 GGUF 文件启动；Ollama 在推理前增加服务管理、模型登记和导入步骤，`ollama serve` 就绪不代表模型已经加载，通常要等首次 `run` 或 API 请求。vLLM 与 SGLang 则在启动 serving engine 时加载模型并进行运行时初始化。因而比较启动时间时，应把 Ollama 的服务启动、一次性导入、模型冷加载分开记录；并注明 GGUF Q4/Q5 与 FP8 Safetensors 的精度差异。当前项目的结果代表端到端部署路线取舍，不是单独的推理引擎排名。

### Ollama 与 llama.cpp 的关系

本项目的 Ollama 路线和 llama.cpp 路线使用同一份 `UD-Q4_K_XL` GGUF。Ollama 的 `ollama ps` 显示 runner 为 `llamacpp`，说明实际 GGUF 推理由 llama.cpp runner 执行；Ollama 在其上提供模型登记、blob 存储、服务生命周期、OpenAI-compatible API 和并发调度。直接 llama.cpp 路线则由 `llama-server` 直接加载 GGUF。两者的对比口径是同一 GGUF 上的两条服务部署路线，启动指标需要分别记录 API ready、模型加载和首次请求。

## 语言大模型

语言模型分支的模型索引和实测概要见 [llm/README.md](llm/README.md)。

### Qwen3.8-Flash-Next

| 后端路线 | 状态 | 部署记录 |
|---|---|---|
| llama.cpp + GGUF UD-Q4_K_XL / UD-Q5_K_XL | E1–E4 已完成 | [llama.cpp](llm/qwen3.8-flash-next/llama.cpp.md) |
| vLLM + FP8 Safetensors | E1–E4 已完成 | [vLLM](llm/qwen3.8-flash-next/vllm.md) |
| SGLang + FP8 Safetensors | E1–E4 已完成；32K 使用 Triton attention backend | [SGLang](llm/qwen3.8-flash-next/sglang.md) |
| Ollama + GGUF UD-Q4_K_XL | E1–E4 完成；daemon API ready 中位数 1.088 s，首次模型请求墙钟 82.471 s | [Ollama](llm/qwen3.8-flash-next/ollama.md) |

四条路线实测摘要：

- **llama.cpp + Q4**：本轮显存占用最低，E1 峰值约 77.6 GiB；这主要体现 Q4 相对 FP8 的权重表示与部署路线优势。
- **vLLM + FP8**：当前已测到较强的并发扩展，8192 目标上下文、并发 8 时约 108.58 tok/s；32K 目标上下文、并发 4 时约 72.29 tok/s。SGLang 本轮只测到并发 2，尚无 C4/C8 数据。
- **SGLang + FP8**：当前配置启动就绪中位数约 91 秒，vLLM 约 202 秒；配置与缓存状态存在差异。32K 单并发已测通，65K 因当前 KV pool 容量不足未测。
- **Ollama + UD-Q4_K_XL**：E1–E4 已完成；daemon API ready 中位数约 1.088 秒，首次模型请求墙钟约 82.471 秒，其中 Ollama 报告的模型加载时间约 80.950 秒。该路线使用 llama.cpp runner，启动指标按 daemon 与模型请求分开记录。

由于 llama.cpp 使用 GGUF Q4/Q5，而 vLLM 和 SGLang 使用 FP8 Safetensors，这些数据反映的是部署路线取舍，不是纯框架排名。详细矩阵、显存与启动记录、质量检查及比较限制见 [Qwen3.8-Flash-Next 项目实测](llm/qwen3.8-flash-next/README.md)。

## 视频

### Wan2.2-T2V-A14B

文生视频路线已跑通，运行记录见 [Wan2.2-T2V-A14B](video/wan2.2-t2v-a14b/record.md)。

### Wan2.2-TI2V-5B

文本 / 图像生视频路线待开始，运行记录见 [Wan2.2-TI2V-5B](video/wan2.2-ti2v-5b/record.md)。

## 图片与视觉

当前处于路线规划阶段。后续确定模型和任务后，会在本节下增加对应模型介绍。模块说明见 [vision/README.md](vision/README.md)。

## 共用测评与记录

| 内容 | 位置 | 用途 |
|---|---|---|
| 模型索引与分支实测概要 | [llm/README.md](llm/README.md) | 汇总语言模型路线，方便横向查看 |
| 模型和后端详细记录 | 各模型项目目录 | 记录部署步骤、参数、结果和限制 |
| 工作负载配置 | configs/benchmark/ | 版本化定义上下文、并发、输出长度和请求数 |
| 测评工具与说明 | [benchmarks/README.md](benchmarks/README.md) | 通过已启动的 OpenAI-compatible 服务执行一致测量 |
| 结构化结果 | results/ | 保存环境信息和聚合指标 |

比较结果应标明模型版本、硬件、软件栈、权重格式与精度、上下文和并发设置。权重表示或精度不同的路线应按端到端部署方案解读；缺少数据的项目标为“未测”，不据此推断能力上限。

## 公开与维护边界

- 不提交模型权重、视频 / 图片大文件、临时日志、云端 URL、Cookie 或访问令牌。
- 运行 API 示例前，通过环境变量注入凭据，例如 MODELSCOPE_API_TOKEN，不要把令牌写入源文件。
- 对外发布前仍需确定代码托管平台、仓库可见性和许可证。
