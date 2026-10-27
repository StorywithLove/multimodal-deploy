# Inference Engineering Portfolio / 推理工程作品集

覆盖 AMD MI300X / ROCm 推理测评、模型平台 API 实践与国产 GPU 云端部署的推理工程作品集。这里记录已经验证的部署路径、关键实测结果、共用测评工具和工程经验，并提供文本、图像、语音、OCR 与向量检索的实践入口。

项目按模态与任务组织，具体模型放在相应主题下；后续增加模型时沿用相同的阅读结构。这里展示项目概览，语言模型分支保留更多比较数据，各模型项目页记录完整测试细节。

## 仓库目录

| 目录 | 主要功能 |
|---|---|
| `configs/` | 版本化的基准工作负载和实验配置。 |
| `llm/` | 语言模型部署路线、模型索引、测试概要和各模型详细记录。 |
| `platforms/` | ModelScope、Hugging Face 的 API 调用、数据集实践与能力探索，以及模力方舟 / 沐曦的多模态云端部署、API 服务和排障记录。 |
| `results/` | 结构化基准结果和实验输出，不存放模型权重。 |
| `video/` | 视频模型的部署脚本、配置和运行记录。 |
| `vision/` | 图片与视觉模型的路线规划和相关实践。 |

`platforms/` 按平台划分：`platforms/modelscope/` 与 `platforms/huggingface/` 各自维护 `model_inference/`（在线推理调用示例）、`datasets/`（数据集处理与微调实践）和 `capabilities/`（平台专区及模型能力边界记录）；`platforms/moark/metax/` 按考试主题记录沐曦云端模型部署与服务化实践。模型调用脚本按模型命名；共享 API 调用模板使用描述性名称。

## 模型平台与国产算力实践

完整入口见 [平台索引](platforms/README.md)。

| 平台 | 功能与记录 |
|---|---|
| [ModelScope](platforms/modelscope/README.md) | 模型推理 API 调用示例、数据集实践和平台能力记录。 |
| [Hugging Face](platforms/huggingface/README.md) | 模型推理 API、数据集及 Hub 平台能力探索。 |
| 模力方舟 / 沐曦 | 厂商适配环境下的单次推理、OpenAI 风格 API、产物保存和排障；六类主题入口如下。 |

### 模力方舟 / 沐曦

2026-10-22 已取得沐曦 L1 认证（用户确认）。六份文档保留各阶段的原始证据边界；认证通过不替代单项运行日志或统一性能测评。

| 主题 | 模型与功能 | 部署记录 |
|---|---|---|
| 文本生成 | Qwen3-0.6B、Qwen3-8B；Transformers 单次推理、vLLM API、SSH 隧道与模型名匹配 | [文本生成模型](platforms/moark/metax/文本生成模型.md) |
| 图像生成 | Z-Image-Turbo、Qwen-Image-2512；Diffusers 生图、KV Cache 绕过、系统内存 OOM 排障 | [图形生成模型](platforms/moark/metax/图形生成模型.md) |
| 语音识别 | Whisper、Qwen3-ASR-1.7B；音频转写、文本保存、FastAPI 上传识别 | [语音识别模型](platforms/moark/metax/语音识别模型.md) |
| 语音合成 | IndexTTS2、Qwen3-TTS Base；参考音频克隆、WAV 保存、base64 API 与采样率处理 | [语音合成模型](platforms/moark/metax/语音合成模型.md) |
| OCR | PaddleOCR-VL 示例跑通；FireRed-OCR 单次识别与上传 API 代码已整理，未归档运行结果 | [OCR模型](platforms/moark/metax/OCR模型.md) |
| 向量检索与重排 | Qwen3-Embedding-8B、Chroma、bge-reranker-v2-m3；1024 维接口、22 片入库及 Top-10 重排 | [部署向量化模型与向量数据库的使用](platforms/moark/metax/部署向量化模型与向量数据库的使用.md) |

这些记录同时覆盖预置模型目录核查、依赖与镜像匹配、模型管道选择、端口管理和 API 验收。不同硬件、精度和任务的结果分别记录，不直接混入下文 MI300X 的 E1–E4 框架测评。

## 推理引擎与部署流程

下表概括四种引擎在本项目中的典型模型接入与启动方式。模型格式和具体参数会随模型、框架版本与硬件路线变化。

| 推理引擎 | 运行方式 | 大致流程 |
|---|---|---|
| llama.cpp | 直接读取 GGUF；由 `llama-server` 加载模型并提供推理 API | `llama-server -m <模型.gguf>` → 加载权重与运行环境 → 服务就绪 |
| Ollama | 服务进程管理模型；本项目的 GGUF 路线通过 Modelfile 登记模型，并由 llama.cpp runner 推理 | `ollama serve` → `Modelfile` 指向 GGUF → `ollama create` 导入 → `ollama run` 或 API 首次请求时加载模型 |
| vLLM | 以模型仓库或本地检查点启动 serving engine；MI300X 路线使用 FP8 Safetensors，沐曦实践另含 BF16 文本与池化模型 | `vllm serve <模型仓库或路径>` → 加载权重并初始化引擎 → API 服务就绪 |
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

`vision/` 模块仍处于路线规划阶段，见 [vision/README.md](vision/README.md)。已完成的模力方舟 / 沐曦生图实践记录在 [图形生成模型](platforms/moark/metax/图形生成模型.md)，OCR 方案与证据边界见 [OCR模型](platforms/moark/metax/OCR模型.md)。

## 共用测评与记录

| 内容 | 位置 | 用途 |
|---|---|---|
| 模型索引与分支实测概要 | [llm/README.md](llm/README.md) | 汇总语言模型路线，方便横向查看 |
| 模型和后端详细记录 | 各模型项目目录 | 记录部署步骤、参数、结果和限制 |
| 工作负载配置 | configs/benchmark/ | 版本化定义上下文、并发、输出长度和请求数 |
| 测评工具与说明 | [configs/benchmark/README.md](configs/benchmark/README.md) | 通过已启动的 OpenAI-compatible 服务执行一致测量 |
| 结构化结果 | results/ | 保存环境信息和聚合指标 |

比较结果应标明模型版本、硬件、软件栈、权重格式与精度、上下文和并发设置。权重表示或精度不同的路线应按端到端部署方案解读；缺少数据的项目标为“未测”，不据此推断能力上限。

## 公开与维护边界

- 不提交模型权重、视频 / 图片大文件、临时日志、云端 URL、Cookie 或访问令牌。
- 运行 API 示例前，通过环境变量注入凭据，例如 MODELSCOPE_API_TOKEN，不要把令牌写入源文件。
- 对外发布前仍需确定代码托管平台、仓库可见性和许可证。
