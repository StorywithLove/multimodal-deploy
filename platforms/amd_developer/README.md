# AMD AI 学院学习与实践

- 学院入口：[AMD AI 学院](https://developer.amd.com.cn/academy)
- 实践入口：[Radeon Cloud Gallery](https://developer.amd.com.cn/radeon/)
- 模板核对日期：2026-10-25
- 本页把 Radeon Cloud 当前获取到的 41 个模板整理为课程学习总览和模板分类与功能。

> 说明：课程页面只返回前端壳，未直接取得官方“课程—Notebook”绑定清单。因此下表中的“直接对应”是基于模板标题、可读取的 Notebook 内容与课程笔记进行的内容匹配，不等同于 AMD 官方一对一指定作业。模板页面和内容会变化，实际学习时以 [Radeon Cloud Gallery](https://developer.amd.com.cn/radeon/) 当前版本为准。

## 一、课程学习总览

| 课程 | 课程记录 | 视频进度 | Notebook 实践 | 当前优先 Notebook |
| --- | --- | --- | --- | --- |
| 扩散模型与 AMD AI 生成训练 | [课程笔记](diffusion_models_and_amd_ai_training.md) | 已完成 | 5051 已生成图像；5052 已生成视频 | [5051：Image Generation](https://developer.amd.com.cn/radeon/templates/5051/preview) → [5052：Video Generation](https://developer.amd.com.cn/radeon/templates/5052/preview) |
| AI 智能体 | [课程笔记](ai_agents.md) | 入门篇、进阶篇已完成 | 5048 已完成 | [5048：Agents 101](https://developer.amd.com.cn/radeon/templates/5048/preview) → [5061：Agents 201](https://developer.amd.com.cn/radeon/templates/5061/preview) |
| 大语言模型（LLM）基础与优化 | [课程笔记](llm_basics_and_optimization.md) | 4/4 节已完成 | 尚未开始 | [4978：HF on AMD](https://developer.amd.com.cn/radeon/templates/4978/preview) → [5018：AI on AMD](https://developer.amd.com.cn/radeon/templates/5018/preview) |

课程笔记记录理论、概念和课程脉络；Notebook 的实际步骤、运行日志、结果截图和阻塞问题统一记录在 [notebook_ROCm/](notebook_ROCm/) 中。不要为尚未学习的模板预先创建空记录。

课程笔记的本地配图统一存放在 [images/](images/)，并使用相对路径引用。

### 2026-10-30：5051 与 5052 实践

实践环境为 ModelScope NVIDIA A10 24 GB，CUDA PyTorch，而非 AMD ROCm。

- **5051 图像生成**：使用 SDXL Base 1.0、BF16，默认 1664×928、30 步，成功生成图片。用户实测启用 cuDNN 时耗时 15.0 秒，PyTorch 峰值分配显存 10.08 GiB；另一组禁用 cuDNN 时约 14.7 秒、15.31 GiB，单次结果不代表稳定性能差异。脚本与后台日志入口分别为 [5051 脚本](../../scripts/amd-academy/5051_image_generation.py) 和 [运行脚本](../../scripts/amd-academy/run_5051_image_generation.sh)，支持保存去噪预览和 GIF；该动图实现尚未记录实测结果。
- **5052 视频生成**：已下载 Wan 2.2 TI2V-5B、UMT5 和 VAE，启动 ComfyUI 并由用户确认浏览器可访问，成功生成两个视频。日志显示两次任务各 20 步、耗时 77.96 秒和 37.04 秒，具体提示词和两次配置差异尚未核实；[运行日志](../../results/5052-video-generation/5052_video_generation_20261030_172506_DDqAN1.log)为运行期间快照，视频保存在本地 `results/5052-video-generation/`，不纳入 Git。入口为 [5052 脚本](../../scripts/amd-academy/5052_video_generation.py) 和 [运行脚本](../../scripts/amd-academy/run_5052_video_generation.sh)。当前实例访问地址：[ComfyUI](https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2318316/proxy/8000/)，通用格式为 `https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-<实例编号>/proxy/<端口>/`；实例编号及登录状态变化后需重新确认。模型与输出位于 `/root/tmp/ComfyUI-5052`，实例回收前应备份。

### 2026-10-31：5048 Agents 101 实践

基于本地 Qwen3.5-9B vLLM 服务完成 [5048 Agents 101 Notebook](notebook_ROCm/5048_agents_101.ipynb)，验证了三种工具接入方式：

- **自定义 Python 函数工具**：将本地日期时间函数注册给 Pydantic AI Agent，并由模型调用。
- **Python MCP 工具**：通过 `MCPServerStdio` 启动 `mcp_server_time`，完成时区日期时间查询。
- **Node.js MCP 工具**：启动 Airbnb MCP server 并完成住宿搜索；这是搜索演示，预订仍需在 Airbnb 页面确认。

当前智能体课程进度：5048 已完成，下一步实践 [5061 Agents 201](https://developer.amd.com.cn/radeon/templates/5061/preview)。
## 二、41 个模板的分类与功能

以下按 Notebook 的**主要学习目标**归为 8 个大类。一个模板可能同时涉及多个方向，例如智能体依赖 vLLM、视觉模型也涉及微调；为方便查找，每个模板只在最主要的大类中列出一次。

“内容可读”表示本次核对能够读取 Gallery 预览页中的 Markdown 与代码单元格；“未配置路径”只表示 Gallery 元数据没有提供 Notebook 入口，不代表模板一定不可运行。

### 2.1 分类与 Notebook 对照表

下表将大类和对应 Notebook 合并展示。第二列采用“**ID · Notebook：主要功能**”格式；未特别标注的模板均为“内容可读”。

| 大类 | 对应 Notebook（ID、名称与主要功能） |
| --- | --- |
| **LLM 基础、提示工程与推理服务（7）**<br>Transformer、提示词、vLLM 部署及推理基准 | [4978 · HF on AMD](https://developer.amd.com.cn/radeon/templates/4978/preview)：理解 Tokenizer、Embedding、Attention 和语言模型流程。<br>[4015 · Modern LLM Notebook](https://developer.amd.com.cn/radeon/templates/4015/preview)：认识 Tokenizer、Token ID、词表和文本数值化。<br>[5018 · AI on AMD](https://developer.amd.com.cn/radeon/templates/5018/preview)：构建聊天机器人并实验提示词、temperature、top-k 和 top-p。<br>[5037 · LLM Serving Inference with vLLM](https://developer.amd.com.cn/radeon/templates/5037/preview)：配置 ROCm 并启动本地 vLLM 模型服务。<br>[5049 · GPU Optimization for LLM Inference - Lab 1](https://developer.amd.com.cn/radeon/templates/5049/preview)：启动 vLLM、调用 API 并测量推理性能。<br>[5050 · GPU Optimization for LLM Inference - Lab 2](https://developer.amd.com.cn/radeon/templates/5050/preview)：配置 AITER、vLLM 和 DeepSeek-R1，观察显存、质量与性能。<br>[29 · vllm-workshop](https://developer.amd.com.cn/radeon/templates/29/preview)：对比 vLLM v0/v1、Prefix Caching 及 TTFT、TPOT 等指标。 |
| **模型训练、微调与强化学习（4）**<br>LoRA、Unsloth、GRPO、后训练及模型转换 | [257 · Hello ROCm](https://developer.amd.com.cn/radeon/templates/257/preview)：使用 Gemma 4 和 emotion 数据集进行单卡 LoRA 微调。<br>[5074 · Finetuning Your Own R1 Reasoning Model on Unsloth](https://developer.amd.com.cn/radeon/templates/5074/preview)：使用 Unsloth 与 GRPO 训练推理模型并设计奖励函数。<br>[5053 · Reinforcement Learning with LLMs](https://developer.amd.com.cn/radeon/templates/5053/preview)：实践大语言模型强化学习与后训练流程。<br>[722 · Unsloth and llama.cpp](https://developer.amd.com.cn/radeon/templates/722/preview)：训练、转换或本地推理；**未配置 Notebook 路径，入口需核验**。 |
| **智能体与自动优化（7）**<br>工具调用、MCP、A2A、多智能体及智能化性能优化 | [5048 · Agents 101](https://developer.amd.com.cn/radeon/templates/5048/preview)：使用 vLLM、Pydantic AI 和 MCP 构建查询智能体。<br>[5061 · Agents 201](https://developer.amd.com.cn/radeon/templates/5061/preview)：使用 Google ADK、A2A、本地 vLLM 和工具调用构建多智能体系统。<br>[320 · Run Hermes-Agent with vLLM on Radeon GPU](https://developer.amd.com.cn/radeon/templates/320/preview)：连接本地 vLLM 与 Hermes-Agent，实践技能封装和自主工作流。<br>[5 · AgenticAI 推理自优化](https://developer.amd.com.cn/radeon/templates/5/preview)：用智能体测量、分析、调优并验证模型推理。<br>[26 · inference-skill](https://developer.amd.com.cn/radeon/templates/26/preview)：使用智能体辅助基础模型推理优化。<br>[6 · Agent Development](https://developer.amd.com.cn/radeon/templates/6/preview)：使用智能体辅助推理性能分析和优化。<br>[247 · openclaw](https://developer.amd.com.cn/radeon/templates/247/preview)：OpenClaw 开发工具或智能体工作流；**未配置 Notebook 路径，具体功能需核验**。 |
| **生成式图像与视频（3）**<br>Diffusers、SDXL、ComfyUI 及音视频生成 | [5051 · Diffusion Models - Image Generation](https://developer.amd.com.cn/radeon/templates/5051/preview)：使用 Diffusers 和 SDXL 进行文生图并调整生成参数。<br>[5052 · Diffusion Models - Video Generation](https://developer.amd.com.cn/radeon/templates/5052/preview)：配置 ComfyUI、模型组件和工作流生成视频。<br>[1196 · comfyui-ltx](https://developer.amd.com.cn/radeon/templates/1196/preview)：使用 ComfyUI 和 LTX-2.3 生成带声音的视频。 |
| **视觉、多模态与向量检索（7）**<br>检测、VLM、OCR、视频理解及向量数据库 | [3991 · Ultralytics Workshop 1](https://developer.amd.com.cn/radeon/templates/3991/preview)：微调 YOLO OBB，评估检测结果并进行视频跟踪。<br>[3590 · Ultralytics Workshop2](https://developer.amd.com.cn/radeon/templates/3590/preview)：用 YOLO 选取视频关键帧，再由 VLM 基于视觉证据回答问题。<br>[3589 · HunyuanOCR v1.5](https://developer.amd.com.cn/radeon/templates/3589/preview)：处理多语言、表格和低分辨率文档并导出结构化结果。<br>[3102 · seekdb-hipvs](https://developer.amd.com.cn/radeon/templates/3102/preview)：对比 CPU/GPU 向量检索性能和召回质量。<br>[3863 · MiniCPM5-2B](https://developer.amd.com.cn/radeon/templates/3863/preview)：运行 MiniCPM5-2B 视觉语言模型；**未配置 Notebook 路径**。<br>[2568 · MiniCPM-v46](https://developer.amd.com.cn/radeon/templates/2568/preview)：运行 MiniCPM-v4.6 视觉语言模型；**未配置 Notebook 路径**。<br>[2332 · MiniCPM5-1B](https://developer.amd.com.cn/radeon/templates/2332/preview)：运行 MiniCPM5-1B 视觉语言模型；**未配置 Notebook 路径**。 |
| **GPU 编程、迁移与性能工具（9）**<br>HIP、Triton、TileLang、CUDA 迁移、调试与性能分析 | [2880 · Hello GPU](https://developer.amd.com.cn/radeon/templates/2880/preview)：从简单算子和计时入门，理解 CPU/GPU 工作与数据位置。<br>[5020 · HIP Programming Basics](https://developer.amd.com.cn/radeon/templates/5020/preview)：学习 HIP 语法、内存管理和 Kernel 启动。<br>[5021 · HIP Kernel Programming](https://developer.amd.com.cn/radeon/templates/5021/preview)：理解线程、内存和执行模型并编写 AMD GPU Kernel。<br>[5019 · CUDA-to-HIP Conversion](https://developer.amd.com.cn/radeon/templates/5019/preview)：将 CUDA 程序迁移到 HIP。<br>[5032 · Triton Kernel Development on AMD Radeon GPUs](https://developer.amd.com.cn/radeon/templates/5032/preview)：使用 Triton 编写、运行和优化 Radeon GPU Kernel。<br>[2 · Kernel Development & Optimizations with Triton](https://developer.amd.com.cn/radeon/templates/2/preview)：使用 Triton 开发和优化 AMD GPU Kernel。<br>[656 · Tilelang On Radeon](https://developer.amd.com.cn/radeon/templates/656/preview)：使用 TileLang 编写和优化算子。<br>[5054 · GPU Debugging with ROCgdb](https://developer.amd.com.cn/radeon/templates/5054/preview)：使用 ROCgdb 调试 ROCm/HIP 程序。<br>[5055 · GPU Profiling with rocprof](https://developer.amd.com.cn/radeon/templates/5055/preview)：使用 rocprof 采集性能轨迹并定位瓶颈。 |
| **具身智能与机器人仿真（2）**<br>仿真数据、VLA 训练、策略复现及闭环评估 | [7 · 在仿真中合成机器人操作数据](https://developer.amd.com.cn/radeon/templates/7/preview)：使用 Genesis 生成操作数据、训练 SmolVLA 并进行闭环评估。<br>[1194 · ROCm Embodied AI Policy Replication](https://developer.amd.com.cn/radeon/templates/1194/preview)：检查 ROCm 设备、缓存和权限，为具身策略复现实验准备环境。 |
| **通用与测试模板（2）**<br>空白环境与工程测试入口 | [46 · amd-test-git](https://developer.amd.com.cn/radeon/templates/46/preview)：AMD 测试或 Git 集成模板；**未配置 Notebook 路径，具体功能需核验**。<br>[8 · Blank Notebook](https://developer.amd.com.cn/radeon/templates/8/preview)：空白 Notebook 环境；**未配置 Notebook 路径**。 |
| **合计（41）** | 每个模板仅归入一个主要类别。 |

## 三、建议实践顺序

1. **扩散模型**：先完成 [5051 图像生成](https://developer.amd.com.cn/radeon/templates/5051/preview)，再完成 [5052 视频生成](https://developer.amd.com.cn/radeon/templates/5052/preview)，最后按兴趣扩展 [1196 LTX 音视频](https://developer.amd.com.cn/radeon/templates/1196/preview)。
2. **AI 智能体**：先完成 [5048 Agents 101](https://developer.amd.com.cn/radeon/templates/5048/preview)，再完成 [5061 Agents 201](https://developer.amd.com.cn/radeon/templates/5061/preview)；[5037](https://developer.amd.com.cn/radeon/templates/5037/preview)、[29](https://developer.amd.com.cn/radeon/templates/29/preview) 用于补本地 vLLM 服务。
3. **LLM 基础与优化**：先完成 [4978 HF on AMD](https://developer.amd.com.cn/radeon/templates/4978/preview)，再完成 [5018 AI on AMD](https://developer.amd.com.cn/radeon/templates/5018/preview)；[3102](https://developer.amd.com.cn/radeon/templates/3102/preview) 只补 RAG 的向量检索环节，量化仍需另行设计完整实验。
4. 三门课程完成后，再按求职主线选择 vLLM 服务、性能分析、Kernel、微调或具身智能方向的延伸 Notebook。

## 四、信息口径与实践边界

本清单依据 [Radeon Cloud Gallery](https://developer.amd.com.cn/radeon/) 于 2026-10-25 的模板快照整理，共 41 个模板：34 个模板具有可读取的 Notebook 路径和内容，7 个模板未配置 Notebook 路径。原始抓取文本、元数据和检查脚本属于整理过程中的中间材料，已清理；需要查看模板时可直接打开表格中的 Gallery 链接。

“内容可读”只表示可以检查教学内容，不能证明模型下载、依赖安装、GPU 运行和实验结果已经成功。实践完成后，在 [notebook_ROCm/](notebook_ROCm/) 新建记录，并把成功依据、结果产物和阻塞问题补回课程进度表。
