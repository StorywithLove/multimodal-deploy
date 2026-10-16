# 语言模型 / LLM

这里记录语言模型的可验证部署与推理服务测评。当前重点不是堆叠模型数量，而是建立同一硬件上 llama.cpp、vLLM 与 SGLang 的选择边界。

## 当前主线

Qwen3.8-Flash-Next / Unsloth UD-Q4_K_XL 已在单张 ModelScope AMD MI300X 192GB（ROCm）上通过 ROCm 版 llama.cpp 完成：

- 文本 CLI 冒烟测试；
- OpenAI-compatible API；
- llama-server 内置 Web UI，经 DSW 登录态网关完成实际对话。

下一阶段先完成 llama.cpp 的四组基线，再以**相同工作负载契约**接入 vLLM，最后视服务特征引入 SGLang：

1. E1：单请求基线；
2. E2：并发 / 槽位；
3. E3：长上下文 / KV cache；
4. E4：量化质量、速度与显存权衡。

GGUF Q4 与原生 BF16、FP8 等权重格式不能被表述为“纯框架胜负”。跨格式结果是部署路线比较；只有在模型版本、精度、上下文和负载尽量对齐时，才是框架对照。

## 目录约定

`<model-name>/` 下的 `record.md` 分为两部分：

1. **模型部署 / Deployment**：必要环境、关键命令、异常与已验证的接口；
2. **实验测评 / Evaluation**：实验假设、待办、引用的统一配置和结果结论。

性能压测不在每个文档里重新实现。所有支持 OpenAI-compatible API 的后端共用仓库根目录的 `benchmarks/openai_server_matrix.py` 与 `configs/benchmark/`；结构化结果保存到 `results/`。

## 已部署与计划

| 模型 | 任务 | 后端 / 格式 | 状态 | 运行环境 | 记录 |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next / UD-Q4_K_XL | 文本 LLM 推理与服务 | llama.cpp ROCm / GGUF Q4 | CLI、API、Web UI 已验证；E1–E4 待测 | ModelScope AMD MI300X 192GB | [record](qwen3.8-flash-next/record.md) |

后续顺序：先完成该 GGUF 路线的 E1–E4，再准备可比的原生权重路径接入 vLLM；当连续批处理、前缀缓存或更高并发成为瓶颈时，引入 SGLang。
