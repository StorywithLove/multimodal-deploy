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

`<model-name>/` 下按推理后端分开记录部署文档；共同的测试契约仍放在仓库根目录：

- `llama.cpp.md`：本模型的 `llama.cpp + GGUF` 路线与 E1–E4 结果；
- `vllm.md`：本模型的 `vLLM + FP8 Safetensors` 路线；
- `sglang.md`：本模型的 `SGLang + FP8 Safetensors` 路线。

三份文档共享硬件和基准方法，但每份文档都必须在开头写明后端、权重格式、精度和模型目录，避免把 GGUF Q4/Q5 与原生 FP8 误当成同一权重的框架对照。

性能压测不在每个文档里重新实现。所有支持 OpenAI-compatible API 的后端共用仓库根目录的 `benchmarks/openai_server_matrix.py` 与 `configs/benchmark/`；结构化结果保存到 `results/`。

## 已部署与计划

| 模型 | 任务 | 后端 / 格式 | 状态 | 运行环境 | 记录 |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next / UD-Q4_K_XL + UD-Q5_K_XL | 文本 LLM 推理与服务 | llama.cpp ROCm / GGUF | E1–E4 已完成 | AMD MI300X 192GB | [llama.cpp](qwen3.8-flash-next/llama.cpp.md) |
| Qwen3.8-Flash-Next / FP8 | 文本 LLM 推理与服务 | vLLM / Safetensors FP8 | v0.31.0 源码构建完成；`--enforce-eager` 启动、健康检查和单请求通过；默认 V1 编译路径仍有动态形状错误 | AMD MI300X 192GB | [vLLM](qwen3.8-flash-next/vllm.md) |
| Qwen3.8-Flash-Next / FP8 | 文本 LLM 推理与服务 | SGLang / Safetensors FP8 | 待 MI300X 启动预检 | AMD MI300X 192GB | [SGLang](qwen3.8-flash-next/sglang.md) |

后续顺序：先完成该 GGUF 路线的 E1–E4，再用同一官方 FP8 Safetensors 权重分别做 vLLM 与 SGLang 的单请求预检；只有 E1 参数和启动结果稳定后，才扩展到 E2/E3。
