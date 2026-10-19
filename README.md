# Inference Engineering Portfolio / 推理工程作品集

面向 AMD MI300X / ROCm 的模型部署与推理测评作品集。仓库沉淀**已经验证的部署路径、共用基准工具、结果与工程坑点**；Notion 保留学习主线、取舍和下一步决策。

这里追问的不是“哪个框架绝对最快”，而是：

> 在同一硬件、可比模型表示和明确服务负载下，什么时候该选择 llama.cpp、vLLM，后续什么时候值得引入 SGLang？

## 项目路线

- 根目录负责项目级索引、公开边界和共享测评方法。
- 语言模型路线的具体主线与框架选择见 [llm/README.md](llm/README.md)。
- Qwen3.8-Flash-Next 的 UD-Q4_K_XL GGUF / llama.cpp 部署步骤、异常处理和实验待办见 [llama.cpp.md](llm/qwen3.8-flash-next/llama.cpp.md)；原生 FP8 Safetensors / vLLM 的构建与启动记录见 [vllm.md](llm/qwen3.8-flash-next/vllm.md)，SGLang 使用独立的后端文档。
- 统一测评器只连接已启动的 OpenAI-compatible 服务，配置、脚本和结果之间的关系见 [benchmarks/README.md](benchmarks/README.md)。

## 仓库结构

```text
multimodal-deploy/
├─ llm/                         # 语言模型的部署记录与结论
│  └─ qwen3.8-flash-next/
│     ├─ llama.cpp.md            # llama.cpp + GGUF 部署与实验
│     ├─ vllm.md                 # vLLM + FP8 Safetensors
│     └─ sglang.md               # SGLang + FP8 Safetensors
├─ vision/                       # 图片模型路线
├─ video/                        # 视频模型路线与既有产出
├─ benchmarks/
│  ├─ openai_server_matrix.py    # llama.cpp / vLLM / SGLang 共用压测器
│  └─ README.md
├─ configs/benchmark/            # 版本化的工作负载契约
├─ results/                      # 小型 JSON 汇总；大日志和模型不入库
├─ docs/runs/                    # 可公开的运行索引
└─ lessons/                      # 可复用的兼容性、网络与部署经验
```

## 记录与测评如何协作

| 位置 | 回答的问题 | 产物 |
|---|---|---|
| 每个后端的部署文档 | 如何部署、改了什么、为什么这样选、结果意味着什么 | 简洁操作路径、异常处理、结论 |
| `configs/benchmark/` | 什么算作一次可比的负载 | 上下文、并发、输出长度、请求数 |
| `benchmarks/` | 如何一致地测量 | 通用 OpenAI-compatible 客户端与采样逻辑 |
| `results/` | 实际发生了什么 | 带环境与聚合指标的 JSON 结果 |

因此，模型文档不是重复粘贴压测日志；它链接配置和结果，并解释结论。基准脚本也不绑定某一个推理后端，只要求已运行的 OpenAI-compatible 服务端点。

## 当前状态

<table>
<thead>
<tr>
<th>模态</th>
<th>模型</th>
<th>分支 / 路线</th>
<th>任务</th>
<th>状态</th>
<th>运行环境</th>
<th>记录</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">语言</td>
<td rowspan="2">Qwen3.8-Flash-Next</td>
<td>UD-Q4_K_XL / GGUF + llama.cpp</td>
<td>文本 LLM 推理与服务</td>
<td>llama.cpp E1–E4 已完成</td>
<td>ModelScope AMD MI300X 192GB</td>
<td><a href="llm/qwen3.8-flash-next/llama.cpp.md">llama.cpp</a></td>
</tr>
<tr>
<td>FP8 Safetensors + vLLM</td>
<td>文本 LLM 推理与服务</td>
<td>vLLM E1–E4 已完成</td>
<td>AMD MI300X 192GB / ROCm 7.2.3</td>
<td><a href="llm/qwen3.8-flash-next/vllm.md">vLLM</a></td>
</tr>
<tr>
<td>视频</td>
<td>Wan2.2-T2V-A14B</td>
<td>T2V</td>
<td>文生视频</td>
<td>已跑通</td>
<td>ModelScope AMD MI300X（ROCm）</td>
<td><a href="video/wan2.2-t2v-a14b/record.md">record</a></td>
</tr>
<tr>
<td>视频</td>
<td>Wan2.2-TI2V-5B</td>
<td>TI2V</td>
<td>文 / 图生视频</td>
<td>待开始</td>
<td>ModelScope AMD MI300X（ROCm）</td>
<td><a href="video/wan2.2-ti2v-5b/record.md">record</a></td>
</tr>
<tr>
<td>图片 / 视觉</td>
<td>待选模型</td>
<td>待选路线</td>
<td>图像生成、视觉理解与微调</td>
<td>规划中</td>
<td>待定</td>
<td><a href="vision/README.md">vision</a></td>
</tr>
</tbody>
</table>

状态口径：

- **已验证**：至少完成真实环境的启动与目标接口 / 产出检查。
- **有基准**：已提交配置和结构化结果，能与后续后端比较。
- **能解释**：能说明资源瓶颈、质量取舍和框架选择边界。

## 公开与维护边界

- 不提交模型权重、视频 / 图片大文件、临时日志、云端 URL、Cookie 或访问令牌。
- 运行 API 示例前，通过环境变量注入凭据，例如 `MODELSCOPE_API_TOKEN`，不要把令牌写入源文件。
- 对外发布前仍需确定代码托管平台、仓库可见性和许可证；本地 Git 仓库可在这些选择确定后直接推送。
