---
title: Qwen3.8-Flash-Next 在 AMD MI300X 上的可复现部署：从 GGUF 到 vLLM FP8
date: 2026-10-19
tags:
  - AMD GPU激励计划
  - AMD GPU
  - ROCm
  - vLLM
  - llama.cpp
  - Qwen3.8-Flash-Next
status: draft
---

# Qwen3.8-Flash-Next 在 AMD MI300X 上的可复现部署：从 GGUF 到 vLLM FP8

> 本文是一份可复现的 AMD GPU 部署与测评记录。所有性能数字都来自仓库中的原始 JSON 和固定脚本；没有把不同权重格式、不同框架的结果包装成“纯框架排名”。

## 简介

Qwen3.8-Flash-Next 具备较大的模型规模和较长的上下文能力，在 AMD GPU 上部署时，既要关注模型能否成功加载，也要关注权重格式、推理框架、显存占用、并发吞吐和长上下文延迟之间的实际取舍。本文记录一次基于单张 AMD Instinct MI300X 192GB 的完整实践：先使用 `llama.cpp` 加载 GGUF 量化模型建立可运行基线，再使用 ROCm 环境下的 vLLM 加载 FP8 Safetensors，最后通过统一的 OpenAI-compatible 接口和固定负载进行对照测评。

文章重点不在于给出一个脱离环境的“最快框架”结论，而在于展示如何从环境检查、服务启动、健康探针、故障定位到 E1–E4 测评逐步建立可复现证据。读者可以据此复用启动参数、测评方法和结果判读思路，并清楚区分已完成的 llama.cpp/vLLM 实验与尚未完成的 SGLang 路线。

## 摘要

这次实践使用 ModelScope AMD 实例上的单张 AMD Instinct MI300X 192GB，目标是把 `Qwen/Qwen3.8-Flash-Next` 分别跑通为：

1. `llama.cpp + GGUF UD-Q4_K_XL`：先验证模型加载、OpenAI-compatible API、Web UI 和统一测评脚本；
2. `vLLM 0.31.0 + FP8 Safetensors`：在 ROCm 7.2.3 / `gfx942` 上从源码构建，定位 Torch Dynamo 的 GDN/PLE 动态形状故障，并在补丁后跑通 compiled O1/O2；
3. SGLang FP8：原计划在本日继续验证，但受 192GB AMD GPU 资源紧张影响，延后到下一轮，不把它写成已完成结果。

最终得到的不是一句“能启动”，而是一套包括启动参数、健康检查、固定负载、显存采样、请求成功率、原始结果和复现命令的证据链。

结果图例：✅ 已通过 / 请求成功；⚠️ 有性能拐点或适用边界；🔬 单轮观测或需要进一步验证；❌ 失败。

## 1. 为什么要同时保留两条路线

`llama.cpp` 路线使用 Unsloth 的 `UD-Q4_K_XL` GGUF，下载体积约 104 GiB；本次启动日志识别为 `text` 模态，因此只覆盖文本推理和服务接口。`vLLM` 路线使用官方 FP8 Safetensors，模型目录约 173 GiB，目标是验证原生 FP8 serving 路线。

这两条路线同时改变了后端和权重精度，所以结论应表述为“部署路线与精度配置对照”，不能直接说成 llama.cpp 与 vLLM 的纯框架胜负。图片、视频和视觉微调属于独立路线，不能从本次 GGUF 文本实验外推。

## 2. 环境与版本

| 项目 | 配置 |
|---|---|
| GPU | AMD Instinct MI300X，192GB HBM |
| GPU 架构 | `gfx942` |
| ROCm | 7.2.3 |
| vLLM | 0.31.0，源码构建 |
| FP8 权重 | `Qwen/Qwen3.8-Flash-Next-FP8`，细粒度 FP8，block size 128 |
| GGUF 权重 | `UD-Q4_K_XL`，另测 `UD-Q5_K_XL` |
| 服务协议 | OpenAI-compatible API |
| 测评 | E1 单请求、E2 并发、E3 长上下文、E4 量化与质量抽样 |

仓库中的 [llama.cpp 部署笔记](https://github.com/StorywithLove/multimodal-deploy/blob/main/llm/qwen3.8-flash-next/llama.cpp.md) 记录了 GGUF 下载、ROCm 编译、UI 和服务端细节；[vLLM 部署笔记](https://github.com/StorywithLove/multimodal-deploy/blob/main/llm/qwen3.8-flash-next/vllm.md) 记录了源码构建、补丁、启动档位和复测命令。

## 3. llama.cpp + GGUF：先建立可运行基线

核心启动方式如下，实际路径按实例环境替换：

```bash
llama-server \
  -m /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl 999 -c 10240 --parallel 1 --jinja \
  --host 127.0.0.1 --port 8000
```

启动后先验证 `/health` 和 `/v1/models`，再运行仓库的统一 OpenAI-compatible 测评器。这样可以把“模型能回答”与“性能数据可复现”分开。

### E1：单请求基线

| 目标上下文 | 实际 Prompt | TTFT | P50 / P95 | Completion 吞吐 | HBM 峰值 | 结果 |
|---:|---:|---:|---:|---:|---:|:---:|
| 512 | 433 | 78.9 ms | 6.481 / 6.527 s | 39.31 tok/s | 77.56 GiB | ✅ 8/8 |
| 4096 | 2,801 | 83.0 ms | 6.644 / 6.711 s | 38.32 tok/s | 77.58 GiB | ✅ 8/8 |
| 8192 | 5,507 | 91.4 ms | 6.763 / 6.804 s | 37.67 tok/s | 77.56 GiB | ✅ 8/8 |

⚠️ 随目标上下文增加，TTFT 约从 79 ms 增至 91 ms，completion 吞吐从 39.31 降至 37.67 tok/s；HBM 峰值稳定在约 77.6 GiB。

### E2：并发拐点

| 客户端并发 | Completion 吞吐（512 / 4096 / 8192，tok/s） | 各上下文 P95 最大值 | HBM 峰值范围 | 结果 |
|---:|---:|---:|---:|:---:|
| 1 | 38.58 / 35.52 / 34.81 | 9.167 s | 80.392–80.414 GiB | ✅ |
| 2 | 54.10 / 63.65 / 62.28 | 14.276 s | 80.397–80.418 GiB | ✅ |
| 4 | 99.17 / 72.56 / 69.20 | 18.386 s | 80.373–80.396 GiB | ✅ |
| 8 | 117.92 / 70.02 / 65.10 | 31.246 s | 80.383–80.393 GiB | ✅ |

⚠️ 三个上下文平均 completion 吞吐约为 36.30、60.01、80.31、84.35 tok/s；并发从 4 增至 8 时吞吐只小幅增加，但最差 P95 从 18.386 s 增至 31.246 s，因此当前配置的高效档位更接近客户端并发 4。

### E3：长上下文

| 目标上下文 | 实际 Prompt | TTFT | P50 / P95 | Completion 吞吐 | HBM 峰值 | 结果 |
|---:|---:|---:|---:|---:|---:|:---:|
| 8192 | 5,507 | 89.0 ms | 6.757 / 6.780 s | 37.790 tok/s | 79.26 GiB | ✅ 8/8 |
| 32768 | 21,740 | 123.0 ms | 7.079 / 7.091 s | 36.048 tok/s | 79.26 GiB | ✅ 8/8 |
| 65536 | 43,385 | 186.3 ms | 7.513 / 7.527 s | 33.925 tok/s | 79.26 GiB | ✅ 8/8 |

🔬 单 Slot 下 24/24 请求成功；实际最长 Prompt 为 43,385 tokens，completion 吞吐从 37.790 降至 33.925 tok/s，HBM 维持在约 79.24 GiB。

把 32768 上下文扩展到客户端并发 4 后，⚠️ completion 吞吐降至 20.213 tok/s，P95 升至 89.116 s。这说明长上下文下的主要风险是 Prefill/KV Cache 调度竞争，而不是本轮观察到的 OOM。HBM 采样只能提供方向性证据，不能精确拆分权重和 KV Cache。

| 32768 上下文补充 | TTFT | P50 / P95 | Completion 吞吐 | HBM 峰值 | 结果 |
|---|---:|---:|---:|---:|:---:|
| 客户端并发 1 | 135.2 ms | 7.068 / 7.095 s | 36.060 tok/s | 81.395 GiB | ✅ 8/8 |
| 客户端并发 4 | 123.4 ms | 49.922 / 89.116 s | 20.213 tok/s | 81.400 GiB | ⚠️ 8/8 |

## 4. E4：Q4 与 Q5——量化不是越大越好

在相同 `llama-server` 参数、同一 Mini-8 质量题集和 4096 输出上限下：

| 指标 | `UD-Q4_K_XL` | `UD-Q5_K_XL` |
|---|---:|---:|
| E4 serving completion 吞吐 | 约 38.04 tok/s | 约 38.31 tok/s |
| HBM 峰值 | 约 77.6 GiB | 约 97.4 GiB |
| 启动到 `/health` 中位数 | 81.593 s | 102.444 s |
| Mini-8 成功 | ✅ 8/8 | ✅ 8/8 |
| 4096 轮长度截断 | ✅ 0/8 | ✅ 0/8 |

结果：✅ 两种量化的 4096 轮均 8/8 成功且无长度截断；⚠️ Q5 的单位 token 速度只高约 0.7%，但 HBM 多约 19.9 GiB，启动中位数高约 25.6%。🔬 本轮 Mini-8 没有证明 Q5 有稳定质量优势，因此当前 MI300X + llama.cpp 配置优先选 Q4；Q5 只作为质量敏感场景的备选，而不是默认升级。

## 5. vLLM FP8：故障定位比“一条命令成功”更重要

预编译 ROCm wheel 受宿主机 glibc 2.35 与 wheel 标签要求不匹配，Docker 又没有可用 daemon，因此最后选择在当前 ROCm 7.2.3 / `gfx942` 环境中从源码构建 vLLM 0.31.0。

实际排障链如下：

1. vLLM 0.30.0 源码构建后 EngineCore 在 600 秒内没有 ready；
2. 升级到 0.31.0 后，eager 路径可以启动；
3. 默认 compiled 路径在 Qwen3.8-Flash-Next 的 GDN/PLE 动态形状处理中出现 Torch Dynamo guard / reshape 错误；
4. 最终补丁固定 PLE 输入 buffer 形状，移除 `query_start_loc` / `ngram_context` 的动态维声明，并把 GDN reshape 中的 `-1` 改为显式 head 数；
5. O1、O2 compiled 服务均通过 `/health`、`/v1/models` 和 chat 冒烟，实例重启后 O2 也能从持久环境恢复。

O2 的首启命令：

```bash
export VLLM_PY=/mnt/workspace/artifacts/qwen3.8-flash-next-vllm/vllm0310-20261017/venv-vllm-0310/bin/python
export MODEL_DIR=/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-FP8
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export PYTORCH_ROCM_ARCH=gfx942
export VLLM_ENGINE_READY_TIMEOUT_S=1200

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3.8-flash-next-fp8 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --dtype auto \
  --gpu-memory-utilization 0.92 --cpu-offload-gb 8 \
  --max-model-len 8192 --max-num-seqs 2 \
  --optimization-level 2
```

健康检查不应只看进程是否存在：至少要确认 `/health` 返回 200、模型列表可读、chat 请求返回 HTTP 200 且 `finish_reason=stop`。

## 6. vLLM 测评结果

### E1：O1/O2 单请求

| 配置 / 目标上下文 | 实际 Prompt | Prompt tok/s | Completion tok/s | TTFT | P50 / P95 | 结果 |
|---|---:|---:|---:|---:|---:|:---:|
| O1 / 512 | 433 | 48.06 | 28.42 | 0.324 s | 8.956 / 9.015 s | ✅ 8/8 |
| O2 / 512 | 433 | 60.84 | 35.97 | 0.311 s | 7.061 / 7.089 s | ✅ 8/8 |
| O2 / 4096 | 2,801 | 377.66 | 34.52 | 0.548 s | 7.331 / 7.355 s | ✅ 8/8 |
| O2 / 8192 | 5,507 | 757.57 | 35.22 | 0.417 s | 7.203 / 7.222 s | ✅ 8/8 |

🔬 各测评单元均为 8/8 成功；O1/O2 只有单轮数据，不能据此宣称 O2 的稳定性能优势，但 O2 是当前可复现的默认服务档位。

### E2：并发

| 目标上下文（实际 Prompt） | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| 512（433） | 35.91 / 7.07 s | 55.44 / 8.92 s | 87.33 / 11.51 s | 116.77 / 17.13 s |
| 4096（2,801） | 34.50 / 7.34 s | 53.66 / 9.40 s | 79.07 / 12.62 s | 101.77 / 19.48 s |
| 8192（5,507） | 35.20 / 7.21 s | 55.46 / 9.12 s | 82.25 / 12.17 s | 108.58 / 18.36 s |

✅ 全矩阵 96/96 请求成功。⚠️ 单元格格式为 `completion tok/s / 请求 P50`；并发增加提升聚合吞吐，同时提高单请求延迟。

### E3：长上下文

| 目标上下文（实际 Prompt） | Prompt tok/s | Completion tok/s | TTFT | P50 / P95 | 结果 |
|---|---:|---:|---:|---:|:---:|
| 8192（5,507） | 729.52 | 33.91 | 0.457 s | 7.48 / 7.50 s | ✅ 8/8 |
| 32768（21,740） | 2762.22 | 32.53 | 0.734 s | 7.77 / 7.78 s | ✅ 8/8 |
| 65536（43,385） | 5587.96 | 32.97 | 0.625 s | 7.67 / 7.70 s | ✅ 8/8 |

| 32768 补充 | Prompt tok/s | Completion tok/s | TTFT | P50 / P95 | 结果 |
|---|---:|---:|---:|---:|:---:|
| 客户端并发 4 | 6138.96 | 72.29 | 0.730 s | 13.735 / 14.219 s | ✅ 8/8 |

🔬 单并发全矩阵 24/24 请求成功，补充单元 8/8 成功；⚠️ 补充单元吞吐约 72.29 tok/s，但 P50/P95 延迟升至约 13.735 / 14.219 s。

### E4：FP8 启动与质量补充

| 项目 | 结果 | 判读 |
|---|---|---|
| O2 E1 配置启动就绪 | ✅ 3/3，平均 202.577 s，中位数 202.098 s | 含权重加载、编译、KV Cache 分配与服务初始化，不等同于 TTFT |
| 默认 thinking 的 Mini-8 | ✅ 8/8，0 空回答，0 次 `length` | 🔬 接口可用，但 8/8 响应含 `</think>`，存在过程性文本前缀 |
| `enable_thinking=false` 的 Mini-8 | ✅ 8/8，0 空回答，0 次 `length` | ✅ 8/8 正文不含 `</think>` 或过程性前缀 |

⚠️ E4 证明 FP8 服务启动和质量探针可复现；Mini-8 只是可用性与输出行为抽样，不能替代人工质量评分，也不能与 llama.cpp Q4/Q5 结果归因成纯框架差异。

## 7. 可复现性与边界

本项目把“可复现”拆成四层：

- 环境：GPU 架构、ROCm、Torch/vLLM 版本、源码 commit 和权重格式；
- 服务：启动参数、端口、健康检查和模型列表；
- 负载：固定上下文、并发、输出上限、请求数和 warmup；
- 证据：原始 JSON、日志、结果摘要与复跑命令。

原始材料位于：

- [llama.cpp 部署与 E1–E4](https://github.com/StorywithLove/multimodal-deploy/blob/main/llm/qwen3.8-flash-next/llama.cpp.md)
- [vLLM 部署、补丁与 E1–E4](https://github.com/StorywithLove/multimodal-deploy/blob/main/llm/qwen3.8-flash-next/vllm.md)
- [统一 benchmark 说明](https://github.com/StorywithLove/multimodal-deploy/blob/main/benchmarks/README.md)
- [结果目录](https://github.com/StorywithLove/multimodal-deploy/tree/main/results/llm/qwen3.8-flash-next)
- [社区先例与活动规则调查](https://github.com/StorywithLove/multimodal-deploy/blob/main/llm/qwen3.8-flash-next/community-prior-art.md)

本文不声称补丁已经进入所有上游版本，也不声称当前数字代表所有 MI300X、所有 vLLM 版本或所有量化格式。SGLang FP8 与文本生成以外的多模态路线需要用同一套“先冒烟、再 benchmark”的纪律另行验证。

## 8. 与 AMD × ModelScope 激励计划的关系

按 2026-09-30 发布的 [AMD × ModelScope 开发者激励计划 2.0](https://www.modelscope.cn/posts/437059)，开发者可以在魔搭社区“开发者实践”发布原创 AMD GPU / ROCm 文章，带指定标签 `AMD GPU激励计划`，经审核通过后每篇获得 125 ADP 积分，每个 ID 最多 3 篇。符合条件的新 ADP 注册者另有一次性 100 小时 AMD GPU Notebook 权益；旧的普惠 700 小时 CreateSpace 路线已下线，具体资格仍以活动页面和账号状态为准。

因此本文的投稿目标是提交一份可审核的原创实践证据，而不是承诺“一篇文章直接兑换固定 GPU 小时”。投稿前还需要补齐：

1. 魔搭文章正文与 `AMD GPU激励计划` 标签；
2. 公开仓库链接、commit 和可脱敏的原始结果；
3. 环境、代码、结果、异常与解决方法的完整说明；
4. 明确区分已完成的 llama.cpp/vLLM 结果和计划中的 SGLang 结果；
5. 确认同一内容没有在新旧活动中重复申领。

## 结论

在 MI300X 192GB 上，`llama.cpp + UD-Q4_K_XL` 已形成稳定的文本 serving 基线；`vLLM 0.31.0 + FP8` 在补丁后的 O2 compiled 路径上完成了从启动到长上下文测评的闭环。当前最有价值的产出不是某个单点 tok/s，而是能够解释失败原因、复现修复过程、保留原始证据，并且对不同精度路线的比较边界保持诚实。

下一步是用同一套健康检查和负载契约验证 SGLang FP8；A10 24GB 则优先从 Qwen3-14B-AWQ、Qwen-Image 24GB 量化配方和 Wan2.2-TI2V-5B 开始，而不是直接挑战需要更大显存的模型。
