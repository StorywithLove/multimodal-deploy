# A10 24GB 模型候选清单（2026-10-19）

目标是让单张 NVIDIA A10 24GB 先跑出稳定的文本、图片和视频基线。显存预算不要按 24GB 全部占满：文本模型要为 KV cache 和 CUDA workspace 留余量，图像/视频模型要为 VAE、文本编码器和临时张量留余量。

## 结论先行

| 任务 | 首选 | 运行时 | A10 24GB 判断 |
|---|---|---|---|
| 文本生成 | `Qwen/Qwen3-14B-AWQ` | vLLM | 推荐平衡点；先从 8K context、单请求开始 |
| 文本生成 / 本地服务 | `Qwen/Qwen3-14B-GGUF:Q4_K_M` | llama.cpp | 推荐低风险路线；可逐步测 Q4/Q5 |
| 文本生成 / 更强尝试 | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | vLLM | 30.5B 总参数、3.3B 激活；只建议单并发、4K–8K context 压测 |
| 文生图 | `Qwen/Qwen-Image` | Diffusers + TorchAO `int8wo` + CPU offload | 官方 README 有 24GB 配方，优先级高；不是 vLLM 模型 |
| 文生图 / 基线 | `stabilityai/stable-diffusion-xl-base-1.0` | Diffusers / ComfyUI | 权重规模较小，适合作为最容易跑通的图像基线 |
| 文/图生视频 | `Wan-AI/Wan2.2-TI2V-5B` | Wan2.2 官方代码 | 官方命令以至少 24GB GPU 为目标，需要 offload、dtype conversion、T5 CPU |
| 文/图生视频 | `zai-org/CogVideoX-2b` | Diffusers | 官方支持 INT8/TorchAO 与 CPU offload，适合第一条视频基线 |
| 文/图生视频 | `ltxv-2b-0.9.8-distilled` 或对应 FP8/Q8 变体 | LTX-Video / ComfyUI | 轻量、适合快速迭代；13B 版本放到第二阶段 |
| 低风险视频 | `Wan2.1-T2V-1.3B` | Wan2.1 / Diffusers | 先做 480P；官方建议单卡低显存场景开启 offload 和 T5 CPU |
| 暂不建议 | `Wan2.2-T2V-A14B` | Wan2.2 | 官方 README 以至少 80GB GPU 为目标，A10 不应作为首轮目标 |

## 文本生成

### 1. Qwen3-14B-AWQ：首个 vLLM 候选

官方模型卡把它标为 AWQ 4-bit，原生上下文 32768；官方也给出了 vLLM serve 方式。A10 属于 Ampere，当前 vLLM 量化兼容表对 Ampere 标记了 AWQ、GPTQ 和 GGUF 支持，但具体 vLLM 版本和 kernel 仍要在实例上做 smoke test。

```bash
vllm serve Qwen/Qwen3-14B-AWQ \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90
```

首轮固定 `max_model_len=8192`，确认 `/health`、chat、实际显存和 8 请求成功率，再扩展到 16K 或并发 2。不要一开始打开 131K YaRN；Qwen 官方也提示，短于 32K 的常规工作负载不必启用 YaRN。

### 2. Qwen3-14B-GGUF：llama.cpp 路线

官方 GGUF 页面列出 `Q4_K_M`、`Q5_K_M`、`Q6_K` 等量化，并提供 `llama-server -hf Qwen/Qwen3-14B-GGUF:Q4_K_M` 的启动方式。它适合在 A10 上做“模型尺寸、量化、上下文、吞吐”的低风险测评。

```bash
llama-server -hf Qwen/Qwen3-14B-GGUF:Q4_K_M \
  -ngl 999 -c 8192 --parallel 1 \
  --host 127.0.0.1 --port 8000
```

建议顺序是 Q4_K_M → Q5_K_M，而不是直接上 Q8。量化版本比较时沿用本项目的 E1/E4 口径：实际 token、TTFT、completion tok/s、HBM/显存峰值、启动到 health 的时间和固定小型质量题集。

### 3. Qwen3-30B-A3B-GPTQ-Int4：第二阶段压力点

这是 30.5B 总参数、3.3B 激活专家的 MoE，官方提供 GPTQ 4-bit 和 vLLM serve 示例。它的“激活参数少”不等于显存只按 3.3B 计算：权重、KV cache、workspace 和上下文仍需要预算。因此在 A10 上只把它作为实验候选：`max_num_seqs=1`、4K context 起步，必要时开启 CPU offload；若启动即 OOM，先回退到 Qwen3-14B-AWQ。

## 文生图

### 1. Qwen-Image：最值得先试的 24GB 图像模型

Qwen 官方模型卡把 Qwen-Image 标为 20B BF16 文生图模型，但官方 README 另提供了受限显存方案：24GB GPU 使用 TorchAO `int8wo` 量化 Transformer，并启用 `pipe.enable_model_cpu_offload()`；16GB 方案则使用 bitsandbytes NF4。这个路径是 Diffusers/torchao，不是 vLLM serving。

首轮建议使用官方 24GB 配方，但将分辨率从 1664×928 保守降到 1024×1024，确认单图成功和峰值显存后再提高分辨率。A10 与官方示例硬件不同，所以“官方有 24GB 配方”是可行性依据，不是对 A10 性能的保证。

### 2. SDXL：最稳妥的图像基线

`stabilityai/stable-diffusion-xl-base-1.0` 官方模型卡提供 Diffusers 用法，仓库中的基础权重规模远小于 Qwen-Image，A10 24GB 上通常不需要先做激进量化。它适合作为图像管线、采样器、VAE、分辨率和显存采样的基线；跑通后再比较 Qwen-Image 的文字渲染和中文能力。

`FLUX.1-schnell` 不作为第一轮：官方 transformer 文件约 23.8GB，尚未算 VAE、文本编码器和运行时 workspace。即使使用社区 NF4/GGUF，也应把 CPU offload 和单图低分辨率作为前提，不能按原始 BF16 checkpoint 直接套到 24GB。

## 视频大模型

### 1. Wan2.2-TI2V-5B：高质量优先

Wan2.2 官方仓库明确给出 `ti2v-5B` 的单 GPU 命令，并说明 1280×704 的任务可在至少 24GB VRAM 的 GPU 上运行；命令包含 `--offload_model True --convert_model_dtype --t5_cpu`。这与 A10 24GB 的约束匹配，但 A10 的速度会低于 4090，因此第一轮应接受较长生成时间，并准备降低分辨率或帧数。

```bash
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --prompt "A small robot walks through a rainy Wuhan street at night." \
  --save_file output.mp4
```

仓库已有 [Wan2.2-TI2V-5B 部署脚本](../../video/wan2.2-ti2v-5b/deploy.sh) 和 [记录模板](../../video/wan2.2-ti2v-5b/record.md)，但它们目前记录的是 MI300X/ROCm 路线。迁移到 A10 时要单独记录 CUDA 依赖、峰值显存和生成耗时，不要把 AMD 结果移植成 NVIDIA 结论。

### 2. CogVideoX-2B：最容易形成视频基线

官方模型卡给出 Diffusers FP16 和 INT8/TorchAO 的显存优化方式，2B 版本定位为兼容性和运行成本优先，视频规格为 6 秒、720×480、8 FPS。它比 5B/14B 更适合 A10 首轮验证，先跑 INT8 + model CPU offload，再视显存余量切回 FP16。

### 3. LTX-Video 2B distilled：快速迭代

官方仓库将 2B distilled 版本定位为较轻量、适合快速生成，并列出 13B 与 2B 的 FP8 变体；仓库还说明 8-bit LTX-VideoQ8 可以在 8GB RTX 4060 上生成 720×480×121。对 A10 而言，2B distilled/Q8 更适合做低延迟预览，13B distilled 则作为第二阶段的质量尝试。

### 4. Wan2.1-T2V-1.3B：低风险回退

Wan2.1 官方 README 建议 1.3B 版本主要使用 480P，并对单张消费级 GPU 建议开启 model offload 和 T5 CPU。它的画质上限不如 5B/14B，但适合先验证视频环境、输出编码、帧数和显存采样链路。

## 推荐实验顺序

1. **文本**：Qwen3-8B-AWQ 做 vLLM 基线，随后 Qwen3-14B-AWQ；并行保留 Qwen3-14B-GGUF Q4_K_M 的 llama.cpp 路线。
2. **图片**：SDXL 做最稳妥的管线基线，再跑 Qwen-Image 官方 24GB TorchAO + CPU offload 配方。
3. **视频**：CogVideoX-2B 或 Wan2.1-T2V-1.3B 先验证链路；再用 Wan2.2-TI2V-5B 做质量路线。
4. **压力测试**：最后才尝试 Qwen3-30B-A3B-GPTQ-Int4、LTX-Video 13B FP8 或社区 Wan2.1 14B GGUF；这些都应标记为“候选/实验中”，不能预先写成已跑通。

每个模型至少保存：模型 ID 与 revision、量化格式、启动命令、GPU 峰值、首次输出/首帧时间、总耗时、输出规格、失败日志和是否启用 CPU offload。这样 A10 的结果才可以和 MI300X 的推理基础设施实验形成同一套作品证据。

## 主要官方参考

- [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ)
- [Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ)
- [Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4)
- [Qwen3-14B-GGUF](https://huggingface.co/Qwen/Qwen3-14B-GGUF)
- [Qwen-Image 24GB quantization recipe](https://huggingface.co/Qwen/Qwen-Image/blob/refs%2Fpr%2F27/README.md)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Wan2.2 official repository](https://github.com/Wan-Video/Wan2.2)
- [CogVideoX-5B model card](https://huggingface.co/THUDM/CogVideoX-5b)
- [LTX-Video official repository](https://github.com/Lightricks/LTX-Video)
