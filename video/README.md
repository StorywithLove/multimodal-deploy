# 视频大模型（video）

视频生成与视频理解类模型部署与测评。运行环境：魔搭 ModelScope AMD GPU 实例（AMD Instinct MI300X 192GB，gfx942，ROCm + PyTorch 预装）。

## 约定

每个模型一个子目录，包含：

```
<model-name>/
├─ deploy.sh     # 部署/推理脚本（云端 Linux 执行）
├─ record.md     # 环境、命令、踩坑、产出、测评
└─ outputs/      # 产出视频与日志（大文件不入库）
```

## 模型清单

### 视频生成（通义万相 Wan2.2）

| 模型 | 任务 | 模型 ID | 大小 | 状态 | 记录 |
|---|---|---|---|---|---|
| Wan2.2-T2V-A14B | 文生视频（MoE） | `Wan-AI/Wan2.2-T2V-A14B` | ~126GB | 部署中 | [record](wan2.2-t2v-a14b/record.md) |
| Wan2.2-TI2V-5B | 文/图生视频 | `Wan-AI/Wan2.2-TI2V-5B` | ~15GB | 待开始 | [record](wan2.2-ti2v-5b/record.md) |

### 视频理解（VLM）

| 模型 | 任务 | 模型 ID | 状态 | 记录 |
|---|---|---|---|---|
| Qwen3-VL-72B | 视频/图像理解 | `Qwen/Qwen3-VL-72B-Instruct` | 待开始 | — |
| Qwen2.5-VL-72B | 视频/图像理解 | `Qwen/Qwen2.5-VL-72B-Instruct` | 待开始 | — |

## 通用避坑（ROCm）

- Wan2.2 官方代码 `wan/modules/model.py` **直接调用 `flash_attention(...)`**，而 `flash_attn` 在 ROCm 上需单独编译。
  解决：改为使用 `wan/modules/attention.py` 自带的 SDPA 回退封装 `attention()`（改 1 行 import），无需编译 flash-attn。
- `/tmp` 通常很小（本机仅 30G），模型权重请下载到空间更大的临时盘（本机为 `/root/tmp`，在 `/` 上，约 561G 空闲）。
- 权重是 fp32，务必加 `--convert_model_dtype` 转 bf16 提速省显存。
