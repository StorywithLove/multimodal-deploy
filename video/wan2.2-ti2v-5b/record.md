# Wan2.2-TI2V-5B 部署记录

## 基本信息

- 模型 ID：`Wan-AI/Wan2.2-TI2V-5B`
- 任务：文生视频 + 图生视频（5B 稠密模型，Wan2.2-VAE 压缩比 16×16×4，720P@24fps）
- 官方仓库：https://github.com/Wan-Video/Wan2.2
- 模型页：https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B
- 大小：约 15GB
- 状态：待开始

## 运行环境

- 平台：魔搭 ModelScope（AMD GPU 实例）
- GPU：AMD Instinct MI300X 192GB（gfx942）
- 软件：ROCm + PyTorch（镜像预装）
- 日期：待填

## 关键命令

```bash
export MODELSCOPE_CACHE=/root/tmp/modelscope
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir /root/tmp/Wan2.2-TI2V-5B

git clone https://github.com/Wan-Video/Wan2.2.git /root/tmp/Wan2.2
cd /root/tmp/Wan2.2
grep -v flash_attn requirements.txt > req_amd.txt
pip install -r req_amd.txt
sed -i 's/from .attention import flash_attention/from .attention import attention as flash_attention/' wan/modules/model.py

# 文生视频
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir /root/tmp/Wan2.2-TI2V-5B \
  --convert_model_dtype --offload_model True \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file /mnt/workspace/outputs/wan2.2_ti2v_demo.mp4
```

## 踩坑记录

- 同 T2V-A14B：ROCm 无 flash-attn，需改 `wan/modules/model.py` 走 SDPA 回退。
- `--size` 用 `1280*704` 或 `704*1280`。

## 产出

- `outputs/wan2.2_ti2v_demo.mp4`
- 生成耗时 / 显存峰值：待填

## 测评

- 与 T2V-A14B 的对比（速度 / 质量 / 显存）：待填

## 下一步

- 跑通后填写并更新根 README 状态表。
