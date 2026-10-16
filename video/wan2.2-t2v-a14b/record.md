# Wan2.2-T2V-A14B 部署记录

## 基本信息

- 模型 ID：`Wan-AI/Wan2.2-T2V-A14B`
- 任务：文生视频（Text-to-Video，MoE 双专家：high_noise + low_noise）
- 官方仓库：https://github.com/Wan-Video/Wan2.2
- 安装文档：https://github.com/Wan-Video/Wan2.2/blob/main/INSTALL.md
- 模型页：https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B
- 大小：约 126GB（两专家 fp32 各 ~57GB + T5 11GB + VAE 0.5GB）
- 状态：已跑通（2026-10-13 首跑；MoE 两专家 bf16，单卡 MI300X）（口径：已跑通 / 调过 / 能解释）

## 运行环境

- 平台：魔搭 ModelScope（AMD GPU 实例）
- GPU：AMD Instinct MI300X 192GB（gfx942）
- 软件：ROCm + PyTorch（镜像预装，勿升级 torch）
- 日期：2026-10-13

## 磁盘与下载位置

| 挂载点 | 大小 | 可用 | 用途 |
|---|---|---|---|
| `/`（rootfs） | 591G | 561G | **临时下载（本模型放这里，`/root/tmp`）** |
| `/tmp`（/dev/vda） | 30G | 28G | 太小，勿用 |
| `/mnt/data`（NFS） | 1.0P | 970T | 网络盘，慢；仅备份/持久产出 |
| `/etc/dsw/instance`（virtiofs） | 7.0T | 6.1T | 备用 |

选择：`export MODELSCOPE_CACHE=/root/tmp/modelscope`，权重下载到 `/root/tmp/Wan2.2-T2V-A14B`。

## 关键命令

```bash
# 自检
rocm-smi
python -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 下载（~126GB）
export MODELSCOPE_CACHE=/root/tmp/modelscope
modelscope download --model Wan-AI/Wan2.2-T2V-A14B --local_dir /root/tmp/Wan2.2-T2V-A14B

# 代码 + 依赖（跳过 flash_attn 与 numpy 钉版本）
git clone https://github.com/Wan-Video/Wan2.2.git /root/tmp/Wan2.2
cd /root/tmp/Wan2.2
grep -vE "flash_attn|numpy" requirements.txt > req_amd.txt
pip install -r req_amd.txt
pip install -U "numpy>=2,<3"    # 不要降级 numpy，否则预装 scipy 崩
# ROCm 无 flash-attn → 走 attention.py 的 SDPA 回退
sed -i 's/from .attention import flash_attention/from .attention import attention as flash_attention/' wan/modules/model.py
# 跳过 s2v/animate 的 decord/librosa 依赖（t2v 用不到）
sed -i -e 's/^from .speech2video import WanS2V/# &/' -e 's/^from .animate import WanAnimate/# &/' wan/__init__.py

# 生成
python generate.py --task t2v-A14B --size 1280*720 \
  --ckpt_dir /root/tmp/Wan2.2-T2V-A14B \
  --convert_model_dtype --offload_model False --t5_cpu \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file /mnt/workspace/outputs/wan2.2_t2v_demo.mp4

# 清理临时下载
rm -rf /root/tmp/Wan2.2-T2V-A14B /root/tmp/modelscope
```

## 踩坑记录

- **numpy 被降级导致 scipy 崩溃**：`requirements.txt` 钉 `numpy<2`，pip 把预装的 numpy 2.3.5 降到 1.26.4；预装 scipy 1.18.1 需要 numpy 2 → `AttributeError: module 'numpy' has no attribute 'long'`。
  解决：`pip install -U "numpy>=2,<3"`；安装依赖时跳过 numpy 那行（`grep -vE "flash_attn|numpy"`）。
  副作用：同环境里 transformers 也被从 5.15.1 降到 4.51.3，会破坏预装 vllm（vllm 0.28 要求 transformers>=5.5）；若两套都要用，建议 WAN 单独建 venv 隔离。
- **decord 缺失导致 import wan 失败**：`wan/__init__.py` 强制 `from .speech2video import WanS2V`（及 `.animate`），二者依赖 `decord`，而 decord 无 Python 3.12 wheel → `ModuleNotFoundError: No module named 'decord'`。
  解决（t2v 用不到 s2v/animate，直接跳过）：`sed -i -e 's/^from .speech2video import WanS2V/# &/' -e 's/^from .animate import WanAnimate/# &/' wan/__init__.py`
  备选：`pip install eva-decord librosa`（eva-decord 提供 decord，py3.12 有 wheel；但会牵入 numba、可能再动 numpy）。
- **flash-attn 阻塞**：`wan/modules/model.py` 直接调用 `flash_attention(...)`，ROCm 无 `flash_attn` 会 AssertionError。
  解决：改为 `from .attention import attention as flash_attention`，走 `attention.py` 的 SDPA 回退。副作用：SDPA 回退忽略 padding mask（本模型 `window_size=(-1,-1)`，t2v 场景影响很小）。
  性能版可选：编译 ROCm flash-attention（CK/Triton，`GPU_ARCHS=gfx942 python setup.py install`）。
- **磁盘选错**：`/tmp` 仅 30G，模型放不下；用 `/root/tmp`（在 `/` 上）。
- **fp32 权重**：不加 `--convert_model_dtype` 会更慢更吃显存；192GB 建议开 bf16，`--offload_model False` 提速。

## 产出

- 成片：云端 `/mnt/workspace/outputs/wan2.2_t2v_demo.mp4`（本地副本 `D:\用户资料\Downloads\wan2.2_t2v_demo.mp4`）
- 规格：1280*720、81 帧、16 fps → 时长约 5s（按 81/16）
- 生成耗时：约 70–82 分钟（122.52 s/it × 40 步，首跑；SDPA 回退，未装 flash-attn）
- 显存峰值：约 64GB（amd-smi：65592 / 196288 MB）
- 环境：MI300X 单卡，bf16 两专家，`--offload_model False --t5_cpu`

## 测评

- 状态：已跑通（首跑成功出片）
- 生成质量 / 指令遵循 / 运动连贯性：待评（对照官方 Demo：https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7 ，官方示例 prompt 与本模型一致）
- 速度基线：122.5 s/it（SDPA 回退）；装 ROCm flash-attn 后需复测并记录加速比
- 参数对比（480P vs 720P、步数 40 vs 50、帧数）：待做

## 官方参考与对比

- 官方 Demo 视频（README「Video Demos」）：https://github.com/user-attachments/assets/b63bfa58-d5d7-4de6-a1a2-98970b06d9a7
- 官方效率表（各 GPU 的「总耗时 / 峰值显存」）：https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/assets/comp_effic.png
- 本模型用的正是官方示例 prompt（"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."），可与官方示例直接对比。
- 在线样例/画廊：ModelScope 模型页 https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B ；官方站 https://wan.video ；ComfyUI 教程 https://docs.comfy.org/tutorials/video/wan/wan2_2
- 说明：仓库 `examples/` 只放输入素材（i2v_input.JPG、talk.wav、pose.mp4 等），未提交输出成片；成片见上面 Demo/画廊。

## 下一步

- [x] 首跑出片（本记录）
- [ ] 装 ROCm flash-attention 并撤销 `model.py` 的 SDPA 补丁，复测 s/it 与官方效率表对比
- [ ] 参数对比：480P vs 720P、步数 40 vs 50、帧数 81 vs 121
- [ ] 视频理解线：Qwen3-VL-72B / Qwen2.5-VL-72B
- 结论：显存不是瓶颈（64/192GB），瓶颈是 attention 实现（SDPA 回退 → 慢）。
