#!/usr/bin/env bash
# Wan2.2-T2V-A14B 部署脚本（魔搭 ModelScope AMD MI300X / ROCm）
# 用法: bash deploy.sh
# 说明: 权重下载到临时盘（默认 /root/tmp，在 / 上，约 561G 空闲），部署完可删除。
set -euo pipefail

MODEL_ID="Wan-AI/Wan2.2-T2V-A14B"
REPO_URL="https://github.com/Wan-Video/Wan2.2.git"

TMP_DIR="${TMP_DIR:-/root/tmp}"                 # 临时下载目录（非持久盘，>=140GB）
MODEL_DIR="${MODEL_DIR:-$TMP_DIR/Wan2.2-T2V-A14B}"
CODE_DIR="${CODE_DIR:-$TMP_DIR/Wan2.2}"
OUT_DIR="${OUT_DIR:-/mnt/workspace/outputs}"    # 产出放持久盘

PROMPT="${PROMPT:-Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.}"
SIZE="${SIZE:-1280*720}"

echo "==> 0. 环境自检"
rocm-smi || true
python -c "import torch;print('torch', torch.__version__, 'hip', torch.version.hip, 'cuda_ok', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h "$TMP_DIR"

echo "==> 1. 准备目录"
mkdir -p "$TMP_DIR" "$OUT_DIR"
export MODELSCOPE_CACHE="$TMP_DIR/modelscope"

echo "==> 2. 下载模型（约 126GB）"
pip install -U modelscope
modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"

echo "==> 3. 获取代码 + 依赖（跳过 flash_attn）"
if [ ! -d "$CODE_DIR" ]; then git clone "$REPO_URL" "$CODE_DIR"; fi
cd "$CODE_DIR"
# 跳过 flash_attn（ROCm 不装）与 numpy 钉版本（避免把预装 numpy 2 降级，破坏 scipy）
grep -vE "flash_attn|numpy" requirements.txt > req_amd.txt
pip install -r req_amd.txt
pip install -U "numpy>=2,<3"   # 保险：保持 numpy 2.x
# ROCm 无 flash-attn：改走 attention.py 的 SDPA 回退（无需编译 flash-attn）
sed -i 's/from .attention import flash_attention/from .attention import attention as flash_attention/' wan/modules/model.py
grep -n "attention import" wan/modules/model.py
# 跳过 s2v/animate 的强依赖（decord/librosa），t2v 用不到；decord 无 py3.12 wheel
sed -i -e 's/^from .speech2video import WanS2V/# &/' -e 's/^from .animate import WanAnimate/# &/' wan/__init__.py

echo "==> 4. 生成视频"
python generate.py --task t2v-A14B --size "$SIZE" \
  --ckpt_dir "$MODEL_DIR" \
  --convert_model_dtype --offload_model False --t5_cpu \
  --prompt "$PROMPT" \
  --save_file "$OUT_DIR/wan2.2_t2v_demo.mp4"

echo "==> 完成，输出: $OUT_DIR/wan2.2_t2v_demo.mp4"
echo "==> 清理临时下载: rm -rf '$MODEL_DIR' '$TMP_DIR/modelscope'"
