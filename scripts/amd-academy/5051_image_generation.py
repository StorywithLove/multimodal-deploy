"""在 NVIDIA CUDA 单卡上复现 AMD 5051 的 SDXL 文生图实验。

依赖：CUDA 版 PyTorch、modelscope、diffusers、transformers、accelerate、
safetensors、Pillow。模型与图片默认位于临时目录 /root/tmp；请及时备份产物。
示例：python 5051_image_generation.py --aspect-ratio 16:9 --seed 42
"""

import argparse
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

MODEL_ROOT = Path('/root/tmp/model')
OUTPUT_DIR = Path('/root/tmp/result')
MODEL_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
PROMPT = (
    'Two anthropomorphic cats wearing cozy boxing robes and bright gloves,\n'
    'standing in a boxing ring under a hard spotlight.\n'
    'Sweat mist drifts in the air, the crowd is blurred in the background with a bokeh effect.\n'
    'Cinematic style, dramatic lighting, detailed close-up of quick jabs and agile footwork.'
)
ASPECT_RATIOS = {
    '1:1': (1328, 1328), '16:9': (1664, 928), '9:16': (928, 1664),
    '4:3': (1472, 1104), '3:4': (1140, 1472),
    '3:2': (1584, 1056), '2:3': (1056, 1584),
}

# 在加载第三方库之前设置缓存。
os.environ.setdefault('HF_HOME', str(MODEL_ROOT / 'huggingface'))
os.environ.setdefault('MODELSCOPE_CACHE', str(MODEL_ROOT / 'modelscope_cache'))

import torch
from diffusers import DiffusionPipeline
from modelscope import snapshot_download
from PIL import Image


def prepare_environment():
    """创建目录，检查 NVIDIA CUDA 与 notebook 使用的 BF16 精度。"""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
        raise RuntimeError('需要可用的 NVIDIA CUDA 版 PyTorch；当前不是 CUDA 环境。')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('本实验使用 BF16，请使用支持 BF16 的 NVIDIA GPU。')
    print(f'模型下载目录: {MODEL_ROOT}\n图片输出目录: {OUTPUT_DIR}', flush=True)
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}', flush=True)


def parse_args():
    """读取生成参数，默认值与 notebook 一致。"""
    parser = argparse.ArgumentParser(description='AMD 5051 SDXL CUDA 图像生成')
    parser.add_argument('--prompt', default=PROMPT, help='图像描述')
    parser.add_argument('--aspect-ratio', choices=ASPECT_RATIOS, default='16:9', help='预设宽高比')
    parser.add_argument('--width', type=int, help='自定义宽度，须与 --height 同时使用且为 8 的倍数')
    parser.add_argument('--height', type=int, help='自定义高度，须与 --width 同时使用且为 8 的倍数')
    parser.add_argument('--steps', type=int, default=30, help='去噪步数，默认 30')
    parser.add_argument('--guidance-scale', type=float, default=7.0, help='提示词引导强度，默认 7.0')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，默认 42')
    parser.add_argument('--output', type=Path, help='输出图片路径，默认在结果目录生成唯一文件名')
    args = parser.parse_args()
    if (args.width is None) != (args.height is None):
        parser.error('--width 和 --height 必须同时指定')
    if args.width is not None and (args.width <= 0 or args.height <= 0 or args.width % 8 or args.height % 8):
        parser.error('宽度和高度必须为正数且为 8 的倍数')
    if args.steps <= 0:
        parser.error('--steps 必须为正数')
    if args.width is None:
        args.width, args.height = ASPECT_RATIOS[args.aspect_ratio]
        # notebook 的 3:4 宽度 1140 不能被 8 整除，调整到最近的合法值。
        if args.width % 8:
            args.width -= args.width % 8
    return args


def generate_animation(pipe, args, frames_dir):
    """生成图片并保存每步 latent 的近似预览，合成 GIF。"""
    frame_paths = []

    def save_step_preview(pipeline, step, timestep, callback_kwargs):
        # 一般是[4, H/8, W/8]，但也可能是[1, 4, H/8, W/8]，取第一个 batch。
        latent = callback_kwargs['latents'][0].float()
        weights = torch.tensor([[60, -60, 25, -70], [60, -5, 15, -50],
                                [60, 10, -5, -35]], device=latent.device,
                               dtype=torch.float32).T   # 4*3
        biases = torch.tensor([150, 140, 130], device=latent.device, dtype=torch.float32)
        # lagent:lxy, weights:lr -> out:rxy
        rgb = torch.einsum('lxy,lr->rxy', latent, weights) + biases[:, None, None]
        array = rgb.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)
        path = frames_dir / f'step_{step + 1:03d}.png'
        Image.fromarray(array).save(path)
        frame_paths.append(path)
        return callback_kwargs

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        # 每次step去噪后调用一次save_step_prview, 并把['latents']传入
        image = pipe(prompt=args.prompt, width=args.width, height=args.height,
                     num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                     generator=torch.Generator(device='cuda:0').manual_seed(args.seed),
                     callback_on_step_end=save_step_preview,
                     callback_on_step_end_tensor_inputs=['latents']).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
    reserved = torch.cuda.max_memory_reserved() / 1024 ** 3

    if not frame_paths:
        raise RuntimeError('没有收到逐步回调，未生成预览帧。')
    frames = []
    for path in frame_paths:
        with Image.open(path) as frame:
            frames.append(frame.convert('RGB'))
    gif_path = frames_dir.parent / 'denoising.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=False)
    return image, gif_path, len(frame_paths), elapsed, allocated, reserved


def main():
    """下载预训练模型，生成图片和逐步去噪动图。"""
    args = parse_args()
    prepare_environment()
    model_dir = snapshot_download(MODEL_ID, cache_dir=str(MODEL_ROOT))
    print(f'模型实际路径: {model_dir}', flush=True)
    pipe = DiffusionPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16).to('cuda:0')
    # 使用 cuDNN 默认设置；原 notebook 禁用了 cuDNN。
    print(f'开始生成: {args.width}x{args.height}, steps={args.steps}, '
          f'guidance_scale={args.guidance_scale}, seed={args.seed}', flush=True)
    output = args.output or OUTPUT_DIR / f'sdxl_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'输出文件已存在，请指定其他路径：{output}')
    run_dir = Path(tempfile.mkdtemp(prefix=f'{output.stem}_', dir=str(output.parent)))
    frames_dir = run_dir / 'frames'
    frames_dir.mkdir()
    image, gif_path, frame_count, elapsed, allocated, reserved = generate_animation(pipe, args, frames_dir)
    image.save(output)
    print(f'图片已保存: {output}\n动图: {gif_path}\n逐步预览目录: {frames_dir}\n'
          f'预览帧数: {frame_count}\n生成及保存预览耗时: {elapsed:.1f} 秒\n'
          f'PyTorch 峰值分配显存: {allocated:.2f} GiB；峰值预留显存: {reserved:.2f} GiB')
    print('模型和默认输出位于临时目录，请及时备份需要保留的文件。')


if __name__ == '__main__':
    main()
