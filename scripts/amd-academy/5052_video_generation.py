"""复现 AMD 5052：在 NVIDIA CUDA 单卡运行 Wan 2.2 5B 的 ComfyUI 工作流。

依赖：Linux、git、CUDA 版 PyTorch 和 pip。首次运行安装 ComfyUI 依赖并下载约 18 GB 模型。
默认使用 /root/tmp/ComfyUI-5052；临时实例回收前请备份 output 中的视频。
在 tmux 中运行配套 shell 脚本；通过 SSH 端口转发访问本机 127.0.0.1:8000。
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path('/root/tmp/ComfyUI-5052')
COMFY_COMMIT = '5b80addafd24bda5b2f9f7a35e32dbd40823c3fd'
WORKFLOW_COMMIT = '3e89967349dd95fa68ce3b36966f41d642af8ea7'
MODEL_FILES = (
    ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged', 'diffusion_models',
     'wan2.2_ti2v_5B_fp16.safetensors', '456f901338bd9eadbded3828b819109a9b68e8a525ca5cf8d0049a69fcfeca1e'),
    ('Comfy-Org/Wan_2.2_ComfyUI_Repackaged', 'vae',
     'wan2.2_vae.safetensors', 'e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156'),
    ('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'text_encoders',
     'umt5_xxl_fp8_e4m3fn_scaled.safetensors', 'c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68'),
)


def prepare_environment():
    """检查已安装的 NVIDIA CUDA 版 PyTorch，不替换现有 GPU 环境。"""
    import torch
    if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
        raise RuntimeError('需要可用的 NVIDIA CUDA 版 PyTorch。')
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'ComfyUI 与视频目录: {ROOT}；实例回收前请备份 output。', flush=True)


def prepare_comfyui(skip_install):
    """固定原 notebook 的 ComfyUI 版本；已有目录不强制覆盖。"""
    if not ROOT.exists():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', 'https://github.com/Comfy-Org/ComfyUI.git', str(ROOT)], check=True)
        subprocess.run(['git', '-C', str(ROOT), 'checkout', '--detach', COMFY_COMMIT], check=True)
    if not (ROOT / 'main.py').is_file():
        raise RuntimeError(f'{ROOT} 不含 ComfyUI main.py，请检查目录。')
    version = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if version != COMFY_COMMIT:
        raise RuntimeError(f'已有 ComfyUI 版本 {version} 与 notebook 的 {COMFY_COMMIT} 不一致，请检查目录。')
    if not skip_install:
        # 与 notebook 一样，不通过 requirements 替换 CUDA 版 torch 及相关包。
        requirements = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
        import re
        requirements = [line for line in requirements if not re.match(r'^\s*(torch|torchvision|torchaudio)(?:\s|[<>=!~;\[]|$)', line, re.I)]
        with tempfile.TemporaryDirectory() as temp_dir:
            filtered = Path(temp_dir) / 'requirements.txt'
            filtered.write_text('\n'.join(requirements) + '\nhuggingface_hub\n', encoding='utf-8')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(filtered)], check=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_models():
    """只下载 notebook 指定的三个文件，检查完整性并复用已有文件。"""
    from huggingface_hub import hf_hub_download
    for repo, category, filename, expected_hash in MODEL_FILES:
        target = ROOT / 'models' / category / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            if sha256_file(target) != expected_hash:
                raise RuntimeError(f'已有文件校验失败，请检查并手动处理：{target}')
            print(f'复用已校验模型: {target}', flush=True)
            continue
        print(f'下载模型: {repo} / {filename}', flush=True)
        downloaded = Path(hf_hub_download(repo_id=repo, filename=f'split_files/{category}/{filename}',
                                          local_dir=str(ROOT / 'model-downloads')))
        if sha256_file(downloaded) != expected_hash:
            raise RuntimeError(f'下载文件校验失败：{downloaded}')
        os.replace(downloaded, target)
        print(f'模型已就绪: {target}', flush=True)


def prepare_workflow():
    """安装 notebook 指定版本的模板，保留用户已有工作流。"""
    target = ROOT / 'user' / 'default' / 'workflows' / 'video_wan2_2_5B_ti2v.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = f'https://raw.githubusercontent.com/Comfy-Org/workflow_templates/{WORKFLOW_COMMIT}/templates/video_wan2_2_5B_ti2v.json'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        import json
        json.loads(data)
        target.write_bytes(data)
    print(f'工作流: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='AMD 5052 Wan 2.2 ComfyUI NVIDIA CUDA 工作流')
    parser.add_argument('--port', type=int, default=8000, help='本机监听端口，默认 8000')
    parser.add_argument('--skip-install', action='store_true', help='跳过依赖安装')
    parser.add_argument('--lowvram', action='store_true', help='启用 ComfyUI 低显存模式')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在 1 到 65535 之间')
    prepare_environment()
    prepare_comfyui(args.skip_install)
    prepare_models()
    prepare_workflow()
    command = [sys.executable, '-u', 'main.py', '--listen', '127.0.0.1', '--port', str(args.port)]
    if args.lowvram:
        command.append('--lowvram')
    print(f'启动 ComfyUI: http://127.0.0.1:{args.port}/；远程访问请通过 SSH 转发该端口。', flush=True)
    print('打开界面后加载工作流，设置提示词并提交任务；视频默认保存在 output。', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
