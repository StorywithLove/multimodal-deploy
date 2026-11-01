# 5051 图像生成：A10 运行记录

- 时间：2026-10-30，ModelScope NVIDIA A10。
- 模型：stabilityai/stable-diffusion-xl-base-1.0；BF16。
- 配置：1664 × 928，30 步，guidance scale 7.0，seed 42。
- 生成耗时：14.7 秒；PyTorch 生成阶段峰值分配显存：15.23 GiB（非整卡显存峰值）。
- 模型下载约 6 分 13 秒，管线加载约 37 秒；按日志文件时间估算首次运行总耗时约 7 分 32 秒。
- Python 和日志写入退出码均为 0。
- 图片：`sdxl_20261030_111008_104497.png`；原始日志：`5051_image_generation_20261030_110240_LTkPyF.log`。
- 图片和日志已通过 SHA-256 与远端原文件核对一致。
