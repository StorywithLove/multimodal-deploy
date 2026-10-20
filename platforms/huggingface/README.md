# Hugging Face

本目录用于记录 Hugging Face 托管平台上的模型 API、数据集和平台能力实践，目录职责与 `platforms/modelscope/` 对齐。

| 子目录 | 功能 |
|---|---|
| `model_inference/` | 托管推理 API 调用示例；每个具体模型单独一个 Python 文件，并注明模型 ID 和端点类型。 |
| `datasets/` | 数据集读取、清洗、后处理和微调数据准备。 |
| `capabilities/` | Spaces、Inference Providers、模型卡等平台功能与能力边界记录。 |

凭据通过环境变量或本地 `.env` 提供，不提交令牌。在线 API 示例与本地推理框架测试分别记录。
