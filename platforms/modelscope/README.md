# ModelScope

本目录用于记录 ModelScope 托管平台上的模型 API、数据集和专区能力实践。

| 子目录 / 文件 | 功能 |
|---|---|
| `model_inference/` | 调用模型推理 API。每个模型单独建一个 Python 示例，以模型名命名；通用 OpenAI 兼容调用模板单独保留。 |
| `datasets/` | 数据集读取、清洗、后处理，以及后续微调实验。 |
| `capabilities/` | 创空间、智能工坊、实验场、AIGC 等专区的平台能力与边界记录。 |
| `.env.example` | API Token、服务端点和模型 ID 的环境变量模板，不含真实凭据。 |

旧的 `llm/modelscope_api.py` 没有固定模型名，实际模型 ID 通过 `MODELSCOPE_MODEL_ID` 环境变量提供，因此迁移为 `model_inference/openai_compatible_chat.py` 通用示例。确定具体模型和端点后，再增加对应的 `<model_name>.py` 文件，并在文件开头记录 ModelScope 模型 ID、部署端点类型和调用前提。

运行示例前设置 `.env` 中的 `MODELSCOPE_API_TOKEN`、`MODELSCOPE_BASE_URL`、`MODELSCOPE_MODEL_ID`，或在 Shell 中导出相同名称的环境变量。不要提交 `.env`。
