# 模型平台实践

这里记录模型平台上的 API 调用、数据集实践、平台能力探索和平台云端算力部署；具体硬件环境与实测范围在各记录中注明，与 `llm/`、`video/`、`vision/` 中的既有部署路线分开。

| 平台 | 入口 | 内容 |
|---|---|---|
| ModelScope | [platforms/modelscope](modelscope/README.md) | 模型推理 API、数据集、平台专区能力。 |
| Hugging Face | [platforms/huggingface](huggingface/README.md) | 模型推理 API、数据集、平台能力。 |
| 模力方舟 / 沐曦 | [文本生成模型](moark/metax/文本生成模型.md)、[图形生成模型](moark/metax/图形生成模型.md)、[语音识别模型](moark/metax/语音识别模型.md)、[语音合成模型](moark/metax/语音合成模型.md) | 文本、生图、ASR 与 TTS 部署及排障；语音任务用户已确认完成，各项证据边界见正文。 |

平台 API 凭据通过忽略的 `.env` 文件或环境变量提供；不要将 Token、私有端点凭据写入源文件或提交到 Git。模型 API 示例与本地部署基准的测试口径不同，不能直接混入本地推理框架排名。
