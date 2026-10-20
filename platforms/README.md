# 模型平台实践

这里记录托管模型平台上的 API 调用、数据集实践和平台能力探索，与 `llm/`、`video/`、`vision/` 中的本地部署路线分开。

| 平台 | 入口 | 内容 |
|---|---|---|
| ModelScope | [platforms/modelscope](modelscope/README.md) | 模型推理 API、数据集、平台专区能力。 |
| Hugging Face | [platforms/huggingface](huggingface/README.md) | 模型推理 API、数据集、平台能力。 |

平台 API 凭据通过忽略的 `.env` 文件或环境变量提供；不要将 Token、私有端点凭据写入源文件或提交到 Git。模型 API 示例与本地部署基准的测试口径不同，不能直接混入本地推理框架排名。
