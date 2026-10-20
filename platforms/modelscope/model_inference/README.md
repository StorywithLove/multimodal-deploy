# ModelScope 模型推理调用

每个模型示例单独使用一个 Python 文件，建议文件名采用模型名的小写 snake_case，例如 `qwen3_8_flash_next.py`；在文件顶部注明平台模型 ID 和所需的 API 端点类型。

`openai_compatible_chat.py` 是原 `llm/modelscope_api.py` 的通用调用模板。它从环境变量读取模型 ID 和端点，不代表某一个固定模型，也不构成基准测试。
