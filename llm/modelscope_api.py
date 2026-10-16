import os

from openai import OpenAI


api_key = os.environ.get('MODELSCOPE_API_TOKEN')
base_url = os.environ.get('MODELSCOPE_BASE_URL')
model_id = os.environ.get('MODELSCOPE_MODEL_ID')
if not api_key:
    raise RuntimeError(
        'Set MODELSCOPE_API_TOKEN before running this example; never hard-code tokens in source files.'
    )
if not base_url:
    raise RuntimeError(
        'Set MODELSCOPE_BASE_URL to the ModelScope deployment endpoint before running this example.'
    )
if not model_id:
    raise RuntimeError(
        'Set MODELSCOPE_MODEL_ID before running this example.'
    )

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': '你好'
        }
    ],
    stream=True
)

for chunk in response:
	if chunk.choices:
		print(chunk.choices[0].delta.content, end='', flush=True)

# Example: python llm/modelscope_api.py
