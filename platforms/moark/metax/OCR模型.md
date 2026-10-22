# 模力方舟 / 沐曦：OCR 模型部署记录

记录日期：2026-10-22。依据本轮会话历史与用户补充说明整理。**沐曦 L1 认证已于当日通过并拿到（用户确认）**；FireRed-OCR 单项没有任何归档的运行输出，不能由整体认证反推单项日志或识别正文。以下为交付代码和复现要点，未在本地重新执行云端推理。

## 范围与产物

| 路线 | 结果 |
|---|---|
| PaddleOCR-VL-1.5 | 用户明确表示原始示例已经跑通；未提供完整识别正文 |
| FireRed-OCR 单次识别 | 已交付改写代码，未归档运行结果 |
| FireRed-OCR FastAPI | 已交付服务与 curl 方案，未归档启动日志或响应 |
| 沐曦 L1 整体认证 | 用户确认 2026-10-22 已通过并取得认证；凭证文件未进入仓库 |

任务指定：

- 模型：`/mnt/moark-models/FireRed-OCR`。
- 输入：`/mnt/moark-models/L1_exam/ocr_test_image.jpg`。
- 正文：`/data/exam/ocr_output.txt`。
- 服务端口：8188，`POST /v1/vision/ocr`。
- 请求：multipart/form-data，上传字段 `file`，模型字段 `model=FireRed-OCR`。
- 返回：`{"text":"识别结果"}`，保留模型生成的多行文本 / Markdown。

本页对应考试 OCR 主题。用户原话“OCI模型”在此按明确交接说明更正为 OCR；本项目 OCI 指 Oracle Cloud，不能用作文字识别缩写。相关四篇：[文本生成](文本生成模型.md)、[图像生成](图形生成模型.md)、[语音识别](语音识别模型.md)、[语音合成](语音合成模型.md)；第六篇为 [向量化与向量数据库](部署向量化模型与向量数据库的使用.md)。

## 环境与依赖

| 项目 | 记录与边界 |
|---|---|
| 硬件 | 模力方舟沐曦环境；OCR 阶段没有独立完整硬件快照 |
| 框架 | Transformers、PyTorch、Pillow、Accelerate；服务使用 FastAPI、Uvicorn、python-multipart |
| Transformers | 会话讨论过安装 5.0.0；没有最终版本输出，不能标成已验证版本 |
| 模型类型 | FireRed-OCR 的 Qwen3-VL 路线，使用 AutoModelForImageTextToText / AutoProcessor |
| 证据源 | 本地 2026-10-22 会话中 OCR 代码交付；没有原始运行输出 |

### 坑一：不能只替换 PaddleOCR 模型路径

PaddleOCR-VL 示例包含 `OCR:`、`Table Recognition:`、spotting 放大与特定 `images_kwargs`。FireRed-OCR 应采用其模型对应的多模态消息、处理器与文档解析指令。下列 INSTRUCTION 是会话整理版提示词，**不是逐字核验过的官方训练 prompt**；若严格复现官方输出，应对照官方 `conv_for_infer.py`。

### 坑二：解码时不要无条件丢弃最后一个 token

原样例使用 `outputs[0][prompt_len:-1]`；如果生成因长度上限截断，最后一个 token 可能是正文。改为仅去掉输入部分，再用 `skip_special_tokens=True` 清理。整页识别 4096 token 可能不足，需要观察实际输出再调整，不能凭代码长度判定完整识别。

### 坑三：架构与包版本需要先检查

```bash
python -c "import transformers; from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print(transformers.__version__); print('qwen3_vl' in CONFIG_MAPPING)"
```

本轮没有归档 FireRed 特定异常，不虚构报错。若架构不在映射中，先匹配其官方依赖或独立环境；不要直接升级后继续假定 ASR / TTS 的固定版本仍兼容。参数 `dtype` 与 `torch_dtype` 的接受情况需以实际版本为准。

## 任务一：单次识别与保存

保存到云端 `/data/exam/ocr_single.py`；Notebook 可直接运行代码。下面保留会话交付逻辑，未宣称重新实测：

```python
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH  = "/mnt/moark-models/FireRed-OCR"
IMAGE_PATH  = "/mnt/moark-models/L1_exam/ocr_test_image.jpg"
OUTPUT_PATH = Path("/data/exam/ocr_output.txt")
DEVICE = "cuda"

MAX_PIXELS = 1280 * 28 * 28
MIN_PIXELS = 4 * 28 * 28

# FireRed-OCR 的 prompt 在训练时固定为「文档图 → Markdown」的长指令，
# 而不是 PaddleOCR-VL 的 "OCR:" / "Table Recognition:" 这类短指令。
INSTRUCTION = (
    "You are an AI assistant specialized in converting document images to Markdown format. "
    "Recognize all text in the image verbatim, preserving the reading order and layout. "
    "Use Markdown headings and lists where appropriate, render formulas as LaTeX, "
    "and render tables as HTML <table>. "
    "Output only the recognized content, without any extra explanation."
)

assert Path(MODEL_PATH).is_dir(), MODEL_PATH
assert Path(IMAGE_PATH).is_file(), IMAGE_PATH

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,          # transformers 4.x 写作 torch_dtype=
    device_map="auto",
).eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 像素预算挂在 image_processor 上；不是 PaddleOCR-VL 那种 apply_chat_template 的 images_kwargs["size"]
img_proc = processor.image_processor
if hasattr(img_proc, "max_pixels"):
    img_proc.max_pixels = MAX_PIXELS
if hasattr(img_proc, "min_pixels"):
    img_proc.min_pixels = MIN_PIXELS

image = Image.open(IMAGE_PATH).convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": INSTRUCTION},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)

prompt_len = inputs["input_ids"].shape[-1]
result = processor.batch_decode(
    outputs[:, prompt_len:],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0].strip()

if not result:
    raise RuntimeError("识别结果为空")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(result + "\n", encoding="utf-8")
print(result)
print(f"saved -> {OUTPUT_PATH} ({len(result)} chars)")
```

### 启动与验收

```bash
mkdir -p /data/exam
cd /data/exam
python ocr_single.py
test -s /data/exam/ocr_output.txt && head -c 500 /data/exam/ocr_output.txt
```

人工核对图片与识别内容；文件非空不是准确率证明。当前未收集这些命令的实际输出。

## 任务二：FastAPI 图片上传

保存到云端 `/data/exam/ocr_api.py`。文件扩展名和 PIL 解码均检查；EXIF 方向纠正用于避免上传照片方向错误。锁用于串行使用共享模型，不代表已经测过并发性能。

```python
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_NAME = "FireRed-OCR"
MODEL_PATH = f"/mnt/moark-models/{MODEL_NAME}"
MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "4096"))
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

OCR_INSTRUCTION = (
    "You are an AI assistant specialized in converting document images to Markdown format. "
    "Recognize all text in the image verbatim, preserving the reading order and layout. "
    "Use Markdown headings and lists where appropriate, render formulas as LaTeX, "
    "and render tables as HTML <table>. "
    "Output only the recognized content, without any extra explanation."
)

lock = threading.Lock()


def build_messages(image):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": OCR_INSTRUCTION},
        ],
    }]


def run_ocr(app, image):
    processor, model = app.state.processor, app.state.model
    inputs = processor.apply_chat_template(
        build_messages(image),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.batch_decode(
        outputs[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@asynccontextmanager
async def lifespan(app):
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    img_proc = processor.image_processor
    if hasattr(img_proc, "max_pixels"):
        img_proc.max_pixels = 1280 * 28 * 28
    if hasattr(img_proc, "min_pixels"):
        img_proc.min_pixels = 4 * 28 * 28

    app.state.processor = processor
    app.state.model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,   # transformers 5.x 改为 dtype=torch.bfloat16
        device_map="auto",
    ).eval()
    try:
        yield
    finally:
        del app.state.model
        torch.cuda.empty_cache()


app = FastAPI(title="FireRed-OCR Service", lifespan=lifespan)


class OcrResponse(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/vision/ocr", response_model=OcrResponse)
def ocr(file: UploadFile = File(...), model: str = Form(MODEL_NAME)):
    try:
        if model != MODEL_NAME:
            raise HTTPException(400, f"仅支持 {MODEL_NAME}")

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(400, "不支持的图片扩展名")

        raw = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not raw:
            raise HTTPException(400, "图片为空")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "图片超过 20 MB")

        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(400, f"图片无法解析: {exc}") from exc

        with lock:
            text = run_ocr(app, image)
        if not text:
            raise HTTPException(422, "识别结果为空")
        return OcrResponse(text=text)
    finally:
        file.file.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8188, workers=1)
```

### 启动与验收

先释放 Notebook 模型，停止旧 8188 服务，再启动一个 worker。reload 可能引起重复导入或重载，实验中不启用。

```bash
cd /data/exam
python -m uvicorn ocr_api:app --host 0.0.0.0 --port 8188 --workers 1
```

另一个服务器终端：

```bash
curl --fail-with-body http://127.0.0.1:8188/health
curl --fail-with-body http://127.0.0.1:8188/v1/vision/ocr \
  -F "file=@/mnt/moark-models/L1_exam/ocr_test_image.jpg" \
  -F "model=FireRed-OCR" \
  -o /data/exam/ocr_api_response.json
cat /data/exam/ocr_api_response.json
```

该 API 不自动覆盖任务一的 `ocr_output.txt`。需要分别保留文件与接口响应。若必须扩大输出上限，可在启动前设置 `OCR_MAX_NEW_TOKENS=8192`；这只是参数方案，不是本轮实测。

## 参考

- [FireRed-OCR 官方模型卡](https://huggingface.co/FireRedTeam/FireRed-OCR)
- [FireRed-OCR 官方仓库](https://github.com/FireRedTeam/FireRed-OCR)
- [模力方舟 OCR 部署教程](https://ai.gitee.com/docs/compute/deploy/ocr_model)
- [FastAPI 文件与表单](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/)
