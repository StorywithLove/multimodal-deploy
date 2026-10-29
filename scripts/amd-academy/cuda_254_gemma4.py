"""在 NVIDIA CUDA 单卡上复现 Gemma 4 情绪分类 LoRA 实验。

依赖：CUDA 版 PyTorch、modelscope、transformers、accelerate、datasets、
trl、peft、scikit-learn、pandas、tqdm。需使用支持 Gemma 4 的 transformers。
模型、数据与结果默认位于临时目录 /root/tmp；实例结束前请自行备份所需产物。
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

MODEL_ROOT = Path('/root/tmp/model')
OUTPUT_DIR = Path('/root/tmp/result')
DATA_ROOT = MODEL_ROOT / 'datasets'

# 在加载第三方库前设置缓存，并先显示本次使用的目录。
print(f'模型下载目录: {MODEL_ROOT}', flush=True)
print(f'模型输出目录: {OUTPUT_DIR}', flush=True)
os.environ.setdefault('HF_HOME', str(MODEL_ROOT / 'huggingface'))
os.environ.setdefault('HF_DATASETS_CACHE', str(DATA_ROOT / 'hf_cache'))
os.environ.setdefault('MODELSCOPE_CACHE', str(MODEL_ROOT / 'modelscope_cache'))

import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, DatasetDict, load_dataset
from modelscope import snapshot_download
from modelscope.hub.snapshot_download import dataset_snapshot_download
from peft import LoraConfig
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

MODEL_ID = 'google/gemma-4-E4B-it'
DATASET_ID = 'AI-ModelScope/emotion'
LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
LABEL_PATTERN = re.compile(r'\b(sadness|joy|love|anger|fear|surprise)\b', re.IGNORECASE)
SEED = 42
SYSTEM_PROMPT = ('You are an emotion classification assistant.\n'
                 "Read the user's text and answer with exactly one label.\n"
                 'Only choose from: sadness, joy, love, anger, fear, surprise.\n'
                 'Return only the label and nothing else.')


def prepare_environment():
    """创建临时目录，检查 NVIDIA CUDA 和 BF16，并固定随机种子。"""
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
        raise RuntimeError('需要可用的 NVIDIA CUDA 版 PyTorch；当前不是 CUDA 环境。')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('本实验使用 BF16，请使用支持 BF16 的 NVIDIA GPU。')
    random.seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print(f'PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}')
    print('单卡 BF16 LoRA；请确认显存与磁盘空间足够。')


def download_resources():
    """从 ModelScope 下载模型和数据集到临时目录。"""
    model_dir = snapshot_download(MODEL_ID, cache_dir=str(MODEL_ROOT))
    dataset_dir = dataset_snapshot_download(DATASET_ID, cache_dir=str(DATA_ROOT))
    print(f'模型实际路径: {model_dir}\n数据集实际路径: {dataset_dir}')
    return Path(model_dir), Path(dataset_dir)


def load_emotion_dataset(dataset_dir):
    """读取 parquet 并按固定随机种子选择训练、验证和测试样本。"""
    data_files = {}
    for split in ('train', 'validation', 'test'):
        files = sorted(str(p) for p in (dataset_dir / 'data').glob(f'{split}-*.parquet'))
        if not files:
            raise FileNotFoundError(f'未找到 {split} parquet 文件：{dataset_dir / "data"}')
        data_files[split] = files
    raw = load_dataset('parquet', data_files=data_files, cache_dir=str(DATA_ROOT / 'hf_cache'))
    subsets = {}
    for split, limit in (('train', 4000), ('validation', 400), ('test', 400)):
        data = raw[split]
        if not isinstance(data.features['label'], ClassLabel):
            data = data.cast_column('label', ClassLabel(names=LABELS))
        data = data.shuffle(seed=SEED)
        subsets[split] = data.select(range(min(limit, len(data))))
    return DatasetDict(subsets)


def make_messages(text):
    """构造训练和评估共用的分类提示词。"""
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'Classify the emotion of this text:\n\n{text}'}]


def prepare_training_data(dataset):
    """把数据转换为 TRL 对话式 prompt/completion 格式。"""
    def convert(example):
        """将一条文本与标签转换为对话训练样本。"""
        return {'prompt': make_messages(example['text']),
                'completion': [{'role': 'assistant', 'content': LABELS[example['label']]}]}
    return dataset.map(convert, remove_columns=dataset['train'].column_names)


def load_model_and_tokenizer(model_dir):
    """加载对话模板和 BF16 基础模型，并放到第一张 CUDA 卡。"""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # 将结束符号作为填充符号
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        template = model_dir / 'chat_template.jinja'
        if not template.is_file():
            raise FileNotFoundError(f'模型未提供 chat template：{template}')
        # 将结构化的system/user/assistant消息转为模型所需格式
        tokenizer.chat_template = template.read_text(encoding='utf-8')
    tokenizer.apply_chat_template(make_messages('Hello'), tokenize=False, add_generation_prompt=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to('cuda:0')
    # 关闭生成时使用的KV缓存。
    model.config.use_cache = False
    # 将 tokenizer 的 pad/bos/eos token id 复制到模型配置和生成配置中。
    for key in ('pad_token_id', 'bos_token_id', 'eos_token_id'):
        value = getattr(tokenizer, key)
        setattr(model.config, key, value)
        setattr(model.generation_config, key, value)
    return model, tokenizer


def generate_label(model, tokenizer, text):
    """贪心解码/最大概率生成分类结果，同时保留完整生成文本供格式检查。"""
    inputs = tokenizer.apply_chat_template(make_messages(text), tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt')
    inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    # 输出token ID转为文字；inputs['input_ids'].shape[-1]为原始内容和生成内容切分点
    decoded = tokenizer.decode(output[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    match = LABEL_PATTERN.search(decoded)
    return (match.group(1).lower() if match else 'INVALID'), decoded


def evaluate(model, tokenizer, test_data, stage):
    """在固定测试集上评估并立即保存指标、原始预测和混淆矩阵。"""
    model.eval()
    rows = []
    for example in tqdm(test_data, desc=f'评估 {stage}'):
        label, decoded = generate_label(model, tokenizer, example['text'])
        truth = LABELS[example['label']]
        rows.append({'text': example['text'], 'true_label': truth, 'pred_label': label,
                     'decoded_text': decoded, 'strict_format': decoded.strip().lower() in LABELS,
                     'correct': truth == label})
    predictions = pd.DataFrame(rows)
    y_true, y_pred = predictions['true_label'], predictions['pred_label']
    metrics = {'accuracy': accuracy_score(y_true, y_pred),
               'macro_f1': f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0),
               'weighted_f1': f1_score(y_true, y_pred, labels=LABELS, average='weighted', zero_division=0),
               'invalid_predictions': int((y_pred == 'INVALID').sum()),
               'strict_format_rate': float(predictions['strict_format'].mean()),
               'evaluated_examples': len(predictions)}
    predictions.to_csv(OUTPUT_DIR / f'{stage}_predictions.csv', index=False)
    # 生成分类报告和混淆矩阵，并保存为 CSV。
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / f'{stage}_classification_report.csv')
    names = LABELS + ['INVALID']
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=names), index=names, columns=names).to_csv(
        OUTPUT_DIR / f'{stage}_confusion_matrix.csv')
    # 保存指标为 JSON，确保 UTF-8 编码和中文可读。
    save_json(OUTPUT_DIR / f'{stage}_metrics.json', metrics)
    print(f'{stage} 指标: {metrics}')
    return metrics, predictions


def save_json(path, data):
    """将指标保存为 UTF-8 JSON。"""
    with path.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_trainer(model, tokenizer, training_data):
    """按原 Notebook 配置，创建单卡 BF16 LoRA SFTTrainer。"""
    # r为低秩，lora_alpha为缩放参数，task_type为因果语言模型，target_modules为LoRA适配器位置
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                      task_type='CAUSAL_LM', target_modules='all-linear')
    # 数据量与更新频率：batch为4，梯度累积4步(有效bach16)，验证batch为1，训练集完整遍历一边
    # 优化和学习率：权重衰减，学习率线性变化，前50个更新逐步升高，AdamW优化器
    # 日志和保存：每5步打印日志，每25步验证和保存，最多保留2个检查点
    # 显存与精度：启用梯度检查点/反向传播时重新计算部分激活值，bf16，不拼接短样本packing
    # 样本：complete_only拟合completion不拟合prompt。
    args = SFTConfig(output_dir=str(OUTPUT_DIR), per_device_train_batch_size=4,
        per_device_eval_batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-4,
        weight_decay=0.01, lr_scheduler_type='linear', warmup_steps=50, num_train_epochs=1,
        logging_steps=5, eval_strategy='steps', eval_steps=25, save_strategy='steps',
        save_steps=25, save_total_limit=2, gradient_checkpointing=True, bf16=True, fp16=False,
        tf32=False, max_length=256, packing=False, completion_only_loss=True,
        remove_unused_columns=False, dataloader_num_workers=2, optim='adamw_torch',
        report_to='none', seed=SEED, data_seed=SEED)
    trainer = SFTTrainer(model=model, train_dataset=training_data['train'],
        eval_dataset=training_data['validation'], peft_config=lora, args=args,
        processing_class=tokenizer)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    if not trainable:
        raise RuntimeError('未发现可训练的 LoRA 参数。')
    print(f'可训练参数: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)')
    return trainer


def train_and_save(trainer, tokenizer, checkpoint=None):
    """执行训练并在后评估前保存适配器、tokenizer 和训练指标。"""
    result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    save_json(OUTPUT_DIR / 'train_metrics.json', result.metrics)
    trainer.model.eval()
    trainer.model.config.use_cache = False  # 前后评估使用相同缓存设置
    return trainer.model


def save_comparison(pre_metrics, post_metrics, pre, post):
    """保存前后指标与逐样本对照，并验证评估样本顺序。"""
    if not pre[['text', 'true_label']].equals(post[['text', 'true_label']]):
        raise ValueError('前后评估样本不一致，停止生成对比结果。')
    pd.DataFrame([{'stage': 'pre_finetuning', **pre_metrics},
                  {'stage': 'post_finetuning', **post_metrics}]).to_csv(
                      OUTPUT_DIR / 'gemma4_emotion_before_after_metrics.csv', index=False)
    combined = pre.add_prefix('pre_').join(post.add_prefix('post_'))
    combined.to_csv(OUTPUT_DIR / 'gemma4_emotion_prediction_examples.csv', index=False)
    combined[combined['pre_pred_label'] != combined['post_pred_label']].to_csv(
        OUTPUT_DIR / 'gemma4_emotion_changed_predictions.csv', index=False)


def resolve_checkpoint(resume):
    """显式续训时选择最新检查点，检查必需文件；新训练避免覆盖旧检查点。"""
    checkpoints = sorted(
        (p for p in OUTPUT_DIR.glob('checkpoint-*')
         if p.is_dir() and p.name.removeprefix('checkpoint-').isdigit()),
        key=lambda p: int(p.name.removeprefix('checkpoint-')))
    if not resume:
        if checkpoints or (OUTPUT_DIR / 'adapter_config.json').exists():
            raise RuntimeError('输出目录已有训练产物；续训请加 --resume，新实验请先备份并使用干净的结果目录。')
        return None
    if not checkpoints:
        raise FileNotFoundError(f'{OUTPUT_DIR} 中没有 checkpoint，无法续训。')
    checkpoint = checkpoints[-1]
    required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_config.json']
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not any((checkpoint / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin')):
        missing.append('adapter_model.safetensors 或 adapter_model.bin')
    if missing:
        raise RuntimeError(f'最新检查点 {checkpoint} 不完整，缺少 {missing}；请检查或恢复备份。')
    print(f'续训检查点: {checkpoint}', flush=True)
    return checkpoint


def load_baseline(test_data):
    """读取既有基线结果，并确认与当前固定测试样本一致。"""
    metrics_path = OUTPUT_DIR / 'pre_finetuning_metrics.json'
    predictions_path = OUTPUT_DIR / 'pre_finetuning_predictions.csv'
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError('续训需要原实验的基线指标与预测文件，请先恢复这两个文件。')
    with metrics_path.open(encoding='utf-8') as file:
        metrics = json.load(file)
    predictions = pd.read_csv(predictions_path, keep_default_na=False)
    expected = [(item['text'], LABELS[item['label']]) for item in test_data]
    if list(zip(predictions['text'], predictions['true_label'])) != expected:
        raise ValueError('当前测试样本与保存的基线不一致，停止续训。')
    print(f'已读取原实验的 {len(predictions)} 条基线预测。')
    return metrics, predictions


def main():
    """执行新实验，或通过 --resume 从最近的完整检查点继续。"""
    parser = argparse.ArgumentParser(description='Gemma 4 CUDA LoRA 训练')
    parser.add_argument('--resume', action='store_true', help='从结果目录中最新检查点恢复训练')
    args = parser.parse_args()
    prepare_environment()
    checkpoint = resolve_checkpoint(args.resume)
    model_dir, dataset_dir = download_resources()
    dataset = load_emotion_dataset(dataset_dir)
    model, tokenizer = load_model_and_tokenizer(model_dir)
    if checkpoint:
        pre_metrics, pre = load_baseline(dataset['test'])
    else:
        pre_metrics, pre = evaluate(model, tokenizer, dataset['test'], 'pre_finetuning')
    trainer = build_trainer(model, tokenizer, prepare_training_data(dataset))
    trained_model = train_and_save(trainer, tokenizer, checkpoint)
    post_metrics, post = evaluate(trained_model, tokenizer, dataset['test'], 'post_finetuning')
    save_comparison(pre_metrics, post_metrics, pre, post)
    print(f'实验完成；产物位于 {OUTPUT_DIR}，请及时备份需要保留的文件。')


if __name__ == '__main__':
    main()
