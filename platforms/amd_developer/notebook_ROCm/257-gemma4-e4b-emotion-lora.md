# Gemma 4 E4B-it 情绪分类单卡 LoRA 微调（ModelScope）

- 来源：[Radeon Cloud 模板 257](https://developer.amd.com.cn/radeon/templates/257/preview)
- 分析日期：2026-10-26
- 模型：`google/gemma-4-E4B-it`；E4B-it（4 Billion Effective Parameters, Instruction-Tuned），即有效参数规模为 40 亿的指令微调版本。
- 微调方法：LoRA（Low-Rank Adaptation，低秩适配）。
- 数据：`AI-ModelScope/emotion`（`dair-ai/emotion` 镜像）
- 状态：已完成1 epoch、250步 LoRA 训练及同一批400条测试样本的前后评估；结果文件已同步到本地。
- 实际平台：ModelScope DSW，申请规格 AMD MI300X 192GB；终端查询设备显存约 191.7 GiB。

## 一、实验目标与方法

将英文文本分类为 sadness、joy、love、anger、fear、surprise 六类。数据转换成 system/user prompt 与 assistant 标签 completion，使用模型自带官方 chat template，以 SFTTrainer 训练 LoRA，比较训练前后 accuracy、macro-F1、非法预测数和混淆矩阵，保存适配器、tokenizer、训练指标与预测 CSV。

模型使用 ModelScope `snapshot_download` 下载，再从本地路径加载；数据使用 `dataset_snapshot_download` 下载仓库，以 `datasets.load_dataset("parquet", ...)` 读取，并恢复 ClassLabel 标签类型，绕开 MsDataset 的版本桥接问题。不使用 Hugging Face Hub 登录或 HF_TOKEN，但仍使用 transformers、datasets、TRL、PEFT 等开源库。

这是单卡 LoRA，并非全参数微调，也不是默认 QLoRA。原版为 ROCm 环境设计，不启用 bitsandbytes 4bit。ROCm 下通过 `torch.cuda` 访问 GPU 是正常的统一接口。

## 二、部署

### 2.1 环境配置与验证（2026-10-26）

本次使用 ModelScope DSW 镜像，与课程使用的 `/opt/venv` 预置环境不同。镜像已有可用的 ROCm PyTorch，在此基础上创建 `/mnt/workspace/.venv` 安装训练依赖。

| 检查项 | 实际输出 |
| --- | --- |
| Python | 创建虚拟环境时为 CPython 3.12.13，基础解释器 `/usr/bin/python` |
| 系统 torch 路径 | `/usr/local/lib/python3.12/dist-packages/torch/__init__.py` |
| PyTorch | `2.12.0+git6bbd260`（平台定制构建） |
| ROCm / HIP | `7.2.53211` |
| GPU | 终端及 Notebook 可用性均为 True；Notebook 检测到1张卡 |
| 设备显存 | 终端查询约191.7 GiB |
| BF16 | 终端检查为 True |
| 设备名称 | 查询返回空字符串；MI300X 型号来自申请规格，未由该输出确认 |
| Notebook | 用户确认环境验证成功；导入与配置单元已打印 torch 版本、GPU 状态 |

这些结果证明依赖导入和设备访问通过，不代表模型加载或训练已完成。ROCm 下继续使用 `torch.cuda`，接口名称不代表 NVIDIA CUDA 后端。

#### 配置步骤

以下在云实例 Linux 终端执行。先检查镜像已有 torch，再建立继承系统包的虚拟环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
cd /mnt/workspace
uv venv --system-site-packages .venv
source .venv/bin/activate
```

继承系统包不等于禁止另装 torch。安装时使用排除文件保护平台 GPU 软件包，并明确目标解释器：

```bash
printf 'torch\ntorchvision\ntorchaudio\ntriton\n' > /mnt/workspace/rocm-excludes.txt

uv pip install --python /mnt/workspace/.venv/bin/python \
  modelscope transformers accelerate datasets trl peft \
  scikit-learn pandas tqdm ipykernel \
  --exclude /mnt/workspace/rocm-excludes.txt \
  -i https://mirrors.cloud.tencent.com/pypi/simple/
```

排除项及其依赖不参与本次解析，由平台环境提供；兼容性仍需真实导入检查。微调代码没有调用 vLLM，因此未沿用课程命令中的 vLLM 和 ROCm vLLM wheel 源，也不对全部包使用 `-U`。本次保留原版 BF16 LoRA 配置，后续按实际运行结果调整。

注册虚拟环境为 Notebook 内核：

```bash
/mnt/workspace/.venv/bin/python -m ipykernel install --user \
  --name gemma4-lora \
  --display-name "Python (gemma4-lora)"
```

保存并刷新页面，在 Select Kernel 中选择 `Python (gemma4-lora)`。终端激活环境不会自动切换 Notebook 内核；用下列 Notebook 单元格确认解释器和实际导入：

```python
import sys
import torch
from sklearn.metrics import accuracy_score, f1_score
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from modelscope import snapshot_download
from datasets import load_dataset

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
```

解释器应为 `/mnt/workspace/.venv/bin/python`，torch 和 HIP 应保留平台版本。通过后跳过教程安装单元，从导入依赖与全局配置开始。安装的包保存在环境中，但新内核仍需执行 import 和变量定义。

#### 环境问题与处理

| 现象 | 原因或依据 | 处理与经验 |
| --- | --- | --- |
| No virtual environment found | uv 没找到目标虚拟环境 | 建立 `.venv`；后续明确 `--python` |
| 下载 torch、nvidia-cublas、nvidia-cudnn-cu13 | 依赖解析选中另一份 torch | 中断；确认原 ROCm torch 仍正常；改用排除文件 |
| unexpected argument --no-deps-package | 本轮指导中给出了无效参数 | 不再使用；不可用整套安装 `--no-deps` 代替 |
| 固定 torch==2.12.0+git6bbd260 后解析无解 | 软件源没有平台定制构建；约束仍要求解析 torch | 放弃此约束方案，安装不再带 `-c rocm-constraints.txt`，改用 `--exclude` |
| h11 的 METADATA 缺失 | 包记录存在但元数据不完整 | 定向重装后，后续安装已不再报告此错误 |
| Checked 10 packages | 仅说明安装记录满足要求 | 不能据此认定导入成功，必须执行 import |
| No module named sklearn.metrics | scikit-learn 导入失败，引发 GenerationMixin、AutoProcessor、SFTTrainer 连锁错误 | 定向重装 scikit-learn；终端导入成功，后续 Notebook 验证成功 |
| 终端成功，Notebook 环境未明确 | 两者可能使用不同解释器 | 注册并切换内核，用 sys.executable 核对 |
| TqdmWarning: IProgress not found | Notebook 组件进度条不可用 | 本次为警告，未阻止配置单元继续输出；未作为训练环境失败依据 |
| GPU 名称为空 | 当前名称查询未返回字符串 | 如实记录；结合申请规格、HIP、显存和可用性判断，不虚构设备名 |

本次单包修复命令如下，仅遇到对应问题时执行：

```bash
uv pip install --python /mnt/workspace/.venv/bin/python \
  --reinstall --no-deps "h11==0.16.0" \
  -i https://mirrors.cloud.tencent.com/pypi/simple/

uv pip install --python /mnt/workspace/.venv/bin/python \
  --reinstall --no-deps scikit-learn \
  -i https://mirrors.cloud.tencent.com/pypi/simple/
```

这里 `--no-deps` 用于单包修复，不用于整套训练环境安装；修复后仍需导入验证。

#### 部署检查点

已完成 ROCm 设备访问、依赖修复和 Notebook 环境验证；后续进度见第三章，避免把环境检查点当作整项实验状态。

环境配置已通过，后续直接使用已注册的内核运行实验。

## 三、运行

已完成微调闭环。配置依据本次实际 Notebook，指标依据本地 CSV 与 JSON；GPU数据仍为运行期间的一次采样。

### 3.1 下载与加载模型、数据

- 从 ModelScope 下载模型及 emotion 数据集，检查本地路径和 parquet 文件布局。
- 加载 tokenizer，确认 chat template 可用；加载基础模型并检查实际设备、精度和参数量。
- 将数据转换为 prompt-completion，核对标签映射、模板化文本及截断后标签是否保留。
- 实际结果：模型已加载，SFTTrainer 已处理4000条训练数据与400条验证数据；下载目录调整为 `/root/tmp/models`（实际返回的模型子目录待补）。

### 3.2 微调前评估

使用固定测试样本、相同 prompt 与生成设置建立基线，保存指标和逐样本预测。关注 accuracy、macro-F1、INVALID 以及严格标签输出情况。

实际结果：已完成，详见下方“结果分析”。

### 3.3 LoRA 训练

已完成训练；trainer_state.json 记录 global_step=250、epoch=1.0，最终检查点为 checkpoint-250。

| 项目 | 本次已知结果 |
| --- | --- |
| 总参数（含当前适配器） | 7,991,600,416 |
| 可训练 LoRA 参数 | 50,499,584，占0.6319% |
| 数据预处理 | 训练4000条、验证400条；tokenize、构建labels、截断及全掩码样本筛选流程已执行；筛选后实际保留数待核对 |
| 预处理可见耗时 | 训练和验证 tokenizing 各约2秒；其余各阶段显示不足1秒；不含下载、模型加载与训练 |
| 训练调用耗时 | 1115.5292秒，约18分36秒；包含周期验证、保存等开销，不含下载、模型加载和独立前后测试评估 |
| 训练进度／loss | 250步，1 epoch；全程平均 train_loss=0.314208；吞吐3.586 samples/s、0.224 steps/s |

#### GPU 占用采样

采样时间：2026-10-26 18:01:06（用户提供的 amd-smi 显示时间；该输出未标注时区）。此表是一个时刻的观察，不代表平均值或峰值。

| 指标 | 瞬时值 |
| --- | --- |
| 整卡显存占用 | 17797 / 196288 MB，约占9.1%（按工具原始单位记录） |
| 进程显存 | 工具显示17.1 GB，PID 30572 |
| GFX 利用率 | 99% |
| Mem-Uti | 3%，不是显存容量占用比例 |
| 功耗 | 269 / 650 W |
| 温度 | 40°C |
| AMD-SMI / ROCm | 26.2.2+c2d9476115 / 7.2.3；HIP构建版本另见环境章节 |

日志中的可训练模块包含 `vision_tower`，说明 all-linear 也给视觉塔挂了适配器。当前为纯文本任务，挂载并不证明这些参数实际获得梯度。此轮先观察 loss 与训练进度；后续实验可核对语言模块路径，比较仅对语言模型应用 LoRA 的配置。

训练已完成；显存峰值尚未采集，不把以上瞬时采样当作整轮资源结论。

### 3.4 微调后评估与产物保存

复用训练后的模型，在同一批测试样本上评估，与基线比较。保存 LoRA adapter、tokenizer、训练指标、前后指标 CSV、逐样本预测和混淆矩阵，并确认云实例结束后产物仍可访问。

实际结果：已完成，详见下方“结果分析”。

### 3.5 结果分析（2026-10-26）

证据目录：[results/257-gemma4-emotion](../../../results/257-gemma4-emotion/)；实际代码：[本次 Notebook](257-gemma4-e4b-emotion-lora.ipynb)。前后预测文件的400条文本与真实标签逐行一致，未发现对齐错误。

#### 整体效果

| 指标 | 微调前 | 微调后 | 变化 |
| --- | ---: | ---: | ---: |
| Accuracy | 62.25%（249/400） | 92.00%（368/400） | +29.75个百分点 |
| Macro-F1 | 0.4751 | 0.8704 | +0.3954 |
| Weighted-F1 | 0.6072 | 0.9188 | +0.3116 |
| INVALID | 2 | 0 | 减少2条 |
| 错误数 | 151 | 32 | 减少119条，错误率相对下降78.8% |

逐样本对比：240条始终正确，128条从错变对，9条从对变错，23条始终错误。净增119条正确，不是仅消除了2条非法输出，说明任务分类结果有明显改善；不能据此推断模型通用能力提升。

#### 各类别效果

| 类别 | 测试数量 | 微调前F1 | 微调后F1 | 微调后Recall |
| --- | ---: | ---: | ---: | ---: |
| sadness | 114 | 0.6690 | 0.9351 | 94.74% |
| joy | 162 | 0.7372 | 0.9483 | 96.30% |
| love | 26 | 0.2439 | 0.7600 | 73.08% |
| anger | 47 | 0.4225 | 0.9318 | 87.23% |
| fear | 36 | 0.4444 | 0.9067 | 94.44% |
| surprise | 15 | 0.3333 | 0.7407 | 66.67% |

六类F1均改善。love和surprise仍是弱项；样本数仅26和15，单条结果即可改变Recall约3.85与6.67个百分点，不宜过度解读小类别的精确排名。sadness与joy合计占69%，因此需要同时看Macro-F1，不能只看Accuracy。

后评估的主要混淆：love→joy 5条、anger→sadness 5条、joy→love 4条、sadness→joy 3条、surprise→fear 3条。这些可以作为下一轮人工复核和补数据的入口；不要根据测试集错误直接改标签或调参后仍把同一测试集当作独立验证。

#### 收敛与耗时

| Step | Validation loss | Mean token accuracy |
| --- | ---: | ---: |
| 25 | 0.4049 | 84.56% |
| 50 | 0.3043 | 89.58% |
| 75 | 0.2255 | 92.98% |
| 100 | 0.1818 | 93.69% |
| 125 | 0.1597 | 94.13% |
| 150 | 0.1484 | 94.52% |
| 175 | 0.1378 | 95.29% |
| 200 | 0.1140 | 95.35% |
| 225 | 0.0936 | 96.29% |
| 250 | 0.0873 | 96.42% |

记录的验证loss连续下降，本轮没有观察到验证loss反弹；仅1 epoch和单个随机种子，不能证明继续训练不会过拟合。Token accuracy衡量教师强制下参与损失计算的token，最终测试Accuracy衡量生成后提取的情绪标签，二者口径不同。

训练调用总耗时1115.53秒，其中10次验证合计477.23秒（约7分57秒，占42.8%），单次约47.72秒。其余638.30秒包含训练、检查点保存及其他开销，不能直接当作纯GPU计算时间。

下一轮提速可优先比较 eval_steps=50或100，以及提高eval batch；若降低验证频率，会减少收敛观察点，需要保持其他配置一致。再分别测试更大训练batch配更少梯度累积、关闭梯度检查点，记录真实耗时与峰值，不能从17GB瞬时显存推算加速倍数。

#### 评估与复现边界

- 这是从原始测试split固定抽取400条的单次实验，未评估完整2000条测试集，也未跨随机种子重复。
- `raw_pred_label` 实际是 extract_label 处理后的结果，不是原始生成文本，现有CSV无法证明严格“只输出一个标签”。后续保存 decoded_text 和格式合规指标。
- 前后使用同一prompt与生成代码，但代码中基础模型评估 use_cache=False，训练后改为True；不宜直接比较前后推理速度。
- LoRA挂载含视觉模块；可训练参数数目不等于纯文本步骤中实际更新的参数数目，需梯度检查才能确认。
- 本地结果目录包含CSV与配置/状态JSON，不含adapter权重，尚不能仅依赖这些文件重载模型；完整权重、tokenizer及基础模型版本仍需另行保留。

### 运行配置（已核对本次 Notebook）

| 项目 | 预览页代码 |
| --- | --- |
| 加载 | AutoModelForCausalLM，本地路径，BF16，加载后 `.to(device)` |
| 数据上限 | train 4000 / validation 400 / test 400，生成评估 400 |
| LoRA | r=16，alpha=32，dropout=0.05，target_modules="all-linear" |
| batch | train=4，eval=1，梯度累积=4，单卡有效 batch=16 |
| 长度与损失 | max_length=256，packing=False，completion_only_loss=True |
| 训练 | 1 epoch，lr=1e-4，warmup=50，adamw_torch |
| 显存措施 | gradient_checkpointing=True；训练前 use_cache=False |
| 周期操作 | 每25步验证及保存，最多保留2个检查点 |
| 后评估 | 复用 trainer.model，不额外加载第二份模型；开启生成缓存 |


### 运行前需核对的代码细节

- `AutoModelForCausalLM` 对该仓库的支持取决于 transformers 版本；先确认模型类加载成功，再判断显存。配置映射或 chat template 报错不能靠降低 batch 解决。
- 当前 `extract_label` 会优先寻找任意合法标签，可能将“解释中包含标签”的输出也算作有效分类。它衡量的是可提取标签准确率；若要衡量严格遵循“只输出标签”，另记原始生成文本和严格格式合规率。
- 原正则在页面反序列化内容中疑似出现退格控制字符，应在实际 Notebook 中核对是否为 `r"\b(sadness|joy|love|anger|fear|surprise)\b"`。
- Gemma 4 的 thinking / chat template 设置与 `max_new_tokens=4` 应一起核对；若生成思考前缀，可能未输出标签就被截断，应先检查原始生成文本。
- 原可选重新加载单元没有将 reload_model 放到 GPU；独立重启后执行时需明确设备。训练后正常评估已复用模型，不必运行该可选单元。
- 原评估 CSV 在末尾统一保存。建议基线评估完成就保存 pre_metrics / pre_preds，避免会话中断丢失。


## 四、总结与下一步

### 已验证的经验

- ModelScope MI300X 镜像中的 ROCm PyTorch 可被虚拟环境继承；安装训练包时需避免另装其他后端的 torch。
- 安装记录满足要求不等于包文件完整或可导入，应以实际 import 和 GPU 检查为依据。
- 终端与 Notebook 使用的解释器需明确对齐，切换内核后再运行教程。

### 实验结论的边界

本轮400条固定测试样本的Accuracy从62.25%提升到92.00%，Macro-F1从0.4751提升到0.8704；训练调用约18分36秒。改进覆盖六类，但少数类样本有限；严格格式合规、跨种子稳定性和峰值显存仍未验证。

### 下一步

备份adapter权重及tokenizer；人工复核9条退化样本和主要混淆类别。下一轮先用验证集比较配置，锁定方案后再做完整测试集评估。

## 五、参考

- [Notebook 257 原始预览](https://developer.amd.com.cn/radeon/templates/257/preview)：本页配置与流程的直接依据，读取日期2026-10-26。
- [Google Gemma 4 内存要求](https://ai.google.dev/gemma/docs/core#gemma-4-inference-memory-requirements)：官方推理内存参考，不能当作训练峰值。
- [官方模型 config.json](https://huggingface.co/google/gemma-4-E4B-it/blob/main/config.json)：架构、词表、层数与dtype的参考；实际训练仍从ModelScope本地权重加载。




