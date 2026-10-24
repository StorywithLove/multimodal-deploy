# AMD AI 学院学习记录：语言模型与扩散模型

- 学习日期：2026-10-23；整理日期：2026-10-24。
- 来源：个人课程随记；课程名称、链接与考试结果待补充。
- 状态：已记录概念和安装要点；尚无部署日志、生成产物或考试通过凭证。

## 1. 语言模型

以基于 Transformer 的自回归语言模型为例：

```text
输入文本 → 分词器 → token ID → 嵌入向量
        → 多层 Transformer → 输出层 → 下一个 token 的概率 → 解码并继续生成
```

分词器使用切分规则和有限词表；嵌入矩阵通常随机初始化，并随模型训练更新。注意力机制使表示包含上下文信息，生成 token 还需要输出层和解码策略。

## 2. 潜在扩散模型

以下以采用 CLIP 和 U-Net 的经典 Stable Diffusion 为例，通过在多个时间步中基于提示优化初始化的随机噪声来逐步提升图像质量。

```text
提示词 → 分词器 → CLIP 文本编码器 → 文本条件
                                      ↓
潜空间随机噪声 → U-Net 预测噪声 + 调度器迭代更新 → 去噪后的潜变量
                                      ↓
                                 VAE 解码 → 图像
```

![Stable Diffusion 各组件与张量尺寸](https://jalammar.github.io/images/stable-diffusion/stable-diffusion-components-and-tensors.png)

图示以 SD 1.x 的 512 × 512 输出为例：文本表示为 77 × 768，潜变量为 4 × 64 × 64；这些尺寸不是所有模型的统一规格。

本文配图来自 Jay Alammar 的 [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)，按原文 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可署名引用，通过原站链接加载。

### 2.1 文本表示生成
（1）分词。基于tokenizer实现。
（2）编码。CLIP 将文本编码为去噪网络使用的条件表示。CLIP（Contrastive Language-Image Pretraining）"语言-图像对比与训练"包含一个图片编码器和一个文本编码器，将两者的输出训练至一致，因此CLIP处理输入文本得到类似图片输入的嵌入向量。

### 2.2 图像表示精炼

![先编码原图，再向潜变量加入不同程度的噪声](https://jalammar.github.io/images/stable-diffusion/stable-diffusion-latent-forward-process-v2.png)

图中加噪发生在压缩后的潜空间；下面的公式描述如何直接构造指定时间步的训练样本。

（1）训练阶段：加噪。训练程序基于VAE编码后的原图潜变量 $x_0$、噪声 $\epsilon$ 构建题目。每次训练通常**随机选 $t$ 和噪声**，直接构造 $x_t$；加噪过程如下：
$$
x_t=\sqrt{\bar{\alpha}_t}\,x_0
+\sqrt{1-\bar{\alpha}_t}\,\epsilon
\qquad
\epsilon\sim\mathcal{N}(0,I)
$$
其中 $x_0$ 为干净数据，$\epsilon$ 为从标准正态分布采样的噪声，$t$ 为时间步，$\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$ 为累计信号保留系数。
为支持 CFG，训练时通常以一定概率随机丢弃文本条件。对每个样本，随机选择保留条件 $c$ 或替换为空条件 $\varnothing$，记实际使用的条件为 $\tilde c$。同一个 U-Net 因此同时学习有条件和无条件噪声预测；两种情况的目标都是实际加入的噪声 $\epsilon$。

U-Net 基于加噪潜变量 $x_t$、时间步 $t$ 和选定条件 $\tilde c$ 执行一次噪声预测：
$$
\hat{\epsilon}=\epsilon_\theta(x_t,t,\tilde c)
$$
最后比较 $\epsilon$ 和 $\hat\epsilon$ ，更新模型参数：
$$
\mathcal{L}=\mathbb{E}_{x_0,t,\epsilon,\tilde c}\left[\|\epsilon-\epsilon_\theta(x_t,t,\tilde c)\|^2\right]
$$
每次训练迭代通常直接构造随机时间步的数据并预测噪声，不执行完整的多步生成，也通常不在此处组合两份 CFG 预测。反向传播更新的是模型参数。

![正向扩散训练样本：向干净数据加入噪声，得到带噪输入](https://jalammar.github.io/images/stable-diffusion/stable-diffusion-forward-diffusion-training-example.png)

可将图中的加噪样例对应到一次训练抽样：随机选择时间步 $t$，采样噪声 $\epsilon$，按前面的公式构造 $x_t$。其中 $t$ 决定信号与噪声的混合比例，实际采样的 $\epsilon$ 保留为预测目标；下图展示预测、计算损失与更新参数的过程。

![U-Net 训练：预测噪声、与真实噪声计算损失，再反向传播更新参数](https://jalammar.github.io/images/stable-diffusion/stable-diffusion-u-net-noise-training-step.png)

这张图用带噪图片直观展示训练逻辑；在 Stable Diffusion 中，实际输入和预测噪声都是潜变量形状的张量。图中的反向传播属于训练阶段，生成时不执行这一步。

（2）生成阶段：预测噪声并迭代移除。基于已经训练好的 U-Net，从随机采样的潜空间噪声开始，每步根据噪声预测更新潜变量，**使其逐步从高噪声状态转向低噪声状态**。完整相邻步采样可写为 $x_T\to x_{T-1}\to\cdots\to x_0$；若只采样 50 步，则通常跨步选取时间位置，并非只从 999 走到 950。生成阶段模型参数固定，每步更新的是潜变量。

![生成过程中不同去噪步骤的潜变量可视化](https://jalammar.github.io/images/stable-diffusion/stable-diffusion-denoising-steps-latents.png)

图示展示生成过程中选取的若干步骤：潜变量经图像解码器可视化后，逐渐呈现轮廓和细节。实际迭代更新发生在潜空间中；为观察过程而解码中间结果，并非每一步生成都必须执行的操作。

① CFG（Classifier-Free Guidance，无分类器引导）：与训练时随机选择条件不同，使用 CFG 的生成过程在每个采样步都对同一个当前潜变量 $x_t$ 和时间步 $t$，分别计算无条件和有条件两份噪声预测（实现时可合并成一次批量前向计算）：
$$
\epsilon_u=\epsilon_\theta(x_t,t,\varnothing),
\qquad
\epsilon_c=\epsilon_\theta(x_t,t,c)
$$
其中 $u$ 表示无文本条件（空提示 $\varnothing$），$c$ 表示有文本条件。然后组合：
$$
\hat\epsilon_{\mathrm{CFG},t}=\epsilon_u+s\left(\epsilon_c-\epsilon_u\right)
$$
其中 $\epsilon_c-\epsilon_u$ 表示文本条件使噪声预测发生的变化，再以 guidance scale $s$ 加权叠加回无条件噪声预测。两者都是预测张量，不是文本贴合度或图片质量评分。$\hat\epsilon_{\mathrm{CFG},t}$ 是当前步组合后的预测，每个采样步都会重新计算，而非仅最后一步使用。

② 去噪更新。将 $\hat\epsilon_{\mathrm{CFG},t}$ 作为预测噪声，交给采样器更新图像潜变量。

- 调度器（噪声 / 时间步调度）：安排各步的噪声等级及生成时选取的时间位置，决定“在哪些噪声等级之间前进”。
- 采样器：利用模型预测，按选定的数值更新规则计算下一步潜变量，并决定是否引入额外随机噪声，负责“怎样走到下一步”。
- 两者配合影响生成速度和结果；术语在不同工具中可能合并，例如 Diffusers 的 scheduler 通常同时封装时间步安排与采样更新。

以下为经典 DDPM 的相邻时间步更新，跨步采样需采用相应采样器与时间步安排所对应的更新公式：
$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\hat\epsilon_{\mathrm{CFG},t}
\right)
+\sigma_t\eta
$$
其中 $\beta_t=1-\alpha_t$；$\eta\sim\mathcal N(0,I)$ 是另外采样的随机噪声，不是 CFG 组合后的预测，末步通常不再加入。每次更新后，用新的潜变量和下一时间步再次执行“分别预测 → CFG 组合 → 采样更新”。

### 2.3 图像上采样
步骤完成后，Stable Diffusion使用Decoder对图片表示进行上采样。
- 颜色：像素空间扩散可以直接使用 RGB 三通道噪声；经典 Stable Diffusion 使用四通道潜空间噪声，各通道没有直接的 RGB 含义，最终由 VAE 解码为彩色图像。
- 分辨率：经典 Stable Diffusion 的潜空间宽、高通常各为输出的 1/8；例如 128 × 128 × 4 的潜变量可解码为 1024 × 1024 × 3 的图像。具体比例取决于模型与 VAE。

![自编码器将图像压缩为潜变量，再由解码器重建图像](https://jalammar.github.io/images/stable-diffusion/stable-diffusion-autoencoder.png)

图中展示的是编码与重建过程。文生图推理从随机潜变量开始，去噪结束后只使用右侧的图像解码器，无需先编码一张原图。

## 3. 参数与使用技巧

| 参数 | 作用与注意事项 |
| --- | --- |
| 采样器 / 调度器 | 决定更新方式，应与模型和工作流匹配。 |
| 采样步数 | 更多步骤通常增加耗时；画质收益可能饱和，并非越多越好。 |
| 引导强度（CFG） | 调整文本条件的影响；过高可能损害画质，不能简单等同于创造性降低。 |
| 分辨率 | 影响耗时与显存；先在模型支持的较低分辨率验证。 |
| 随机种子 | 固定种子便于比较，同时需固定模型、参数和环境；跨设备不保证完全一致。 |
| 输入图像 / 去噪强度 | 图生图需记录输入图和强度，以解释与原图的差异。 |

## 4. 视频生成与 ComfyUI 随记

原始笔记记录了 AMD / ROCm 安装思路，尚未保留可复现命令或验证结果：

1. 核实 GPU、系统、ROCm 与 PyTorch 的兼容组合。
2. 原笔记“从 requirements.txt 删除 CUDA 版 torch”应理解为避免依赖覆盖已安装的 ROCm 版 PyTorch，不能作为通用安装步骤；以所选版本官方说明为准。
3. 典型工作流需要扩散模型、VAE 和文本编码器，具体文件与组件数量以工作流要求为准。
4. 视频模型可能需要额外组件；当前记录不足以确定具体视频模型或确认已跑通视频生成。

## 5. 待完成事项

- [ ] 打开昨日所学课程的考核入口，确认是否有可参加的考试及通过要求。
- [ ] 补充课程名称、链接与实际学习/考试状态。
- [ ] 记录硬件、软件版本、安装命令及实际错误（如有）。
- [ ] 保存一个最小 ComfyUI 工作流、生成样例及完整参数。

## 6. 整理时核对的资料

以下资料用于校正概念，不作为昨日操作完成的证明：
- [CLIP: Connecting text and images](https://openai.com/index/clip/)
- [Diffusers：管线、模型与调度器](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)
- [Stable Diffusion 文生图参数](https://huggingface.co/docs/diffusers/v0.35.0/en/api/pipelines/stable_diffusion/text2img)
- [Diffusion Explainer：交互演示](https://poloclub.github.io/diffusion-explainer/)
- [Jay Alammar：图解 Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
