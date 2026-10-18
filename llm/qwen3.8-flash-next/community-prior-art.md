# Qwen3.8-Flash-Next AMD 部署先例与激励规则调查

调查日期：2026-10-18。范围包括 ModelScope 社区帖子/模型卡、AMD × ModelScope 激励计划页面，以及 vLLM 上游关于 Qwen4Exp/AMD 的相邻故障记录。搜索未命中不能证明互联网上绝对不存在相关内容；结论限定为本次可检索公开资料。

## 与当前问题的关系

| 公开资料 | 已有内容 | 与当前排障的关系 |
| --- | --- | --- |
| [ModelScope：AMD ROCm 上部署 Qwen3.8-27B 多模态大模型](https://www.modelscope.cn/posts/436771) | MI300X / gfx942、ROCm 7.2.3、vLLM 0.28、PyTorch `2.12.0+git6bbd260`，记录服务与吞吐实测 | 环境相近，但它是 Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`），不是 Qwen3.8-Flash-Next 的 `Qwen4Exp` GDN/PLE 路径；未见本文所述的 Dynamo `Ne(128*u0,0)` reshape 修复 |
| [ModelScope：Qwen3.8-Flash-Next 官方模型卡](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/summary) | 列出 vLLM、SGLang、TokenSpeed recipe；介绍模型含 GDN、QSA、n-gram 表等结构 | 是部署入口和模型结构资料，不是 AMD MI300X + 自编译 vLLM O1 动态形状问题的复现/修复帖 |
| [ModelScope AMD 组织](https://modelscope.cn/organization/amd)中的 Quark Flash-Next 权重；其[部署卡](https://huggingface.co/amd/Qwen3.8-Flash-Next-Quark-MXFP4-PLEFP8)记载 MI350X/MI355X + SGLang | AMD 专用 MXFP4 experts / FP8 PLE checkpoint，SGLang 有 MI350/MI355 部署与 GSM8K 复现说明 | 同模型家族、AMD 实测，但硬件、权重精度/量化与框架都不同；不能视为当前官方 FP8 checkpoint 在 MI300X 上的 vLLM 编译问题已解决 |
| [vLLM Issue #61486：AMD Qwen4Exp PIECEWISE 回放读到过期的 PLE `query_start_loc` 尾部](https://github.com/vllm-project/vllm/issues/61486)及[关联修复提交](https://github.com/devtyagi3909/vllm/commit/bb5dc2b40a1e0a98dfcd7216611168808ef40356) | AMD Qwen3.8-Flash-Next 的 PLE 在 PIECEWISE graph replay 中可能因固定缓冲区尾部陈旧而错误分配 n-gram IDs；提交建议保留固定 buffer 并补齐尾部，报告中有 CPU 与 GPU 正确性检查 | 是最接近的同架构 AMD/PLE 先例，但属于**运行期 graph replay 数据正确性**，不是当前的 Dynamo 编译期 `reshape(...,-1,128)` guard。该页面显示 issue 仍为 Todo/开放状态；关联提交不等于已合并到用户安装的 vLLM |

本次在 ModelScope 精确检索 `Qwen3.8-Flash-Next + AMD/ROCm + vLLM`、`Qwen4Exp` 和 Dynamo 报错特征时，没有发现已发布同一 `Ne(128*u0,0)` / GDN `-1` reshape 修复的社区帖子。已有大量相关部署与优化材料，所以未来分享不宜声称“首次在 AMD 上跑通 Qwen3.8-Flash-Next”。更扎实的定位是：**MI300X/gfx942、ROCm 7.2.3、源码构建 vLLM 0.31.0、FP8 checkpoint 的 compiled/O1 动态形状故障最小复现与局部补丁验证**，再补上可复现启动时间、接口正确性、benchmark 和与 llama.cpp/SGLang 的公平对照。

## GPU 激励口径（按调查日公开规则）

- 较早的[AMD × ModelScope 1.0 说明](https://www.modelscope.cn/posts/433641)列出技术文章每篇 25 GPU 小时、最多 50 小时，以及独立的 CreateSpace 资源申请路线。
- 更新的[AMD × ModelScope 开发者激励计划 2.0](https://www.modelscope.cn/posts/437059)（2026-09-30）对新参与者列的是：合格原创实践审核通过后每篇 125 ADP 积分、每个 ID 最多 3 篇；符合条件的新 ADP 注册者另有一次性 100 小时 AMD GPU Notebook 权益。页面还说明旧的普惠 700 小时 CreateSpace 路径下线，1.0 已参与者按原阶段规则处理。
- 因此，不应按“现在发一篇文章就必得 25 GPU 小时”来规划。新旧活动资格、ADP 账号关联及积分兑换方式会影响实际所得；投稿前应按自己的账号参与状态核对 2.0 页面和活动入口。奖励不是保证审核通过或固定兑得某时长。

## 建议的分享差异化证据

1. 明确区分 ModelScope 已有的 Qwen3.8-27B 与本项目 Qwen3.8-Flash-Next / `Qwen4Exp`，列出模型 revision、GPU 架构、ROCm、Torch、vLLM commit 和 checkpoint 精度。
2. 给出原始完整错误、CPU 最小探针、目标源码上下文、方法级补丁、备份/回退方法及补丁后的完整启动结果；不能把 CPU 探针通过写成服务修复完成。
3. 若 O1 服务起来，除了 health/chat，再验证短 prefill 与不同请求序列的 PIECEWISE PLE 正确性，回应 Issue #61486 所示的相邻回放风险。
4. 启动时间与推理性能单独统计；分别说明模型权重加载、Dynamo 编译和服务 ready 的时间，避免把 eager、`mode=0`、O1 混为同一性能档。
5. 三框架横向对比前，先统一模型任务、输入输出 token、并发、KV/cache 策略及量化口径；当前 llama.cpp `UD-Q4_K_XL` 与 vLLM/SGLang FP8 是不同精度路线，结论应标为“框架+精度配置对比”，不能解读成纯框架排名。
