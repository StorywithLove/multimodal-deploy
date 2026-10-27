# GPU 实例使用与维护

按平台与硬件记录实例资源、连接方式、额度及维护经验；具体使用状态以实例与日志为准。

## AMD Radeon Cloud

### 国际站、中国站与积分

- 下方 RCC 自动探测使用 Radeon Cloud **国际站**，服务地址为 `radeon-global.anruicloud.com`。
- 2026-10-26 实测国际站 CLI 登录成功；中国站登录返回 403，调整请求后路由返回 404，尚未验证成功。这是当时的实测结果，不代表中国站永久不支持 CLI。
- 用户实际操作：使用国际站登录邮箱登录 AMD Developer Program 并完成关联，国际站页面显示 `Connected`；完成认证获得开发者积分后，在 [开发者积分兑换页面](https://developer.amd.com.cn/points/redeem)领取算力优惠券，再到国际站 `Redeem Credits` 兑换，供国际站实例使用。
- 国际站页面明确提示可从 AMD Developer China 获取优惠券并在此兑换。国际站账号余额从5增至15 GPU-hours，用户确认通过上述流程获得额度。
- [官方开发者计划文档](https://amd-aim.github.io/radeon-cloud-docs/zh-cn/guides/developer-program/)目前仍描述中国站兑换，并称全球站不使用该计划，与国际站页面及本次实际操作存在差异；以当前页面和实际兑换结果为本次记录依据。

### OCI 自动探测（国际站）

2026-10-26部署。代码见 [scripts/rcc-probe.py](scripts/rcc-probe.py)；OCI安装目录 `/opt/rcc-probe`，独立venv安装 `amd-oneclick-sdk==1.0.2`，系统服务以opc用户运行。认证文件仅保存在服务器 `auth/config.json`（0600），不入库，不复制SSH私钥。

#### 策略

- systemd `rcc-probe.timer` 每小时00、30分触发，首次成功读取余额时开始计时，3天后不再新增申请。
- 规格 `production-1 / one-click / resource-16c55g1u`，目录报价1 GPU-hour/小时；实际扣费以余额为准。不启用SSH，探测不执行工作负载。
- 单次最多等待约120秒，每10秒查状态；发现ready立即请求删除，成功释放须再次确认not_found。
- 独立 `rcc-probe-cleanup.timer` 每分钟检查过期且属于探测的实例。两个任务共用文件锁；主任务退出或超时后，清理任务可继续重试。
- 每次仅在账号无实例时创建；只删除记录ID匹配的探测实例，不接管其他实例。该账号不应同时手动创建实例；正式训练前先暂停并确认探测实例已清理。
- 累计观测余额下降达到0.25 GPU-hour后停止新增申请；充值不会抵消已观察到的下降。它是停止阈值，并非服务端硬预算，单次扣费和余额延迟可能导致超出；该账号的其他使用也可能触发阈值。
- 认证、查询或删除异常不应被当作资源不足；创建结果不确定时停止新增申请，需人工确认，避免误删或重复提交。
- 网络故障时无法保证即时释放；日志出现cleanup_pending或error时需检查。这里没有配置外部消息通知。

#### 运维命令（OCI终端）

```bash
# 暂停新增申请；清理watchdog保持运行
touch /opt/rcc-probe/PAUSE
sudo systemctl stop rcc-probe.timer

# 查看状态与脱敏事件
sudo systemctl list-timers 'rcc-probe*'
tail -n 20 /opt/rcc-probe/events.jsonl

# 按北京时间两小时区间汇总样本
RCC_PROBE_HOME=/opt/rcc-probe /opt/rcc-probe/venv/bin/python /opt/rcc-probe/probe.py report

# 恢复（不会重置原来的3天截止时间或额度基线）
rm -f /opt/rcc-probe/PAUSE
sudo systemctl start rcc-probe.timer
```

`events.jsonl`记录UTC时间、申请ID、结果、等待时长、清理确认及余额。汇总换算北京时间；成功率表示“120秒窗口内达到ready”的比例，不是完整空闲率。按已提交样本统计；已有实例、暂停、认证失败等跳过事件不计入成功率。3天样本有限，需结合各时段样本数判断，不预先承诺可用时间。

#### 首夜结果（10月27日10:30，截至北京时间）

23次探测中15次达到 ready；04:00–08:00为8/8，08:00–10:00为2/4，10:00–10:30为0/2。成功启动通常约22–26秒；仅一天样本，尚不足以确定稳定时段。

两次清理请求报错，清理任务随后确认实例已释放。余额由最初5升至15 GPU-hours，最近一次探测前后均为15；不能据此断言整晚无扣费。已修复充值后余额下降可能漏计的问题，并通过模拟测试。两项定时器继续运行，计划于10月29日23:43停止新增申请。

#### 部署注意

OCI的SELinux阻止了从home目录启动的系统服务，最终改用 `/opt/rcc-probe` 与系统 `/usr/bin/python3` 创建venv，并恢复目录默认SELinux标签；没有关闭SELinux。服务单元在 `/etc/systemd/system/rcc-probe*.service` 和 `.timer`。

凭据为现有CLI scoped token，过期需重新授权；不得把服务器认证文件加入仓库。用户日常在本机使用stock环境的RCC不受影响，但它操作的是同一云账号。

## NVIDIA GPU：ModelScope DSW

- 用户所见免费 GPU 环境为24GB显存，曾计划使用 A10；实际型号与时长以实例页面为准。[A10 模型候选清单](configs/benchmark/a10-24gb-model-shortlist-2026-10-19.md)。
- 尚无该候选清单对应的 A10 微调完成记录。

## 国产 GPU：模力方舟 / 沐曦

- 已完成文本、图像、语音、OCR 和向量检索等实践，实例部署与证据见 [沐曦文本生成实践](platforms/moark/metax/文本生成模型.md)；具体 GPU 型号与资源以实际实例为准。

## AMD MI300X：ModelScope DSW

- 已使用 MI300X 192GB 完成 Gemma 4 情绪分类 LoRA 训练；环境与结果见 [微调记录](platforms/amd_developer/notebook_ROCm/257-gemma4-e4b-emotion-lora.md)。
