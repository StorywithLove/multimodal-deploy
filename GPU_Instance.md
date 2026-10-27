# GPU 实例使用与维护

目标：在本机 VS Code 中连接云端 GPU 实例，模块化开发，并管理代码和实验产物。主要路线为 AMD Radeon Cloud 国际站与 ModelScope Notebook；时长、实例状态和连接能力以平台页面及实测为准。

| 路线 | 资源与当前状态 | VS Code 连接 |
| --- | --- | --- |
| AMD Radeon Cloud 国际站 | 用户提供约330可消耗开发者积分、每天最多兑换20小时；积分是待兑换额度，不等于已到账330 GPU-hours | RCC 登录、公钥配置和申请流程已验证；带 SSH 的实例端到端连接待实测 |
| ModelScope Notebook | 用户提供普通GPU约220小时时长；余额、机型与实例限制以当前页面为准 | A10 已验证 Tailscale 私有 SSH 与 VS Code Remote-SSH；重启后恢复连接成功 |

## AMD Radeon Cloud

### 国际站、中国站与积分

- 下方 RCC 自动探测使用 Radeon Cloud **国际站**，服务地址为 `radeon-global.anruicloud.com`。
- 2026-10-26 实测国际站 CLI 登录成功；中国站登录返回 403，调整请求后路由返回 404，尚未验证成功。这是当时的实测结果，不代表中国站永久不支持 CLI。
- 用户实际操作：使用国际站登录邮箱登录 AMD Developer Program 并完成关联，国际站页面显示 `Connected`；完成认证获得开发者积分后，在 [开发者积分兑换页面](https://developer.amd.com.cn/points/redeem)领取算力优惠券，再到国际站 `Redeem Credits` 兑换，供国际站实例使用。
- 国际站页面明确提示可从 AMD Developer China 获取优惠券并在此兑换。国际站账号余额从5增至15 GPU-hours，用户确认通过上述流程获得额度。
- [官方开发者计划文档](https://amd-aim.github.io/radeon-cloud-docs/zh-cn/guides/developer-program/)目前仍描述中国站兑换，并称全球站不使用该计划，与国际站页面及本次实际操作存在差异；以当前页面和实际兑换结果为本次记录依据。

### 计划中的 VS Code 连接

本机 `stock` 环境可运行 `rcc`，国际站账号和公钥已配置。用户截图显示剩余330开发者积分，按本人反馈每天最多兑换20小时；到期和到账条件以平台实际提示为准。探测在北京时间04:00–08:00共8次均达到 ready，可优先在04:00或06:00尝试，但仅是一次夜间的初步观察，不能保证每天有资源。OCI 按30分钟继续三天探测；正式训练前先暂停探测并确认探测实例已释放。

```powershell
conda activate stock
rcc credits status
rcc instance create --env production-1 --pod-type one-click --resource-template resource-16c55g1u
rcc instance status
rcc instance ssh --print
```

这是目标路径：国际站申请过且已配置公钥，但带 SSH 的 ready 实例尚未实测登录。成功后使用 `ssh --print` 输出配置 VS Code Remote - SSH；保存数据并按平台指引释放实例。

### OCI 自动探测（国际站）

2026-10-26部署。代码见 [scripts/rcc-probe.py](scripts/rcc-probe.py)；OCI安装目录 `/opt/rcc-probe`，独立venv安装 `amd-oneclick-sdk==1.0.2`，系统服务以opc用户运行。认证文件仅保存在服务器 `auth/config.json`（0600），不入库，不复制SSH私钥。

#### 2026-10-28 保留实例计划

- 北京时间10月28日06:00–08:00（不含08:00），定时申请首次达到 ready 时保留实例，供持久化和 VS Code SSH 研究；窗口内请求启用 SSH。公钥已配置，但连接尚未端到端验证。
- 成功后脚本和清理任务均跳过该实例，不再提交探测；实例可能持续扣除 GPU-hours，需用户使用完后手动释放。若窗口内未成功，普通探测按原计划继续。
- 原有3天期限、0.25 GPU-hour 观测下降停止阈值、已有实例和认证异常保护仍生效，可能使窗口内不发起申请；余额阈值不是硬扣费上限。
- 本机检查：`rcc instance status`；就绪后 `rcc instance ssh --print` 查看连接方式。结束实验并确认产物保存后，用 `rcc instance delete --yes` 释放，随后 `rcc instance status` 核实实例消失。保留标记不会自动清除，释放后如需恢复探测需先检查 OCI 状态并人工处理。

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

### llama.cpp 网关记录与 SSH

[项目 llama.cpp 记录](llm/qwen3.8-flash-next/llama.cpp.md)中服务绑定实例内 `127.0.0.1:8000`，经 `<workspace-id>-proxy-8000.dsw-gateway-...` 登录态 HTTPS 网关访问。[Notion 大模型笔记](https://app.notion.com/p/3fa12e6731688158a60ecb9ebba3e8b0)也记录了8000端口和网关关系。这验证的是 HTTP 服务，不证明 SSH 转发；域名解析的网关 IP 不能直接当作实例公网 IP。

历史 ModelScope 实例曾检测到 `/etc/dsw/runtime/bin/sshd -p 22 -D`，并收到 `SSH-2.0-OpenSSH_10.0`；公钥认证开启，root密码登录受限。当前实例仍需单独验证，实例内部有 sshd 不证明本机可连接。

2026-10-27 A10 实例内部的系统 sshd 监听 22，root 密码登录禁用；该监听本身不证明外部可达。随后使用 Tailscale 用户态网络和独立的回环 sshd 建立了私有 SSH：Windows 公钥登录、SCP 小文件上传及 SHA-256 核对成功；服务切换至 `/mnt/workspace/lcx/modelscope-remote-ssh`。重启后恢复服务，新实例复用原 Tailscale IP，Windows SSH 和用户的 VS Code Remote-SSH 均连接成功。当前经过 DERP 中继，SSH 端口转发尚未端到端验证。部署、登录和重启恢复步骤见 [ModelScope 远程 SSH](platforms/modelscope/remote-ssh/README.md)。

同日 ProxyClient 实测：以网页路径中的工作区编号作为实例 ID 时返回 `Entity does not exist`；改用登录回调中的 `instanceId` 后，返回 `Get proxy url failed`，连接仍未建立。未取得底层错误，不能据此断言是账号权限、实例归属、地域或网络中的哪一项；需核对 PAI 控制台实例归属和地域。

### 其他连接方式与 VS Code

1. 已验证 Tailscale SSH 与 VS Code Remote - SSH；使用时打开远端项目目录并选择远端 Python／Jupyter 内核。
2. 若免费实例可在本人阿里云 PAI 控制台查看并授权，官方 [DSW ProxyClient](https://www.alibabacloud.com/help/en/pai/connect-to-a-dsw-instance-over-ssh) 可通过阿里云身份及 SSH 公钥连接，无需 NAT/EIP；它要求实例 ID、地域和有权访问的云账号凭据。**ModelScope 平台免费实例是否提供这类权限尚未验证。**
3. 若平台提供直连地址，可按 [DSW 直连文档](https://www.alibabacloud.com/help/en/pai/dsw-direct-connection)使用公钥与指定端口。阿里云公网直连的网络组件可能单独计费；不自行创建付费资源。
4. 连接后在 VS Code Remote - SSH 打开 `/mnt/workspace`，选择远端 Python／Jupyter 内核；Remote - SSH 编辑远端文件，不自动同步本地项目，代码可用 Git 管理。

本机已检测到 OpenSSH 和 VS Code；`proxyclient.exe` 未在 PATH 中，默认 ProxyClient 配置文件不存在。ProxyClient 与平台直连路线均未完成端到端验证；WebIDE 可作为恢复入口。

[ModelScope 开源教程](https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-04.md)说明 `/mnt/workspace` 用于持久保存；其他路径不应假定跨实例保留。教程示例额度与用户页面不同，具体以当前账号为准。

## 国产 GPU：模力方舟 / 沐曦

- 已完成文本、图像、语音、OCR 和向量检索等实践，实例部署与证据见 [沐曦文本生成实践](platforms/moark/metax/文本生成模型.md)；具体 GPU 型号与资源以实际实例为准。

## AMD MI300X：ModelScope DSW

- 已使用 MI300X 192GB 完成 Gemma 4 情绪分类 LoRA 训练；环境与结果见 [微调记录](platforms/amd_developer/notebook_ROCm/257-gemma4-e4b-emotion-lora.md)。
