# GPU 实例使用与维护

## 当前实例对比（2026-10-28）

| 对比项 | AMD Radeon Cloud 国际站 | ModelScope Notebook（A10） |
| --- | --- | --- |
| SSH 连接 | 本机通过 `radeon-gpu` 别名和公钥直连实例；重建后地址和端口可能变化。 | 通过 Tailscale 私有网络间接连接，当前经 DERP 中继；配置和恢复步骤见 [ModelScope 远程 SSH](platforms/modelscope/remote-ssh/README.md)。 |
| VS Code | 配置实例 sshd 的 TCP 转发后，使用本机 VS Code Remote-SSH 直连。 | 可使用平台浏览器工作区的终端配置连接，也已验证本机 VS Code Remote-SSH 经 Tailscale 连接。 |
| 网络与模型下载 | 当前实例直连多个公网网站超时；平台出口代理可访问部分站点，但有访问限制。Codex 设备码登录经本机代理的 SSH 反向转发成功；实例内 Clash 节点直连超时，平台代理拒绝连接节点端口。 | 按用户当前使用经验，暂未遇到上述网络限制，可快速下载 ModelScope 模型；具体速度和其他站点可达性随实例及网络变化。 |

AMD 网络测试为当前实例的结果，不代表平台所有实例；ModelScope 下载速度尚无本次量化测量。

目标：在本机 VS Code 中连接云端 GPU 实例，模块化开发，并管理代码和实验产物。主要路线为 AMD Radeon Cloud 国际站与 ModelScope Notebook；时长、实例状态和连接能力以平台页面及实测为准。

| 路线 | 资源与当前状态 | VS Code 连接 |
| --- | --- | --- |
| AMD Radeon Cloud 国际站 | 用户提供约330可消耗开发者积分、每天最多兑换20小时；积分是待兑换额度，不等于已到账330 GPU-hours | 当前实例已验证 SSH 与 VS Code Remote-SSH；需在实例内启用 TCP 端口转发 |
| ModelScope Notebook | 用户提供普通GPU约220小时时长；余额、机型与实例限制以当前页面为准 | A10 已验证 Tailscale 私有 SSH 与 VS Code Remote-SSH；重启后恢复连接成功 |

## AMD Radeon Cloud

### 国际站、中国站与积分

- 下方 RCC 自动探测使用 Radeon Cloud **国际站**，服务地址为 [radeon-global.anruicloud.com](https://radeon-global.anruicloud.com/)。
- 2026-10-26 实测国际站 CLI 登录成功；中国站登录返回 403，调整请求后路由返回 404，尚未验证成功。这是当时的实测结果，不代表中国站永久不支持 CLI。
- 用户实际操作：使用国际站登录邮箱登录 AMD Developer Program 并完成关联，国际站页面显示 `Connected`；完成认证获得开发者积分后，在 [开发者积分兑换页面](https://developer.amd.com.cn/points/redeem)领取算力优惠券，再到国际站 `Redeem Credits` 兑换，供国际站实例使用。
- 国际站页面明确提示可从 AMD Developer China 获取优惠券并在此兑换。国际站账号余额从5增至15 GPU-hours，用户确认通过上述流程获得额度。
- [官方开发者计划文档](https://amd-aim.github.io/radeon-cloud-docs/zh-cn/guides/developer-program/)目前仍描述中国站兑换，并称全球站不使用该计划，与国际站页面及本次实际操作存在差异；以当前页面和实际兑换结果为本次记录依据。

### AMD GPU 连接与环境（2026-10-28）

#### Codex：通过本机代理登录

当前 AMD 实例使用平台出口代理请求 Codex 设备码返回 403；通过 SSH 将本机 HTTP 代理反向转发后，已完成 ChatGPT 设备码登录，CLI 模型请求和交互界面均验证成功。本机代理需监听 `127.0.0.1:7890`，在本机 PowerShell 运行并保持连接：

```powershell
ssh -N -R 127.0.0.1:17890:127.0.0.1:7890 radeon-gpu
```

在 AMD 远程终端设置代理，再运行插件自带 CLI（路径随插件更新而变化）：

```bash
export HTTP_PROXY=http://127.0.0.1:17890 HTTPS_PROXY=http://127.0.0.1:17890
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
/root/.vscode-server/extensions/openai.chatgpt-26.51021.22027/bin/linux-x86_64/codex --no-daemon
```

首次登录可用同一可执行文件运行 `login --device-auth`，在本机浏览器完成设备码授权。代理变量只对当前终端及其子进程生效；VS Code 插件使用该转发尚未验证。SSH 转发或本机代理停止后远程访问会中断；登录凭据不得提交到仓库。

本机已配置 `radeon-gpu` SSH 别名，VS Code Remote-SSH 可选择该别名；当前实例直连命令（实例重建后地址、端口可能变化）：

```powershell
ssh -i "$env:USERPROFILE\.radeon-cloud\keys\code12ab@gmail.com\id_ed25519" -p 35841 root@36.150.116.206
```

#### VS Code Remote-SSH：启用端口转发

2026-10-28 当前实例实测：SSH 登录正常，但平台启动的 sshd 使用 `/tmp/amd-oneclick-sshd_config`，默认 `AllowTcpForwarding no`，导致 VS Code 报 `administratively prohibited`。在实例终端执行以下命令，验证配置后重载服务，重新连接 `radeon-gpu` 已成功：

```bash
cp /tmp/amd-oneclick-sshd_config /tmp/amd-oneclick-sshd_config.bak
sed -i 's/^AllowTcpForwarding no$/AllowTcpForwarding yes/' /tmp/amd-oneclick-sshd_config
sshd -t -f /tmp/amd-oneclick-sshd_config && kill -HUP "$(cat /tmp/amd-oneclick-sshd.pid)"
```

`sshd -t` 成功时不输出；重载后需重新建立 Remote-SSH 连接。可用 `sshd -T -f /tmp/amd-oneclick-sshd_config | grep '^allowtcpforwarding'` 检查配置文件，应显示 `allowtcpforwarding yes`；最终以重新连接实测为准。该文件位于 `/tmp` 且由平台管理，实例重启或重建后是否保留尚未验证；新实例需先检查实际配置，确认平台允许修改后再操作。启用转发会改变实例 SSH 访问能力。若需撤销，可将备份恢复到原路径，验证配置后同样重载 sshd。

实测 Ubuntu 24.04、Python 3.12、ROCm 7.2.1、PyTorch 2.9.1、Transformers 4.57.6、vLLM 0.16.1 开发版和 JupyterLab 4.5.6；PyTorch 已识别 GPU。使用预装环境：

```bash
source /opt/venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

用户数据放 `/workspace`：当前约 93 GiB 可用，足够一般代码与实验，但大模型权重、数据和检查点可能超过容量，需提前估算、清理或使用外部存储。`/root` 位于容器的根 overlay，可作大模型临时缓存，`/root/.cache` 也是常见默认缓存位置；显示约 1.8 TiB 是底层文件系统的可用空间，并非已确认的个人配额，实例删除后不保留。真正的 `/tmp` 是 4 GiB 的内存 tmpfs，不适合放大模型。此次创建未显式要求持久化，不能仅凭容量确认 `/workspace` 可跨实例保留；重要数据另行备份，并在控制台核对工作区类型。[官方存储说明](https://amd-aim.github.io/radeon-cloud-docs/zh-cn/guides/storage/)。

### CLI 连接

本机 `stock` 环境可运行 `rcc`，国际站账号和公钥已配置。用户截图显示剩余330开发者积分，按本人反馈每天最多兑换20小时；到期和到账条件以平台实际提示为准。探测在北京时间04:00–08:00共8次均达到 ready，可优先在04:00或06:00尝试，但仅是一次夜间的初步观察，不能保证每天有资源。OCI 按30分钟继续三天探测；正式训练前先暂停探测并确认探测实例已释放。

```powershell
conda activate stock
rcc credits status
rcc instance create --env production-1 --pod-type one-click --resource-template resource-16c55g1u
rcc instance status
rcc instance ssh --print
```

2026-10-28 已实测当前实例的 SSH 公钥登录；实例重建后用 `ssh --print` 获取新连接信息。

### OCI 自动探测（国际站）

2026-10-26部署。代码见 [scripts/rcc-probe.py](scripts/rcc-probe.py)；OCI安装目录 `/opt/rcc-probe`，独立venv安装 `amd-oneclick-sdk==1.0.2`，系统服务以opc用户运行。认证文件仅保存在服务器 `auth/config.json`（0600），不入库，不复制SSH私钥。

#### 2026-10-28 保留实例计划

- 北京时间10月28日06:00–08:00（不含08:00），定时申请首次达到 ready 时保留实例，供持久化和 VS Code SSH 研究；窗口内请求启用 SSH。后续已验证 SSH 与 VS Code Remote-SSH，配置见上文。
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
- 可在 OCI 创建 `/opt/rcc-probe/REQUEST_PERSISTENT` 标记，让未来保留窗口内的申请显式传 `--use-pvc`；普通短时探测不受影响。标记不改变现有实例，也不创建新的保留窗口。服务端不支持或创建结果不明确时按原有保护停止，不自动降级或重复创建；申请持久化不等于挂载成功，结果缺少 `persistent_available` 时状态为未知，须核查平台。删除标记可关闭此选项。
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
