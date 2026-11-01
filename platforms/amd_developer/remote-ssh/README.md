# AMD Radeon Cloud：远程 SSH 与开发环境

## 装包先换源，不要先上代理（2026-11-01）

**绝大多数"下载卡住 / timed out"不需要反向代理，换镜像源就够了。** 平台白名单只放行了 PyPI 的索引域，真正下发 wheel 的 `files.pythonhosted.org` 被压到 43 KB/s：

| 源 | 实测速度 |
|---|---|
| `files.pythonhosted.org`（uv/pip 默认） | 43 KB/s |
| `pypi.tuna.tsinghua.edu.cn` | 5.3 MB/s |
| `mirrors.aliyun.com` | 49.8 MB/s |

```bash
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/   # uv，0.10.9 实测生效
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/      # pip
```

400 MB 的依赖按默认源要两小时以上，换镜像后十几秒。变量只对当前 shell 及其子进程生效，先 export 再跑安装脚本。

判断是限速而不是卡死：uv 打完一屏 `Downloading xxx (4.1MiB)` 只是待下清单，不是进度条；用 `ps -eo pid,etime,time,args | grep uv` 看 CPU 时间几乎不涨，`/proc/net/dev` 下行只有几十 KB/s。**索引通不等于包体通**——本次先验证了 `pypi.org/simple` 返回 200 就以为可用，实际卡在包体域上。

需要访问 PyPI 之外的站点（GitHub、Hugging Face 等）时才看下面的代理方案。

本文记录当前实例的连接、VS Code Remote-SSH、Codex 代理和环境配置。实例重建后地址、端口及平台配置可能变化。

### AMD GPU 连接与环境（2026-10-28）

#### OCI 中转反向代理（2026-11-01，已停止）

**状态：2026-11-01 已停止并禁用。** 隧道可以打通，但下载吞吐只有 51 KB/s（详见下方实测表），装包不可用；而 PyPI 本来就在平台白名单内，日常无需隧道。OCI 上的 `radeon-proxy.service`、`radeon-tunnel.service` 均已 `disable --now`，AMD 侧 `127.0.0.1:17890` 已释放，OCI 公钥也已从 `/root/.ssh/authorized_keys` 移除（该键带 exec 能力，停用后没有必要留着）。下面的内容保留，供需要时重新部署或排障；AMD 侧 sshd 的 `AllowTcpForwarding yes` 保留未回滚（VS Code Remote-SSH 端口转发也依赖它）。

AMD 实例的出口受平台白名单代理限制：直连 github、1.1.1.1、任何节点 IP 全部超时；平台注入的 `notebook-egress-proxy...:3128` 对任意目标 CONNECT 返回 403，只放行白名单域名。因此在实例内跑 clash/mihomo 不可行——节点连不上，而且 Go 核心不吃 `HTTP_PROXY`。本机反向代理可用，但依赖 Windows 开机。

改用 OCI 那台常驻机器做出口。**方向必须反着来**：AMD 主动出站会被 403 挡住，只能由 OCI 主动拨入 AMD，把 OCI 本地代理反推过去。

```
AMD 127.0.0.1:17890
   ↑ sshd 反向转发（restrict,permitlisten="127.0.0.1:17890"）
OCI radeon-tunnel.service（opc 用户级单元）
   → OCI 127.0.0.1:18888（radeon-proxy.service，CONNECT 代理）
   → OCI 国际出口 → 外网
```

OCI 直连出口极快（github 36ms、huggingface 31ms、pypi 20ms），不需要 glados 节点。

**组件**

- OCI `/data/lcx/radeon-tunnel/connect_proxy.py`：只支持 CONNECT 的最小代理，监听 `127.0.0.1:18888`。
- OCI `radeon-proxy.service`（系统单元，`opc` 用户运行）：启动上面的脚本。
- OCI `radeon-tunnel.service`（**opc 用户单元** `~/.config/systemd/user/`）：`ssh -NT -o ExitOnForwardFailure=yes -R 127.0.0.1:17890:127.0.0.1:18888 root@<AMD实例> -p <端口>`，`Restart=always`，已 `loginctl enable-linger opc`。
- OCI 公钥在 AMD `/root/.ssh/authorized_keys`，前缀 `no-pty,no-agent-forwarding,no-X11-forwarding,permitlisten="127.0.0.1:17890"`，只能开这一个端口。

  这里**不能用 `restrict`**：`restrict` 本身已包含 `no-port-forwarding`，`permitlisten` 压不住它，结果是每次连接都报 `remote port forwarding failed for listen port 17890`，而端口其实是空闲的。必须显式列出要禁的项。

**为什么用用户单元而不是系统单元**：Oracle Linux 9 SELinux 为 Enforcing，系统单元运行在 `init_t` 域，执行 `ssh_exec_t` 被拒绝，`Failed to locate executable /usr/bin/ssh: Permission denied`：

```
avc: denied { execute } for comm="(ssh)" name="ssh"
scontext=system_u:system_r:init_t tcontext=system_u:object_r:ssh_exec_t permissive=0
```

`systemd-run --uid=opc` 同样失败，只有用户级单元（非受限域）能 exec ssh。

**AMD 侧启用转发**

```bash
cp /tmp/amd-oneclick-sshd_config /tmp/amd-oneclick-sshd_config.bak
sed -i 's/^AllowTcpForwarding no$/AllowTcpForwarding yes/' /tmp/amd-oneclick-sshd_config
sshd -t -f /tmp/amd-oneclick-sshd_config && kill -HUP "$(cat /tmp/amd-oneclick-sshd.pid)"
```

终端里设置（只对当前终端及子进程生效）：

```bash
export HTTP_PROXY=http://127.0.0.1:17890 HTTPS_PROXY=http://127.0.0.1:17890
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
```

2026-11-01 从 AMD 实测：github 200、api.github 200 1.9s、huggingface 200 1.2s、pypi 200 0.76s。

**实测吞吐（2026-11-01，1MB 文件）**

| 路径 | 速度 |
|---|---|
| OCI 本地代理 → 外网 | 14 MB/s |
| AMD → OCI（AMD 发出） | 0.7 MB/s |
| OCI → AMD（AMD 下载，隧道的实际方向） | **51 KB/s** |

瓶颈是跨洋链路的**下行**方向：OCI↔AMD TCP 握手 71–218ms 且抖动，高 RTT 叠加丢包把吞吐压到几十 KB/s。同一代理在 OCI 本地访问 github 仅 35ms；经隧道则要 8–27s。

**结论：这条隧道只够跑小请求，不要指望它装包。** 而且多数情况下并不需要它——平台 egress proxy 的白名单本来就放行 PyPI 镜像：

```
https://mirrors.aliyun.com/pypi/simple/      200  0.10s
https://pypi.tuna.tsinghua.edu.cn/simple/    200  0.20s
https://pypi.org/simple/                     200  0.49s
```

`uv`/`pip` 直接用平台默认代理即可（实测 `Resolved 1 package in 235ms`），阿里云镜像 180ms 更快。隧道留给确实需要、且流量小的访问。
- **实例重建后必须手动重接**，IP、端口、主机密钥全变：

```bash
# OCI：改 unit 里的 IP 与 -p 端口后
ssh-keygen -R '[旧IP]:旧端口' -f /home/opc/.ssh/known_hosts_radeon
systemctl --user restart radeon-tunnel.service
```

AMD 侧同时要重做 `AllowTcpForwarding`，并把 OCI 公钥重新加回 `authorized_keys`。

**写 authorized_keys 的坑**：该文件末尾没有换行符，直接 `>>` 追加会和最后一把公钥粘成一行，导致全部密钥失效、SSH 登录被拒（2026-11-01 实际踩过）。追加前先补换行：

```bash
printf '\n' >> /root/.ssh/authorized_keys
printf '%s\n' 'restrict,permitlisten="127.0.0.1:17890" ssh-ed25519 AAAA... comment' >> /root/.ssh/authorized_keys
```

**排障：客户端被杀后 AMD 端口仍被占**

客户端异常退出（如本机 `Stop-Process`）时，sshd 侧对应的子进程可能不释放 `-R` 监听，表现为端口能连上但数据不通、OCI 侧每 12 秒报一次 `remote port forwarding failed`。实例上 `ss`/`lsof` 都没装，用 `/proc` 找持有者：

```bash
python3 - <<'PY'
import os
port_hex = format(17890, "04X")
inodes = {f.split()[9] for path in ("/proc/net/tcp","/proc/net/tcp6")
          if os.path.exists(path)
          for f in open(path).read().splitlines()[1:]
          if f[3] == "0A" and f[1].split(":")[-1] == port_hex}
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        for fd in os.listdir("/proc/%s/fd" % pid):
            link = os.readlink("/proc/%s/fd/%s" % (pid, fd))
            if link.startswith("socket:[") and link[8:-1] in inodes:
                print(pid, open("/proc/%s/cmdline" % pid).read().replace("\0"," ").strip()); break
    except OSError:
        pass
PY
```

输出的 `sshd: root` 子进程即为僵尸会话，`kill <pid>` 后重启隧道服务即可。

**回滚**：OCI 上 `systemctl --user disable --now radeon-tunnel.service`、`sudo systemctl disable --now radeon-proxy.service`；AMD 上还原 sshd 备份并重载，删除 `authorized_keys` 中 OCI 那条。

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


## AMD Radeon Cloud

### 国际站、中国站与积分

- 下方 RCC 自动探测使用 Radeon Cloud **国际站**，服务地址为 [radeon-global.anruicloud.com](https://radeon-global.anruicloud.com/)。
- 2026-10-26 实测国际站 CLI 登录成功；中国站登录返回 403，调整请求后路由返回 404，尚未验证成功。这是当时的实测结果，不代表中国站永久不支持 CLI。
- 用户实际操作：使用国际站登录邮箱登录 AMD Developer Program 并完成关联，国际站页面显示 `Connected`；完成认证获得开发者积分后，在 [开发者积分兑换页面](https://developer.amd.com.cn/points/redeem)领取算力优惠券，再到国际站 `Redeem Credits` 兑换，供国际站实例使用。
- 国际站页面明确提示可从 AMD Developer China 获取优惠券并在此兑换。国际站账号余额从5增至15 GPU-hours，用户确认通过上述流程获得额度。
- [官方开发者计划文档](https://amd-aim.github.io/radeon-cloud-docs/zh-cn/guides/developer-program/)目前仍描述中国站兑换，并称全球站不使用该计划，与国际站页面及本次实际操作存在差异；以当前页面和实际兑换结果为本次记录依据。

### 连接与开发环境

当前实例通过公钥 SSH 直连；启用 TCP 转发后已验证 VS Code Remote-SSH。连接、环境、Codex 代理和 RCC 命令见 [AMD 远程 SSH](README.md)。

### OCI 自动探测（国际站）

**2026-11-01：网页资源池只读监测。** `radeon-monitor.timer` 北京时间每小时 00、30 分运行，持续至 2026-12-01 14:45（不含），只读取国际站网页的 `/api/pools/status`，不创建实例。记录位于 OCI `/data/lcx/codex/radeon-monitor/observations.jsonl`；目标为 `one-click / resource-16c55g1u`，状态仅表示该次快照，不能保证之后创建成功。`unknown` 和 `error` 不表示资源池已满。

程序优先使用权限为 0600 的 `session.cookies`。缺少会话或接口返回 401 时，调用 OCI Codex 的 Gmail 连接读取新验证码并重新登录；重新登录失败后至少等待一小时再尝试。每次登录后的 Cookie 先通过接口验证，再保存。2026-11-01 实测初次登录和后续 Cookie 复用都成功；账号权限、平台接口、Codex/Gmail 连接或限流变化仍可能导致续期失败。运行结果与原因在上述 JSONL 和 `journalctl -u radeon-monitor.service` 中查看，不将验证码或 Cookie 输出到服务日志。

代码见 [radeon_monitor.py](radeon_monitor.py) 和 [radeon_session_login.py](radeon_session_login.py)，服务单元见 [radeon-monitor.service](radeon-monitor.service) 与 [radeon-monitor.timer](radeon-monitor.timer)。

**2026-10-31：已停止自动探测和 Launch。** OCI 上 `rcc-probe.timer` 与 `rcc-probe-cleanup.timer` 均已停用并停止，对应服务均为 inactive，`/opt/rcc-probe/PAUSE` 标记已创建。停止清理定时器前，RCC 实例状态查询返回 `not_found`；本地状态文件仍保留旧的 `held_instance` 记录，之后如需恢复，应先核对实例和状态文件。

2026-10-26部署。代码见 [rcc-probe.py](rcc-probe.py)；OCI安装目录 `/opt/rcc-probe`，独立venv安装 `amd-oneclick-sdk==1.0.2`，系统服务以opc用户运行。认证文件仅保存在服务器 `auth/config.json`（0600），不入库，不复制SSH私钥。仓库内移动脚本不改变 OCI 上已经部署的副本。

2026-10-29 更新：实例存在期间定期读取 RCC 日志与状态，只记录预设的故障类别及日志采集次数，不保存原始日志；未匹配不等于没有故障。报告按是否请求持久化分组，历史样本混合时段且两组数量不均，不能直接推断因果。一次验证中，请求和不请求持久化各一次，均在约34–36秒后变为 not_found；两轮各有3次日志采集成功，均未匹配已知类别，余额读数均为63 GPU-hours。实际持久化是否可用仍以实例返回值为准。

#### 2026-10-28 更新：每天早晨申请并保留

> 2026-10-30 起的新规则：北京时间 04:00–08:40 每 20 分钟申请一次。每轮先请求持久化；失败且实例确认已释放后，再尝试非持久化。04:00–04:59 达到 ready 后立即清理；05:00–08:59 达到 ready 后保留。创建结果不明确、清理未确认或达到预算阈值时不回退。两种请求均记录 `persistent_requested`；持久化实际是否可用仍需核查。10 月 29 日维持原时间表。

- 每天北京时间 05:30 开始每 20 分钟申请一次，最后一次为 08:50，09:00 后不申请；不再使用原三天截止时间。达到 ready 后保留，清理任务不会删除，其他定时申请跳过已有实例。
- 请求启用 SSH 和持久化（`/opt/rcc-probe/REQUEST_PERSISTENT` 已启用）。持久化是否实际挂载须核查，创建结果不明确时暂停，不自动尝试无持久化，避免重复创建。
- 工作结束并保存产物后，在本机运行 `rcc instance delete --yes`，再运行 `rcc instance status` 确认释放；下次探测会确认实例不存在并自动清除保留标记。保留期间的余额变化不计入短时探测预算。
- 每日探测余额下降达到 0.25 GPU-hour、余额不足、已有实例或创建结果不明时停止后续申请；每日预算基线在下一天的首次探测重置。网络故障可能影响状态与清理，日志中出现 `error` 或 `cleanup_pending` 时需核查。

#### 2026-10-28 原保留实例计划（历史）

- 北京时间10月28日06:00–08:00（不含08:00），定时申请首次达到 ready 时保留实例，供持久化和 VS Code SSH 研究；窗口内请求启用 SSH。后续已验证 SSH 与 VS Code Remote-SSH，配置见 [AMD 远程 SSH](README.md)。
- 成功后脚本和清理任务均跳过该实例，不再提交探测；实例可能持续扣除 GPU-hours，需用户使用完后手动释放。若窗口内未成功，普通探测按原计划继续。
- 原有3天期限、0.25 GPU-hour 观测下降停止阈值、已有实例和认证异常保护仍生效，可能使窗口内不发起申请；余额阈值不是硬扣费上限。
- 本机检查：`rcc instance status`；就绪后 `rcc instance ssh --print` 查看连接方式。结束实验并确认产物保存后，用 `rcc instance delete --yes` 释放，随后 `rcc instance status` 核实实例消失。保留标记不会自动清除，释放后如需恢复探测需先检查 OCI 状态并人工处理。

#### 原三天探测策略（历史）

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
