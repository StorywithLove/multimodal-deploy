# ModelScope NVIDIA 实例：Tailscale 与 SSH

本文记录本机 Windows 连接 ModelScope 临时 GPU 容器的配置与重启恢复。当前 A10 实例已验证公钥 SSH 登录和 SCP 小文件上传及 SHA-256 一致；Tailscale 使用 DERP 中继，尚未建立直连。SFTP 和 SSH 端口转发尚未完成端到端验证。

## 1. 前期 Tailscale 部署

参考 [ModelScope 社区教程](https://www.modelscope.cn/posts/435314)和 [Tailscale 用户态网络文档](https://tailscale.com/docs/concepts/userspace-networking)。方案是本机与实例加入同一 tailnet，实例以用户态运行 Tailscale Serve，将私有端口 2222 转发到仅监听 `127.0.0.1:2223` 的独立 OpenSSH 服务。该入口不是 ModelScope 官方 SSH 接口，不修改实例原有 22 端口。

- 本机 Tailscale：`D:\Programs\Tailscale`，使用与实例相同的 Tailscale 账号登录。
- 本机 SSH 专用私钥：`%USERPROFILE%\.ssh\id_ed25519_modelscope`；其公钥配置于远端 `authorized_key.pub`。
- 当前实例：Ubuntu 22.04 x86_64，Tailscale 1.104.2 官方静态包已校验；没有 `/dev/net/tun`，因此使用用户态模式。
- 服务、状态、独立 SSH 配置和主机密钥现位于 `/mnt/workspace/lcx/modelscope-remote-ssh`。这些状态文件含敏感凭据，不要提交到 Git；旧的 `/root/tmp/modelscope-remote-ssh` 留作回退。
- 2026-10-27 从临时目录切换到持久目录后重新 SSH 登录成功；不代表实例重建后的恢复已验证。
- 2026-10-27 用户重启并重新进入实例后，Tailscale 与 SSH 服务已运行；本机通过 `modelscope-a10` 公钥登录并识别 NVIDIA A10，用户确认 VS Code Remote-SSH 连接成功。新实例复用了持久化身份和原 Tailscale IP；仅代表本次重启验证结果，不代表服务已自动启动。
- VS Code Server 安装根目录配置为 `/mnt/workspace/lcx/vscode-server`；更换为 CPU 实例后，Server 从该持久目录启动，启动日志未见重新下载。VS Code 更新或平台存储变化时可能仍需重新下载或配置。

本机 PowerShell 登录：

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_modelscope" `
  -o "UserKnownHostsFile=$env:USERPROFILE\.ssh\known_hosts_modelscope_test" `
  -p 2222 root@100.100.2.61
```

这个地址是本次设备的 Tailscale 私有地址，重建后可能改变；以 Tailscale status 或管理页面为准。首次连接或主机密钥改变时，先在 WebIDE 用 `ssh-keygen -lf /mnt/workspace/lcx/modelscope-remote-ssh/ssh_host_ed25519_key.pub` 独立核对指纹，不要盲目接受变更。

本次网络检查：Windows 和实例均报告 UDP 可用，双方 NAT 映射会随目标变化，直连未建立；先前经旧金山、后来经香港 DERP。Windows 控制服务检测到本地 `127.0.0.1:7890` 代理，不能据此断定代理是中继的原因。可在两端用 `tailscale netcheck`，在本机用 `tailscale ping <实例Tailscale IP>` 对比，连接方式以实际输出为准。

## 2. 实例重启后的重新建立

以下步骤假设 `/mnt/workspace/lcx/modelscope-remote-ssh` 在新实例仍存在且文件完整。重启前先备份训练 adapter 和 tokenizer；`/mnt/workspace` 的实际保留规则与账号配额以平台为准。不要在旧实例仍运行时同时启动同一份 Tailscale 身份。

1. 在 ModelScope 网页打开新实例的 WebIDE 终端，确认目录、程序及配置存在，且旧进程未运行：

```bash
cd /mnt/workspace/lcx/modelscope-remote-ssh
ls -ld .
ls -l tailscale_1.104.2_amd64/tailscale tailscale_1.104.2_amd64/tailscaled sshd_config
pgrep -af tailscaled || true
ss -lnt | grep ':2223' || true
```

如果已有相关进程或端口被占用，先确认归属，不要重复启动或停止其他服务。确认未运行后，在 WebIDE 终端执行：

```bash
cd /mnt/workspace/lcx/modelscope-remote-ssh
umask 077
nohup ./tailscale_1.104.2_amd64/tailscaled \
  --tun=userspace-networking \
  --socket="$PWD/tailscaled.sock" \
  --state="$PWD/tailscaled.state" \
  > "$PWD/tailscaled.log" 2>&1 < /dev/null &
sleep 3
./tailscale_1.104.2_amd64/tailscale --socket="$PWD/tailscaled.sock" status
```

如果状态显示 `Logged out`，运行 `./tailscale_1.104.2_amd64/tailscale --socket="$PWD/tailscaled.sock" up`，在浏览器完成登录和设备批准；不要公开登录链接。若权限或网络出错，查看 `tailscaled.log`，其中可能含敏感信息。

2. 启动独立 sshd 并配置 tailnet 转发：

```bash
mkdir -p /run/sshd
/usr/sbin/sshd -t -f "$PWD/sshd_config" && \
  /usr/sbin/sshd -f "$PWD/sshd_config" -E "$PWD/sshd.log"
ss -lnt | grep ':2223'
./tailscale_1.104.2_amd64/tailscale --socket="$PWD/tailscaled.sock" \
  serve --bg --tcp=2222 tcp://127.0.0.1:2223
./tailscale_1.104.2_amd64/tailscale --socket="$PWD/tailscaled.sock" serve status
./tailscale_1.104.2_amd64/tailscale --socket="$PWD/tailscaled.sock" ip -4
ssh-keygen -lf ssh_host_ed25519_key.pub
```

如需 Serve 批准，按终端提示操作。确认监听为 `127.0.0.1:2223`、Serve 显示 tailnet only，并在本机使用第 1 节命令登录；如果 IP 改变，替换命令中的地址并核对指纹。容器重启会终止进程，当前未配置自动启动。

如果持久目录不存在或文件不完整，重新按第 1 节方案配置并重新核对主机指纹；不可假设旧状态或旧 IP 一定可复用。

本机已经配置 SSH 别名 `modelscope-a10`，VS Code Remote-SSH 可选择该名称；新实例的 Tailscale IP 若变化，需要更新本机 SSH config。SSH 保活不能防止网络中断，长任务应使用 `tmux` 等会话工具并保存检查点。重新分配的实例镜像和 SSH 组件也可能变化；恢复过程应以实际检查结果为准。

## NVIDIA GPU：ModelScope DSW

- 用户所见免费 GPU 环境为24GB显存，曾计划使用 A10；实际型号与时长以实例页面为准。历史 A10 模型候选清单已移除。
- 尚无该候选清单对应的 A10 微调完成记录。

### llama.cpp 网关记录与 SSH

[项目 llama.cpp 记录](../../../llm/qwen3.8-flash-next/llama.cpp.md)中服务绑定实例内 `127.0.0.1:8000`，经 `<workspace-id>-proxy-8000.dsw-gateway-...` 登录态 HTTPS 网关访问。[Notion 大模型笔记](https://app.notion.com/p/3fa12e6731688158a60ecb9ebba3e8b0)也记录了8000端口和网关关系。这验证的是 HTTP 服务，不证明 SSH 转发；域名解析的网关 IP 不能直接当作实例公网 IP。

历史 ModelScope 实例曾检测到 `/etc/dsw/runtime/bin/sshd -p 22 -D`，并收到 `SSH-2.0-OpenSSH_10.0`；公钥认证开启，root密码登录受限。当前实例仍需单独验证，实例内部有 sshd 不证明本机可连接。

2026-10-27 A10 实例内部的系统 sshd 监听 22，root 密码登录禁用；该监听本身不证明外部可达。随后使用 Tailscale 用户态网络和独立的回环 sshd 建立了私有 SSH：Windows 公钥登录、SCP 小文件上传及 SHA-256 核对成功；服务切换至 `/mnt/workspace/lcx/modelscope-remote-ssh`。重启后恢复服务，新实例复用原 Tailscale IP，Windows SSH 和用户的 VS Code Remote-SSH 均连接成功。当前经过 DERP 中继，SSH 端口转发尚未端到端验证。部署、登录和重启恢复步骤见 [ModelScope 远程 SSH](README.md)。

同日 ProxyClient 实测：以网页路径中的工作区编号作为实例 ID 时返回 `Entity does not exist`；改用登录回调中的 `instanceId` 后，返回 `Get proxy url failed`，连接仍未建立。未取得底层错误，不能据此断言是账号权限、实例归属、地域或网络中的哪一项；需核对 PAI 控制台实例归属和地域。

### 其他连接方式与 VS Code

1. 已验证 Tailscale SSH 与 VS Code Remote - SSH；使用时打开远端项目目录并选择远端 Python／Jupyter 内核。
2. 若免费实例可在本人阿里云 PAI 控制台查看并授权，官方 [DSW ProxyClient](https://www.alibabacloud.com/help/en/pai/connect-to-a-dsw-instance-over-ssh) 可通过阿里云身份及 SSH 公钥连接，无需 NAT/EIP；它要求实例 ID、地域和有权访问的云账号凭据。**ModelScope 平台免费实例是否提供这类权限尚未验证。**
3. 若平台提供直连地址，可按 [DSW 直连文档](https://www.alibabacloud.com/help/en/pai/dsw-direct-connection)使用公钥与指定端口。阿里云公网直连的网络组件可能单独计费；不自行创建付费资源。
4. 连接后在 VS Code Remote - SSH 打开 `/mnt/workspace`，选择远端 Python／Jupyter 内核；Remote - SSH 编辑远端文件，不自动同步本地项目，代码可用 Git 管理。

本机已检测到 OpenSSH 和 VS Code；`proxyclient.exe` 未在 PATH 中，默认 ProxyClient 配置文件不存在。ProxyClient 与平台直连路线均未完成端到端验证；WebIDE 可作为恢复入口。

[ModelScope 开源教程](https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-04.md)说明 `/mnt/workspace` 用于持久保存；其他路径不应假定跨实例保留。教程示例额度与用户页面不同，具体以当前账号为准。


## AMD MI300X：ModelScope DSW

- 已使用 MI300X 192GB 完成 Gemma 4 情绪分类 LoRA 训练；环境与结果见 [微调记录](../../amd_developer/notebook_ROCm/257-gemma4-e4b-emotion-lora.md)。
