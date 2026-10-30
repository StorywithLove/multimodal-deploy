# GPU 实例横向对比

以下记录截至 2026-10-28 的使用情况；实例规格、额度和网络条件以平台页面及实际测试为准。

| 对比项 | AMD Radeon Cloud 国际站 | NVIDIA：ModelScope Notebook（A10） | 国产 GPU：模力方舟 / 沐曦 |
| --- | --- | --- | --- |
| 资源与实践 | 用户提供约 330 开发者积分、每天最多兑换 20 小时；积分是待兑换额度。 | 用户提供普通 GPU 约 220 小时时长；曾使用 A10，具体实例以平台页面为准。 | 已进行文本、图像、语音、OCR 和向量检索实践；具体型号和资源以实例为准。 |
| SSH | 本机使用 `radeon-gpu` 别名和公钥直连；实例重建后地址、端口可能变化。 | 使用 Tailscale 私有网络连接，已验证公钥登录及 SCP；当前使用 DERP 中继。 | SSH 连接方式尚未在本项目中核实。 |
| VS Code | 实例启用 SSH TCP 端口转发后，已验证 Remote-SSH。 | 已验证通过 Tailscale 使用 Remote-SSH，重启后曾成功恢复连接。 | 尚无本项目内的 Remote-SSH 验证记录。 |
| 远程 Codex 使用 | 经 SSH 反向转发本机代理，CLI 登录和请求成功；侧边栏代理未验证。 | CLI 可用；侧边栏模块加载异常。 | 尚未部署验证，安装难度未知。 |
| 网络与模型下载 | 当前实例直接访问多个公网网站超时；平台出口代理有访问限制。Codex CLI 登录及请求曾通过本机代理转发成功。 | 按用户使用经验可快速下载 ModelScope 模型；本次没有量化速度数据。 | 暂无可比较的网络测试记录。 |
| 详细记录 | [AMD 远程 SSH 与实例运维](platforms/amd_developer/remote-ssh/README.md) | [ModelScope 远程 SSH 与实例记录](platforms/modelscope/remote-ssh/README.md) | [模力方舟实例记录](platforms/moark/remote-ssh/README.md) |

AMD 网络测试只代表当前实例，ModelScope 的下载速度尚未量化。ModelScope 上的 AMD MI300X 训练记录见 [Gemma 4 微调](platforms/amd_developer/notebook_ROCm/257-gemma4-e4b-emotion-lora.md)；它与上表的 NVIDIA A10 是不同实例。

[腾讯云 Cloud Studio](https://cloudstudio.net/) 每日签到可领 2 机时，可与 ModelScope 每天的 8 小时配合使用；机时按实例规格折算，2 机时约可运行 A10 36 分钟，具体额度和有效期以平台页面为准。
