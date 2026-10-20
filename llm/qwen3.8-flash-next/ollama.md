# Qwen3.8-Flash-Next / Ollama / GGUF

- 硬件目标：AMD Instinct MI300X 192GB，ROCm 7.2.3，`gfx942`
- 模型：`unsloth/Qwen3.8-Flash-Next-GGUF`
- 量化：`UD-Q4_K_XL`，四个 GGUF 分片，约 104 GiB
- 对照：与 [llama.cpp / GGUF](llama.cpp.md) 使用同一量化文件；另一组为 [vLLM / FP8](vllm.md) 与 [SGLang / FP8](sglang.md)
- 结果目录：`/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next`
- Ollama 模型目录：`/root/tmp/qwen3.8-flash-next/ollama-models`
- 状态：已安装 Ollama CLI（版本 `0.40.6`），并确认 `/usr/local/lib/ollama/rocm_v7_2` 存在；2026-10-20 已将模型重新导入 `/root/tmp/qwen3.8-flash-next/ollama-models`，CLI/GPU 冒烟通过，E1 单并发基线、E2 并发矩阵、E3 长上下文以及 E4 的 daemon API、首次模型请求和 Mini-8 质量抽样均完成。

本页沿用 [SGLang 部署记录](sglang.md)的流程结构，并复用仓库里的 OpenAI-compatible benchmark 客户端。Ollama 导入 GGUF 时不会重新量化；对于多分片 GGUF，Modelfile 的 `FROM` 应匹配全部原始分片。[Ollama GGUF 导入说明](https://github.com/ollama/ollama/blob/main/docs/import.mdx)、[Modelfile 说明](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx)

> **导入兼容性闸门：** 2026-08-31 的报告记录了这款 Unsloth `UD-Q4_K_XL` 在 Ollama 0.33.2 导入时，所有分片复制完成后仍在 GGUF 校验阶段失败；当时维护者表示 Qwen3.8-Flash-Next 尚未受支持。[问题记录](https://github.com/ollama/ollama/issues/18146) 因此先记下 `ollama --version`，并为 Ollama 使用独立模型存储目录。若 `ollama create` 再次报 `failed to validate GGUF ... without compatibility patches`，保留版本和完整日志，不用同一版本反复导入，也不要开始性能测评。该历史报告不代表当前版本或 AMD 上必然失败。

## 模块 A：模型部署 / Deployment

### 0. 公共变量与结果目录

**存储规则：不要将模型权重、GGUF 分片、Ollama blobs 或权重缓存写入 ModelScope Notebook 的 `/mnt/workspace` 持久化目录。** ModelScope 文档说明该挂载为 100GB 持久化空间；本项目将原始 GGUF 和 Ollama 导入副本都放在 `/root/tmp`。导入会产生约 104 GiB 的第二份权重，因此开始前检查 `/root/tmp` 至少有 120 GiB 可用空间；否则先停止，不要开始 `ollama create`。`/mnt/workspace` 仅保存代码、配置、日志和结果。[ModelScope Notebook 存储说明](https://github.com/modelscope/ms-cookbook/blob/main/content/chapters/chapter-04.md)

```bash
set +e
set -uo pipefail
cd /mnt/workspace/multimodal-deploy

export WORK=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$WORK/Qwen3.8-Flash-Next-GGUF"
export OLLAMA_MODELS="$WORK/ollama-models"
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export OLLAMA_TAG=qwen3.8-flash-next-ollama-ud-q4-k-xl
export OLLAMA_HOST=127.0.0.1:11435
export SERVER_LOG="$RESULT_DIR/qwen3.8-flash-next-ollama-server.log"

mkdir -p "$MODEL_DIR" "$OLLAMA_MODELS" "$RESULT_DIR"
df -h /root/tmp /mnt/workspace
du -sh "$MODEL_DIR" "$OLLAMA_MODELS" 2>/dev/null || true
quota -s 2>/dev/null || true
```

Ollama 在本页使用 `127.0.0.1:11435`，这样可以与安装器可能启动的默认 `127.0.0.1:11434` 服务并存。服务只监听回环地址，不绑定公网地址。`llama.cpp.md` 里实际使用过的 `*-proxy-8000.dsw-gateway-...` 是 DSW 的 HTTP(S) 登录态服务网关；它可以把 Web 服务交给已登录的浏览器访问，但这个 HTTPS 链接本身不是 SSH 地址。若要在本地 VS Code 用 Remote-SSH 连接 DSW，应通过 DSW ProxyClient 或已配置的 SSH 入口；连接后再将远程 11435 转发到本机回环地址。[DSW ProxyClient 远程连接说明](https://help.aliyun.com/zh/pai/connect-to-a-dsw-instance-over-ssh)、[VS Code SSH 端口转发说明](https://code.visualstudio.com/docs/remote/ssh)。

### 1. 检查 AMD / ROCm 基础环境

```bash
cat /opt/rocm/.info/version 2>/dev/null || true
rocm-smi
rocm-smi --version 2>/dev/null || true
rocm-smi --showmeminfo vram --json
rocminfo 2>/dev/null | grep -m 5 -E 'Name:.*gfx|gfx942' || true
ls -l /dev/kfd /dev/dri/renderD* 2>/dev/null
id
df -h / /root/tmp /mnt/workspace
free -h
```

Ollama 的 Linux AMD GPU 包包含 ROCm 用户态运行库，主机仍需兼容的 ROCm 7 驱动。安装包与 GPU 要求见 [Ollama Linux 安装说明](https://github.com/ollama/ollama/blob/main/docs/linux.mdx) 和 [GPU 故障排查](https://github.com/ollama/ollama/blob/main/docs/troubleshooting.mdx)。

**基于本次提供的输出：**`rocm-smi` 看到了 Device 0（NPS1），`rocminfo` 看到了 `gfx942`，且 `/dev/kfd` 与 `/dev/dri/renderD128`–`renderD135` 存在，硬件/ROCm 枚举正常。`free -h` 的可用内存约 198 GiB；此前 `/root/tmp` 所在盘可用 469 GiB。ModelScope 持久化挂载的 `df` 曾显示池化网络容量，但该值不能代表个人配额；用户本次确认清理 Ollama 副本后 `/mnt/workspace` 目录占用为 7.0 GB。模型文件统一放 `/root/tmp`，导入前重新检查可用空间。简要 `rocm-smi` 输出没有给 VRAM 总量，部署前用上面的 `--showmeminfo vram --json` 再确认。

需要留意：提供的设备节点权限为 `crw------- root root`（0600）。同一个 root 身份启动的前台 `ollama serve` 可以访问；安装脚本创建的非 root `ollama` systemd 服务则可能无法打开设备。先用 `id` 确认当前终端身份，并让 Ollama 服务进程身份与设备访问权限匹配；本实例当前展示的设备节点没有给组权限，所以仅把服务用户加入 `render` 组未必足够。Ollama/ROCm 的常见配置是通过 `render` 组或 udev 规则授予 GPU 设备访问权；本部署先沿用当前 root 终端运行前台服务，若切换成 systemd 服务再单独处理权限。[Ollama AMD GPU 说明](https://github.com/ollama/ollama/blob/main/docs/gpu.mdx)、[AMD ROCm 设备权限说明](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.3.3/install/prerequisites.html#configuring-permissions-for-gpu-access)

### 2. 安装 Ollama 与 AMD ROCm 运行库

`.tar.zst` 需要 `zstd`。先安装解压工具，再运行官方安装脚本；脚本检测到 AMD GPU 时会尝试一并安装 ROCm 附加包。[Linux 安装说明](https://github.com/ollama/ollama/blob/main/docs/linux.mdx)、[安装脚本的 GPU 检测逻辑](https://github.com/ollama/ollama/blob/main/scripts/install.sh)

这里的 `/usr/local` 是 Ollama **程序与运行库**的安装前缀，例如 `/usr/local/bin/ollama` 和 `/usr/local/lib/ollama`；它不是已有的“系统自带 Ollama”目录，也不是模型权重目录。安装脚本根据当前 `PATH` 选择安装位置，本次输出明确显示为 `/usr/local`。模型文件统一放在 `/root/tmp`：原始 GGUF 位于 `/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-GGUF`；导入后 Ollama 管理的模型 blob 由 `OLLAMA_MODELS` 指向 `/root/tmp/qwen3.8-flash-next/ollama-models`。ModelScope 持久化目录 `/mnt/workspace` 只保存项目文件与结果，不保存模型权重。

```bash
sudo apt-get update
sudo apt-get install -y zstd
zstd --version

curl -fsSL https://ollama.com/install.sh | sh

ollama --version | tee "$RESULT_DIR/ollama-version.txt"
command -v ollama
find /usr/local/lib/ollama -maxdepth 1 -type d -name 'rocm*' -print
```

如果安装脚本报 `curl: (16) Error in the HTTP2 framing layer`，随后出现 `gzip: stdin: unexpected end of file`，说明下载包传输中断，gzip/tar 错误和 `ollama: command not found` 都是后续结果。此时可用 curl 强制 HTTP/1.1，将两个官方包先下载到临时目录、校验后再解压；本机输出的安装前缀是 `/usr/local`。脚本日志若显示下载 `.tgz`，恢复函数会使用 `.tgz`；若 `.tar.zst` 可用则优先使用它。此恢复流程用于安装脚本未完成时，不要和成功安装后的手动补包流程重复执行：

若下载速度很低，可在运行恢复函数**之前**，在同一终端设置 curl 代理。例如代理运行在 DSW 实例本机的 7890 端口时：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

若代理不在 DSW 实例本机，将 `127.0.0.1` 换成 DSW 能访问到的代理主机地址。代理只影响包下载；若要给当前正在运行的 curl 换代理，需先 `Ctrl+C` 中断再重新运行函数。该下载命令不启用续传，因此中断后会从头下载。

```bash
export OLLAMA_INSTALL_TMP=/root/tmp/ollama-install
mkdir -p "$OLLAMA_INSTALL_TMP"
set -o pipefail

download_extract_ollama_bundle() {
  local bundle_name="$1"
  local base_url="https://ollama.com/download/$bundle_name"
  if curl --http1.1 -fsSI -L "$base_url.tar.zst" >/dev/null 2>&1; then
    local archive="$OLLAMA_INSTALL_TMP/$bundle_name.tar.zst"
    if ! curl --http1.1 --retry 5 --retry-all-errors -fL -o "$archive" "$base_url.tar.zst"; then
      echo "[ERROR] 下载 $bundle_name.tar.zst 失败。"
      return 1
    fi
    if ! zstd -t "$archive"; then
      echo "[ERROR] $archive 未通过 zstd 完整性检查。"
      return 1
    fi
    if ! zstd -dc "$archive" | sudo tar -xf - -C /usr/local; then
      echo "[ERROR] 解压 $bundle_name.tar.zst 失败。"
      return 1
    fi
  else
    local archive="$OLLAMA_INSTALL_TMP/$bundle_name.tgz"
    if ! curl --http1.1 --retry 5 --retry-all-errors -fL -o "$archive" "$base_url.tgz"; then
      echo "[ERROR] 下载 $bundle_name.tgz 失败。"
      return 1
    fi
    if ! gzip -t "$archive"; then
      echo "[ERROR] $archive 未通过 gzip 完整性检查。"
      return 1
    fi
    if ! sudo tar -xzf "$archive" -C /usr/local; then
      echo "[ERROR] 解压 $bundle_name.tgz 失败。"
      return 1
    fi
  fi
}

if download_extract_ollama_bundle ollama-linux-amd64; then
  if download_extract_ollama_bundle ollama-linux-amd64-rocm; then
    /usr/local/bin/ollama --version | tee "$RESULT_DIR/ollama-version.txt"
    command -v ollama
    find /usr/local/lib/ollama -maxdepth 1 -type d -name 'rocm*' -print
  else
    echo '[ERROR] Ollama 核心包已解压，但 ROCm 包失败；先修复下载后再启动服务。'
  fi
else
  echo '[ERROR] Ollama 核心包失败；先修复下载后再继续。'
fi
```

#### 仅在 ROCm 目录缺失时手动补装

下面这段命令只处理 **ROCm 附加运行库**：`curl` 下载 ROCm 包，`zstd -d` 解压，`tar` 将文件写入 `/usr/local`，最后的 `find` 检查运行库目录是否出现。它不会安装 Ollama 核心程序、启动 Ollama 服务，也不会导入 GGUF。

**执行条件：**只有前面的 `find /usr/local/lib/ollama ...` 没有任何输出时，才执行此补装命令。若输出了 `/usr/local/lib/ollama/rocm_v7_2`，说明 ROCm 包已安装，直接跳过。上面的恢复函数本身会依次下载核心包和 ROCm 包；恢复函数成功且找到 `rocm*` 目录后，不要再运行下面这段，以免重复下载和解压。

目标目录必须与 Ollama 安装前缀一致；当前实例的安装输出是 `/usr/local`：

```bash
curl -fsSL https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst \
  | zstd -d | sudo tar -xf - -C /usr/local
find /usr/local/lib/ollama -maxdepth 1 -type d -name 'rocm*' -print
```

**本次用户提供的结果：**`ollama --version` 已输出客户端版本 `0.40.6`，`command -v` 指向 `/usr/local/bin/ollama`，并找到 `/usr/local/lib/ollama/rocm_v7_2`。因此核心程序和 ROCm 附加包都已就位；`Warning: could not connect to a running Ollama instance` 只表示运行版本命令时尚无可连接的服务，不代表安装失败。无需再运行安装或 ROCm 补装命令，继续执行第 0 步变量设置，然后启动第 3 步的服务。

安装脚本可能创建 systemd 服务。此部署固定用 `127.0.0.1:11435` 的前台 `ollama serve` 进程；若新开终端，重新导出服务所需环境变量。不要把 SGLang/vLLM venv 的 `LD_LIBRARY_PATH` 带入 Ollama 进程，以免混用两套 ROCm 用户态库。

### 3. 启动 Ollama 服务

如果终端 A 里旧的 Ollama 服务仍在运行，先按 `Ctrl+C` 停止它；`OLLAMA_MODELS` 只在服务启动时读取。然后在**终端 A**用新目录启动服务。`OLLAMA_MODELS` 指向 `/root/tmp`；E1 固定 10240 context 容量和单并发。10240 为服务容量，给 8192 目标输入、256 输出 token 和模板余量留空间。Ollama 的 OpenAI-compatible API 不提供逐请求的 context 参数，模型的 context 通过 Modelfile 设置。[Ollama OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx)

```bash
cd /mnt/workspace/multimodal-deploy
export WORK=/root/tmp/qwen3.8-flash-next
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_CONTEXT_LENGTH=10240
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_DEBUG=1
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export SERVER_LOG="$RESULT_DIR/qwen3.8-flash-next-ollama-server.log"
mkdir -p "$OLLAMA_MODELS" "$RESULT_DIR"
df -h "$OLLAMA_MODELS" "$RESULT_DIR"

ollama serve 2>&1 | tee -a "$SERVER_LOG"
```

**正常启动时终端 A 会保持运行，不会返回 shell 提示符。**日志出现 `Listening on 127.0.0.1:11435` 表示服务已开始监听；出现 `inference compute ... library=ROCm compute=gfx942 ... total="191.7 GiB"` 表示 Ollama 已选中 ROCm 并识别到 MI300X。启动探测中如果先出现 `libcuda.so.1: cannot open shared object file`，但随后报告 `library=ROCm`，这是对本机不适用的 CUDA 后端探测失败，不影响已成功选择 ROCm。首次启动生成 `/root/.ollama/id_ed25519` 也是正常的 Ollama 身份密钥初始化。

日志里的 `vram-based default context ... default_num_ctx=262144` 是根据 191.7 GiB 显存计算出的自动默认值记录，不表示此刻已加载模型或分配了该长度的 KV cache。实际配置中已收到 `OLLAMA_CONTEXT_LENGTH=10240`；第 5 步的 E1 Modelfile 也会显式设 `num_ctx 10240`。模型加载后可用 `ollama ps` 查看实际运行上下文。[上下文长度配置说明](https://github.com/ollama/ollama/blob/main/docs/faq.mdx)

保持终端 A 运行。另开**终端 B**，设置相同的 `OLLAMA_HOST` 并验证服务；看到 `Ollama is running` 后再继续。若 `/root/tmp` 下已有 4 个 GGUF 分片，跳过第 4 步下载部分，直接复核分片后执行第 5 步重新导入；仅当原始分片缺失时才重新下载。模型创建和 CLI/API 请求都会连接此地址。

```bash
export OLLAMA_HOST=127.0.0.1:11435
curl -fsS http://127.0.0.1:11435/
```

### 3.1 可选：在本地 VS Code 访问 Ollama

如果本地 VS Code 已通过 DSW ProxyClient 配好 Remote-SSH：在 VS Code 的 **Ports** 面板选择 **Forward a Port**，填写远端端口 `11435`，然后从本机访问 VS Code 显示的 localhost 地址（例如 `http://127.0.0.1:11435`）。也可以在本机 PowerShell 的 SSH 配置已生成后运行：

```powershell
ssh -N -L 127.0.0.1:11435:127.0.0.1:11435 dsw-mi300x
```

先将 `dsw-mi300x` 替换成 ProxyClient 写入本机 SSH 配置的实际 Host 别名。隧道运行期间，本机客户端访问 `http://127.0.0.1:11435/v1/chat/completions`；DSW 实例上的 Ollama 仍只监听实例自己的 `127.0.0.1:11435`。本机 11435 若已占用，可把命令左侧本地端口改成其他空闲端口。若只需要在本机浏览器通过 DSW 登录态网关访问，应使用平台为该服务实际生成的访问链接；不要把 `proxy-8000` 链接改成 `proxy-11435` 后就假设路由已经创建。benchmark 脚本仍在实例内部跑时继续使用实例内的 `127.0.0.1:11435`。

### 4. 下载并检查 GGUF

沿用 [llama.cpp 记录](llama.cpp.md)的 ModelScope 来源与量化版本，只下载 `UD-Q4_K_XL`：

若原始 GGUF 已在 `/root/tmp/qwen3.8-flash-next/Qwen3.8-Flash-Next-GGUF` 且确认有 4 个分片，不要重复下载；只运行下面的分片检查，再继续第 5 步。

```bash
export WORK=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$WORK/Qwen3.8-Flash-Next-GGUF"
export OLLAMA_MODELS="$WORK/ollama-models"
mkdir -p "$MODEL_DIR"

modelscope download unsloth/Qwen3.8-Flash-Next-GGUF \
  --repo-type model \
  --include 'UD-Q4_K_XL/*' \
  --include '*.json' '*.md' \
  --cache-dir "$MODEL_DIR"

mapfile -t GGUF_SHARDS < <(
  find "$MODEL_DIR" -type f \
    -name 'Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf' -print | sort
)
if [ "${#GGUF_SHARDS[@]}" -eq 4 ]; then
  printf '%s\n' "${GGUF_SHARDS[@]}"
  export GGUF_DIR="$(dirname "${GGUF_SHARDS[0]}")"
  du -sh "$GGUF_DIR"
  df -h "$OLLAMA_MODELS"
else
  echo "[ERROR] 预期 4 个 GGUF 分片，实际找到 ${#GGUF_SHARDS[@]} 个；先检查下载目录。"
fi
```

实际仓库文件名形如 `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`。先确认 4 个原始分片都在同一目录；不要只把 `00001` 分片交给 Ollama。模型来源：[Unsloth GGUF 仓库](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)。

### 5. 编写 Modelfile 并导入

`GGUF_DIR` 是 shell 环境变量；如果第 4 步和本步骤在不同终端执行，它不会自动保留。下面先在当前终端重新查找 4 个原始分片，再生成 Modelfile，避免 `FROM` 被展开为空。官方 Modelfile 支持以 wildcard 匹配 split GGUF 全部分片。这里保留 GGUF 内的模型元数据与模板，不手工覆盖 Qwen 模板；先创建 10240 context 的 E1 模型：

```bash
set -o pipefail
export WORK=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$WORK/Qwen3.8-Flash-Next-GGUF"
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_TAG=qwen3.8-flash-next-ollama-ud-q4-k-xl
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
mkdir -p "$WORK" "$RESULT_DIR"
mkdir -p "$OLLAMA_MODELS"

AVAILABLE_KIB=$(df -Pk /root/tmp | awk 'NR==2 {print $4}')

mapfile -t GGUF_SHARDS < <(
  find "$MODEL_DIR" -type f \
    -name 'Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf' -print | sort
)
if [ -z "$AVAILABLE_KIB" ] || [ "$AVAILABLE_KIB" -lt $((120 * 1024 * 1024)) ]; then
  echo "[ERROR] /root/tmp 剩余空间不足 120 GiB，停止导入；当前约 ${AVAILABLE_KIB:-未知} KiB。"
elif [ "${#GGUF_SHARDS[@]}" -eq 4 ]; then
  export GGUF_DIR="$(dirname "${GGUF_SHARDS[0]}")"
  printf 'GGUF_DIR=%s\n' "$GGUF_DIR"
  printf '%s\n' "${GGUF_SHARDS[@]}"

  cat > "$WORK/Modelfile.ollama" <<EOF
FROM $GGUF_DIR/Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf
PARAMETER num_ctx 10240
PARAMETER temperature 0.6
EOF

  cat "$WORK/Modelfile.ollama"
  ollama create "$OLLAMA_TAG" -f "$WORK/Modelfile.ollama" 2>&1 \
    | tee "$RESULT_DIR/qwen3.8-flash-next-ollama-create.log"
else
  echo "[ERROR] 预期 4 个 GGUF 分片，实际找到 ${#GGUF_SHARDS[@]} 个；检查下载目录 $MODEL_DIR，未创建模型。" >&2
  printf '%s\n' "${GGUF_SHARDS[@]}"
fi
```

若当前 Ollama 版本要求 split shards 显式列出，可按官方说明把 `FROM` wildcard 改为 4 行 `FROM /完整路径/分片名.gguf`，每行对应一个原始分片；不要合并或重命名分片后再导入。[Split GGUF 导入格式](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx#build-from-a-gguf-model)

`ollama create` 的 `gathering model components` / `copying file sha256:...` 阶段是在把原始 GGUF 分片写入 `OLLAMA_MODELS`，不是重新量化，也还没有开始推理。每个百分比对应正在复制的文件；前一个文件显示 `100%` 后，进度会转到下一个文件。终端或 `tee` 记录的 spinner 重绘可能让同一进度行重复出现。模型约 104 GiB；如果想确认 `/root/tmp` 下的目标目录仍在增长，可另开终端查看：

```bash
watch -n 10 'du -sh /root/tmp/qwen3.8-flash-next/ollama-models; df -h /root/tmp'
```

进度百分比或目标目录大小仍在变化时，让 `ollama create` 继续运行。若长时间完全不变，再检查进程和存储状态；不要因为 spinner 周期性重绘就中断导入。

**失败处理：** 若创建失败，记下完整的 Ollama 版本、错误阶段和日志。`ollama create` 可能已将分片 blob 写入独立的 `OLLAMA_MODELS` 目录；不要在共享模型库中手工删除 blob。先确认该隔离目录中没有成功创建的其他模型，再决定是否清理或换新版重试。

### 6. 检查模型、API 与 AMD GPU

```bash
export OLLAMA_HOST=127.0.0.1:11435
export WORK=/root/tmp/qwen3.8-flash-next
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_TAG=qwen3.8-flash-next-ollama-ud-q4-k-xl
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next

ollama list | tee "$RESULT_DIR/ollama-list.txt"
ollama show "$OLLAMA_TAG" | tee "$RESULT_DIR/ollama-show.txt"
ollama show --modelfile "$OLLAMA_TAG" \
  | tee "$RESULT_DIR/ollama-show-modelfile.txt"
if grep -Fq "$OLLAMA_MODELS/blobs/" "$RESULT_DIR/ollama-show-modelfile.txt"; then
  echo "[OK] Ollama 模型 blob 位于 $OLLAMA_MODELS"
else
  echo "[ERROR] 当前模型 blob 路径不在 $OLLAMA_MODELS；不要继续 benchmark。"
fi
du -sh "$OLLAMA_MODELS"
df -h /root/tmp /mnt/workspace
curl -fsS http://127.0.0.1:11435/ | tee "$RESULT_DIR/ollama-root.txt"
curl -fsS http://127.0.0.1:11435/api/tags \
  | tee "$RESULT_DIR/ollama-tags.json"
curl -fsS http://127.0.0.1:11435/v1/models \
  | tee "$RESULT_DIR/ollama-models.json"
```

`/`、`/api/tags` 和 `/v1/models` 用于检查服务与模型名；模型加载后再看 `ollama ps`、Ollama server log 和 `rocm-smi`，确认实际走 AMD GPU，而不是仅因 API 返回成功就认定 GPU 已使用。`ollama show --modelfile` 可复核从 GGUF 读取到的模板和参数。

**此前成功导入记录（旧路径，2026-10-20）：**用户输出显示模型 tag 曾出现在 `ollama list`、`/api/tags` 和 `/v1/models`，大小约 111 GB，架构 `qwen4exp`，参数量 176.9B；该副本随后已从 `/mnt/workspace` 清理，并按本页流程重新导入 `/root/tmp`。`ollama show` 的原生 `context length 262144` 是 GGUF 元数据；Modelfile 的运行参数 `num_ctx 10240` 是本次 E1 配置，两者含义不同。此前 `ollama show --modelfile` 显示 `TEMPLATE {{ .Prompt }}`，这是通用简化模板；本次重新导入后的 CLI 冒烟回答正常。Ollama 的量化摘要显示 `Q4_K_M`，但 `/api/tags` 同时记录父分片 `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`、总大小约 111 GB，与所选 Unsloth UD-Q4_K_XL 文件一致。[Unsloth GGUF 仓库](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)

**本次重新导入与冒烟结果（2026-10-20）：**`ollama show --modelfile` 的 `FROM` 指向 `/root/tmp/qwen3.8-flash-next/ollama-models/blobs/sha256-4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082`，路径校验通过；模型目录占用 `104G`。`/root` 文件系统为 591G 总量、216G 已用、345G 可用（39%）；`/mnt/workspace` 的 `df` 显示共享网络池容量，不能代表个人配额；此前用户执行 `du -sh /mnt/workspace` 确认目录占用 7.0G。`ollama run` 正常返回中文自我介绍；`ollama ps` 显示模型大小 82 GB、`100% GPU`、context 10240、runner `llamacpp`、驻留 `Forever`。`rocm-smi` 显示 VRAM 总量 `205822885888` B、使用 `83448569856` B（约 77.7 GiB）。随后 E1 通过 OpenAI-compatible API 完成 24/24 请求；原始数据见上方 E1 结果表和 JSON。

如果上一段的 `grep` 显示 blob 不在 `/root/tmp` 下，先停止后续操作并修正 Ollama 服务的 `OLLAMA_MODELS`，不要开始 benchmark。确认路径正确后，启动模型并检查 GPU：

```bash
ollama run "$OLLAMA_TAG" '请用中文简要说明你是什么模型。'

ollama ps | tee "$RESULT_DIR/ollama-ps-after-smoke.txt"
rocm-smi --showmeminfo vram --json \
  | tee "$RESULT_DIR/ollama-rocm-smi-after-smoke.json"
du -sh "$OLLAMA_MODELS"
df -h /root/tmp /mnt/workspace
```

再检查 OpenAI-compatible Chat Completions；该接口使用 Ollama 模型 tag：

```bash
curl -fsS http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$OLLAMA_TAG\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话说明你是什么模型。\"}],\"max_tokens\":256,\"temperature\":0}" \
  | tee "$RESULT_DIR/ollama-smoke.json"
```

### 7. 重启后检查模型存储

先在终端 A 按 `Ctrl+C`，用同样的 `OLLAMA_MODELS=/root/tmp/qwen3.8-flash-next/ollama-models`、`OLLAMA_HOST` 和 context / parallel 设置重新运行 `ollama serve`；再在终端 B 执行：

```bash
export OLLAMA_HOST=127.0.0.1:11435
ollama list
curl -fsS http://127.0.0.1:11435/api/tags
ollama run qwen3.8-flash-next-ollama-ud-q4-k-xl '只回复：模型重启后可用。'
```

这一步验证 Ollama 服务进程重启后，`/root/tmp` 下的模型 manifest/blob 仍能重新加载。`/root/tmp` 是实例本地临时空间，不提供实例重建后的持久性；若实例重建，需要重新下载或导入模型。

## 模块 B：实验测评 / Evaluation

### 共同口径

- E1–E3 复用 [统一负载配置](../../configs/benchmark/mi300x-serving-matrix.json) 和 [openai_server_matrix.py](../../benchmarks/openai_server_matrix.py)，请求 Ollama 的 `/v1/chat/completions`。
- E4 质量抽样复用 [Mini-8 题集](../../benchmarks/quality/mtbench-mini-8.json) 和 [openai_quality_probe.py](../../benchmarks/openai_quality_probe.py)。
- 本组 Ollama 与 llama.cpp 使用同一份 `UD-Q4_K_XL` GGUF；记录两边版本、context 容量、并发限制和启动参数。该实验比较的是 GGUF 模型在 Ollama 管理服务栈与 llama.cpp server 中的表现。
- vLLM/SGLang 使用 FP8 Safetensors，与本组 GGUF 的跨组结果应标为**部署路线比较**，不能写成纯框架胜负。
- Ollama `/api/tags` 的快速响应只说明 API 服务已起来，不代表模型已加载。不能把通用 `/health` ready 时间直接当作 Ollama 模型冷启动时间；模型冷加载时间应从首次请求测量并单独标注。

### E1：单请求基线

确认 Ollama 已完成一次 chat 请求、`ollama ps` 显示模型驻留、服务并发为 1 后运行。测量期间不要从 Web UI 或其他客户端发送请求。

```bash
cd /mnt/workspace/multimodal-deploy
python -u benchmarks/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --model qwen3.8-flash-next-ollama-ud-q4-k-xl \
  --label qwen3.8-flash-next-ollama-ud-q4-k-xl-e1 \
  --contexts 512,4096,8192 --concurrencies 1 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e1-raw.json
```

该客户端每格发送 8 个请求、1 个 warmup、最多生成 256 tokens，并记录返回 token 用量、延迟、吞吐与 `rocm-smi` HBM 采样。目标 context 是负载提示词目标；Ollama 模型的 `num_ctx=10240` 是服务容量，实际 prompt token 数以 JSON 结果为准。

**E1 结果（2026-10-20）：**三个档位均为 8/8 成功、HTTP 200、每请求生成 256 tokens；总计 24/24，无失败。TTFT 是每档单次探针，不代表冷加载首 token 延迟。HBM 是整卡已用量，模型在测试前已驻留。

| 目标 context | 实际 prompt tokens | completion tok/s | TTFT | P50 | P95 | HBM 峰值整卡已用 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 433 | 34.850 | 0.090 s | 6.519 s | 6.535 s | 77.756 GiB |
| 4096 | 2801 | 33.999 | 0.099 s | 6.675 s | 6.702 s | 77.822 GiB |
| 8192 | 5507 | 33.358 | 0.126 s | 6.811 s | 6.827 s | 77.796 GiB |

结果文件：[E1 原始 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e1-raw.json)。随着输入增长，输出吞吐从 34.85 降至 33.36 tok/s；本轮整卡 HBM 峰值约 77.8 GiB，测试期间增量约 0.024 GiB。该显存值包含已驻留模型的占用，不能解释为请求自身只需这些显存。

### E2：并发扩展（E1 通过后）

E2 不预设并发 2，也不直接用并发 8 跑完整矩阵。先在最长的目标上下文 8192 探测 C8；本轮 C8 上线探测已通过。探测通过后，在相同 max-parallel 配置下运行完整矩阵 C1/C2/C4/C8。若 C8 单格探测失败，再回退 C4 并在 8192 下复测。测试期间不从其他客户端发送请求。

**C8 上线探测结果：**用户确认请求正常，GPU 显存峰值约 85 GB。同步的 [ROCm SMI 快照](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e2-max-parallel-8-rocm-smi-after-probe-raw.json)记录已用 87,103,221,760 B（约 81.12 GiB），仍有约 110.57 GiB 可用。下一步保持 `OLLAMA_NUM_PARALLEL=8`，直接跑完整矩阵。

如果服务需要重新启动，终端 A 使用 `OLLAMA_NUM_PARALLEL=8`，模型目录仍放在 `/root/tmp`：

```bash
set -o pipefail
cd /mnt/workspace/multimodal-deploy
export WORK=/root/tmp/qwen3.8-flash-next
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_CONTEXT_LENGTH=10240
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_DEBUG=1
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export SERVER_LOG="$RESULT_DIR/qwen3.8-flash-next-ollama-e2-max-parallel-8-server.log"
mkdir -p "$OLLAMA_MODELS" "$RESULT_DIR"
ollama serve 2>&1 | tee -a "$SERVER_LOG"
```

终端 B 保持同一个 max-parallel 8 服务，直接运行 E2 完整矩阵：

```bash
set -euo pipefail
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_TAG=qwen3.8-flash-next-ollama-ud-q4-k-xl
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
cd /mnt/workspace/multimodal-deploy
python -u benchmarks/openai_server_matrix.py \
  --config configs/benchmark/mi300x-serving-matrix.json \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --model "$OLLAMA_TAG" \
  --label qwen3.8-flash-next-ollama-ud-q4-k-xl-e2-max-parallel-8-context512-4096-8192-concurrency1-2-4-8 \
  --contexts 512,4096,8192 --concurrencies 1,2,4,8 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e2-max-parallel-8-context512-4096-8192-concurrency1-2-4-8-raw.json
ollama ps | tee "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e2-max-parallel-8-ollama-ps-after.txt"
rocm-smi --showmeminfo vram --json \
  | tee "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e2-max-parallel-8-rocm-smi-after-raw.json"
```

若完整矩阵中出现失败单元或 OOM，保留 benchmark JSON、服务日志和显存快照，再评估是否回退 max-parallel 4；部分完成的矩阵不记作完整成功。

**E2 结果：**[完整原始 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e2-max-parallel-8-context512-4096-8192-concurrency1-2-4-8-raw.json)。12 格共 96/96 个正式请求成功；最高整卡 HBM 采样为 81.168 GiB，包含已驻留模型。

| 目标上下文 | C1 tok/s | C2 tok/s | C4 tok/s | C8 tok/s | C8 P95 |
|---|---:|---:|---:|---:|---:|
| 512 | 33.26 | 51.36 | 73.08 | 85.99 | 16.934 s |
| 4096 | 33.47 | 49.63 | 61.26 | 61.46 | 26.385 s |
| 8192 | 32.86 | 46.43 | 54.85 | 50.70 | 33.311 s |

C8 在 512 档继续提升吞吐；4096 档较 C4 基本持平；8192 档则比 C4 低约 7.6%，且 P95 更高。因此 C8 能稳定处理该负载，但不是所有上下文下的最高吞吐档。

### E3：长上下文（单并发）

单并发长上下文测评：目标输入 32768，模型 context 设为 34816，为输出和模板留出余量。先创建独立 tag，再以 `OLLAMA_CONTEXT_LENGTH=34816`、`OLLAMA_NUM_PARALLEL=1` 重启服务，然后跑 8192/32768 两档。Ollama 的显存需求会随并行槽位数和 context 增长，因此 E3 将并行槽位设为 1（[官方并发与显存说明](https://github.com/ollama/ollama/blob/main/docs/faq.mdx)）。矩阵自带 warmup，不额外重复发短请求。

首次创建 ctx32k tag 失败，错误为 `invalid split GGUF ... must use llama.cpp split filename pattern`。原因是复用 `ollama show --modelfile` 输出的 `sha256-...` 内部 blob 作为 `FROM`；多分片 GGUF 应将 `FROM` 指向保留原始分片文件名的 wildcard（[Ollama GGUF 导入说明](https://github.com/ollama/ollama/blob/main/docs/import.mdx)）。下方修正版从 `/root/tmp` 的四个原始分片创建 tag，并保留现有 Modelfile 的模板和其他参数。

修正版已成功创建 ctx32k tag，输出确认四个模型权重层均为 `using existing layer`。随后矩阵前的 `curl http://127.0.0.1:11435/` 返回 HTTP 502，因此 benchmark 尚未启动。E3 检查命令将显式绕过代理，并在服务重启后短暂重试；若仍失败，再检查终端 A 的服务日志和监听状态。

以下代码块都在已有终端中执行。创建 tag 和矩阵命令使用子 shell，错误只结束当前步骤，父终端会返回提示符；不要用 `bash -c` 或 `exec bash -i` 包裹整段。多分片 GGUF 的 `FROM` 必须使用原始分片文件名的 wildcard，不能直接使用 `ollama show --modelfile` 输出的 `sha256-...` 内部 blob 路径。

```bash
# 终端 B：当前服务仍运行时，创建长上下文模型 tag。
set +e
export WORK=/root/tmp/qwen3.8-flash-next
export MODEL_DIR="$WORK/Qwen3.8-Flash-Next-GGUF"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_TAG=qwen3.8-flash-next-ollama-ud-q4-k-xl
export OLLAMA_TAG_CTX32K=qwen3.8-flash-next-ollama-ud-q4-k-xl-ctx32k
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
(
  set -euo pipefail
  mkdir -p "$WORK" "$RESULT_DIR"
  mapfile -t GGUF_SHARDS < <(
    find "$MODEL_DIR" -type f \
      -name 'Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf' -print | sort
  )
  if [ "${#GGUF_SHARDS[@]}" -ne 4 ]; then
    echo "[ERROR] 原始 GGUF 分片数量为 ${#GGUF_SHARDS[@]}，预期为 4：$MODEL_DIR"
    exit 1
  fi
  GGUF_DIR=$(dirname "${GGUF_SHARDS[0]}")
  for shard in "${GGUF_SHARDS[@]}"; do
    if [ "$(dirname "$shard")" != "$GGUF_DIR" ]; then
      echo '[ERROR] 4 个 GGUF 分片不在同一个目录；停止 E3。'
      exit 1
    fi
  done
  GGUF_GLOB="$GGUF_DIR/Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf"
  ollama show --modelfile "$OLLAMA_TAG" > "$WORK/Modelfile.ollama-ctx32k"
  if ! grep -Fq "$OLLAMA_MODELS/blobs/" "$WORK/Modelfile.ollama-ctx32k"; then
    echo '[ERROR] 原模型 blob 不在 /root/tmp；停止 E3。'
    exit 1
  fi
  sed -i "s|^FROM .*|FROM $GGUF_GLOB|" \
    "$WORK/Modelfile.ollama-ctx32k"
  sed -i -E 's/^(PARAMETER[[:space:]]+num_ctx)[[:space:]]+.*/\1 34816/' \
    "$WORK/Modelfile.ollama-ctx32k"
  if ! grep -Fq "FROM $GGUF_GLOB" "$WORK/Modelfile.ollama-ctx32k"; then
    echo '[ERROR] Modelfile 的 FROM 没有指向原始四分片 GGUF；停止 E3。'
    exit 1
  fi
  if ! grep -Eq '^PARAMETER[[:space:]]+num_ctx[[:space:]]+34816[[:space:]]*$' \
    "$WORK/Modelfile.ollama-ctx32k"; then
    echo '[ERROR] Modelfile 中没有确认到 num_ctx 34816；停止 E3。'
    exit 1
  fi
  ollama create "$OLLAMA_TAG_CTX32K" -f "$WORK/Modelfile.ollama-ctx32k"
)
e3_create_status=$?
if [ "$e3_create_status" -eq 0 ]; then
  echo "[OK] 已创建 $OLLAMA_TAG_CTX32K；终端 B 保留，可继续下面的 E3 步骤。"
else
  echo "[ERROR] E3 tag 创建失败（退出码 $e3_create_status）；终端 B 保留，请修复后重试。"
fi
unset e3_create_status
```

```bash
# 终端 A：tag 创建完成后，Ctrl+C 停止 E2 服务，再以单并发和 34,816 context 重启。
set +e
set -o pipefail
cd /mnt/workspace/multimodal-deploy
export WORK=/root/tmp/qwen3.8-flash-next
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_CONTEXT_LENGTH=34816
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_DEBUG=1
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export SERVER_LOG="$RESULT_DIR/qwen3.8-flash-next-ollama-e3-ctx32k-server.log"
mkdir -p "$OLLAMA_MODELS" "$RESULT_DIR"
ollama serve 2>&1 | tee -a "$SERVER_LOG"
```

```bash
# 终端 B：服务监听后直接运行两档单并发矩阵，每格 8 个请求和 1 次 warmup。
set +e
(
  set -euo pipefail
  export OLLAMA_HOST=127.0.0.1:11435
  export OLLAMA_TAG_CTX32K=qwen3.8-flash-next-ollama-ud-q4-k-xl-ctx32k
  export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
  export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
  mkdir -p "$RESULT_DIR"
  curl --noproxy '*' --retry 10 --retry-all-errors --retry-delay 2 \
    --retry-max-time 30 --max-time 5 -fsS "http://$OLLAMA_HOST/"
  ollama show --modelfile "$OLLAMA_TAG_CTX32K" \
    | tee "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e3-ctx32k-modelfile.txt"
  if ! grep -Fq '/root/tmp/qwen3.8-flash-next/ollama-models/blobs/' \
    "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e3-ctx32k-modelfile.txt"; then
    echo '[ERROR] E3 模型 blob 不在 /root/tmp；停止，不运行基准。'
    exit 1
  fi
  cd /mnt/workspace/multimodal-deploy
  python -u benchmarks/openai_server_matrix.py \
    --config configs/benchmark/mi300x-serving-matrix.json \
    --endpoint http://127.0.0.1:11435/v1/chat/completions \
    --model "$OLLAMA_TAG_CTX32K" \
    --label qwen3.8-flash-next-ollama-ud-q4-k-xl-e3-ctx32k \
    --contexts 8192,32768 --concurrencies 1 \
    --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e3-ctx32k-raw.json
  ollama ps | tee "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e3-ctx32k-ollama-ps-after.txt"
  rocm-smi --showmeminfo vram --json \
    | tee "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e3-ctx32k-rocm-smi-after-raw.json"
)
e3_matrix_status=$?
if [ "$e3_matrix_status" -eq 0 ]; then
  echo '[OK] E3 矩阵执行完成；终端 B 保留，请同步结果文件。'
else
  echo "[ERROR] E3 矩阵失败（退出码 $e3_matrix_status）；终端 B 保留。"
fi
unset e3_matrix_status
```

**E3 结果（2026-10-20）：**长上下文 tag 在 `OLLAMA_CONTEXT_LENGTH=34816`、`OLLAMA_NUM_PARALLEL=1` 下完成 2 个单并发测试格，共 16/16 个正式请求成功、HTTP 200。8192 档实际 prompt 为 5507 tokens，32768 档为 21740 tokens；两档每请求均生成 256 tokens。整卡 HBM 峰值约 79.07 GiB，测试期间相对基线增量约 0.024 GiB。原始证据见 [E3 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e3-ctx32k-raw.json)。

| 目标上下文 | 实际 prompt tokens | completion tok/s | TTFT | P50 / P95 | HBM 峰值 / 增量 |
|---:|---:|---:|---:|---:|---:|
| 8192 | 5507 | 33.477 | 0.121 s | 6.782 / 6.804 s | 79.072 / 0.024 GiB |
| 32768 | 21740 | 31.756 | 0.190 s | 7.152 / 7.167 s | 79.074 / 0.024 GiB |

E3 同时验证了服务重启后仍能从 `/root/tmp/qwen3.8-flash-next/ollama-models` 加载长上下文 tag。

### E4：启动就绪与 Mini-8 质量抽样

E4 同时记录三轮启动时间，和 llama.cpp、vLLM、SGLang 保持相同轮数。Ollama 的 `GET /` 只表示 daemon API 已监听，可能早于模型加载，因此单独记录 `process_launch_to_api_ready_s`；模型加载证据使用原生 `/api/generate` 首次请求的墙钟耗时，以及响应中的 `load_duration`、`total_duration`（纳秒，字段定义见 [Ollama API usage 指标](https://github.com/ollama/ollama/blob/main/docs/api/usage.mdx)）。两类指标都标记 `cache_state=uncontrolled`，不称作冷启动，也不把 daemon ready 当作模型 ready。

先在终端 A 按 `Ctrl+C` 停止 E3 前台服务，确认 `11435` 没有健康服务。下面的探针每轮自己启动并停止一个 Ollama 进程，使用基础模型 tag、`context=10240`、`num_parallel=1`，只测 daemon API 到 `GET /` 返回 HTTP 200 的时间。探针不会运行模型请求：

```bash
# 终端 B：E4 三轮 daemon API 启动时间。
set +e
export WORK=/root/tmp/qwen3.8-flash-next
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
mkdir -p "$RESULT_DIR"
(
  set +e
  if curl --noproxy '*' -fsS --max-time 5 "http://$OLLAMA_HOST/" >/dev/null 2>&1; then
    echo '[ERROR] 11435 仍有健康服务；先在终端 A 按 Ctrl+C 停止 E3 服务。'
  else
    for run in 1 2 3; do
      echo "=== Ollama E4 API startup $run/3 ==="
      python -u benchmarks/server_startup_probe.py \
        --health "http://$OLLAMA_HOST/" \
        --output "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-startup-api-${run}-raw.json" \
        --cache-state uncontrolled --timeout 1200 -- \
        env \
          OLLAMA_HOST="$OLLAMA_HOST" \
          OLLAMA_MODELS="$OLLAMA_MODELS" \
          OLLAMA_CONTEXT_LENGTH=10240 \
          OLLAMA_NUM_PARALLEL=1 \
          OLLAMA_KEEP_ALIVE=-1 \
          OLLAMA_DEBUG=1 \
          ollama serve
      probe_rc=$?
      echo "startup_probe_rc=$probe_rc"
      if [ "$probe_rc" -ne 0 ]; then
        break
      fi
    done
  fi
)
e4_startup_status=$?
echo "E4 API startup block finished with status $e4_startup_status; terminal B remains available."
unset e4_startup_status
```

探针的三个 JSON 中，`metric=process_launch_to_health_ready`、`health=http://127.0.0.1:11435/`，表示 daemon API 就绪时间。每轮同时生成同名 `.server.log`。探针完成后，在终端 A 用 E4 的基础模型配置启动服务，再立即执行一次原生首请求；不要先用 `ollama run` 或质量脚本预热模型：

**E4 daemon API 启动结果（2026-10-20）：**三轮探针均返回 `ready=true`、退出码 0，`startup_s` 分别为 `0.9792`、`1.0881`、`1.0999` s；中位数为 `1.0881` s。这个指标只覆盖 `ollama serve` 进程启动到 `GET /` 健康返回，不包含模型加载、权重映射或首次推理，因此不能与其他后端的模型加载就绪时间直接混用。三轮原始 JSON 与 server log 使用下列文件名保存：`qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-startup-api-{1,2,3}-raw.json` 及对应的 `.server.log`。

**E4 首次模型请求结果（2026-10-20）：**`curl_exit_code=0` 且 `done=true`。API 请求墙钟耗时为 `82.4713 s`；Ollama 报告的 `load_duration` 为 `80.9496 s`，`total_duration` 为 `82.4481 s`；本次请求的 `prompt_eval_count=60`、`eval_count=32`。这一步包含模型首次加载，因此应与上面的 daemon API ready 分开报告；两者不能相加为严格的端到端冷启动时间，因为服务探针与模型请求的缓存状态均为 `uncontrolled`。原始计时证据见 [E4 首次模型请求 timing JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-first-model-response-timing.json)。

```bash
# 终端 A：启动 E4 质量与首模型请求所需的基础模型服务。
set +e
set -o pipefail
cd /mnt/workspace/multimodal-deploy
export WORK=/root/tmp/qwen3.8-flash-next
export OLLAMA_MODELS="$WORK/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_CONTEXT_LENGTH=10240
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_DEBUG=1
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
export SERVER_LOG="$RESULT_DIR/qwen3.8-flash-next-ollama-e4-server.log"
mkdir -p "$OLLAMA_MODELS" "$RESULT_DIR"
ollama serve 2>&1 | tee -a "$SERVER_LOG"
```

```bash
# 终端 B：服务监听后，先记录一次模型首次请求的墙钟耗时和原生 usage。
set +e
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_TAG=qwen3.8-flash-next-ollama-ud-q4-k-xl
export RESULT_DIR=/mnt/workspace/multimodal-deploy/results/llm/qwen3.8-flash-next
mkdir -p "$RESULT_DIR"
if curl --noproxy '*' -fsS --max-time 5 "http://$OLLAMA_HOST/" >/dev/null; then
  payload=$(printf '{"model":"%s","prompt":"请用一句中文说明你是什么模型。","stream":false,"keep_alive":-1,"options":{"temperature":0,"num_predict":32}}' "$OLLAMA_TAG")
  first_request_json="$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-first-model-response-raw.json"
  first_t0=$(date +%s%N)
  curl --noproxy '*' -fsS --max-time 900 \
    -H 'Content-Type: application/json' \
    -d "$payload" "http://$OLLAMA_HOST/api/generate" \
    > "$first_request_json"
  first_rc=$?
  first_t1=$(date +%s%N)
  python - "$first_request_json" "$RESULT_DIR/qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-first-model-response-timing.json" \
    "$first_t0" "$first_t1" "$first_rc" <<'PY'
import json
import sys
from pathlib import Path

response_path, timing_path, t0, t1, rc = sys.argv[1:]
try:
    response = json.loads(Path(response_path).read_text(encoding="utf-8"))
except Exception as exc:
    response = {"parse_error": str(exc)}
start_ns = int(t0)
end_ns = int(t1)
timing = {
    "schema_version": 1,
    "metric": "first_model_request_wall_time",
    "cache_state": "uncontrolled",
    "curl_exit_code": int(rc),
    "request_wall_s": (end_ns - start_ns) / 1e9,
    "load_duration_ns": response.get("load_duration"),
    "total_duration_ns": response.get("total_duration"),
    "prompt_eval_count": response.get("prompt_eval_count"),
    "eval_count": response.get("eval_count"),
    "done": response.get("done"),
}
Path(timing_path).write_text(json.dumps(timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(timing, ensure_ascii=False))
PY
  echo "首模型请求已保存：$first_request_json"
else
  echo '[ERROR] 11435 尚未返回 API ready；不发送首模型请求。'
fi
```

`first-model-response-timing.json` 的 `request_wall_s` 是 API 请求墙钟耗时，`load_duration_ns` 是 Ollama 报告的模型加载时间；两者都包含在 E4 记录中，但不与 `/` 的 daemon ready 时间相加。本次首请求已经完成；下方 Mini-8 质量抽样结果也已记录。

沿用相同的 GGUF 与模型 tag，先用 `temperature=0` 和 4096 上限验证固定题集，不在正式抽样期间并发运行其他请求：

```bash
cd /mnt/workspace/multimodal-deploy
python -u benchmarks/openai_quality_probe.py \
  --questions benchmarks/quality/mtbench-mini-8.json \
  --endpoint http://127.0.0.1:11435/v1/chat/completions \
  --model qwen3.8-flash-next-ollama-ud-q4-k-xl \
  --label qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-quality \
  --max-tokens 4096 --temperature 0 \
  --output results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-quality-raw.json
```

**E4 Mini-8 结果（2026-10-20）：**8 个题目全部 HTTP 200 且成功，空回答和长度截断均为 0；总 completion 为 8565 tokens，平均端到端耗时 37.844 s，平均观测输出速度 34.986 tok/s。各题均以 `stop` 结束；其中 writing 和 humanities 的长输出分别耗时 147.407 s 和 87.934 s，属于题目输出长度差异，不能单独解释为模型质量差异。该结果只证明接口、生成和结果落盘流程可用，没有人工评分，因此不构成回答质量排名。原始证据见 [E4 Mini-8 JSON](../../results/llm/qwen3.8-flash-next/qwen3.8-flash-next-ollama-ud-q4-k-xl-e4-quality-raw.json)。

## 状态

- [x] 确认 AMD/ROCm 设备与 `/root/tmp` 可用空间（根盘剩余 345G）
- [x] 安装并记录 Ollama 版本（`0.40.6`）；确认 ROCm 目录 `rocm_v7_2`
- [x] 下载并确认 4 个 UD-Q4_K_XL GGUF 分片齐全
- [x] 重新导入模型到 `/root/tmp/qwen3.8-flash-next/ollama-models`（未写入 `/mnt/workspace`）
- [x] CLI 冒烟与 GPU 使用确认（`ollama ps` 显示 `100% GPU`、context `10240`）
- [x] 当前 root 模型目录下的 OpenAI-compatible API 测评请求（E1 24/24 成功）
- [x] 确认服务重启后模型仍可加载（E3 ctx32k tag）
- [x] E1 单请求基线（3 档 context，24/24 成功）
- [x] E2 并发扩展（12 格均 8/8 请求成功）
- [x] E3 长上下文（8192/32768，16/16 成功）
- [x] E4 daemon API 启动就绪时间（3 轮，中位数 1.0881 s；只代表 API ready）
- [x] E4 Mini-8 质量抽样（8/8 成功、0 空回答、0 长度截断）
- [x] E4 首次模型请求耗时（墙钟 82.4713 s；`load_duration` 80.9496 s）
