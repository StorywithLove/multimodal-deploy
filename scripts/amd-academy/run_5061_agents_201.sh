#!/usr/bin/env bash
# 依次准备模型与教程文件、独立环境、模型服务和多智能体演示。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="${AGENTS_201_ROOT:-/root/tmp/agents-201}"
VENV="${AGENTS_201_VENV:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_BIN="${HF_BIN:-hf}"
VLLM_BIN="${VLLM_BIN:-vllm}"
OLLAMA_BIN="${OLLAMA_BIN:-ollama}"
LOG_DIR="${LOG_DIR:-/root/tmp/logs}"
MODEL_DIR="$ROOT/models/Qwen3.5-9B"
GGUF_DIR="$ROOT/models/Qwen3.5-9B-GGUF"
GGUF_FILE="$GGUF_DIR/Qwen3.5-9B-Q4_K_M.gguf"
REQUIREMENTS="$ROOT/google-adk-agentic-ai-tutorial/requirements/requirements.txt"
STEP="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STEP" in
  models|env|run|all) ;;
  -h|--help) printf '用法: %s [models|env|run|all] [Python 参数...]\nmodels: 下载模型和教程文件；env: 创建环境；run: 启动服务和演示；all: 依次执行（默认）。\n' "$0"; exit 0 ;;
  *) printf '未知步骤: %s；使用 --help 查看用法。\n' "$STEP" >&2; exit 2 ;;
esac

mkdir -p -- "$LOG_DIR" || exit 1
LOG_FILE="$(mktemp "$LOG_DIR/5061_agents_201_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")" || exit 1
printf '日志: %s\n' "$LOG_FILE"

workflow() {
  set -e
  export AGENTS_201_ROOT="$ROOT" OLLAMA_BASE_URL="http://127.0.0.1:11434"
  export VLLM_MODEL="Qwen/Qwen3.5-9B" OLLAMA_MODEL="ollama_chat/qwen3.5:9b"
  export OPENAI_API_BASE="http://127.0.0.1:8088/v1" OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  vllm_pid='' ollama_pid=''
  cleanup() {
    [[ -z "$vllm_pid" ]] || kill "$vllm_pid" 2>/dev/null || true
    [[ -z "$ollama_pid" ]] || kill "$ollama_pid" 2>/dev/null || true
  }
  trap cleanup EXIT

  if [[ "$STEP" == models || "$STEP" == all ]]; then
    printf '\n[1/3] 准备模型和教程文件\n'
    command -v "$HF_BIN" >/dev/null || { printf '需要 Hugging Face CLI（hf），请先安装或设置 HF_BIN。\n' >&2; return 1; }
    mkdir -p -- "$MODEL_DIR" "$GGUF_DIR"
    "$HF_BIN" download Qwen/Qwen3.5-9B --local-dir "$MODEL_DIR"
    "$HF_BIN" download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir "$GGUF_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/5061_agents_201.py" --stage prepare --root "$ROOT"
  fi

  if [[ "$STEP" == env || "$STEP" == all ]]; then
    printf '\n[2/3] 准备独立 Python 环境\n'
    [[ -f "$REQUIREMENTS" ]] || { printf '缺少依赖清单；请先运行 models 步骤。\n' >&2; return 1; }
    [[ -x "$VENV/bin/python" ]] || "$PYTHON_BIN" -m venv "$VENV"
    "$VENV/bin/python" -m pip install -r "$REQUIREMENTS"
    printf '环境: %s\n' "$VENV"
  fi

  if [[ "$STEP" == run || "$STEP" == all ]]; then
    printf '\n[3/3] 启动模型服务和多智能体演示\n'
    [[ -x "$VENV/bin/python" && -f "$GGUF_FILE" && -f "$MODEL_DIR/config.json" && -d "$ROOT/runtime/utils" ]] || { printf '准备不完整；请先运行 models 和 env 步骤。\n' >&2; return 1; }
    command -v curl >/dev/null
    command -v "$VLLM_BIN" >/dev/null || { printf '找不到 vLLM；请在基础环境安装兼容 GPU 的版本，或设置 VLLM_BIN。\n' >&2; return 1; }
    command -v "$OLLAMA_BIN" >/dev/null || { printf '找不到 Ollama；请先安装或设置 OLLAMA_BIN。\n' >&2; return 1; }
    if ! curl -fsS http://127.0.0.1:8088/v1/models >/dev/null 2>&1; then
      "$VLLM_BIN" serve "$MODEL_DIR" --served-model-name "$VLLM_MODEL" --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only --port 8088 --gpu-memory-utilization 0.6 &
      vllm_pid=$!
    fi
    if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      "$OLLAMA_BIN" serve &
      ollama_pid=$!
    fi
    local i ready
    for endpoint in http://127.0.0.1:8088/v1/models http://127.0.0.1:11434/api/tags; do
      ready=0
      for ((i=0; i<300; i++)); do
        if curl -fsS "$endpoint" >/dev/null 2>&1; then ready=1; break; fi
        if [[ "$endpoint" == *8088* && -n "$vllm_pid" ]] && ! kill -0 "$vllm_pid" 2>/dev/null; then break; fi
        if [[ "$endpoint" == *11434* && -n "$ollama_pid" ]] && ! kill -0 "$ollama_pid" 2>/dev/null; then break; fi
        sleep 2
      done
      (( ready )) || { printf '模型服务未就绪: %s\n' "$endpoint" >&2; return 1; }
    done
    curl -fsS http://127.0.0.1:8088/v1/models | "$VENV/bin/python" -c 'import json,sys; names=[x["id"] for x in json.load(sys.stdin)["data"]]; sys.exit(0 if "Qwen/Qwen3.5-9B" in names else "vLLM 未提供 Qwen/Qwen3.5-9B: " + str(names))'
    if ! "$OLLAMA_BIN" show qwen3.5:9b >/dev/null 2>&1; then
      printf 'FROM %s\nTEMPLATE """{{ .Prompt }}"""\nRENDERER qwen3.5\nPARSER qwen3.5\n' "$GGUF_FILE" > "$ROOT/Qwen3.5.Modelfile"
      "$OLLAMA_BIN" create qwen3.5:9b -f "$ROOT/Qwen3.5.Modelfile"
    fi
    "$VENV/bin/python" -u "$SCRIPT_DIR/5061_agents_201.py" --root "$ROOT" "$@"
  fi
}

workflow "$@" 2>&1 | tee -a "$LOG_FILE"
statuses=("${PIPESTATUS[@]}")
printf '运行退出码: %s；日志写入退出码: %s\n' "${statuses[0]}" "${statuses[1]}" | tee -a "$LOG_FILE"
if (( statuses[0] != 0 )); then exit "${statuses[0]}"; fi
exit "${statuses[1]}"
