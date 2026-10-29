#!/usr/bin/env bash
# 在 tmux 内运行：同步显示并保存日志，保留 Python 的退出状态。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-/root/tmp/logs}"
mkdir -p -- "$LOG_DIR" || exit 1
LOG_FILE="$(mktemp "$LOG_DIR/cpu_254_gemma4_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")" || exit 1

printf '脚本: %s\nPython: %s\n日志: %s\n' "$SCRIPT_DIR/cpu_254_gemma4.py" "$PYTHON_BIN" "$LOG_FILE" | tee -a "$LOG_FILE"
"$PYTHON_BIN" -u "$SCRIPT_DIR/cpu_254_gemma4.py" "$@" 2>&1 | tee -a "$LOG_FILE"
statuses=("${PIPESTATUS[@]}")
printf 'Python 退出码: %s；日志写入退出码: %s\n' "${statuses[0]}" "${statuses[1]}" | tee -a "$LOG_FILE"
if (( statuses[0] != 0 )); then
    exit "${statuses[0]}"
fi
exit "${statuses[1]}"
