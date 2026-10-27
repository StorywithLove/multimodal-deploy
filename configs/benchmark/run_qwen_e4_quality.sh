#!/usr/bin/env bash
set -u

# Run the same MT-Bench Mini-8 quality sample against Q4 and Q5 on an
# ephemeral Linux benchmark instance.  The instance may keep raw results in
# /mnt/workspace/results; reviewed JSON files can later be copied into the
# repository's results/llm/qwen3.8-flash-next/ directory.

WORK="${WORK:-/root/tmp/qwen3.8-flash-next}"
MODEL_DIR="${MODEL_DIR:-$WORK/Qwen3.8-Flash-Next-GGUF}"
SERVER="${SERVER:-$WORK/llama.cpp/build/bin/llama-server}"
RESULT_ROOT="${RESULT_ROOT:-results}"
QUESTIONS="${QUESTIONS:-configs/benchmark/quality/mtbench-mini-8.json}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"

mkdir -p "$RESULT_ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

find_first_shard() {
  local quant="$1"
  find "$MODEL_DIR" -type f \
    -name "*UD-${quant}_K_XL*00001*.gguf" -print -quit
}

SERVER_PID=""
cleanup_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

wait_for_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl --noproxy '*' -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

run_one() {
  local quant="$1"
  local model="$2"
  local label="qwen3.8-flash-next-ud-${quant,,}-k-xl-llamacpp-e4-quality-${MAX_TOKENS}"
  local output="$RESULT_ROOT/${label}.json"
  local log="$RESULT_ROOT/${label}.server.log"
  local probe_status

  echo "[e4-quality] starting ${quant}: ${model}"
  "$SERVER" -m "$model" -ngl 999 -c 10240 --parallel 1 --jinja \
    --host "$HOST" --port "$PORT" >"$log" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_health; then
    echo "[e4-quality] ${quant} did not become healthy; tailing ${log}" >&2
    tail -n 120 "$log" >&2 || true
    cleanup_server
    return 1
  fi

  python3 -u configs/benchmark/openai_quality_probe.py \
    --questions "$QUESTIONS" \
    --endpoint "http://${HOST}:${PORT}/v1/chat/completions" \
    --label "$label" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output "$output"
  probe_status=$?

  cleanup_server
  echo "[e4-quality] ${quant} output: ${output}"
  echo "[e4-quality] ${quant} server log: ${log}"
  return "$probe_status"
}

[ -x "$SERVER" ] || die "llama-server not found or not executable: $SERVER"
[ -f "$QUESTIONS" ] || die "question file not found: $QUESTIONS"

if ! python3 - "$QUESTIONS" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
items = data.get("items") if isinstance(data, dict) else None
expected = {
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
}
categories = {item.get("category") for item in items} if isinstance(items, list) else set()
if not isinstance(items, list) or len(items) != 8 or categories != expected:
    raise SystemExit(
        f"expected exactly 8 MT-Bench Mini-8 categories, got {len(items) if isinstance(items, list) else 0}: "
        + ", ".join(sorted(str(category) for category in categories))
    )
print(f"[e4-quality] question set OK: {path} ({len(items)} items)")
PY
then
  die "invalid MT-Bench Mini-8 question file: $QUESTIONS"
fi

Q4="$(find_first_shard Q4)"
Q5="$(find_first_shard Q5)"
[ -f "$Q4" ] || die "Q4 first shard not found under $MODEL_DIR"
[ -f "$Q5" ] || die "Q5 first shard not found under $MODEL_DIR"

if curl --noproxy '*' -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  die "${HOST}:${PORT} already has a healthy service; stop it before running this script"
fi

overall_status=0
run_one Q4 "$Q4" || overall_status=1
run_one Q5 "$Q5" || overall_status=1

echo "--- generated quality files ---"
find "$RESULT_ROOT" -maxdepth 1 -type f \
  -name 'qwen3.8-flash-next-ud-*-llamacpp-e4-quality-*.json' -print | sort
echo "Finished with status ${overall_status}."
echo "The terminal will stay open; press Enter to close."
read -r _
exit "$overall_status"
