#!/usr/bin/env bash
# 在 Ubuntu 上每 10 秒下载一个 10MB 资源，逐轮记录字节数 / 速率到 CSV。
#
# 用法：
#   ./dl-10m-every-10s.sh                          # 默认源，间隔 10s，Ctrl-C 停止
#   URL=https://your.host/10mb.bin ./dl-10m-every-10s.sh
#   INTERVAL=10 DURATION=600 ./dl-10m-every-10s.sh # 跑 10 分钟
#   SAVE_DIR=./downloads ./dl-10m-every-10s.sh     # 保留文件，默认丢弃到 /dev/null
#
# CSV 列：ts,seq,rc,http_code,bytes,rate_Bps,time_total
#   速率单位是 B/s；找短下载：awk -F, 'NR>1 && $5<10485760' <log>.csv
#
# 节奏语义：以「上一轮的计划开始时刻 + INTERVAL」为目标，下载快则补足等待，
# 累计不漂移；某一轮耗时超过 INTERVAL 时立即开始下一轮并在控制台告警。
set -uo pipefail

URL="${URL:-https://speed.cloudflare.com/__down?bytes=10485760}"
INTERVAL="${INTERVAL:-10}"
DURATION="${DURATION:-0}"          # 0 = 不限时
SAVE_DIR="${SAVE_DIR:-}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-10}"
MAX_TIME="${MAX_TIME:-120}"
EXPECT_BYTES="${EXPECT_BYTES:-10485760}"

STAMP=$(date +%Y%m%d-%H%M%S)
LOG="${LOG:-dl-10m-${STAMP}.csv}"
ERRLOG="${ERRLOG:-${LOG%.csv}.err.log}"

command -v curl >/dev/null || { echo "缺少 curl：sudo apt update && sudo apt install -y curl" >&2; exit 1; }
[ -n "$SAVE_DIR" ] && mkdir -p "$SAVE_DIR"

printf 'ts,seq,rc,http_code,bytes,rate_Bps,time_total\n' > "$LOG"
printf '源      : %s\n' "$URL"
printf '间隔    : %ss   时长: %s\n' "$INTERVAL" "$([ "$DURATION" -gt 0 ] && echo "${DURATION}s" || echo '不限')"
printf '日志    : %s\n\n' "$LOG"

seq=0
next_start=$(date +%s)
SECONDS=0

trap 'echo; echo "已停止：共 $seq 轮，日志 $LOG"; exit 0' INT TERM

while :; do
  [ "$DURATION" -gt 0 ] && [ "$SECONDS" -ge "$DURATION" ] && break

  now=$(date +%s)
  [ "$now" -lt "$next_start" ] && sleep $(( next_start - now ))

  seq=$(( seq + 1 ))
  ts=$(date -Iseconds)
  if [ -n "$SAVE_DIR" ]; then dest="${SAVE_DIR}/${STAMP}-${seq}.bin"; else dest=/dev/null; fi

  metrics=$(curl -sS -L --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" \
              -o "$dest" \
              -w '%{http_code},%{size_download},%{speed_download},%{time_total}' \
              "$URL" 2>>"$ERRLOG")
  rc=$?
  case "$metrics" in *,*,*,*) ;; *) metrics="-,0,0,0" ;; esac

  printf '%s,%d,%d,%s\n' "$ts" "$seq" "$rc" "$metrics" >> "$LOG"

  bytes=${metrics#*,*}; bytes=${bytes%%,*}
  rate=${metrics##*,};  rate=${metrics%,*}
  printf '[%s] #%d rc=%d bytes=%s speed=%.2fMB/s\n' \
    "$ts" "$seq" "$rc" "$bytes" "$(awk -v r="${metrics%%,*}" -v b="$bytes" 'BEGIN{print (b>0?b/1048576:0)}')"
  if [ "$bytes" -lt "$EXPECT_BYTES" ] 2>/dev/null && [ "$rc" -eq 0 ]; then
    printf '   ! 只有 %s 字节，短于预期 %s（源被截断或限速）\n' "$bytes" "$EXPECT_BYTES"
  fi

  now=$(date +%s)
  next_start=$(( next_start + INTERVAL ))
  if [ "$now" -ge "$next_start" ]; then
    printf '   ! 本轮耗时 ≥ %ss，间隔无法维持，下一轮立即开始\n' "$INTERVAL"
    next_start=$now
  fi
done

echo "完成：共 $seq 轮，日志 $LOG"
