#!/usr/bin/env bash
set -euo pipefail
export HOME=/data/lcx/codex CODEX_HOME=/data/lcx/codex/.codex
cd /data/lcx/codex
exec /data/lcx/codex/.local/bin/codex exec --ephemeral --skip-git-repo-check \
  --sandbox read-only --color never --model gpt-5.6-luna \
  -c 'mcp_servers.radeon_capacity.command="/data/lcx/codex/radeon-gmail/venv/bin/python"' \
  -c 'mcp_servers.radeon_capacity.args=["/data/lcx/codex/radeon-gmail/radeon_gmail_mcp.py"]' \
  -c 'mcp_servers.radeon_capacity.tool_timeout_sec=120' \
  'Call the radeon_pool_status MCP tool once. Report the one-click 1 GPU state, fit_count, computed_at and stale. If unavailable, report the error. Do not launch or delete instances.' </dev/null
