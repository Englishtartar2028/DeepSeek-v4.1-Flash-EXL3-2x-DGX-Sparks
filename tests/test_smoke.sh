#!/usr/bin/env bash
# tests/test_smoke.sh — 17*19, tools, optional image. Requires a healthy server.
set -euo pipefail
BASE="${1:-http://127.0.0.1:8888}"
MODEL="${2:-DeepSeek-v4.1-Flash-EXL3}"
AUTH=()
[ -n "${VLLM_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer ${VLLM_API_KEY}")

body=$(curl -fsS "${AUTH[@]}" "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is 17*19? Reply with the integer only."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
echo "$body"
echo "$body" | grep -E '323' >/dev/null \
  || { echo "smoke FAIL: expected 323 in the completion" >&2; exit 1; }
echo "smoke OK: 17*19 -> 323"
