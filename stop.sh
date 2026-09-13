#!/usr/bin/env bash
# stop.sh — stop the DeepSeek-V4.1-Flash EXL3 vLLM server started by start.sh
#
# Removes dsv41-exl3-head on this machine and dsv41-exl3-worker on the
# worker. Weights and compile caches stay on disk.
#
# Equivalent to: ./start.sh stop
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/start.sh" stop
