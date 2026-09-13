#!/bin/bash
# Dump the Python (and native) stacks of every vLLM process in this container
# (py-spy; the container needs --cap-add SYS_PTRACE). Used by start.sh's hang
# detector. The native frames tell a CPU spin from a CUDA sync on a hung kernel.
for p in $(ps -eo pid,comm --no-headers | awk '$2 ~ /VLLM|vllm|python/ {print $1}'); do
    echo "=== pid $p $(tr '\0' ' ' </proc/"$p"/cmdline 2>/dev/null | cut -c1-100) ==="
    timeout 60 py-spy dump --pid "$p" 2>&1 | head -150
    case "$(cat /proc/"$p"/comm 2>/dev/null)" in
        VLLM::Worker*) echo "--- native"; timeout 90 py-spy dump --native --pid "$p" 2>&1 | head -80 ;;
    esac
done
