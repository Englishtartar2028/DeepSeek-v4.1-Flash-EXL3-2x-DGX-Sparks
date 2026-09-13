#!/usr/bin/env bash
# memguard.sh — kill the local vLLM container before the kernel OOM killer
# wedges a DGX Spark.
#
# On GB10 host RAM *is* GPU memory: every cudaMalloc comes straight out of
# /proc/meminfo MemAvailable (measured: a 4 GiB cudaMalloc drops MemAvailable
# by 4 GiB before anything touches it). When the pool is exhausted the kernel
# OOM killer picks small desktop daemons (their RSS is ordinary anon memory,
# the 100 GiB of weights is not attributed to any process) and the box wedges
# until a hard reboot — that is exactly what happened on 2026-09-11 (see
# HANDOFF.md). This guard watches MemAvailable and kills the container while
# the host can still act.
#
# usage: memguard.sh <container> <threshold-GiB> <logfile> [pidfile]
#   DSV41_MEM_GUARD_INTERVAL  seconds between samples (default 1)
#   DSV41_MEM_GUARD_STRIKES   consecutive low samples before the kill (default 2)
#   DSV41_MEM_GUARD_REPORT    seconds between periodic MemAvailable lines (default 30)
#
# Exits when the container is gone. Never touches other containers.
set -u
container="${1:?container name}"
threshold_gib="${2:?threshold GiB}"
logfile="${3:?logfile}"
pidfile="${4:-}"
interval="${DSV41_MEM_GUARD_INTERVAL:-1}"
strikes_needed="${DSV41_MEM_GUARD_STRIKES:-2}"
report_every="${DSV41_MEM_GUARD_REPORT:-30}"

[ -n "$pidfile" ] && echo $$ >"$pidfile"
mkdir -p "$(dirname "$logfile")"
threshold_kb=$(awk -v g="$threshold_gib" 'BEGIN { printf "%d", g * 1048576 }')

say() { printf '%s memguard[%s]: %s\n' "$(date '+%F %T')" "$(hostname)" "$*" >>"$logfile"; }

avail_kb() { awk '/^MemAvailable:/ { print $2 }' /proc/meminfo; }
running() { docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; }

say "armed: container=$container threshold=${threshold_gib}GiB interval=${interval}s strikes=$strikes_needed"
strikes=0
last_report=0
low_water=$(avail_kb)
# Wait up to 60 s for the container to appear (it is started right after us).
for _ in $(seq 1 60); do running && break; sleep 1; done
while running; do
    now_kb=$(avail_kb)
    [ "$now_kb" -lt "$low_water" ] && low_water=$now_kb
    epoch=$(date +%s)
    if [ $((epoch - last_report)) -ge "$report_every" ]; then
        # top RSS processes: tells GPU-side (unattributed) from host-side growth
        top=$(ps -eo rss,comm --sort=-rss 2>/dev/null | awk 'NR>1 && NR<=4 { printf "%s=%dMiB ", $2, $1/1024 }')
        say "MemAvailable=$((now_kb / 1024)) MiB (low-water $((low_water / 1024)) MiB) top: ${top}"
        last_report=$epoch
    fi
    if [ "$now_kb" -lt "$threshold_kb" ]; then
        strikes=$((strikes + 1))
        say "LOW MemAvailable=$((now_kb / 1024)) MiB < ${threshold_gib} GiB (strike $strikes/$strikes_needed)"
        if [ "$strikes" -ge "$strikes_needed" ]; then
            say "KILLING $container (host would wedge otherwise)"
            docker kill "$container" >>"$logfile" 2>&1 || true
            sleep 2
            say "after kill: MemAvailable=$(( $(avail_kb) / 1024 )) MiB"
            [ -n "$pidfile" ] && rm -f "$pidfile"
            exit 3
        fi
    else
        strikes=0
    fi
    sleep "$interval"
done
say "container gone; low-water MemAvailable was $((low_water / 1024)) MiB — exiting"
[ -n "$pidfile" ] && rm -f "$pidfile"
exit 0
