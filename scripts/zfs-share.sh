# scripts/zfs-share.sh — optional replica of EXL3 + slim Engram with zfs send|recv.
# Sourced by start.sh. Default weight path is NFSv4 (scripts/nfs-share.sh), matching
# DeepSeek-v4.1-Flash-DGX-Sparks. Use WEIGHT_SYNC=zfs only when a pool exists.
#
# GLM EXL3 2x uses rsync of the HF cache. Native DS4.1 uses NFSv4 + packed
# per-rank Engram. vLLM cpu_offload also works over NFS at load (no decode mmap).
# This ZFS path is the optional node-local replica when a pool exists.
#
# Expected datasets (override in .env):
#   HEAD:   ${ZFS_POOL}/dsv41-exl3     mountpoint = MODEL_HOST
#           ${ZFS_POOL}/dsv41-engram   only shards 47+48 + index + config
#   WORKER: same dataset names, mountpoints WORKER_MODEL_DIR / WORKER_ENGRAM_DIR
#
# One-time host setup (root, both nodes):
#   sudo apt-get install -y zfsutils-linux
#   sudo zpool create -f "$ZFS_POOL" <vdev>          # or import an existing pool
#   sudo zfs create -o recordsize=1M -o atime=off -o compression=lz4 \
#        -o mountpoint="$MODEL_HOST" "${ZFS_POOL}/dsv41-exl3"
#   sudo zfs create -o recordsize=1M -o atime=off -o compression=lz4 \
#        -o mountpoint="$HEAD_ENGRAM_ZFS_MOUNT" "${ZFS_POOL}/dsv41-engram"
#   # populate engram dataset with shards 47+48 (do NOT send the 476G native tree)
#   sudo zfs allow -u "$WORKER_USER" create,destroy,mount,receive,snapshot,rollback \
#        "$ZFS_POOL"
# Worker recv also works with: sudo -n zfs ... (NOPASSWD).

ZFS_POOL="${ZFS_POOL:-models}"
ZFS_HEAD_MODEL_DS="${ZFS_HEAD_MODEL_DS:-${ZFS_POOL}/dsv41-exl3}"
ZFS_HEAD_ENGRAM_DS="${ZFS_HEAD_ENGRAM_DS:-${ZFS_POOL}/dsv41-engram}"
ZFS_WORKER_MODEL_DS="${ZFS_WORKER_MODEL_DS:-${ZFS_POOL}/dsv41-exl3}"
ZFS_WORKER_ENGRAM_DS="${ZFS_WORKER_ENGRAM_DS:-${ZFS_POOL}/dsv41-engram}"
# Bulk transfer rides CX7, not the 10.0.0.x management alias.
ZFS_WORKER_IP="${ZFS_WORKER_IP:-10.0.22.2}"
ZFS_SSH="${ZFS_SSH:-${WORKER_USER}@${ZFS_WORKER_IP}}"
ZFS_SNAP_PREFIX="${ZFS_SNAP_PREFIX:-dsv41-sync}"
WEIGHT_SYNC="${WEIGHT_SYNC:-auto}"
HEAD_ENGRAM_ZFS_MOUNT="${HEAD_ENGRAM_ZFS_MOUNT:-$HOME/dsv41-engram-zfs}"

zfs_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$ZFS_SSH" "$@"; }

_zfs_bin() {
    if command -v zfs >/dev/null 2>&1; then
        printf '%s' zfs
        return 0
    fi
    if [ -x /usr/sbin/zfs ]; then
        printf '%s' /usr/sbin/zfs
        return 0
    fi
    return 1
}

_ZFS_HEAD_WRAP=""
zfs_head_detect() {
    local bin
    bin="$(_zfs_bin)" || return 1
    if "$bin" list >/dev/null 2>&1; then
        _ZFS_HEAD_WRAP="$bin"
        return 0
    fi
    if sudo -n "$bin" list >/dev/null 2>&1; then
        _ZFS_HEAD_WRAP="sudo -n $bin"
        return 0
    fi
    return 1
}

zfs_head() {
    [ -n "$_ZFS_HEAD_WRAP" ] || zfs_head_detect || return 1
    # shellcheck disable=SC2086
    $_ZFS_HEAD_WRAP "$@"
}

# Remote zfs wrapper: prefer unprivileged zfs allow, then sudo -n.
_ZFS_WORKER_WRAP=""
zfs_worker_detect() {
    if zfs_ssh "zfs list >/dev/null 2>&1"; then
        _ZFS_WORKER_WRAP=zfs
        return 0
    fi
    if zfs_ssh "/usr/sbin/zfs list >/dev/null 2>&1"; then
        _ZFS_WORKER_WRAP=/usr/sbin/zfs
        return 0
    fi
    if zfs_ssh "sudo -n zfs list >/dev/null 2>&1"; then
        _ZFS_WORKER_WRAP="sudo -n zfs"
        return 0
    fi
    if zfs_ssh "sudo -n /usr/sbin/zfs list >/dev/null 2>&1"; then
        _ZFS_WORKER_WRAP="sudo -n /usr/sbin/zfs"
        return 0
    fi
    _ZFS_WORKER_WRAP=""
    return 1
}

zfs_worker() {
    [ -n "$_ZFS_WORKER_WRAP" ] || zfs_worker_detect || return 1
    zfs_ssh "$_ZFS_WORKER_WRAP $*"
}

zfs_dataset_for_path() {
    local path="$1" src fstype
    path="$(readlink -f "$path" 2>/dev/null || printf '%s' "$path")"
    fstype="$(findmnt -n -o FSTYPE --target "$path" 2>/dev/null || true)"
    [ "$fstype" = "zfs" ] || return 1
    src="$(findmnt -n -o SOURCE --target "$path" 2>/dev/null || true)"
    [ -n "$src" ] || return 1
    printf '%s' "$src"
}

zfs_dataset_exists() {
    local name="$1"
    zfs_head list -H -o name "$name" >/dev/null 2>&1
}

zfs_worker_dataset_exists() {
    local name="$1"
    zfs_worker list -H -o name "$name" >/dev/null 2>&1
}

zfs_bytes_for_engram() {
    local total=0 shard
    for shard in $ENGRAM_SHARDS; do
        total=$((total + $(stat -c '%s' "$ENGRAM_DIR/$shard" 2>/dev/null || echo 0)))
    done
    total=$((total + $(stat -c '%s' "$ENGRAM_DIR/model.safetensors.index.json" 2>/dev/null || echo 0)))
    total=$((total + $(stat -c '%s' "$ENGRAM_DIR/config.json" 2>/dev/null || echo 0)))
    printf '%s' "$total"
}

zfs_df_avail_bytes() {
    local path="$1"
    df -PB1 "$path" 2>/dev/null | awk 'NR==2{print $4}'
}

zfs_tools_ok() {
    _zfs_bin >/dev/null 2>&1 || return 1
    zfs_head list >/dev/null 2>&1 || return 1
}

# 0 = use ZFS, 1 = caller should fall back, 2 = hard fail (WEIGHT_SYNC=zfs)
zfs_should_use() {
    case "$WEIGHT_SYNC" in
        rsync|off|0) return 1 ;;
    esac
    if ! zfs_tools_ok; then
        [ "$WEIGHT_SYNC" = "zfs" ] && return 2
        return 1
    fi
    if ! zfs_ssh true 2>/dev/null; then
        warn "ZFS_SSH=${ZFS_SSH} unreachable — trying WORKER_SSH=${WORKER_SSH} for send"
        ZFS_SSH="$WORKER_SSH"
        if ! zfs_ssh true 2>/dev/null; then
            [ "$WEIGHT_SYNC" = "zfs" ] && return 2
            return 1
        fi
    fi
    if ! zfs_worker_detect; then
        [ "$WEIGHT_SYNC" = "zfs" ] && return 2
        return 1
    fi
    local model_ds engram_ds
    model_ds="$(zfs_dataset_for_path "$MODEL_HOST" 2>/dev/null || true)"
    [ -n "$model_ds" ] && ZFS_HEAD_MODEL_DS="$model_ds"
    if ! zfs_dataset_exists "$ZFS_HEAD_MODEL_DS"; then
        [ "$WEIGHT_SYNC" = "zfs" ] && return 2
        return 1
    fi
    if ! zfs_dataset_exists "$ZFS_HEAD_ENGRAM_DS"; then
        [ "$WEIGHT_SYNC" = "zfs" ] && return 2
        return 1
    fi
    return 0
}

zfs_preflight() {
    local rc=0
    zfs_should_use || rc=$?
    if [ "$rc" = 1 ]; then
        log "ZFS weight sync not available — will rsync if WEIGHT_SYNC=${WEIGHT_SYNC}"
        return 1
    fi
    if [ "$rc" = 2 ]; then
        die "WEIGHT_SYNC=zfs but send/recv is not ready.
  Head:  zfsutils-linux + dataset ${ZFS_HEAD_MODEL_DS} (EXL3) and ${ZFS_HEAD_ENGRAM_DS} (shards 47+48 only).
  Worker ${ZFS_SSH}: zfs recv into ${ZFS_WORKER_MODEL_DS} / ${ZFS_WORKER_ENGRAM_DS}
         (passwordless: zfs allow -u ${WORKER_USER} create,mount,receive,snapshot ${ZFS_POOL}
          or sudo -n zfs).
  Do not send the 476 GiB native tree — Engram dataset is shards 47+48 + index + config.
  WEIGHT_SYNC=auto falls back to rsync; WEIGHT_SYNC=rsync forces it."
    fi

    local need avail
    need=$(( $(du -sb "$MODEL_HOST" | awk '{print $1}') + $(zfs_bytes_for_engram) ))
    avail="$(zfs_ssh "df -PB1 '${WORKER_HOME:-/home/${WORKER_USER}}' 2>/dev/null | awk 'NR==2{print \$4}'" || true)"
    if [ -n "${avail:-}" ] && [ "$avail" -lt "$need" ]; then
        die "worker has $((avail/1024/1024/1024)) GiB free, need ~$((need/1024/1024/1024)) GiB for a local ZFS replica (EXL3 + Engram 47+48). Free NVMe or WEIGHT_SYNC=rsync will fail the same way."
    fi
    log "ZFS preflight OK  head ${ZFS_HEAD_MODEL_DS} + ${ZFS_HEAD_ENGRAM_DS}  →  ${ZFS_SSH} ${ZFS_WORKER_MODEL_DS} + ${ZFS_WORKER_ENGRAM_DS}"
    return 0
}

zfs_list_snaps() {
    local ds="$1" side="$2"
    if [ "$side" = worker ]; then
        zfs_worker "list -H -t snapshot -o name -s creation $ds" 2>/dev/null \
            | awk -F@ '{print $2}' | grep "^${ZFS_SNAP_PREFIX}-" || true
    else
        zfs_head list -H -t snapshot -o name -s creation "$ds" 2>/dev/null \
            | awk -F@ '{print $2}' | grep "^${ZFS_SNAP_PREFIX}-" || true
    fi
}

zfs_send_one() {
    local src_ds="$1" dst_ds="$2" mount="$3" label="$4"
    local common new recv_cmd
    new="${ZFS_SNAP_PREFIX}-$(date -u +%Y%m%dT%H%M%SZ)"
    log "ZFS snapshot ${src_ds}@${new} (${label})"
    zfs_head snapshot "${src_ds}@${new}" \
        || die "zfs snapshot ${src_ds}@${new} failed (need snapshot permission)"

    common="$(comm -12 \
        <(zfs_list_snaps "$src_ds" head | sort) \
        <(zfs_list_snaps "$dst_ds" worker | sort) \
        | tail -n 1 || true)"

    recv_cmd="$_ZFS_WORKER_WRAP recv -F -u -o mountpoint=${mount} ${dst_ds}"
    if [ -n "$common" ]; then
        log "ZFS incremental send ${src_ds}@${common} → @${new}  (${label}, CX7 ${ZFS_SSH})"
        zfs_head send -i "${src_ds}@${common}" "${src_ds}@${new}" \
            | zfs_ssh "$recv_cmd" \
            || die "zfs send -i ${src_ds}@${common} ${src_ds}@${new} | recv ${dst_ds} failed"
    else
        log "ZFS full send ${src_ds}@${new} → ${dst_ds}  (${label}, CX7 ${ZFS_SSH})"
        zfs_head send "${src_ds}@${new}" \
            | zfs_ssh "$recv_cmd" \
            || die "zfs send ${src_ds}@${new} | recv ${dst_ds} failed"
    fi
    zfs_worker "set canmount=on ${dst_ds}" >/dev/null 2>&1 || true
    zfs_worker "mount ${dst_ds}" >/dev/null 2>&1 || true
    zfs_ssh "sudo -n chown -R ${WORKER_USER}:${WORKER_USER} '${mount}' 2>/dev/null || true" || true
    if ! zfs_ssh "test -e '${mount}'"; then
        die "after zfs recv, ${ZFS_SSH}:${mount} is missing"
    fi
    log "ZFS ${label} mounted on worker at ${mount}"
}

zfs_sync_weights() {
    zfs_worker_detect || die "worker cannot zfs list/recv as ${ZFS_SSH}"
    zfs_send_one "$ZFS_HEAD_MODEL_DS" "$ZFS_WORKER_MODEL_DS" "$WORKER_MODEL_DIR" "EXL3"
    zfs_send_one "$ZFS_HEAD_ENGRAM_DS" "$ZFS_WORKER_ENGRAM_DS" "$WORKER_ENGRAM_DIR" "Engram"
    # Point docker mounts at the received datasets (node-local).
    local mp
    mp="$(zfs_worker "get -H -o value mountpoint $ZFS_WORKER_MODEL_DS" | tr -d '\r')"
    [ -n "$mp" ] && [ "$mp" != "-" ] && [ "$mp" != "none" ] && WORKER_MODEL_DIR="$mp"
    mp="$(zfs_worker "get -H -o value mountpoint $ZFS_WORKER_ENGRAM_DS" | tr -d '\r')"
    [ -n "$mp" ] && [ "$mp" != "-" ] && [ "$mp" != "none" ] && WORKER_ENGRAM_DIR="$mp"
    export WORKER_MODEL_DIR WORKER_ENGRAM_DIR
    log "worker ZFS mounts: model=${WORKER_MODEL_DIR} engram=${WORKER_ENGRAM_DIR}"
}
