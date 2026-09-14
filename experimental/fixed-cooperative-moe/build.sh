#!/usr/bin/env bash
# Build only; never selects a serving profile or touches a running container.
set -euo pipefail
upstream_checkout=${1:?Usage: build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}
output_dir=${2:?Usage: build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY}
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
upstream_pin=02aef45cd681b960a00afcd0749a4ab99e6c1bfe
test "$(git -C "$upstream_checkout" rev-parse "$upstream_pin^{commit}")" = "$upstream_pin"
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
test -z "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" || {
  echo 'Refusing a nonempty build output directory.' >&2; exit 1;
}
mkdir -- "$output_dir/upstream"
git -C "$upstream_checkout" archive "$upstream_pin" exllamav3/exllamav3_ext |
  tar -x -C "$output_dir/upstream"
cp -- "$source_dir/native/goal50_fixed_coop.cu" "$output_dir/goal50_fixed_coop.cu"
cp -- "$source_dir/native/goal50_fixed_coop_kernel.cuh" "$source_dir/native/exl3_moe_coop.cuh" \
  "$output_dir/upstream/exllamav3/exllamav3_ext/quant/"
cp -- "$source_dir/goal50_fixed_coop_runtime.py" "$output_dir/goal50_fixed_coop_runtime.py"
"${NVCC:-/usr/local/cuda/bin/nvcc}" -std=c++17 -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr \
  -gencode arch=compute_121a,code=sm_121a -shared -Xcompiler -fPIC --ptxas-options=-v \
  -I "$output_dir/upstream/exllamav3/exllamav3_ext" \
  "$output_dir/goal50_fixed_coop.cu" -o "$output_dir/goal50-fixed-coop.so" \
  > "$output_dir/goal50-fixed-coop-build.log" 2>&1
sha256sum "$output_dir/goal50-fixed-coop.so" "$output_dir/goal50_fixed_coop_runtime.py"
