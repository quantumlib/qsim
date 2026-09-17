#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CIRCUIT="${1:-${ROOT_DIR}/circuits/circuit_q24}"
THREADS="${2:-}"
if [[ -z "${THREADS}" ]]; then
  THREADS="$(lscpu -p=CORE 2>/dev/null | awk -F, '!/^#/ {print $1}' | sort -u | wc -l)"
  THREADS="${THREADS//[[:space:]]/}"
fi
THREADS="${THREADS:-1}"
RESULTS="${QSIM_BENCH_RESULTS:-/tmp/qsim-tiled-benchmark-$(date +%Y%m%d-%H%M%S).tsv}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
LOG_DIR="${RESULTS%.tsv}.logs"
mkdir -p "${LOG_DIR}"

SYSTEM_INFO="${RESULTS%.tsv}.system.txt"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "kernel=$(uname -srv)"
  lscpu 2>/dev/null || true
  echo "--- process status ---"
  grep -E '^(Cpus_allowed_list|Mems_allowed_list|Threads):' /proc/self/status || true
  if command -v numactl >/dev/null 2>&1; then
    echo "--- numactl hardware ---"
    numactl --hardware || true
  fi
} >"${SYSTEM_INFO}"

make -C "${ROOT_DIR}" qsim tiled >/dev/null
printf 'label\telapsed_seconds\tcpu_percent\tmetrics\tlog\n' >"${RESULTS}"

record() {
  local label="$1"; shift
  local safe_label="${label//[^[:alnum:]_.-]/_}"
  local log="${LOG_DIR}/${safe_label}.log"
  local timing="${TMP_DIR}/${safe_label}.time"
  /usr/bin/time -f '%e\t%P' -o "${timing}" "$@" >"${log}" 2>&1 || true
  local elapsed cpu metrics
  elapsed="$(cut -f1 "${timing}")"
  cpu="$(cut -f2 "${timing}")"
  metrics="$(grep '^backend=' "${log}" | head -1 | tr '\t' ' ' || true)"
  printf '%s\t%s\t%s\t%s\t%s\n' "${label}" "${elapsed:-NA}" "${cpu:-NA}" "${metrics}" "${log}" >>"${RESULTS}"
  printf '%-32s %8s s %s\n' "${label}" "${elapsed:-NA}" "${metrics}"
}

echo "Circuit: ${CIRCUIT}"
echo "Physical-core threads: ${THREADS}"
echo "Results: ${RESULTS}"

# Explicit staged configurations used for the performance matrix.  The
# implementation remains the same binary; these settings isolate tile
# partitioning, fusion, swizzling, NUMA policy, and SMT cooperation.
record "tile-only" "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -d 30 \
  -t "${THREADS}" -i 1 -l 16 -f 1 -s linear -n off -v 2
record "tile-plus-batching" "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -d 30 \
  -t "${THREADS}" -i 1 -l 16 -f 3 -s linear -n off -v 2
record "tile-batching-swizzle" "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -d 30 \
  -t "${THREADS}" -i 1 -l 16 -f 3 -s xor_swizzle -n off -v 2
record "tile-batching-swizzle-numa" "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -d 30 \
  -t "${THREADS}" -i 1 -l 16 -f 3 -s xor_swizzle -n on -v 2
record "tile-batching-swizzle-numa-smt" "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -d 30 \
  -t "${THREADS}" -i 2 -l 16 -f 3 -s xor_swizzle -n on -v 2

for fused in 2 3 4; do
  record "baseline-f${fused}" "${SCRIPT_DIR}/qsim_base.x" -c "${CIRCUIT}" -d 30 -t "${THREADS}" -f "${fused}" -v 1
done

for tile in 0 12 16; do
  for swizzle in linear round_robin numa_group xor_swizzle dynamic; do
    record "tile${tile}-${swizzle}-numa-off-smt1" \
      "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -t "${THREADS}" -i 1 \
      -d 30 -l "${tile}" -f 3 -s "${swizzle}" -n off -v 2
    record "tile${tile}-${swizzle}-numa-on-smt2" \
      "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -t "${THREADS}" -i 2 \
      -d 30 -l "${tile}" -f 3 -s "${swizzle}" -n on -v 2
  done
done

if [[ "${PERF:-0}" == 1 ]] && command -v perf >/dev/null 2>&1; then
  perf stat -e cycles,instructions,cache-references,cache-misses \
    "${SCRIPT_DIR}/qsim_tiled.x" -c "${CIRCUIT}" -t "${THREADS}" \
    -i 1 -f 3 -s xor_swizzle -n off -v 1 \
    >"${TMP_DIR}/perf.stdout" 2>"${RESULTS%.tsv}.perf.txt" || true
fi

if command -v numactl >/dev/null 2>&1; then
  numactl --hardware >"${RESULTS%.tsv}.numactl.txt" 2>&1 || true
fi

echo "Completed benchmark matrix: ${RESULTS}"
