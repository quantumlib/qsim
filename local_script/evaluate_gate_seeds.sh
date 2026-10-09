#!/bin/bash
# Compare gate-batch lookahead budgets without changing the planner.
# Usage: ./local_script/evaluate_gate_seeds.sh [circuit] [threads]
# Each run always includes the empty and resident candidates; this varies
# only the number of additional later-gate lookahead seeds.

set -euo pipefail
cd "$(dirname "$0")/.."

CIRCUIT=${1:-circuits/circuit_q24}
THREADS=${2:-4}
BIN=${BIN:-apps/qsim_gate_batch.x}

if [ ! -x "$BIN" ]; then
  echo "missing executable: $BIN" >&2
  exit 1
fi

echo "circuit=$CIRCUIT threads=$THREADS tile_qubits=19 fused=3 floor=5 commute=1"
printf '%-16s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n' lookahead_seeds total_s batches executable swaps plan_s swap_s gate_s

for seeds in 0 1 8 64; do
  output=$(OMP_NUM_THREADS="$THREADS" "$BIN" \
    -c "$CIRCUIT" -s 1 -t "$THREADS" -f 3 -l 19 -e 5 -x 1 -g "$seeds" -v 2 2>&1)
  total=$(printf '%s\n' "$output" | sed -n 's/^simu time is \([^ ]*\) seconds\./\1/p')
  stats=$(printf '%s\n' "$output" | sed -n 's/^gate-batch runner: .*gate batches, .*raw gates, \([^ ]*\) executable gates, \([^ ]*\) qubit swaps\./\1 \2/p')
  batches=$(printf '%s\n' "$output" | sed -n 's/^gate-batch runner: \([^ ]*\) gate batches.*/\1/p')
  executable=$(printf '%s\n' "$stats" | awk '{print $1}')
  swaps=$(printf '%s\n' "$stats" | awk '{print $2}')
  breakdown=$(printf '%s\n' "$output" | sed -n 's/^breakdown: plan \([^ ]*\) s, swap passes \([^ ]*\) s, fuse \([^ ]*\) s, gate passes \([^ ]*\) s\./\1 \2 \4/p')
  plan_s=$(printf '%s\n' "$breakdown" | awk '{print $1}')
  swap_s=$(printf '%s\n' "$breakdown" | awk '{print $2}')
  gate_s=$(printf '%s\n' "$breakdown" | awk '{print $3}')
  printf '%-16s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n' "$seeds" "$total" "$batches" "$executable" "$swaps" "$plan_s" "$swap_s" "$gate_s"
done
