#!/usr/bin/env bash
# Extension A, tuning grid: plain-20 and plain-56 at five initial learning
# rates, seed 0, paper's two divisions by 10, full 64k schedule, no warm-up.
# The 0.1 runs are also replication seed 0 (runs/plain{20,56}-s0), so they are
# reused if present rather than retrained.
# Run from the project root:  bash scripts/run_extension_a.sh
#
# Two runs at a time, for the same reason as run_core.sh. Resumable.
set -euo pipefail

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
JOBS=${JOBS:-2}

mkdir -p build
queue=build/queue_extension_a.txt
: > "$queue"

for depth in 20 56; do
  for lr in 0.01 0.03 0.1 0.3 0.5; do
    if [ "$lr" = "0.1" ]; then
      out="runs/plain${depth}-s0"
    else
      out="runs/plain${depth}-lr${lr}-s0"
    fi
    if [ -f "$out/results.json" ]; then
      echo "skip $out (already done)" >&2
    else
      echo "plain $depth 0 $out $lr" >> "$queue"
    fi
  done
done

# Same lane split as run_core.sh: a 56-layer run fills the GPU by itself, a
# 20-layer run leaves it half idle, so pair them instead of pairing like with
# like.
awk '$2==56' "$queue" > build/queue_ext_long.txt
awk '$2==20' "$queue" > build/queue_ext_short.txt
paste -d'\n' build/queue_ext_long.txt build/queue_ext_short.txt | grep -v '^$' \
  | xargs -P "$JOBS" -L 1 bash -c \
  'PYTHONPATH=. "$0" -m src.train --arch "$1" --depth "$2" --seed "$3" --out "$4" --lr "$5"' "$PY"

# Selection: lowest evaluation-mode training error at the last iteration.
"$PY" -m src.tune runs/plain20-lr*-s0 runs/plain20-s0 runs/plain56-lr*-s0 runs/plain56-s0 \
    --out reports/extension_a_tuning.md
