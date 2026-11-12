#!/usr/bin/env bash
# Extension A, tuning grid: plain-20 and plain-56 at five initial learning
# rates, seed 0, paper's two divisions by 10, full 64k schedule, no warm-up.
# The 0.1 runs are also replication seed 0 (runs/plain{20,56}-s0), so we skip
# them here unless they are missing.
# Run from the project root:  bash scripts/run_extension_a.sh
set -euo pipefail

PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

for lr in 0.01 0.03 0.1 0.3 0.5; do
  for depth in 20 56; do
    if [ "$lr" = "0.1" ]; then
      out="runs/plain${depth}-s0"
      [ -f "$out/results.json" ] && continue
    else
      out="runs/plain${depth}-lr${lr}-s0"
    fi
    $PY -m src.train --arch plain --depth "$depth" --seed 0 --lr "$lr" --out "$out"
  done
done

# Selection: lowest evaluation-mode training error at the last iteration.
$PY -m src.tune runs/plain20-lr*-s0 runs/plain20-s0 runs/plain56-lr*-s0 runs/plain56-s0 \
    --out reports/extension_a_tuning.md
