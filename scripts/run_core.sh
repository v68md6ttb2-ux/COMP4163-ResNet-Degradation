#!/usr/bin/env bash
# Core replication: plain-20/56 and ResNet-20/56, three seeds each (12 runs),
# full 64,000-iteration recipe of the paper plus the course protocol
# (45k/5k split, per-epoch evaluation, mixed precision).
# Run from the project root:  bash scripts/run_core.sh
set -euo pipefail

PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python

for seed in 0 1 2; do
  for arch in plain resnet; do
    for depth in 20 56; do
      $PY -m src.train --arch "$arch" --depth "$depth" --seed "$seed" \
          --out "runs/${arch}${depth}-s${seed}"
    done
  done
done

$PY -m src.plot \
    --network "plain-20:runs/plain20-s*" --network "plain-56:runs/plain56-s*" \
    --readoff "plain-20:1.5:9.7" --readoff "plain-56:5.9:13.3" \
    --title "plain networks (cf. Figure 6 left)" --out figures/fig6_plain.png
$PY -m src.plot \
    --network "ResNet-20:runs/resnet20-s*" --network "ResNet-56:runs/resnet56-s*" \
    --readoff "ResNet-20:0.8:8.8" --readoff "ResNet-56:0.1:7.0" \
    --title "residual networks (cf. Figure 6 middle)" --out figures/fig6_resnet.png

$PY -m src.summarize runs/plain20-s* runs/plain56-s* runs/resnet20-s* runs/resnet56-s* \
    --out reports/summary.md
