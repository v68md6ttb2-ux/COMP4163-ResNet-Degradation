#!/usr/bin/env bash
# Core replication: plain-20/56 and ResNet-20/56, three seeds each (12 runs),
# full 64,000-iteration recipe of the paper plus the course protocol
# (45k/5k split, per-epoch evaluation, mixed precision).
#
# Run from the project root:  bash scripts/run_core.sh
#
# Two runs at a time. The 32x32 networks have such small kernels that a single
# process leaves the GPU half idle: two concurrent runs measure 27 ms/iter each
# against 24 ms/iter for one run alone, so two at a time is ~1.75x the
# throughput of one (three at a time is slower again). Each run is a separate
# process with its own seed, so running them together changes nothing about
# the numbers, only the wall-clock.
#
# Resumable: a run whose results.json already exists is not repeated.
set -euo pipefail

PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
JOBS=${JOBS:-2}

mkdir -p build
queue=build/queue_core.txt
: > "$queue"

for seed in 0 1 2; do
  for arch in plain resnet; do
    for depth in 20 56; do
      out="runs/${arch}${depth}-s${seed}"
      if [ -f "$out/results.json" ]; then
        echo "skip $out (already done)" >&2
      else
        echo "$arch $depth $seed $out" >> "$queue"
      fi
    done
  done
done

# A 56-layer run already keeps the GPU busy on its own (two of them together
# give only 5% more aggregate throughput, because the larger kernels fill the
# device), so a second 56-layer run only slows the first by 2x. The 20-layer
# runs are the opposite: tiny kernels leave the device half idle. So alternate
# one deep run with one shallow run, and each worker keeps its own lane.
awk '$2==56' "$queue" > build/queue_core_long.txt
awk '$2==20' "$queue" > build/queue_core_short.txt
paste -d'\n' build/queue_core_long.txt build/queue_core_short.txt | grep -v '^$' \
  | xargs -P "$JOBS" -L 1 bash -c \
  'PYTHONPATH=. "$0" -m src.train --arch "$1" --depth "$2" --seed "$3" --out "$4"' "$PY"

"$PY" -m src.plot \
    --network "plain-20:runs/plain20-s*" --network "plain-56:runs/plain56-s*" \
    --readoff "plain-20:1.5:9.7" --readoff "plain-56:5.9:13.3" \
    --title "plain networks (cf. Figure 6 left)" --out figures/fig6_plain.png
"$PY" -m src.plot \
    --network "ResNet-20:runs/resnet20-s*" --network "ResNet-56:runs/resnet56-s*" \
    --readoff "ResNet-20:0.8:8.8" --readoff "ResNet-56:0.1:7.0" \
    --title "residual networks (cf. Figure 6 middle)" --out figures/fig6_resnet.png

"$PY" -m src.summarize runs/plain20-s* runs/plain56-s* runs/resnet20-s* runs/resnet56-s* \
    --out reports/summary.md
