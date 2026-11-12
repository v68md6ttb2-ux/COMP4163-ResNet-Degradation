"""Tuning table for extension A: one row per (depth, initial learning rate).

Reads results.json from run directories given as arguments, groups by depth,
and prints the evaluation-mode training error at the last iteration (the
card's selection criterion) next to the validation error (reported, not used
for selection).  The selected rate per depth is the row with the lowest
training error.

Usage:
    python -m src.tune runs/plain20-lr*-s0 runs/plain20-s0 runs/plain56-lr*-s0 runs/plain56-s0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows: dict[int, list[dict]] = defaultdict(list)
    for run in args.runs:
        path = Path(run) / "results.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        rows[r["depth"]].append(r)

    lines = ["| Depth | Initial lr | Train 45k (%) | Val 5k (%) | Test 10k (%) | Minutes | Status |",
             "|---|---|---|---|---|---|---|"]
    selected = {}
    for depth in sorted(rows):
        runs = sorted(rows[depth], key=lambda r: r["args"]["lr"])
        ok = [r for r in runs if not r["diverged"]]
        best = min(ok, key=lambda r: r["final"]["train_err"]) if ok else None
        selected[depth] = best
        for r in runs:
            lr = r["args"]["lr"]
            mark = " **selected**" if r is best else ""
            status = f"diverged at {r['diverged_iteration']}" if r["diverged"] else f"iter {r['iterations_run']}"
            lines.append(
                f"| {depth} | {lr:g}{mark} | {r['final']['train_err']:.2f} | "
                f"{r['final']['val_err']:.2f} | {r['final']['test_err']:.2f} | "
                f"{r['elapsed_s'] / 60:.1f} | {status} |"
            )
    for depth, r in selected.items():
        if r is not None:
            lines.append("")
            lines.append(f"plain-{depth}: selected initial rate {r['args']['lr']:g} "
                         f"(training error {r['final']['train_err']:.2f}% at iteration {r['iterations_run']}).")

    text = "\n".join(lines) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"saved {out}")
    print(text)


if __name__ == "__main__":
    main()
