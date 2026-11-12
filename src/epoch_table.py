"""Per-epoch table of one run, for the milestone-2 report.

Prints the running training error, the evaluation-mode training error (45k),
the validation error (5k) and the test error (10k) after every epoch, plus a
trailing row at iteration 64,000.

Usage:
    python -m src.epoch_table runs/resnet20-s0 --out reports/milestone2.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    run = Path(args.run)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    lines = [
        f"# {results['arch']}-{results['depth']}, seed {results['seed']}",
        "",
        f"Parameters: {results['parameters']:,}.  Total time: {results['elapsed_s'] / 60:.1f} min.  "
        f"Epoch = 351 iterations of 128 images; the last epoch is partial (to 64,000).",
        "",
        "| Epoch | Iteration | lr | Running train (%) | Train 45k (%) | Val 5k (%) | Test 10k (%) | Loss |",
        "|---|---|---|---|---|---|---|---|",
    ]
    with open(run / "metrics.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lines.append(
                f"| {row['epoch']} | {row['iteration']} | {float(row['lr']):.4g} | "
                f"{float(row['running_train_err']):.2f} | {float(row['train_err']):.2f} | "
                f"{float(row['val_err']):.2f} | {float(row['test_err']):.2f} | "
                f"{float(row['epoch_loss']):.4f} |"
            )
    final = results["final"]
    lines += [
        "",
        f"At iteration {final['iteration']}: test error **{final['test_err']:.2f}%** "
        f"against the paper's Table 6 value of 8.75% (one run, all 50,000 training images).",
    ]
    text = "\n".join(lines) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
