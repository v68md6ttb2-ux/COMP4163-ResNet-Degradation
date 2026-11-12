"""Summarise runs into the table the topic card asks for.

For every network: the training (45k), validation (5k) and test (10k) errors
of every seed at the end of the schedule, their mean, and their spread
(largest minus smallest).  A diverged run is listed with its divergence
iteration and left out of the mean.  The two differences of means required by
the card are printed below the table.

Usage:
    python -m src.summarize runs/plain20-s* runs/plain56-s* runs/resnet20-s* runs/resnet56-s* \
        --out reports/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ARCH_ORDER = {"plain": 0, "resnet": 1}
MEASURES = [("train_err", "train 45k"), ("val_err", "val 5k"), ("test_err", "test 10k")]


def load_rows(runs: list[str]) -> dict[tuple[str, int], list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run in runs:
        path = Path(run) / "results.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        groups[(r["arch"], r["depth"])].append(r)
    for rows in groups.values():
        rows.sort(key=lambda r: r["seed"])
    return dict(sorted(groups.items(), key=lambda kv: (ARCH_ORDER.get(kv[0][0], 9), kv[0][1])))


def mean_spread(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), max(values) - min(values)


def name(arch: str, depth: int) -> str:
    return f"plain-{depth}" if arch == "plain" else f"ResNet-{depth}"


def group_means(rows: list[dict]) -> dict[str, float]:
    """Means over the non-diverged seeds, keyed by measure."""
    ok = [r for r in rows if not r["diverged"]]
    out = {}
    for key, _ in MEASURES:
        out[key] = mean_spread([r["final"][key] for r in ok])[0]
    out["running_train_err"] = mean_spread([r["final"]["running_train_err"] for r in ok])[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="reports/summary.md")
    args = p.parse_args()

    groups = load_rows(args.runs)
    lines = ["| Network | Seed | Train 45k (%) | Val 5k (%) | Test 10k (%) | Minutes | Status |",
             "|---|---|---|---|---|---|---|"]
    for (arch, depth), rows in groups.items():
        for r in rows:
            status = f"diverged at {r['diverged_iteration']}" if r["diverged"] else f"iter {r['iterations_run']}"
            lines.append(
                f"| {name(arch, depth)} | {r['seed']} | {r['final']['train_err']:.2f} | "
                f"{r['final']['val_err']:.2f} | {r['final']['test_err']:.2f} | "
                f"{r['elapsed_s'] / 60:.1f} | {status} |"
            )
        ok = [r for r in rows if not r["diverged"]]
        for label, fn in (("mean", statistics.fmean), ("spread", None)):
            cells = []
            for key, _ in MEASURES:
                values = [r["final"][key] for r in ok]
                cells.append(f"{mean_spread(values)[0]:.2f}" if label == "mean"
                             else f"{mean_spread(values)[1]:.2f}")
            lines.append(f"| {name(arch, depth)} | **{label}** | " + " | ".join(cells) + " | | |")
        if any(r["diverged"] for r in rows):
            lines.append(f"| {name(arch, depth)} | **note** | {len(rows) - len(ok)} of {len(rows)} "
                         f"seeds diverged and are excluded from mean/spread | | | | |")

    # The two differences of means the card asks for, with the card's decision rule.
    lines.append("")
    means = {k: group_means(v) for k, v in groups.items()}
    spreads = {}
    for k, v in groups.items():
        ok = [r for r in v if not r["diverged"]]
        spreads[k] = {key: mean_spread([r["final"][key] for r in ok])[1]
                      for key in [key for key, _ in MEASURES] + ["running_train_err"]}

    def diff_text(a, b, key, unit="pp"):
        if a not in means or b not in means:
            return None
        d = means[a][key] - means[b][key]
        bound = max(spreads[a][key], spreads[b][key])
        verdict = "real" if d > bound else "not clear of the seed spread"
        return (f"{name(*a)} minus {name(*b)}: {key} {means[a][key]:.2f} - {means[b][key]:.2f} "
                f"= {d:+.2f} {unit}; larger spread {bound:.2f}; {verdict}")

    plain = (("plain", 56), ("plain", 20))
    res = (("resnet", 20), ("resnet", 56))
    for key in ("train_err", "running_train_err"):
        t = diff_text(plain[0], plain[1], key)
        if t:
            lines.append(f"- {t}")
    t = diff_text(res[0], res[1], "test_err")
    if t:
        lines.append(f"- {t} (paper: 8.75 - 6.97 = 1.78, one run each)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["network", "seed", "train_err", "val_err", "test_err", "minutes", "status"])
        for (arch, depth), rows in groups.items():
            for r in rows:
                writer.writerow([name(arch, depth), r["seed"], f"{r['final']['train_err']:.2f}",
                                 f"{r['final']['val_err']:.2f}", f"{r['final']['test_err']:.2f}",
                                 f"{r['elapsed_s'] / 60:.1f}",
                                 "diverged" if r["diverged"] else "ok"])

    print("\n".join(lines))
    print(f"\nsaved {out} and {csv_path}")


if __name__ == "__main__":
    main()
