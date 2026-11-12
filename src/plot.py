"""Training curves as in the paper's Fig. 6: thin line per seed, bold mean.

Three panels per figure: the running training error (augmented minibatches,
training mode), the training error in evaluation mode on the 45k training
images, and the test error on the 10k test images.

Usage:
    python -m src.plot --network "plain-20:runs/plain20-s*" \
                       --network "plain-56:runs/plain56-s*" \
        --title "plain networks (cf. Fig. 6 left)" --out figures/fig6_plain.png
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [("running_train_err", "training error (running, augmented)"),
          ("train_err", "training error (45k, eval mode)"),
          ("test_err", "test error (10k, eval mode)")]
GRID = np.arange(0, 64_001, 500)


def read_metrics(run_dir: str) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {"iteration": []}
    with open(Path(run_dir) / "metrics.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cols.setdefault("iteration", []).append(float(row["iteration"]))
            for key, _ in PANELS:
                cols.setdefault(key, []).append(float(row[key]))
    return {k: np.asarray(v) for k, v in cols.items()}


def expand(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", action="append", required=True,
                   help='LABEL:GLOB, e.g. "plain-20:runs/plain20-s*"')
    p.add_argument("--readoff", action="append", default=[],
                   help='LABEL:TRAIN:TEST read-off from Figure 6, drawn as dotted lines')
    p.add_argument("--out", default="figures/curves.png")
    p.add_argument("--title", default="")
    p.add_argument("--ymax", type=float, default=None)
    args = p.parse_args()

    networks = []
    for spec in args.network:
        label, pattern = spec.split(":", 1)
        runs = expand(pattern)
        if not runs:
            raise SystemExit(f"no runs match {pattern!r}")
        networks.append((label, runs))

    readoffs = {}
    for spec in args.readoff:
        label, train, test = spec.split(":")
        readoffs[label] = (float(train), float(test))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for i, (label, runs) in enumerate(networks):
        color = colors[i % len(colors)]
        stacked = {key: [] for key, _ in PANELS}
        for run in runs:
            m = read_metrics(run)
            for j, (key, _) in enumerate(PANELS):
                stacked[key].append(np.interp(GRID, m["iteration"], m[key]))
                axes[j].plot(m["iteration"] / 1e3, m[key], color=color, lw=0.6, alpha=0.45)
        for j, (key, _) in enumerate(PANELS):
            mean = np.mean(stacked[key], axis=0)
            axes[j].plot(GRID / 1e3, mean, color=color, lw=2.0, label=label)
            if label in readoffs:
                axes[j].axhline(readoffs[label][0 if j < 2 else 1], color=color,
                                lw=1.0, ls=":", alpha=0.9)

    for ax, (_, title) in zip(axes, PANELS):
        ax.set_xlabel("iteration ($\\times 10^3$)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error (%)")
    axes[0].legend(frameon=False)
    if args.ymax is not None:
        axes[0].set_ylim(0, args.ymax)
    if args.title:
        fig.suptitle(args.title + "   (thin = seed, bold = mean, dotted = Fig. 6 read-off)")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
