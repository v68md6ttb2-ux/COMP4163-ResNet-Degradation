"""Extension A figure: final error against the initial learning rate.

Two panels, both at iteration 64,000: the training error, which is the
pre-registered selection metric, and the test error, which it is not.

The left panel is what the tuning rule reads. It is also where the interesting
thing happens: plain-56 does not merely get worse as the rate rises, it stops
training altogether, and those runs are drawn as crosses at their recorded
value (89.98% is chance for ten classes). plain-20 at the same rates is fine,
which is the visual form of hypothesis H1.

A log error axis is used because the points span 0.43% to 90% and the shape of
the low end is part of the result.

Usage:
    python -m src.plot_tuning --out figures/extension_a_lr.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1 and 2 of the reference palette, validated for light mode:
# CVD dE 24.7 (protan), 32.7 (tritan), 33.6 normal vision, contrast >= 3:1.
COLORS = {"20": "#2a78d6", "56": "#eb6834"}
LABELS = {"20": "plain-20", "56": "plain-56"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d9d8d4"


def load_points(patterns: list[str]) -> dict[str, dict[str, list[tuple[float, float, float, bool, int | None]]]]:
    """depth -> list of (lr, train_err, test_err, diverged, diverged_iteration)."""
    out: dict[str, list] = {"20": [], "56": []}
    for pattern in patterns:
        for run in sorted(glob.glob(pattern)):
            res = Path(run) / "results.json"
            args = Path(run) / "args.json"
            if not res.exists() or not args.exists():
                continue
            r = json.loads(res.read_text(encoding="utf-8"))
            a = json.loads(args.read_text(encoding="utf-8"))
            depth = str(a["depth"])
            f = r["final"]
            out[depth].append((float(a["lr"]), f["train_err"], f["test_err"],
                               bool(r["diverged"]), r["diverged_iteration"]))
    for depth in out:
        out[depth].sort(key=lambda t: t[0])
    return out


def panel(ax, points, key, title, ylabel=False):
    """key: 1 = train error, 2 = test error."""
    for depth in ("20", "56"):
        pts = points[depth]
        if not pts:
            continue
        lrs = [p[0] for p in pts]
        errs = [p[key] for p in pts]
        color = COLORS[depth]

        live = [(lr, e) for lr, e, p in zip(lrs, errs, pts) if not p[3]]
        ax.plot([x for x, _ in live], [y for _, y in live], color=color, lw=2.0,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=LABELS[depth], zorder=3)
        dead = [(lr, e, p) for lr, e, p in zip(lrs, errs, pts) if p[3]]
        if dead:
            ax.plot([x for x, _, _ in dead], [y for _, y, _ in dead], color=color, lw=0,
                    marker="x", markersize=9, markeredgewidth=2.0, zorder=3)

        # Selected point: the lowest training error among the runs that trained.
        if key == 1 and live:
            sel_lr, sel_e = min(live, key=lambda t: t[1])
            ax.scatter([sel_lr], [sel_e], s=190, facecolors="none", edgecolors=color,
                       linewidths=2.0, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 0.5])
    ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "0.5"])
    ax.minorticks_off()
    ax.set_yticks([1, 3, 10, 30, 100])
    ax.set_yticklabels(["1", "3", "10", "30", "100"])
    ax.set_xlabel("initial learning rate", color=INK_MUTED)
    if ylabel:
        ax.set_ylabel("error at iteration 64,000 (%)", color=INK_MUTED)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(True, which="major", color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[
        "runs/plain20-s0", "runs/plain20-lr*-s0",
        "runs/plain56-s0", "runs/plain56-lr*-s0",
    ])
    p.add_argument("--out", default="figures/extension_a_lr.png")
    args = p.parse_args()

    points = load_points(args.runs)
    for depth in ("20", "56"):
        print(f"plain-{depth}: {len(points[depth])} rates -> "
              f"{[round(x[0], 3) for x in points[depth]]}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    panel(axes[0], points, 1, "training error — the selection metric", ylabel=True)
    panel(axes[1], points, 2, "test error — reported, not used for selection")

    # Annotate the diverged points once, in the left panel where they matter.
    # 0.3 and 0.5 sit close together on a log axis, so each label is pushed to
    # its own side of its marker rather than centred on it.
    ax = axes[0]
    dead_lrs = sorted(lr for lr, tr, te, div, it in points["56"] if div)
    for lr, tr, te, div, it in points["56"]:
        if div:
            left = lr == dead_lrs[0]
            ax.annotate(f"iter {it}", xy=(lr, tr),
                        xytext=(-9 if left else 9, -3), textcoords="offset points",
                        ha="right" if left else "left", va="center",
                        fontsize=8, color=INK_MUTED)

    # Direct labels for the two selected rates (identity is never colour alone).
    for depth, lr, off, ha in (("20", _selected_lr(points["20"]), (9, 9), "left"),
                               ("56", _selected_lr(points["56"]), (-9, -2), "right")):
        ax.annotate(f"selected {lr:g}", xy=(lr, [q[1] for q in points[depth] if q[0] == lr][0]),
                    xytext=off, textcoords="offset points", fontsize=8.5,
                    color=INK_MUTED, ha=ha, va="center")

    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=COLORS["56"], lw=0, marker="x",
                          markersize=8, markeredgewidth=2.0))
    labels.append("plain-56 diverged (89.98% = chance)")
    ax.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK,
              loc="upper left", handletextpad=0.5)
    fig.suptitle("Selection at the paper's starting rates: the deep plain network runs out of "
                 "stable learning rate", fontsize=11.5, color=INK)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"saved {out}")


def _selected_lr(pts) -> float:
    live = [(lr, tr) for lr, tr, te, div, it in pts if not div]
    return min(live, key=lambda t: t[1])[0] if live else float("nan")


if __name__ == "__main__":
    main()
