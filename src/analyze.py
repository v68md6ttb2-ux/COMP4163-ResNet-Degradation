"""Diagnostics used to argue *why* the degradation problem happens.

responses -- standard deviation of each layer's response (paper Fig. 7).
             The paper measures the 3x3 layer outputs *after BN and before
             ReLU/addition*, i.e. exactly the BatchNorm outputs here.

gradients -- per-layer backpropagated gradient norms, to check the paper's
             claim that gradients do not vanish in the plain nets.

Usage:
    python -m src.analyze responses --runs runs/plain20 runs/plain56 \
        runs/resnet20 runs/resnet56 --out figures/fig7.png
    python -m src.analyze gradients --runs runs/plain20 runs/plain56 \
        --out figures/gradients.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn

from .data import CIFAR10Data
from .models import build_model


def load_run(run_dir: str, device: torch.device):
    run_dir = Path(run_dir)
    args = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    model = build_model(args["arch"], args["depth"], args.get("shortcut", "A")).to(device)
    checkpoint = torch.load(run_dir / "last.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, args


def response_stds(model: nn.Module, batches, device: torch.device, num_batches: int):
    modules = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    values: dict[str, list[float]] = {name: [] for name, _ in modules}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            values[name].append(output.detach().float().std().item())
        return hook

    handles = [m.register_forward_hook(make_hook(name)) for name, m in modules]
    with torch.no_grad():
        for images, _ in batches[:num_batches]:
            model(images)
    for h in handles:
        h.remove()
    return [float(np.mean(values[name])) for name, _ in modules]


def gradient_norms(model: nn.Module, batches, device: torch.device):
    model.train()
    images, targets = batches[0]
    model.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(images), targets)
    loss.backward()

    norms = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.weight.grad is not None:
            norms.append(float(module.weight.grad.detach().norm().item()))
    model.zero_grad(set_to_none=True)
    return norms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("responses", "gradients"))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--num-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs in length")

    data = CIFAR10Data(root=args.data_dir, download=True, device=device)

    def test_batches():
        for i in range(0, data.test_x.shape[0], args.batch_size):
            yield data.normalise(data.test_x[i : i + args.batch_size]), data.test_y[i : i + args.batch_size]

    batches = list(test_batches())

    curves = []
    for run in args.runs:
        model, run_args = load_run(run, device)
        if args.mode == "responses":
            curve = response_stds(model, batches, device, args.num_batches)
        else:
            curve = gradient_norms(model, batches, device)
        curves.append(curve)
        print(f"{run}: {len(curve)} layers, range [{min(curve):.3g}, {max(curve):.3g}]")

    if args.mode == "responses":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label, curve in zip(labels, curves):
            axes[0].plot(range(len(curve)), curve, marker=".", lw=1.2, label=label)
            axes[1].plot(range(len(curve)), sorted(curve, reverse=True), marker=".", lw=1.2, label=label)
        axes[0].set_title("layer index (original order)")
        axes[1].set_title("layer index (sorted by magnitude)")
        axes[0].set_ylabel(r"std of layer response")
        for ax in axes:
            ax.set_xlabel("layer index")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Layer response magnitudes (cf. paper Fig. 7)")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, curve in zip(labels, curves):
            ax.semilogy(range(len(curve)), curve, marker=".", lw=1.2, label=label)
        ax.set_xlabel("conv layer index")
        ax.set_ylabel("gradient norm (log scale)")
        ax.set_title("Per-layer gradient norms (cf. paper Sec. 4.1)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = Path(args.out or f"figures/{args.mode}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)

    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps({"labels": labels, "curves": curves}, indent=2), encoding="utf-8")
    print(f"saved {out} and {json_out}")


if __name__ == "__main__":
    main()
