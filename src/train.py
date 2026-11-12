"""CIFAR-10 training in the course protocol (paper Sec. 4.2 + topic card).

Recipe: SGD momentum 0.9, weight decay 1e-4, He (MSRA, fan-in) init, BN,
batch 128, lr 0.1 divided by 10 at 32k and 48k iterations, stop at 64k.

Course protocol:
  * one epoch is 351 iterations of 128 images (45,000 // 128, last partial
    batch dropped); the schedule is counted in iterations;
  * the 45,000 training images are the first 45,000 of the concatenated
    batches, the last 5,000 are validation;
  * after every epoch and at iteration 64,000, evaluation-mode errors on the
    un-augmented 45k / 5k / 10k images, plus the running training error on the
    augmented minibatches (training mode, the way the network sees them);
  * mixed precision (autocast float16 + gradient scaler) in every run,
    evaluation in float32;
  * a run that produces a non-finite loss is stopped and recorded as
    diverged at that iteration; it is not replaced.

Example:
    python -m src.train --arch plain  --depth 20 --seed 0 --out runs/plain20-s0
    python -m src.train --arch resnet --depth 20 --seed 0 --out runs/resnet20-s0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

from .data import NUM_TRAIN, CIFAR10Data
from .models import build_model, count_parameters

DEFAULT_ITERS = 64_000
LR_MILESTONES = (32_000, 48_000)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=("plain", "resnet"), required=True)
    p.add_argument("--depth", type=int, required=True, help="6n+2, e.g. 20 or 56")
    p.add_argument("--shortcut", choices=("A", "B", "C"), default="A")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--lr-milestones", type=int, nargs="+", default=list(LR_MILESTONES))
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--warmup-iters", type=int, default=0,
                   help="paper warms up only 110+ layer nets: 400 iters at lr 0.01")
    p.add_argument("--warmup-lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true", help="debugging only; every reported run uses AMP")
    args = p.parse_args()
    args.amp = not args.no_amp
    if args.out is None:
        args.out = f"runs/{args.arch}{args.depth}-s{args.seed}"
    return args


def learning_rate_at(it: int, args: argparse.Namespace) -> float:
    if args.warmup_iters and it < args.warmup_iters:
        return args.warmup_lr
    return args.lr * (args.lr_gamma ** sum(it >= m for m in args.lr_milestones))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    epoch_iters = NUM_TRAIN // args.batch_size

    data = CIFAR10Data(root=args.data_dir, download=not args.no_download, device=device)
    model = build_model(args.arch, args.depth, args.shortcut).to(device)
    num_params = count_parameters(model)
    (out_dir / "model.txt").write_text(str(model), encoding="utf-8")
    print(f"device={device}  {args.arch}-{args.depth}  seed={args.seed}  params={num_params:,}  "
          f"epoch={epoch_iters} iters x {args.batch_size}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "iteration", "lr", "running_train_err", "train_err",
                     "val_err", "test_err", "epoch_loss", "elapsed_s"])

    start = time.time()
    history: deque[tuple[int, int]] = deque(maxlen=epoch_iters)  # (correct, seen) per minibatch
    it = 0
    epoch = 0
    diverged = False
    diverged_at: int | None = None
    last = {"iteration": 0, "running_train_err": float("nan"), "train_err": float("nan"),
            "val_err": float("nan"), "test_err": float("nan")}

    while it < args.iters and not diverged:
        epoch += 1
        model.train()
        perm = torch.randperm(NUM_TRAIN, generator=generator, device=device)
        loss_sum, seen, correct = 0.0, 0, 0
        lr = learning_rate_at(it, args)
        for k in range(epoch_iters):
            if it >= args.iters:
                break
            idx = perm[k * args.batch_size : (k + 1) * args.batch_size]
            x = data.augment(data.train_x[idx], generator)
            y = data.train_y[idx]

            lr = learning_rate_at(it, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)
            if not torch.isfinite(loss).all():
                diverged, diverged_at = True, it + 1
                print(f"DIVERGED at iteration {diverged_at} (loss={loss.item()})", flush=True)
                break

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            it += 1
            batch_correct = (logits.detach().argmax(1) == y).sum().item()
            history.append((batch_correct, y.numel()))
            loss_sum += loss.item()
            seen += y.numel()
            correct += batch_correct

        # Evaluation after every (possibly partial) epoch, in evaluation mode,
        # on un-augmented images.
        run_correct = sum(c for c, _ in history)
        run_seen = sum(n for _, n in history)
        last = {
            "iteration": it,
            "running_train_err": 100.0 * (1.0 - run_correct / run_seen),
            "train_err": data.error(model, data.train_x, data.train_y),
            "val_err": data.error(model, data.val_x, data.val_y),
            "test_err": data.error(model, data.test_x, data.test_y),
        }
        epoch_loss = loss_sum / max(seen, 1)
        elapsed = time.time() - start
        writer.writerow([epoch, it, f"{lr:.6g}", f"{last['running_train_err']:.4f}",
                         f"{last['train_err']:.4f}", f"{last['val_err']:.4f}",
                         f"{last['test_err']:.4f}", f"{epoch_loss:.6f}", f"{elapsed:.1f}"])
        csv_file.flush()
        print(f"epoch {epoch:3d}  iter {it:6d}  lr={lr:.4g}  run={last['running_train_err']:5.2f}%  "
              f"train={last['train_err']:5.2f}%  val={last['val_err']:5.2f}%  "
              f"test={last['test_err']:5.2f}%  loss={epoch_loss:.4f}  {elapsed / 60:.1f} min",
              flush=True)

    csv_file.close()
    torch.save({"model": model.state_dict(), "iteration": it}, out_dir / "last.pth")
    results = {
        "arch": args.arch,
        "depth": args.depth,
        "shortcut": args.shortcut,
        "seed": args.seed,
        "parameters": num_params,
        "epochs_completed": epoch,
        "iterations_run": it,
        "diverged": diverged,
        "diverged_iteration": diverged_at,
        "final": last,
        "elapsed_s": time.time() - start,
        "args": vars(args),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFINAL iter={it}  train={last['train_err']:.2f}%  val={last['val_err']:.2f}%  "
          f"test={last['test_err']:.2f}%  running={last['running_train_err']:.2f}%  "
          f"diverged={diverged}  time={(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
