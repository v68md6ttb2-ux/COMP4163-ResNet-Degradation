"""CIFAR-10 data in the course protocol (topic card, "Split" and "Measurement").

Split: concatenate the five CIFAR-10 training batches in order, hold out the
last 5,000 images as validation, train on the first 45,000.

Normalisation: the paper subtracts the per-pixel mean (no standard-deviation
division).  The mean is computed on the 45,000 training images only and is
subtracted *after* augmentation, so the zero padding of a shifted image
becomes -mean, exactly as the paper's "pad with zeros, then crop" gives.

Augmentation: pad 4 zeros on each side, random 32x32 crop, random horizontal
flip, independently per example.  Everything happens on the GPU in float32;
the images are kept on the device as uint8 (138 MB for the 45,000 training
images) and converted per batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets

NUM_TRAIN = 45_000
NUM_VAL = 5_000
PAD = 4
IMAGE = 32


class CIFAR10Data:
    """CIFAR-10 tensors on ``device`` with the 45k/5k split of the protocol."""

    def __init__(self, root: str = "data", download: bool = True, device: torch.device | str = "cuda"):
        train = datasets.CIFAR10(root=root, train=True, download=download)
        test = datasets.CIFAR10(root=root, train=False, download=download)

        # torchvision keeps the five training batches concatenated in order,
        # so index order is the protocol's "concatenate in order".
        x = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()  # uint8 (N,3,32,32)
        y = torch.tensor(train.targets, dtype=torch.long)
        xt = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
        yt = torch.tensor(test.targets, dtype=torch.long)
        assert x.shape[0] == NUM_TRAIN + NUM_VAL == 50_000

        self.train_x = x[:NUM_TRAIN].to(device)
        self.train_y = y[:NUM_TRAIN].to(device)
        self.val_x = x[NUM_TRAIN:].to(device)
        self.val_y = y[NUM_TRAIN:].to(device)
        self.test_x = xt.to(device)
        self.test_y = yt.to(device)

        # Per-pixel mean over the 45,000 training images, shape (3, 32, 32).
        self.mean = self._per_pixel_mean(self.train_x)

    @staticmethod
    def _per_pixel_mean(x: Tensor, chunk: int = 5_000) -> Tensor:
        total = torch.zeros(3, IMAGE, IMAGE, dtype=torch.float64, device=x.device)
        for i in range(0, x.shape[0], chunk):
            total += x[i : i + chunk].to(torch.float64).div_(255.0).sum(dim=0)
        return (total / x.shape[0]).to(torch.float32)

    def normalise(self, x_uint8: Tensor) -> Tensor:
        """uint8 (B,3,32,32) -> float32 with the training per-pixel mean removed."""
        return x_uint8.to(torch.float32).div_(255.0).sub_(self.mean)

    def augment(self, x_uint8: Tensor, generator: torch.Generator) -> Tensor:
        """Zero-pad 4, random 32x32 crop, random horizontal flip, subtract mean."""
        x = x_uint8.to(torch.float32).div_(255.0)
        x = F.pad(x, (PAD, PAD, PAD, PAD))
        b = x.shape[0]
        span = 2 * PAD + 1
        ar = torch.arange(IMAGE, device=x.device)

        # One independent crop offset per example, gathered row-wise then column-wise.
        i = torch.randint(0, span, (b,), generator=generator, device=x.device)
        j = torch.randint(0, span, (b,), generator=generator, device=x.device)
        rows = (i.view(b, 1) + ar).view(b, 1, IMAGE, 1).expand(b, 3, IMAGE, 2 * PAD + IMAGE)
        x = x.gather(2, rows)
        cols = (j.view(b, 1, 1) + ar).view(b, 1, 1, IMAGE).expand(b, 3, IMAGE, IMAGE)
        x = x.gather(3, cols)

        flip = (torch.rand(b, generator=generator, device=x.device) < 0.5).view(b, 1, 1, 1)
        x = torch.where(flip, torch.flip(x, dims=[3]), x)
        return x.sub_(self.mean)

    @torch.no_grad()
    def error(self, model: torch.nn.Module, x: Tensor, y: Tensor, batch: int = 500) -> float:
        """Top-1 error (%) in evaluation mode on un-augmented images."""
        model.eval()
        wrong = 0
        for i in range(0, x.shape[0], batch):
            logits = model(self.normalise(x[i : i + batch]))
            wrong += (logits.argmax(1) != y[i : i + batch]).sum().item()
        return 100.0 * wrong / x.shape[0]
