"""CIFAR-10 plain and residual networks from He et al., CVPR 2016.

Depth is 6n + 2: one 3x3 conv stem, three stages of n basic blocks (two
3x3 convs each), and one fully-connected layer.

    n = 3  -> 20-layer network
    n = 9  -> 56-layer network

The paper's CIFAR-10 experiments (Sec. 4.2, Fig. 6) use n in {3, 5, 7, 9}.
Both architectures have exactly the same number of parameters, depth,
width and FLOPs; the only difference is the shortcut connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature-map sizes and widths of the three stages (paper Sec. 4.2 / Table on p.7).
STAGE_WIDTHS = (16, 32, 64)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def _init_weights(module: nn.Module) -> None:
    """He et al. (2015) "MSRA" initialization, as used by the paper.

    Fan-in (not fan-out): the cited paper's derivation uses the number of
    input channels, and this is the variant that keeps the forward
    variance constant through ReLU layers.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class BasicBlock(nn.Module):
    """Two 3x3 convs plus a shortcut (paper Fig. 2).

    shortcut:
        "A" -- identity everywhere, zero-padding + stride 2 when the
               dimensions increase.  Parameter-free (used for CIFAR-10).
        "B" -- identity, except a 1x1 projection where dimensions increase.
        "C" -- 1x1 projection on every shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, shortcut: str = "A"):
        super().__init__()
        if shortcut not in ("A", "B", "C"):
            raise ValueError(f"unknown shortcut option: {shortcut!r}")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.shortcut = shortcut

        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.identity = stride == 1 and in_ch == out_ch
        needs_projection = shortcut == "C" or (shortcut == "B" and not self.identity)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False) if needs_projection else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.proj is not None:
            residual = self.proj(x)
        elif self.identity:
            residual = x
        else:  # option A: subsample with stride 2 and zero-pad extra channels
            residual = x[:, :, ::2, ::2] if self.stride != 1 else x
            pad = self.out_ch - self.in_ch
            assert pad >= 0, "option A only supports increasing channel counts"
            residual = F.pad(residual, (0, 0, 0, 0, 0, pad))

        return F.relu(out + residual, inplace=True)


class ResNetCIFAR(nn.Module):
    """Residual network for 32x32 CIFAR images (paper Sec. 4.2)."""

    def __init__(self, n: int, shortcut: str = "A", num_classes: int = 10):
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.shortcut = shortcut

        self.conv1 = conv3x3(3, STAGE_WIDTHS[0])
        self.bn1 = nn.BatchNorm2d(STAGE_WIDTHS[0])

        stages = []
        in_ch = STAGE_WIDTHS[0]
        for stage_idx, width in enumerate(STAGE_WIDTHS):
            blocks = []
            for block_idx in range(n):
                # Downsample at the first conv of the 2nd and 3rd stages.
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                blocks.append(BasicBlock(in_ch, width, stride, shortcut))
                in_ch = width
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        self.fc = nn.Linear(STAGE_WIDTHS[-1], num_classes)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        for stage in self.stages:
            x = stage(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class PlainCIFAR(nn.Module):
    """Plain (non-residual) network for 32x32 CIFAR images.

    Same depth, width, parameter count and FLOPs as the residual
    counterpart, obtained by removing the shortcut connections.
    """

    def __init__(self, n: int, num_classes: int = 10):
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n

        layers: list[nn.Module] = [
            conv3x3(3, STAGE_WIDTHS[0]),
            nn.BatchNorm2d(STAGE_WIDTHS[0]),
            nn.ReLU(inplace=True),
        ]
        in_ch = STAGE_WIDTHS[0]
        for stage_idx, width in enumerate(STAGE_WIDTHS):
            for conv_idx in range(2 * n):
                stride = 2 if stage_idx > 0 and conv_idx == 0 else 1
                layers += [
                    conv3x3(in_ch, width, stride),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
                in_ch = width
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(STAGE_WIDTHS[-1], num_classes)]
        self.features = nn.Sequential(*layers)
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def depth_to_n(depth: int) -> int:
    """Convert a 6n+2 depth into n, e.g. 20 -> 3, 56 -> 9."""
    if depth < 8 or (depth - 2) % 6 != 0:
        raise ValueError(f"depth must be 6n+2 with n >= 1, got {depth}")
    return (depth - 2) // 6


def build_model(arch: str, depth: int, shortcut: str = "A", num_classes: int = 10) -> nn.Module:
    n = depth_to_n(depth)
    if arch == "plain":
        return PlainCIFAR(n, num_classes=num_classes)
    if arch == "resnet":
        return ResNetCIFAR(n, shortcut=shortcut, num_classes=num_classes)
    raise ValueError(f"unknown arch: {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for arch in ("plain", "resnet"):
        for depth in (20, 56):
            model = build_model(arch, depth)
            print(f"{arch:6s} depth={depth:3d}  params={count_parameters(model):,}")
