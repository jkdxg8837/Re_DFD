# SPDX-License-Identifier: Apache-2.0
"""Torch-only helpers shared by tools/phaseab_gpu.py and tools/position_metrics.py (task2.md, runs/phaseAB/)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_blur_hw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on the spatial dims (H, W) of [B, C, T, H, W]; reflect padding, radius ceil(3 sigma)."""
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = (k / k.sum()).to(x.dtype)
    B, C, T, H, W = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(B * T * C, 1, H, W)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return y.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)


def radial_power_spectrum(x: torch.Tensor, nbins: int = 40) -> torch.Tensor:
    """Radially averaged spatial power spectrum of latents [B, C, T, H, W], averaged over B, C, T.
    Frequencies are normalised to [0, 0.5] cycles/pixel (radius sqrt(fy^2 + fx^2), clipped at 0.5)."""
    B, C, T, H, W = x.shape
    P = torch.fft.fft2(x.float() - x.float().mean(dim=(-2, -1), keepdim=True)).abs().pow(2).mean(dim=(0, 1, 2))
    fy = torch.fft.fftfreq(H).view(-1, 1)
    fx = torch.fft.fftfreq(W).view(1, -1)
    r = (fy ** 2 + fx ** 2).sqrt().clamp(max=0.5)
    idx = (r / 0.5 * (nbins - 1)).round().long().flatten()
    s = torch.zeros(nbins).index_add_(0, idx, P.flatten().cpu())
    n = torch.zeros(nbins).index_add_(0, idx, torch.ones_like(P.flatten().cpu()))
    return s / n.clamp_min(1)
