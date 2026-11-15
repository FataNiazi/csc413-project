"""Random-shift augmentation, verbatim from multimodal_rl's sear/models.

Kept byte-for-byte (including `last_shift` / `repeat_last`, which lets the real
batch reuse the shift drawn for the sim batch) because Alpha3 pretrained under
exactly this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RandomShiftsAug(nn.Module):
  def __init__(self, pad, default_mode="bilinear", padding_mode="zeros", align_corners=False):
    super().__init__()
    self.pad = pad
    self.default_mode = default_mode
    self.padding_mode = padding_mode
    self.align_corners = align_corners
    self.last_shift = None

  def _make_grid(self, n, h, w, device, dtype, repeat_last):
    """Per-axis so non-square inputs work; EXACTLY the original when h == w.

    The side-by-side multi-view control feeds 84x168. Both the coordinate
    ranges and the shift scale are per-pixel quantities, so each has to use its
    own axis length -- sharing h for both, as the square version did, would
    stretch the shift along the wider axis rather than translating it.
    """
    def _axis(size):
      eps = 1.0 / (size + 2 * self.pad)
      return torch.linspace(
        -1.0 + eps, 1.0 - eps, size + 2 * self.pad, device=device, dtype=dtype
      )[:size]

    ax, ay = _axis(w), _axis(h)
    gx = ax.unsqueeze(0).repeat(h, 1)          # x varies along the width
    gy = ay.unsqueeze(1).repeat(1, w)          # y varies along the height
    base_grid = torch.stack([gx, gy], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)

    if repeat_last and (self.last_shift is not None):
      shift = self.last_shift
    else:
      shift = torch.randint(
        0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=device, dtype=dtype
      )
      scale = torch.tensor(
        [2.0 / (w + 2 * self.pad), 2.0 / (h + 2 * self.pad)],
        device=device, dtype=dtype)
      shift = shift * scale
      self.last_shift = shift
    return base_grid + shift

  def forward(self, x, repeat_last=False, mode=None):
    n, _, h, w = x.size()
    x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
    grid = self._make_grid(n, h, w, x.device, x.dtype, repeat_last)
    return F.grid_sample(
      x,
      grid,
      mode=(mode or self.default_mode),
      padding_mode=self.padding_mode,
      align_corners=self.align_corners,
    )
