"""Fisheye projection as a post-process on a wide pinhole render.

MuJoCo renders a strict pinhole -- `fovy` is the whole intrinsic model, and
there is no lens object to give optical properties to. A fisheye therefore
cannot be rendered directly; it has to be produced by rendering a WIDE pinhole
image and resampling it through a fisheye projection.

The resampling is a fixed grid per parameter set, so it costs one
`grid_sample` per frame and runs batched on the GPU alongside the render.

Geometry
--------
For an output pixel at normalised radius ``rn`` in [0, 1] (1 = the edge of the
fisheye circle), the model gives the ray angle from the optical axis:

    rectilinear   theta = atan(rn * tan(theta_max))      (a pinhole; d = 0)
    equidistant   theta = rn * theta_max                 (r = f*theta; d = 1)

and blends between them with ``distortion`` d, then adds a residual barrel term
``k1``. Real add-on fisheyes sit between the two, nearer equidistant.

That ray is then looked up in the pinhole source at radius

    r_src = tan(theta) / tan(fov_render / 2)

which is why **the render FOV must cover the fisheye's full field**: a pinhole
image simply does not contain rays outside its own frustum, so no warp can
invent them. `build_grid` refuses rather than silently sampling the border.

The practical ceiling is around 140 deg. tan blows up toward 90 deg, so a wider
render spends almost all its pixels on the periphery -- at 140 deg the edge is
already stretched 2.75x relative to the centre. Past that, the honest fix is
cubemap rendering, not a bigger fovy.
"""

from __future__ import annotations

import math

import torch


def build_grid(
  out_hw: tuple[int, int],
  fov_render_deg: float,
  fisheye_fov_deg: float,
  distortion: float = 1.0,
  k1: float = 0.0,
  center: tuple[float, float] = (0.0, 0.0),
  vignette_r: float = 1.0,
  device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sampling grid + validity mask for one fisheye parameter set.

  Args:
    out_hw: (H, W) of the fisheye output.
    fov_render_deg: vertical FOV the SOURCE pinhole was rendered at.
    fisheye_fov_deg: full field of the fisheye being modelled.
    distortion: 0 = rectilinear, 1 = equidistant. Cheap add-on lenses sit near 1.
    k1: residual barrel term, theta -> theta * (1 + k1 * (theta/theta_max)^2).
    center: principal-point offset in half-image units; an add-on lens is never
      perfectly centred on the sensor.
    vignette_r: normalised radius where the lens aperture cuts the image off.
      >= 1.415 means no visible circle at all (the frame corners are at
      sqrt(2)); smaller values crop a circle, which is what a small lens in
      front of a larger sensor actually looks like.
    device: torch device.

  Returns:
    grid: (1, H, W, 2) for `torch.nn.functional.grid_sample`, align_corners=False.
    valid: (1, 1, H, W) float mask, 0 outside the vignette circle.
  """
  th_max = math.radians(fisheye_fov_deg) / 2.0
  th_render = math.radians(fov_render_deg) / 2.0
  if th_max >= th_render - 1e-6:
    raise ValueError(
      f"fisheye_fov {fisheye_fov_deg} needs a render FOV strictly wider than "
      f"it ({fov_render_deg} given). The pinhole source contains no rays "
      "beyond its own frustum, so the warp cannot fabricate them."
    )

  h, w = out_hw
  ys = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1).expand(h, w)
  xs = torch.linspace(-1.0, 1.0, w, device=device).view(1, w).expand(h, w)
  xs = xs - center[0]
  ys = ys - center[1]
  rn = torch.sqrt(xs * xs + ys * ys)

  # rn is in half-image units, so the frame EDGE is 1 and the corners are
  # sqrt(2). theta_max is defined at the edge; corners simply run past it,
  # which is correct -- a circular image does not fill a square frame.
  t_rect = torch.atan(rn * math.tan(th_max))
  t_equi = rn * th_max
  theta = (1.0 - distortion) * t_rect + distortion * t_equi
  if k1:
    theta = theta * (1.0 + k1 * (theta / th_max) ** 2)
  theta = theta.clamp(max=th_render - 1e-4)

  r_src = torch.tan(theta) / math.tan(th_render)
  scale = torch.where(rn > 1e-8, r_src / rn.clamp_min(1e-8),
                      torch.ones_like(rn))
  gx, gy = xs * scale, ys * scale

  valid = ((rn <= vignette_r) & (gx.abs() <= 1.0) & (gy.abs() <= 1.0)).float()
  grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
  return grid, valid.view(1, 1, h, w)


def apply(
  src: torch.Tensor,
  grid: torch.Tensor,
  valid: torch.Tensor,
  fill: float = 0.0,
) -> torch.Tensor:
  """Resample a (B,C,H,W) pinhole render through a grid from `build_grid`."""
  b = src.shape[0]
  out = torch.nn.functional.grid_sample(
    src, grid.expand(b, -1, -1, -1), mode="bilinear",
    padding_mode="border", align_corners=False,
  )
  return out * valid + fill * (1.0 - valid)


def fisheye(
  src: torch.Tensor,
  fov_render_deg: float,
  fisheye_fov_deg: float,
  out_hw: tuple[int, int] | None = None,
  **kw,
) -> torch.Tensor:
  """One-shot convenience wrapper. Rebuilds the grid every call -- fine for
  figures, wasteful in a training loop: cache `build_grid` there."""
  hw = out_hw or (src.shape[-2], src.shape[-1])
  grid, valid = build_grid(hw, fov_render_deg, fisheye_fov_deg,
                           device=src.device, **kw)
  return apply(src, grid, valid)
