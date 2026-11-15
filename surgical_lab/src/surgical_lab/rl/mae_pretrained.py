"""Frozen Alpha3 SimOrReal MAE encoder as an rsl-rl CNNModel.

The RL-side half of Alpha3, ported from muse's
`python/scripts/torch_networks.py::MAEPretrained` (registry key
`simOrReal_pretrained_frozen`). Integrates the same way `resnet_e2e.py` does:
each 2D observation group gets an encoder, the latents concat with the 1D
branch and feed the MLP head.

Alpha3's train/test mismatch is preserved deliberately: pretraining showed the
transformer only the 25% visible tokens (36 of 144), and this forward runs it
over all 144 unmasked -- a 4x change in sequence length it never saw. That was
live during Alpha3's ~90-100% real result, so it is reproduced, not fixed.

`freeze=True` sets `requires_grad=False` on the conv stem, positional embedding
and transformer -- and NOT `.eval()`, matching muse. That is harmless here
because the transformer is built with dropout=0 and there is no BatchNorm, so
train/eval mode is numerically identical. `proj` stays trainable: it is a new
layer with no pretrained weights, exactly as in muse.

Unlike SB3 -- where orthogonal init silently overwrote the loaded checkpoint and
forced a `post_init_restore()` workaround -- rsl-rl defines `init_weights()` but
never calls it, so weights loaded in __init__ survive. `load_pretrained` still
reports how many tensors matched rather than trusting a silent `strict=False`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict
from vit_pytorch.vit import Transformer


class _EarlyCNNSpatial(nn.Module):
  """EarlyCNN returning the spatial map. Same weights as pretraining's EarlyCNN."""

  def __init__(self, in_channels: int, encoder_dim: int):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, encoder_dim // 8, 4, stride=2, padding=1)
    self.conv2 = nn.Conv2d(encoder_dim // 8, encoder_dim // 4, 4, stride=2, padding=1)
    self.conv3 = nn.Conv2d(encoder_dim // 4, encoder_dim // 2, 4, stride=2, padding=1)
    self.conv4 = nn.Conv2d(encoder_dim // 2, encoder_dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    return self.conv4(x)


class ExactAdaptiveAvgPool2d(nn.Module):
  """AdaptiveAvgPool2d as two fixed matmuls: out = L @ x @ R.

  Adaptive average pooling is a separable linear map -- output cell i averages
  input rows floor(i*H/O)..ceil((i+1)*H/O)-1 -- so it can be written exactly as
  matrix multiplication. The torchscript ONNX exporter cannot express
  AdaptiveAvgPool2d when the output size is not a factor of the input size
  (ABLATIONS.md issue #4); this form exports as two MatMuls with ZERO drift
  (fp32 associativity only, ~2e-7), unlike muse's bilinear-interpolate
  substitute, which was a different function with unmeasured drift.
  """

  def __init__(
    self, in_hw: int | tuple[int, int], out_hw: int | tuple[int, int]
  ):
    super().__init__()
    # Per-axis: the side-by-side arm pools a non-square map (10x21 -> 12x24),
    # so height and width no longer share a matrix.
    in_h, in_w = (in_hw, in_hw) if isinstance(in_hw, int) else tuple(in_hw)
    out_h, out_w = (out_hw, out_hw) if isinstance(out_hw, int) else tuple(out_hw)

    def pool_matrix(n_in: int, n_out: int) -> torch.Tensor:
      m = torch.zeros(n_out, n_in)
      for i in range(n_out):
        s = (i * n_in) // n_out
        e = -(-((i + 1) * n_in) // n_out)  # ceil division
        m[i, s:e] = 1.0 / (e - s)
      return m

    self.register_buffer("left", pool_matrix(in_h, out_h))
    self.register_buffer("right", pool_matrix(in_w, out_w).T)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (self.left @ x) @ self.right


def swap_pool_for_onnx_export(
  encoder: "MAEEncoder", image_hw: int | tuple[int, int]
) -> None:
  """Replace the encoder's AdaptiveAvgPool2d with its exact matmul form.

  Probes the conv stem with a dummy image to learn its output spatial size
  (84 -> 10 for the Alpha3 geometry), builds the equivalent linear map, and
  verifies the swap end-to-end on random input before letting it stand.
  """
  if isinstance(encoder.adapt_pool, ExactAdaptiveAvgPool2d):
    return
  dev = next(encoder.parameters()).device
  img_h, img_w = (image_hw, image_hw) if isinstance(image_hw, int) else tuple(image_hw)
  probe = torch.zeros(1, encoder.frame_channels, img_h, img_w, device=dev)
  with torch.no_grad():
    in_hw = tuple(encoder.early_conv(probe).shape[-2:])
    old_pool = encoder.adapt_pool
    encoder.adapt_pool = ExactAdaptiveAvgPool2d(in_hw, encoder.patch_grid).to(dev)

    # Verify on CPU: Ampere runs fp32 matmul through TF32 tensor cores
    # (10-bit mantissa), which pollutes a GPU-side comparison with ~5e-4 noise
    # that the deployed ONNX MatMul (true fp32) does not have. CPU matmul is
    # strict fp32, so this measures the real functional difference.
    import copy

    cpu_new = copy.deepcopy(encoder).cpu().eval()
    cpu_old = copy.deepcopy(encoder).cpu().eval()
    cpu_old.adapt_pool = copy.deepcopy(old_pool).cpu()
    x = torch.rand(4, encoder.frame_channels * encoder.n_frames, img_h, img_w)
    err = float((cpu_new(x) - cpu_old(x)).abs().max())
  if err > 1e-5:
    encoder.adapt_pool = old_pool
    raise RuntimeError(
      f"exact-pool swap changed encoder output by {err:.2e} (strict fp32); "
      "refusing to export"
    )
  print(f"[MAEEncoder] adapt_pool -> exact matmul ({in_hw}->{encoder.patch_grid}), "
        f"max|diff| {err:.2e} (strict-fp32 CPU check)")


class MAEEncoder(nn.Module):
  """Pretrained MAE trunk -> `features_dim` vector. Exposes the CNNModel contract."""

  def __init__(
    self,
    input_channels: int = 3,
    features_dim: int = 256,
    *,
    patch_grid: int | tuple[int, int] = 12,
    encoder_dim: int = 256,
    depth: int = 4,
    heads: int = 4,
    mlp_dim: int = 512,
    dim_head: int = 64,
    frame_stack: int = 1,
    checkpoint_path: str | Path | None = None,
    freeze: bool = True,
  ):
    super().__init__()
    if input_channels % frame_stack != 0:
      raise ValueError(
        f"input_channels {input_channels} not divisible by frame_stack {frame_stack}"
      )
    self.n_frames = frame_stack
    self.frame_channels = input_channels // frame_stack
    # (rows, cols). The side-by-side arm is 12x24 over an 84x168 frame; the
    # single-camera arms stay 12x12. Stored as a pair so num_patches and the
    # pos_embedding length follow from one place.
    self.patch_grid = (
      (patch_grid, patch_grid) if isinstance(patch_grid, int) else tuple(patch_grid)
    )
    self.num_patches = self.patch_grid[0] * self.patch_grid[1]

    self.early_conv = _EarlyCNNSpatial(self.frame_channels, encoder_dim)
    # 84x84 leaves a 10x10 map, which this UPSAMPLES to the 12x12 patch grid.
    # Alpha3's quirk; kept.
    self.adapt_pool = nn.AdaptiveAvgPool2d(self.patch_grid)
    self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim))
    self.transformer = Transformer(
      encoder_dim, depth, heads, dim_head, mlp_dim, dropout=0.0
    )
    self.proj = nn.Linear(encoder_dim, features_dim)

    self.output_dim = features_dim
    self.output_channels = None  # flattened output, as CNNModel requires

    if checkpoint_path is not None:
      self.load_pretrained(checkpoint_path)
    if freeze:
      self.freeze_trunk()

  def freeze_trunk(self) -> None:
    """requires_grad=False on the pretrained parts; `proj` stays trainable."""
    for p in self.early_conv.parameters():
      p.requires_grad = False
    self.pos_embedding.requires_grad = False
    for p in self.transformer.parameters():
      p.requires_grad = False

  def load_pretrained(self, path: str | Path) -> None:
    """Load a DualOutputMAE checkpoint, keeping only the encoder-side tensors.

    Pretraining saves a flat state_dict whose encoder lives under
    `early_conv.*`, `pos_embedding` and `encoder.transformer.*`; everything else
    is decoder and is dropped.
    """
    path = Path(path)
    if not path.is_absolute():
      path = Path(__file__).resolve().parents[3] / path
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck and "enc_opt" in ck:
      ck = ck["model"]  # a resume.pt rather than a bare state_dict

    mapped: dict[str, torch.Tensor] = {}
    for k, v in ck.items():
      if k.startswith("early_conv.") or k == "pos_embedding":
        mapped[k] = v
      elif k.startswith("encoder.transformer."):
        mapped[k[len("encoder.") :]] = v

    own = self.state_dict()
    usable = {k: v for k, v in mapped.items() if k in own and own[k].shape == v.shape}
    missing = [
      k for k in own
      if k not in usable and not k.startswith("proj.")
    ]
    shape_clash = [
      k for k, v in mapped.items() if k in own and own[k].shape != v.shape
    ]
    if shape_clash:
      raise ValueError(
        f"checkpoint/model shape mismatch for {shape_clash[:4]} -- the encoder "
        f"geometry does not match what was pretrained ({path})"
      )
    if missing:
      raise ValueError(
        f"checkpoint {path} is missing encoder tensors {missing[:6]} "
        f"({len(missing)} total); refusing to run a partly-random 'pretrained' "
        f"encoder"
      )
    self.load_state_dict(usable, strict=False)
    print(
      f"[MAEEncoder] loaded {len(usable)} pretrained tensors from {path} "
      f"(proj left at init, as in muse)"
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b * self.n_frames, self.frame_channels, h, w)
    x = self.early_conv(x)
    x = self.adapt_pool(x)
    x = x.flatten(2).transpose(1, 2)          # (B*F, N, D)
    x = x + self.pos_embedding
    x = self.transformer(x)                    # no masking at RL time
    x = x.mean(dim=1)                          # mean-pool over tokens
    x = x.view(b, self.n_frames, -1).mean(dim=1)
    return self.proj(x)


class MAEPretrainedModel(CNNModel):
  """Drop-in CNNModel using frozen Alpha3 MAE encoders."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, Any] | None = None,
    cnns: nn.ModuleDict | None = None,
    hidden_dims: tuple[int] | list[int] = [256, 256],  # noqa: B006
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    cnn_cfg = dict(cnn_cfg or {})
    features_dim = int(cnn_cfg.pop("features_dim", 256))
    checkpoint_path = cnn_cfg.pop("checkpoint_path", None)
    freeze = bool(cnn_cfg.pop("freeze", True))
    encoder_dim = int(cnn_cfg.pop("encoder_dim", 256))
    _pg = cnn_cfg.pop("patch_grid", 12)
    if isinstance(_pg, str):
      _pg = [int(v) for v in _pg.split(",")]
    patch_grid = int(_pg) if isinstance(_pg, (int, float)) else tuple(int(v) for v in _pg)
    depth = int(cnn_cfg.pop("depth", 4))
    heads = int(cnn_cfg.pop("heads", 4))
    mlp_dim = int(cnn_cfg.pop("mlp_dim", 512))
    frame_stack = int(cnn_cfg.pop("frame_stack", 1))

    self._get_obs_dim(obs, obs_groups, obs_set)

    if cnns is not None:
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError(
          "The 2D observations must be identical for all models sharing CNN encoders."
        )
      print("Sharing MAE encoders between models.")
      _cnns = cnns
    else:
      _cnns = {
        group: MAEEncoder(
          input_channels=self.obs_channels_2d[idx],
          features_dim=features_dim,
          patch_grid=patch_grid,
          encoder_dim=encoder_dim,
          depth=depth,
          heads=heads,
          mlp_dim=mlp_dim,
          frame_stack=frame_stack,
          checkpoint_path=checkpoint_path,
          freeze=freeze,
        )
        for idx, group in enumerate(self.obs_groups_2d)
      }

    self.cnn_latent_dim = sum(int(c.output_dim) for c in _cnns.values())

    MLPModel.__init__(
      self,
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    if isinstance(_cnns, nn.ModuleDict):
      self.cnns = _cnns
    else:
      self.cnns = nn.ModuleDict(_cnns)
