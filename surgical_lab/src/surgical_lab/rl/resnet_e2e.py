"""ResNet18Lite CNN model for rsl-rl, ported from muse's torch_networks.py.

Exact architecture used by the Unity `ABL_a3b_resnet_e2e_chain` policy that
reached 80% real success: stem (7x7/s2 conv + BN + ReLU + 3x3/s2 maxpool),
three BasicBlock stages (32, 64, 128 channels, stride 1/2/2), global average
pool, fc -> features_dim. Trained END-TO-END by PPO (no pretraining). Inputs
are RGB in [0, 1].

Integrates with mjlab's rsl-rl stack the same way SpatialSoftmaxCNNModel does:
each 2D observation group gets an encoder; latents concat with the (constant)
1D branch and feed the MLP head.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict


class BasicBlock(nn.Module):
  def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
    super().__init__()
    self.c1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
    self.b1 = nn.BatchNorm2d(out_ch)
    self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
    self.b2 = nn.BatchNorm2d(out_ch)
    self.act = nn.ReLU(inplace=True)
    self.down: nn.Module | None = None
    if stride != 1 or in_ch != out_ch:
      self.down = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    idt = x if self.down is None else self.down(x)
    x = self.act(self.b1(self.c1(x)))
    x = self.b2(self.c2(x))
    return self.act(x + idt)


class ResNet18Lite(nn.Module):
  """muse ResNet18Lite; exposes output_dim/output_channels like mjlab CNNs."""

  def __init__(self, input_channels: int = 3, features_dim: int = 256):
    super().__init__()
    self.stem = nn.Sequential(
      nn.Conv2d(input_channels, 32, 7, 2, 3, bias=False),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(3, 2, 1),
    )
    self.l1 = self._layer(32, 32, 2, 1)
    self.l2 = self._layer(32, 64, 2, 2)
    self.l3 = self._layer(64, 128, 2, 2)
    self.avg = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(128, features_dim)
    self.output_dim = features_dim
    self.output_channels = None  # flattened output, as CNNModel requires

  @staticmethod
  def _layer(i: int, o: int, n: int, s: int) -> nn.Sequential:
    return nn.Sequential(
      *([BasicBlock(i, o, s)] + [BasicBlock(o, o) for _ in range(n - 1)])
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # inputs already in [0, 1]
    x = self.stem(x)
    x = self.l1(x)
    x = self.l2(x)
    x = self.l3(x)
    return self.fc(self.avg(x).flatten(1))


class ResNetE2EModel(CNNModel):
  """Drop-in CNNModel using ResNet18Lite encoders (muse a3b recipe)."""

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

    self._get_obs_dim(obs, obs_groups, obs_set)

    if cnns is not None:
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError(
          "The 2D observations must be identical for all models sharing CNN encoders."
        )
      print("Sharing ResNet encoders between models.")
      _cnns = cnns
    else:
      _cnns = {
        group: ResNet18Lite(
          input_channels=self.obs_channels_2d[idx], features_dim=features_dim
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
