from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def distance_to_ground(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance to ground in world frame."""
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids, 2]


def noisy_camera_rgb(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  brightness_range: tuple[float, float] = (-0.1, 0.1),
  contrast_range: tuple[float, float] = (0.8, 1.2),
  noise_std: float = 0.02,
) -> torch.Tensor:
  """RGB observation with noise augmentation for sim2real transfer (B, C, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  rgb_data = sensor.data.rgb  # (B, H, W, 3)
  assert rgb_data is not None, f"Camera '{sensor_name}' has no RGB data"
  rgb_data = rgb_data.permute(0, 3, 1, 2).float() / 255.0  # (B, 3, H, W)

  # Apply random brightness shift per sample
  brightness = torch.empty(rgb_data.shape[0], 1, 1, 1, device=rgb_data.device)
  brightness.uniform_(brightness_range[0], brightness_range[1])
  rgb_data = rgb_data + brightness

  # Apply random contrast scaling per sample
  contrast = torch.empty(rgb_data.shape[0], 1, 1, 1, device=rgb_data.device)
  contrast.uniform_(contrast_range[0], contrast_range[1])
  mean = rgb_data.mean(dim=(1, 2, 3), keepdim=True)
  rgb_data = (rgb_data - mean) * contrast + mean

  # Apply Gaussian noise
  rgb_data = rgb_data + torch.randn_like(rgb_data) * noise_std

  return torch.clamp(rgb_data, 0.0, 1.0)
