"""Reward terms for the reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from surgical_lab.tasks.reach.mdp.commands import ReachCommand

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot", site_names=("tool_tip",))


def position_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gaussian kernel on the EE-to-target distance (1 at the target, ->0 far away)."""
  robot: Entity = env.scene[asset_cfg.name]
  command = cast(ReachCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  error = torch.sum(torch.square(command.target_pos - ee_pos_w), dim=-1)
  return torch.exp(-error / std**2)
