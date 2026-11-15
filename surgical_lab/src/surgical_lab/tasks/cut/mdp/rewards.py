from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from surgical_lab.tasks.cut.mdp.commands import CutCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def cutting_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  cutting_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for reaching the object and cutting at the target."""
  command = cast(CutCommand, env.command_manager.get_term(command_name))
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]

  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  object_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos

  reach_error = torch.sum(torch.square(ee_pos_w - object_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)

  cut_error = torch.sum(torch.square(goal_pos_w - object_pos_w), dim=-1)
  cutting = torch.exp(-cut_error / cutting_std**2)

  return reaching * (1.0 + cutting)


def goal_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  threshold: float = 0.01,
) -> torch.Tensor:
  """Binary reward for successful cut at the goal position."""
  command = cast(CutCommand, env.command_manager.get_term(command_name))
  obj: Entity = env.scene[object_name]

  object_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos

  error = torch.norm(object_pos_w - goal_pos_w, dim=-1)
  return (error < threshold).float()
