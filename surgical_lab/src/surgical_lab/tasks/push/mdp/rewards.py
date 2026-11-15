from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from surgical_lab.tasks.push.mdp.commands import PushCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  pushing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for reaching and pushing the block to the goal."""
  command = cast(PushCommand, env.command_manager.get_term(command_name))
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]

  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  object_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos

  reach_error = torch.sum(torch.square(ee_pos_w - object_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)

  push_error = torch.sum(torch.square(goal_pos_w[:, :2] - object_pos_w[:, :2]), dim=-1)
  pushing = torch.exp(-push_error / pushing_std**2)

  return reaching * (1.0 + pushing)


def goal_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  threshold: float = 0.01,
) -> torch.Tensor:
  """Binary reward for block being at the goal position."""
  command = cast(PushCommand, env.command_manager.get_term(command_name))
  obj: Entity = env.scene[object_name]

  object_pos_w = obj.data.root_link_pos_w
  goal_pos_w = command.target_pos

  xy_error = torch.norm(object_pos_w[:, :2] - goal_pos_w[:, :2], dim=-1)
  return (xy_error < threshold).float()
