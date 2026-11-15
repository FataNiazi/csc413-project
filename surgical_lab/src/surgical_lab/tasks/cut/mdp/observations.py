from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from surgical_lab.tasks.cut.mdp.commands import CutCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def object_to_target_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from object to target in base frame."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, CutCommand):
    raise ValueError(f"Command {command_name} is not a CutCommand")

  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]

  object_pos_w = obj.data.root_link_pos_w
  target_pos_w = command.target_pos

  distance_vec_w = target_pos_w - object_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b
