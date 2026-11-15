"""Reward functions for the lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  z_offset: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.

  Unlike the original reward function in mjlab, this reward function allows for a z-axis offset.
  This is useful for encouraging the robot to grasp the object lower in the z-axis.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  offset_target_pos = command.target_pos + torch.tensor(
    [0.0, 0.0, z_offset], device=env.device
  )
  position_error = torch.sum(torch.square(offset_target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
  z_offset: float = 0.0,
) -> torch.Tensor:
  """Reward for bringing the object to the target position.

  Unlike the original reward function in mjlab, this reward function allows for a z-axis offset.
  This is useful for encouraging the robot to grasp the object lower in the z-axis.
  """
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  offset_target_pos = command.target_pos + torch.tensor(
    [0.0, 0.0, z_offset], device=env.device
  )
  position_error = torch.sum(
    torch.square(offset_target_pos - obj.data.root_link_pos_w), dim=-1
  )
  return torch.exp(-position_error / std**2)


def not_lifted(
  env: ManagerBasedRlEnv,
  object_name: str,
  lift_threshold: float,
) -> torch.Tensor:
  """Penalty for the object not being lifted."""
  obj: Entity = env.scene[object_name]
  object_z = obj.data.root_link_pos_w[:, 2]
  return (object_z < lift_threshold).float()


def grasp_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  reaching_std: float,
  ee_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  jaw_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  object_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Reward for closing the jaw when near the object.

  Incentivizes proper grasping by rewarding jaw closure in proximity to the
  object, rather than entering a hollow object and opening the jaw to hook it.

  When ``object_cfg`` is provided with ``site_names``, the distance is measured
  to the specified site (e.g. a grasp target on the surface) instead of the
  object center.
  """
  robot: Entity = env.scene[ee_cfg.name]
  obj: Entity = env.scene[object_name]

  ee_pos_w = robot.data.site_pos_w[:, ee_cfg.site_ids].squeeze(1)

  if object_cfg is not None and object_cfg.site_names:
    target_pos_w = obj.data.site_pos_w[:, object_cfg.site_ids].squeeze(1)
  else:
    target_pos_w = obj.data.root_link_pos_w

  distance_sq = torch.sum(torch.square(ee_pos_w - target_pos_w), dim=-1)
  reaching = torch.exp(-distance_sq / reaching_std**2)

  # jaw range is [0, 0.5] on the PSM: 0 = closed, 0.5 = fully open
  jaw_pos = robot.data.joint_pos[:, jaw_cfg.joint_ids].squeeze(1)
  jaw_closed = 1.0 - torch.clamp(jaw_pos / 0.5, 0.0, 1.0)

  return reaching * jaw_closed


def open_jaw_lift_penalty(
  env: ManagerBasedRlEnv,
  object_name: str,
  lift_threshold: float,
  jaw_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for lifting the object while the jaw is open.

  Prevents the exploit of entering a hollow object's interior and opening the
  jaw to hook-lift it. Should be used with a negative reward weight.
  """
  robot: Entity = env.scene[jaw_cfg.name]
  obj: Entity = env.scene[object_name]

  object_z = obj.data.root_link_pos_w[:, 2]
  is_lifted = (object_z > lift_threshold).float()

  jaw_pos = robot.data.joint_pos[:, jaw_cfg.joint_ids].squeeze(1)
  jaw_open = torch.clamp(jaw_pos / 0.5, 0.0, 1.0)

  return is_lifted * jaw_open


def penetration_penalty(
  env: ManagerBasedRlEnv,
  object_name: str,
  interior_radius: float,
  ee_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for the end-effector being inside the hollow interior of an object.

  Computes the horizontal (xy) distance from the EE to the object center.
  When the EE is closer than ``interior_radius``, a penalty proportional to
  the penetration depth is returned. Should be used with a negative weight.
  """
  robot: Entity = env.scene[ee_cfg.name]
  obj: Entity = env.scene[object_name]

  ee_pos_w = robot.data.site_pos_w[:, ee_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w

  # Horizontal distance only (xy plane relative to object).
  diff_xy = ee_pos_w[:, :2] - obj_pos_w[:, :2]
  dist_xy = torch.norm(diff_xy, dim=-1)

  # Penalty when EE is inside the interior radius.
  penetration = torch.clamp(interior_radius - dist_xy, min=0.0) / interior_radius

  return penetration
