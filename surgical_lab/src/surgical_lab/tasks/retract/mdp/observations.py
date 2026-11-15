"""Observations for tissue retraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from surgical_lab.tasks.retract.mdp.cloth import flex_vertex_pos_w

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _ee_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def ee_to_corner(
  env: ManagerBasedRlEnv, vertex_id: int, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Vector from the gripper to the tissue's grasp corner."""
  return flex_vertex_pos_w(env, vertex_id) - _ee_pos_w(env, asset_cfg)


def corner_to_goal(
  env: ManagerBasedRlEnv, vertex_id: int, goal_pos: tuple[float, float, float]
) -> torch.Tensor:
  """Vector from the grasp corner to the retraction endpoint."""
  goal = torch.tensor(goal_pos, device=env.device).expand(env.num_envs, 3)
  return goal + env.scene.env_origins - flex_vertex_pos_w(env, vertex_id)


def corner_pos(
  env: ManagerBasedRlEnv, vertex_id: int
) -> torch.Tensor:
  """Grasp corner position, relative to the env origin."""
  return flex_vertex_pos_w(env, vertex_id) - env.scene.env_origins


def ee_pos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Gripper position, relative to the env origin."""
  return _ee_pos_w(env, asset_cfg) - env.scene.env_origins


def gripper_state(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Gripper opening (the finger joints)."""
  robot: Entity = env.scene[asset_cfg.name]
  return robot.data.joint_pos[:, asset_cfg.joint_ids]
