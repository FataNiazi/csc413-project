"""Rewards for tissue retraction: reach the corner, grasp it, drag it to the goal."""

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


def _goal_w(env: ManagerBasedRlEnv, goal_pos) -> torch.Tensor:
  goal = torch.tensor(goal_pos, device=env.device).expand(env.num_envs, 3)
  return goal + env.scene.env_origins


def staged_retract_reward(
  env: ManagerBasedRlEnv,
  vertex_id: int,
  goal_pos: tuple[float, float, float],
  reaching_std: float,
  dragging_std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reach the corner, then drag it to the goal.

  Staged, so dragging only pays once the gripper is at the corner. Without the gate the
  policy is rewarded for the corner drifting goalward on its own.
  """
  corner = flex_vertex_pos_w(env, vertex_id)
  ee = _ee_pos_w(env, asset_cfg)
  goal = _goal_w(env, goal_pos)

  reach_err = torch.sum(torch.square(ee - corner), dim=-1)
  reaching = torch.exp(-reach_err / reaching_std**2)

  drag_err = torch.sum(torch.square(corner - goal), dim=-1)
  dragging = torch.exp(-drag_err / dragging_std**2)

  return reaching * (1.0 + dragging)


def grasp_reward(
  env: ManagerBasedRlEnv,
  vertex_id: int,
  distance_threshold: float,
  closed_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Pay for closing the gripper while it is actually at the corner."""
  robot: Entity = env.scene[asset_cfg.name]
  corner = flex_vertex_pos_w(env, vertex_id)
  ee = _ee_pos_w(env, asset_cfg)

  near = torch.norm(ee - corner, dim=-1) < distance_threshold
  finger = robot.data.joint_pos[:, asset_cfg.joint_ids].mean(dim=-1)
  closed = finger < closed_threshold
  return (near & closed).float()


def goal_reward(
  env: ManagerBasedRlEnv,
  vertex_id: int,
  goal_pos: tuple[float, float, float],
  threshold: float,
) -> torch.Tensor:
  """Sparse success: the corner is at the retraction endpoint."""
  corner = flex_vertex_pos_w(env, vertex_id)
  goal = _goal_w(env, goal_pos)
  return (torch.norm(corner - goal, dim=-1) < threshold).float()


def tissue_lifted_off_table(
  env: ManagerBasedRlEnv, vertex_id: int, height: float
) -> torch.Tensor:
  """Shaping: the grasp corner is off the table (retraction implies lifting it clear)."""
  corner_z = flex_vertex_pos_w(env, vertex_id)[:, 2] - env.scene.env_origins[:, 2]
  return torch.clamp(corner_z / height, 0.0, 1.0)
