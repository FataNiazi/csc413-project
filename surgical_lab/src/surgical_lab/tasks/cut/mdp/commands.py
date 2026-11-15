from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CutCommand(CommandTerm):
  """Command that manages cutting an object at a target location."""

  cfg: CutCommandCfg

  def __init__(self, cfg: CutCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.entity_name]
    self.target: Entity = env.scene[cfg.target_name]

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    object_pos_w = self.object.data.root_link_pos_w

    error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    at_goal = (error < self.cfg.success_threshold).float()

    self.episode_success = torch.maximum(self.episode_success, at_goal)

    self.metrics["position_error"] = error
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    position_error = self.metrics["position_error"]
    return position_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    self.episode_success[env_ids] = 0.0

    # Sample target pose
    g = self.cfg.target_pose_range
    target_lower = torch.tensor([g.x[0], g.y[0], g.z[0]], device=self.device)
    target_upper = torch.tensor([g.x[1], g.y[1], g.z[1]], device=self.device)
    target_pos = sample_uniform(target_lower, target_upper, (n, 3), device=self.device)
    target_pos = target_pos + self._env.scene.env_origins[env_ids]
    self.target_pos[env_ids] = target_pos

    target_yaw = sample_uniform(g.yaw[0], g.yaw[1], (n,), device=self.device)
    target_quat = quat_from_euler_xyz(
      torch.zeros(n, device=self.device),
      torch.zeros(n, device=self.device),
      target_yaw,
    )
    target_pose = torch.cat([target_pos, target_quat], dim=-1)
    self.target.write_mocap_pose_to_sim(target_pose, env_ids=env_ids)

    # Sample object pose
    t = self.cfg.object_pose_range
    object_lower = torch.tensor([t.x[0], t.y[0], t.z[0]], device=self.device)
    object_upper = torch.tensor([t.x[1], t.y[1], t.z[1]], device=self.device)
    object_pos = sample_uniform(object_lower, object_upper, (n, 3), device=self.device)
    object_pos = object_pos + self._env.scene.env_origins[env_ids]

    object_yaw = sample_uniform(t.yaw[0], t.yaw[1], (n,), device=self.device)
    object_quat = quat_from_euler_xyz(
      torch.zeros(n, device=self.device),
      torch.zeros(n, device=self.device),
      object_yaw,
    )
    object_pose = torch.cat([object_pos, object_quat], dim=-1)
    velocity = torch.zeros(n, 6, device=self.device)

    self.object.write_root_link_pose_to_sim(object_pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class CutCommandCfg(CommandTermCfg):
  """Configuration for cut command."""

  entity_name: str = "object"
  target_name: str = "target"
  success_threshold: float = 0.02

  @dataclass
  class TargetPoseRangeCfg:
    x: tuple[float, float] = (-0.1, 0.1)
    y: tuple[float, float] = (-0.05, 0.1)
    z: tuple[float, float] = (0.001, 0.001)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  target_pose_range: TargetPoseRangeCfg = field(default_factory=TargetPoseRangeCfg)

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (-0.1, 0.1)
    y: tuple[float, float] = (-0.05, 0.1)
    z: tuple[float, float] = (0.005, 0.005)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  def build(self, env: ManagerBasedRlEnv) -> CutCommand:
    return CutCommand(self, env)
