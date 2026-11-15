"""Command for the reach task: a randomly sampled end-effector target pose.

Mirrors Orbit-Surgical's ``UniformPoseCommand`` for the PSM reach task, but built on
mjlab's ``CommandTerm`` (the same base class our push/lift commands use). The target is
sampled in a box in the robot base frame each resampling interval; a mocap marker follows
it so it is visible, and the EE-to-target error is tracked as a metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import CommandTermCfg
from mjlab.managers.command_manager import CommandTerm
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ReachCommand(CommandTerm):
  """Sample a target EE position and expose it; a mocap marker tracks it."""

  cfg: ReachCommandCfg

  def __init__(self, cfg: ReachCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.target: Entity = env.scene[cfg.target_name]
    # Resolve the EE site once (same mechanism SceneEntityCfg uses for reward/obs terms).
    site_ids, _ = self.robot.find_sites(cfg.ee_site_name)
    self._ee_site_id = site_ids[0]

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _ee_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self._ee_site_id]

  def _update_metrics(self) -> None:
    position_error = torch.norm(self.target_pos - self._ee_pos_w(), dim=-1)
    at_goal = (position_error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    self.metrics["position_error"] = position_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["position_error"] < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0

    r = self.cfg.target_pose_range
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    offset = sample_uniform(lower, upper, (n, 3), device=self.device)
    # Sample relative to the robot's ACTUAL base position, so the target lands in the tool's
    # workspace regardless of whether envs are tiled by env_origins (they are for some tasks,
    # not others). Ranges are therefore base-frame offsets.
    base_pos = self.robot.data.root_link_pos_w[env_ids]
    self.target_pos[env_ids] = base_pos + offset

    # Move the mocap marker to the new target (pose is (x,y,z, qw,qx,qy,qz)).
    pose = torch.zeros(n, 7, device=self.device)
    pose[:, :3] = self.target_pos[env_ids]
    pose[:, 3] = 1.0  # identity quaternion
    self.target.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass


@dataclass
class ReachCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  target_name: str = "target"
  ee_site_name: str = "tool_tip"
  success_threshold: float = 0.005

  @dataclass
  class TargetPoseRangeCfg:
    """Target position box, as an offset from the robot base position (meters)."""

    x: tuple[float, float] = (-0.05, 0.05)
    y: tuple[float, float] = (-0.13, -0.03)
    z: tuple[float, float] = (0.0, 0.08)

  target_pose_range: TargetPoseRangeCfg = field(default_factory=TargetPoseRangeCfg)

  def build(self, env: ManagerBasedRlEnv) -> ReachCommand:
    return ReachCommand(self, env)
