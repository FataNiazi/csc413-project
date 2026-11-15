from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

EPS = 1e-8


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
  angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp(min=EPS)
  axis = axis_angle / angle
  x, y, z = axis.unbind(-1)
  zeros = torch.zeros_like(x)
  K = torch.stack(
    (
      torch.stack((zeros, -z, y), dim=-1),
      torch.stack((z, zeros, -x), dim=-1),
      torch.stack((-y, x, zeros), dim=-1),
    ),
    dim=-2,
  )
  eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(
    *axis_angle.shape[:-1], 3, 3
  )
  angle = angle.squeeze(-1)
  sa = torch.sin(angle)[..., None, None]
  ca = torch.cos(angle)[..., None, None]
  return eye + sa * K + (1 - ca) * torch.matmul(K, K)


@dataclass(kw_only=True)
class CartesianDeltaActionCfg(ActionTermCfg):
  """Configuration for Cartesian delta action term."""

  actuator_names: tuple[str, ...]
  """Tuple of actuator names or regex expressions that the action will be mapped to."""
  position_only: bool = True
  """Whether to only apply the position component of the action."""
  fk_fn: Callable
  """Function to compute the forward kinematics of the robot."""
  ik_fn: Callable
  """Function to compute the inverse kinematics of the robot."""
  pos_scale: float = 0.05
  """Scale factor for the position component of the action."""
  rot_scale: float = 0.5
  """Scale factor for the rotation component of the action."""
  max_target_lead: float | None = 0.02
  """Max distance (m) the integrated EE target may lead the actual EE pose.

  ``None`` disables the clamp. The real-robot increment pipeline
  (medrct_onnx -> dvrk_planning IncrementTeleopController) has no equivalent
  anti-windup, so policies meant for direct transfer train without it so the
  deployed dynamics match training.
  """
  min_target_z: float | None = None
  """Hard floor on the integrated EE target's z, in the FK (robot base) frame.

  A virtual fixture. The target is an integrator, so clamping the STORED target
  -- rather than the incoming delta -- gives exactly the behaviour a safety
  limit should have: commands to go down saturate and stop, commands to go up
  take effect immediately from where the target actually is, and nothing winds
  up in between waiting to be undone.

  On the real arm this is set by a calibration step: jog the tool down until it
  just grazes the floor, record that z, and enter it before running inference.
  Training with the same clamp means the deployed dynamics match, and it means
  the policy never has to perceive its own height to stay safe -- which matters
  because it cannot: the tip moves only 0.344 px per mm at 84 px, so one pixel
  is 2.9 mm and the safe band is a few mm (docs/tissue_floor_safety.md).

  ``None`` disables it, which is the default -- pushblock and liftpeg are
  unaffected.
  """
  frame: str = "end_effector"
  """Frame the position delta is expressed in: "end_effector" or "base".

  The real-robot pipeline integrates increments in the robot BASE frame, so
  direct-transfer policies must train with frame="base". "end_effector" rotates
  the delta by the current target orientation (the original behaviour).
  """
  position_axes: tuple[int, ...] = (0, 1, 2)
  """Which position axes the action drives; the rest hold their target value.

  The deployed pushblock pipeline exposes exactly two continuous actions (px at
  index 0, pz at index 1 -- planar pushing), so a direct-transfer policy trains
  with position_axes=(0, 1): a 2-dim action driving the two horizontal base
  axes, vertical target frozen. Requires position_only=True when not (0, 1, 2).
  """
  extra_actuator_names: tuple[str, ...] = ()
  """Tuple of extra actuator names or regex expressions that the action will be mapped to."""
  extra_scale: float = 0.04
  """Scale factor for the extra component of the action."""
  extra_offset: float = 0.02
  """Offset for the extra component of the action."""

  def build(self, env: ManagerBasedRlEnv) -> CartesianDeltaAction:
    return CartesianDeltaAction(self, env)


class CartesianDeltaAction(ActionTerm):
  """Action term that applies a Cartesian delta to the robot."""

  _entity: Entity

  def __init__(self, cfg: CartesianDeltaActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._entity.find_joints_by_actuator_names(
      cfg.actuator_names
    )
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names

    self._num_joints = len(joint_ids)
    self._position_axes = tuple(cfg.position_axes)
    if self._position_axes != (0, 1, 2) and not cfg.position_only:
      raise ValueError("position_axes subsets require position_only=True.")
    self._action_dim = len(self._position_axes) if cfg.position_only else 6
    self._position_only = cfg.position_only

    extra_joint_ids, extra_joint_names = self._entity.find_joints_by_actuator_names(
      cfg.extra_actuator_names
    )
    self._extra_joint_ids = torch.tensor(
      extra_joint_ids, device=self.device, dtype=torch.long
    )
    self._extra_joint_names = extra_joint_names

    self._num_extra_joints = len(extra_joint_ids)
    self._action_dim += self._num_extra_joints

    self._fk_fn = cfg.fk_fn
    self._ik_fn = cfg.ik_fn

    self._pos_scale = cfg.pos_scale
    self._rot_scale = cfg.rot_scale
    self._max_target_lead = cfg.max_target_lead
    self._min_target_z = cfg.min_target_z
    if cfg.frame not in ("end_effector", "base"):
      raise ValueError(f"Unknown frame '{cfg.frame}'; use 'end_effector' or 'base'.")
    self._base_frame_deltas = cfg.frame == "base"
    self._extra_scale = cfg.extra_scale
    self._extra_offset = cfg.extra_offset

    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    # Full 3-vector position delta; reduced position_axes scatter into it.
    self._pos_delta3 = torch.zeros(self.num_envs, 3, device=self.device)
    self._axes_index = torch.tensor(
      self._position_axes, device=self.device, dtype=torch.long
    )

    self._ee_target_tf = (
      torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
    )
    self._q_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def joint_names(self) -> list[str]:
    """Arm joints, in the order the FK/IK index them."""
    return list(self._joint_names)

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint_ids

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    n_pos = len(self._position_axes)
    self._processed_actions[..., 0:n_pos] = (
      self._raw_actions[..., 0:n_pos] * self._pos_scale
    )
    self._pos_delta3.zero_()
    self._pos_delta3[..., self._axes_index] = self._processed_actions[..., 0:n_pos]

    if not self._position_only:
      self._processed_actions[..., 3:6] = self._raw_actions[..., 3:6] * self._rot_scale
    if self._num_extra_joints:
      self._processed_actions[..., self._action_dim - self._num_extra_joints :] = (
        self._raw_actions[..., self._action_dim - self._num_extra_joints :]
        * self._extra_scale
        + self._extra_offset
      )

    self._integrate_target()

    q_arm = self._entity.data.joint_pos[:, self._joint_ids]
    self._q_target = self._ik_fn(self._ee_target_tf, q_arm[..., -1], q_arm)

  def _integrate_target(self) -> None:
    """Advance the stored EE target by one policy-step delta.

    The delta is per policy step, so this must not run inside the decimation loop.
    """
    pos_delta = self._pos_delta3

    # Use stored target transform to prevent drift
    R_curr = self._ee_target_tf[..., :3, :3]
    p_curr = self._ee_target_tf[..., :3, 3]

    if self._base_frame_deltas:
      # Deltas are already in the base frame (the real-robot increment
      # convention): integrate directly, independent of tool orientation.
      pos_world = pos_delta
    else:
      pos_world = torch.einsum("...ij,...j->...i", R_curr, pos_delta)
    p_new = p_curr + pos_world

    if not self._position_only:
      rot_delta = self._processed_actions[..., 3:6]
      R_delta = _axis_angle_to_matrix(rot_delta)
      R_new = torch.matmul(R_curr, R_delta)
    else:
      R_new = R_curr

    # Anti-windup: the target is an integrator, so if the arm cannot keep up it
    # would otherwise run away to an unreachable pose and never recover. Keep it
    # within one step's reach of where the arm actually is. Disabled (None) for
    # direct-transfer policies: the real increment pipeline has no equivalent,
    # and training with a clamp the deployment lacks changes the dynamics.
    if self._max_target_lead is not None:
      q_arm = self._entity.data.joint_pos[:, self._joint_ids]
      p_actual = self._fk_fn(q_arm)[..., :3, 3]
      offset = p_new - p_actual
      dist = torch.linalg.norm(offset, dim=-1, keepdim=True)
      max_dist = self._max_target_lead
      p_new = torch.where(
        dist > max_dist, p_actual + offset * (max_dist / dist.clamp(min=EPS)), p_new
      )

    # Virtual fixture: never let the target descend past the calibrated floor.
    # Applied to the stored target, so it cannot accumulate below the limit.
    if self._min_target_z is not None:
      p_new = torch.cat(
        [p_new[..., :2], p_new[..., 2:3].clamp_min(self._min_target_z)], dim=-1
      )
    self._ee_target_tf[..., :3, 3] = p_new
    self._ee_target_tf[..., :3, :3] = R_new

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0

    if env_ids is None:
      env_ids = slice(None)

    q_arm = self._entity.data.joint_pos[env_ids][:, self._joint_ids]
    self._ee_target_tf[env_ids] = self._fk_fn(q_arm)
    self._q_target[env_ids] = q_arm

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._q_target, joint_ids=self._joint_ids)

    if self._num_extra_joints:
      self._entity.set_joint_position_target(
        self._processed_actions[..., self._action_dim - self._num_extra_joints :],
        joint_ids=self._extra_joint_ids,
      )
