"""ONNX metadata for surgical_lab policies.

mjlab's :func:`~mjlab.rl.exporter_utils.get_base_metadata` describes a joint-position
policy: it records per-joint action scale, joint order, stiffness and damping, and
assumes the action term is named ``joint_pos``. A Cartesian policy emits end-effector
deltas, so that description does not apply and mjlab raises ``KeyError: 'joint_pos'``.

This module records what a deployment target actually needs to reproduce the Cartesian
control loop on hardware: the observation layout, the control rate, the action contract
(scale, frame, integration and anti-windup semantics), and the PSM kinematic constants
that the FK/IK were trained against. A real PSM whose DH parameters differ from these
will not reproduce sim behaviour, so they are pinned here rather than assumed.
"""

from __future__ import annotations

import inspect

import torch
from mjlab.envs import ManagerBasedRlEnv

from surgical_lab.mdp.actions.cartesian_actions import CartesianDeltaAction
from surgical_lab.robots.psm import psm_kinematics

# The action term integrates the EE target once per control step, in the target's own
# frame, and clamps it to lead the measured EE by at most max_target_lead. A deployment
# that re-derives the target from measured FK every step, or that integrates at the
# servo rate instead of the policy rate, will not match the trained dynamics.
_TARGET_INTEGRATION = (
  "p_target <- p_target + R_target @ (action[:3] * pos_scale), once per control step; "
  "then clamped so ||p_target - fk(q_measured)|| <= max_target_lead; "
  "q_target = ik(T_target); target re-synced to fk(q) on episode reset"
)


def _kinematic_constants() -> dict[str, float]:
  """Pin the DH constants the trained FK/IK were built against."""
  params = inspect.signature(psm_kinematics.compute_psm_fk).parameters
  return {
    name: float(p.default)
    for name, p in params.items()
    if name != "joint_pos" and p.default is not inspect.Parameter.empty
  }


def _observation_metadata(env: ManagerBasedRlEnv) -> dict:
  """Observation layout for the actor group, mirroring mjlab's base metadata."""
  term_scale: list = []
  term_flatten_history_dim: list = []
  term_history_length: list = []
  term_clip: list = []
  names = env.observation_manager.active_terms["actor"]

  for name in names:
    cfg = env.observation_manager.get_term_cfg("actor", name)

    if cfg.scale is None:
      term_scale.append(1.0)
    else:
      scale = cfg.scale
      term_scale.append(
        scale.cpu().tolist() if isinstance(scale, torch.Tensor) else scale
      )

    term_clip.append(
      [float("-inf"), float("inf")] if cfg.clip is None else list(cfg.clip)
    )
    term_flatten_history_dim.append(cfg.flatten_history_dim)
    term_history_length.append(cfg.history_length)

  return {
    "observation_names": names,
    "observation_terms_scale": term_scale,
    "observation_terms_flatten_history_dim": term_flatten_history_dim,
    "observation_terms_history_length": term_history_length,
    "observation_terms_clip": term_clip,
  }


def get_diffik_metadata(
  env: ManagerBasedRlEnv,
  run_path: str,
  action_term_name: str = "ee_delta",
  gripper_term_name: str = "gripper",
  clip_actions: float | None = None,
) -> dict[str, list | str | float]:
  """Metadata for a differential-IK end-effector delta policy (+ optional gripper).

  The policy emits EE position deltas in [-1, 1] that only mean something once scaled by
  `delta_pos_scale` and resolved against the arm's Jacobian, so the deployment needs the
  scale, the frame, the joint order and the control rate. Same contract as
  get_cartesian_metadata, but for mjlab's DifferentialIKAction rather than our analytic
  PSM term.
  """
  term = env.action_manager.get_term(action_term_name)
  cfg = term.cfg
  robot = env.scene["robot"]
  joint_limits = robot.data.joint_pos_limits[0].cpu()

  metadata: dict[str, list | str | float] = {
    "run_path": run_path,
    "policy_type": "differential_ik_delta",
    # Control loop.
    "control_dt": env.step_dt,
    "control_freq_hz": 1.0 / env.step_dt,
    "decimation": env.cfg.decimation,
    "physics_dt": env.cfg.sim.mujoco.timestep,
    # Action contract.
    "action_dim": env.action_manager.total_action_dim,
    "action_frame": f"{cfg.frame_type}:{cfg.frame_name}",
    "action_relative": cfg.use_relative_mode,
    "action_delta_pos_scale": cfg.delta_pos_scale,
    "action_delta_ori_scale": cfg.delta_ori_scale,
    "action_position_weight": cfg.position_weight,
    "action_orientation_weight": cfg.orientation_weight,
    "ik_damping": cfg.damping,
    "ik_max_dq": cfg.max_dq,
    "action_clip": "none" if clip_actions is None else clip_actions,
    "solver": "damped_least_squares (mjlab DifferentialIKAction)",
    # Robot.
    "arm_actuator_names": list(cfg.actuator_names),
    "joint_names": list(robot.joint_names),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "joint_pos_lower_limits": joint_limits[:, 0].tolist(),
    "joint_pos_upper_limits": joint_limits[:, 1].tolist(),
  }

  if gripper_term_name in env.action_manager.active_terms:
    gcfg = env.action_manager.get_term(gripper_term_name).cfg
    metadata["gripper_type"] = "binary"
    metadata["gripper_open_command"] = str(gcfg.open_command_expr)
    metadata["gripper_close_command"] = str(gcfg.close_command_expr)

  metadata.update(_observation_metadata(env))
  return metadata


def get_cartesian_metadata(
  env: ManagerBasedRlEnv,
  run_path: str,
  action_term_name: str = "cartesian_delta",
  clip_actions: float | None = None,
) -> dict[str, list | str | float]:
  """Metadata for a Cartesian end-effector delta policy.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    action_term_name: Name of the Cartesian action term in the action manager.
    clip_actions: Symmetric action clip applied by the RL runner, if any. Not part of
      the env, so the caller must pass it from the agent config.
  """
  term = env.action_manager.get_term(action_term_name)
  if not isinstance(term, CartesianDeltaAction):
    raise TypeError(
      f"Action term '{action_term_name}' is {type(term).__name__}, "
      "expected CartesianDeltaAction."
    )
  cfg = term.cfg
  robot = env.scene["robot"]

  # Joints in the order the FK/IK index them; the policy's joint targets are
  # meaningless if the deployment applies them in a different order. Only the names
  # travel: the integer ids are mjlab-internal and do not address a real PSM.
  arm_joint_names = term.joint_names
  joint_limits = robot.data.joint_pos_limits[0].cpu()

  metadata: dict[str, list | str | float] = {
    "run_path": run_path,
    "policy_type": "cartesian_delta",
    # Control loop.
    "control_dt": env.step_dt,
    "control_freq_hz": 1.0 / env.step_dt,
    "decimation": env.cfg.decimation,
    "physics_dt": env.cfg.sim.mujoco.timestep,
    # Action contract.
    "action_dim": term.action_dim,
    "action_frame": "end_effector",
    "action_position_only": cfg.position_only,
    "action_pos_scale": cfg.pos_scale,
    "action_rot_scale": cfg.rot_scale,
    "action_max_target_lead": cfg.max_target_lead,
    "action_clip": "none" if clip_actions is None else clip_actions,
    "target_integration": _TARGET_INTEGRATION,
    # Kinematic chain. IK maps the EE target to these joints, in this order.
    "arm_joint_names": arm_joint_names,
    # compute_psm_fk/ik control the tool control point (l_yaw2ctrlpnt = 0.0102), which
    # matches the model's `tool_tip` site to ~5 um -- NOT `tool_midpoint` (off by ~2.9 mm).
    # A deployment that servos tool_midpoint would place the EE ~3 mm from where the policy
    # expects, so the frame must be named exactly.
    "ee_frame": "tool_tip",
    "ik_solver": "surgical_lab.robots.psm.psm_kinematics:compute_psm_ik",
    "fk_solver": "surgical_lab.robots.psm.psm_kinematics:compute_psm_fk",
    # Robot state.
    "joint_names": list(robot.joint_names),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "joint_pos_lower_limits": joint_limits[:, 0].tolist(),
    "joint_pos_upper_limits": joint_limits[:, 1].tolist(),
    "command_names": list(env.command_manager.active_terms),
  }
  metadata.update(_kinematic_constants())
  metadata.update(_observation_metadata(env))
  return metadata
