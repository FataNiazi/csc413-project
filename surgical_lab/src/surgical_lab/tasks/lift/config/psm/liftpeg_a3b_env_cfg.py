"""LiftPeg on the a3b recipe, built from measured Unity ground truth.

Composed from `psm_push_wide_a3b_env_cfg()` so the PSM, blue floor, DR pipeline
and PPO recipe stay literally the same objects as pushblock's. Everything
task-specific comes from `liftpeg_scene.py`, whose constants are traceable to
`docs/liftpeg_unity_ground_truth.md`.

The task is peg TRANSFER: a green triangular ring starts threaded on a yellow
post and must be lifted clear and carried to a fixed goal. Not the claw-machine
lift the v1-v4 attempts implemented -- those were built from a different task's
prefab that shares the Unity folder.
"""

from __future__ import annotations

import math
import os

# MUST precede the a3b import: A3B_UNITY_LAYOUT is read at a3b's module scope
# and defaults to "0", which silently selects the NON-Unity robot
# (get_psm_robot_cfg at HOME_KEYFRAME) instead of _unity_robot_cfg at
# PSM_BASE_POS, and also skips stripping the default sun and dimming the
# headlight. Liftpeg is a Unity replication unconditionally, so the flag is
# forced rather than inherited. This same default has now bitten three separate
# times on this task; do not remove.
os.environ.setdefault("A3B_UNITY_LAYOUT", "1")

import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as LR
from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b
from surgical_lab.tasks.lift.config.psm.liftpeg_grasp import (
  contact_both_jaws,
  quat_apply,
  _jaw_cmd,
)

# Home pose in OUR joint convention. Unity's values do not transfer -- the two
# URDFs parameterize the parallelogram differently -- so this is the IK solution
# for Unity's measured tool POSE (position 0.0000 mm, orientation 0.00000).
HOME_JOINTS = {
  # Tool at (ANCHOR_X, ANCHOR_Y, 72.98) mm -- pushblock's x/y anchor so the
  # real-robot home carries over, raised to Unity's liftpeg height. Solved in
  # the assembled env; it comes out as a PURE VERTICAL descent (every joint ~0)
  # because pushblock's anchor sits directly below the RCM.
  "outer_yaw": 0.0,
  "outer_pitch": 0.0,
  "main_insertion": 0.043267,   # qpos 0.254330 minus lower_tool's 0.211063
  "tool_roll": 0.0,
  "tool_pitch": 0.0,
  "tool_yaw": 0.0,
  "jaw": 0.0,
  "pitch_bottom_joint": 0.0,
  "pitch_top_joint": 0.0,
  "pitch_end_joint": 0.0,
  "pitch_front_joint": 0.0,
}

LIFT_TOL = 0.008

# Camera arm configuration. The shaped-vision run grasps well (held_frac 0.75)
# and cannot lift (2.3 mm) -- the signature of a view that resolves xy but not
# height. At 84x84 the ring's bar is 1.21 px and the whole grasp geometry is
# sub-2-pixel, so these are the three things worth varying.
CAM_RES = int(os.environ.get("LIFTPEG_CAM_RES", "84"))
# Side-camera fovy DR band, degrees. Default is the historical (58, 68); see the
# note on the camera_fovy event for the measured real value and why they differ.
_FOVY_DR = tuple(
  float(x) for x in os.environ.get("LIFTPEG_CAM_FOVY_DR", "58.0,68.0").split(",")
)
CAM_VIEW = os.environ.get("LIFTPEG_CAM_VIEW", "side")
"""side | head | multi | wrist | sbs.

wrist = the tool-mounted fisheye alone; sbs = front|wrist concatenated on
width, the layout the 84x168 MAE was pretrained on.
"""


def _block_pos(env) -> torch.Tensor:
  return env.scene["block"].data.root_link_pos_w - env.scene.env_origins


def _goal(env) -> torch.Tensor:
  """Goal square centre in env coords (per-env, since it is DR'd)."""
  return env.scene["goal"].data.root_link_pos_w - env.scene.env_origins


# Success and failure now live in liftpeg_rewards so that every arm is scored
# against exactly the same task definition and only the shaping differs. Both
# changed after v7; see the docstrings there for the measurements behind it.
block_at_goal = LR.block_at_goal
block_lost = LR.block_lost
success_metric = LR.m_success
success_bonus = LR.success_bonus
lost_penalty = LR.lost_penalty

# Old name, kept so existing scripts/checkpoint configs still import.
block_below_ground = LR.block_lost
dropped_penalty = LR.lost_penalty


# --- Unity reward constants ----------------------------------------------
# AgentHelper works in UNITY UNITS (1 u = 0.05 m), so every tolerance and
# exponent has to be converted:
#   ExpDistanceReward(cur, tgt, tol, maxD, k) = exp(-k * clamp(d - tol, 0, maxD))
#   LinearDistanceReward(cur, tgt, tol, orig) = 1 - (d - tol)/(d0 - tol)
_U = 0.05
EXP_TOL = 0.05 * _U          # 0.05 u  -> 2.5 mm
EXP_K = 4.0 / _U             # 4.0 per u -> 80 per m
LIN_TOL = 0.05 * _U          # 2.5 mm
LIFT_THRESH = 0.05 * _U      # SparseLift: block risen > 0.05 u = 2.5 mm
JAW_NEAR = 0.07 * _U         # JawOpenEarly window, in metres
JAW_FAR = 0.3 * _U
SUCCESS_RADIUS = 0.12 * _U   # 0.12 u -> 6 mm, and requires a grasp

# Unity scales JawEntropy at 0.1 against a CONTINUOUS jaw, so its per-step
# |d jaw| is small. Ours is binary (0.0 <-> 0.5), making every toggle a full
# 0.5 swing: at 0.1 that is -0.050 per toggle, against a maximum +0.010 for a
# full step of ideal approach. A 5:1 tax on touching the jaw at all -- freezing
# it shut is the rational policy, which is the opposite of the task. Rescaled
# so a toggle costs 0.002, i.e. a fraction of a reach step rather than 5x it.
JAW_ENTROPY_SCALE = 0.004

# PickPoint is the grasp site on the triangle, not its centroid. Unity puts it
# 6.61 mm out and 13.6 mm up from the block origin; 6.61 mm matches the face
# inradius (7.217 mm support minus the 1 mm bar half-thickness = 6.22, and
# Unity's own collider centre is 6.59), so it sits mid-face near the top edge.
PICKPOINT_LOCAL = (0.0, 0.00659, 0.0136)


def _pickpoint(env) -> torch.Tensor:
  """PickPoint in world coords, carried by the block's orientation."""
  blk = env.scene["block"]
  q = blk.data.root_link_quat_w
  off = torch.tensor(PICKPOINT_LOCAL, device=env.device).expand(q.shape[0], 3)
  return blk.data.root_link_pos_w + quat_apply(q, off) - env.scene.env_origins


def _grasped(env) -> torch.Tensor:
  """Contact-based grasp: BOTH jaw pads touching the block.

  Unity uses a trigger-volume GraspDetector, which has no MuJoCo equivalent.
  Requiring both pads is stricter and physical, and it is what gates success,
  SparseGrasp and SparseLift.
  """
  return contact_both_jaws(env)


class unity_reward:
  """Faithful port of LiftPegAgent.UpdateReward.

  Unity accumulates the six terms and emits the DELTA (`AddReward(current -
  reward)`), so this returns the per-step change in the summed potential --
  the same structure a3b uses.
  """

  def __init__(self, cfg, env):
    n, dev = env.num_envs, env.device
    self._prev = torch.zeros(n, device=dev)
    self._fresh = torch.ones(n, dtype=torch.bool, device=dev)
    self._jaw_entropy = torch.zeros(n, device=dev)
    self._prev_jaw = torch.zeros(n, device=dev)
    self._orig_d = torch.ones(n, device=dev)     # |origBlock - goal|
    self._orig_z = torch.zeros(n, device=dev)

  def reset(self, env_ids=None) -> None:
    idx = slice(None) if env_ids is None else env_ids
    self._fresh[idx] = True
    self._jaw_entropy[idx] = 0.0
    self._prev_jaw[idx] = 0.0

  def __call__(self, env) -> torch.Tensor:
    tool = a3b._tool_pos(env)
    pick = _pickpoint(env)
    block = _block_pos(env)
    goal = _goal(env)
    grasped = _grasped(env)
    jaw = _jaw_cmd(env)

    # capture the episode's starting geometry on the first call after reset
    if self._fresh.any():
      f = self._fresh
      self._orig_d[f] = torch.norm(block - goal, dim=-1)[f].clamp_min(1e-4)
      self._orig_z[f] = block[:, 2][f]

    d_pick = torch.norm(pick - tool, dim=-1)
    # Unity's exponential alone is unusable here. Measured with a random-action
    # probe (256 envs x 600 steps): the tool never gets closer than 35.7 mm,
    # nothing ever grasps, and BlockDistance sits at 0.018 with a gradient of
    # ~0.0005 per mm -- flat. Unity gets away with it because ResetTool starts
    # the tool directly ABOVE the block, making approach a 1-D descent; we hold
    # the home pose fixed so it is reproducible on the real arm, which leaves up
    # to ~18 mm of lateral offset to solve with no signal.
    #
    # So add a scale-free LINEAR reach potential -- the same form pushblock
    # actually trained on -- and keep Unity's exponential as the near-field
    # term where it does have slope. Deliberate, documented deviation.
    reach = -d_pick / 0.05
    block_distance = torch.exp(-EXP_K * (d_pick - EXP_TOL).clamp_min(0.0))

    d_goal = torch.norm(block - goal, dim=-1)
    goal_distance = 1.0 - (d_goal - LIN_TOL) / (self._orig_d - LIN_TOL)

    sparse_grasp = (grasped & (jaw < 0.1)).float()
    sparse_lift = (grasped & (block[:, 2] - self._orig_z > LIFT_THRESH)).float()

    jaw_open_early = torch.where(
      (~grasped) & (d_pick < JAW_FAR) & (d_pick > JAW_NEAR) & (jaw < 0.01),
      -torch.ones_like(d_pick),
      torch.zeros_like(d_pick),
    )

    self._jaw_entropy = self._jaw_entropy - (jaw - self._prev_jaw).abs()
    self._prev_jaw = jaw

    potential = (
      1.0 * reach
      + 1.0 * block_distance
      + 1.0 * goal_distance
      + 1.0 * sparse_grasp
      + 1.0 * sparse_lift
      + 0.1 * jaw_open_early
      + JAW_ENTROPY_SCALE * self._jaw_entropy
    )
    delta = torch.where(self._fresh, torch.zeros_like(potential),
                        potential - self._prev)
    self._prev = potential
    self._fresh = torch.zeros_like(self._fresh)
    return delta


def m_reach_mm(env) -> torch.Tensor:
  """Stage 1: tool -> PickPoint distance, mm. Reduced with min over the episode."""
  return torch.norm(_pickpoint(env) - a3b._tool_pos(env), dim=-1) * 1000.0


def m_grasp_rate(env) -> torch.Tensor:
  """Stage 2: did both jaw pads ever hold the block this episode."""
  return _grasped(env).float()


def m_lift_mm(env) -> torch.Tensor:
  """Stage 3: block height above the plate, mm."""
  return (_block_pos(env)[:, 2] - LS.PEG_Z) * 1000.0


def m_goal_mm(env) -> torch.Tensor:
  """Stage 4: block -> goal distance, mm."""
  return torch.norm(_block_pos(env) - _goal(env), dim=-1) * 1000.0


def _critic_state(env) -> torch.Tensor:
  tool = a3b._tool_pos(env)
  block = _block_pos(env)
  goal = _goal(env)
  return torch.cat([tool, block, block - tool, goal - block], dim=-1)


def _teacher_state(env) -> torch.Tensor:
  """Privileged observation for the state-based teacher (22-D).

  `_critic_state` is deliberately NOT reused here. Twelve dimensions of pure
  position are enough for a VALUE function, but an ACTOR cannot work from them:

  * **No ring orientation.** The triangle is 3-fold symmetric and its yaw is
    randomised over a full 120 deg symmetry period, while the tool's
    orientation is FIXED (`position_only=True`). Which face is presented is
    therefore the single most important fact about the state, and a policy
    reading only centroid positions cannot distinguish two ring orientations at
    all. It would fail structurally -- and be misread as "state-based RL cannot
    do this task".
  * **No jaw state.** The jaw is binary, so a policy that cannot observe
    whether it is currently holding has no way to decide whether to keep
    holding.

  Orientation is supplied as the three candidate grasp points relative to the
  tool rather than as a yaw angle. A raw angle is discontinuous at the wrap and
  ambiguous under the ring's own 3-fold symmetry; three offsets are continuous,
  symmetry-respecting, directly actionable, and do not bake in a choice of
  which face to take -- that stays the policy's decision.
  """
  tool = a3b._tool_pos(env)
  block = _block_pos(env)
  goal = _goal(env)
  blk = env.scene["block"]
  q = blk.data.root_link_quat_w
  origin = blk.data.root_link_pos_w - env.scene.env_origins

  feats = [tool, block, block - tool, goal - block]
  for yaw_deg in LS.BLOCK_BAR_YAWS_DEG:
    a = math.radians(yaw_deg)
    local = torch.tensor(
      [-math.sin(a) * LS.BLOCK_BAR_RADIAL,
       math.cos(a) * LS.BLOCK_BAR_RADIAL,
       LR.PICKPOINT_LOCAL[2]], device=env.device
    ).expand(q.shape[0], 3)
    feats.append(origin + quat_apply(q, local) - tool)
  feats.append(_jaw_cmd(env).unsqueeze(-1))
  return torch.cat(feats, dim=-1)


def _m_held(env) -> torch.Tensor:
  from surgical_lab.tasks.lift.config.psm import liftpeg_shaped as SH
  return SH.m_held(env)


def _m_grip_mm(env) -> torch.Tensor:
  from surgical_lab.tasks.lift.config.psm import liftpeg_shaped as SH
  return SH.m_grip_mm(env)


def _camera_multi(env) -> torch.Tensor:
  """Both views side by side as one (3, H, 2W) image.

  Concatenated on WIDTH rather than stacked on channels so the CNN's 3-channel
  input and the SpatialSoftmax head need no change -- convolutions do not care
  how wide the field is. Stacking to 6 channels would mean editing the model
  config, and this arm is meant to vary the VIEW and nothing else.
  """
  a = a3b._camera_rgb_normalized(env, "liftcam_cam")
  b = a3b._camera_rgb_normalized(env, "liftcam2_cam")
  return torch.cat([a, b], dim=-1)


# --- wrist camera -----------------------------------------------------------
# The real wrist unit is a fisheye, so a raw pinhole render is the wrong
# picture: render WIDE and resample through the fitted warp. These constants are
# the ones scripts/dump_mae_sim_data_liftpeg.py dumped the MAE datasets with and
# must stay in lockstep with it -- otherwise the policy sees a different lens
# than the encoder was pretrained on.
WRIST_SRC = int(os.environ.get("LIFTPEG_WRIST_SRC", "252"))
# 135, not the 165 this started at. The warp reads an 89.5 deg fisheye out of a
# RECTILINEAR source render, and a 165 deg pinhole spends almost all its pixels
# on extreme angles the fisheye never samples: only a 48x48 patch of the 252px
# render was used, then bilinearly stretched to 84x84 -- a 2.4x magnification,
# which is what made the wrist look soft next to the natively-rendered front
# cam. At 135 the same render gives 1.33 source px per output px.
#
# 135 is also a FLOOR, not a preference. With the vignette off the frame
# corners need rays 66.5 deg off axis, so a source narrower than ~133 deg
# cannot supply them -- and build_grid CLAMPS such rays to the frustum edge
# rather than marking them invalid, so the failure is smeared corners rather
# than an error.
WRIST_RENDER_FOV = float(os.environ.get("LIFTPEG_WRIST_RENDER_FOV", "135.0"))
# Off by default. The fitted value (1.3469) cuts a dark circle into the corners
# -- frame corners sit at sqrt(2)=1.414 -- but the real wrist frames are square
# cropped and show no vignette, so rendering one is a sim-only artefact. Only
# the validity mask depends on this; the sampling geometry does not.
WRIST_VIGNETTE_R = float(os.environ.get("LIFTPEG_WRIST_VIGNETTE_R", "99.0"))
# The CEM fit (out/stage_b_e22_nooff.json, best_warp[6]) includes a -5.317 deg
# roll: the add-on lens does not sit square on the sensor. build_grid has no
# roll parameter -- wrist_cem/harness.py applies it by rotating the grid after
# the fact -- so calling build_grid directly, as this module and the dump
# script both did, silently dropped it and left every sim wrist frame rotated
# 5.3 deg against the real camera it was calibrated to.
WRIST_ROLL_DEG = float(os.environ.get("LIFTPEG_WRIST_ROLL_DEG", "-5.317"))
_WRIST_GRID: dict[str, tuple] = {}


def build_wrist_grid(device, out_hw: tuple[int, int] | None = None):
  """The wrist fisheye grid. THE definition -- do not re-derive it elsewhere.

  Both the RL observation and scripts/dump_mae_sim_data_liftpeg.py must warp
  identically, or the policy is fed a different lens than its encoder was
  pretrained on. They used to hold separate copies of these constants, and the
  copies drifted: the dump script and this module both silently dropped the
  fit's -5.317 deg roll because fe.build_grid has no roll argument. One
  function now, called by both.
  """
  from surgical_lab.utils import fisheye as fe

  h, w = out_hw or (CAM_RES, CAM_RES)
  # Fitted wrist fisheye from the CEM (out/stage_b_e22_nooff.json, best_warp).
  g, v = fe.build_grid(
    (h, w), WRIST_RENDER_FOV, 89.464, distortion=0.9424, k1=0.030,
    center=(-0.2096, 0.0591), vignette_r=WRIST_VIGNETTE_R, device=device,
  )
  if WRIST_ROLL_DEG:
    # Same rotation wrist_cem/harness.py:warp_grid applies.
    c = math.cos(math.radians(WRIST_ROLL_DEG))
    sn = math.sin(math.radians(WRIST_ROLL_DEG))
    R = torch.tensor([[c, -sn], [sn, c]], device=device, dtype=g.dtype)
    g = g @ R.T
  return g, v


def _wrist_grid(env):
  """Cached per device; the geometry itself lives in build_wrist_grid."""
  key = str(env.device)
  if key not in _WRIST_GRID:
    _WRIST_GRID[key] = build_wrist_grid(env.device)
  return _WRIST_GRID[key]


def _stretch(img: torch.Tensor) -> torch.Tensor:
  """Per-frame min-max, mirroring the deployment's cv::normalize."""
  flat = img.flatten(1)
  lo = flat.min(dim=1).values[:, None, None, None]
  hi = flat.max(dim=1).values[:, None, None, None]
  return (img - lo) / (hi - lo).clamp_min(1e-6)


def _camera_wrist(env) -> torch.Tensor:
  """Wrist view, fisheye-warped, as (B, 3, CAM_RES, CAM_RES)."""
  from surgical_lab.utils import fisheye as fe

  grid, valid = _wrist_grid(env)
  raw = manipulation_mdp.camera_rgb(env, "wristcam_cam")  # (B, 3, SRC, SRC)
  return _stretch(fe.apply(raw, grid, valid))


def _camera_sbs(env) -> torch.Tensor:
  """Front | wrist concatenated on WIDTH -> (B, 3, CAM_RES, 2*CAM_RES).

  Each view is stretched on its own rather than jointly: the rig is two
  independent cameras, each running its own cv::normalize, so a joint stretch
  would couple their exposures in a way the real hardware never does.
  """
  return torch.cat(
    [a3b._camera_rgb_normalized(env, "liftcam_cam"), _camera_wrist(env)], dim=-1
  )


def _nan_guard(env, x: torch.Tensor, where: str) -> torch.Tensor:
  """Keep a rare non-finite state from killing a multi-hour run, and record it.

  A 6-hour vision run died at iteration 3060 -- after reaching 0.88 success --
  with "observation group 'critic' contains NaN". It could not be reproduced:
  4 million env-steps under the trained policy and 6 million under random
  actions were clean, against roughly 150 million env-steps before the crash.
  At that rate hunting it by sampling costs more than it is worth.

  So instead: replace the non-finite rows with zeros so the run survives, count
  them into a metric so the frequency is visible rather than silently
  swallowed, and dump the offending env's full state ONCE so the next
  occurrence identifies the cause instead of just repeating the symptom.
  """
  bad = ~torch.isfinite(x)
  if not bool(bad.any()):
    return x
  rows = bad.any(dim=-1)
  env._liftpeg_nan_count = getattr(env, "_liftpeg_nan_count", 0) + int(rows.sum())
  if not getattr(env, "_liftpeg_nan_dumped", False):
    env._liftpeg_nan_dumped = True
    i = int(rows.nonzero()[0])
    blk, rb = env.scene["block"], env.scene["robot"]
    print(f"\n[NAN] first non-finite {where} at env {i}, step "
          f"{getattr(env, 'common_step_counter', -1)}", flush=True)
    print(f"[NAN]   columns: {bad[i].nonzero().flatten().tolist()}", flush=True)
    for nm, v in (("block_pos", blk.data.root_link_pos_w[i]),
                  ("block_quat", blk.data.root_link_quat_w[i]),
                  ("block_vel", blk.data.root_link_vel_w[i]),
                  ("joint_pos", rb.data.joint_pos[i]),
                  ("joint_vel", rb.data.joint_vel[i])):
      print(f"[NAN]   {nm}: {v.tolist()}", flush=True)
  return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def nan_state(env) -> torch.Tensor:
  """Terminate any env whose SIMULATOR state has gone non-finite.

  An 11-hour run died with the whole physics state NaN -- block position and
  velocity, and all twelve joint positions and velocities. Sanitising the
  observation, which is what the previous guard did, cannot help: the sim
  itself is corrupt and will keep emitting NaN every step until something
  rewrites qpos. A reset does exactly that, so terminating is the recovery.

  Registered as a normal (non-timeout) termination, so the episode counts as a
  failure rather than being quietly excused, and `nan_rows` records how often
  it happens.
  """
  d = env.sim.data
  bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for nm in ("qpos", "qvel"):
    v = getattr(d, nm, None)
    if v is None:
      continue
    t = v if torch.is_tensor(v) else torch.as_tensor(v)
    if t.ndim >= 2 and t.shape[0] == env.num_envs:
      bad |= (~torch.isfinite(t)).any(dim=tuple(range(1, t.ndim)))
  if bool(bad.any()):
    env._liftpeg_nan_count = getattr(env, "_liftpeg_nan_count", 0) + int(bad.sum())
  return bad


def _guarded(fn):
  """Wrap an observation term so a non-finite value cannot kill the run.

  The previous guard covered only `_shaped_state`, but the critic group also
  carries `joint_pos` and `actions` -- and joint_pos was the term that was NaN
  when the run crashed. Guarding one term of a group is guarding none.
  """
  def _inner(env, *a, **k):
    return _nan_guard(env, fn(env, *a, **k), getattr(fn, "__name__", "obs"))
  _inner.__name__ = getattr(fn, "__name__", "guarded")
  return _inner


def _m_phased(env) -> torch.Tensor:
  """Ring left the post below the height that clears it -- through the shaft."""
  from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as _LR
  return _LR.terms(env)["phased"].float()


# --- controller-fault guard ------------------------------------------------
# The dVRK does not trip on force -- it has no force sensor. It trips when the
# PID's SETPOINT diverges from the MEASURED joint by more than a tolerance, and
# on 2026-09-08 that is exactly what killed a liftpeg run on the rig:
#
#   PSM1_PID: tracking error, mask (1 for error): 0 0 1 0 0 0 0
#   errors:      ... 0.0500361 ...        <- joint 3 (insertion), metres
#   tolerances:  ... 0.0500000 ...
#   measured:    ... 0.143109  ...
#   setpoint:    ... 0.193146  ...
#   -> desired state FAULT, "detected power loss (actuators)", arm disabled
#
# The tool was pressed against something, could not advance, and the commanded
# setpoint kept integrating downward until it led the achieved position by
# 50 mm. The sim has the SAME quantity -- the Cartesian action term stores a
# target pose that it integrates every step -- and a3b sets
# `max_target_lead=None`, i.e. the target may run arbitrarily far ahead of the
# arm with no consequence at all. That is why nothing in sim ever discouraged
# the behaviour that faults the real robot.
#
# So this terminates on the CONDITION rather than on a force: the same
# divergence, measured the same way, with margin under the real 50 mm trip.
TARGET_LEAD_FAULT = float(os.environ.get("LIFTPEG_TARGET_LEAD_FAULT", "0"))
# Hard floor on the COMMANDED tool z, the second half of "do not grasp into the
# peg". Raising the grip target aims the jaw higher; this stops the integrator
# from driving the target below a height at all, however the policy behaves.
# Unset by default -- the tool must still descend to the grip point, so a floor
# set above it makes the task impossible.
#
# ROBOT BASE FRAME, not world -- this is handed to the cartesian action term,
# which clamps the integrated target's z in the FK frame
# (cartesian_actions.py: `p_new[..., 2:3].clamp_min(min_target_z)`). For this
# arm that number is NEGATIVE, as tissue's calibrated TISSUE_MIN_TARGET_Z
# (-0.3272) already shows. Measured offset on the liftpeg scene:
# world_z ~= base_z + 0.3273, so a floor at world 18 mm is about -0.3093.
# The gate below was `> 0.0`, which silently discarded every usable value and
# left this knob inert since it was added. Metres. 0 disables.
MIN_TARGET_Z = float(os.environ.get("LIFTPEG_MIN_TARGET_Z", "0"))


def _target_lead(env) -> torch.Tensor:
  """|commanded target - achieved tool position|, metres. The sim twin of the
  dVRK's PID tracking error."""
  term = env.action_manager.get_term("cartesian_delta")
  p_target = term._ee_target_tf[..., :3, 3]
  q_arm = term._entity.data.joint_pos[:, term._joint_ids]
  p_actual = term._fk_fn(q_arm)[..., :3, 3]
  return torch.norm(p_target - p_actual, dim=-1)


def m_target_lead_mm(env) -> torch.Tensor:
  return _target_lead(env) * 1000.0


def target_lead_fault(env) -> torch.Tensor:
  """Terminate: the arm is commanding into something it cannot move through.

  Off unless LIFTPEG_TARGET_LEAD_FAULT is set, so every existing run keeps its
  exact termination set.
  """
  if TARGET_LEAD_FAULT <= 0.0:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return _target_lead(env) > TARGET_LEAD_FAULT


def _m_nan(env) -> torch.Tensor:
  """Cumulative count of non-finite observation rows -- 0 on a healthy run."""
  return torch.full((env.num_envs,), float(getattr(env, "_liftpeg_nan_count", 0)),
                    device=env.device)


def _shaped_state(env) -> torch.Tensor:
  """Privileged state for the shaped task (31-D).

  Differs from `_teacher_state` in what it says about the ring: instead of the
  three BAR CENTRES it gives the two JAW-ALIGNED grip points and the wall
  thickness at each. The jaw's closing axis is fixed in world, so those two
  points are the only places a wall can actually be straddled -- the bar centres
  are only reachable when the yaw happens to line up. Handing the policy the
  right target beats asking it to rediscover the geometry.
  """
  from surgical_lab.tasks.lift.config.psm import liftpeg_shaped as SH

  tool = a3b._tool_pos(env)
  block = _block_pos(env)
  goal = _goal(env)
  P, T = SH.grip_targets(env)
  feats = [tool, block, block - tool, goal - block,
           P[:, 0] - tool, P[:, 1] - tool, T,
           _jaw_cmd(env).unsqueeze(-1),
           SH.shaped_terms(env)["held"].float().unsqueeze(-1)]
  return _nan_guard(env, torch.cat(feats, dim=-1), "_shaped_state")


def place_fixtures(env, env_ids: torch.Tensor | None) -> None:
  """Place peg, block, camera and key light for the resetting envs.

  All in ONE event and from one random draw, so the peg and the block cannot
  disagree about where the post is.

  Fixed geometry cannot tile across mujoco_warp worlds, so both ride mocap
  bodies whose poses are written every reset -- the same pattern a3b uses.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  origins = env.scene.env_origins[env_ids]

  # Peg: uniform in Unity's 25 x 25 mm SpawnGround square.
  jitter = (torch.rand((n, 2), device=env.device) * 2.0 - 1.0) * LS.SPAWN_HALF
  peg_xy = torch.tensor(LS.SPAWN_CENTER, device=env.device).unsqueeze(0) + jitter
  peg = torch.cat(
    [peg_xy, torch.full((n, 1), LS.PEG_Z, device=env.device)], dim=-1
  )
  ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(n, 4)
  env.scene["peg"].write_mocap_pose_to_sim(
    torch.cat([origins + peg, ident], dim=-1), env_ids=env_ids
  )

  key = torch.tensor(LS.LIGHT_POS, device=env.device).unsqueeze(0).expand(n, 3)
  env.scene["keylight"].write_mocap_pose_to_sim(
    torch.cat([origins + key, ident], dim=-1), env_ids=env_ids
  )

  # Goal square, drawn BEFORE the camera: hemisphere mode has to know where it
  # is in order to keep it in frame.
  gj = (torch.rand((n, 2), device=env.device) * 2.0 - 1.0) * 0.004
  gpos = torch.cat([
    torch.tensor(LS.GOAL_SQUARE_CENTER, device=env.device).unsqueeze(0) + gj,
    torch.zeros((n, 1), device=env.device),
  ], dim=-1)

  # The primary camera takes the head-on pose when that view is selected; the
  # multi arm keeps the side view here and adds the head-on one as liftcam2.
  _side = CAM_VIEW != "head"
  cpos = LS.CAM_POS if _side else LS.CAM_POS_HEAD
  cpitch = None if _side else LS.CAM_PITCH_HEAD
  # Jitter is drawn HERE rather than via a separate dr.cam_pos/dr.cam_quat
  # event pair: this event already writes the camera mocap pose on reset, so a
  # second event would either be overwritten by this one or overwrite it,
  # depending on registration order. One writer, one source of truth.
  # The cam_fovy event is ordered BEFORE this one, so the drawn value is
  # already in the model and the visibility check can use the ACTUAL fovy
  # rather than a nominal that no env has.
  _fovy_drawn = None
  _ci = getattr(env, "_liftpeg_cam_index", None)
  if _ci is None:
    _m = env.sim.mj_model
    _ci = next(i for i in range(_m.ncam) if "liftcam/" in _m.camera(i).name)
    env._liftpeg_cam_index = _ci
  try:
    _fovy_drawn = env.sim.model.cam_fovy[env_ids, _ci].reshape(-1).float()
  except Exception:
    _fovy_drawn = None

  if LS.CAM_HEMI:
    # Everything that must stay visible on the first frame: the tool anywhere
    # in its home box, the top of the post, the plate corners, and the goal.
    _tip = LS._tip_box_corners(env.device).unsqueeze(0).expand(n, -1, -1)
    _plate = LS.PEG_PLATE_HALF[0]
    _cor = torch.tensor([[+1, +1], [+1, -1], [-1, +1], [-1, -1]],
                        device=env.device, dtype=torch.float32) * _plate
    _plate_pts = torch.cat(
      [peg_xy.unsqueeze(1) + _cor.unsqueeze(0),
       torch.full((n, 4, 1), LS.PEG_Z, device=env.device)], dim=-1)
    _post = torch.cat([peg_xy, torch.full((n, 1), LS.PEG_POST_TOP,
                                          device=env.device)], dim=-1)
    _gh = 0.5 * LS.GOAL_SQUARE_SIDE
    _gc = torch.tensor([[+1, +1], [+1, -1], [-1, +1], [-1, -1]],
                       device=env.device, dtype=torch.float32) * _gh
    _goal_pts = torch.cat(
      [gpos[:, :2].unsqueeze(1) + _gc.unsqueeze(0),
       torch.zeros((n, 4, 1), device=env.device)], dim=-1)
    _pts = torch.cat([_tip, _plate_pts, _post.unsqueeze(1), _goal_pts], dim=1)
    # Aim at the middle of the workspace rather than a fixed floor point, so a
    # camera on the far side still centres what matters.
    _look = 0.5 * (_pts.amin(dim=1) + _pts.amax(dim=1))
    _fovy = torch.full((n,), LS.CAM_FOVY, device=env.device)
    if _fovy_drawn is not None:
      _fovy = _fovy_drawn
    _stand = torch.cat([peg_xy, torch.full((n, 1), LS.PEG_Z, device=env.device)],
                       dim=-1)
    cam, quat, _ok = LS.camera_pose_hemisphere(n, env.device, _look, _pts,
                                               _fovy, _stand, _goal_pts,
                                               peg_xy)
    env._liftpeg_cam_reject = int((~_ok).sum())
  else:
    cam, quat = LS.camera_pose_batch(n, env.device, base_pos=cpos,
                                     base_pitch_deg=cpitch)
  env.scene["liftcam"].write_mocap_pose_to_sim(
    torch.cat([origins + cam, quat], dim=-1), env_ids=env_ids
  )
  # Scene has no .keys(); gate on the configured view instead.
  if CAM_VIEW == "multi":
    # Drawn independently of liftcam: two physically separate mounts do not
    # share a mounting error, and correlating them would let the policy cancel
    # the jitter by comparing the views.
    cam2, q2 = LS.camera_pose_batch(n, env.device, base_pos=LS.CAM_POS_HEAD,
                                    base_pitch_deg=LS.CAM_PITCH_HEAD)
    env.scene["liftcam2"].write_mocap_pose_to_sim(
      torch.cat([origins + cam2, q2], dim=-1), env_ids=env_ids
    )
  env.scene["goal"].write_mocap_pose_to_sim(
    torch.cat([origins + gpos, ident], dim=-1), env_ids=env_ids
  )

  # Block, placed from the SAME draw in the SAME event. An earlier version
  # stashed peg_xy on the env and read it back from a second event, which
  # assumed both the event ordering and the env_ids slice lined up -- they did
  # not, and the block spawned beside the post instead of threaded on it.
  # THE task's randomisation: where the triangle sits on the shaft, and its
  # yaw. The hole clears the post by LS.RING_CLEARANCE, so the ring can rest
  # anywhere in that annulus; yaw spans 120 deg, covering every distinct edge
  # the policy might grasp. This is the analogue of pushblock's block pose DR.
  ang = torch.rand(n, device=env.device) * 6.2831853
  rad = torch.rand(n, device=env.device).sqrt() * LS.RING_CLEARANCE
  blk = torch.stack(
    [
      peg_xy[:, 0] + rad * torch.cos(ang),
      peg_xy[:, 1] + rad * torch.sin(ang),
      torch.full((n,), LS.PEG_Z + LS.BLOCK_SPAWN_DZ, device=env.device),
    ],
    dim=-1,
  )
  yaw = LS.TRI_YAW_RANGE[0] + torch.rand(n, device=env.device) * (
    LS.TRI_YAW_RANGE[1] - LS.TRI_YAW_RANGE[0]
  )
  quat = torch.stack(
    [torch.cos(yaw * 0.5), torch.zeros_like(yaw), torch.zeros_like(yaw),
     torch.sin(yaw * 0.5)], dim=-1
  )
  root = torch.cat([origins + blk, quat], dim=-1)
  block = env.scene["block"]
  block.write_root_link_pose_to_sim(root, env_ids=env_ids)
  block.write_root_com_velocity_to_sim(
    torch.zeros((n, 6), device=env.device), env_ids=env_ids
  )



# ---------------------------------------------------------------------------
# Extended domain randomisation.
#
# An audit of what liftpeg actually randomised (docs/liftpeg_domain_randomisation.md)
# found appearance well covered and PHYSICS almost entirely uncovered: floor
# friction was the only dynamics term, and the robot started from exactly the
# same pose every episode. That is the wrong shape for a task whose sim-to-real
# gap runs through a hand-clamped endoscope and a cable-driven wrist -- and we
# had just found the contact model was wrong by 1.3 mm, which is the argument
# for randomising contact rather than trusting one calibration.
#
# Home-pose spread follows OmniReset's "reaching resets" (Yin et al., ICLR
# 2026): start the gripper across the workspace rather than at one point, so
# the policy sees the approach from many places instead of memorising one.
# NOTE this is only the first of their four reset distributions -- near-object,
# stable-grasp and near-goal resets are the parts that attack long-horizon
# exploration, and are NOT implemented here.
# ---------------------------------------------------------------------------

EXTRA_DR = os.environ.get("LIFTPEG_EXTRA_DR", "1") == "1"

# Home pose, in JOINT space. Only the three position DOFs are perturbed:
# outer_yaw and outer_pitch swing the tip laterally about the RCM,
# main_insertion slides it along the shaft. tool_roll/pitch/yaw are left alone
# on purpose -- the action term is position_only, so the policy cannot correct
# an orientation error, and randomising them would change the task rather than
# harden it.
# Calibrated to a target tool-tip spread, not guessed -- see the measured
# numbers in docs/liftpeg_domain_randomisation.md.
HOME_YAW_JITTER = float(os.environ.get("LIFTPEG_HOME_YAW_JIT", "0.020"))
# Pitch is deliberately SMALLER than yaw. The arm is mounted tilted, so
# outer_pitch is not a lateral DOF: measured at 0.040 rad it moves the tip
# +-20 mm vertically against only +-6 mm in y, while outer_yaw at the same
# angle is purely lateral. Equal angles would have given a lopsided box.
HOME_PITCH_JITTER = float(os.environ.get("LIFTPEG_HOME_PITCH_JIT", "0.011"))
HOME_INSERT_JITTER = float(os.environ.get("LIFTPEG_HOME_INSERT_JIT", "0.004"))
LOWER_TOOL_OFFSET = 0.211063


def randomize_home_joints(
  env, env_ids: torch.Tensor | None = None, asset_name: str = "robot"
) -> None:
  """Scatter the tool's starting pose about the nominal home.

  Written as one event rather than three `reset_joints_by_offset` terms
  because that helper applies a SINGLE range to every joint it is given, and
  these three want different ones -- a radian of insertion is 1 m of travel
  while a radian of yaw is a rotation about the RCM.

  Runs AFTER `lower_tool`, and writes absolute positions derived from the
  entity defaults, so it composes with the fixed insertion offset instead of
  fighting it.
  """
  robot = env.scene[asset_name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  names = ("outer_yaw", "outer_pitch", "main_insertion")
  jit = (HOME_YAW_JITTER, HOME_PITCH_JITTER, HOME_INSERT_JITTER)
  ids = torch.tensor([robot.joint_names.index(x) for x in names],
                     device=env.device, dtype=torch.long)
  # From default_joint_pos, NOT the live joint_pos: entity .data is stale
  # inside a reset event, so reading it jitters about wherever the PREVIOUS
  # episode happened to end. That produced a 72 mm spread in tool height
  # instead of the intended 20, and it looked like a plausible range.
  q = robot.data.default_joint_pos[env_ids][:, ids].clone()
  # `lower_tool` adds this to main_insertion in its own event; default_joint_pos
  # predates it, so add it here or the tool starts 211 mm too high.
  q[:, 2] += LOWER_TOOL_OFFSET
  for k, w in enumerate(jit):
    q[:, k] += (torch.rand(n, device=env.device) * 2.0 - 1.0) * w
  robot.write_joint_position_to_sim(q, joint_ids=ids, env_ids=env_ids)
  # The two setters index differently. write_joint_position_to_sim turns
  # env_ids into a column itself; set_joint_position_target does a plain
  # `[env_ids, joint_ids]`, whose advanced indices must broadcast -- with more
  # than one joint that needs env_ids as an explicit column or it raises.
  robot.set_joint_position_target(q, joint_ids=ids,
                                  env_ids=env_ids.unsqueeze(-1))


# Contact softness (solref / solimp) is NOT randomised, and cannot be in this
# build. `geom_solref` is materialised as a stride-0 EXPANDED tensor -- one
# world's memory viewed across every world -- so a per-env write silently
# collapses. Verified directly: writing the pattern
# [0.004, 0.005, ... 0.011] to eight envs reads back as 0.004 eight times, and
# torch warns "Use of index_put_ on expanded tensors is deprecated". mjlab's
# own dr._randomize_model_field hits the same wall (every env got an identical
# 0.00551), so this is a limitation of the field, not of how it was called.
#
# Do not "fix" this by reintroducing a solref DR term without first checking
# that geom_solref.stride(0) != 0. Randomising the ring's FRICTION and MASS
# covers part of the same contact uncertainty and does work.

def _add_extended_dr(cfg, block_geoms, peg_geoms) -> None:
  from mjlab.envs.mdp import dr
  from surgical_lab.tasks.push.config.psm.env_cfg import PSM_MAIN_JOINTS

  # -- home pose (OmniReset "reaching resets", position only)
  cfg.events["home_pose"] = EventTermCfg(mode="reset", func=randomize_home_joints)

  # -- ring inertial properties. scale, not abs: the mesh-derived mass is the
  #    right centre to vary about and an absolute range would silently ignore
  #    any future change to the geometry.
  cfg.events["block_mass"] = EventTermCfg(
    mode="reset", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("block", body_names=("block",)),
            "operation": "scale", "ranges": {0: (0.7, 1.4)}},
  )

  # -- contact friction on the surfaces that actually matter: the ring, and
  #    the post/plate it slides on. Floor friction was already randomised;
  #    these are the ones the grasp and the unthread depend on.
  cfg.events["block_friction"] = EventTermCfg(
    mode="reset", func=dr.geom_friction,
    params={"asset_cfg": SceneEntityCfg("block", geom_names=block_geoms),
            "operation": "abs", "ranges": {0: (0.4, 1.4)}, "shared_random": True},
  )
  cfg.events["peg_friction"] = EventTermCfg(
    mode="reset", func=dr.geom_friction,
    params={"asset_cfg": SceneEntityCfg("peg", geom_names=peg_geoms),
            "operation": "abs", "ranges": {0: (0.2, 0.9)}, "shared_random": True},
  )

  # -- contact softness. solref[0] is the restitution time constant. It must
  #    stay at or above 2x the physics timestep (0.002 -> 0.004) or driven
  #    contacts go soft again -- that is exactly the bug that let the ring sink
  #    1.26 mm into the post. The range is therefore ONE-SIDED about nominal:
  #    stiffer is free, softer is not allowed at all.
  # -- camera intrinsics
  cfg.events["camera_fovy"] = EventTermCfg(
    mode="reset", func=dr.cam_fovy,
    params={"asset_cfg": SceneEntityCfg("liftcam", camera_names=("liftcam",)),
            # [45, 70] not [54, 66]. The OAK-1's native VFOV (~55 deg, and the crop
            # preserves full height so that IS the effective FOV) sits below
            # sim's nominal 60, and the measured real/sim apparent scale of
            # 1.23x needs coverage beyond what a +-6 deg band reaches: the old
            # range spanned rendered plate widths of 34-46 px against a real 48.
            # operation is "abs", so this range -- not CAM_FOVY -- sets the
            # distribution. Centred so the MEASURED real projection (plate
            # 48 px wide at 84 px) lands near the median rather than out in the
            # tail: a uniform [45, 70] has median 57.5 deg, which renders the
            # plate at 39 px and put the real rig at the 80th percentile.
            # Centring on the image measurement is robust even if the physical
            # plate is not the 60 mm sim assumes, because it is measured in
            # pixels, not inferred from geometry.
            "operation": "abs",
            # Hemisphere mode randomises the lens as well as the pose: the
            # visibility check keeps the workspace framed, so a wide range is
            # safe here in a way it was not when the pose was near-fixed.
            # OVERRIDABLE via LIFTPEG_CAM_FOVY_DR="lo,hi". The real OAK side
            # camera measures 40.10 x 40.15 deg -- fx 1479.46 calibrated at
            # 1920x1080, on-chip 1080x1080 centre crop, 2*atan(540/1479.46) --
            # so this 58-68 band is ~1.5x too wide, and every policy so far has
            # trained on a camera the rig does not have. It is why the real
            # plate measured LARGER than in 6143 of 6144 sim draws.
            "ranges": {0: ((35.0, 85.0) if LS.CAM_HEMI else _FOVY_DR)}},
  )

  # -- actuation. The dVRK is cable driven, so gain error and encoder offset
  #    are the two calibration errors that actually show up on hardware.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="reset", func=dr.pd_gains,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=PSM_MAIN_JOINTS),
            "operation": "scale",
            "kp_range": (0.8, 1.25), "kd_range": (0.8, 1.25)},
  )
  cfg.events["encoder_bias"] = EventTermCfg(
    mode="reset", func=dr.encoder_bias,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=PSM_MAIN_JOINTS),
            "bias_range": (-0.004, 0.004)},
  )
  # ABS, not scale: the model ships dof_damping and dof_frictionloss at exactly
  # ZERO on every arm joint, so a multiplicative range is a silent no-op. These
  # absolute values give a cable-driven arm some stiction and viscous drag,
  # which is the dominant un-modelled effect on the real PSM.
  cfg.events["joint_friction"] = EventTermCfg(
    mode="reset", func=dr.joint_friction,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=PSM_MAIN_JOINTS),
            "operation": "abs", "ranges": {0: (0.0, 0.002)}},
  )
  cfg.events["joint_damping"] = EventTermCfg(
    mode="reset", func=dr.joint_damping,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=PSM_MAIN_JOINTS),
            "operation": "abs", "ranges": {0: (0.0, 0.01)}},
  )

# --- optional experiment arms --------------------------------------------
#
# Kept as functions rather than inlined so the "is this run byte-identical to
# the old one?" question has one place to look, and so a flag-off build is
# literally a pair of early returns.


def _apply_proprio(cfg, play: bool = False) -> None:
  """Experiment 1/3: replace the actor's constant-zeros branch with real
  proprioception -- (dx, dy, dz, djaw) from the episode's home pose.

  The zeros term exists only because rsl-rl's MLP head refuses an empty 1-D
  branch (see `push/config/psm/env_cfg.py::_zeros_obs`); at ONNX export the
  zeros are baked into the graph so the deployed model has a single camera
  input. Putting a real signal there is therefore not an addition, it is a
  REPLACEMENT, and it changes the deployment story: the exported graph gains a
  genuine second input, `scripts/bake_zeros_input.py` no longer applies, and
  whatever runs the policy on the robot has to assemble and feed those four
  numbers. That is a deliberate, flagged trade -- see liftpeg_proprio.py for
  why displacement-from-home is the only form of this observation that can
  survive the sim-to-real frame offset.
  """
  from surgical_lab.tasks.lift.config.psm import liftpeg_proprio as LP

  if not (LP.PROPRIO or LP.PROPRIO_DR):
    return

  term = ObservationTermCfg(func=LP.proprio_obs)
  if LP.PROPRIO_DR:
    from mjlab.utils.noise.noise_cfg import GaussianNoiseCfg

    # Per-STEP sensor noise only. The per-episode gain and offset live inside
    # `proprio_obs` because mjlab's native bias model
    # (NoiseModelWithAdditiveBias) accumulates its bias across resets instead
    # of resampling it -- `reset` does `bias = noise.apply(bias)`, which for an
    # additive cfg is a random WALK -- and it has no notion of a multiplicative
    # per-episode scale at all. Reading the source beat trusting the name.
    #
    # Per-dimension std, not a scalar: 0.2 mm on a translation and 0.01 rad on
    # a jaw angle are the same order of magnitude only by coincidence of units.
    term.noise = GaussianNoiseCfg(
      mean=0.0,
      std=(LP.PROPRIO_STEP_SIGMA_XYZ,) * 3 + (LP.PROPRIO_STEP_SIGMA_JAW,),
    )
    # Native latency. The real loop is encoder -> ROS -> inference, which is
    # not instantaneous; 0-2 control steps is 0-40 ms. Costs one DelayBuffer.
    term.delay_min_lag = 0
    term.delay_max_lag = LP.PROPRIO_DELAY_MAX_LAG
    term.delay_per_env = True
    # mjlab drops `noise` on any group with enable_corruption=False, silently.
    # In PLAY that is what we want (an eval should see the clean signal); in
    # TRAINING it would mean the DR arm trains on a clean signal while its run
    # name claims otherwise, so it is asserted rather than hoped for.
    assert play or cfg.observations["actor"].enable_corruption, (
      "actor group has corruption disabled during TRAINING, so "
      "ObservationTermCfg.noise would be silently dropped and the DR arm "
      "would be the no-DR arm under a different name")

  cfg.observations["actor"].terms = {"proprio": term}
  cfg.events["reset_home"] = EventTermCfg(mode="reset", func=LP.reset_home)
  # A proprio observation that never moves is the failure this experiment must
  # be able to detect; a flat 0.000 here means the term is mis-wired, not that
  # the policy is standing still.
  cfg.metrics["prop_dz_mm"] = MetricsTermCfg(func=LP.m_proprio_dz_mm, reduce="max")


def _apply_force_penalty(cfg) -> None:
  """Experiment 2: tax the contact force delivered into anything but the ring.

  Weight is read from LIFTPEG_FORCE_PENALTY and 0 means the term is not
  registered at all. See liftpeg_force.py for what is measured and why the
  ring's own weight is free.
  """
  from surgical_lab.tasks.lift.config.psm import liftpeg_force as LF

  if LF.FORCE_PENALTY == 0.0:
    return
  assert LF.FORCE_PENALTY > 0.0, (
    f"LIFTPEG_FORCE_PENALTY={LF.FORCE_PENALTY} is negative, which would PAY "
    "for crashing into the stand. It is a penalty magnitude; the sign is in "
    "`force_penalty`.")
  cfg.rewards["force"] = RewardTermCfg(
    func=LF.force_penalty, weight=LF.FORCE_PENALTY * LR.REWARD_SCALE
  )
  # Three metrics, not one. `force_mean_N` next to success_rate/grasp_rate is
  # what exposes the penalty's own failure mode -- a policy that touches
  # nothing, grasps nothing and scores a perfect 0.00 N. `force_peak_N` catches
  # the single hard strike a 2000-step mean hides. `force_robot_N` isolates the
  # component with a real-world referent: the peg-shaft collision.
  cfg.metrics["force_mean_N"] = MetricsTermCfg(func=LF.m_force_mean_n, reduce="mean")
  cfg.metrics["force_peak_N"] = MetricsTermCfg(func=LF.m_force_peak_n, reduce="max")
  cfg.metrics["force_robot_N"] = MetricsTermCfg(func=LF.m_force_robot_n, reduce="max")
  # CUMULATIVE load per fixture, newton-seconds over the whole episode, split
  # by what is being pressed. The three above are instantaneous summaries; this
  # is the total the scene actually absorbed, which is the safety quantity --
  # and split by target it says WHICH failure it was: stand = shaft strike,
  # goal = the set-down, floor = a drop or a scrape. reduce="last" reads the
  # accumulator at the terminal step, i.e. the episode total.
  cfg.metrics["imp_stand_Ns"] = MetricsTermCfg(func=LF.m_impulse_stand, reduce="last")
  cfg.metrics["imp_floor_Ns"] = MetricsTermCfg(func=LF.m_impulse_floor, reduce="last")
  cfg.metrics["imp_goal_Ns"] = MetricsTermCfg(func=LF.m_impulse_goal, reduce="last")
  # The accumulator must be zeroed at a KNOWN point, not inferred from a step
  # counter -- the same reason LR.reset_latches is an explicit event.
  cfg.events["reset_impulse"] = EventTermCfg(mode="reset", func=LF.reset_impulse)


def psm_lift_peg_a3b_env_cfg(
  play: bool = False, reward: str = "unity", state_only: bool = False
) -> ManagerBasedRlEnvCfg:
  """LiftPeg on the a3b recipe.

  `reward` selects the shaping arm; everything else -- scene, home pose, DR,
  camera, action space, success test -- is identical across arms, so a
  difference between runs is attributable to the reward and nothing else. See
  `liftpeg_rewards.py` for what each arm is and why it exists.
  """
  if reward not in LR.VARIANTS and reward != "e_shaped":
    raise ValueError(f"unknown reward arm {reward!r}; have "
                     f"{sorted(LR.VARIANTS) + ['e_shaped']}")
  cfg = a3b.psm_push_wide_a3b_env_cfg(play=play)

  ents = dict(cfg.scene.entities)
  for gone in ("goal", "topdown", "block"):
    ents.pop(gone, None)
  ents["peg"] = EntityCfg(spec_fn=LS.get_peg_spec)
  ents["block"] = EntityCfg(spec_fn=LS.get_block_spec)
  ents["liftcam"] = EntityCfg(spec_fn=LS.get_cam_spec)
  if CAM_VIEW == "multi":
    ents["liftcam2"] = EntityCfg(spec_fn=LS.get_cam2_spec)
  ents["keylight"] = EntityCfg(spec_fn=LS.get_keylight_spec)
  ents["goal"] = EntityCfg(spec_fn=LS.get_goal_spec)
  # Unity's home pose, in our joint convention.
  # MERGE, never replace: a3b's dict carries entries we must not drop.
  robot = ents["robot"]
  robot.init_state.joint_pos = {**robot.init_state.joint_pos, **HOME_JOINTS}
  cfg.scene.entities = ents

  from surgical_lab.mdp.actions import BinaryJointPositionActionCfg

  cfg.actions["cartesian_delta"].position_axes = (0, 1, 2)
  if MIN_TARGET_Z != 0.0:
    cfg.actions["cartesian_delta"].min_target_z = MIN_TARGET_Z
  cfg.actions["jaw_pos"] = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("jaw",),
    open_command_expr={"jaw": 0.5},
    close_command_expr={"jaw": 0.0},
  )

  cfg.observations["critic"].terms["state"] = ObservationTermCfg(
    func=_shaped_state if reward == "e_shaped" else _critic_state)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if getattr(s, "name", "") != "topdown_cam"
  ) + (
    CameraSensorCfg(
      name="liftcam_cam",
      camera_name="liftcam/liftcam",
      height=CAM_RES,
      width=CAM_RES,
      data_types=("rgb",),
      enabled_geom_groups=(0, 1, 2),
      use_shadows=True,
      use_textures=True,
    ),
  )
  if CAM_VIEW in ("wrist", "sbs"):
    # Rendered WIDE (WRIST_SRC, WRIST_RENDER_FOV); _camera_wrist resamples it
    # down to CAM_RES through the fisheye grid.
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
      CameraSensorCfg(
        name="wristcam_cam",
        camera_name="robot/wrist_camera",
        height=WRIST_SRC,
        width=WRIST_SRC,
        data_types=("rgb",),
        enabled_geom_groups=(0, 1, 2),
        use_shadows=True,
        use_textures=True,
      ),
    )
    cfg.observations["camera"].terms = {
      "liftcam_rgb": ObservationTermCfg(
        func=_camera_wrist if CAM_VIEW == "wrist" else _camera_sbs
      ),
    }
  elif CAM_VIEW == "multi":
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
      CameraSensorCfg(
        name="liftcam2_cam",
        camera_name="liftcam2/liftcam2",
        height=CAM_RES,
        width=CAM_RES,
        data_types=("rgb",),
        enabled_geom_groups=(0, 1, 2),
        use_shadows=True,
        use_textures=True,
      ),
    )
    cfg.observations["camera"].terms = {
      "liftcam_multi": ObservationTermCfg(func=_camera_multi),
    }
  elif CAM_VIEW in ("side", "head"):
    cfg.observations["camera"].terms = {
      "liftcam_rgb": ObservationTermCfg(
        func=a3b._camera_rgb_normalized, params={"sensor_name": "liftcam_cam"}
      ),
    }

  if reward == "e_shaped":
    from surgical_lab.tasks.lift.config.psm import liftpeg_shaped as SH
    shaping = SH.arm_e_shaped()
  else:
    shaping = LR.VARIANTS[reward] or unity_reward
  w = LR.REWARD_SCALE
  cfg.rewards = {
    "shaping": RewardTermCfg(func=shaping, weight=w),
    "success": RewardTermCfg(func=LR.success_bonus, weight=w),
    "lost": RewardTermCfg(func=LR.lost_penalty, weight=w),
  }
  if reward == "e_shaped" and SH.RELEASE_PEN > 0.0:
    # OUTSIDE the potential, deliberately. `place` already pays for the ring
    # being low, but shaping telescopes to phi(T) - phi(0), so a ring dropped
    # from height and one set down gently reach an identical terminal
    # potential -- no tuning of `place` can separate them. A one-shot event
    # charge can. See the note on release_penalty.
    cfg.rewards["release"] = RewardTermCfg(func=SH.release_penalty(), weight=w)
    # The metric and its reset event are registered AFTER the cfg.metrics
    # assignment below -- that assignment REPLACES the dict, so anything added
    # here would be silently dropped.
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "nan_state": TerminationTermCfg(func=nan_state),
    "success": TerminationTermCfg(func=LR.block_at_goal),
    "lost": TerminationTermCfg(func=LR.block_lost),
  }
  # Always LOGGED so the divergence is visible even when it is not fatal; only
  # TERMINATES when the threshold is set. A run that faults the real arm should
  # not look identical in sim to one that does not.
  cfg.metrics["target_lead_mm"] = MetricsTermCfg(func=m_target_lead_mm, reduce="max")
  if TARGET_LEAD_FAULT > 0.0:
    cfg.terminations["target_lead_fault"] = TerminationTermCfg(func=target_lead_fault)
  # Arm C pays its staged value EVERY step rather than the change in it, so
  # ending the episode forfeits the rest of the income. Leaving the success
  # termination in would make hovering at the goal strictly better than setting
  # the ring down -- the arm would optimise the exact opposite of the task.
  # Consequence worth being explicit about: with no cutoff, `success` stops
  # being a one-shot bonus and becomes a standing +2/step for keeping the ring
  # on the square. That is the correct incentive for a non-terminating dense
  # formulation, and it is the largest term in the arm by design.
  if reward == "c_dense":
    cfg.terminations.pop("success")

  # success_rate is the END of reach -> grasp -> unthread -> transport, so it
  # reads 0.0000 whether stage 1 or stage 4 is failing. These separate the
  # stages, so a stalled run says WHERE it stalled. clear_rate is the one v7
  # was missing and the one that matters now: it is the 30 mm unthread that
  # every previous run has died on.
  cfg.metrics = {
    "success_rate": MetricsTermCfg(func=LR.m_success, reduce="max"),
    "reach_mm": MetricsTermCfg(
      func=LR.EpisodeMin, params={"fn": LR.m_reach_mm}, reduce="last"
    ),
    "grasp_rate": MetricsTermCfg(func=LR.m_grasp_rate, reduce="max"),
    # mean, not max: a rim-rest that registers once would read 1.0 under max and
    # tell us nothing. This is the fraction of steps actually holding.
    "held_frac": MetricsTermCfg(func=_m_held, reduce="mean"),
    "grip_mm": MetricsTermCfg(func=_m_grip_mm, reduce="min"),
    # Non-zero means _nan_guard fired. Should be 0 for the whole run; if it
    # climbs, the one-shot dump in the log names the offending state.
    "nan_rows": MetricsTermCfg(func=_m_nan, reduce="max"),
    # Phasing no longer gates success -- it is reported so a physics artifact
    # cannot grow unnoticed behind a criterion that silently excluded it.
    "phased_rate": MetricsTermCfg(func=_m_phased, reduce="max"),
    "lift_mm": MetricsTermCfg(func=LR.m_lift_mm, reduce="max"),
    # Absolute heights, in mm, directly comparable to the 36 mm post top --
    # unlike lift_mm, which is measured from the 6 mm resting pose.
    #
    # carry_z_mm is THE one to read. lift_mm is max-reduced and so reports the
    # episode's ceiling; a ring that spikes to 54 mm and then sinks to 30 mm
    # for the whole carry still logs ~48 mm of lift. carry_z_mm reports the
    # FLOOR of the transport leg, which is what the real robot actually drags
    # the peg base with. It must sit above 36 mm with margin.
    "ring_z_mm": MetricsTermCfg(func=LR.m_ring_z_mm, reduce="max"),
    "depart_z_mm": MetricsTermCfg(func=LR.m_depart_z_mm, reduce="last"),
    "carry_z_mm": MetricsTermCfg(func=LR.m_carry_z_min_mm, reduce="last"),
    "clear_rate": MetricsTermCfg(func=LR.m_clear_rate, reduce="max"),
    "goal_mm": MetricsTermCfg(
      func=LR.EpisodeMin, params={"fn": LR.m_goal_mm}, reduce="last"
    ),
  }

  if reward == "e_shaped":
    # Registered HERE, not beside the reward term: `cfg.metrics = {...}` above
    # replaces the dict wholesale, so an earlier entry vanishes without error.
    #
    # How DEEP the jaw went. `grip_mm` cannot answer this -- it is a 3-D
    # distance to the grip point, so a tool 4 mm below the target and one 4 mm
    # above report the same number. tool_dip_mm is the towing-risk number: the
    # jaw tips hang below the tool point, so the deepest excursion before the
    # ring clears is what reaches the peg base.
    cfg.metrics["grasp_z_mm"] = MetricsTermCfg(func=SH.m_grasp_z_mm, reduce="last")
    cfg.metrics["tool_dip_mm"] = MetricsTermCfg(func=SH.m_tool_dip_mm, reduce="last")
    cfg.events["reset_grasp_depth"] = EventTermCfg(
      mode="reset", func=SH.reset_grasp_depth
    )
    # Decoupled from RELEASE_PEN. Gating the METRIC on the penalty's weight
    # meant that turning the penalty off also stopped measuring the thing --
    # which is backwards, and is what happened to the grasp-depth runs.
    cfg.metrics["release_z_mm"] = MetricsTermCfg(
      func=SH.m_release_z_mm, reduce="last"
    )
    # Without the reset the accumulator survives the episode boundary and one
    # that never lets go reports the PREVIOUS episode's release height -- the
    # trap reset_impulse and reset_latches already exist to avoid.
    cfg.events["reset_release_z"] = EventTermCfg(
      mode="reset", func=SH.reset_release_z
    )

  events = {}
  for k, v in cfg.events.items():
    if "block" in k or "goal" in k or k == "place_fixtures":
      continue
    acfg = (v.params or {}).get("asset_cfg")
    if acfg is not None and getattr(acfg, "name", "") == "topdown":
      acfg.name = "liftcam"
      if getattr(acfg, "light_names", None):
        acfg.light_names = ("keylight",)
        acfg.name = "keylight"
      if getattr(acfg, "camera_names", None):
        acfg.camera_names = ("liftcam",)
    events[k] = v
  # Order matters: fixtures draws the peg pose that reset_block reads.
  events["place_fixtures"] = EventTermCfg(mode="reset", func=place_fixtures)
  from surgical_lab.tasks.lift.config.psm import liftpeg_a3_dr as _A3W
  if _A3W.WALL:
    _prev_spec = cfg.scene.spec_fn

    def _with_wall(spec):
      if _prev_spec is not None:
        _prev_spec(spec)
      _A3W.add_wall(spec)

    cfg.scene.spec_fn = _with_wall
  # Alpha3 appearance DR. Off unless LIFTPEG_A3_DR=1, so a3b's distribution is
  # untouched. Registered AFTER place_fixtures because the mount term writes
  # cam_pos for the wrist camera and must not be overwritten by it.
  from surgical_lab.tasks.lift.config.psm import liftpeg_a3_dr as _A3DR
  if _A3DR.A3_DR:
    _A3DR.register(events)
  elif _A3W.WALL:
    # add_wall() hangs the barrier off a mocap body created at the ENV ORIGIN,
    # and the only thing that ever moves it to its hand-placed pose is a3_dr's
    # `wall_pose` reset event -- which was registered only under
    # LIFTPEG_A3_DR=1. With DR off (browser teleop, eval renders, sim-vs-real
    # stills) the barrier therefore sat in the middle of the workspace, a black
    # slab straight through the peg. Register the same event with the jitter
    # zeroed: nominal pose, nominal grey, spec size. This is the DR-off branch
    # only, so the randomised placement training uses is untouched.
    events["a3_wall"] = EventTermCfg(
      mode="reset", func=_A3DR.wall_pose,
      params={"pos": _A3W.WALL_POS, "yaw_deg": _A3W.WALL_YAW_DEG,
              "pos_dr": (0.0, 0.0, 0.0), "yaw_dr": 0.0,
              "grey": (0.05, 0.05), "size_dr": None})
  # `cleared` (ring ever off the post) gates success and the lost test, so it
  # has to be cleared at a known point rather than inferred by whichever
  # manager happens to run first.
  events["reset_latches"] = EventTermCfg(mode="reset", func=LR.reset_latches)
  cfg.events = events

  # --- optional arms, all OFF by default ----------------------------------
  #
  # Each of these is gated on an environment variable that defaults to off, and
  # each is REGISTERED ONLY WHEN ON rather than registered-and-disabled. That
  # distinction is deliberate: rsl-rl prints, and W&B logs, one row per reward /
  # metric / observation term, so a term that exists at weight 0 still changes
  # the shape of the actor's input or the contents of the term table, and every
  # earlier run stops being byte-for-byte comparable. With the flags unset this
  # whole block is a no-op. (Asserted by tests/test_liftpeg_flags_off.py.)
  _apply_proprio(cfg, play=play)
  _apply_force_penalty(cfg)

  # a3b strips the default sun and dims mjlab's headlight ONLY under
  # A3B_UNITY_LAYOUT, which defaults to "0". Liftpeg is a Unity replication
  # unconditionally, and leaving these on gave three light sources where Unity
  # has one (a 0.7 directional sun + a 0.6 headlight + the point light), which
  # blew the scene out. Force them here so the flag cannot silently change the
  # lighting.
  cfg.scene.terrain.lights = ()

  _prev_spec_fn = cfg.scene.spec_fn

  def _unity_lighting(spec):
    if _prev_spec_fn is not None:
      _prev_spec_fn(spec)
    a = LS.HEADLIGHT_AMBIENT
    spec.visual.headlight.ambient = [a, a, a]
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

  cfg.scene.spec_fn = _unity_lighting

  # The ring passing through the post was diagnosed as tunnelling and "fixed"
  # by dropping the timestep to 2 ms. That was wrong. Replaying the failing
  # checkpoint shows the ring crossing at 0.34 mm per step against a 2.0 mm
  # bar, with a post contact present in the 5 preceding steps 100% of the time
  # -- tunnelling is not geometrically available at that speed and the contact
  # was never missing. The cause is contact SOFTNESS (LS.CONTACT_SOLREF), and
  # with that fixed the 2 ms timestep buys nothing, so it goes back to
  # pushblock's 5 ms and the sim runs 2.5x faster.
  # Overridable so the softness/timestep trade can be re-measured rather than
  # re-argued. decimation follows so the 20 ms control step is invariant -- the
  # policy must not change when the physics rate does, or the comparison is
  # between two different tasks.
  cfg.sim.mujoco.timestep = float(os.environ.get("LIFTPEG_DT", "0.002"))
  cfg.decimation = max(1, int(round(0.02 / cfg.sim.mujoco.timestep)))
  # solref's restitution time constant must stay at ~2x the timestep or a
  # driven contact goes soft again. Asserted rather than commented because the
  # two live in different files and the failure is silent.
  assert LS.CONTACT_SOLREF[0] >= 2.0 * cfg.sim.mujoco.timestep - 1e-9, (
    f"CONTACT_SOLREF {LS.CONTACT_SOLREF[0]} is under 2x the timestep "
    f"{cfg.sim.mujoco.timestep}; contacts will be too soft to hold the ring"
  )
  # The sector collision model is many more geoms than the three bars, and the
  # ring resting on the plate alone puts every sector's bottom face in contact.
  # Measured need is ~77 at reset against a default that is far lower, and the
  # failure is a hard ValueError from put_data, not a silent degradation.
  if LS.BLOCK_COLLISION == "sectors":
    cfg.sim.nconmax = max(getattr(cfg.sim, "nconmax", 0) or 0, 256)

  rec = getattr(cfg, "recorders", None)
  if rec and "policy_cam_video" in rec:
    # ON for vision runs, OFF for state runs.
    #
    # Vision: the camera is already rendered every step for the policy, so the
    # recorder only copies frames -- and crucially it logs through the
    # TRAINER's own wandb handle. The sidecar cannot: two processes writing one
    # wandb run means the trainer owns the step counter and the sidecar's
    # log() calls are dropped from history. Its mp4s uploaded and then had no
    # panel to appear in, which looked exactly like "no videos".
    #
    # State: there is no camera otherwise, so the recorder forces a render pass
    # over every env and costs 41% throughput (15442 -> 9150 steps/s at 4096
    # envs). Use the sidecar there instead.
    _vision_run = "camera" in cfg.observations
    if os.environ.get("LIFTPEG_INTRAIN_VIDEO", "1" if _vision_run else "0") == "1":
      r = rec["policy_cam_video"]
      # Film what the ACTOR receives, not a fixed sensor. The wrist is
      # rendered wide and fisheye-warped before the policy sees it, and the
      # side-by-side arm is a concatenation of two sensors -- filming
      # liftcam_cam in either case shows a front-camera view the actor is
      # never given, which reads as a miswired run when you watch the clip.
      if CAM_VIEW == "wrist":
        r.obs_func = _camera_wrist
      elif CAM_VIEW == "sbs":
        r.obs_func = _camera_sbs
      else:
        r.sensor_name = "liftcam_cam"
      r.num_track = 8            # 8 x 2100 x 84x84x3 = 356 MB of GPU buffer
      r.max_frames = 2100        # episodes are 2000 steps; do not truncate
      r.interval_s = float(os.environ.get("LIFTPEG_VIDEO_EVERY_S", "600"))
      r.upscale = 3
    else:
      rec.pop("policy_cam_video", None)

  # Privileged-state teacher: no camera anywhere. The actor reads the same
  # 12-D state the critic already gets, so there is no CNN and no render pass
  # -- which was ~95% of an iteration's cost. Its purpose is diagnostic as much
  # as practical: if state-based PPO solves the lift, perception is the
  # bottleneck and the distillation path is right; if it plateaus at ~1 mm too,
  # the problem is exploration / the binary jaw, not perception.
  if state_only:
    cfg.scene.sensors = ()
    cfg.observations.pop("camera", None)
    cfg.recorders = {}
    # Swap in the 22-D actor-grade state. The 12-D critic vector carries no
    # ring orientation and no jaw state, so an actor reading it could not tell
    # two ring orientations apart -- see _teacher_state.
    cfg.observations["critic"].terms["state"] = ObservationTermCfg(
      func=_shaped_state if reward == "e_shaped" else _teacher_state)

    # RENDER-ONLY camera, for wandb rollout videos. A state run is otherwise
    # blind to us: the metrics say what the ring did but not how, and the last
    # collision bug was found by watching a video, not by reading a number.
    #
    # The sensor is added to the SCENE but never to cfg.observations, and that
    # is asserted below rather than trusted: a camera leaking into an
    # observation group would silently turn the privileged-state teacher into
    # a vision policy, and every comparison against the old runs would be
    # invalid while still looking perfectly healthy.
    #
    # Off by default -- it costs a render pass on a run whose whole point is
    # not having one.
    if os.environ.get("LIFTPEG_STATE_VIDEO", "0") == "1":
      from surgical_lab.mdp.recorders import PolicyCamVideoRecorderCfg

      vid_res = int(os.environ.get("LIFTPEG_STATE_VIDEO_RES", "128"))
      cfg.scene.sensors = (
        CameraSensorCfg(
          name="rollout_cam",
          camera_name="liftcam/liftcam",
          height=vid_res,
          width=vid_res,
          data_types=("rgb",),
          # Groups 0-2 are the VISUAL meshes; group 3 is collision. Rendering
          # the visual set is what the operator recognises, but it is also why
          # the jaw looks buried in the ring -- see
          # docs/liftpeg_collision_vs_visual.md before reading a video as a
          # physics bug.
          enabled_geom_groups=(0, 1, 2),
          use_shadows=True,
          use_textures=True,
        ),
      )
      cfg.recorders["policy_cam_video"] = PolicyCamVideoRecorderCfg(
        sensor_name="rollout_cam",
        num_track=8,
        interval_s=float(os.environ.get("LIFTPEG_STATE_VIDEO_EVERY_S", "900")),
        fps=int(round(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation))),
        upscale=2,
      )
      assert "camera" not in cfg.observations, (
        "render camera leaked into the observation groups -- the state teacher "
        "would be reading pixels")
      for _g, _grp in cfg.observations.items():
        for _t in _grp.terms:
          assert "cam" not in _t, f"camera term {_t!r} in observation group {_g!r}"

  if EXTRA_DR:
    # Geom names come from the built spec, not a literal list: the sector count
    # is configurable (LIFTPEG_BLOCK_SECTORS) and the "bars" variant has three
    # geoms instead of twelve, so a hardcoded list would silently randomise a
    # subset and look like it worked.
    n_sec = LS.BLOCK_SECTORS if LS.BLOCK_COLLISION == "sectors" else len(
      LS.BLOCK_BAR_YAWS_DEG)
    _block_geoms = tuple(f"block_bar{i}" for i in range(n_sec))
    _add_extended_dr(cfg, _block_geoms, ("peg_post", "peg_plate"))

    # ORDER matters: place_fixtures reads the drawn fovy so the visibility
    # check can use the actual value rather than a nominal no env has. Events
    # run in dict order and _add_extended_dr appends, so move camera_fovy ahead
    # of place_fixtures.
    _order = [k for k in cfg.events if k != "camera_fovy"]
    _order.insert(_order.index("place_fixtures"), "camera_fovy")
    cfg.events = {k: cfg.events[k] for k in _order}
    _e = list(cfg.events)
    assert _e.index("camera_fovy") < _e.index("place_fixtures")

  # Guard EVERY observation term, not just the state one. joint_pos is what was
  # NaN when the 11-hour run died, and it was unguarded.
  for _grp in cfg.observations.values():
    for _name, _term in _grp.terms.items():
      if not getattr(_term.func, "_liftpeg_guarded", False):
        _term.func = _guarded(_term.func)
        _term.func._liftpeg_guarded = True

  return cfg
