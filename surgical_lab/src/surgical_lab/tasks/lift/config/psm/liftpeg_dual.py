"""LiftPeg with TWO PSMs -- a proof-of-concept scene for bimanual dVRK tasks.

Everything about the single-arm task (peg, ring, goal, camera, DR, reward,
terminations) is inherited unchanged from `psm_lift_peg_a3b_env_cfg`; this only
adds a second arm, `robot2`, with its own action terms, reset/DR events and
observations. The reward still scores `robot` alone: the point here is to find
out whether the scene, the physics and the action pipeline survive a second
arm, not to define a bimanual task yet.

Placement. `robot2` is `robot` rotated by DUAL_PSM2_YAW_DEG about the vertical
and translated so its nominal tool point lands at DUAL_PSM2_TOOL_XY. The
default (180 deg, 40 mm to +x of arm 1's tool) puts the two arms facing each
other across the peg with their tools side by side, the way two PSMs approach a
shared site on the real cart. The Cartesian actions integrate in each arm's own
base frame (FK/IK on its joints), so the mount pose needs no action changes.

Arm-arm contact. The stock PSM collision set only lets the jaw geoms touch the
world (contype 0 / conaffinity 1), so two of them would pass straight through
each other. With DUAL_PSM_ARM_CONTACT=1 (default) each arm gets its own contype
bit and has the OTHER arm's bit in its conaffinity, so every collision geom of
one arm -- shaft included -- collides with the other arm, while still never
colliding with itself or with anything it did not collide with before.

No wrist-camera bracket. The plan for this scene is state-based only, so the
printed bracket (PSM_WRIST_MOUNT) is stripped from both arms whatever that flag
says. It is a visual-only geom; the wrist camera itself stays, unrendered, so
the bracket DR event (a3_mount_pose) finds no bracket and skips.

No barrier wall. The single-arm scene's black slab (LIFTPEG_A3_WALL) is a
visual occluder for the camera policy; with a second arm in the workspace it is
not wanted, so it is left out here whatever that flag says.
"""

import copy
import math
import os

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.spec_config import CollisionCfg

from surgical_lab.tasks.lift.config.psm import liftpeg_a3_dr as A3DR
from surgical_lab.tasks.lift.config.psm.liftpeg_a3b_env_cfg import (
  psm_lift_peg_a3b_env_cfg,
)
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b


def _floats(name: str, default: str) -> tuple[float, ...]:
  return tuple(float(x) for x in os.environ.get(name, default).split(","))


PSM2_TOOL_XY = _floats("DUAL_PSM2_TOOL_XY", f"{a3b.ANCHOR_X + 0.040},{a3b.ANCHOR_Y}")
PSM2_YAW_DEG = float(os.environ.get("DUAL_PSM2_YAW_DEG", "180"))
ARM_CONTACT = os.environ.get("DUAL_PSM_ARM_CONTACT", "1") == "1"

# Collision bits. Bit 0 is the world (terrain, peg, ring, goal), which the jaw
# geoms already reach through conaffinity 1.
_WORLD, _ARM1, _ARM2 = 1, 2, 4
_GRIPPER_RE = r"^(lg|rg|tool_pitch_capsule).*_collision$"


def _quat_mul(a, b):
  aw, ax, ay, az = a
  bw, bx, by, bz = b
  return (
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  )


def _arm_collision(base: CollisionCfg, own: int, other: int) -> CollisionCfg:
  """`base` with arm-arm contact switched on, the world contacts unchanged."""
  c = copy.deepcopy(base)
  c.contype = {_GRIPPER_RE: own, ".*_collision": own}
  c.conaffinity = {_GRIPPER_RE: _WORLD | other, ".*_collision": other}
  return c


def _without_wrist_mount(spec_fn):
  def fn():
    spec = spec_fn()
    for g in list(spec.geoms):
      if g.name == "wrist_camera_mount":
        spec.delete(g)
    return spec

  return fn


def tool_pos(env, asset_name: str = "robot2") -> torch.Tensor:
  """Tool midpoint of an arm, env frame."""
  r = env.scene[asset_name]
  return r.data.site_pos_w[:, r.site_names.index("tool_midpoint")] - env.scene.env_origins


# --- arm-arm contact: sensor, penalty, termination, metric -------------------
#
# Any contact between the two arms is treated as a failure: on the real cart it
# is two instruments clashing inside the patient. One MuJoCo contact sensor
# watches the whole subtree of arm 1 against the whole subtree of arm 2, so it
# fires on ANY arm-arm contact the physics can produce. The parts that can touch
# are the ones with collision geoms: the instrument shaft (main_insertion_link),
# wrist roll and pitch, and the jaws (shaft-shaft, shaft-jaw and shaft-wrist
# contacts all verified to fire).
#
# KNOWN GAP: the parallelogram links, the instrument carriage (pitch_end_link)
# and the base have NO collision geometry. With the default mount (both shafts
# vertical, 40 mm apart) the two carriages sit ~3 mm apart at the home pose and
# their meshes overlap under motion, unseen by this sensor. Fixing that needs
# collision geoms on those links AND a mount that keeps them apart.
ARM_CONTACT_SENSOR = "arm_arm_contact"
# Cost of one contact step, in return units (the reward manager scales weights
# by the step dt, so the weight below is -penalty / dt). Sized per task: it
# should outweigh what the policy could earn in the rest of the episode.
ARM_CONTACT_PENALTY = float(os.environ.get("DUAL_ARM_CONTACT_PENALTY", "10"))
ARM_CONTACT_TERMINATE = os.environ.get("DUAL_ARM_CONTACT_TERMINATE", "1") == "1"


def arm_arm_contact(env) -> torch.Tensor:
  """(num_envs,) bool: any part of arm 1 touching any part of arm 2."""
  found = env.scene[ARM_CONTACT_SENSOR].data.found
  return (found.reshape(found.shape[0], -1) > 0).any(dim=-1)


def arm_arm_contact_f(env) -> torch.Tensor:
  return arm_arm_contact(env).float()


def add_arm_contact_safety(cfg, penalty: float = ARM_CONTACT_PENALTY,
                           terminate: bool = ARM_CONTACT_TERMINATE):
  """Penalise (and by default end the episode on) any arm-arm contact.

  Call AFTER the task sets cfg.rewards / terminations / metrics, since those
  are often replaced wholesale."""
  from mjlab.managers.metrics_manager import MetricsTermCfg
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  if not ARM_CONTACT:
    raise ValueError("arm-arm contact safety needs DUAL_PSM_ARM_CONTACT=1: "
                     "with it off the arms pass through each other and no "
                     "contact is ever generated to detect.")
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name=ARM_CONTACT_SENSOR,
      primary=ContactMatch(mode="subtree", pattern="mocap_base", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="mocap_base", entity="robot2"),
      fields=("found",),
    ),
  )
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  cfg.rewards["arm_contact"] = RewardTermCfg(func=arm_arm_contact_f,
                                             weight=-penalty / dt)
  if terminate:
    cfg.terminations["arm_contact"] = TerminationTermCfg(func=arm_arm_contact)
  # max: 1 if the episode had ANY arm-arm contact. Should go to ~0 in training.
  cfg.metrics["arm_contact_rate"] = MetricsTermCfg(func=arm_arm_contact_f,
                                                   reduce="max")
  return cfg


def psm_lift_peg_dual_env_cfg(
  play: bool = False, reward: str = "e_shaped", state_only: bool = True
):
  # The wall flag is read while the base cfg is built (spec edit + pose event),
  # so turn it off for that call only; the single-arm tasks keep their wall.
  wall, A3DR.WALL = A3DR.WALL, False
  try:
    cfg = psm_lift_peg_a3b_env_cfg(play=play, reward=reward, state_only=state_only)
  finally:
    A3DR.WALL = wall

  ents = dict(cfg.scene.entities)
  r1 = ents["robot"]
  r2 = copy.deepcopy(r1)

  # base2 = T2 + R(yaw) (base1 - T1): the same arm, swung about its tool point.
  yaw = math.radians(PSM2_YAW_DEG)
  cy, sy = math.cos(yaw), math.sin(yaw)
  bx, by, bz = r1.init_state.pos
  dx, dy = bx - a3b.ANCHOR_X, by - a3b.ANCHOR_Y
  r2.init_state.pos = (
    PSM2_TOOL_XY[0] + cy * dx - sy * dy,
    PSM2_TOOL_XY[1] + sy * dx + cy * dy,
    bz,
  )
  qz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
  r2.init_state.rot = _quat_mul(qz, r1.init_state.rot)

  r1.spec_fn = r2.spec_fn = _without_wrist_mount(r1.spec_fn)
  if ARM_CONTACT:
    (c1,) = r1.collisions
    r1.collisions = (_arm_collision(c1, _ARM1, _ARM2),)
    r2.collisions = (_arm_collision(c1, _ARM2, _ARM1),)
  ents["robot2"] = r2
  cfg.scene.entities = ents

  # Actions: arm 1's terms, then the same terms for arm 2 -> [xyz1, jaw1, xyz2, jaw2].
  for k in list(cfg.actions):
    a = copy.deepcopy(cfg.actions[k])
    a.entity_name = "robot2"
    cfg.actions[f"{k}_2"] = a

  # Reset + physics DR for arm 2. Appearance DR (tool colours) is left on arm 1
  # only; it is a vision concern and this arm is not in a policy's view yet.
  events = dict(cfg.events)
  for k in ("reset_base", "reset_robot_joints", "lower_tool", "home_pose",
            "pd_gains", "encoder_bias", "joint_friction", "joint_damping"):
    if k not in events:
      continue
    e = copy.deepcopy(events[k])
    e.params = dict(e.params or {})
    if k == "home_pose":
      e.params["asset_name"] = "robot2"
    elif "asset_cfg" in e.params:
      e.params["asset_cfg"].name = "robot2"
    else:
      e.params["asset_cfg"] = SceneEntityCfg("robot2")
    events[f"{k}_2"] = e
  cfg.events = events

  # Critic (and, for the state task, actor) also sees arm 2.
  terms = cfg.observations["critic"].terms
  jp = copy.deepcopy(terms["joint_pos"])
  jp.params["asset_cfg"].name = "robot2"
  terms["joint_pos_2"] = jp
  tp = copy.deepcopy(terms["joint_pos"])
  tp.func, tp.params = tool_pos, {"asset_name": "robot2"}
  terms["tool_pos_2"] = tp
  return cfg
