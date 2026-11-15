"""Faithful mjlab replication of the muse/Unity `ABL_a3b_resnet_e2e_chain` env
(branch fresh-pushblockwide-build): wide fixed goal bar, top-down camera,
planar 2-D increments, potential-based shaping, DR matched to the a3b config.

Everything is deliberately matched to Unity (see docs/direct_s2r_notes_for_later.md
for the improvements we chose NOT to make yet). Unit conversion: 1 Unity unit
= 5 cm, inferred from Unity's 0.01 u/step tool motion vs the real robot's
0.0005 m/step (pushblock.yaml position_speed_multiplier).

Unity ground truth (extracted from the .unity/.prefab files, PushBlockVec.unity
+ PushBlockVecEnv.prefab + LNDBaseEnv.prefab + scene overrides):
  goal   : green Plane bar 6 u x 1.5 u at (0, 0, -2.15) rel. tool start; FIXED
  block  : cube 0.5 u (DR scale 0.3-0.5, mass 0.01-0.3, red-ish), yaw 0-90 deg,
           spawned in a ~3 u square around the tool start, goal band excluded
  camera : top-down at height ~5 u, fov 60, DR pos +-0.1 u xy / +0-0.4 u z,
           pitch 87-93 deg, yaw +-3 deg
  action : 2 continuous in [-1,1] -> tip moves action * 0.01 u per 0.02 s
           decision (= 0.0005 m real), planar only, no vertical, no jaw
  reward : potential delta of [1*exp(-2u * (|tip-block| - 1.5*s)+)
           + 2*linear_z(block->goal) - step/2000], +2 on success
  success: block centre inside the goal band; MaxStep 2000 (40 s at 50 Hz)
"""

import math
import os
from typing import Any

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import last_action as base_mdp_last_action
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from surgical_lab.mdp import CartesianDeltaActionCfg
from surgical_lab.mdp.recorders import PolicyCamVideoRecorderCfg
from surgical_lab.robots.psm.psm_constants import get_psm_robot_cfg
from surgical_lab.tasks.push.config.psm.env_cfg import (
  PSM_MAIN_JOINTS,
  _psm_ik_fn,
  psm_push_env_cfg,
)
from surgical_lab.robots.psm.psm_kinematics import compute_psm_fk

# ---------------------------------------------------------------------------
# Unity -> metric constants (1 u = 5 cm)
U = 0.05
# Workspace anchor: the tool tip's default (x, y) in the env/world frame; the
# whole Unity layout is expressed relative to the tool start. Verified against
# a rendered frame during bring-up; adjust here if the PSM default pose moves.
ANCHOR_X = 0.0
ANCHOR_Y = 0.0127  # measured tool_midpoint default (env frame); was a guess before
# "Camera-down"/goal direction is +y (away from the PSM base).
# --- UNITY-EXACT LAYOUT (A3B_UNITY_LAYOUT=1) -------------------------------
# From a live scene dump of PushBlockVec (SceneDumper.cs, edit+play) plus
# muse/config/push_block/pushblock_withdr.yaml. 1 u = 50 mm EXACTLY: the
# psm_lnd import carries lossyScale = 20 and the URDF is metres.
#
# Geometry: frame 282 mm wide (camera 5 u up, fovy 60, aspect 1), tool->goal
# 3.922 u = 196 mm, block 15-25 mm (scale DR), shaft entering at 31.5 deg off
# vertical from an RCM ~0.26 m up and ~0.15 m behind the tool, point light at
# the camera with position/intensity DR, jaw CLOSED.
#
# TWO EYES-OPEN DEVIATIONS FROM PHYSICAL REALITY, both Unity's own:
#   1. The arm pose needs RCM-to-tip 0.29-0.42 m across the workspace, beyond
#      the real PSM's insertion range (0.05..0.24 -> max ~0.17 m). With the
#      REAL range the tip stalls at y=0.098 against a goal band starting at
#      0.171 (probe_reach.py) -- the task is unsolvable. Unity's arm simply
#      ignores the limit, so under UNITY_LAYOUT the sim model's insertion
#      range is EXTENDED to (0.05, 0.55). The deploy contract (image -> 2D
#      delta) never sees a joint value, and the Unity-trained policy with
#      this exact un-physical arm transferred to the real rig at 100%.
#   2. Real's frames are ~1.7x tighter than Unity's (block 74 px real vs
#      26 px unity). Unity transferred ACROSS that scale gap, so we inherit
#      it rather than fight it.
UNITY_LAYOUT = os.environ.get("A3B_UNITY_LAYOUT", "0") == "1"
# Tool home-pose ablation under UNITY_LAYOUT:
#   A3B_UNITY_HOME=1 (default) Unity's tool home: jaw CLOSED, tip at the
#                    dump's 13.6 mm push height
#   A3B_UNITY_HOME=0 our previous tool home on the same scene: jaw OPEN
#                    (JAW_OPEN below) and the tool lowered to ~1 mm
UNITY_HOME = os.environ.get("A3B_UNITY_HOME", "1") == "1"
if UNITY_LAYOUT:
  GOAL_CENTER_Y = ANCHOR_Y + 3.922 * U    # 0.2088 m
else:
  GOAL_CENTER_Y = ANCHOR_Y + 2.15 * U     # 0.1202 m

# --- Unity arm pose (A3B_UNITY_LAYOUT=1) -----------------------------------
# Shaft 31.5 deg off vertical at reset, RCM at Unity's measured position
# (insertion tube pivot (-0.59, 5.25, 4.86) u):
#   RCM target = (0, ANCHOR_Y - 3.088 u, 5.25 u) = (0, -0.142, 0.2625)
# Probe facts (probe_rcm.py): shaft angle EQUALS outer_pitch; base->RCM offset
# is (y -0.0886, z +0.0876).
#
# CORRECTION (see docs/lab_pc_port.md): the shaft now hangs VERTICALLY, as it
# does in Unity and on the real rig. The old 31.5 deg tilt came from treating
# "Unity's measured RCM (insertion tube pivot (-0.59, 5.25, 4.86) u)" as a
# fixed point -- but those are main_insertion_link's coordinates in the PLAY
# dump, and that body rides the TRANSLATING shaft: it reads
# (-0.588, 5.247, 4.860) in PLAY against (0, 11.684, 1.772) in EDIT. PLAY is a
# mid-episode snapshot with the arm swung out to 11 deg, so the angle was
# reverse-engineered from a pose Unity is never at rest in. In EDIT the tool
# tip sits at x = 0.00001, z = 1.77185 -- directly beneath base_link
# (0, 1.772), 0.00 deg -- matching the "tool resets to local z = 1.77 u" the
# camera block quotes.
#
# Base position follows from that. Tilted, the base had to sit BEHIND the
# workspace so the slanted shaft still landed the tip on the anchor, and each
# insertion depth needed its own offset because depth slid the tip goalward
# along the shaft. Vertical, the tip hangs directly below the RCM: depth
# changes height only, so ONE base position serves every variant, and putting
# the RCM on the anchor means base y = ANCHOR_Y + 0.0886. Verified in sim --
# straightening the arm without moving the base left the tip 188 mm short of
# the anchor, i.e. off-frame in a 282 mm view.
SHAFT_PITCH = 0.0  # vertical, as in Unity and on the real rig
_PITCH_INIT = SHAFT_PITCH
_BASE_TO_RCM_Y = 0.0886  # probe_rcm.py
PSM_BASE_POS = (0.0, ANCHOR_Y + _BASE_TO_RCM_Y, 0.1749)
GOAL_HALF_X = 3.0 * U                     # bar is 6 u wide  -> half 0.15 m
GOAL_HALF_Y = 0.75 * U                    # 1.5 u deep       -> half 0.0375 m
# Unity DR draws localScale U(0.3, 0.5) EVERY reset -> mean cube 0.4 u = 20 mm
# full width. The legacy 12.5 mm was calibrated against real frames in the
# legacy 2.4x-tighter framing.
BLOCK_HALF = (0.2 * U) if UNITY_LAYOUT else (0.25 * U * 0.5)
SPAWN_HALF = 1.5 * U                      # ~3 u spawn square -> +-0.075 m
# Camera solved from the Unity reference frames rather than the config (the
# frames span only ~2.3 u vertically): height 0.115 m with fovy 60 gives a
# 0.133 m-tall footprint; centering the view at y=0.085 puts the tool stub at
# ~15% of the frame and the goal band top at ~70%, band running off the bottom
# -- the Unity composition.
# Camera height and FOV move TOGETHER: the ground coverage
#   frame_half = CAM_HEIGHT * tan(CAM_FOVY/2)
# is what the whole layout (block spawn band, camera DR) is expressed against,
# so raising the camera without narrowing the lens would silently rescale the
# task. Override both via A3B_CAM_HEIGHT / A3B_CAM_FOVY; CAM_FOVY defaults to
# whatever preserves the original 58.3 mm half-width at the chosen height.
#
# WHY THIS IS A KNOB: at the original 0.101 m / 60 deg the PSM's 416 mm shaft
# extends 4.1x PAST the camera, so it flares into a wide trapezoid. Measured
# over 300 frames, the tool covers 18.9% of the top image row against 5.3% on
# the real robot and 7.6% in Unity, tapering 4.18x top-to-tip against real's
# 2.09x. The scene was shrunk to Unity's framing (12.5 mm block vs Unity's
# 25 mm) and the camera pulled in to match, but the robot stayed full size.
# Raising the camera and narrowing the lens keeps coverage and removes the
# flare.
# DEFAULT RAISED 0.101 -> 0.30 on 2026-08-01 (0.25 first, then tuned to 0.30
# to match the tool's apparent WIDTH, not just its taper). The FRAMING is unchanged --
# CAM_FOVY below narrows to hold frame_half at 58.3 mm -- so goal composition,
# block-to-frame ratio and spawn band are all untouched. Only the tool's
# perspective changes, which is the one thing that was wrong.
#
# Verified by measurement rather than by matching Unity's nominal units: a pass
# that rebuilt the scene at 1 u = 0.05 m (block 25 mm, camera 5 u, fovy 60) made
# things WORSE -- the goal band stopped running off the bottom of frame
# (top edge row 32 vs real's 65) and green coverage went 22% -> 28%. The a3b
# Unity build's camera does not obey that conversion; its rendered frames span
# only ~2.3 u, which is why the original value was solved from frames in the
# first place. Measured composition, 300 frames per domain:
#
#     domain        green%   goal top row   runs off bottom   block px
#     real           21.6%       64.8            100%            71
#     unity          21.3%       65.8             96%            26
#     ours (0.101)   22.2%       64.8            100%            80
#
# i.e. our framing already matched REAL more closely than Unity's own frames do.
# Do not "fix" the geometry to Unity's config numbers.
#
# HEIGHT CHOSEN BY TOOL WIDTH. Raising the camera shrinks ELEVATED objects while
# leaving ground-plane ones untouched (frame_half is held), because the
# magnification on something at height z is H/(H-z). Total tool pixel area,
# 200 frames each, against real's 211 px:
#     H=0.25 (fovy 26.3)  256 px  1.21x   block 63  green 22.0%
#     H=0.30 (fovy 22.0)  214 px  1.01x   block 65  green 22.1%   <- chosen
#     H=0.32 (fovy 20.7)  211 px  1.00x   block 57  green 22.0%
#     H=0.35 (fovy 18.9)  192 px  0.91x   block 68  green 22.2%
# 0.30 over 0.32 because its block size (65) is nearer real's 71 at equal tool
# match. Framing is invariant across all of them, as designed.
#
# RESIDUAL, not fixable by camera height: our per-row profile still differs from
# real (row5 3.0 vs 4.7, row15 6.1 vs 4.3). Real's shaft is widest at the top and
# tapers monotonically; ours bulges mid-frame at the wrist/jaw assembly, which is
# genuinely chunkier in the PSM model and sits near the floor where camera height
# barely acts (row30 moves only 4.55 -> 4.05 from H=0.25 to 0.60). Closing that
# needs the tool mesh, not the camera.
CAM_HEIGHT = float(os.environ.get("A3B_CAM_HEIGHT", str(5.0 * U) if UNITY_LAYOUT else "0.30"))
_BASE_HALF = 0.101 * math.tan(math.radians(60.0) / 2.0)  # 0.0583 m, the design coverage
CAM_FOVY = float(
  os.environ.get(
    "A3B_CAM_FOVY",
    "60.0" if UNITY_LAYOUT
    else str(2.0 * math.degrees(math.atan(_BASE_HALF / CAM_HEIGHT))),
  )
)
# Unity resets the tool to local z = 1.772 u with the camera at local z ~ 0.
CAM_CENTER_Y = (ANCHOR_Y + 1.772 * U) if UNITY_LAYOUT else 0.0449
MAX_DECISIONS = 2000.0

# --- ablation knobs -------------------------------------------------------
# Unity's a3b_resnet_e2e.yaml is the only config we have seen actually solve
# this task (28M steps). Our recipe drifted from it, and 253M steps of job
# 52194268 never moved off a ~20% success plateau. These two env vars let the
# sweep move one factor at a time without five copies of this file.
#
#   A3B_TIME_DRAG=1   (default) include Unity's -step/MaxStep in the potential
#   A3B_REGULARIZERS=0 (default) omit action_rate_l2 + joint_pos_limits, which
#                      Unity's reward does not have at all
USE_TIME_DRAG = os.environ.get("A3B_TIME_DRAG", "1") == "1"
USE_REGULARIZERS = os.environ.get("A3B_REGULARIZERS", "0") == "1"


def get_a3b_block_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="block")
  body.add_freejoint(name="block_joint")
  body.add_geom(
    name="block_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(BLOCK_HALF,) * 3,
    mass=0.2 if UNITY_LAYOUT else 0.05,  # Unity Rigidbody base mass (DR overrides)
    rgba=(0.8, 0.15, 0.15, 1.0),
    # Slide friction 0, NOT the mujoco default 1.0. MuJoCo combines contact
    # friction as the element-wise MAX of the two geoms (at equal priority),
    # so a default-friction block pinned every block-floor contact at 1.0 and
    # silently disabled the floor_friction DR (Unity: 0.0-0.3). Measured
    # consequence: at unity tool height (13.6 mm, ~90% of a 15 mm DR block)
    # the floor grips and the block SOMERSAULTS 180 deg at friction 1.0,
    # while at Unity's <=0.3 it slides for every DR size (probe_roll2.py).
    # With a near-zero value here the pair max IS the floor's per-reset DR
    # draw (floor range starts at 0.05). Not exactly 0.0: mujoco_warp warns
    # friction < MJ_MINMU with condim=3 can NaN. Legacy path keeps the
    # (buggy but trained-with) mujoco default 1.0.
    friction=(0.01, 0.005, 0.0001) if UNITY_LAYOUT else (1.0, 0.005, 0.0001),
  )
  return spec


def get_wide_goal_spec() -> mujoco.MjSpec:
  """Fixed green goal bar + the top-down agent camera.

  The goal never moves or changes colour (Unity's ResetGoal is disabled on the
  wide branch), so it is a static visual-only geom. The camera rides a mocap
  body so dr.body_pos/body_quat can jitter its pose like Unity's camera DR.
  """
  spec = mujoco.MjSpec()
  # Mocap: in mujoco_warp there is ONE model and N worlds, so fixed geometry
  # cannot differ per env. Mocap poses live in per-world data; the
  # place_fixtures reset event writes env_origin + design pose for every env
  # (the same pattern PushCommand uses for its target).
  goal = spec.worldbody.add_body(name="goal", mocap=True)
  # PLANE, not BOX. A box goal showed a hard 1-pixel black seam along its near
  # edge: the goal sits at y~0.120 while the camera is at y~0.054, so a
  # top-down view catches the box's vertical side wall, and mujoco_warp's
  # shader returns zero diffuse AND zero specular once ndotl == 0 (render.py).
  # A vertical face lit from above therefore keeps only the ambient term and
  # renders near-black. The wall projects to ~1 px at any thickness (the view
  # is near-vertical), so thinning it only changed how OFTEN it landed on a
  # pixel centre -- 0.1 mm hit 6% of camera draws, 6 mm hit 45%. Elevation
  # changed nothing, ruling out self-shadowing and z-fighting. A plane has no
  # vertical face at all: 0 of 256 draws. size = (half_x, half_y, grid spacing).
  goal.add_geom(
    name="goal_geom",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(GOAL_HALF_X, GOAL_HALF_Y, 0.01),
    pos=(ANCHOR_X, GOAL_CENTER_Y, 0.0003),
    rgba=(0.2, 0.8, 0.2, 1.0),
    contype=0,
    conaffinity=0,
  )
  return spec


def get_topdown_cam_spec() -> mujoco.MjSpec:
  """The agent camera on its own mocap body (placed per env by place_fixtures)."""
  spec = mujoco.MjSpec()
  cam_frame = spec.worldbody.add_body(name="topdown_cam_frame", mocap=True)
  cam_frame.pos = (ANCHOR_X, CAM_CENTER_Y, CAM_HEIGHT)
  # A MuJoCo camera at identity already looks along -z (straight down). Rotate
  # 180 deg about z so image-down = world +y: the goal bar sits at the bottom of
  # the frame exactly as in Unity.
  cam_frame.add_camera(
    name="topdown_cam",
    fovy=CAM_FOVY,
    quat=(0.0, 0.0, 0.0, 1.0),
  )
  if UNITY_LAYOUT:
    # Unity's scene light is a POINT light (the object is just NAMED "Spot
    # Light"), warm white, base pose (0, 5, 0) u -- i.e. at the camera. Riding
    # the same mocap frame gives per-env placement for free, and dr.light_pos
    # (a LOCAL offset on this body) becomes the per-env position DR that
    # light_px/py/pz is in Unity. mujoco_warp shadows non-directional lights
    # positionally (shadow ray toward lightpos, render.py:466/489), so the
    # splayed, off-axis shadows Unity shows finally exist here too.
    light = cam_frame.add_light(name="scene_light")
    light.type = mujoco.mjtLightType.mjLIGHT_POINT
    light.pos = (0.0, 0.0, 0.0)
    light.castshadow = True
    light.diffuse = (1.0, 0.99, 0.94)
    # Distance falloff: brightness = 1/(a0 + a1 d + a2 d^2). a2 = 1/0.30^2 puts
    # unit gain at the camera-height floor distance and less at the corners --
    # the moving hotspot unity/real both show. The old "spot renders flat"
    # finding was zero attenuation coefficients, not a renderer limitation.
    light.attenuation = (0.0, 0.0, 1.0 / (CAM_HEIGHT * CAM_HEIGHT))
  # Debug-only oblique views, for eyeballing the 3-D layout (tool length vs goal
  # distance vs block size) which a top-down view cannot show. Off by default so
  # training never pays for them; A3B_DEBUG_CAMS=1 to enable.
  if os.environ.get("A3B_DEBUG_CAMS", "0") == "1":
    import numpy as _np

    def _look_at(eye, target):
      f = _np.array(target, float) - _np.array(eye, float)
      f /= _np.linalg.norm(f)
      # MuJoCo camera looks down its -z with +y up.
      zc = -f
      up = _np.array([0.0, 0.0, 1.0])
      xc = _np.cross(up, zc); xc /= _np.linalg.norm(xc)
      yc = _np.cross(zc, xc)
      R = _np.stack([xc, yc, zc], axis=1)
      q = _np.empty(4)
      mujoco.mju_mat2Quat(q, R.flatten())
      return tuple(float(v) for v in q)

    look = (ANCHOR_X, (ANCHOR_Y + GOAL_CENTER_Y) * 0.5, 0.0)
    span = GOAL_CENTER_Y - ANCHOR_Y
    for nm, eye in (
      ("oblique_cam", (ANCHOR_X + 1.1 * span, look[1] - 0.9 * span, 0.9 * span)),
      ("side_cam", (ANCHOR_X + 1.7 * span, look[1], 0.28 * span)),
    ):
      # NOT mocap: place_fixtures only writes the goal and topdown poses, so a
      # mocap debug cam would sit at the world origin and frame nothing.
      c = spec.worldbody.add_body(name=nm + "_body")
      c.pos = eye
      c.add_camera(name=nm, fovy=45.0, quat=_look_at(eye, look))
  return spec


def _unity_robot_cfg() -> EntityCfg:
  """PSM cfg with the Unity arm pose (see SHAFT_PITCH/PSM_BASE_POS above) and
  named visual geoms so tool-colour DR can target parts individually.

  Deep-copied: get_psm_robot_cfg hands back the module-level HOME_KEYFRAME, and
  mutating it in place would leak the tilted pose into the legacy path."""
  import copy

  from surgical_lab.robots.psm.psm_constants import get_spec

  cfg = copy.deepcopy(get_psm_robot_cfg(tool_type="lnd"))
  cfg.init_state.pos = PSM_BASE_POS
  # The four-bar followers MUST be set with outer_pitch: they are coupled by
  # equality constraints (psm_lnd.xml: bottom/top = -outer_pitch,
  # end/front = +outer_pitch), and a reset that writes only outer_pitch leaves
  # the linkage torn until the solver slams it back -- probe_rcm.py measured
  # the resulting pose 10+ cm off at 0.55 rad.
  cfg.init_state.joint_pos = {
    **cfg.init_state.joint_pos,
    "outer_pitch": _PITCH_INIT,
    "pitch_bottom_joint": -_PITCH_INIT,
    "pitch_end_joint": _PITCH_INIT,
    "pitch_top_joint": -_PITCH_INIT,
    "pitch_front_joint": _PITCH_INIT,
  }

  def _named_spec() -> mujoco.MjSpec:
    spec = get_spec("lnd")
    # The URDF's visual meshes compile unnamed, so SceneEntityCfg(geom_names=)
    # cannot address them. Name each after its parent body.
    counts: dict[str, int] = {}
    for g in spec.geoms:
      if g.name == "":
        body = g.parent.name
        i = counts.get(body, 0)
        counts[body] = i + 1
        g.name = f"{body}_visual_{i}"
    # SIM-ONLY: extend the insertion range to Unity's un-physical workspace
    # (real PSM: 0.05..0.24). See the UNITY-EXACT LAYOUT comment block -- with
    # the real range the Unity-scale task is unsolvable, and Unity's own arm
    # (which transferred at 100%) ignores this limit too.
    for j in spec.joints:
      if j.name == "main_insertion":
        j.range = (0.05, 0.55)
    return spec

  cfg.spec_fn = _named_spec

  # Tool-block contact friction 0.35, from the Unity physics dump: the tool
  # colliders carry NO material (PhysX default 0.6), the block's "Slippery" is
  # 0.1, and PhysX AVERAGES -> 0.35. Our gripper geoms have priority=1, so
  # their value IS the contact value; the previous 1.0 made the tip grab the
  # block ~3x harder than Unity's.
  new_collisions = []
  for col in cfg.collisions:
    col = copy.deepcopy(col)
    fr = dict(col.friction or {})
    for pat, val in list(fr.items()):
      if "lg" in pat or "rg" in pat or "tool_pitch_capsule" in pat:
        fr[pat] = (0.35,) + tuple(val[1:])
    col.friction = fr
    new_collisions.append(col)
  cfg.collisions = tuple(new_collisions)

  # SIM-ONLY stiffness bump on the outer/insertion actuators. At Unity's arm
  # extension (insertion 0.35-0.53) gravity sag on the long lever leaves a
  # tracking error the planar action cannot correct (z is uncontrolled), and
  # the tip drifted up to +22 mm mid-push -- above the 15-16 mm end of the
  # block-scale DR, so small blocks would be ridden over near the goal. 4x
  # stiffness / 2x damping (critical scaling) cuts the sag ~4x. Unity's
  # UrdfJointController is effectively rigid, so this MOVES TOWARD the
  # reference, not away from it.
  acts = []
  for a in cfg.articulation.actuators:
    a = copy.deepcopy(a)
    if any(n in ("outer_yaw", "outer_pitch", "main_insertion") for n in a.target_names_expr):
      a.stiffness = a.stiffness * 4.0
      a.damping = a.damping * 2.0
    acts.append(a)
  cfg.articulation.actuators = tuple(acts)

  # Tool-block contact friction 0.35, matching Unity's dump: the tool
  # colliders carry NO material (PhysX default 0.6) and the block's
  # "Slippery" is 0.1, combined by Average -> 0.35. Our gripper geoms carry
  # priority=1, so THEIR friction is the pair's friction verbatim; the
  # inherited 1.0 made the tip grab the block ~3x harder than Unity.
  for coll in cfg.collisions:
    for pattern in list(coll.friction.keys()):
      f = coll.friction[pattern]
      if isinstance(f, tuple) and len(f) == 3 and f[0] == 1:
        coll.friction[pattern] = (0.35, f[1], f[2])
  return cfg


# ---------------------------------------------------------------------------
# mdp terms (Unity PushBlockAgent semantics)


def _block_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  # env-frame: the design constants (GOAL_CENTER_Y etc.) are per-env.
  return env.scene["block"].data.root_link_pos_w - env.scene.env_origins


def _tool_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  idx = robot.site_names.index("tool_midpoint")
  return robot.data.site_pos_w[:, idx] - env.scene.env_origins


_BLOCK_GEOM_ID: int | None = None


def _block_full_width(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env block full width in metres, read from the live model.

  Unity re-randomizes the block's localScale on EVERY reset, so the reward's
  `s` term cannot be the design constant once scale DR is on -- a policy would
  otherwise be scored against a block size that is not the one in front of it.
  `geom_size` is (nworld, ngeom, 3) half-extents; the cube is isotropic so
  axis 0 suffices.

  SceneEntityCfg(geom_names=...) does not resolve to ids for this entity (it
  returns slice(None)), so the geom is looked up by its qualified model name
  once and cached.
  """
  global _BLOCK_GEOM_ID
  if _BLOCK_GEOM_ID is None:
    mj = env.sim.mj_model
    names = [mj.geom(i).name for i in range(mj.ngeom)]
    _BLOCK_GEOM_ID = names.index("block/block_geom")
  return env.sim.model.geom_size[:, _BLOCK_GEOM_ID, 0] * 2.0


def _critic_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged state for the value function (9-D).

  The ACTOR is camera-only because the deployed medrct_onnx node can only bind
  an image -- that is the sim2real contract and it does not change. The CRITIC
  is never deployed, so there is no reason to blind it: giving it the geometry
  directly makes value estimates far less noisy than regressing them out of
  84x84 pixels, and it lets us drop the critic's CNN entirely (that second CNN
  was ~half of the PPO update cost).

  Self-contained on purpose: the base push cfg's `ee_to_block`/`block_to_goal`
  terms read the `push_command` manager, which a3b removes (cfg.commands = {}),
  so they cannot be reused here.
  """
  block = _block_pos(env)
  tool = _tool_pos(env)
  goal = torch.tensor(
    (ANCHOR_X, GOAL_CENTER_Y, 0.0), device=env.device, dtype=torch.float32
  ).unsqueeze(0)
  return torch.cat([block - tool, goal - block, tool], dim=-1)


class a3b_shaping:
  """Potential-based shaping identical to Unity's UpdateReward.

  Unity computes potential = 1*ExpDistanceReward(tip, block, 1.5*s, 100, 2)
  + 2*LinearDistanceReward(block.z, goal.z, s, block_start.z) - step/MaxStep and
  rewards the per-step DELTA (AddReward(current - prev)). Distances are in
  Unity units, so the exp multiplier 2/u becomes 2/U per metre. The linear term
  is a ratio and needs no unit conversion.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._prev = torch.zeros(env.num_envs, device=env.device)
    self._block_start_y = torch.zeros(env.num_envs, device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._fresh[env_ids] = True

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    block = _block_pos(env)
    tool = _tool_pos(env)

    # Latch the block's start y on the first step after each reset.
    if self._fresh.any():
      ids = torch.where(self._fresh)[0]
      self._block_start_y[ids] = block[ids, 1]

    # Per-env, because scale DR changes it every reset (see _block_full_width).
    s = _block_full_width(env)  # Unity block localScale (full width), metres
    d = torch.linalg.norm(tool - block, dim=-1)
    exp_term = torch.exp(-(2.0 / U) * (d - 1.5 * s).clamp(min=0.0))

    dist = (block[:, 1] - GOAL_CENTER_Y).abs() - s
    orig = (self._block_start_y - GOAL_CENTER_Y).abs() - s
    linear_term = 1.0 - dist / orig.clamp(min=1e-6)

    potential = exp_term + 2.0 * linear_term

    # Unity's time drag. This was dropped once, on the reasoning that it was an
    # artifact of Unity's 2000-decision episode and that action_rate_l2 /
    # joint_pos_limits regularize instead. That was wrong: those two penalize
    # motion, not TIME, so without this term timing out is free. Job 52194268
    # showed exactly that -- episode length sat at 1725/2000 and ROSE over
    # training while success_rate stayed flat. Inside a potential delta it is a
    # constant -1/MAX_DECISIONS per step, so a full-length episode bleeds ~1.0,
    # half the success bonus.
    if USE_TIME_DRAG:
      potential = potential - env.episode_length_buf.float() / MAX_DECISIONS

    delta = potential - self._prev
    # First step after reset: Unity's `reward` accumulator restarts at 0, so
    # the first AddReward is the full potential; replicate by zeroing prev.
    delta = torch.where(self._fresh, potential, delta)
    self._prev = potential
    self._fresh = torch.zeros_like(self._fresh)
    return delta


def block_in_goal(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Unity IsInsideGoal: block centre inside the goal bounds (x AND y)."""
  block = _block_pos(env)
  in_x = (block[:, 0] - ANCHOR_X).abs() <= GOAL_HALF_X
  in_y = (block[:, 1] - GOAL_CENTER_Y).abs() <= GOAL_HALF_Y
  return in_x & in_y


def success_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Unity TaskComplete(2.0): one-shot +2 on the success step."""
  return block_in_goal(env).float() * 2.0


def success_metric(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Binary success for the metrics manager. Paired with reduce="max" this
  becomes a per-episode flag, so its logged mean IS the success rate."""
  return block_in_goal(env).float()


def place_fixtures(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Write per-env mocap poses for the goal bar and camera frame.

  Fixed geometry cannot tile across mujoco_warp worlds (one shared model), so
  the bar and camera ride mocap bodies whose per-world poses are set here every
  reset: env_origin + the design-frame constant."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  origins = env.scene.env_origins[env_ids]
  n = len(env_ids)

  def pose(px, py, pz):
    p = torch.zeros(n, 7, device=env.device)
    p[:, 0] = origins[:, 0] + px
    p[:, 1] = origins[:, 1] + py
    p[:, 2] = origins[:, 2] + pz
    p[:, 3] = 1.0  # identity quat (the camera's own quat lives on the camera)
    return p

  env.scene["goal"].write_mocap_pose_to_sim(pose(0.0, 0.0, 0.0), env_ids=env_ids)
  env.scene["topdown"].write_mocap_pose_to_sim(
    pose(ANCHOR_X, CAM_CENTER_Y, CAM_HEIGHT), env_ids=env_ids
  )


JAW_OPEN = 0.5
"""Jaw angle held for the whole run (matches the lift task's open command)."""


def _open_jaw(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Hold the jaw open. Sets the joint AND its actuator target: writing the
  position alone would just be driven back to the closed default."""
  robot = env.scene["robot"]
  if "jaw" not in robot.joint_names:
    return
  ids = torch.tensor(
    [robot.joint_names.index("jaw")], device=env.device, dtype=torch.long
  )
  # The two setters want DIFFERENT shapes, so build both. write_joint_position
  # does `qpos[_resolve_env_ids(env_ids), q_slice]`, which turns env_ids into a
  # column and broadcasts to (n, 1); set_joint_position_target does a plain
  # `[env_ids, joint_ids]`, whose advanced indices broadcast to (n,).
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  val = torch.full((len(env_ids), 1), JAW_OPEN, device=env.device, dtype=torch.float32)
  robot.write_joint_position_to_sim(val, joint_ids=ids, env_ids=env_ids)
  robot.set_joint_position_target(val.squeeze(-1), joint_ids=ids, env_ids=env_ids)


def _light_diffuse(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, ranges: dict
) -> None:
  """Randomize light brightness. mjlab ships light_pos/light_dir DR but not
  intensity, so drive the light_diffuse model field through the same engine."""
  from mjlab.envs.mdp.dr._core import _randomize_model_field

  _randomize_model_field(
    env,
    env_ids,
    "light_diffuse",
    entity_type="light",
    ranges=ranges,
    distribution="uniform",
    operation="abs",
    default_axes=[0, 1, 2],
  )


def _camera_rgb_normalized(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """camera_rgb with the per-frame min-max stretch the deployment applies.

  medrct_onnx's image_processor.cpp runs `cv::normalize(img, out, 0, 1,
  NORM_MINMAX)`, which rescales every frame so its darkest pixel is 0 and its
  brightest is 1 -- NOT a fixed /255. Training on fixed-scale images and
  deploying through a contrast stretch would be a silent input-distribution
  shift, so reproduce the stretch here. Output is (B, 3, 84, 84), the
  deployed contract.
  """
  img = manipulation_mdp.camera_rgb(env, sensor_name)  # (B, 3, H, W) in [0,1]

  # Per-frame min-max stretch over all channels/pixels, mirroring cv::normalize.
  flat = img.flatten(1)
  lo = flat.min(dim=1).values[:, None, None, None]
  hi = flat.max(dim=1).values[:, None, None, None]
  return (img - lo) / (hi - lo).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Env cfg


def psm_push_wide_a3b_env_cfg(
  play: bool = False, video_interval_s: float = 300.0
) -> ManagerBasedRlEnvCfg:
  cfg = psm_push_env_cfg(play=play)

  cfg.scene.entities = {
    "robot": _unity_robot_cfg() if UNITY_LAYOUT else get_psm_robot_cfg(tool_type="lnd"),
    "block": EntityCfg(spec_fn=get_a3b_block_spec),
    "goal": EntityCfg(spec_fn=get_wide_goal_spec),
    "topdown": EntityCfg(spec_fn=get_topdown_cam_spec),
  }

  # No command term: the goal is fixed (Unity ResetGoal is disabled). Empty
  # dict, not None: mjlab's train.py does `"motion" in cfg.env.commands`.
  cfg.commands = {}
  # Base cfg's curriculum modulates joint_vel_hinge, which we removed.
  cfg.curriculum = {}

  # Planar 2-D increments in the BASE frame, 0.5 mm per 50 Hz decision --
  # numerically identical to deployment (position_speed_multiplier 0.0005).
  cfg.actions.pop("joint_pos", None)
  cfg.actions["cartesian_delta"] = CartesianDeltaActionCfg(
    entity_name="robot",
    actuator_names=PSM_MAIN_JOINTS,
    position_only=True,
    position_axes=(0, 1),
    frame="base",
    fk_fn=compute_psm_fk,
    ik_fn=_psm_ik_fn,
    pos_scale=0.0005,
    max_target_lead=None,
  )

  # Unity: MaxStep 2000 decisions at 0.02 s = 40 s episodes. decimation stays 4
  # (0.005 * 4 = 0.02 s), matching Unity's 50 Hz decision rate.
  cfg.episode_length_s = 40.0

  # --- observations: camera-only ACTOR (plus the constant-zeros 1D branch the
  # rsl-rl MLP head requires; export bakes the zeros into the graph), and a
  # PRIVILEGED state-only critic.
  #
  # The critic used to carry the same zeros + camera as the actor, a leftover
  # from replicating SB3-a3b (which had no privileged critic). We dropped Unity
  # recipe replication, and the known-good mjlab run this recipe is based on
  # (psm_cart_s2r_8192) had a 15-D privileged critic. Restoring that costs the
  # actor nothing, removes a full 84x84 CNN from every PPO epoch, and gives the
  # value function geometry it previously had to hallucinate from pixels.
  def _zeros_obs(env, dim: int = 3):
    return torch.zeros(env.num_envs, dim, device=env.device)

  cfg.observations["actor"].terms = {
    "zeros": ObservationTermCfg(func=_zeros_obs, params={"dim": 3}),
  }
  cfg.observations["critic"].terms = {
    "state": ObservationTermCfg(func=_critic_state),
    "joint_pos": ObservationTermCfg(
      func=manipulation_mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=PSM_MAIN_JOINTS)},
    ),
    "actions": ObservationTermCfg(func=base_mdp_last_action),
  }

  # Render natively at the deployed 84x84. Rendering higher and pooling down
  # costs pixels quadratically (and forces the env count down to fit memory)
  # for a cosmetic softness match that Unity doesn't actually do.
  cam_cfg = CameraSensorCfg(
    name="topdown_cam",
    camera_name="topdown/topdown_cam",
    height=84,
    width=84,
    data_types=("rgb",),
    enabled_geom_groups=(0, 1, 2),
    # Shadows ON. mujoco_warp ray-traces them (render.py:484, gated on
    # use_shadows AND light_castshadow) and a shadowed pixel drops to
    # NO_LIGHT_AMBIENT_FALLBACK = 0.3, so the effect is strong.
    #
    # This looked broken for a while: toggling the flag was bit-identical. The
    # cause was the LIGHT DIRECTION, not the renderer. The default sun points
    # straight down (0,0,-1) and the camera also looks straight down, so every
    # shadow falls exactly behind the object casting it. Even at 33 degrees a
    # 12.5 mm block sitting 6 mm off the floor throws a ~3 px shadow that hides
    # under itself at 1.39 mm/px. Verified with the block hovered clear of the
    # floor under a 55-degree light: z=8 mm -> 99 px changed, z=30 mm -> 190 px,
    # max|diff| 0.18. The light_dir DR below now spans up to 45 degrees off
    # vertical, which is what makes them visible.
    use_shadows=True,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      "topdown_rgb": ObservationTermCfg(
        func=_camera_rgb_normalized, params={"sensor_name": "topdown_cam"}
      ),
    },
    enable_corruption=False,
    concatenate_terms=True,
  )

  # --- rewards. Unity's reward is EXACTLY potential-delta + 2.0 on success --
  # no regularizers at all. We added action_rate_l2 and joint_pos_limits from
  # the mjlab baseline; joint_pos_limits then grew to 15x the entire positive
  # task signal (-0.180/episode vs +0.012 at iteration 2572 of job 52194268).
  # Default is now Unity's two terms only.
  from mjlab.envs import mdp as base_mdp

  cfg.rewards = {
    "a3b_shaping": RewardTermCfg(func=a3b_shaping, weight=1.0),
    "success": RewardTermCfg(func=success_bonus, weight=1.0),
  }

  if USE_REGULARIZERS:
    # Kept behind a flag so the sweep can test whether they help at all.
    # joint_pos_limits scope MUST stay the 6 actuated joints: ".*" also matches
    # the PSM's passive parallel four-bar linkage (pitch_bottom/end/top/front),
    # which deflects with outer_pitch, so one physical violation is counted
    # five times. Measured under directed actions (600 steps, 64 envs):
    #   ".*" total 0.0787  vs  PSM_MAIN_JOINTS total 0.0189   (4.16x)
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
      func=base_mdp.action_rate_l2, weight=-0.01
    )
    cfg.rewards["joint_pos_limits"] = RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=-0.01,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=PSM_MAIN_JOINTS)},
    )

  # --- terminations: time-out + block-in-goal (Unity has no other ends).
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "success": TerminationTermCfg(func=block_in_goal),
  }

  # --- metrics: success rate. reduce="max" turns the per-step binary into a
  # per-episode flag (1 if the block ever reached the goal), so the mean the
  # runner logs IS the success rate.
  cfg.metrics = {
    "success_rate": MetricsTermCfg(func=success_metric, reduce="max"),
  }

  # --- events: keep robot reset; respawn the block uniformly in the Unity
  # spawn square (goal band excluded by capping y, vs Unity's rejection
  # sampling -- a slight simplification), yaw 0-90 deg like Unity.
  keep = {}
  for name in ("reset_base", "reset_robot_joints"):
    if cfg.events and name in cfg.events:
      keep[name] = cfg.events[name]
  cfg.events = keep

  # Block spawn, matched to Unity by RATIO rather than by absolute units. The
  # Unity env's SpawnGround is a 10x10 Plane scaled 0.3 -> extents +-1.5 u, its
  # camera sits at local y = 5 with fov 60 -> frame half-extent 5*tan30 =
  # 2.887 u, and the tool resets to local z = 1.77 u. So in Unity:
  #     spawn half / frame half = 1.5 / 2.887 = 0.52
  #     tool offset / frame half = 1.77 / 2.887 = 0.61
  # Ratios avoid the unit ambiguity (our scene geometry uses 1 u = 0.05 m but
  # the camera height implies ~0.02 m/u, so absolute Unity offsets do not
  # transfer). Spawn is centred on the CAMERA AXIS, as it is in Unity, where
  # SpawnGround sits at local z = 0.
  #
  # The old range started at y = 0.005, BEHIND the tool at y = 0.0127, so the
  # block could spawn between the tool and the camera -- forcing the tool to
  # retract up and out of frame to get behind it. Unity never does this: its
  # tool at 1.77 u sits beyond the spawn edge at 1.5 u, so the block is always
  # on the goal side of the tool.
  frame_half = CAM_HEIGHT * math.tan(math.radians(CAM_FOVY) / 2.0)  # 0.0583 m by design
  spawn_half = 0.52 * frame_half
  # Unity's tool (1.77 u) sits 0.27 u beyond the spawn edge (1.5 u), so the
  # block never spawns on top of it. Keep that clearance -- 0.27/2.887 = 9.4%
  # of the frame half-extent -- or the recentred spawn would reach to within
  # 2 mm of the tool.
  spawn_y_lo = max(CAM_CENTER_Y - spawn_half, ANCHOR_Y + 0.094 * frame_half)
  spawn_y_hi = min(CAM_CENTER_Y + spawn_half, GOAL_CENTER_Y - GOAL_HALF_Y - 2 * BLOCK_HALF)
  cfg.events["reset_block"] = EventTermCfg(
    mode="reset",
    func=reset_root_state_uniform,
    params={
      "asset_cfg": SceneEntityCfg("block"),
      "pose_range": {
        "x": (ANCHOR_X - spawn_half, ANCHOR_X + spawn_half),
        "y": (spawn_y_lo, spawn_y_hi),
        "z": (BLOCK_HALF, BLOCK_HALF),
        "yaw": (0.0, 1.5708),
      },
      "velocity_range": {},
    },
  )

  # --- DR matched to the a3b config (metric equivalents of the Unity ranges).
  cfg.events["block_color"] = EventTermCfg(
    mode="reset",
    func=dr.geom_rgba,
    params={
      "asset_cfg": SceneEntityCfg("block", geom_names=("block_geom",)),
      "operation": "abs",
      "distribution": "uniform",
      "ranges": {0: (0.6, 1.0), 1: (0.05, 0.2), 2: (0.05, 0.2), 3: (1.0, 1.0)},
    },
  )
  cfg.events["block_mass"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("block", body_names=("block",)),
      "operation": "abs",
      "distribution": "uniform",
      "ranges": (0.01, 0.3),
    },
  )
  # Unity's CONTACT friction, not its yaml range. The physics dump shows the
  # block carries a "Slippery" material (0.1), the ground's DR writes
  # U(0, 0.3), and PhysX combines by AVERAGE -> the actual block-floor
  # contact is U(0.05, 0.2). With the block geom at friction 0 (see
  # get_a3b_block_spec) MuJoCo's max-combine makes the floor value BE the
  # contact value, so this range reproduces Unity's contact distribution
  # exactly. Legacy path keeps the raw yaml range it trained with.
  cfg.events["floor_friction"] = EventTermCfg(
    mode="reset",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("terrain"),
      "operation": "abs",
      "distribution": "uniform",
      "ranges": {0: (0.05, 0.2) if UNITY_LAYOUT else (0.0, 0.3)},
    },
  )
  cfg.events["floor_color"] = EventTermCfg(
    mode="reset",
    func=dr.geom_rgba,
    params={
      "asset_cfg": SceneEntityCfg("terrain"),
      "operation": "abs",
      "distribution": "uniform",
      "ranges": {0: (0.0, 0.4), 1: (0.6, 0.8), 2: (0.95, 1.0), 3: (1.0, 1.0)},
    },
  )
  # NOTE: no goal-colour DR. Unity's target_rb 0.25..0.75 drives r and b from a
  # SINGLE sampled value against a fixed green; mapping it onto dr.geom_rgba,
  # which samples channels independently, let r and b both reach 0.75 against
  # g = 0.8 -- (0.75, 0.8, 0.75) is near-grey, and under a bright light draw it
  # clipped to pure white, erasing the goal entirely. Reproducing Unity here
  # needs one shared sample across r and b, not two independent ones, so the
  # goal stays a fixed green until that exists.
  # Light brightness (Unity light_intensity 5..35). Unity's intensity scale has
  # no metric equivalent in MuJoCo, whose light_diffuse is a 0..1 RGB gain, so
  # this reproduces the ~7x dynamic range rather than the absolute numbers.
  cfg.events["light_intensity"] = EventTermCfg(
    mode="reset",
    func=_light_diffuse,
    params={"ranges": {0: (0.15, 1.0), 1: (0.15, 1.0), 2: (0.15, 1.0)}},
  )
  # Block scale DR. Unity re-randomizes localScale to U(0.3, 0.5) on every reset
  # (PushBlockAgent.cs:57-61 -- `scale` sets localScale ABSOLUTELY, it is not a
  # multiplier; the prefab's 0.5 is only the authoring default and never occurs
  # at runtime). That is +-25% about a mean of 0.4.
  #
  # Applied as a RATIO about BLOCK_HALF. Under UNITY_LAYOUT the ratio IS the
  # absolute Unity range: 0.75-1.25 x 20 mm = 15-25 mm = Unity's U(0.3, 0.5) u
  # at 50 mm/u exactly. On the legacy path it reproduces Unity's SPREAD around
  # the real-frame-calibrated 12.5 mm mean instead.
  #
  # shared_random keeps the cube isotropic: one sample drives all three axes.
  cfg.events["block_scale"] = EventTermCfg(
    mode="reset",
    func=dr.geom_size,
    params={
      "asset_cfg": SceneEntityCfg("block"),
      "operation": "scale",
      "distribution": "uniform",
      "ranges": {0: (0.75, 1.25), 1: (0.75, 1.25), 2: (0.75, 1.25)},
      "shared_random": True,
    },
  )
  cfg.events["place_fixtures"] = EventTermCfg(mode="reset", func=place_fixtures)
  # Jaw: Unity keeps it CLOSED (near zero), and the open-jaw pose was a local
  # deviation from the Unity home pose (along with the ~1 mm tip height).
  # UNITY_LAYOUT + UNITY_HOME holds the closed default; A3B_UNITY_HOME=0
  # reproduces the previous open-jaw home on the unity scene, and legacy
  # keeps the open jaw it was trained with.
  if not (UNITY_LAYOUT and UNITY_HOME):
    cfg.events["open_jaw"] = EventTermCfg(mode="reset", func=_open_jaw)
  # Start the tool at the block's CENTRE OF MASS: the planar action freezes z at
  # the reset height, and the default insertion leaves the tip at 4.9 cm --
  # sailing OVER the block. An earlier +0.036 aimed at the block's full height
  # (~1.3 cm), i.e. its top face, which torqued the block and rolled it instead
  # of pushing it.
  #
  # This is the INSERTION JOINT, not a height: it slides the tool along the
  # shaft axis, and tool z follows only through FK on an RCM-constrained arm.
  # The map is nonlinear and saturates past ~0.046, so it was bisected.
  #
  # Bisected on TOOL_TIP, which is what contacts the block and sits ~3.7 mm
  # BELOW tool_midpoint. 0.04394 -> tool_tip 1.03 mm, just clear of the floor
  # (tool_midpoint 4.73 mm, block COM 6.13 mm). Re-bisect after anything that
  # changes the arm pose: opening the jaw already shifted this map once, since
  # the splayed jaw tips clear the floor differently from a closed one.
  from mjlab.envs.mdp.events import reset_joints_by_offset as _rjbo

  cfg.events["lower_tool"] = EventTermCfg(
    mode="reset",
    func=_rjbo,
    params={
      # Tool tip height. Legacy 0.04394 -> tip 1.2 mm (tuned by hand against the
      # old 12.5 mm block). NOTE the sign: MORE insertion LOWERS the tip.
      #
      # UNITY_LAYOUT: 0.211063 puts the tool_tip control point 12.547 mm above
      # the floor -- Unity's home height, read off a PLAY dump captured with
      # the scene actually at home (tool_tip worldPosition y = 0.25093 u x
      # 50 mm/u, shaft vertical at 0.004 deg, tip directly beneath base_link).
      # Total insertion is 0.3111 (default 0.10 + this offset; range extended
      # in _named_spec).
      #
      # In that episode Unity's block drew 20.68 mm, centroid 12.30 mm, so the
      # tip sits at 102% of the centroid -- the tool pushes through the centre
      # of mass, which is the whole point of the height.
      #
      # An earlier dump showed tool_tip at 0.10034 u (5.02 mm) with insertion
      # 0.240 rather than 0.1362; that was not the home state. The original
      # 0.2598 chased "13.6 mm" and was closer than the 5 mm reading that
      # briefly replaced it.
      #
      # 12.547 mm is the CONTROL POINT; the jaw geometry bottoms out 0.52 mm
      # higher. Block DR spans 15-25 mm (centroid 7.5-12.5 mm), and the tool
      # height is fixed, so the tip is at the centroid only for the ~20 mm mean
      # block -- above it for small draws, below it for large ones.
      #
      # A3B_UNITY_HOME=0 is now a pure jaw ablation (open vs closed) at the
      # same height: vertical, insertion no longer slides the tip goalward, so
      # there is no reason to give the variants different geometry.
      "position_range": ((0.211063, 0.211063) if UNITY_LAYOUT else (0.04394, 0.04394)),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("main_insertion",)),
    },
  )
  if UNITY_LAYOUT:
    # Point-light POSITION DR, the real counterpart of Unity's light_px/py/pz.
    # The light rides the topdown mocap frame (see get_topdown_cam_spec), so
    # light_pos is a LOCAL offset from the camera position and per-env
    # placement comes from the mocap pose. Unity ranges, y-up -> z-up about
    # the camera at (0, 5, 0):
    #   light_px -5..5 u   -> local x +-0.25 m
    #   light_pz -3.5..1.5 -> local y -1.5*U..+3.5*U (Unity +z is base-ward)
    #   light_py 4..6      -> local z +-0.05 m
    cfg.events["light_pos"] = EventTermCfg(
      mode="reset",
      func=dr.light_pos,
      params={
        "asset_cfg": SceneEntityCfg("topdown", light_names=("scene_light",)),
        "operation": "abs",
        "distribution": "uniform",
        "ranges": {
          0: (-5.0 * U, 5.0 * U),
          1: (-1.5 * U, 3.5 * U),
          2: (-1.0 * U, 1.0 * U),
        },
      },
    )
    # Tool-part greyscale DR (Unity tool_shaft_r / tool_roll_r /
    # tool_gripper_r; the cannula has no mjlab counterpart). Unity drives r=g=b
    # from ONE sample; shared_random reproduces that (one draw shared across
    # the term's channels/geoms) instead of sampling channels independently,
    # which is what ruled out goal-colour DR below.
    for name, geoms, lo, hi in (
      ("tool_shaft_color", ("main_insertion_link",), 0.05, 0.35),
      ("tool_roll_color", ("tool_roll_link_visual_0",), 0.05, 0.25),
      (
        "tool_gripper_color",
        (
          "tool_pitch_link_visual_0",
          "tool_yaw_link_visual_0",
          "tool_gripper1_link_visual_0",
          "tool_gripper2_link_visual_0",
        ),
        0.7,
        0.9,
      ),
    ):
      cfg.events[name] = EventTermCfg(
        mode="reset",
        func=dr.geom_rgba,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=geoms),
          "operation": "abs",
          "distribution": "uniform",
          "ranges": {0: (lo, hi), 1: (lo, hi), 2: (lo, hi), 3: (1.0, 1.0)},
          "shared_random": True,
        },
      )
  else:
    # Light DIRECTION, not position. The legacy scene light is DIRECTIONAL
    # (light_type=1), and a directional light has no position -- the renderer
    # reads only light_dir. This term used to call dr.light_pos, which
    # measurably varied light_pos per world while the render ignored every one
    # of those values. It was a silent no-op for the entire project.
    #
    # Unity randomizes the light's POSITION (light_px -5..5, light_py 4..6,
    # light_pz -3.5..1.5 u) aimed at the scene, so the equivalent here is the
    # angular spread that induces. dir is normalized downstream, so these are
    # ratios against a -1 z component.
    cfg.events["light_dir"] = EventTermCfg(
      mode="reset",
      func=dr.light_dir,
      params={
        "asset_cfg": SceneEntityCfg("terrain"),
        "operation": "abs",
        "distribution": "uniform",
        # x,y against a fixed -1 z, so |xy| = 0.45 is ~32 degrees off vertical.
        # The first pass used +-1.0 / -0.875..0.375 (up to 45 deg) and produced
        # shadow streaks visibly longer and harder than real's, which sit
        # closer to overhead. Some obliqueness is REQUIRED -- at 0 deg the
        # shadow hides entirely behind the object and use_shadows is a no-op.
        "ranges": {0: (-0.45, 0.45), 1: (-0.45, 0.30), 2: (-1.0, -1.0)},
      },
    )
  # Camera pose, matched to Unity by RATIO. Unity's camera sits at local y = 5
  # with fov 60 (frame half-extent 2.887 u); ours sits at 0.101 m (frame half-
  # extent 0.0583 m). Applying Unity's ABSOLUTE offsets to a camera 2.5x lower
  # over-perturbed the view by 2.5x, which is what pushed the tool out of frame.
  # Unity is y-up, so its camera_py is HEIGHT -> our z, camera_pz -> our y.
  #   camera_px  +-0.1 u  = +-3.5% of frame half-extent
  #   camera_pz  0..0.4 u = 0..13.9% of frame half-extent
  #   camera_py  +-0.1 u  = +-2% of camera height
  cam_dx = 0.035 * frame_half  # ~2.0 mm
  cam_dy = 0.139 * frame_half  # ~8.1 mm
  cam_dz = 0.02 * CAM_HEIGHT  # ~2.0 mm
  cfg.events["camera_pos"] = EventTermCfg(
    mode="reset",
    func=dr.cam_pos,
    params={
      "asset_cfg": SceneEntityCfg("topdown", camera_names=("topdown_cam",)),
      "axes": [0, 1, 2],
      "ranges": {0: (-cam_dx, cam_dx), 1: (0.0, cam_dy), 2: (-cam_dz, cam_dz)},
    },
  )
  # Camera angle -- was missing entirely. Unity camera_rx 87..93 deg is tilt
  # away from straight-down (= our pitch), camera_ry -3..3 deg is rotation
  # about the view axis (= our yaw, since the camera looks along -z).
  _DEG = math.pi / 180.0
  cfg.events["camera_quat"] = EventTermCfg(
    mode="reset",
    func=dr.cam_quat,
    params={
      "asset_cfg": SceneEntityCfg("topdown", camera_names=("topdown_cam",)),
      "pitch_range": (-3.0 * _DEG, 3.0 * _DEG),
      "yaw_range": (-3.0 * _DEG, 3.0 * _DEG),
    },
  )
  # NOTE: pushblock_dr.yaml randomizes no field of view, so fovy stays fixed at
  # the 60 deg set on the sensor. The previous (57, 63) DR was not from Unity.

  # --- misc
  cfg.scene.terrain.materials = ()
  cfg.scene.terrain.textures = ()
  if UNITY_LAYOUT:
    # Drop the default directional sun: the point light on the camera frame is
    # the only scene light, as in Unity (whose fill is a dim ambient skybox --
    # the renderer's ambient/headlight terms play that role here). An earlier
    # note here claimed the renderer applies no spatial falloff for
    # non-directional lights; that experiment had zero attenuation
    # coefficients, which IS no falloff. See the light in get_topdown_cam_spec.
    cfg.scene.terrain.lights = ()

    # Dim the headlight on the MERGED spec. mjlab's scene.xml ships a strong
    # shadow-free camera-axis fill (diffuse 0.6, ambient 0.3) that mujoco_warp
    # bakes into the render context (io.py reads vis.headlight), washing the
    # point light's shadows out to a faint tint. Setting spec.visual on an
    # ENTITY spec does nothing -- MjSpec.attach() does not propagate visual
    # settings -- so this must run on the scene's master spec, which is what
    # SceneCfg.spec_fn receives. Unity's only fill is its ~0.21 ambient
    # skybox; match that and keep just a whisper of diffuse.
    def _dim_headlight(spec: mujoco.MjSpec) -> None:
      spec.visual.headlight.diffuse = (0.08, 0.08, 0.08)
      spec.visual.headlight.ambient = (0.21, 0.21, 0.21)

    cfg.scene.spec_fn = _dim_headlight
  cfg.viewer.body_name = "tool_pitch_link"

  if play:
    cfg.episode_length_s = int(1e9)
  else:
    cfg.recorders["policy_cam_video"] = PolicyCamVideoRecorderCfg(
      sensor_name="topdown_cam",
      num_track=20,
      interval_s=video_interval_s,
      fps=int(round(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation))),
      upscale=4,
    )

  return cfg
