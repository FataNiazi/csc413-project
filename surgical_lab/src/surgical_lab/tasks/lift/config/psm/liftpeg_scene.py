"""LiftPeg scene geometry, rebuilt from measured Unity ground truth.

Every constant here is traceable to `docs/liftpeg_unity_ground_truth.md`, which
was extracted from live edit- and play-mode dumps of `LiftPeg2DCam.unity`. The
earlier v1-v4 attempts were built from a *different* task's prefab that happens
to share the Unity folder; nothing from them survives here.

Frame mapping, verified by the tool/base coincidence at the zero pose:

    (x, y, z)_unity -> (x, -z, y)_spec ;  env = spec + (0, 0.1, 0) m

Meshes carry Unity's 0.01 importer scale factor, so `scale=0.01` on the raw
`.obj` lands exactly on the measured sizes. Both meshes are Z-up already (the
post/prism axis is the obj's +z), so no reorientation is needed for MuJoCo --
Unity's 270 deg X rotation on these objects is just its own Y-up conversion.
"""

from __future__ import annotations

import math
import os

import mujoco

# --- asset paths ---------------------------------------------------------
# peg.obj and triangle.obj ship in the repo, byte-identical to the Unity copies
# in muse/unity/Assets/Tasks/SharedAssets/Resources, so a fresh clone runs
# without the muse checkout. LIFTPEG_MESH_DIR still overrides.
_MESH_DIR = os.environ.get(
  "LIFTPEG_MESH_DIR",
  os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
               "robots", "psm", "xmls", "assets"),
)
MESH_SCALE = 0.01          # Unity .obj importer scale factor

# Ring WALL-HEIGHT scale, for matching the sim ring to the physical one.
#
# Measured 2026-09-16, the printed original ring against triangle.obj:
#
#              sim      real     
#   plan h    21.70    23.00 mm   sim 5.6% small
#   bore Od   12.00    12.00 mm   match
#   wall h    15.00    20.00 mm   sim 25% SHORT
#
# Both rings rest on the plate, so the bottom is pinned and the 5 mm lands
# entirely on the TOP: the sim ring's top is 20.99 mm, the real one's 25.99.
# A policy that grasps 2.1 mm below the sim top is 7.1 mm below the real top,
# which is the overbite seen on hardware and never in sim.
#
# This knob scales ONLY the wall height, to isolate that variable -- the real
# ring is not a uniform rescale of this mesh (bore matches while plan does
# not), so a proper fix is a new mesh, not a scale factor. 20/15 = 1.3333
# reproduces the real ring's height.
#
# The factor belongs in component 2, established by MEASUREMENT rather than
# from mesh_quat: MuJoCo RECOMPUTES the mesh frame after scaling, so the quat
# read off an unscaled compile does not predict which component moves which
# body axis. Scaling each in turn: [0] and [1] change the plan only, [2]
# takes the wall 15.00 -> 20.00 mm with the plan untouched.
RING_Z_SCALE = float(os.environ.get("LIFTPEG_RING_Z_SCALE", "1.0"))
MESH_SCALE_XYZ = [MESH_SCALE, MESH_SCALE, MESH_SCALE * RING_Z_SCALE]

# --- layout, env frame, metres -------------------------------------------
def _pair(name, default):
  raw = os.environ.get(name)
  return tuple(float(x) for x in raw.split(",")) if raw else default


FLOOR_Z = 0.0
# The POST sits directly under the tooltip, so descent is 1-D as in Unity.
# pushblock's anchor (ANCHOR_X, ANCHOR_Y) = (0, 12.7) mm.
SPAWN_CENTER = (0.0, 0.0127)
# Half-width of the square the STAND BASE spawns in. Default 2 mm: light
# jitter only, because the ring's pose on the post was meant to be the task's
# randomisation source.
#
# But the post is what the tool descends onto, and at 2 mm it barely moves. The
# 20-episode sim/real calibration (2026-09-19) measured a lateral disagreement
# of about 6.5 mm between where the camera model puts the ring and where the
# robot has to go to grasp it -- i.e. OUTSIDE this distribution entirely. A
# policy trained at 2 mm can memorise the descent target; one trained wider has
# to actually find the ring in the image. Overridable so that can be tested
# without forking the scene.
SPAWN_HALF = float(os.environ.get("LIFTPEG_SPAWN_HALF_M", "0.002"))

# The triangle's pose ON THE SHAFT is the real randomisation -- the analogue of
# pushblock's block start pose. The hole gives 3.39 mm of clearance over the
# 2.2 mm post, so it can sit anywhere in that annulus, and yaw spans 0-120 deg
# (the three edges are 120 deg apart, so that covers every distinct edge the
# policy might choose to grasp).
RING_CLEARANCE = (BLOCK_BAR_RADIAL_FWD := 0.00659) - 0.001 - 0.0022
TRI_YAW_RANGE = (0.0, 2.0943951)   # 120 deg

# Green goal square, 20.6 mm -- the side of a square enclosing the triangle
# from top-down. Placed BEHIND the mount: the plate spans y in [-17.3, 42.7]
# about the post, so the square must clear y < -27.6 to sit on bare floor.
GOAL_SQUARE_SIDE = 0.0206
# To the RIGHT of the mount, not behind it. The plate spans x in [-30, +30] mm
# about the post, so the square must clear x > 40.3 to sit on bare floor.
# 45 mm keeps it well inside frame once camera DR (+-5 mm position, +-3 deg
# yaw/pitch) and the goal's own +-4 mm jitter are stacked: worst case ~65 mm
# off-axis against an ~83 mm half-width at this distance.
# NOTE the sign: env +x renders to screen LEFT under this camera (the same
# flip found during the pushblock arrow-convention work), so screen-right is -x.
# Overridable so the calibrated layout can be applied without a code edit;
# the scene-editor match puts the goal 3.5 mm -x and 11 mm +y of nominal.
GOAL_SQUARE_CENTER = _pair("LIFTPEG_GOAL_CENTER", (-0.045, 0.0127))
# RED, deliberately: the triangle is DarkGreen, and a green goal made the two
# most similar-looking objects in an 84x84 frame. pushblock paired a RED block
# with a GREEN goal; ours is the same contrast, with the roles swapped.
GOAL_SQUARE_RGB = (0.85, 0.15, 0.15)
PEG_Z = 0.006                   # peg body origin; plate 0-6 mm, post 6-36 mm
BLOCK_SPAWN_DZ = 0.00795        # Unity: Block = Peg + (0.05, 0.159, 0) u
BLOCK_SPAWN_DX = 0.0025
GOAL = (0.0, 0.0115, 0.060)     # Unity (0, 60, 88.5) mm -- FIXED in env frame
CAM_POS = _pair("LIFTPEG_CAM_POS", (0.0, 0.0925, 0.110))  # Unity (0, 110, 7.5) mm
# 55, not Unity's 60. The deployed pipeline crops the OAK-1 frame horizontally
# to a square using the FULL height, so the effective vertical FOV is the
# camera's native VFOV (~55 deg for the IMX378) -- not whatever sim picked.
# Centring the DR here rather than on 60 puts the real rig near the middle of
# the training distribution instead of out at its 80th percentile.
CAM_FOVY = float(os.environ.get("LIFTPEG_CAM_FOVY", "55.0"))
# 40, not Unity's 45. From the nominal camera the tool tip at home sits 17.8
# deg off the optical axis while the peg sits only 7.5 deg off -- the aim is
# lopsided, and it spends 65% of the frame's half-angle on the tool before any
# randomisation. Pitching to the bisector puts both at 12.7 deg (46%), which is
# what makes a tool-visibility guarantee affordable at all. The pitch DR spans
# 40 +- 4, so the real rig's ~45 deg stand stays inside the distribution.
CAM_PITCH_DEG = float(os.environ.get("LIFTPEG_CAM_PITCH", "40.0"))
# Unity `Spot Light` (a POINT light despite the name): one per env, fixed at
# (0, 250, 0) u = env (0, 100, 250) mm -- directly above the env origin, NOT
# ride-along on the camera the way pushblock's is. Intensity 20, range 1 m,
# Hard shadows.
LIGHT_POS = (0.0, 0.100, 0.250)
LIGHT_DIFFUSE = (1.0, 0.9862, 0.9377)
LIGHT_SPECULAR = (0.3, 0.3, 0.3)
LIGHT_CASTSHADOW = True
# Unity's intensity/range pair is a URP photometric curve and does not map onto
# MuJoCo's 1/(a + b d + c d^2) directly. Normalise the quadratic term at the
# workspace so illumination there is ~full, which is the same trick a3b uses on
# its camera light. Distance light -> block: |(0,.1,.25)-(0,.03,.006)|.
LIGHT_WORKDIST = math.sqrt(0.07 ** 2 + 0.244 ** 2)
LIGHT_ATTENUATION = (0.0, 0.0, 1.0 / (LIGHT_WORKDIST ** 2))
# Unity's fill is its ambient skybox. RenderSettings is a SCENE setting, not a
# component, so the dumper does not capture it -- this is a3b's ~0.21 estimate
# carried over, and is the one lighting number here NOT measured.
HEADLIGHT_AMBIENT = 0.21

# --- materials, read from the scene dump's MeshRenderer entries -----------
# Unity asset -> RGB. The floor is `BlueFloor`, the SAME material pushblock
# uses: a3b clears terrain materials/textures (flat, untextured) and its
# floor_color DR spans R (0, 0.4) / G (0.6, 0.8) / B (0.95, 1) -- this colour
# sits inside all three, so that DR is centred on this exact floor.
FLOOR_RGB = (0.35, 0.795, 1.0)      # Assets/Tasks/SharedAssets/Materials/BlueFloor.mat
PEG_RGB = (1.0, 0.8, 0.0)           # MountMaterial -- the mount is YELLOW
BLOCK_RGB = (0.0, 0.7, 0.15)        # DarkGreen -- the lifted triangle is GREEN
# a3b's floor_color DR, reused verbatim so the two tasks randomise identically.
FLOOR_RGB_DR = ((0.0, 0.4), (0.6, 0.8), (0.95, 1.0))

# --- peg (static scenery: post + base plate) -----------------------------
PEG_POST_RADIUS = 0.0022        # Unity CapsuleCollider r 0.0022 -> d 4.4 mm
# "capsule" or "cylinder". Unity uses a CapsuleCollider, so capsule is also the
# faithful choice; it is switchable only so scripts/probe_liftpeg_pairsupport.py
# can score the two against each other rather than assume.
POST_GEOM = os.environ.get("LIFTPEG_POST_GEOM", "capsule")

# THE fix for the ring passing through the post.
#
# MuJoCo's default solref is (0.02, 1.0): a 20 ms constraint restitution time
# constant. Penetration is bled off over that window, so at a 5 ms timestep the
# contact needs ~4 steps to push back properly. A gripper driving the ring
# sideways at a steady ~0.3 mm per control step penetrates FASTER than the
# constraint recovers, so the ring simply walks through the 4.4 mm post while
# the solver is perpetually catching up.
#
# Measured on the failing checkpoint (`probe_liftpeg_replay_phasing.py`,
# 256 envs, 600 steps):
#
#   solref 0.020, 10 iterations    39.8% of envs lose the ring off the post
#   solref 0.020, 100 iterations   43.8%   <- solver iterations are irrelevant
#   solref 0.004, 10 iterations     0.0%
#
# In every escape the ring was flat (5.9 deg tilt, so not tipping off) and
# moving 0.34 mm per step against a 2.0 mm bar (so tunnelling is geometrically
# unavailable), and a post contact was present in the preceding 5 steps 100% of
# the time. The contact was always there; it was too soft to hold.
#
# Keep this at 2x the physics timestep, which is MuJoCo's own guidance for the
# stiffest stable setting. `psm_lift_peg_a3b_env_cfg` asserts the relationship
# rather than trusting the two constants to be edited together.
CONTACT_SOLREF = _pair("LIFTPEG_SOLREF", (0.004, 1.0))
# MuJoCo's default, unchanged: the sweep shows solimp makes no difference to
# this failure (0.0% escape with the default once solref is fixed), so there is
# no reason to deviate from it.
CONTACT_SOLIMP = _pair("LIFTPEG_SOLIMP", (0.9, 0.95, 0.001, 0.5, 2.0))


def _stiffen(geom):
  """Apply the task's contact parameters to a collision geom."""
  geom.solref = list(CONTACT_SOLREF)
  geom.solimp = list(CONTACT_SOLIMP)
  return geom
PEG_POST_TOP = 0.036            # mesh spans -6..+30 mm about the origin
PEG_PLATE_HALF = (0.030, 0.030, 0.003)   # 60 x 60 x 6 mm

# --- block (the graspable triangular ring) -------------------------------
BLOCK_MASS = 0.1
BLOCK_BAR_HALF = (0.0103, 0.001, 0.0075)  # Side1/2/3: 20.6 x 2.0 x 15.0 mm
BLOCK_BAR_RADIAL = 0.00659                # BoxCollider local centre offset
BLOCK_BAR_Z = 0.0075                      # mesh spans 0..15 mm from origin
# Derived from triangle.obj, NOT from Unity's Side1/2/3 world eulers. Those are
# expressed in Unity's frame and do not survive the (x, -z, y) mapping: reading
# them across literally put the bars' normals at 0/120/240 deg, a 90 deg
# rotation off, so each collision bar sat over a CORNER of the visual triangle
# and the jaws closed straight through the edges.
# Support-function analysis of the mesh: face normals at 90/210/330 deg with
# support 7.217 mm; the 30/150/270 directions are the corners (11.434 mm).
# A bar at yaw t has its normal at 90 + t, so the faces need t = 0/120/240.
BLOCK_BAR_YAWS_DEG = (0.0, 120.0, 240.0)

# Ring collision model. "sectors" follows the visual mesh via a convex
# decomposition; "bars" restores Unity's three Side1/2/3 BoxColliders, which
# every run before 2026-08-15 used and which left the ring's corner mass with no
# collision geometry at all. Keep `bars` available so those runs stay
# reproducible -- checkpoints trained against it are not valid under `sectors`.
# BLOCK_BAR_YAWS_DEG / BLOCK_BAR_RADIAL stay regardless: they describe the three
# GRASP points (used by the teacher's observation and the scripted controller),
# which is a separate concern from collision.
BLOCK_COLLISION = os.environ.get("LIFTPEG_BLOCK_COLLISION", "sectors")
BLOCK_SECTORS = int(os.environ.get("LIFTPEG_BLOCK_SECTORS", "12"))


_OUTER_R: float | None = None


def outer_radius() -> float:
  """Ring outer radius in metres, from the section actually compiled.

  Cached and lazy: _block_section() compiles a throwaway MjSpec, too expensive
  to run at import and pointless for anything that never touches liftpeg.
  """
  global _OUTER_R
  if _OUTER_R is None:
    import numpy as _np
    _OUTER_R = float(_np.linalg.norm(_block_section()[0], axis=-1).max())
  return _OUTER_R


def quat_from_forward_up(fwd, up):
  """MuJoCo camera quat: the camera looks down its own -z, with +y up."""
  fx, fy, fz = fwd
  n = math.sqrt(fx * fx + fy * fy + fz * fz)
  fwd = (fx / n, fy / n, fz / n)
  zx, zy, zz = (-fwd[0], -fwd[1], -fwd[2])          # camera +z is -forward
  ux, uy, uz = up
  d = ux * zx + uy * zy + uz * zz                    # re-orthogonalise up
  ux, uy, uz = ux - d * zx, uy - d * zy, uz - d * zz
  n = math.sqrt(ux * ux + uy * uy + uz * uz)
  yx, yy, yz = ux / n, uy / n, uz / n
  xx = yy * zz - yz * zy                             # x = y cross z
  xy = yz * zx - yx * zz
  xz = yx * zy - yy * zx
  m = ((xx, yx, zx), (xy, yy, zy), (xz, yz, zz))     # columns are the axes
  t = m[0][0] + m[1][1] + m[2][2]
  if t > 0.0:
    s = math.sqrt(t + 1.0) * 2.0
    return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s)
  if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
    s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
    return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s)
  if m[1][1] > m[2][2]:
    s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
    return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s,
            (m[1][2] + m[2][1]) / s)
  s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
  return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s,
          (m[1][2] + m[2][1]) / s, 0.25 * s)


# Head-on view: down at the ring's own height, looking along -y, for the
# vision arms. The 45-degree side view resolves xy well and height poorly --
# the shaped-vision run grasps (held_frac 0.75) and cannot lift (2.3 mm), which
# is what a camera that cannot see height would produce.
CAM_POS_HEAD = _pair("LIFTPEG_CAM_POS_HEAD", (0.0, 0.130, 0.030))
CAM_PITCH_HEAD = float(os.environ.get("LIFTPEG_CAM_PITCH_HEAD", "10.0"))

# --- camera domain randomisation ---------------------------------------------
# Liftpeg shipped with NONE. The placement event wrote CAM_POS and camera_quat()
# as constants expanded across envs, so every frame the policy has ever seen
# came from one exact camera pose -- while pushblock, which reached hardware,
# randomised both (dr.cam_pos / dr.cam_quat, +-2..8 mm and +-3 deg). A policy
# trained on a single viewpoint has no reason to be robust to the endoscope
# being clamped 10 mm off, and no way to tell us it isn't.
#
# These are deliberately WIDER than pushblock's: the mount is positioned by
# hand, so a centimetre and ~10 degrees is the honest reproducibility of the
# real setup, not a stress test.
CAM_POS_JITTER = _pair("LIFTPEG_CAM_POS_JITTER", (0.005, 0.005, 0.005))
CAM_PITCH_JITTER_DEG = float(os.environ.get("LIFTPEG_CAM_PITCH_JITTER", "4.0"))
CAM_YAW_JITTER_DEG = float(os.environ.get("LIFTPEG_CAM_YAW_JITTER", "4.0"))
# Roll stays generous while the others shrink: rotating a point about the
# optical axis cannot increase max(|x|,|y|) beyond its radius, so roll is the
# one angular axis that costs nothing against a square frame.
CAM_ROLL_JITTER_DEG = float(os.environ.get("LIFTPEG_CAM_ROLL_JITTER", "10.0"))
# Radial distance from the look-at point, as a multiplier on the nominal
# 155.6 mm. Distance and fovy BOTH scale apparent size, so randomising only one
# leaves a direction uncovered: fovy is pure magnification while distance also
# changes perspective. Randomising both spans the plane those two axes define,
# which is what we actually do not know.
CAM_DIST_SCALE = _pair("LIFTPEG_CAM_DIST_SCALE", (0.90, 1.12))
CAM_LOOKAT = _pair("LIFTPEG_CAM_LOOKAT", (0.0, -0.0175, 0.0))
CAM_DR = os.environ.get("LIFTPEG_CAM_DR", "1") == "1"

# --- hemisphere mode ----------------------------------------------------------
# Arbitrary camera POSITION, ANGLE and FOV: the camera is placed anywhere on a
# hemisphere around the workspace (never below the floor) and aimed at it.
#
# The hard part is not the camera, it is that the ACTION SPACE IS WORLD-FIXED
# (dx, dy, dz in env coordinates). Orbit the camera to the far side and "right
# in the image" becomes -x in the world instead of +x, so the policy cannot
# learn a fixed visual servo -- it has to infer the viewpoint from the image and
# invert the mapping. The scene is asymmetric enough to make that possible (the
# red goal sits to one side, the arm enters from a fixed direction), but it is a
# genuine capability jump rather than more robustness.
#
# Distance is held to a band around the current one so the workspace does not
# shrink to nothing, and every sample is rejected unless the tool, peg, plate
# and goal are all in frame.
# Box the tool tip can land in under the home-pose DR, env-frame metres, with
# headroom. Measured over 1024 resets at the current jitter (yaw 0.020, pitch
# 0.011, insertion 0.004): x [-5.25, +5.23], y [+9.83, +15.54],
# z [+59.76, +78.73] mm. scripts/check_tool_visibility.py re-measures it.
HOME_TIP_BOX = ((-0.0070, 0.0070), (0.0090, 0.0165), (0.0570, 0.0800))
CAM_VIS_MARGIN = float(os.environ.get("LIFTPEG_CAM_VIS_MARGIN", "0.92"))


def _tip_box_corners(device):
  """The 8 corners of HOME_TIP_BOX -- keeping all of them in frame keeps the
  tool visible for ANY home draw, not just the one that happened."""
  import itertools
  import torch as _t
  return _t.tensor(list(itertools.product(*HOME_TIP_BOX)), device=device,
                   dtype=_t.float32)


CAM_HEMI = os.environ.get("LIFTPEG_CAM_HEMI", "0") == "1"
CAM_HEMI_AZIM = _pair("LIFTPEG_CAM_HEMI_AZIM", (0.0, 360.0))
CAM_HEMI_ELEV = _pair("LIFTPEG_CAM_HEMI_ELEV", (20.0, 68.0))
CAM_HEMI_DIST = _pair("LIFTPEG_CAM_HEMI_DIST", (0.75, 1.25))
CAM_HEMI_ROLL = float(os.environ.get("LIFTPEG_CAM_HEMI_ROLL", "12.0"))
CAM_HEMI_TRIES = int(os.environ.get("LIFTPEG_CAM_HEMI_TRIES", "40"))


def points_in_frame(pos, quat, pts, fovy, margin=None):
  """Are ALL pts (n,k,3) inside the frame of each camera? -> (n,) bool."""
  import torch as _t
  from mjlab.utils.lab_api.math import matrix_from_quat
  R = matrix_from_quat(quat)
  d = pts - pos.unsqueeze(1)
  local = _t.einsum("nij,nkj->nki", R.transpose(1, 2), d)
  depth = -local[..., 2]
  f = fovy.reshape(-1).float()
  t = (_t.tan(_t.deg2rad(0.5 * f))
       * (CAM_VIS_MARGIN if margin is None else margin)).unsqueeze(-1)
  ok = (depth > 1e-4) & (local[..., 0].abs() <= t * depth) \
       & (local[..., 1].abs() <= t * depth)
  return ok.all(dim=1)


def goal_unoccluded(pos, goal_pts, plate_xy):
  """True where the camera can actually SEE the goal corners.

  points_in_frame only proves a point projects inside the image. The goal
  square sits on the floor ~15 mm beyond the edge of the 60x60x6 mm peg plate,
  so a low camera on the far side has the plate squarely between it and the
  goal -- in frame, and invisible. This is an exact slab test of each
  camera->corner ray against the plate box; it is the difference between
  "the goal is in shot" and "the goal is in shot and not behind the stand".
  """
  import torch as _t
  lo = _t.cat([plate_xy - PEG_PLATE_HALF[0],
               _t.zeros_like(plate_xy[:, :1])], dim=-1).unsqueeze(1)
  hi = _t.cat([plate_xy + PEG_PLATE_HALF[0],
               _t.full_like(plate_xy[:, :1], PEG_Z)], dim=-1).unsqueeze(1)
  o = pos.unsqueeze(1)
  d = goal_pts - o
  inv = 1.0 / _t.where(d.abs() < 1e-9, _t.full_like(d, 1e-9), d)
  t1, t2 = (lo - o) * inv, (hi - o) * inv
  tmin = _t.minimum(t1, t2).amax(dim=-1)
  tmax = _t.maximum(t1, t2).amin(dim=-1)
  # a hit strictly between the camera (t=0) and the corner (t=1) blocks it
  blocked = (tmax >= tmin) & (tmax > 1e-4) & (tmin < 1.0 - 1e-4)
  return ~blocked.any(dim=1)


def camera_pose_hemisphere(n, device, look, pts, fovy, stand, goal_pts,
                           plate_xy):
  """Sample a camera anywhere on the hemisphere that still frames everything.

  look (n,3) aim point, pts (n,k,3) everything that must stay visible, fovy
  (n,) degrees already drawn by the cam_fovy event, stand (n,3) the peg-stand
  base. The hemisphere is centred on the STAND and the radius is a fraction of
  the current stand-to-camera distance, because that is the constraint that
  actually matters -- it fixes how large the workspace appears. The camera then
  aims at `look` (the workspace centre), which is a few mm off the stand, so
  aim and distance are decoupled.
  """
  import torch as _t
  from mjlab.utils.lab_api.math import quat_from_matrix
  d2r = math.pi / 180.0
  # nominal = the stand-to-camera distance in the un-randomised scene
  _stand0 = _t.tensor([SPAWN_CENTER[0], SPAWN_CENTER[1], PEG_Z], dtype=_t.float32)
  nom = float((_t.tensor(CAM_POS) - _stand0).norm())

  def draw(m, shrink=0.0):
    az = (_t.rand(m, device=device) * (CAM_HEMI_AZIM[1] - CAM_HEMI_AZIM[0])
          + CAM_HEMI_AZIM[0]) * d2r
    # Retries narrow toward the middle of the elevation band rather than
    # toward the top: a camera directly overhead looks down the tool shaft and
    # the shaft occludes everything the visibility test just certified.
    lo, hi = CAM_HEMI_ELEV
    mid = 0.5 * (lo + hi)
    lo = lo + (mid - lo) * shrink
    hi = hi - (hi - mid) * shrink
    el = (_t.rand(m, device=device) * (hi - lo) + lo) * d2r
    r = (_t.rand(m, device=device) * (CAM_HEMI_DIST[1] - CAM_HEMI_DIST[0])
         + CAM_HEMI_DIST[0]) * nom
    off = _t.stack([r * _t.cos(el) * _t.cos(az),
                    r * _t.cos(el) * _t.sin(az),
                    r * _t.sin(el)], dim=-1)
    roll = (_t.rand(m, device=device) * 2 - 1) * CAM_HEMI_ROLL * d2r
    return off, roll

  def orient(p, lk, roll):
    fwd = lk - p
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    world_up = _t.tensor([0.0, 0.0, 1.0], device=device).expand_as(fwd)
    # degenerate when looking straight down; nudge the reference then
    bad = (fwd * world_up).sum(-1).abs() > 0.999
    world_up = _t.where(bad.unsqueeze(-1),
                        _t.tensor([0.0, 1.0, 0.0], device=device).expand_as(fwd),
                        world_up)
    right = _t.linalg.cross(fwd, world_up, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    up = _t.linalg.cross(right, fwd, dim=-1)
    cr, sr = _t.cos(roll).unsqueeze(-1), _t.sin(roll).unsqueeze(-1)
    up = up * cr + _t.linalg.cross(fwd, up, dim=-1) * sr
    zc = -fwd
    yc = up - (up * zc).sum(-1, keepdim=True) * zc
    yc = yc / yc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    xc = _t.linalg.cross(yc, zc, dim=-1)
    return quat_from_matrix(_t.stack([xc, yc, zc], dim=-1))

  off, roll = draw(n)
  pos = stand + off
  quat = orient(pos, look, roll)
  ok = (points_in_frame(pos, quat, pts, fovy)
        & goal_unoccluded(pos, goal_pts, plate_xy))
  for i in range(CAM_HEMI_TRIES):
    if bool(ok.all()):
      break
    bad = (~ok).nonzero(as_tuple=True)[0]
    # bias later attempts upward: a high camera sees a flat workspace best
    o2, r2 = draw(len(bad), shrink=i / CAM_HEMI_TRIES)
    pos[bad] = stand[bad] + o2
    quat[bad] = orient(pos[bad], look[bad], r2)
    ok[bad] = (points_in_frame(pos[bad], quat[bad], pts[bad], fovy[bad])
               & goal_unoccluded(pos[bad], goal_pts[bad], plate_xy[bad]))
  return pos, quat, ok


def camera_pose_batch(n, device, base_pos=None, base_pitch_deg=None, jitter=None):
  """Per-env camera position and quaternion, nominal pose plus jitter.

  Returns (pos [n,3], quat [n,4] wxyz). With jitter off this reproduces
  CAM_POS / camera_quat() exactly, so the DR arm and the fixed arm differ in
  the randomisation and nothing else.

  Angles are applied as pitch (tilt away from horizontal, the existing knob),
  then YAW about the world z axis, then ROLL about the camera's own optical
  axis. Roll matters more than it looks: it is the one rotation that does not
  move the workspace in frame, so a policy can silently learn the image's
  gravity direction as a cue and lose it the moment the mount is clocked a few
  degrees.
  """
  import torch
  from mjlab.utils.lab_api.math import quat_from_matrix

  if jitter is None:
    jitter = CAM_DR
  pos0 = CAM_POS if base_pos is None else base_pos
  pitch0 = CAM_PITCH_DEG if base_pitch_deg is None else base_pitch_deg

  pos = torch.tensor(pos0, device=device, dtype=torch.float32).expand(n, 3).clone()
  d2r = math.pi / 180.0
  pitch = torch.full((n,), pitch0 * d2r, device=device)
  yaw = torch.zeros(n, device=device)
  roll = torch.zeros(n, device=device)

  if jitter:
    def u(scale):
      return (torch.rand(n, device=device) * 2.0 - 1.0) * scale
    # Radial move FIRST, about the look-at point, so the workspace stays framed
    # -- a pure positional jitter of the same magnitude would swing the aim off
    # the peg instead of dollying toward it.
    look = torch.tensor(CAM_LOOKAT, device=device, dtype=torch.float32)
    lo_d, hi_d = CAM_DIST_SCALE
    k = torch.rand(n, 1, device=device) * (hi_d - lo_d) + lo_d
    pos = look.unsqueeze(0) + (pos - look.unsqueeze(0)) * k
    j = torch.stack([u(CAM_POS_JITTER[0]), u(CAM_POS_JITTER[1]),
                     u(CAM_POS_JITTER[2])], dim=-1)
    pos = pos + j
    pitch = pitch + u(CAM_PITCH_JITTER_DEG * d2r)
    yaw = u(CAM_YAW_JITTER_DEG * d2r)
    roll = u(CAM_ROLL_JITTER_DEG * d2r)

  cp, sp = torch.cos(pitch), torch.sin(pitch)
  ca, sa = torch.cos(yaw), torch.sin(yaw)
  z0 = torch.zeros(n, device=device)
  # Nominal forward (0, -cos p, -sin p) and up (0, -sin p, cos p), then rotated
  # about world z by the yaw. Written out rather than built as a matmul so the
  # zero x-component stays visible -- it is why the yaw terms look asymmetric.
  fwd = torch.stack([cp * sa, -cp * ca, -sp], dim=-1)
  up = torch.stack([sp * sa, -sp * ca, cp], dim=-1)
  # Roll about the optical axis. fwd . up is identically zero above, so
  # Rodrigues collapses to the two-term form.
  cr, sr = torch.cos(roll).unsqueeze(-1), torch.sin(roll).unsqueeze(-1)
  up = up * cr + torch.linalg.cross(fwd, up, dim=-1) * sr

  zc = -fwd                                   # camera +z is -forward
  yc = up - (up * zc).sum(-1, keepdim=True) * zc
  yc = yc / yc.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  xc = torch.linalg.cross(yc, zc, dim=-1)
  m = torch.stack([xc, yc, zc], dim=-1)       # columns are the camera axes
  del z0
  return pos, quat_from_matrix(m)


def camera_quat(pitch_deg: float | None = None) -> tuple[float, float, float, float]:
  """Unity's camera is euler (45, 0, 0): pitched down, looking along +z_unity.

  Unity forward (0, -sin45, cos45) and up (0, cos45, sin45) map through
  (x, -z, y) to forward (0, -c, -c) and up (0, -c, c) with c = cos45 -- i.e.
  looking toward -y and downward, which from (0, 92.5, 110) mm puts the optical
  axis on the floor at env y = -17.5 mm. That is ~29 mm beyond the peg, exactly
  as Unity frames it (its axis hits the floor at z = 117.5 mm vs the peg at
  88.5 mm); with fov 60 at ~156 mm the workspace sits comfortably in frame,
  just below centre.
  """
  p = CAM_PITCH_DEG if pitch_deg is None else pitch_deg
  c = math.cos(math.radians(p))
  s = math.sin(math.radians(p))
  return quat_from_forward_up((0.0, -c, -s), (0.0, -s, c))


def get_peg_spec() -> mujoco.MjSpec:
  """Static peg: a 4.4 mm post on a 60 mm plate. Mocap so it can be respawned.

  Unity gives the Peg colliders but no Rigidbody -- it is scenery that gets
  teleported on reset, never simulated. A mocap body reproduces that exactly:
  it collides, but its pose is written rather than integrated.
  """
  spec = mujoco.MjSpec()
  spec.add_mesh(name="peg_mesh", file=os.path.join(_MESH_DIR, "peg.obj"),
                scale=[MESH_SCALE] * 3)
  body = spec.worldbody.add_body(name="peg", mocap=True)
  # Visual: the real mesh, group 2 so it renders but never collides.
  vis = body.add_geom(name="peg_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname="peg_mesh", group=2, contype=0, conaffinity=0)
  vis.rgba = PEG_RGB + (1.0,)
  # Collision: primitives matched to Unity's Capsule + Box colliders. Cheaper
  # and far more robust than mesh-mesh contact at 4096 envs.
  #
  # The post MUST be a capsule, not a cylinder. mujoco_warp routes
  # (CYLINDER, BOX) through its convex/CCD narrowphase, and at this scale that
  # path silently returns nothing: `scripts/probe_liftpeg_pairsupport.py`
  # measures 4 contacts and 0.507 mm of penetration in MuJoCo against 0 in warp
  # for the identical pose, so the ring simply passed through the shaft. That
  # is why neither the 2 ms timestep nor a larger contact buffer helped -- no
  # contact was ever generated to integrate. (CAPSULE, BOX) is a native
  # PRIMITIVE routine and does not go near the CCD path.
  #
  # This is also closer to Unity, which uses a CapsuleCollider here; the
  # cylinder was the deviation. Half-length is shortened by the radius so the
  # cap TOP lands exactly on PEG_POST_TOP and the collider still ends where the
  # rendered mesh ends -- a camera-only policy must not see one post and feel
  # another.
  half = 0.5 * (PEG_POST_TOP - PEG_Z)
  if POST_GEOM == "capsule":
    post = body.add_geom(
      name="peg_post", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      size=(PEG_POST_RADIUS, half - PEG_POST_RADIUS, 0.0),
      pos=(0.0, 0.0, half), group=3,
    )
  else:
    post = body.add_geom(
      name="peg_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(PEG_POST_RADIUS, half, 0.0), pos=(0.0, 0.0, half), group=3,
    )
  _stiffen(post)
  post.rgba = (0.7, 0.7, 0.7, 0.0)
  plate = body.add_geom(
    name="peg_plate", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=PEG_PLATE_HALF, pos=(0.0, 0.0, -PEG_Z + PEG_PLATE_HALF[2]), group=3,
  )
  _stiffen(plate)
  plate.rgba = (0.7, 0.7, 0.7, 0.0)
  return spec


def _block_section():
  """The ring's true cross-section, read off the visual mesh. Cached.

  Returns (outer_hull_xy, inner_radius, z0, z1) in the BODY frame -- note the
  body frame, not the mesh frame: `triangle.obj` compiles with a non-identity
  `mesh_quat` (0.5, 0.5, -0.5, 0.5), and reading `mesh_vert` without applying it
  reports the extents permuted.

  The shape is a prism: vertices exist only on the two end faces, so the 2-D
  hull plus the z range describes it completely.
  """
  global _SECTION_CACHE
  if _SECTION_CACHE is not None:
    return _SECTION_CACHE
  import numpy as np
  from scipy.spatial import ConvexHull

  s = mujoco.MjSpec()
  s.add_mesh(name="m", file=os.path.join(_MESH_DIR, "triangle.obj"),
             scale=list(MESH_SCALE_XYZ))
  b = s.worldbody.add_body(name="b")
  b.add_geom(name="g", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m")
  m = s.compile()
  v0, nv = m.mesh_vertadr[0], m.mesh_vertnum[0]
  V = np.array(m.mesh_vert[v0:v0 + nv])
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, np.array(m.mesh_quat[0], dtype=float))
  W = V @ R.reshape(3, 3).T + np.array(m.mesh_pos[0])
  xy = W[:, :2]
  hull = xy[ConvexHull(xy).vertices]
  _SECTION_CACHE = (hull, float(np.linalg.norm(xy, axis=1).min()),
                    float(W[:, 2].min()), float(W[:, 2].max()))
  return _SECTION_CACHE


_SECTION_CACHE = None


def _sector_prisms(k: int):
  """Convex decomposition of the ring into `k` angular sector prisms.

  A single mesh geom cannot represent this shape: MuJoCo collides meshes by
  their CONVEX HULL, which would fill the central hole and make the ring
  impossible to thread onto the post. Slicing the annulus radially gives pieces
  that are individually convex, so each one's hull is exact.

  The inner edge of each sector CIRCUMSCRIBES the hole (tangent at the sector
  mid-angle) rather than cutting a chord through it. An inscribed chord would
  push collision material inward past the true bore and could bind on the post;
  circumscribing errs the harmless way, leaving a thin uncovered sliver at the
  bore that no grasp ever touches.
  """
  import numpy as np
  from scipy.spatial import ConvexHull

  hull, inner, z0, z1 = _block_section()

  def hull_ray(theta):
    d = np.array([math.cos(theta), math.sin(theta)])
    best = float("inf")
    for i in range(len(hull)):
      a, b = hull[i], hull[(i + 1) % len(hull)]
      e = b - a
      den = d[0] * (-e[1]) + d[1] * e[0]
      if abs(den) < 1e-12:
        continue
      t = (a[0] * (-e[1]) + a[1] * e[0]) / den
      if t <= 1e-9:
        continue
      s_ = np.dot(t * d - a, e) / np.dot(e, e)
      if -1e-6 <= s_ <= 1 + 1e-6:
        best = min(best, t)
    return best

  out = []
  for i in range(k):
    ta, tb = 2 * math.pi * i / k, 2 * math.pi * (i + 1) / k
    rin = inner / math.cos((tb - ta) / 2.0)
    pts = [[rin * math.cos(ta), rin * math.sin(ta)],
           [rin * math.cos(tb), rin * math.sin(tb)]]
    for t in (ta, tb):
      r = hull_ray(t)
      pts.append([r * math.cos(t), r * math.sin(t)])
    for v in hull:  # any outer corner falling inside this sector
      if 1e-9 < (math.atan2(v[1], v[0]) - ta) % (2 * math.pi) < (tb - ta) - 1e-9:
        pts.append(list(v))
    P = np.array(pts)
    P = P[ConvexHull(P).vertices]
    area = 0.5 * abs(sum(P[j][0] * P[(j + 1) % len(P)][1]
                         - P[(j + 1) % len(P)][0] * P[j][1]
                         for j in range(len(P))))
    verts = np.vstack([np.c_[P, np.full(len(P), z0)],
                       np.c_[P, np.full(len(P), z1)]])
    out.append((verts, area))
  return out


def get_block_spec() -> mujoco.MjSpec:
  """The graspable object: `triangle.obj`, 20.6 x 18.7 x 15.0 mm, 0.1 kg.

  Collision defaults to a convex decomposition that follows the visual mesh
  (`LIFTPEG_BLOCK_COLLISION=sectors`). Set `bars` to restore Unity's original
  three Side1/2/3 BoxColliders, which is what every run before 2026-08-15 used.

  **Why the bars were replaced.** The outer boundary is a triangle and the hole
  is a circle, so the wall is 1.217 mm thick at the edge midpoints and 5.434 mm
  at the corners. Three slabs laid along the edges cover the thin walls and
  nothing else: 38.4% of the cross-section -- 53.3 mm^2, all of it the corner
  mass -- rendered as solid while colliding with nothing. A jaw closing on a
  corner met no resistance and shut straight through the ring. That is visible
  in the v9 teacher rollouts and is NOT a rendering artifact.

  The sector decomposition covers 97.9% at k=12 with zero intrusion into the
  bore. Mass is distributed by area rather than split evenly, so the inertia
  tensor is closer to the real part than the three-bar model managed.
  """
  spec = mujoco.MjSpec()
  spec.add_mesh(name="block_mesh", file=os.path.join(_MESH_DIR, "triangle.obj"),
                scale=list(MESH_SCALE_XYZ))
  body = spec.worldbody.add_body(name="block")
  body.add_freejoint(name="block_joint")
  vis = body.add_geom(name="block_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname="block_mesh", group=2, contype=0, conaffinity=0)
  vis.rgba = BLOCK_RGB + (1.0,)

  if BLOCK_COLLISION == "bars":
    for i, yaw_deg in enumerate(BLOCK_BAR_YAWS_DEG):
      a = math.radians(yaw_deg)
      # Unity offsets each bar along its own local +y by BLOCK_BAR_RADIAL, so
      # the three bars close into a triangle rather than crossing at the centre.
      g = body.add_geom(
        name=f"block_bar{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=BLOCK_BAR_HALF, group=3,
        pos=(-BLOCK_BAR_RADIAL * math.sin(a),
             BLOCK_BAR_RADIAL * math.cos(a),
             BLOCK_BAR_Z),
        quat=(math.cos(a / 2.0), 0.0, 0.0, math.sin(a / 2.0)),
      )
      _stiffen(g)
      # Alpha 0 normally: group 3 is collision-only and must not appear in the
      # policy's camera. LIFTPEG_SHOW_COLLISION=1 makes the sector wedges
      # visible for inspection in viser -- the visual mesh OVERHANGS the
      # collision, so judging an interpenetration from the rendered mesh alone
      # is misleading. Never set it for training.
      g.rgba = BLOCK_RGB + (
        (0.55,) if os.environ.get("LIFTPEG_SHOW_COLLISION") == "1" else (0.0,))
      g.mass = BLOCK_MASS / len(BLOCK_BAR_YAWS_DEG)
    return spec

  prisms = _sector_prisms(BLOCK_SECTORS)
  total = sum(a for _, a in prisms)
  for i, (verts, area) in enumerate(prisms):
    spec.add_mesh(name=f"block_sec{i}", uservert=verts.ravel().tolist())
    g = body.add_geom(name=f"block_bar{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=f"block_sec{i}", group=3)
    _stiffen(g)
    g.rgba = BLOCK_RGB + (0.0,)
    g.mass = BLOCK_MASS * area / total
  return spec


def get_goal_spec() -> mujoco.MjSpec:
  """Green goal square on the floor, visual only.

  The goal used to be an abstract point at 60 mm with no renderer -- invisible
  to a camera-only policy, which is why neither a human nor the policy could
  tell where to put the triangle. This makes it observable. Visual only
  (contype 0): success is a position test, so the marker never perturbs the
  grasp or costs contact budget.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal", mocap=True)
  g = body.add_geom(
    name="goal_square", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.5 * GOAL_SQUARE_SIDE, 0.5 * GOAL_SQUARE_SIDE, 0.0002),
    # COLLIDABLE, unlike the original visual-only marker. Success is now "the
    # ring is touching the goal", read from the solver, which removes a whole
    # class of threshold argument: the previous test needed a height cutoff
    # (DOWN_Z) to mean "set down", and the goal sits only 4.7 mm from the 6 mm
    # peg plate while the ring is 22.9 mm across -- so a ring delivered to the
    # goal could perch on the plate at 7.2 mm, satisfy the XY test, and never
    # satisfy the height test. Contact has no such gap.
    #
    # 0.4 mm tall, so resting on it rather than the floor is a negligible step.
    pos=(0.0, 0.0, 0.0002), contype=1, conaffinity=1, group=2,
  )
  g.rgba = GOAL_SQUARE_RGB + (1.0,)
  return spec


def get_cam_spec(name: str = "liftcam") -> mujoco.MjSpec:
  """Camera on a mocap body so it tiles per-env. Pose is written at reset."""
  spec = mujoco.MjSpec()
  frame = spec.worldbody.add_body(name=f"{name}_frame", mocap=True)
  frame.add_camera(name=name, fovy=CAM_FOVY,
                   pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
  return spec


def get_cam2_spec() -> mujoco.MjSpec:
  """Second camera entity, for the multi-view arm."""
  return get_cam_spec("liftcam2")


def get_keylight_spec() -> mujoco.MjSpec:
  """Unity's single point light, on its OWN mocap body.

  Deliberately NOT on the camera frame: pushblock mounts its light there, but
  Unity's liftpeg light is fixed above the env origin and does not move with
  camera DR. Mounting it on the camera would couple the two.
  """
  spec = mujoco.MjSpec()
  frame = spec.worldbody.add_body(name="keylight_frame", mocap=True)
  light = frame.add_light(name="keylight")
  light.type = mujoco.mjtLightType.mjLIGHT_POINT
  light.pos = (0.0, 0.0, 0.0)
  light.castshadow = LIGHT_CASTSHADOW
  light.diffuse = LIGHT_DIFFUSE
  light.specular = LIGHT_SPECULAR
  # Without this the light has NO distance falloff and the scene blows out.
  light.attenuation = LIGHT_ATTENUATION
  return spec
