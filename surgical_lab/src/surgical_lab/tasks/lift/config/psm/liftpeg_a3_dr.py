"""Appearance domain randomisation for Alpha3 liftpeg.

Gated by ``LIFTPEG_A3_DR=1`` and registered from the env cfg, so the a3b
distribution is unchanged when it is off. Everything here was written after an
audit (``scripts/dr_audit.py``) found most of liftpeg's appearance DR was doing
nothing at all:

* **Colour on any part with a material never rendered.** A geom with
  ``matid >= 0`` ignores ``geom_rgba`` -- the renderer reads the material.
  Measured by painting each geom red then blue: every material-backed geom gave
  a render delta of exactly 0.00. That covered tool shaft, tool roll link and
  gripper, i.e. three of liftpeg's four colour terms. Only the floor ever
  worked. Those parts are randomised here through ``mat_rgba`` instead, keyed
  by material -- so parts sharing ``silver`` change together, which is what a
  real tool does anyway.

* **Light intensity never varied per env.** ``light_diffuse`` was a stride-0
  expanded tensor, so all worlds shared one brightness. It is not that the
  field cannot hold per-world values: nothing had *declared* it.
  ``@requires_model_fields`` is what makes ``expand_model_fields`` allocate
  real per-world memory, and it is generic -- there is no whitelist. Declaring
  the field is the entire fix.

The mount rides the shaft as a RIGID UNIT with its camera. The bracket and the
lens are one physical part, so randomising the bracket alone slides the housing
off the camera -- visible as the bracket occluding a corner of the wrist view
at +10 mm rise. Both move together here.
"""

from __future__ import annotations

import math
import os

import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields

A3_DR = os.environ.get("LIFTPEG_A3_DR", "0") == "1"
# The real bench has a black barrier wall to one side. It is invisible to the
# FRONT camera (0.0-1.4 % dark pixels, never a band) but sweeps through the
# WRIST view as the arm travels -- the dark band there runs 0 to 35 px wide
# within a single episode, which is what identifies it as a world-fixed object
# rather than a lens vignette. Modelling it as vignetting would have been
# wrong in a way that got worse the further the arm moved.
WALL = os.environ.get("LIFTPEG_A3_WALL", "1") == "1"
_BOX = 6


def add_wall(spec):
  """Black barrier on a mocap body, so its pose is per-env like every other
  fixture. A worldbody geom would sit at one place for all envs -- the whole
  scene is tiled in ONE coordinate space and offset by env_origins."""
  body = spec.worldbody.add_body(name="blackwall", pos=[0.0, 0.0, 0.0])
  body.mocap = True
  g = body.add_geom(name="blackwall_geom", type=_BOX,
                    size=[WALL_SIZE[0], WALL_SIZE[1], WALL_SIZE[2]],
                    pos=[0, 0, 0], rgba=[0.05, 0.05, 0.06, 1.0])
  # Visual only: it must never collide with the arm.
  g.contype, g.conaffinity, g.group = 0, 0, 2


def _u(n, lo, hi, device):
  return torch.rand(n, device=device) * (hi - lo) + lo


def _cache(env):
  c = getattr(env, "_a3dr", None)
  if c is not None:
    return c
  mj = env.sim.mj_model
  c = {}
  name2mat = {mj.material(i).name: i for i in range(mj.nmat)}
  c["mat"] = {k: name2mat[k] for k in
              ("robot/black", "robot/silver", "robot/mount_white",
               "robot/off_white") if k in name2mat}
  # Geoms whose colour is NOT material-backed, so geom_rgba is the right field.
  def gids(pred):
    return [g for g in range(mj.ngeom)
            if pred(mj.geom(g).name or "") and int(mj.geom_matid[g]) < 0]
  c["stand"] = gids(lambda n: n.startswith("peg/"))
  c["ring"] = gids(lambda n: n.startswith("block/"))
  c["goal"] = gids(lambda n: n.startswith("goal/"))
  c["wall_mocap"] = (int(mj.body_mocapid[mj.body("blackwall").id])
                     if any((mj.body(b).name or "") == "blackwall"
                            for b in range(mj.nbody)) else None)
  c["wall_g"] = next(
    (g for g in range(mj.ngeom)
     if (mj.geom(g).name or "") == "blackwall_geom"), None)
  c["mount_g"] = next(
    (g for g in range(mj.ngeom)
     if (mj.geom(g).name or "").endswith("wrist_camera_mount")), None)
  c["wrist_ci"] = next(
    (i for i in range(mj.ncam) if "wrist_camera" in mj.camera(i).name), None)
  m = env.sim.model
  if c["mount_g"] is not None:
    c["mount_pos0"] = m.geom_pos[0, c["mount_g"]].clone()
    c["mount_quat0"] = m.geom_quat[0, c["mount_g"]].clone()
  if c["wrist_ci"] is not None:
    c["cam_pos0"] = m.cam_pos[0, c["wrist_ci"]].clone()
    c["cam_quat0"] = m.cam_quat[0, c["wrist_ci"]].clone()
  env._a3dr = c
  return c


def _qmul(a, b):
  w1, x1, y1, z1 = a.unbind(-1)
  w2, x2, y2, z2 = b.unbind(-1)
  return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


# --------------------------------------------------------------------------


@requires_model_fields("mat_rgba")
def material_colors(env, env_ids, ranges: dict):
  """Colour every material-backed part. The field the RENDERER actually reads."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  c = _cache(env)
  m = env.sim.model
  n = len(env_ids)
  for name, (lo, hi) in ranges.items():
    mid = c["mat"].get(name)
    if mid is None:
      continue
    # One shared draw across the three channels plus a small per-channel tint,
    # so a part stays recognisably itself instead of turning an arbitrary hue.
    base = _u(n, lo, hi, env.device).unsqueeze(-1)
    tint = 1.0 + (torch.rand((n, 3), device=env.device) * 2.0 - 1.0) * 0.06
    m.mat_rgba[env_ids, mid, :3] = (base * tint).clamp(0.0, 1.0)


def _hsv_to_rgb(h, s, v):
  """h in degrees, s/v in [0,1]; all (n,) -> (n,3)."""
  c = v * s
  hp = (h / 60.0) % 6.0
  x = c * (1.0 - torch.abs(hp % 2.0 - 1.0))
  z = torch.zeros_like(c)
  i = hp.floor().long().clamp(0, 5).unsqueeze(-1)
  r = torch.stack([c, x, z, z, x, c], -1).gather(-1, i).squeeze(-1)
  g = torch.stack([x, c, c, x, z, z], -1).gather(-1, i).squeeze(-1)
  b = torch.stack([z, z, x, c, c, x], -1).gather(-1, i).squeeze(-1)
  m = v - c
  return torch.stack([r + m, g + m, b + m], -1).clamp(0.0, 1.0)


def stand_color(env, env_ids, spec: dict):
  """Colour the stand around its nominal HUE rather than per channel."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  c = _cache(env)
  gs = c.get("stand") or []
  if not gs:
    return
  n = len(env_ids)
  j = float(spec["h_jitter"])
  h = _u(n, spec["h_deg"] - j, spec["h_deg"] + j, env.device)
  sat = _u(n, spec["s"][0], spec["s"][1], env.device)
  val = _u(n, spec["v"][0], spec["v"][1], env.device)
  col = _hsv_to_rgb(h, sat, val)
  m = env.sim.model
  for g in gs:
    m.geom_rgba[env_ids, g, :3] = col


def object_colors(env, env_ids, ranges: dict):
  """Colour the fixtures. These have no material, so geom_rgba is correct."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  c = _cache(env)
  m = env.sim.model
  n = len(env_ids)
  for key, chans in ranges.items():
    gs = c.get(key) or []
    if not gs:
      continue
    col = torch.stack([_u(n, lo, hi, env.device) for lo, hi in chans], -1)
    for g in gs:
      m.geom_rgba[env_ids, g, :3] = col


@requires_model_fields("light_diffuse", "light_ambient")
def light_appearance(env, env_ids, gain: tuple, ambient: tuple, tint: float):
  """Per-env key-light brightness, tint and ambient fill.

  The whole point of declaring the fields: without the decorator these are
  stride-0 and every env shares one value, which is what liftpeg has always
  done while appearing to randomise brightness.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  m = env.sim.model
  n = len(env_ids)
  g = _u(n, gain[0], gain[1], env.device).unsqueeze(-1)
  t = 1.0 + (torch.rand((n, 3), device=env.device) * 2.0 - 1.0) * tint
  m.light_diffuse[env_ids, 0] = (g * t).clamp(0.0, 8.0)
  a = _u(n, ambient[0], ambient[1], env.device).unsqueeze(-1)
  m.light_ambient[env_ids, 0] = a.expand(-1, 3).clone()


@requires_model_fields("geom_pos", "geom_quat", "cam_fovy")
def mount_pose(env, env_ids, spin_deg: float, rise_m: float,
               wrist_fov: float = 0.0, rise_offset_m: float = 0.0):
  """Slide and spin the wrist bracket on the shaft, camera rigidly with it.

  Also pins the wrist camera's render fov. The XML ships fovy 90, but the real
  unit is a fisheye: the pipeline renders WIDE and resamples that to the fitted
  fisheye. Rendering at 90 and warping as though it were 165 does not merely
  distort -- it fills the frame with the tool and throws the scene away.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  c = _cache(env)
  if c["mount_g"] is None:
    return
  m = env.sim.model
  n = len(env_ids)
  a = _u(n, -spin_deg, spin_deg, env.device) * (math.pi / 180.0)
  dz = _u(n, -rise_m, rise_m, env.device) + rise_offset_m
  ca, sa = torch.cos(a), torch.sin(a)
  h = a * 0.5
  qz = torch.stack([torch.cos(h), torch.zeros_like(h), torch.zeros_like(h),
                    torch.sin(h)], -1)

  def spin(p0):
    x = p0[0] * ca - p0[1] * sa
    y = p0[0] * sa + p0[1] * ca
    return torch.stack([x, y, p0[2] + dz], -1)

  m.geom_pos[env_ids, c["mount_g"]] = spin(c["mount_pos0"])
  m.geom_quat[env_ids, c["mount_g"]] = _qmul(
    qz, c["mount_quat0"].unsqueeze(0).expand(n, -1))
  if c["wrist_ci"] is not None:
    m.cam_pos[env_ids, c["wrist_ci"]] = spin(c["cam_pos0"])
    m.cam_quat[env_ids, c["wrist_ci"]] = _qmul(
      qz, c["cam_quat0"].unsqueeze(0).expand(n, -1))
    if wrist_fov > 0:
      m.cam_fovy[env_ids, c["wrist_ci"]] = wrist_fov


# --------------------------------------------------------------------------
# ranges. Wide on purpose: these feed an MAE that has to survive a real rig we
# have matched in geometry but never in lighting.

MATERIAL_RANGES = {
  "robot/black": (0.06, 0.55),        # shaft, nominal 0.30
  "robot/silver": (0.55, 1.00),       # tool links, nominal 0.80
  "robot/mount_white": (0.62, 1.00),  # bracket, nominal 0.93
  "robot/off_white": (0.62, 1.00),
}
OBJECT_RANGES = {
  # Independent per-channel boxes. Fine for these two -- they stay recognisably
  # green and red across the whole box -- but NOT for the stand, see below.
  "ring": ((0.00, 0.40), (0.35, 1.00), (0.02, 0.45)),    # green
  "goal": ((0.50, 1.00), (0.02, 0.42), (0.02, 0.38)),    # red
}
# The stand is sampled in HSV instead. Drawing R and G independently over a wide
# box is a box in RGB whose CORNERS are not the colour you started from: at
# (0.84, 0.49, 0.13) the yellow plate rendered orange and at (0.66, 0.63, 0.16)
# olive, a hue swing of about +-40 deg. Randomising around the nominal hue keeps
# "yellow stand" an identity the policy can rely on while still covering white
# balance and material differences, which is what the randomisation is for.
# Nominal PEG_RGB (1.0, 0.8, 0.0) is exactly H=48 deg, S=1, V=1.
STAND_HSV = {"h_deg": 48.0, "h_jitter": 8.0, "s": (0.72, 1.00), "v": (0.55, 1.00)}
# CALIBRATED against the real captures, not guessed. The first ranges rendered
# sim 1.57x brighter than real on the front camera and 2.41x on the wrist, with
# sim p95 clipping at 254 -- every real frame sat below the dimmest 5 % of sim,
# so the two domains did not overlap on brightness at all. Blown highlights
# also desaturated sim (0.575 against real's 0.713), which is why the fix is
# less light rather than more colour.
LIGHT_GAIN = tuple(float(x) for x in os.environ.get(
  "LIFTPEG_A3_LIGHT_GAIN", "0.55,2.10").split(","))
LIGHT_AMBIENT = tuple(float(x) for x in os.environ.get(
  "LIFTPEG_A3_LIGHT_AMBIENT", "0.02,0.22").split(","))
LIGHT_TINT = 0.10
# Half-extents. Long and tall enough to read as a wall rather than a slab from
# every pose the wrist reaches.
WALL_SIZE = (0.0035, 0.131, 0.060)
# Nominal pose, placed BY HAND in the scene editor against the real footage.
# An automated fit on the wrist band width could not recover it -- sweeping x
# alone produced the inverted pattern (wide early, absent late) because the
# wall's bearing, not just its distance, is what sets when it enters view.
WALL_POS = tuple(float(x) for x in os.environ.get(
  "LIFTPEG_A3_WALL_POS", "-0.140,0.010,0.060").split(","))
WALL_YAW_DEG = float(os.environ.get("LIFTPEG_A3_WALL_YAW", "0.0"))
# WIDE on purpose. The hand-placed pose is correct for the real bench, but a
# sim episode only carries the tool from x = 0 to about -37 mm, where a wall at
# -162 mm never enters the wrist view: measured, the sim wrist reached 14 % dark
# pixels against a real 22 % mean and 46 % p95. Randomising the barrier's
# distance, height and bearing gives the encoder the black-barrier appearance
# across the range the real footage actually shows, without pretending to know
# where the wall is. The near limit stays well clear of the goal at -48.5 mm so
# the barrier can never occlude the task itself.
WALL_POS_DR = (0.075, 0.060, 0.040)
WALL_YAW_DR = 28.0
# Half-extents, randomised: a taller barrier reaches the wrist view from
# further out without having to be brought into the workspace to do it.
WALL_SIZE_DR = ((0.002, 0.006), (0.090, 0.180), (0.055, 0.165))
WALL_GREY = (0.02, 0.16)

# The bracket is positioned by hand on the shaft, so these were set wide. But
# the RISE is a slide ALONG THE SHAFT, which on the real rig cannot vary: the
# bracket is clamped once and stays there (measured 2026-09-17: 35-55 mm, top
# of the mount 55 mm from its base). Randomising it asks the policy to be
# robust to a variation reality does not produce, and drags the sim image
# distribution away from the real one it has to match.
#
# RISE_OFFSET pins the bracket at a FIXED displacement instead, for runs that
# want one chosen standoff rather than a distribution.
MOUNT_SPIN_DEG = float(os.environ.get("LIFTPEG_MOUNT_SPIN_DEG", "15.0"))
MOUNT_RISE_M = float(os.environ.get("LIFTPEG_MOUNT_RISE_M", "0.010"))
# DEFAULT IS +10 mm, deliberately -- this is where the real bracket sits.
#
# Measured on the rig 2026-09-19: with the camera at this standoff the distance
# from the EE jaw to the top of the camera mount is 53 mm, and that position
# gave the best grasp height we have seen on hardware. Every run in the current
# line -- J-far, J-far2, J3 -- pins or centres the bracket here, so the default
# now matches the hardware instead of the 0 mm the scene happened to ship with.
#
# NOTE the two knobs stack: dz = U(-MOUNT_RISE_M, +MOUNT_RISE_M) + OFFSET. The
# RISE_M default is still the legacy +-10 mm, so taking both defaults spans
# 0..20 mm, which is NOT what the real clamped bracket does. Current runs pass
# LIFTPEG_MOUNT_RISE_M=0.001 for +9..+11 mm. Narrowing the RISE_M default to
# match is a live question, not yet settled -- J3 (+-1 mm) escaped the
# hold-still plateau at iteration 355 while J-far2 (pinned, 0 mm) took until
# ~1900, but that is n=1 per arm and seed could explain it.
MOUNT_RISE_OFFSET_M = float(os.environ.get("LIFTPEG_MOUNT_RISE_OFFSET_M", "0.010"))
# Wide pinhole the fisheye warp resamples from. Must match the --wrist-render-fov
# the dump passes to fisheye.build_grid, or the mapping is nonsense.
WRIST_RENDER_FOV = float(os.environ.get("LIFTPEG_WRIST_RENDER_FOV", "135.0"))


@requires_model_fields("geom_rgba", "geom_size")
def wall_pose(env, env_ids, pos, yaw_deg, pos_dr, yaw_dr, grey, size_dr=None):
  """Place the barrier per env, with jitter, and vary how black it is."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  c = _cache(env)
  mi = c.get("wall_mocap")
  if mi is None or mi < 0:
    # Loud, once. An earlier version caught the lookup failure and returned
    # silently, so the wall stayed at the origin through an entire dataset dump
    # and the only symptom was a dark-pixel fraction that refused to change.
    if not getattr(env, "_a3dr_wall_warned", False):
      env._a3dr_wall_warned = True
      print("[a3_dr] blackwall body not found -- wall DR is a no-op", flush=True)
    return
  n = len(env_ids)
  base = torch.tensor(pos, device=env.device).unsqueeze(0)
  jit = torch.stack([_u(n, -d, d, env.device) for d in pos_dr], -1)
  p = base + jit
  y = (_u(n, yaw_deg - yaw_dr, yaw_deg + yaw_dr, env.device)
       * (math.pi / 180.0) * 0.5)
  q = torch.stack([torch.cos(y), torch.zeros_like(y), torch.zeros_like(y),
                   torch.sin(y)], -1)
  # Written straight into mocap_pos/quat: `blackwall` is a body added by the
  # scene spec_fn, not a registered scene Entity, so `env.scene["blackwall"]`
  # does not resolve it.
  origins = env.scene.env_origins[env_ids]
  d = env.sim.data
  d.mocap_pos[env_ids, mi] = origins + p
  d.mocap_quat[env_ids, mi] = q
  g = c.get("wall_g")
  if g is not None:
    v = _u(n, grey[0], grey[1], env.device).unsqueeze(-1)
    env.sim.model.geom_rgba[env_ids, g, :3] = v.expand(-1, 3)
    if size_dr is not None:
      env.sim.model.geom_size[env_ids, g] = torch.stack(
        [_u(n, lo, hi, env.device) for lo, hi in size_dr], -1)


def register(events: dict) -> None:
  """Add the appearance terms. Called from the env cfg when A3_DR is on."""
  events["a3_material_colors"] = EventTermCfg(
    mode="reset", func=material_colors, params={"ranges": MATERIAL_RANGES})
  events["a3_object_colors"] = EventTermCfg(
    mode="reset", func=object_colors, params={"ranges": OBJECT_RANGES})
  events["a3_stand_color"] = EventTermCfg(
    mode="reset", func=stand_color, params={"spec": STAND_HSV})
  events["a3_light"] = EventTermCfg(
    mode="reset", func=light_appearance,
    params={"gain": LIGHT_GAIN, "ambient": LIGHT_AMBIENT, "tint": LIGHT_TINT})
  events["a3_mount_pose"] = EventTermCfg(
    mode="reset", func=mount_pose,
    params={"spin_deg": MOUNT_SPIN_DEG, "rise_m": MOUNT_RISE_M,
            "wrist_fov": WRIST_RENDER_FOV,
            "rise_offset_m": MOUNT_RISE_OFFSET_M})
  if WALL:
    events["a3_wall"] = EventTermCfg(
      mode="reset", func=wall_pose,
      params={"pos": WALL_POS, "yaw_deg": WALL_YAW_DEG,
              "pos_dr": WALL_POS_DR, "yaw_dr": WALL_YAW_DR,
              "grey": WALL_GREY, "size_dr": WALL_SIZE_DR})
