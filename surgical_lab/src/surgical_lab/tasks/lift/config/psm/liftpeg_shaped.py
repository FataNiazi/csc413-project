"""Heavily shaped liftpeg reward, built from what the scripted teacher taught us.

Every design choice here is a response to a measured failure earlier in this
task's history. In order:

1. THE GRASP GATE MUST NOT BE SATISFIABLE BY RESTING ON THE RING.
   `contact_both_jaws` only asks whether both pads touch the block, not where.
   The 2026-08-20 run learned to park the jaw on the ring's top rim, which
   registered `grasped` 71.7% of the time while `lift_frac` stayed at 0.0004 --
   it held the reward's grasp stage permanently without ever gripping. The
   sector collision fix made that easier, not harder, by turning three 2 mm
   ridges into a full annular shelf. So `grasped_strict` additionally requires
   the jaw to be CLOSED, at the right HEIGHT, and near the jaw-aligned grip
   POINT.

2. THE GRIP POINT IS SET BY THE JAW AXIS, NOT BY A FACE.
   The jaw's closing axis is fixed in world (position_only actions), so a wall
   can only be straddled where the ring's radial direction is parallel to it.
   The ring is a closed loop, so such a point exists at every yaw. Human demos
   confirm: grip direction sat within 30 deg of the jaw axis in 13/14 episodes.
   The reward guides to that point, and the observation exposes it, so the
   policy is not asked to rediscover the geometry.

3. LIFTING MUST BE WORTH ATTEMPTING.
   The old staged potential paid 1.0 on grasp and (1/32) per mm of lift, so one
   dropped contact cost the equivalent of 32 successful lift steps and freezing
   was optimal. Here the lift stage is worth LIFT_GAIN (>> 1) so the gradient
   points up, and the grasp bonus is small by comparison.
"""

from __future__ import annotations

import math
import os

import torch

from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as LR
from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS
from surgical_lab.tasks.lift.config.psm.liftpeg_grasp import (
  contact_both_jaws,
  quat_apply,
  _jaw_cmd,
)
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

# WHERE ON THE RING THE JAW IS AIMED, measured up from the ring's origin --
# which is its BOTTOM. The ring's mesh spans 0..15 mm from that origin.
#
# Raised 10.32 -> 13.0 mm on 2026-09-11. The ring rests at 5.99 mm on a 6 mm
# plate, so 10.32 mm put the grip target at 16.3 mm absolute, and GRIP_TOL_Z
# lets the tool sit 4 mm under that -- 12.3 mm, barely 6 mm clear of the plate.
# The jaw tips extend below the tool point, so at the bottom of that window
# they reach the peg BASE, and closing then lifting tows the whole fixture.
# That is the "messing up the scene" seen on hardware.
#
# The evidence is a controlled one: the same policy run against a SHALLOWER
# peg -- identical in every way except depth -- barely disturbs the scene. The
# variable is how far into the fixture the jaw goes, not the policy.
#
# 13.0 mm sits 2 mm below the ring's top, putting the target at 19.0 mm
# absolute and the bottom of the tolerance window at 15.0 mm, ~9 mm clear of
# the plate. It stays inside GRIP_TOL_Z of the OLD target (2.68 mm < 4 mm), so
# a warm-started policy still registers as grasping and does not collapse.
GRASP_LOCAL_Z = float(os.environ.get("LIFTPEG_GRASP_Z", "0.01032"))
GRIP_FRAC = float(os.environ.get("LIFTPEG_GRIP_FRAC", "0.5"))
# How close the tool must be to the grip point for a grasp to count.
#
# GRIP_TOL_XY MUST NOT EXCEED HALF THE WALL, or the gate accepts poses with no
# material between the jaws. The ring wall is 4.27 mm at its thinnest and
# 5.13 mm at the median, so the historic +-3.5 mm (a 7 mm window) left
# 0.94-1.37 mm of the window INSIDE the bore and the same again outside the
# ring -- only 4.27 of 7.0 mm on material at the thin end.
#
# The bore is an open annulus from the post (r 2.20) to the hole (r 6.00), so a
# tool centred in the inner slice descends INTO the ring's hole around the post
# and the jaws close on the post, or on nothing. On video that is "it goes deep
# into the peg and never closes" -- observed on both sim and hardware, and
# present in every arm including E, because this value never changed.
#
# 2.0 mm keeps the window on material for the thin wall (6.13..10.13 against a
# 6.00..10.27 wall) and the median (6.56..10.56 against 6.00..11.13).
GRIP_TOL_XY = float(os.environ.get("LIFTPEG_GRIP_TOL_XY_MM", "3.5")) / 1000.0
GRIP_TOL_Z = float(os.environ.get("LIFTPEG_GRIP_TOL_Z_MM", "4.0")) / 1000.0
JAW_CLOSED = 0.25            # binary jaw: open 0.5, closed 0.0

REACH_K = 12.0
GOAL_K = 6.0
LIFT_GAIN = float(os.environ.get("LIFTPEG_LIFT_GAIN", "6.0"))
CARRY_GAIN = 3.0
PLACE_GAIN = 3.0
# "Straight up before sideways." Hardware, 2026-09-09: after the grasp, while
# rising and still below the post top, the tool starts translating and the ring
# tows the whole peg base. Everything after the ring clears is clean.
#
# Applied MULTIPLICATIVELY to the lift stage's gain rather than subtracted from
# the stage total. Subtracting would push `lift` below `grasp_base` (1.0), which
# is also the reach stage's ceiling -- and since the stage is selected by `held`
# rather than by a max, a lift worth less than reach makes LETTING GO a positive
# potential step. Scaling the gain keeps lift in [1.0, 1+LIFT_GAIN] always, so
# dropping the ring can never pay.
LIFT_LATERAL_PEN = float(os.environ.get("LIFTPEG_LIFT_LATERAL_PEN", "0.0"))
assert 0.0 <= LIFT_LATERAL_PEN < 1.0, "LIFTPEG_LIFT_LATERAL_PEN must be in [0,1)"


def _jaw_axis(env) -> torch.Tensor:
  """World-frame jaw closing axis, measured once from the pad geoms."""
  got = getattr(env, "_liftpeg_jaw_axis", None)
  if got is not None:
    return got
  import mujoco

  m = env.sim.mj_model
  left, right = [], []
  for i in range(m.ngeom):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    leaf = n.split("/")[-1]
    if "_collision" not in n:
      continue
    if leaf.startswith("lg"):
      left.append(i)
    elif leaf.startswith("rg"):
      right.append(i)
  ax = torch.tensor([1.0, 0.0], device=env.device)
  if left and right:
    gx = env.sim.data.geom_xpos
    lp = gx[0, torch.tensor(left, device=env.device)].mean(dim=0)
    rp = gx[0, torch.tensor(right, device=env.device)].mean(dim=0)
    v = (rp - lp)[:2]
    if float(torch.norm(v)) > 1e-6:
      ax = v / torch.norm(v)
  env._liftpeg_jaw_axis = ax
  return ax


def _radius_table(env) -> tuple[torch.Tensor, float]:
  """Ring outer radius vs ring-frame angle, and the bore radius."""
  got = getattr(env, "_liftpeg_rad_table", None)
  if got is not None:
    return got
  import numpy as np

  hull, bore, _z0, _z1 = LS._block_section()
  K = 720
  rad = np.zeros(K)
  for i in range(K):
    t = 2 * math.pi * i / K
    d = np.array([math.cos(t), math.sin(t)])
    best = 0.0
    for j in range(len(hull)):
      a, b = hull[j], hull[(j + 1) % len(hull)]
      e = b - a
      den = d[0] * (-e[1]) + d[1] * e[0]
      if abs(den) < 1e-12:
        continue
      tt = (a[0] * (-e[1]) + a[1] * e[0]) / den
      if tt <= 1e-9:
        continue
      ss = np.dot(tt * d - a, e) / np.dot(e, e)
      if -1e-6 <= ss <= 1 + 1e-6:
        best = max(best, tt)
    rad[i] = best
  out = (torch.tensor(rad, dtype=torch.float32, device=env.device), float(bore))
  env._liftpeg_rad_table = out
  return out


def grip_targets(env):
  """The two jaw-aligned grip points, ordered thinnest-wall first.

  Returns (points (n,2,3) env-local, wall thickness (n,2)).
  """
  blk = env.scene["block"]
  q = blk.data.root_link_quat_w
  origin = blk.data.root_link_pos_w - env.scene.env_origins
  tab, bore = _radius_table(env)
  ax = _jaw_axis(env)
  jaw_ang = math.atan2(float(ax[1]), float(ax[0]))
  yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  K = tab.shape[0]
  pts, ths = [], []
  for extra in (0.0, math.pi):
    wa = jaw_ang + extra
    local = wa - yaw
    idx = ((local % (2 * math.pi)) / (2 * math.pi) * K).long() % K
    outer = tab[idx]
    thick = (outer - bore).clamp_min(1e-4)
    r = bore + thick * GRIP_FRAC
    d = torch.stack([torch.full_like(yaw, math.cos(wa)),
                     torch.full_like(yaw, math.sin(wa))], dim=-1)
    p = torch.cat([origin[:, :2] + r.unsqueeze(-1) * d,
                   (origin[:, 2] + GRASP_LOCAL_Z).unsqueeze(-1)], dim=-1)
    pts.append(p)
    ths.append(thick)
  P = torch.stack(pts, dim=1)
  T = torch.stack(ths, dim=1)
  order = torch.argsort(T, dim=1)
  i = torch.arange(P.shape[0], device=env.device)
  return P[i.unsqueeze(1), order], T.gather(1, order)


def grasped_strict(env) -> torch.Tensor:
  """A real pinch: contact, jaw closed, and the tool AT a grip point.

  Contact alone is what the 2026-08-20 run exploited by resting on the rim.
  """
  P, _ = grip_targets(env)
  tool = a3b._tool_pos(env)
  d = P - tool.unsqueeze(1)
  near = ((torch.norm(d[..., :2], dim=-1) < GRIP_TOL_XY)
          & (d[..., 2].abs() < GRIP_TOL_Z)).any(dim=1)
  closed = _jaw_cmd(env) < JAW_CLOSED
  return contact_both_jaws(env) & closed & near


def shaped_terms(env):
  """Task quantities for the shaped reward. Cached per step like LR.terms."""
  key = int(getattr(env, "common_step_counter", -1))
  got = getattr(env, "_liftpeg_shaped_cache", None)
  if got is not None and got[0] == key:
    return got[1]
  t = LR.terms(env)
  P, T = grip_targets(env)
  tool = t["tool"]
  d = P - tool.unsqueeze(1)
  dist = torch.norm(d, dim=-1)
  d_grip = dist.min(dim=1).values
  held = grasped_strict(env)
  out = dict(t)
  out.update(grip=P, thick=T, d_grip=d_grip, held=held)
  env._liftpeg_shaped_cache = (key, out)
  return out


def m_held(env) -> torch.Tensor:
  return shaped_terms(env)["held"].float()


def m_grip_mm(env) -> torch.Tensor:
  return shaped_terms(env)["d_grip"] * 1000.0


def shaped_potential(t) -> torch.Tensor:
  """Monotone staged potential with a LARGE lift stage.

  Stage ceilings never overlap, so progress can never be undone by moving to a
  later stage -- robosuite's nested-ceiling construction. Unlike the old version
  the lift stage spans LIFT_GAIN rather than 1.0, so a step of lift is worth far
  more than the risk of a momentary contact loss.
  """
  reach = 1.0 - torch.tanh(REACH_K * t["d_grip"])            # 0 .. 1
  grasp_base = 1.0
  # off_axis * (1 - clear_h) is non-zero ONLY when the ring is both off the
  # post axis AND below the clearance height: exactly "moving sideways before
  # it is up". Rising straight zeroes it; clearing the post zeroes it.
  sideways_low = t["off_axis"] * (1.0 - t["clear_h"])
  lift = grasp_base + LIFT_GAIN * t["lift_frac"] * (
    1.0 - LIFT_LATERAL_PEN * sideways_low)                   # 1 .. 1+LIFT_GAIN
  carry_base = 1.0 + LIFT_GAIN
  carry = carry_base + CARRY_GAIN * (1.0 - torch.tanh(GOAL_K * t["d_goal_xy"]))
  place_base = carry_base + CARRY_GAIN
  down = ((LR.CLEAR_Z - t["z"]) / max(LR.CLEAR_Z - LR.DOWN_Z, 1e-6)).clamp(0, 1)
  place = place_base + PLACE_GAIN * down

  phi = torch.where(t["held"], lift, reach)
  phi = torch.where(t["cleared"] & t["held"], carry, phi)
  # only pay the set-down stage once the ring is over the goal
  over = t["d_goal_xy"] < LR.GOAL_HALF
  phi = torch.where(t["cleared"] & over, place, phi)
  return phi


class arm_e_shaped:
  """Potential-based shaping on `shaped_potential`."""

  def __init__(self, env=None):
    self.prev = None

  def __call__(self, env) -> torch.Tensor:
    t = shaped_terms(env)
    phi = shaped_potential(t)
    if self.prev is None or self.prev.shape != phi.shape:
      self.prev = phi.detach().clone()
    r = phi - self.prev
    self.prev = phi.detach().clone()
    ids = getattr(env, "reset_buf", None)
    if ids is not None:
      self.prev = torch.where(ids.bool(), phi.detach(), self.prev)
    return r


# --- release height: put it DOWN, do not drop it ---------------------------
# Hardware, 2026-09-10: the wrist Alpha3 policy clears the post cleanly and
# carries to the goal, then lets go in mid-air. The ring falls, bounces and
# ends up off-target, so placement accuracy is poor.
#
# The `place` stage cannot fix this and no amount of tuning it will. Shaping is
# emitted as phi(s') - phi(s) and telescopes to phi(T) - phi(0), so a ring
# DROPPED from 30 mm that lands touching the goal reaches exactly the same
# terminal potential as one lowered gently to the same spot. The reward cannot
# tell them apart, and dropping is faster, so under discounting it is strictly
# preferred. Gentleness was not under-rewarded, it was unrepresented.
#
# So this is charged OUTSIDE the potential, as a one-shot event penalty at the
# instant the ring is let go -- the same shape as `lost_penalty`.
#
# The reference height is human: the demonstrations released roughly 1-2.5 cm
# above the goal area. RELEASE_Z is the top of "free", the low end of that
# band, measured on the ring's ORIGIN (its bottom) which rests at 5.99 mm.
RELEASE_Z = float(os.environ.get("LIFTPEG_RELEASE_Z_MM", "16.0")) / 1000.0
RELEASE_SPAN = float(os.environ.get("LIFTPEG_RELEASE_SPAN_MM", "25.0")) / 1000.0
RELEASE_PEN = float(os.environ.get("LIFTPEG_RELEASE_PEN", "0.0"))


class release_penalty:
  """Charge for letting go of the ring high above the plate.

  Fires once, on the held -> not-held transition, and only after the ring has
  cleared the post: before that a lost grip is already covered by `lost`, and
  the regrasps during the approach are not releases.

  Bounded rather than linear in height. An unbounded charge would make a single
  bad drop dominate a whole episode's return, and the thing being discouraged
  is a discrete choice, not a quantity to be minimised without limit. At full
  scale it is RELEASE_PEN = 2.0, which exactly cancels `success_bonus`: a ring
  flung from RELEASE_Z + RELEASE_SPAN that happens to land on the goal is worth
  nothing, while one set down inside the human band keeps the whole bonus.

  `held` comes from `grasped_strict`, which is hysteretic, so a single-frame
  contact flicker does not read as a release.
  """

  def __init__(self, cfg=None, env=None):
    self.prev = None

  def reset(self, env_ids=None) -> None:
    if self.prev is not None:
      self.prev[slice(None) if env_ids is None else env_ids] = False

  def __call__(self, env) -> torch.Tensor:
    t = shaped_terms(env)
    held = t["held"]
    if self.prev is None or self.prev.shape != held.shape:
      self.prev = held.detach().clone()
      return torch.zeros_like(t["z"])
    released = self.prev & ~held & t["cleared"]
    self.prev = held.detach().clone()
    excess = ((t["z"] - RELEASE_Z) / max(RELEASE_SPAN, 1e-6)).clamp(0.0, 1.0)
    return -RELEASE_PEN * excess * released.float()


def m_release_z_mm(env) -> torch.Tensor:
  """Absolute height at which the ring was let go. Seeded to 0 = never let go."""
  got = getattr(env, "_liftpeg_release_z", None)
  if got is None or got.shape[0] != env.num_envs:
    got = torch.zeros(env.num_envs, device=env.device)
    env._liftpeg_release_z = got
  t = shaped_terms(env)
  prev = getattr(env, "_liftpeg_release_prev", None)
  if prev is None or prev.shape != t["held"].shape:
    env._liftpeg_release_prev = t["held"].detach().clone()
    return got * 1000.0
  fired = prev & ~t["held"] & t["cleared"]
  env._liftpeg_release_prev = t["held"].detach().clone()
  got[:] = torch.where(fired, t["z"], got)
  return got * 1000.0


def reset_release_z(env, env_ids=None) -> None:
  got = getattr(env, "_liftpeg_release_z", None)
  if got is not None:
    got[slice(None) if env_ids is None else env_ids] = 0.0


# --- how deep the jaw actually goes ----------------------------------------
# GRASP_LOCAL_Z was raised on 2026-09-11 to stop the jaw reaching into the peg
# base, and nothing in the metrics could see that change. `grip_mm` is
# `d_grip`, a 3-D distance to the grip point, so a tool 4 mm BELOW the target
# and one 4 mm above it report the same number -- exactly blind to depth.
#
# Two numbers, because they answer different questions:
#   grasp_z_mm  where the jaw closed. Compare against the target (18.99 mm at
#               GRASP_LOCAL_Z 13.0) to see whether the policy followed it up.
#   tool_dip_mm the LOWEST the tool ever got before clearing. This is the
#               towing-risk number: the jaw tips hang below the tool point, so
#               the deepest excursion is what reaches the base, and an episode
#               can grasp at a fine height yet still have dipped on the way in.
class _GraspDepth:
  """Per-episode grasp height and deepest tool excursion, in metres."""

  _FAR = 9.99

  def __init__(self, env):
    n, dev = env.num_envs, env.device
    self.grasp_z = torch.zeros(n, device=dev)      # 0 = never grasped
    self.dip = torch.full((n,), self._FAR, device=dev)
    self.prev_held = torch.zeros(n, dtype=torch.bool, device=dev)
    self.key = -1

  def update(self, env):
    k = int(getattr(env, "common_step_counter", -1))
    if k == self.key:            # idempotent: two metric terms read this
      return self
    self.key = k
    t = shaped_terms(env)
    tool_z = t["tool"][:, 2]
    held = t["held"]
    # First close of the episode only -- later regrasps are recovery, and the
    # question is where it chose to grab.
    first = (~self.prev_held) & held & (self.grasp_z == 0.0)
    self.grasp_z = torch.where(first, tool_z, self.grasp_z)
    self.prev_held = held.detach().clone()
    # Deepest point BEFORE clearing. After the ring is off the post a low tool
    # is the set-down, which is not a towing risk.
    live = ~t["cleared"]
    self.dip = torch.where(live, torch.minimum(self.dip, tool_z), self.dip)
    return self


def _grasp_depth(env) -> _GraspDepth:
  got = getattr(env, "_liftpeg_graspdepth", None)
  if got is None or got.grasp_z.shape[0] != env.num_envs:
    got = _GraspDepth(env)
    env._liftpeg_graspdepth = got
  return got.update(env)


def m_grasp_z_mm(env) -> torch.Tensor:
  """Absolute tool height at the first grasp. 0 = never grasped."""
  return _grasp_depth(env).grasp_z * 1000.0


def m_tool_dip_mm(env) -> torch.Tensor:
  """Lowest absolute tool height reached before the ring cleared the post."""
  d = _grasp_depth(env).dip
  return torch.where(d >= _GraspDepth._FAR, torch.zeros_like(d), d) * 1000.0


def reset_grasp_depth(env, env_ids=None) -> None:
  got = getattr(env, "_liftpeg_graspdepth", None)
  if got is None:
    return
  sl = slice(None) if env_ids is None else env_ids
  got.grasp_z[sl] = 0.0
  got.dip[sl] = _GraspDepth._FAR
  got.prev_held[sl] = False
