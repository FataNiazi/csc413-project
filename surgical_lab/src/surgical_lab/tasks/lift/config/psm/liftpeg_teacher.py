"""Scripted state-based teacher for liftpeg.

A hand-written waypoint controller that reads exact sim state (ring pose, post
pose, goal pose, tool pose) and drives the env's OWN action space. Two jobs:

1.  **Feasibility oracle.** Every liftpeg run so far plateaus at ~1.1 mm of a
    32 mm unthread, and four materially different reward designs all converge
    to the same number -- which is the signature of a ceiling, not of a reward
    problem. A controller that already knows the answer separates "the task
    cannot be done in this action space" from "RL has not found it yet". No
    reward, encoder or distillation work is worth doing until that is settled.

2.  **Teacher for distillation.** If it works, this is the privileged-state
    expert a DextrAH-style RGB student distils from, and the source of
    success-only frames for MAE pretraining. Both are currently blocked on
    liftpeg never having succeeded even once.

The controller is deliberately simple -- direct grasp, no regrasping and no
nudging the ring into a better orientation. That is a real limitation and it is
the point: if the naive grasp fails only at certain ring yaws, that localises
the problem to jaw/ring alignment rather than to control in general.

Why no IK: the action space is already cartesian position deltas
(`CartesianDeltaActionCfg(position_only=True, position_axes=(0,1,2))` at
`pos_scale=0.0005`), so this is a P-controller in task space -- clamp the error
to the per-step displacement budget and emit it. Joint space never enters.
"""

from __future__ import annotations

import math
import os

import torch

from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as LR
from surgical_lab.tasks.lift.config.psm.liftpeg_grasp import contact_both_jaws
from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

# Phase ids for the vectorised state machine.
ALIGN, DESCEND, CLOSE, LIFT, CARRY, PLACE, RELEASE, DONE = range(8)
PHASE_NAMES = ("align", "descend", "close", "lift", "carry", "place",
               "release", "done")

# These are TOOL heights, while CLEAR_Z / Z_REST are BLOCK-ORIGIN heights, and
# the grasp site sits PICKPOINT_LOCAL[2] = 13.6 mm above the block origin. Mixing
# the two frames is the obvious mistake here and it is silent: the first version
# of this file set the lift target to CLEAR_Z + 6 mm, which is BELOW the
# approach height and ~14 mm short of ever clearing the post.
# --- constants measured from 14 successful teleop demos -------------------
# (scripts/analyze_liftpeg_demos.py over data/liftpeg_demos/). Medians with the
# inter-quartile range; the previous values were derived from geometry and were
# wrong in ways that are invisible until you watch a rollout stall.
#
# GRASP_LOCAL_Z is deliberately teacher-local rather than a change to
# LR.PICKPOINT_LOCAL: that constant also defines `d_reach` in the reward and the
# grasp points in the teacher's observation, so moving it would quietly alter
# the training task and break comparison with every run so far.
#
#   grip offset above the ring origin  10.32 mm  IQR [ 9.95 .. 13.13]  (was 13.60)
#   travel height                      71.00 mm  IQR [70.34 .. 72.48]  (was 50.00)
#   ring z at peak lift                46.97 mm  IQR [44.48 .. 48.22]
#   ring z at release                   4.14 mm  IQR [ 4.03 ..  4.28]
#
# The grip offset is the one that matters: DESCEND->CLOSE gates on |dz| < Z_TOL
# = 0.8 mm, so being 3.3 mm high meant that gate could essentially never open.
# 24% of attempts stalled there.
GRASP_LOCAL_Z = float(os.environ.get("LIFTPEG_GRASP_Z", "0.01032"))
# Where between the bore and the outer boundary the tool centre sits. 0.5 is
# mid-wall. Demo grip radii ran 4.9-9.4 mm (median 7.7) against a bore of 6.0.
GRIP_FRAC = float(os.environ.get("LIFTPEG_GRIP_FRAC", "0.5"))
APPROACH_Z = float(os.environ.get("LIFTPEG_APPROACH_Z", "0.071"))
# Rise so the ring reaches ~47 mm: tool sits GRASP_LOCAL_Z above the ring, which
# is at ~6.0 mm when gripped, so the tool goes 16.3 -> 57.3 mm.
LIFT_RISE = float(os.environ.get("LIFTPEG_LIFT_RISE", "0.041"))
# Tool height that puts the ring at the measured 4.1 mm release height.
PLACE_Z = GRASP_LOCAL_Z + 0.0041
CLOSE_STEPS = 40            # let the jaw actually clamp before lifting
RELEASE_STEPS = 25
STUCK_STEPS = 400        # a phase this long has not converged; flag it
DROP_MARGIN = 0.006      # the ring must FALL this far to count as dropped
DROP_LOW = 0.010         # ...and end up this close to its resting height
# The retry loop already recovers from a genuine drop via the stall timeout, so
# whether this detector earns its keep at all is an empirical question.
DROP_RULE = os.environ.get("LIFTPEG_DROP_RULE", "1") == "1"
RETRY_STEPS = int(os.environ.get("LIFTPEG_RETRY_STEPS", "150"))  # retry well before STUCK
MAX_RETRIES = 9          # three passes over the three faces
XY_TOL = 0.0010
Z_TOL = 0.0008
GOAL_XY_TOL = 0.0025

POS_SCALE = 0.0005          # must match CartesianDeltaActionCfg.pos_scale


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  w, u = q[:, 0:1], q[:, 1:4]
  return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


class LiftpegTeacher:
  """Vectorised scripted controller. One phase counter per env."""

  def __init__(self, env):
    self.env = env
    self.n = env.num_envs
    self.dev = env.device
    self._resolve_action_layout()
    self._resolve_jaw_axis()
    self.phase = torch.zeros(self.n, dtype=torch.long, device=self.dev)
    self.timer = torch.zeros(self.n, dtype=torch.long, device=self.dev)
    # Grasp target is latched at CLOSE: after the jaw shuts, the tool follows
    # absolute waypoints, not the ring (which is now moving with it).
    self.latched = torch.zeros((self.n, 3), device=self.dev)
    self.stuck = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
    # Recovery. Humans retry: the demos include attempts that failed and were
    # recovered, and the script previously had no such path -- a stalled
    # DESCEND stayed stalled for the rest of the episode (28/256 of them), and
    # a grasp lost mid-lift was never re-attempted (another 28/256).
    # `face_rank` selects the Nth-best face instead of always the best, so a
    # retry does not repeat the approach that just failed.
    self.face_rank = torch.zeros(self.n, dtype=torch.long, device=self.dev)
    self.retries = torch.zeros(self.n, dtype=torch.long, device=self.dev)
    self.n_retry_grasp = torch.zeros(self.n, dtype=torch.long, device=self.dev)
    self.n_retry_drop = torch.zeros(self.n, dtype=torch.long, device=self.dev)
    self.drop_from_z = torch.zeros(self.n, device=self.dev)
    self.lift_peak = torch.zeros(self.n, device=self.dev)

  # --- one-time introspection ---------------------------------------------

  def _resolve_action_layout(self):
    """Find where the position and jaw slots live, rather than assuming."""
    am = self.env.action_manager
    self.total_dim = am.total_action_dim
    self.pos_slice, self.jaw_slice = None, None
    off = 0
    for name in am.active_terms:
      dim = am.get_term(name).action_dim if hasattr(am, "get_term") else None
      if dim is None:  # fall back to the cfg-declared ordering
        dim = 3 if "cartesian" in name else 1
      if "cartesian" in name:
        self.pos_slice = slice(off, off + dim)
      elif "jaw" in name:
        self.jaw_slice = slice(off, off + dim)
      off += dim
    if self.pos_slice is None or self.jaw_slice is None:
      # Documented default for this task: (dx, dy, dz, jaw).
      self.pos_slice, self.jaw_slice = slice(0, 3), slice(3, 4)
    print(f"[teacher] action layout: total={self.total_dim} "
          f"pos={self.pos_slice} jaw={self.jaw_slice}")

  def _resolve_jaw_axis(self):
    """Measure the jaw's closing axis from the pad geoms.

    The tool's orientation is FIXED for the whole episode (position_only=True),
    so this axis is a constant of the scene and decides which triangle face can
    actually be straddled. Measured rather than assumed, because getting it
    wrong silently produces a controller that always grasps across a corner.
    """
    import mujoco

    m = self.env.sim.mj_model
    left, right = [], []
    for i in range(m.ngeom):
      nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
      if "_collision" not in nm:
        continue
      if "/lg" in nm or nm.startswith("lg"):
        left.append(i)
      elif "/rg" in nm or nm.startswith("rg"):
        right.append(i)
    self.jaw_axis = None
    if left and right:
      gx = self.env.sim.data.geom_xpos
      lp = gx[0, torch.tensor(left, device=self.dev)].mean(dim=0)
      rp = gx[0, torch.tensor(right, device=self.dev)].mean(dim=0)
      axis = (rp - lp)[:2]
      if float(torch.norm(axis)) > 1e-6:
        self.jaw_axis = (axis / torch.norm(axis)).to(self.dev)
    if self.jaw_axis is None:
      self.jaw_axis = torch.tensor([1.0, 0.0], device=self.dev)
      print("[teacher] WARNING: could not measure the jaw axis from pad geoms; "
            "defaulting to world +x. Verify before trusting a sweep.")
    else:
      ang = math.degrees(math.atan2(float(self.jaw_axis[1]),
                                    float(self.jaw_axis[0])))
      print(f"[teacher] jaw closing axis (world xy): "
            f"[{float(self.jaw_axis[0]):+.3f}, {float(self.jaw_axis[1]):+.3f}] "
            f"= {ang:+.1f} deg")

  # --- geometry ------------------------------------------------------------

  def grasp_target(self):
    """Grip where the ring's RADIAL direction lines up with the jaw axis.

    The previous version picked one of the three bar centres by
    |normal . jaw_axis|. That is the wrong abstraction, and it is why retrying
    faces could not close the last few percent: the jaw's closing axis is FIXED
    in world (position_only actions), so a wall can only be straddled where the
    ring's radial direction is parallel to that axis. The ring is a CLOSED LOOP,
    so such a point always exists at any yaw -- there is nothing to select.

    Measured across 14 successful human demos: the angle between the jaw axis
    and the world-frame ring-centre-to-tool direction at the moment of grasp had
    a median of 11.7 deg, with 13/14 inside 30 deg, clustering at 169 deg and
    349 deg -- i.e. exactly +/- the jaw axis. The humans were never choosing a
    face; they were putting the jaw where the wall crossed it.

    Two candidates remain (+u and -u). They are ranked by local wall THICKNESS,
    because the jaw has to open across it: the wall is 1.2 mm at a face and
    5.4 mm at a corner, and the thin side is the one that fits. `face_rank`
    alternates between them on retry.
    """
    blk = self.env.scene["block"]
    q = blk.data.root_link_quat_w
    origin = blk.data.root_link_pos_w - self.env.scene.env_origins
    n = self.n

    # ring yaw about z, from the quaternion
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    jaw_ang = math.atan2(float(self.jaw_axis[1]), float(self.jaw_axis[0]))

    cands = []
    for sign in (0.0, math.pi):
      world_ang = jaw_ang + sign
      local = world_ang - yaw                       # same direction, ring frame
      outer = self._outer_radius(local)             # ring boundary along it
      thick = (outer - self._bore).clamp_min(1e-4)
      r = self._bore + thick * GRIP_FRAC            # sit inside the wall
      d = torch.stack([torch.cos(torch.full_like(yaw, world_ang)),
                       torch.sin(torch.full_like(yaw, world_ang))], dim=-1)
      pt = torch.cat([origin[:, :2] + r.unsqueeze(-1) * d,
                      (origin[:, 2] + GRASP_LOCAL_Z).unsqueeze(-1)], dim=-1)
      cands.append((thick, pt))

    T = torch.stack([c[0] for c in cands], dim=1)   # (n, 2) wall thickness
    P = torch.stack([c[1] for c in cands], dim=1)   # (n, 2, 3)
    order = torch.argsort(T, dim=1)                 # thinnest wall first
    pick = order.gather(1, (self.face_rank % 2).unsqueeze(1)).squeeze(1)
    idx = torch.arange(n, device=self.dev)
    best = P[idx, pick]
    self.last_misalign = torch.zeros(n, device=self.dev)  # aligned by construction
    self.last_thickness = T[idx, pick]
    return best

  def _outer_radius(self, ang):
    """Ring outer boundary at a ring-frame angle, from a precomputed table."""
    if not hasattr(self, "_rad_table"):
      import numpy as np
      hull, bore, _z0, _z1 = LS._block_section()
      K = 720
      th = np.linspace(0.0, 2 * math.pi, K, endpoint=False)
      rad = np.zeros(K)
      for i, t in enumerate(th):
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
      self._rad_table = torch.tensor(rad, dtype=torch.float32, device=self.dev)
      self._bore = float(bore)
    K = self._rad_table.shape[0]
    i = ((ang % (2 * math.pi)) / (2 * math.pi) * K).long() % K
    return self._rad_table[i]

  # --- control -------------------------------------------------------------

  def act(self) -> torch.Tensor:
    env = self.env
    tool = a3b._tool_pos(env)
    grasp = self.grasp_target()
    goal = env.scene["goal"].data.root_link_pos_w - env.scene.env_origins

    # Latch the grasp point the moment the jaw starts closing.
    closing = self.phase == CLOSE
    self.latched = torch.where(closing.unsqueeze(-1), grasp, self.latched)
    held = self.latched

    target = torch.zeros((self.n, 3), device=self.dev)
    jaw_open = torch.ones(self.n, device=self.dev)

    def put(mask, xyz, open_jaw):
      nonlocal target, jaw_open
      target = torch.where(mask.unsqueeze(-1), xyz, target)
      jaw_open = torch.where(mask, torch.full_like(jaw_open,
                                                   1.0 if open_jaw else -1.0),
                             jaw_open)

    z_app = torch.full((self.n, 1), APPROACH_Z, device=self.dev)
    # Relative to the LATCHED grasp height, so the rise is the real 32 mm
    # unthread regardless of where on the shaft this ring happened to sit.
    self._z_lift = held[:, 2] + LIFT_RISE
    z_lift = self._z_lift.unsqueeze(-1)
    z_place = torch.full((self.n, 1), PLACE_Z, device=self.dev)

    put(self.phase == ALIGN, torch.cat([grasp[:, :2], z_app], -1), True)
    put(self.phase == DESCEND, grasp, True)
    put(self.phase == CLOSE, held, False)
    put(self.phase == LIFT, torch.cat([held[:, :2], z_lift], -1), False)
    # The tool holds the ring at a BAR, 6.59 mm off the ring's centre, but
    # success tests the ring's CENTRE against a square of half-width 10.3 mm.
    # Driving the tool to the goal therefore lands the ring 6.59 mm off,
    # spending two thirds of the tolerance before any other error. Offset the
    # target by the live tool-to-ring vector so the RING lands on the goal; live
    # rather than latched, so it self-corrects if the ring shifts in the jaws.
    ring_xy = blk_xy = (env.scene["block"].data.root_link_pos_w
                        - env.scene.env_origins)[:, :2]
    hold_off = tool[:, :2] - ring_xy
    goal_tool_xy = goal[:, :2] + hold_off
    put(self.phase == CARRY, torch.cat([goal_tool_xy, z_lift], -1), False)
    put(self.phase == PLACE, torch.cat([goal_tool_xy, z_place], -1), False)
    put(self.phase == RELEASE, tool, True)
    put(self.phase == DONE, tool, True)

    self._advance(tool, grasp, goal)

    err = target - tool
    act = torch.zeros((self.n, self.total_dim), device=self.dev)
    act[:, self.pos_slice] = (err / POS_SCALE).clamp(-1.0, 1.0)
    act[:, self.jaw_slice] = jaw_open.unsqueeze(-1)
    return act

  def _advance(self, tool, grasp, goal):
    p = self.phase
    self.timer += 1
    xy_err = torch.norm(tool[:, :2] - grasp[:, :2], dim=-1)
    # Measure the RING against the goal, not the tool. The tool now aims
    # deliberately off-centre so the ring lands on target (see act()), so a
    # tool-based test would never fire and CARRY would run until STUCK_STEPS.
    _ring_xy = (self.env.scene["block"].data.root_link_pos_w
                - self.env.scene.env_origins)[:, :2]
    goal_err = torch.norm(_ring_xy - goal[:, :2], dim=-1)

    step = torch.zeros_like(p, dtype=torch.bool)
    step |= (p == ALIGN) & (xy_err < XY_TOL) & (tool[:, 2] > APPROACH_Z - 0.002)
    # BOTH axes, not just z. Checking height alone let the tool descend to the
    # correct depth at the wrong xy and then close on nothing -- which looks
    # exactly like "the grasp failed" and would have been read as a result.
    step |= (p == DESCEND) & (xy_err < XY_TOL) & (
      (tool[:, 2] - grasp[:, 2]).abs() < Z_TOL)
    step |= (p == CLOSE) & (self.timer > CLOSE_STEPS)
    step |= (p == LIFT) & (tool[:, 2] > self._z_lift - Z_TOL)
    step |= (p == CARRY) & (goal_err < GOAL_XY_TOL)
    step |= (p == PLACE) & (tool[:, 2] < PLACE_Z + Z_TOL)
    step |= (p == RELEASE) & (self.timer > RELEASE_STEPS)

    # --- recovery -----------------------------------------------------------
    # A stalled approach retries on the next-best face rather than holding the
    # same failing pose until the episode ends.
    timed_out = (self.timer > RETRY_STEPS) & (p < DONE)
    grasping = p <= CLOSE
    retry_grasp = timed_out & grasping & (self.retries < MAX_RETRIES)
    # A grasp lost after the lift began: restart the approach rather than carry
    # nothing to the goal.
    #
    # This keys on the CONTACT flag even though that flag is measurably noisy
    # (P(grasped | during lift) = 0.730 on a scripted lift), and it therefore
    # fires ~184 times per 256-env run on grasps that were actually fine. That
    # is deliberate. Measured alternatives, 256 envs, 2000-step window:
    #
    #   ~contact & z < CLEAR_Z   (this)      at goal 0.958   CARRY failures  1
    #   fell & low, gated LIFT|CARRY         at goal 0.926   CARRY failures 15
    #   no drop rule at all                  at goal 0.840   CARRY failures 40
    #
    # The trade is asymmetric: a false restart costs ~150 steps to re-grasp,
    # while a missed drop wastes the whole episode carrying air. The noisy flag
    # is FAST, and waiting for the ring to visibly fall is accurate but slow --
    # by the time it fires the tool has translated toward the goal and recovery
    # is much harder. Eager and wrong beats late and right here.
    #
    # Two rules that look more principled and are worse: keying on the ring
    # falling 6 mm fires on a healthy carry that sags (748/run), and extending
    # the phase gate past CARRY catches the PLACE descent, which lowers the ring
    # 47 -> 4 mm by design (739/run).
    ring_z = (self.env.scene["block"].data.root_link_pos_w
              - self.env.scene.env_origins)[:, 2]
    held = contact_both_jaws(self.env)
    dropped = (p > CLOSE) & (p < DONE) & (~held) & (ring_z < LR.CLEAR_Z) & DROP_RULE
    retry_drop = dropped & (self.retries < MAX_RETRIES)
    restart = retry_grasp | retry_drop

    self.n_retry_grasp += retry_grasp.long()
    self.n_retry_drop += retry_drop.long()
    self.drop_from_z = torch.where(retry_drop, ring_z, self.drop_from_z)
    self.retries = torch.where(restart, self.retries + 1, self.retries)
    self.face_rank = torch.where(retry_grasp, self.face_rank + 1, self.face_rank)
    self.lift_peak = torch.where(restart, torch.zeros_like(self.lift_peak),
                                 self.lift_peak)
    # Only give up once the retries are spent.
    self.stuck |= timed_out & (self.retries >= MAX_RETRIES) & ~restart

    self.phase = torch.where(step & (p < DONE), p + 1, p)
    self.phase = torch.where(restart, torch.full_like(p, ALIGN), self.phase)
    self.timer = torch.where(step | restart, torch.zeros_like(self.timer),
                             self.timer)

  def reset(self, env_ids=None):
    idx = slice(None) if env_ids is None else env_ids
    self.phase[idx] = ALIGN
    self.face_rank[idx] = 0
    self.retries[idx] = 0
    self.lift_peak[idx] = 0.0
    self.timer[idx] = 0
    self.latched[idx] = 0.0
    self.stuck[idx] = False
