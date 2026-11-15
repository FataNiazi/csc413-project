"""Rules-based grasping of a flex vertex, with a stochastic failure rate.

Why this is not contact-based
-----------------------------
Two independent reasons, neither of them laziness:

1. It cannot be. MuJoCo Warp's flex narrowphase is vertex-based and cannot feel
   geometry thinner than roughly 3x the cloth's vertex spacing
   (docs/cloth_simulation_report.md, Rule 3). The PSM jaw is ~2 mm across and
   the sheet's spacing is 6.25 mm, so the jaw passes straight through it. On
   top of that, the tissue's floor had to be implemented as a joint limit
   rather than a contact -- mujoco_warp does not replicate flex-vs-geom
   collision across worlds -- which leaves the sheet with no collision at all.

2. It should not be. The real tissue is self-adhesive silicone rubber that can
   stick to itself and tear. No contact model we could fit would predict when a
   real grasp slips. Modelling the grasp as "succeeds with probability p"
   concentrates all of that unmodelled physics into ONE number we can set
   honestly and randomise, instead of spreading it across a dozen parameters we
   would be guessing at. The policy then has to learn the behaviour that
   actually transfers: close, lift, check whether the tissue came with you, and
   go again if it did not. That retry loop is what survived contact with the
   real rig on liftpeg.

The grasp itself has two implementations, selected by TISSUE_GRASP_MODE:

* ``weld`` (the historical default) drives the grabbed patch to the tool tip by
  writing its slide-joint qpos each step. A position write transmits no force,
  so only the patch moves and the rest of the sheet never gets loaded -- that is
  the local tenting that reads as folding paper.
* ``connect`` toggles a real mjEQ_CONNECT equality, which the solver enforces
  with a FORCE that propagates through the mesh. Measured against weld on the
  scripted oracle: peak lift 27.4 mm vs 8.0 mm, and 25/64 successes vs 0/64.
  Requires a live integrator -- a bare ``sim.forward()`` shows nothing.
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Tool tip must be this close to the anchor vertex for a close to count.
# Overridable because it is a fitted tolerance, not a measured length: it
# stands in for the jaw's capture volume plus whatever tip-position error the
# replay carries, so it is exactly the knob to sweep when the pre-grasp pose
# looks offset against the real wrist view.
GRASP_RADIUS = float(os.environ.get("TISSUE_GRASP_RADIUS", "0.010"))
# Test the grasp against the NEAREST free vertex instead of the fixed corner.
#
# The jaw does not seek out the corner -- it closes on whatever tissue is
# between it, and on Sep2 that is a different point along the free edge every
# episode (measured nearest vertex 7, 16, 17, 26 across five). Measuring to
# GRASP_IDS[0] therefore denies grasps that are physically happening: e17 and
# e29 read 12.6 and 12.0 mm to the corner against a 10 mm radius while the jaw
# sat 2.0 and 4.3 mm from the tissue it was touching, and neither ever grasped.
GRASP_NEAREST = os.environ.get("TISSUE_GRASP_NEAREST", "0") == "1"
# The jaw does not take one vertex, it takes a PATCH -- a coherent piece of
# silicone comes away with the forceps. This is a real modelling parameter, not
# a fudge: measured reveal of the goal square goes 0.00 / 0.12 / 0.48 for a
# 1x1 / 2x2 / 3x3 patch, because a single driven vertex just tents locally
# while the pinned edges hold everything else flat.
# In VERTICES, so it must scale with the grid or the physical patch shrinks:
# 3 vertices span 2 spacings, which is 12.5 mm at a 9x9 grid and 6.25 mm at
# 17x17. The measured goal-square reveal (0.00 / 0.12 / 0.48 for 1x1 / 2x2 /
# 3x3) was taken at 9x9, so keeping that behaviour at 17x17 wants 5.
GRASP_PATCH = int(os.environ.get("TISSUE_GRASP_PATCH", "3"))
# How far the grasped patch may be dragged from where it was picked up.
#
# Without this the weld is unbounded: the patch goes exactly where the tool
# goes, the membrane stretches to follow, and the sheet TEARS OPEN -- measured
# 8.05x edge stretch, with a visible hole showing the floor through the middle
# of the tissue. Stiffness cannot fix it (8.05x at both young=3e3 and 1e5),
# because a kinematic position write ignores elastic force by construction.
#
# So the drag is clamped: pulled taut, the tissue stops following the tool.
# Pull further still and it SLIPS out of the jaw, which is what real tissue
# does and gives the policy one more honest way to fail.
# Grip compliance. Below this, tool motion does not move the tissue at all --
# a real grasp has slack in it and does not transmit sub-millimetre jitter. It
# also removes the last of the "did it work?" tell: with the tool nominally
# still, a held grasp was still dragging the sheet 0.30 mm because the arm
# drifts slightly, which is enough to see.
GRASP_SLACK = float(os.environ.get("TISSUE_GRASP_SLACK", "0.0015"))
# MAX_DRAG and SLIP_DRAG were both fitted to keep the sheet intact, and both are
# far below what the rig actually does. Measured over all 30 Sep2 episodes, the
# tool's pull from where the jaw closed peaks at mean 60.8 mm (sd 5.8, range
# 48.6-72.9), and the real arm never lets go -- so at SLIP_DRAG = 32 mm the sim
# would drop the tissue in 30 of 30 real demonstrations, and at MAX_DRAG = 22 mm
# the sheet stops following the tool a third of the way through every one.
MAX_DRAG = float(os.environ.get("TISSUE_MAX_DRAG", "0.022"))
SLIP_DRAG = float(os.environ.get("TISSUE_SLIP_DRAG", "0.032"))
# Probability an attempt actually takes hold. Not a nuisance parameter -- it is
# the knob that buys retry behaviour, and it is domain-randomised per episode.
GRASP_P_RANGE = (0.35, 0.75)
# Jaw command below this reads as closed (open=0.5, closed=0.0).
JAW_CLOSED_BELOW = 0.25


class _GraspState:
  """Per-env grasp bookkeeping plus the model addressing, resolved once."""

  def __init__(self, env: ManagerBasedRlEnv, vertex_ids, entity: str):
    m = env.sim.mj_model
    self.qadr, self.dadr = [], []
    for vid in vertex_ids:
      bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{entity}/tissue_{vid}")
      if bid < 0:
        continue          # pinned vertices have no body; they stay pinned
      ja, jn = int(m.body_jntadr[bid]), int(m.body_jntnum[bid])
      if jn != 3:
        raise ValueError(f"expected 3 slide joints on a vertex body, got {jn}")
      self.qadr.append([int(m.jnt_qposadr[ja + k]) for k in range(3)])
      self.dadr.append([int(m.jnt_dofadr[ja + k]) for k in range(3)])
    if not self.qadr:
      raise ValueError("no free vertices in the grasp patch")
    self.anchor_bid = mujoco.mj_name2id(
      m, mujoco.mjtObj.mjOBJ_BODY, f"{entity}/tissue_{vertex_ids[0]}")
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/jaw")
    self.jaw_qadr = int(m.jnt_qposadr[jid])

    n = env.num_envs
    self.held = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.was_closed = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.p = torch.full((n,), 0.5, device=env.device)
    self.attempts = torch.zeros(n, dtype=torch.int32, device=env.device)
    # tool-tip position at the instant the grasp took hold; the patch is then
    # rigidly translated by (tip_now - anchor), which keeps its shape instead
    # of collapsing every grabbed vertex onto a single point.
    self.anchor = torch.zeros(n, 3, device=env.device)
    # Each grabbed vertex's qpos AT THE INSTANT OF GRASP. Driving from spawn
    # instead teleports the patch the moment a grasp takes hold -- the sheet
    # has sagged ~1.4 mm onto the soft floor limit by then, so qpos=0 pops it
    # back up. That made success visible with the tool standing still (0.807 mm
    # of tissue motion vs 0.000 mm when the grasp failed), which let a human --
    # or a policy -- just open and close on the spot until the tissue twitched.
    # A real grasp gives no such tell: you find out when you lift.
    self.q0 = torch.zeros(n, len(self.qadr), 3, device=env.device)


def patch_ids(count: int, corner, patch: int = GRASP_PATCH):
  """Vertex ids of the `patch` x `patch` block anchored at `corner`, growing
  INTO the sheet (the corner is the outermost vertex)."""
  ix0, iy0 = corner
  sx = 1 if ix0 == 0 else -1
  sy = 1 if iy0 == 0 else -1
  out = []
  for dx in range(patch):
    for dy in range(patch):
      ix, iy = ix0 + sx * dx, iy0 + sy * dy
      if 0 <= ix < count and 0 <= iy < count:
        out.append(ix * count + iy)
  return out


def _free_table(env, st, entity: str):
  """Flex vertex index -> body id, for every vertex the flexcomp gave a body.

  Pinned vertices have no body and cannot be grasped, so they are excluded
  here rather than filtered at every use.
  """
  if getattr(st, "free_vids", None) is not None:
    return st.free_vids, st.free_bids
  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as _c
  m = env.sim.mj_model
  nv = int(env.sim.data.flexvert_xpos.shape[1])
  vids, bids = [], []
  for v in range(nv):
    bid = mujoco.mj_name2id(
      m, mujoco.mjtObj.mjOBJ_BODY, f"{entity}/tissue_{_c._vertex_body_index(v)}")
    if bid >= 0:
      vids.append(v)
      bids.append(bid)
  st.free_vids = torch.tensor(vids, dtype=torch.long, device=env.device)
  st.free_bids = torch.tensor(bids, dtype=torch.long, device=env.device)
  return st.free_vids, st.free_bids


def _eq_table(env, st, entity: str):
  """Equality id for each free vertex, aligned with `_free_table`'s ordering.

  Why a BANK of equalities and not one that gets retargeted: mujoco_warp lays
  eq_obj2id out as a (neq,) field SHARED by every world (verified on the built
  env: eq_data and eq_solref are stride-0 broadcasts too, and only eq_active is
  a real (nworld, neq) field). Retargeting one equality's obj2 therefore points
  all 1024 training envs at whichever vertex the first env to grasp happened to
  pinch -- every other env then drags a vertex its jaw is nowhere near, and
  nothing downstream would say so. This is the limitation 982f446 flagged as
  "WRONG for batched training".

  Selection through eq_active is the way out, so the scene compiles one
  inactive equality per graspable vertex and each env switches on its own.
  """
  if getattr(st, "eq_ids", None) is not None:
    return st.eq_ids
  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as _c
  m = env.sim.mj_model
  fv, _fb = _free_table(env, st, entity)
  ids = []
  for v in fv.tolist():
    eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, _c.grasp_eq_name(v))
    if eid < 0:
      raise ValueError(
        f"no equality named {_c.grasp_eq_name(v)!r}. TISSUE_GRASP_NEAREST=1 "
        "needs the per-vertex equality bank, which add_grasp_equality only "
        "emits when GRASP_NEAREST is on at SCENE BUILD time -- so the flag was "
        "flipped after the env was constructed.")
    ids.append(eid)
  st.eq_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
  return st.eq_ids


def _state(env: ManagerBasedRlEnv, vertex_ids, entity: str) -> _GraspState:
  st = getattr(env, "_tissue_grasp", None)
  if st is None:
    st = _GraspState(env, list(vertex_ids), entity)
    env._tissue_grasp = st
  return st


def reset_grasp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vertex_ids,
  entity: str = "tissue",
) -> None:
  """Clear the grasp and redraw this episode's success probability."""
  st = _state(env, vertex_ids, entity)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  st.held[env_ids] = False
  st.was_closed[env_ids] = False
  st.attempts[env_ids] = 0
  lo, hi = GRASP_P_RANGE
  st.p[env_ids] = torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo


def update_grasp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  vertex_ids,
  entity: str = "tissue",
) -> None:
  """Every step: resolve grasp attempts, then drive the held vertex.

  Runs as a ``mode="step"`` event, which fires after the physics step, so the
  vertex is repositioned and the sheet responds on the following step.
  """
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b
  from surgical_lab.tasks.retract.mdp.cloth import flex_vertex_pos_w

  st = _state(env, vertex_ids, entity)
  tip = a3b._tool_pos(env)
  closed = env.sim.data.qpos[:, st.jaw_qadr] < JAW_CLOSED_BELOW

  # An attempt is the RISING EDGE of closing, not the closed state: holding the
  # jaw shut next to the tissue must not re-roll the dice every step, or a
  # policy that simply parks there would grasp with probability ~1 and the
  # failure rate would teach nothing.
  if GRASP_NEAREST:
    from surgical_lab.tasks.retract.mdp.cloth import flex_vertex_pos_all
    fv, _fb = _free_table(env, st, entity)
    cand = flex_vertex_pos_all(env)[:, fv]            # (n, k, 3)
    dist = torch.norm(cand - tip.unsqueeze(1), dim=-1)
    dmin, st.sel = dist.min(dim=1)                    # nearest free vertex
    vert = cand[torch.arange(cand.shape[0], device=cand.device), st.sel]
    near = dmin < GRASP_RADIUS
  else:
    vert = flex_vertex_pos_w(env, vertex_ids[0])
    near = torch.norm(tip - vert, dim=-1) < GRASP_RADIUS
  attempt = closed & (~st.was_closed) & near & (~st.held)
  won = torch.rand(env.num_envs, device=env.device) < st.p
  st.attempts += attempt.int()
  took = attempt & won
  if bool(took.any()):
    tidx = took.nonzero(as_tuple=True)[0]
    st.anchor[took] = tip[took]
    for pi, qadr in enumerate(st.qadr):
      for k in range(3):
        st.q0[tidx, pi, k] = env.sim.data.qpos[tidx, qadr[k]]
  st.held |= took
  st.held &= closed          # opening the jaw always releases
  st.was_closed = closed

  # Taut, then slipped.
  pull = torch.norm(tip - st.anchor, dim=-1)
  st.held &= pull < SLIP_DRAG

  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as _cfg
  if _cfg.GRASP_MODE == "connect":
    # Hand the grasp to the SOLVER. Toggling eq_active makes it a real
    # constraint force, so the load spreads through the mesh instead of nine
    # vertices being teleported, and the corner is held AT the jaw rather than
    # up to GRASP_RADIUS away from it.
    if not hasattr(env, "_tissue_eq_id"):
      if GRASP_NEAREST:
        # One inactive equality per graspable vertex; the env selects its own
        # by switching it on. See _eq_table for why retargeting one equality
        # cannot work across worlds.
        env._tissue_eq_ids = _eq_table(env, st, entity)
        env._tissue_eq_id = int(env._tissue_eq_ids[0])
        st.eq_sel = env._tissue_eq_ids[0].repeat(env.num_envs).clone()
      else:
        env._tissue_eq_ids = None
        env._tissue_eq_id = mujoco.mj_name2id(
          env.sim.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, _cfg.GRASP_EQ_NAME)
      env._tissue_eq_b1 = int(env.sim.mj_model.eq_obj1id[env._tissue_eq_id])
    eid, eids = env._tissue_eq_id, env._tissue_eq_ids
    if eid >= 0:
      if bool(took.any()):
        ti = took.nonzero(as_tuple=True)[0]
        if GRASP_NEAREST:
          # Point the CONSTRAINT at the vertex actually being pinched. The
          # equalities are compiled one per vertex, so this is a choice of
          # WHICH equality this env activates, not a write to a shared field.
          # Without it the trigger fires on an edge vertex and the solver still
          # drags the corner -- the original "tissue hangs off the jaw"
          # complaint, with extra steps.
          st.eq_sel[ti] = eids[st.sel[ti]]
        # mjEQ_CONNECT locks the relative pose of the REFERENCE configuration,
        # where the jaw sits ~48 mm above the sheet. Activating it as-compiled
        # would try to hold the vertex that far from the jaw. So re-anchor at
        # the instant of the grasp: express the vertex in body1's frame and
        # write it into eq_data, which makes the constraint exactly satisfied
        # at take-time and carries the corner with the jaw thereafter.
        b1 = env._tissue_eq_b1
        bp = env.sim.data.xpos[:, b1]
        bm = env.sim.data.xmat[:, b1].reshape(-1, 3, 3)
        mode = os.environ.get("TISSUE_EQ_ANCHOR", "tip")
        if mode == "vertex":
          # Where the vertex WAS when grabbed. Zero residual at activation, but
          # it then preserves the GRASP_RADIUS slop for the whole lift -- the
          # jaw ends up holding the sheet a measured 9.3 mm away from itself.
          #
          # Refused when batched: this is the ONE anchor that differs per env,
          # and eq_data is a stride-0 broadcast shared by every world, so the
          # last env to grasp would silently set the anchor for all of them.
          if env.num_envs > 1:
            raise ValueError(
              "TISSUE_EQ_ANCHOR=vertex needs a per-env eq_data, but mujoco_warp "
              "shares eq_data across worlds. Single-env replay only.")
          rel = torch.einsum("nij,nj->ni", bm.transpose(1, 2), vert - bp)
        else:
          # The JAW TIP. A real grasp brings the tissue TO the jaw, so the
          # constraint should too. Costs a small activation residual -- the
          # 5-10 mm the corner was away -- against the 48 mm that caused the
          # original flare, so it is a nudge rather than a launch. Constant
          # across envs, which is why the shared eq_data is harmless here.
          rel = torch.as_tensor(_cfg.GRASP_EQ_ANCHOR, device=vert.device,
                                dtype=vert.dtype).expand_as(vert)
        col = st.eq_sel[ti] if GRASP_NEAREST else eid
        env.sim.model.eq_data[ti, col, 0:3] = rel[ti]
        env.sim.model.eq_data[ti, col, 3:6] = 0.0
        if os.environ.get("TISSUE_GRASP_DEBUG") == "1":
          a1 = bp + torch.einsum("nij,nj->ni", bm, rel)
          print(f"[grasp] take: vertex {vert[ti][0].tolist()}  "
                f"anchor1_world {a1[ti][0].tolist()}  "
                f"residual {(a1 - vert)[ti][0].norm().item()*1000:.3f} mm",
                flush=True)
      if not hasattr(env, "_tissue_eq_solref_set"):
        tc, dr = _cfg.GRASP_EQ_SOLREF
        col = eids if GRASP_NEAREST else eid
        env.sim.model.eq_solref[:, col, 0] = tc
        env.sim.model.eq_solref[:, col, 1] = dr
        env._tissue_eq_solref_set = True
      if GRASP_NEAREST:
        # Exactly one equality live per env: clear the whole bank, then switch
        # on the one this env is holding with. A full (n, 64) bool write, which
        # is nothing next to a physics step.
        env.sim.data.eq_active[:, eids] = False
        env.sim.data.eq_active[
          torch.arange(env.num_envs, device=env.device), st.eq_sel] = st.held
      else:
        env.sim.data.eq_active[:, eid] = st.held
    return

  if bool(st.held.any()):
    idx = st.held.nonzero(as_tuple=True)[0]
    # every vertex in the patch shares the same displacement, so qpos (which is
    # displacement from that vertex's own spawn) is the same value for all of
    # them: the patch translates rigidly with the tool.
    disp = (tip - st.anchor).to(torch.float32)
    # clamp the magnitude, keeping direction: the tissue follows the tool until
    # it is taut, then stays put while the tool keeps going.
    mag = torch.norm(disp, dim=-1, keepdim=True).clamp_min(1e-9)
    # slack first, then the taut limit
    eff = (mag - GRASP_SLACK).clamp_min(0.0).clamp_max(MAX_DRAG)
    disp = disp * (eff / mag)
    for pi, (qadr, dadr) in enumerate(zip(st.qadr, st.dadr)):
      for k in range(3):
        # qvel MUST be zeroed too. Without it the stretched membrane's
        # restoring force flings the vertex: a commanded +30 mm lands at
        # -3.3 mm, and the sheet simply refuses to move.
        env.sim.data.qpos[idx, qadr[k]] = st.q0[idx, pi, k] + disp[idx, k]
        env.sim.data.qvel[idx, dadr[k]] = 0.0


def is_held(env: ManagerBasedRlEnv) -> torch.Tensor:
  st = getattr(env, "_tissue_grasp", None)
  if st is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return st.held


def grasp_attempts(env: ManagerBasedRlEnv) -> torch.Tensor:
  st = getattr(env, "_tissue_grasp", None)
  if st is None:
    return torch.zeros(env.num_envs, device=env.device)
  return st.attempts.float()
