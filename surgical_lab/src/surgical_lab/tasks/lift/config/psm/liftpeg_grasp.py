"""Contact-based grasp detection and small helpers for liftpeg.

Unity gates success, `SparseGrasp` and `SparseLift` on `_grasped`, which comes
from a trigger-volume `GraspDetector` on the block. MuJoCo has no trigger
volumes, so we use real contact: BOTH jaw pads must be touching the block.
That is stricter than Unity's volume test and physically meaningful, which is
what we want for something that has to transfer to hardware.
"""

from __future__ import annotations

import os

import torch


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate v by quaternion q (w, x, y, z), batched."""
  w = q[:, 0:1]
  u = q[:, 1:4]
  return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


_JAW_GEOM_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _geom_sets(env):
  """Geom-id sets for (left jaw, right jaw) and the block, cached per model."""
  key = id(env.sim.mj_model)
  if key in _JAW_GEOM_CACHE:
    return _JAW_GEOM_CACHE[key]
  import mujoco

  m = env.sim.mj_model
  left, right, block = [], [], []
  for i in range(m.ngeom):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    if n.startswith("block/") and "bar" in n:
      block.append(i)
    elif "_collision" in n and ("/lg" in n or n.startswith("lg")):
      left.append(i)
    elif "_collision" in n and ("/rg" in n or n.startswith("rg")):
      right.append(i)
  out = (
    torch.tensor(left, dtype=torch.long),
    torch.tensor(right, dtype=torch.long),
    torch.tensor(block, dtype=torch.long),
  )
  _JAW_GEOM_CACHE[key] = out
  return out


# Hysteresis on the grasp signal, in control steps. DEFAULT 0 -- MEASURED USELESS.
#
# Kept because the negative result is worth more than the code: it rules out a
# whole class of explanation. Measured on the sector model with a scripted lift,
# grasp losses are NOT transient. Every loss episode lasted the full 120-step
# lift phase (median 120, none under 8 steps), and only 1 of 5 had any pad still
# touching. 11 of 16 envs held continuously; 5 never grasped at all. The failure
# is binary per env -- acquisition, not retention -- so bridging brief slips
# bridges nothing. Enabling it changed P(grasped | during lift) 0.730 -> 0.731.
#
# The reasoning that motivated it, retained because the ASYMMETRY IS REAL even
# though it is not what stalls training:
#
# The staged potential pays 1.0 + lift_frac while `grasped`, and reach (<= 0.5)
# otherwise. So one dropped step costs ~0.5 against the +0.016 a successful
# 0.5 mm lift step earns -- a 32:1 asymmetry. Measured with the sector
# collision model, a quarter of lift attempts momentarily break contact
# (P(grasped | during lift) = 0.730), which makes ATTEMPTING TO LIFT negative
# expected value and freezing on a secure grasp optimal. That is exactly what
# the 2026-08-20 run did: grasp_rate 1.000 with lift_mm flat at 0.2-1.1 mm for
# 900 iterations while the action std ran to 2.55.
#
# Holding the signal for a few steps after contact drops removes the penalty
# for a momentary slip without changing what a grasp means. The latch only
# survives while at least ONE pad is still touching, so a policy cannot grab,
# leave, and keep collecting -- letting go entirely clears it immediately.
GRASP_HOLD_STEPS = int(os.environ.get("LIFTPEG_GRASP_HOLD", "0"))


def contact_both_jaws(env) -> torch.Tensor:
  """BOTH jaw pads touching the block, held for GRASP_HOLD_STEPS after a slip.

  Keyed on the step counter so repeat calls within one step are idempotent:
  there are two call sites (the cached `terms()` and the `grasp_rate` metric),
  and a naive counter would advance twice per step.
  """
  if GRASP_HOLD_STEPS <= 0:
    return _contact_both_jaws_raw(env)[0]
  key = int(getattr(env, "common_step_counter", -1))
  got = getattr(env, "_liftpeg_grasp_latch", None)
  if got is not None and got[0] == key:
    return got[1]
  raw, any_pad = _contact_both_jaws_raw(env)
  st = getattr(env, "_liftpeg_grasp_hold", None)
  if st is None or st.shape[0] != env.num_envs:
    st = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    env._liftpeg_grasp_hold = st
  # refresh on a real grasp; otherwise decay, but only while a pad still
  # touches -- a clean release ends the latch at once.
  st.copy_(
    torch.where(
      raw,
      torch.full_like(st, GRASP_HOLD_STEPS),
      torch.where(any_pad, (st - 1).clamp(min=0), torch.zeros_like(st)),
    )
  )
  out = raw | (st > 0)
  env._liftpeg_grasp_latch = (key, out)
  return out


def reset_grasp_latch(env, env_ids=None) -> None:
  """Clear the hysteresis state. Called from the reset event."""
  st = getattr(env, "_liftpeg_grasp_hold", None)
  if st is not None:
    st[slice(None) if env_ids is None else env_ids] = 0
  env._liftpeg_grasp_latch = None


def _contact_both_jaws_raw(env):
  """(both pads touching, at least one pad touching).

  mujoco_warp's contact layout differs from MuJoCo's: the active count is
  `data.nacon` (not `ncon`), pairs live in `contact.geom` as (N, 2) rather than
  separate geom1/geom2, and `contact.worldid` maps each contact to its env.
  """
  left, right, block = _geom_sets(env)
  n = env.num_envs
  out = torch.zeros(n, dtype=torch.bool, device=env.device)
  if len(left) == 0 or len(right) == 0 or len(block) == 0:
    return out, out.clone()

  d = env.sim.data
  n_act = int(d.nacon[0])
  if n_act == 0:
    return out, out.clone()

  c = d.contact
  g = c.geom[:n_act]
  wid = c.worldid[:n_act].long()
  # dist < 0 is a penetrating (force-carrying) contact; the buffer also holds
  # near-miss candidates, and counting those would report a grasp too early.
  active = c.dist[:n_act] < 0.0
  g1, g2 = g[:, 0].long(), g[:, 1].long()
  lset = left.to(g1.device)
  rset = right.to(g1.device)
  bset = block.to(g1.device)

  def _hits(js):
    return (
      (torch.isin(g1, js) & torch.isin(g2, bset))
      | (torch.isin(g2, js) & torch.isin(g1, bset))
    ) & active

  hl, hr = _hits(lset), _hits(rset)
  wl = torch.zeros(n, dtype=torch.bool, device=env.device)
  wr = torch.zeros(n, dtype=torch.bool, device=env.device)
  wl[wid[hl]] = True
  wr[wid[hr]] = True
  return wl & wr, wl | wr


def _jaw_cmd(env) -> torch.Tensor:
  """Current jaw joint position, standing in for Unity's GetJointCommand."""
  robot = env.scene["robot"]
  if "jaw" not in robot.joint_names:
    return torch.zeros(env.num_envs, device=env.device)
  return robot.data.joint_pos[:, robot.joint_names.index("jaw")]
