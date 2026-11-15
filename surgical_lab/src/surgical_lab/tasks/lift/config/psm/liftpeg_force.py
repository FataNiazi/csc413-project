"""Penalise the force liftpeg delivers into everything that is NOT the ring.

Why
---
The ideal episode descends without touching the yellow stand, closes on the
ring, unthreads it, carries it over the goal and SETS IT DOWN rather than
dropping it. Nothing in the shaped potential (`liftpeg_shaped.shaped_potential`)
distinguishes any of that from barging through the post and letting go from
30 mm up: the potential only knows where the ring is. On hardware that is the
difference between a run and a bent instrument -- the deployed wrist policy
nearly does the task and then collides with the peg SHAFT.

Force on the RING itself is deliberately not penalised. Gripping it firmly is
the point of the task, and taxing that would teach the policy to hold it
loosely, which is the opposite of what we want.

What is measured
----------------
Two components, both in newtons:

  robot_surf  contact normal force between a ROBOT geom and one of the
              penalised surfaces (floor, peg post, peg plate, goal square).
              This is the direct collision -- the peg-shaft strike -- and it is
              zero in a clean episode.

  press       contact normal force between the RING and those same surfaces,
              MINUS the ring's own weight, clamped at zero. Pushing down on a
              threaded ring transmits into the plate; pushing down on a resting
              ring transmits into the floor or the goal; and dropping it from
              height arrives as an impact spike. None of those show up as a
              robot-surface contact, so penalising only the direct component
              would miss exactly the "place it gently" behaviour we asked for.
              Subtracting m*g is what makes a ring that is merely SITTING there
              -- the start state of every episode, and the desired end state --
              cost nothing.

The allowance is the ring's LIVE per-world mass, not a constant: `block_mass`
DR scales it by 0.7-1.4x (liftpeg_a3b_env_cfg._add_extended_dr), so a fixed
allowance would charge the heavy draws a standing penalty for doing nothing and
give the light draws free press.

Measured on 16 envs (scripts probe, 2026-09-08):
  * ring resting on the plate, no action .... block->surface 0.80-1.39 N,
    which is m*g for the DR'd mass range; robot->surface 0.000 N.
  * tool driven -z into the stand for 200 steps ... robot->surface 8.5 N mean,
    25.3 N peak, dominated by tool_pitch_capsule / rg* against `peg/peg_post`.
So the signal is ~an order of magnitude above the resting baseline, and the
geom pairs it fires on are literally the shaft strike we are trying to remove.

How the force is read
---------------------
mujoco_warp DOES expose a real per-contact force -- no proxy needed.
`d.contact.efc_address` is (naconmax, nmaxpyramid) and indexes `d.efc.force`,
which is (nworld, njmax). The model runs an ELLIPTIC friction cone
(`mj_model.opt.cone == 1`, verified at build time below), so column 0 of that
address block is the contact NORMAL force directly and columns 1..n are the
tangential components -- the same decomposition MuJoCo's `mj_contactForce`
does. Under a pyramidal cone the normal force would instead be the SUM over the
pyramid directions, so the cone is asserted rather than assumed.

Off unless `LIFTPEG_FORCE_PENALTY` is a non-zero weight. At zero the reward and
metric terms are not registered at all -- not merely weighted to zero -- so the
term tables of every existing run stay byte-identical.
"""

from __future__ import annotations

import os

import torch

FORCE_PENALTY = float(os.environ.get("LIFTPEG_FORCE_PENALTY", "0"))
"""Weight on the penalty, in reward units per newton-second. 0 = off.

SIZING. mjlab scales every reward by the 20 ms env dt and logs
`Episode_Reward/<term>` as the episodic sum divided by the 40 s episode, so the
units to compare against are episodic sums:

  shaping   telescopes to (phi_final - phi_initial) * dt. `shaped_potential`
            tops out near 13, so a solved episode is worth about 0.26.
  success   2.0 * dt on the terminating step = 0.04.
  ---------------------------------------------------------------
  a solved episode is therefore worth roughly 0.30 in total.

  force     w * dt * sum(F). One second of contact at 10 N is 50 steps, i.e.
            w * 0.02 * 500 = 10w.

At w = 0.002 a one-second 10 N strike costs 0.02 -- about 7% of a solved
episode, noticeable but survivable -- while LEANING on the post at 4 N for the
whole 40 s costs 0.32, which is more than the task is worth and correctly makes
that policy unlearnable. That separation (a brush is cheap, a lean is fatal) is
the property the weight is chosen for, and it is why the number is not simply
"as large as possible".

The failure mode this invites is a policy that touches NOTHING and therefore
never grasps. `force_mean_N` is logged next to `success_rate` and `grasp_rate`
precisely so that shows up as a number rather than as a mystery.
"""

FORCE_PRESS_W = float(os.environ.get("LIFTPEG_FORCE_PRESS", "1.0"))
"""Relative weight on the ring->surface `press` component vs `robot_surf`.

Separate knob because the two have different confidence levels. A robot geom
touching the stand is unambiguously a mistake. Force transmitted THROUGH the
ring is a proxy -- it is what we have instead of "did it place the ring gently",
and it also fires while the jaws squeeze a ring that is still sitting on the
plate, which is part of a legitimate grasp. Set this to 0 to run the direct
collision penalty alone if `press` turns out to block grasp acquisition.
"""

# Multiplier on the ring's own weight that is free. 1.0 charges any press at
# all above static weight; slightly above 1 absorbs solver ripple on the
# resting contact. Measured ripple on a settled ring is under 5% of m*g.
FORCE_ALLOWANCE = float(os.environ.get("LIFTPEG_FORCE_ALLOWANCE", "1.1"))


_PARTITION: dict[int, tuple] = {}


def _partition(env):
  """(surfaces, robot geoms, ring geoms, ring body id), cached per model.

  Surfaces are the things force must NOT be delivered into: the blue floor
  (`terrain`), the yellow stand (`peg/*`) and the red goal square. The wall
  from LIFTPEG_A3_WALL is deliberately excluded -- it is a visual occluder
  standing in for a real barrier, not a fixture the tool is expected to respect.
  """
  key = id(env.sim.mj_model)
  got = _PARTITION.get(key)
  if got is not None:
    return got
  import mujoco

  m = env.sim.mj_model
  assert int(m.opt.cone) == 1, (
    f"friction cone is {int(m.opt.cone)} (pyramidal), not elliptic. Column 0 "
    "of contact.efc_address is only the NORMAL force under an elliptic cone; "
    "under a pyramidal one the normal force is the sum over the pyramid "
    "directions. Fix _normal_force before trusting this reward."
  )
  names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
           for i in range(m.ngeom)]
  dev = env.device
  surf, robot, ring = [], [], []
  # Broken out so the cumulative metrics can say WHICH fixture took the load:
  # the floor, the yellow stand, or the red goal. Diagnostically these are
  # three different stories -- force into the stand is the shaft strike, force
  # into the goal is the set-down, force into the floor is a drop or a scrape.
  floor, stand, goal = [], [], []
  for i, n in enumerate(names):
    if n == "terrain" or n.startswith("peg/") or n == "goal/goal_square":
      surf.append(i)
      (floor if n == "terrain" else stand if n.startswith("peg/") else goal).append(i)
    elif n.startswith("robot/"):
      robot.append(i)
    elif n.startswith("block/"):
      ring.append(i)
  assert surf and robot and ring, (len(surf), len(robot), len(ring))
  bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block/block")
  assert bid >= 0, "no block/block body -- cannot size the weight allowance"
  got = (
    torch.tensor(surf, device=dev),
    torch.tensor(robot, device=dev),
    torch.tensor(ring, device=dev),
    bid,
    {k: torch.tensor(v, device=dev) for k, v in
     (("floor", floor), ("stand", stand), ("goal", goal)) if v},
  )
  _PARTITION[key] = got
  return got


def _ring_weight(env) -> torch.Tensor:
  """Per-env m*g for the ring, read from the LIVE model so mass DR is tracked."""
  _s, _r, _b, bid, _t = _partition(env)
  mass = env.sim.model.body_mass[..., bid]
  g = abs(float(env.sim.mj_model.opt.gravity[2]))
  if mass.ndim == 0:
    mass = mass.expand(env.num_envs)
  return mass.reshape(-1)[: env.num_envs] * g


def forces(env) -> dict[str, torch.Tensor]:
  """Per-env unwanted contact force, in newtons. Cached per step.

  Cached for the same reason `LR.terms` is: this is read by one reward term and
  two metric terms, and the work is a handful of launch-bound kernels whose
  cost is independent of the batch size.

  `d.nacon[0]` -- NOT `ncon` -- is the active contact count, and it reads 0 on
  the first physics step after a reset (measured: step 0 only, at a 2 ms
  timestep with decimation 10, but treat it as "the first few"). Returning
  zeros there is correct rather than merely safe: at that instant the arm is at
  its home pose, nowhere near the stand, and the only real force in the scene
  is the ring's own weight, which the allowance subtracts anyway.
  """
  key = int(getattr(env, "common_step_counter", -1))
  got = getattr(env, "_liftpeg_force_cache", None)
  if got is not None and got[0] == key:
    return got[1]
  out = _forces_uncached(env)
  env._liftpeg_force_cache = (key, out)
  return out


def _forces_uncached(env) -> dict[str, torch.Tensor]:
  surf, robot, ring, _bid, targets = _partition(env)
  n_env, dev = env.num_envs, env.device
  zeros = torch.zeros(n_env, device=dev)
  d = env.sim.data
  n = int(d.nacon[0])
  if n == 0:
    return {"robot_surf": zeros, "press": zeros.clone(),
            "total": zeros.clone(), "raw_ring_surf": zeros.clone()}

  adr = d.contact.efc_address[:n]                     # (n, nmaxpyramid)
  wid = d.contact.worldid[:n].long()
  # Elliptic cone: column 0 IS the normal force. -1 means "not in efc".
  a0 = adr[:, 0]
  ok = a0 >= 0
  fn = torch.gather(
    d.efc.force[wid], 1, a0.clamp_min(0).long().unsqueeze(-1)
  ).squeeze(-1)
  fn = (fn * ok).abs()

  g = d.contact.geom[:n]
  g1, g2 = g[:, 0].long(), g[:, 1].long()
  on_surf = torch.isin(g1, surf) | torch.isin(g2, surf)
  by_robot = torch.isin(g1, robot) | torch.isin(g2, robot)
  by_ring = torch.isin(g1, ring) | torch.isin(g2, ring)

  def _sum(mask):
    o = torch.zeros(n_env, device=dev)
    if bool(mask.any()):
      o.index_add_(0, wid[mask], fn[mask])
    return o

  # Per-target RAW force -- everything pressing on that fixture, whoever is
  # doing the pressing. Deliberately NOT weight-adjusted: this answers "how
  # hard is this object being loaded", which is the safety question, while the
  # penalty above answers "how much of that is the policy's fault".
  per_target = {}
  for name, ids in targets.items():
    hit = (torch.isin(g1, ids) | torch.isin(g2, ids)) & (by_robot | by_ring)
    per_target[name] = _sum(hit)

  f_robot = _sum(on_surf & by_robot)
  f_ring = _sum(on_surf & by_ring)
  press = (f_ring - FORCE_ALLOWANCE * _ring_weight(env)).clamp_min(0.0)
  out = {"robot_surf": f_robot, "press": press,
         "total": f_robot + FORCE_PRESS_W * press, "raw_ring_surf": f_ring}
  out.update({f"on_{k}": v for k, v in per_target.items()})
  return out


class _Impulse:
  """Per-env running sum of force * dt, in newton-seconds.

  The metrics manager offers mean/last/max but no `sum`, so a genuine episode
  TOTAL has to be accumulated here and read out with reduce="last" -- which,
  at the terminal step, is exactly the episode's cumulative impulse.

  Impulse rather than a bare sum of newtons because the sum's units depend on
  the control rate: change decimation and a plain sum changes with it, while
  N*s is the physical quantity and stays comparable across runs.
  """

  def __init__(self, env):
    n, dev = env.num_envs, env.device
    self.acc = {k: torch.zeros(n, device=dev) for k in ("floor", "stand", "goal")}
    self.key = -1

  def update(self, env):
    k = int(getattr(env, "common_step_counter", -1))
    if k == self.key:            # idempotent: three metric terms read this
      return self.acc
    self.key = k
    f = forces(env)
    dt = float(env.step_dt)
    for name, a in self.acc.items():
      a.add_(f.get(f"on_{name}", torch.zeros_like(a)) * dt)
    return self.acc


def impulse(env) -> dict:
  got = getattr(env, "_liftpeg_impulse", None)
  if got is None or got.acc["floor"].shape[0] != env.num_envs:
    got = _Impulse(env)
    env._liftpeg_impulse = got
  return got.update(env)


def reset_impulse(env, env_ids=None) -> None:
  """Zero the accumulators for the resetting envs, at a known point."""
  got = getattr(env, "_liftpeg_impulse", None)
  if got is None:
    return
  sl = slice(None) if env_ids is None else env_ids
  for a in got.acc.values():
    a[sl] = 0.0


def m_impulse_floor(env) -> torch.Tensor:
  return impulse(env)["floor"]


def m_impulse_stand(env) -> torch.Tensor:
  return impulse(env)["stand"]


def m_impulse_goal(env) -> torch.Tensor:
  return impulse(env)["goal"]


def force_penalty(env) -> torch.Tensor:
  """Negative unwanted force, in newtons. Multiply by LIFTPEG_FORCE_PENALTY.

  Left in raw newtons so the weight is readable as "reward units per newton
  per second" -- mjlab scales every reward by the 20 ms env dt, so a policy
  leaning on the post at 10 N for a whole 40 s episode pays
  10 * 40 * weight.
  """
  return -forces(env)["total"]


def m_force_mean_n(env) -> torch.Tensor:
  """Unwanted force, N. Logged with reduce='mean' -> the episode average.

  This is the metric that catches the failure mode the penalty invites: a
  policy that refuses to touch anything, never grasps and scores a beautiful
  0.00 N. Read it NEXT TO success_rate and grasp_rate, never alone.
  """
  return forces(env)["total"]


def m_force_peak_n(env) -> torch.Tensor:
  """Unwanted force, N, with reduce='max' -> the worst instant of the episode.

  Mean and peak say different things: a single 25 N shaft strike barely moves
  the mean over 2000 steps but is the event that breaks an instrument.
  """
  return forces(env)["total"]


def m_force_robot_n(env) -> torch.Tensor:
  """Just the DIRECT robot-into-fixture component, N, reduce='max'.

  Separated from `press` because it is the one with a real-world referent: it
  is the peg-shaft collision, and it should go to zero even if `press` does not.
  """
  return forces(env)["robot_surf"]
