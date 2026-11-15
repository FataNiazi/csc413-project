"""Reward variants for liftpeg, plus the shared task quantities they read.

v7 (the faithful Unity port) trained for 6000 iterations and never succeeded
once. The tensorboard trace says why, unambiguously:

    Episode_Reward/unity_reward  -0.0005      <- the shaping signal is ZERO
    Episode_Metrics/grasp_rate    1.0000      <- stages 1-2 are solved
    Episode_Metrics/lift_mm       8.73        <- needs 30.0 to clear the post
    Policy/mean_std             157.13        <- the policy is pure noise

Two compounding causes:

1.  **The potential saturates.** Unity's `SparseLift` is a step function that
    fires once the ring rises 2.5 mm and then stays at 1.0. Clearing the post
    needs 30 mm. Nothing whatsoever pays for the remaining 27.5 mm, so the
    optimal policy under that reward is to jiggle the ring 3 mm and stop --
    which is exactly the 8.7 mm plateau we measured.
2.  **Nothing outweighs the entropy bonus.** With a telescoping potential whose
    episodic sum is ~0 and `entropy_coef=0.02`, the only term with a gradient
    is entropy, so the action std ran away to 157 against an action clip of
    1.0. The policy did not fail to learn the task; it was never given a reason
    to prefer any action over any other.

Every variant here therefore has to satisfy two hard requirements: the reward
must keep increasing over the *whole* 30 mm of lift, and its episodic magnitude
must be well clear of the entropy term.

The literature the variants are drawn from:

*   **robosuite** (`environments/manipulation/lift.py`, `pick_place.py`) builds
    staged rewards out of *nested increasing ceilings* -- reaching is capped
    below the grasp bonus, which is capped below the lift bonus -- so the agent
    can never top out inside a stage. It also uses `1 - tanh(10 d)` rather than
    an exponential. At our 67 mm reach distance Unity's `exp(-80 d)` gives
    0.0057 while `1 - tanh(10 d)` gives 0.415: ~70x more signal.
*   **DrS** (arXiv 2404.16779) learns per-stage dense rewards with explicit
    stage indicators, the argument being that one monolithic distance term
    cannot express a multi-stage manipulation task.
*   **Dense2Sparse** (arXiv 2003.02740) anneals a dense reward towards the
    sparse task reward so the final policy is not fitted to the shaping.
"""

from __future__ import annotations

import os
import torch

from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS
from surgical_lab.tasks.lift.config.psm.liftpeg_grasp import (
  contact_both_jaws,
  reset_grasp_latch,
  quat_apply,
  _jaw_cmd,
)
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

# --- reward scale ---------------------------------------------------------

# mjlab multiplies every reward by the env step dt. While liftpeg ran at a 2 ms
# timestep (the mistaken tunnelling "fix") that silently divided the whole
# objective by 2.5 against pushblock's 5 ms, with `entropy_coef` unchanged at
# 0.02, so this carried a 2.5x compensation. The real cause of the ring passing
# through the post turned out to be contact softness, not the timestep, and
# with the timestep back at pushblock's 5 ms the compensation is no longer
# needed -- dt matches the recipe that worked, so the weight is 1.0 again.
#
#   pushblock (0.9939 success): Episode_Reward/a3b_shaping 0.0012, success 0.0010
#   liftpeg v7 (0.0000):        Episode_Reward/unity_reward -0.0005, success 0.0
REWARD_SCALE = 1.0


# --- task geometry --------------------------------------------------------

# The post spans 6..36 mm and the ring's hole runs the full height of its
# 15 mm-tall body from the body origin UP, so the hole is free of the post only
# once the ORIGIN -- which is the ring's BOTTOM -- passes the post top. 2 mm of
# margin on that, i.e. 38 mm, which is 32 mm of lift from Z_REST.
#
# 55ee4a0 (2026-08-25) changed this to PEG_POST_TOP minus the ring's height
# (23 mm), reasoning that the ring is off the post "once the origin passes
# 21 mm". That is backwards: at origin = 21 mm the ring's TOP is level with the
# post top and the post still fills the entire hole. Subtracting the ring's
# height would be right only if the origin were the ring's top.
#
# The 13 mm error cost the whole vision programme. `lift_frac` saturated at
# 17 mm of lift, and `cleared` latched at 23 mm -- which flips the shaped
# potential to its CARRY stage, so every policy trained after 8/25 was paid to
# pull SIDEWAYS on a ring still threaded 13 mm down the shaft. Measured
# 2026-09-05: the wrist / sbs / front CNN runs all plateaued at lift_mm
# 20.8..22.8 with clear_rate 1.00 and success 0.000 after ~22 h, while
# qdch04ik (8/21, the last run before the change) reached lift_mm 37.1 and
# success 0.83 in 5.7 h. The state teacher lifts to 41.1 mm (median of 612
# episodes, p10 40.4), so 38 mm is a height a working policy comfortably makes.
# RAISED from 2 mm to 6 mm on 2026-09-09, and made configurable, after real
# hardware runs showed the tool dragging the whole peg base sideways. 2 mm of
# margin survives sim but not the real robot's calibration error. 6 mm is
# comfortably reachable: the state teacher clears at 41.1 mm (p10 40.4) and
# arm A peaks past 54 mm, so this costs nothing a working policy has.
CLEAR_MARGIN = float(os.environ.get("LIFTPEG_CLEAR_MARGIN_MM", "2.0")) / 1000.0
CLEAR_Z = LS.PEG_POST_TOP + CLEAR_MARGIN
assert CLEAR_MARGIN > 0.0, "LIFTPEG_CLEAR_MARGIN_MM must be positive"
assert CLEAR_Z > LS.PEG_POST_TOP, (
  "CLEAR_Z must sit ABOVE the post top: the ring's origin is its bottom, so a "
  "clear height at or below PEG_POST_TOP calls a still-threaded ring lifted "
  "and hands the shaping over to its lateral carry stage. See 55ee4a0."
)
_CLEAR_XY = LS.PEG_POST_RADIUS + LS.BLOCK_BAR_RADIAL

# Where the ring settles once dropped onto the plate. Measured, not derived:
# `scripts/probe_liftpeg_dropped.py` reports 5.991 mm across 64 envs with zero
# variance (the plate top is at exactly 6 mm, so this is 9 um of solver
# penetration -- it was 36 um before CONTACT_SOLREF was stiffened, which is the
# same fix that stopped the ring passing through the post). Using the constant
# instead of latching the value per episode keeps every reward term stateless.
Z_REST = 0.005991
LIFT_SPAN = CLEAR_Z - Z_REST               # 32.0 mm of lift to clear

GOAL_HALF = 0.5 * LS.GOAL_SQUARE_SIDE
DOWN_Z = 0.004                             # "set down on the floor"
LOST_MARGIN = 0.015                        # slack before an abandoned ring fails

# PickPoint is the grasp site on the triangle, not its centroid: 6.61 mm out
# and 13.6 mm up from the body origin, i.e. mid-face near the top edge.
PICKPOINT_LOCAL = (0.0, 0.00659, 0.0136)

# robosuite's kernel width. 1 - tanh(10 d) is ~0.63 at 5 cm and ~0.10 at 1 cm,
# so it still has usable slope across the entire 67 mm approach.
# --- "straight up before sideways", the 2026-09-09 hardware failure ---------
# Real runs dragged the entire peg base at the moment of PICKUP: the tool
# closed, lifted, and began translating while the ring was still down the
# shaft, so the ring towed the base. After that the transfer was clean.
#
# LIFT_XY_TOL is the radial slack the ring gets before it counts as off-axis.
# _CLEAR_XY is the geometric "hole is free of the post" radius, so allowing
# about that much keeps the penalty off the normal jitter of a held ring.
LIFT_XY_TOL = float(os.environ.get("LIFTPEG_LIFT_XY_TOL_MM", "6.0")) / 1000.0
# ADDITIVE here, and this arm's stages are 1.0 apart, so it is kept under 0.5:
# a fully penalised lift (1.0 - PEN) still sits above the reach stage's 0.5
# ceiling and the stage ordering cannot invert.
#
# Its own env var, deliberately. liftpeg_shaped's arm E applies the same idea
# MULTIPLICATIVELY over a 6.0-wide lift stage, where any value in [0,1) is
# safe. One shared name would let a value legal for one arm abort the import
# of the other -- which it did.
LIFT_LATERAL_PEN = float(os.environ.get("LIFTPEG_LIFT_LATERAL_PEN_ADD", "0.4"))
assert 0.0 <= LIFT_LATERAL_PEN < 0.5, "LIFTPEG_LIFT_LATERAL_PEN_ADD must be < 0.5"

REACH_K = 10.0
GOAL_K = 5.0                               # transport is a longer move


# --- shared per-step quantities -------------------------------------------


class _Latches:
  """Per-env episode latches that more than one term needs.

  Only `cleared` is genuinely stateful: success has to know whether the ring
  was ever lifted off the post, because a ring that ends up on the floor can
  otherwise be *pushed* into the goal square without ever being transferred.
  Latching is monotone (`|=`), so calling `update` more than once per step is
  harmless and no term needs to care what order the managers run in.
  """

  def __init__(self, env):
    self.cleared = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Reported, never scored. Success is contact with the goal alone; phasing
    # is a physics artifact we want visible in the metrics so it cannot grow
    # unnoticed, not a condition quietly gating the score.
    self.phased = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.departed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Ring height AT the first lateral departure, and the LOWEST height it
    # reaches while carrying. Both seeded to the resting height so an episode
    # that never left the post reads 6 mm ("went nowhere") rather than 0.
    self.depart_z = torch.full((env.num_envs,), Z_REST, device=env.device)
    self.carry_z_min = torch.full((env.num_envs,), Z_REST, device=env.device)
    self.carrying = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def update(self, z: torch.Tensor) -> None:
    self.cleared |= z >= CLEAR_Z

  def update_phase(self, d_post: torch.Tensor, z: torch.Tensor) -> None:
    """First departure from the post axis below the height that clears it."""
    first = (~self.departed) & (d_post > _CLEAR_XY)
    self.phased |= first & (z < CLEAR_Z - 0.002)
    self.depart_z = torch.where(first, z, self.depart_z)
    self.departed |= first

  def update_carry(self, z: torch.Tensor, d_goal_xy: torch.Tensor) -> None:
    """Lowest height reached during the TRANSPORT leg.

    `lift_mm` is max-reduced, so a single frame at 54 mm hides a ring that
    spends the whole carry at 30 mm -- which is exactly the real-robot failure
    on 2026-09-09, and exactly what the sim metrics failed to show. This is the
    complementary number: the floor, not the ceiling.

    Gated on `departed`, NOT on `cleared`. Gating on `cleared` looks right and
    is useless: `cleared` first turns true at exactly z == CLEAR_Z, so that
    instant seeds the minimum and the metric reports CLEAR_Z back to you no
    matter what the ring does afterwards. Measured 2026-09-09 on model_6300 --
    p10 38.0, p50 38.2, p90 38.4 against a CLEAR_Z of 38.0, a spread of 0.4 mm
    across 807 episodes, which is the signature of a constant, not a physical
    quantity. `departed` starts the window when the ring actually leaves the
    post axis, which is where the carry genuinely begins.

    Also restricted to `d_goal_xy > GOAL_HALF` so the deliberate set-down at
    the goal does not read as a drag.
    """
    live = self.departed & (d_goal_xy > GOAL_HALF)
    seed = live & ~self.carrying
    self.carry_z_min = torch.where(seed, z, self.carry_z_min)
    self.carry_z_min = torch.where(
      live & self.carrying, torch.minimum(self.carry_z_min, z), self.carry_z_min)
    self.carrying |= live


def latches(env) -> _Latches:
  got = getattr(env, "_liftpeg_latches", None)
  if got is None:
    got = _Latches(env)
    env._liftpeg_latches = got
  return got


def reset_latches(env, env_ids) -> None:
  """Reset event. Registered explicitly so the clear happens at a known point."""
  lat = latches(env)
  sl = slice(None) if env_ids is None else env_ids
  lat.cleared[sl] = False
  lat.phased[sl] = False
  lat.departed[sl] = False
  lat.depart_z[sl] = Z_REST
  lat.carry_z_min[sl] = Z_REST
  lat.carrying[sl] = False
  # Drop the per-step `terms` cache too: a reset changes the state those
  # tensors describe, and anything reading them afterwards must recompute.
  env._liftpeg_terms_cache = None
  # ...and the grasp hysteresis, or a latch set before the reset would carry a
  # phantom grasp into the new episode.
  reset_grasp_latch(env, env_ids)


# `terms` is called once per reward term, once per metric term and once per
# termination -- 11 times per env step. Each call is a few dozen tiny CUDA
# kernels, and at 1024 envs those are launch-bound, so the cost is FIXED rather
# than proportional to batch size. Measured: manager time 34.6 ms at 1024 envs
# vs 36.1 ms at 4096, i.e. flat, which is the signature of exactly this.
#
# Caching is safe because the only side effect is `lat.update(z)`, a monotone
# OR into the `cleared` latch: idempotent for a fixed z, so once per step and
# eleven times per step agree. The cache is keyed on the step counter and
# dropped by `reset_latches`, so nothing survives a reset.
TERMS_CACHE = os.environ.get("LIFTPEG_TERMS_CACHE", "1") == "1"


def terms(env) -> dict[str, torch.Tensor]:
  """Everything the reward variants are built from, computed once per step."""
  if TERMS_CACHE:
    key = int(getattr(env, "common_step_counter", -1))
    got = getattr(env, "_liftpeg_terms_cache", None)
    if got is not None and got[0] == key:
      return got[1]
  out = _terms_uncached(env)
  if TERMS_CACHE:
    env._liftpeg_terms_cache = (key, out)
  return out


def _terms_uncached(env) -> dict[str, torch.Tensor]:
  robot = env.scene["robot"]
  origins = env.scene.env_origins
  tool = a3b._tool_pos(env)

  blk = env.scene["block"]
  q = blk.data.root_link_quat_w
  block = blk.data.root_link_pos_w - origins
  off = torch.tensor(PICKPOINT_LOCAL, device=env.device).expand(q.shape[0], 3)
  pick = blk.data.root_link_pos_w + quat_apply(q, off) - origins

  goal = env.scene["goal"].data.root_link_pos_w - origins
  z = block[:, 2]

  lat = latches(env)
  lat.update(z)
  _peg_xy = (env.scene["peg"].data.root_link_pos_w - env.scene.env_origins)[:, :2]
  _d_post = torch.norm(block[:, :2] - _peg_xy, dim=-1)
  lat.update_phase(_d_post, z)
  _d_goal_xy = torch.norm(block[:, :2] - goal[:, :2], dim=-1)
  lat.update_carry(z, _d_goal_xy)

  return {
    "tool": tool,
    "block": block,
    "pick": pick,
    "goal": goal,
    "z": z,
    "d_reach": torch.norm(pick - tool, dim=-1),
    "d_goal_xy": _d_goal_xy,
    "lift_frac": ((z - Z_REST) / LIFT_SPAN).clamp(0.0, 1.0),
    # How far ABOVE the post top the ring is right now, as a 0..1 fraction of
    # the clearance margin. Unlike `cleared` this is instantaneous, not a
    # latch, which is the whole point -- see staged_potential.
    "clear_h": ((z - LS.PEG_POST_TOP) / CLEAR_MARGIN).clamp(0.0, 1.0),
    # Radial distance of the ring from the post AXIS. Off-axis while still low
    # is the ring dragging the whole peg base -- see staged_potential.
    "d_post": _d_post,
    "off_axis": (_d_post / LIFT_XY_TOL).clamp(0.0, 1.0),
    "grasped": contact_both_jaws(env),
    "cleared": lat.cleared.clone(),
    "phased": lat.phased.clone(),
    "jaw": _jaw_cmd(env),
  }


# --- success / failure ----------------------------------------------------


def _goal_overlap_xy(t) -> torch.Tensor:
  """Does the ring's footprint TOUCH the goal square, in plan view?

  Distance from the ring centre to the square treated as a box, compared
  against the ring's outer radius. The old test asked whether the ring's
  CENTRE was inside the square, which is much stricter than it sounds: the
  ring is 22.9 mm across and the square is 20.6 mm, so a ring resting visibly
  on the square's edge -- unambiguously delivered to a human watching -- could
  be 20 mm off centre and score zero. Touching allows 21.73 mm of centre
  offset against the old 10.30 mm.
  """
  d = (t["block"][:, :2] - t["goal"][:, :2]).abs() - GOAL_HALF
  return torch.norm(d.clamp_min(0.0), dim=-1) <= LS.outer_radius()


_GOAL_CONTACT_IDX: dict = {}


def _touching_goal(env) -> torch.Tensor:
  """Is the ring in solver CONTACT with the goal square?

  The goal geom is a real collider now, so this is the literal test rather
  than a proxy for it. That removes the height threshold entirely, and with it
  a genuine scoring bug: the goal sits only 4.7 mm from the 6 mm peg plate
  while the ring is 22.9 mm across, so a ring delivered to the goal could rest
  on the plate at 7.2 mm, pass the XY test, and fail a `z <= 4 mm` test
  forever. Contact cannot disagree with itself that way.
  """
  d = env.sim.data
  idx = _GOAL_CONTACT_IDX.get(id(env))
  if idx is None:
    m = env.sim.mj_model
    names = [m.geom(i).name for i in range(m.ngeom)]
    blocks = torch.tensor([i for i, n in enumerate(names) if "block_bar" in n],
                          device=env.device)
    goal = torch.tensor([i for i, n in enumerate(names) if "goal_square" in n],
                        device=env.device)
    assert len(blocks) and len(goal), (len(blocks), len(goal))
    idx = (blocks, goal)
    _GOAL_CONTACT_IDX[id(env)] = idx
  blocks, goal = idx
  out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  n = int(d.nacon[0])
  if n == 0:
    return out
  g = d.contact.geom[:n]
  g1, g2 = g[:, 0].long(), g[:, 1].long()
  hit = (torch.isin(g1, blocks) & torch.isin(g2, goal)) | (
        torch.isin(g2, blocks) & torch.isin(g1, goal))
  if bool(hit.any()):
    out[d.contact.worldid[:n].long()[hit]] = True
  return out


def block_at_goal(env) -> torch.Tensor:
  """Success: the ring is TOUCHING the goal square. That is the whole test.

  One physical fact, read from the solver. No height thresholds, no XY proxy,
  and no `cleared` latch -- each of those was an extra condition that could be,
  and was, wrong:

    * `z <= DOWN_Z` (4 mm). The goal sits 4.7 mm from the 6 mm peg plate and
      the ring is 22.9 mm across, so a ring delivered to the goal could rest on
      the plate at 7.2 mm and fail forever while looking delivered.
    * `cleared`, a latch on the ring rising above CLEAR_Z = 38 mm. But CLEAR_Z
      was compared against the ring's ORIGIN, which is its BOTTOM, and the ring
      is 15 mm tall -- it is off the post once the origin passes 21 mm. That
      left a 17 mm dead band where a ring was genuinely lifted, carried and
      placed on the goal, and success was already unreachable. Measured: an
      episode that delivered the ring after peaking at 37.5 mm, half a
      millimetre under the latch, then sat on the target for 1500 steps
      scoring nothing.

  Dropping `cleared` was the deliberate call. Its job was to stop the ring
  being bulldozed along the floor into the square rather than lifted -- but the
  ring starts threaded on a 30 mm post and the goal is 45 mm away, so reaching
  the goal at all already implies it came off the post. The only thing the
  latch uniquely excluded was the ring passing THROUGH the shaft, which is
  0.2% of episodes. That is now reported as a metric instead of silently gating
  the score: a physics artifact we want to see and fix, not score around.
  """
  return _touching_goal(env)


def block_lost(env) -> torch.Tensor:
  """Ring unthreaded and then abandoned -- the episode cannot be recovered.

  Replaces Unity's `IsBlockBelowGround`. That test (`z < 0`) was written for a
  finite Unity platform the block could fall off; our floor is an infinite
  plane, so it only ever fired on solver penetration, which measured -0.036 mm
  on the plate. It therefore terminated episodes essentially at random while
  missing the failure that actually happens: the ring comes off the post, lands
  on the floor away from the goal, and the remaining ~4000 steps are dead.
  """
  t = terms(env)
  on_floor = t["z"] <= DOWN_Z
  # Measured from the same overlap test success uses. With success widened to
  # contact, a fixed centre-distance threshold here would declare a ring that
  # IS touching the square "abandoned" -- the two tests have to move together.
  far = (~_goal_overlap_xy(t)) & (t["d_goal_xy"] > GOAL_HALF + LOST_MARGIN)
  return (t["cleared"] & on_floor & far) | (t["z"] < -0.002)


def success_bonus(env) -> torch.Tensor:
  return block_at_goal(env).float() * 2.0


def lost_penalty(env) -> torch.Tensor:
  return block_lost(env).float() * -1.0


# --- metrics --------------------------------------------------------------


class EpisodeMin:
  """Running per-episode minimum, for use with ``reduce="last"``.

  `MetricsTermCfg.reduce` only offers mean/last/max -- there is no "min", and
  passing one silently falls through to mean, which is how `reach_mm` and
  `goal_mm` came to report episode averages instead of closest approach for the
  whole of v7. Tracking the min here and reporting it as the final step's value
  gets the intended number under the intended name. MetricsManager reads
  `_step_values` before it calls `func.reset`, so the ordering is safe.
  """

  def __init__(self, cfg, env):
    self._fn = cfg.params["fn"]
    self._buf = torch.full((env.num_envs,), float("inf"), device=env.device)

  def reset(self, env_ids=None) -> None:
    self._buf[slice(None) if env_ids is None else env_ids] = float("inf")

  def __call__(self, env, fn=None) -> torch.Tensor:
    torch.minimum(self._buf, self._fn(env), out=self._buf)
    return self._buf


def m_reach_mm(env) -> torch.Tensor:
  return terms(env)["d_reach"] * 1000.0


def m_goal_mm(env) -> torch.Tensor:
  return terms(env)["d_goal_xy"] * 1000.0


def m_grasp_rate(env) -> torch.Tensor:
  return terms(env)["grasped"].float()


# The ring spawns 8 mm above where it settles, so under `reduce="max"` the
# settling drop alone reads ~8.0 mm on every episode no matter what the policy
# does. That is why v7's `lift_mm ~ 8.73` looked like a 9 mm plateau when the
# real lift was near zero. Suppress the metric until the ring is down; the fall
# is 8 mm, i.e. ~5 control steps, so 25 is generous.
SETTLE_STEPS = 25


def m_lift_mm(env) -> torch.Tensor:
  """Height gained above the resting pose. 32.0 mm is a clean unthread."""
  raw = (terms(env)["z"] - Z_REST) * 1000.0
  return torch.where(
    env.episode_length_buf < SETTLE_STEPS, torch.zeros_like(raw), raw
  )


def m_ring_z_mm(env) -> torch.Tensor:
  """ABSOLUTE ring height. Compare directly against the 36 mm post top."""
  raw = terms(env)["z"] * 1000.0
  return torch.where(
    env.episode_length_buf < SETTLE_STEPS, torch.zeros_like(raw), raw
  )


def m_depart_z_mm(env) -> torch.Tensor:
  """Absolute height at which the ring first left the post axis."""
  return latches(env).depart_z * 1000.0


def m_carry_z_min_mm(env) -> torch.Tensor:
  """LOWEST absolute height held during transport. The drag detector.

  Should sit ABOVE the 36 mm post top by a real margin. Below it means the ring
  was towed through the post -- the 2026-09-09 hardware failure.
  """
  return latches(env).carry_z_min * 1000.0


def m_clear_rate(env) -> torch.Tensor:
  """THE stage-3 indicator: did the ring ever come off the post this episode."""
  return terms(env)["cleared"].float()


def m_success(env) -> torch.Tensor:
  return block_at_goal(env).float()


# --- the staged potential shared by arms A and D --------------------------


def staged_potential(t: dict[str, torch.Tensor]) -> torch.Tensor:
  """robosuite's nested-ceiling structure, as a potential.

  Each stage starts above the previous stage's ceiling, so advancing is always
  worth more than perfecting the stage you are in and the agent can never top
  out mid-task:

      reach     0.5 * (1 - tanh(10 d))                        0.0 .. 0.5
      grasp     1.0 + lift_frac - PEN * off_axis * (1-clear_h)  0.6 .. 2.0
      transport 2.0 + (1 - tanh(5 d_goal))                      2.0 .. 3.0

  `lift_frac` is the term v7 was missing: it rises continuously across all
  32 mm from the resting pose to post clearance instead of latching at 2.5 mm.

  THE LIFT-STAGE LATERAL PENALTY, 2026-09-09. Hardware runs dragged the entire
  peg base AT THE MOMENT OF PICKUP -- the tool closed, lifted, and started
  translating while the ring was still down the shaft, towing the base with it.
  Everything after that was clean, which is why the fix belongs in the LIFT
  stage and nowhere else.

  `off_axis * (1 - clear_h)` is non-zero only when the ring is BOTH off the
  post axis AND below the post top: precisely "moving sideways before it is
  up". Rising straight zeroes it (off_axis 0); clearing the post zeroes it
  (clear_h 1). So the cheapest way out is the behaviour we want.

  WHAT THIS REPLACES, and why. The first attempt scaled TRANSPORT by clear_h to
  hold height across the carry. Measured against the checkpoint under hardware
  test, that was aimed at a window which was already fine -- between 30 mm and
  10 mm from the goal only 0.4-2.0 % of held samples sat below the post top,
  with the ring riding at 47-54 mm. Worse, `1 - tanh(GOAL_K d)` PEAKS at the
  goal, so the term applied its greatest pressure exactly where the ring has to
  come down to touch the square: it penalised the set-down, which success
  requires. Over 200 iterations carry_z_mm fell 39.9 -> 34.7 and success
  0.93 -> 0.88. Transport is therefore back to its original form and the
  set-down is unpenalised again.
  """
  reach = 0.5 * (1.0 - torch.tanh(REACH_K * t["d_reach"]))
  lift = (1.0 + t["lift_frac"]
          - LIFT_LATERAL_PEN * t["off_axis"] * (1.0 - t["clear_h"]))
  transport = 2.0 + (1.0 - torch.tanh(GOAL_K * t["d_goal_xy"]))
  phi = torch.where(t["grasped"], lift, reach)
  return torch.where(t["cleared"], transport, phi)


class _PotentialDelta:
  """Emits phi(s') - phi(s), the shaping form both Unity and a3b use."""

  def __init__(self, cfg, env):
    self._prev = torch.zeros(env.num_envs, device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True

  def _phi(self, env) -> torch.Tensor:
    raise NotImplementedError

  def __call__(self, env) -> torch.Tensor:
    phi = self._phi(env)
    delta = torch.where(self._fresh, torch.zeros_like(phi), phi - self._prev)
    self._prev = phi
    self._fresh = torch.zeros_like(self._fresh)
    return delta


# --- ARM A: staged potential ----------------------------------------------


class arm_a_staged(_PotentialDelta):
  """robosuite's stage structure delivered as potential-based shaping.

  Deliberately the *only* change from v7: same delta emission, same +2 success
  bonus, same termination set. If A works and B does not, the answer is that
  the potential's shape was the whole problem.
  """

  def _phi(self, env) -> torch.Tensor:
    return staged_potential(terms(env))


# --- ARM B: Unity's terms, repaired ---------------------------------------

_U = 0.05
LIN_TOL = 0.05 * _U
JAW_NEAR = 0.07 * _U
JAW_FAR = 0.3 * _U
JAW_ENTROPY_SCALE = 0.004


class arm_b_unity_fixed(_PotentialDelta):
  """Unity's six terms, with the two that provably cannot work replaced.

  Kept: GoalDistance, SparseGrasp, JawOpenEarly, JawEntropy, and the linear
  reach potential v6 added. Changed, minimally:

  * `BlockDistance` `exp(-80 d)` -> `1 - tanh(10 d)`. The exponential measures
    0.0057 at the 67 mm the tool actually starts from -- flat to within
    float noise. Unity gets away with it because its `ResetTool` starts the
    tool directly above the block, making approach a 1-D descent; we hold a
    fixed home pose so it is reproducible on the real arm.
  * `SparseLift`'s step function -> `lift_frac`, continuous over the full
    32 mm. This is the term v7 died on.

  This is the conservative arm: if Unity's reward was nearly right and only
  needed the saturation removed, B should be enough.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._jaw_entropy = torch.zeros(env.num_envs, device=env.device)
    self._prev_jaw = torch.zeros(env.num_envs, device=env.device)
    self._orig_d = torch.ones(env.num_envs, device=env.device)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    idx = slice(None) if env_ids is None else env_ids
    self._jaw_entropy[idx] = 0.0
    self._prev_jaw[idx] = 0.0

  def _phi(self, env) -> torch.Tensor:
    t = terms(env)
    d_pick, jaw = t["d_reach"], t["jaw"]

    if self._fresh.any():
      f = self._fresh
      self._orig_d[f] = torch.norm(t["block"] - t["goal"], dim=-1)[f].clamp_min(1e-4)

    reach = -d_pick / 0.05
    block_distance = 1.0 - torch.tanh(REACH_K * d_pick)

    d_goal = torch.norm(t["block"] - t["goal"], dim=-1)
    goal_distance = 1.0 - (d_goal - LIN_TOL) / (self._orig_d - LIN_TOL)

    sparse_grasp = (t["grasped"] & (jaw < 0.1)).float()
    lift = t["grasped"].float() * t["lift_frac"]

    jaw_open_early = torch.where(
      (~t["grasped"]) & (d_pick < JAW_FAR) & (d_pick > JAW_NEAR) & (jaw < 0.01),
      -torch.ones_like(d_pick), torch.zeros_like(d_pick),
    )
    self._jaw_entropy = self._jaw_entropy - (jaw - self._prev_jaw).abs()
    self._prev_jaw = jaw

    return (
      reach + block_distance + goal_distance + sparse_grasp + 2.0 * lift
      + 0.1 * jaw_open_early + JAW_ENTROPY_SCALE * self._jaw_entropy
    )


# --- ARM C: absolute dense staged (robosuite semantics proper) -------------


def arm_c_dense(env) -> torch.Tensor:
  """The staged reward as an ABSOLUTE per-step value, not a delta.

  robosuite pays the staged value every step rather than its change. That is a
  genuinely different optimisation problem: holding a good state keeps earning,
  so there is a standing pull towards the best state reachable rather than only
  towards *improving*. It also cannot be washed out by entropy the way a
  telescoping potential can -- an episode sitting at stage 3 banks ~3/step.

  The trade is that ending the episode now forfeits the remaining income, so
  success must not terminate for this arm (see `psm_lift_peg_a3b_env_cfg`) --
  otherwise the optimal policy is to hover at the goal and never set the ring
  down. Scaled by 0.01 so a full 5000-step episode is worth ~150, not ~15000.
  """
  return 0.01 * staged_potential(terms(env))


# --- ARM D: Dense2Sparse anneal -------------------------------------------

# 6000 iterations x 24 steps per env; anneal over the first 60%.
ANNEAL_STEPS = int(0.6 * 6000 * 24)


class arm_d_anneal(arm_a_staged):
  """Arm A's potential, faded out as training proceeds.

  Dense2Sparse's argument is that a policy trained to convergence under dense
  shaping is partly fitted to the shaping rather than the task, and that fading
  the dense term once it has done its job of getting the agent to the sparse
  reward at all yields a policy that holds up better. The sparse +2 success
  bonus is untouched throughout; only the shaping weight decays, linearly to
  zero over the first 60% of training.

  Riskiest arm: if the anneal outruns the learning it removes the signal before
  the policy is standing on its own. It is included because if it *does* work
  it is the one whose result transfers, and it costs one more GPU slice.
  """

  def __call__(self, env) -> torch.Tensor:
    delta = super().__call__(env)
    step = float(getattr(env, "common_step_counter", 0))
    w = max(0.0, 1.0 - step / ANNEAL_STEPS)
    return delta * w


VARIANTS = {
  "unity": None,          # v7's reward, kept registered as the baseline
  "a_staged": arm_a_staged,
  "b_unity_fixed": arm_b_unity_fixed,
  "c_dense": arm_c_dense,
  "d_anneal": arm_d_anneal,
}
