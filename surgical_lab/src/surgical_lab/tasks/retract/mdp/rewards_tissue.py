"""Potential-based task shaping for tissue retraction, in liftpeg's shape.

The motion is liftpeg's, one for one:

    liftpeg                             tissue
    ------------------------------      ---------------------------------
    descend to the ring's grip point    descend to the tissue corner
    close, `held`                       close, `is_held`
    lift until the ring clears the post lift until the tool is clear of the
                                        floor at the success height
    grasp slipped -> try again          VERIFY failed -> RETRY
    carry toward the goal square        move aside, uncovering the square
    ring resting on the goal            square fully visible, still held, up

so it gets liftpeg's reward: a nested-ceiling staged potential delivered as
phi(s') - phi(s), with the same discipline that made liftpeg work.

    reach     1 - tanh(REACH_K * d_anchor)      0 .. 1        not holding
    lift      1 + LIFT_GAIN * lift_frac         1 .. 1+LG     holding
    reveal    (1+LG) + REVEAL_GAIN * seen       .. +RG        holding AND up

Stage ceilings never overlap, so advancing always beats perfecting the stage
you are in, and no stage can be topped out mid-task.

WHY THIS IS NOT THE EARLIER STAGED POTENTIAL
--------------------------------------------
An earlier version of this file paid `pull_frac = |tool - anchor| / MAX_DRAG`:
displacement in ANY direction, with the reveal stage gated only on `seen > 0`.
The policy found grab-drag-release, exactly as it should have -- over 419
iterations `grasp_attempts` fell 9.4 -> 1.3 and `grasped` 0.90 -> 0.38 while
`revealed_only` climbed to 0.19 and `success_rate` stayed at 0. The sheet is
inextensible and pinned, so a flap shoved aside STAYS aside; dragging and
letting go reveals the square permanently and costs nothing. That is what
replaced this reward with phase-locked imitation.

Two changes close it, both borrowed from liftpeg rather than invented:

  * **The reveal stage is gated on `held & lifted`**, where `lifted` is the
    success criterion's own `tip_z >= SUCCESS_MIN_TIP_Z` clause. This is the
    analogue of liftpeg gating its carry stage on `cleared` -- a physical fact,
    not a proxy. Dragging along the floor while holding now earns nothing from
    `seen`: it sits in the lift stage, whose only currency is height.
  * **The middle stage measures HEIGHT, not displacement.** Lateral motion is
    paid for exactly once, in the reveal stage, and only while the tissue is
    held and up. There is no longer any reward for moving the tissue sideways
    in a way that does not uncover the square.

Releasing is punished automatically and without a bespoke term: `held` goes
false, phi falls from the reveal stage all the way to descend, and the delta is
a large negative. That is the property that makes a potential worth using here.

WHY THERE IS NO RETRY BONUS
---------------------------
The grasp succeeds only 35-75% of the time by construction (`GRASP_P_RANGE`),
so retries are essential, and the potential offers no gradient toward
re-opening the jaw -- opening moves phi nowhere. The previous answer was a
`retry_bonus` paid on the rising edge of the jaw closing, gated only on being
low; the policy parked at the bottom and chattered the jaw 112 times an
episode. It is not reinstated here.

What replaces it is the size of the prize: a successful grasp jumps phi by a
whole stage, from descend's ceiling of 1.0 into the lift stage, and from there
the reveal stage is worth REVEAL_GAIN more. Re-opening is free (delta 0), not
penalised, so an entropic policy explores it and the jump pays for it. This is
liftpeg's situation exactly -- it has no retry term either and learns a jaw
that closes on contact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Reach is liftpeg's REACH_K over the DISTANCE to the grasp corner, and both
# halves of that matter.
#
# The first version of this stage scored HEIGHT with DESCEND_K = 120, inherited
# from the imitation-era file where it was a fine bottom-out signal sitting
# beside a reward that already drove the descent. With imitation gone it was
# the only thing asking the tool to come down, and at the home height it is
# flat: descend_frac reads 0.000003 at 58 mm with a slope of 7e-7 per mm, so
# the policy would have to random-walk to ~25 mm before feeling anything. It
# duly never descended -- 1092 iterations, zero grasp attempts, stage pinned at
# 0 (fir job 58344751). liftpeg's REACH_K = 12 over distance gives 7.4e-3 per
# mm at 60 mm, four orders of magnitude more, and near-constant all the way in.
#
# Distance rather than height also restores the lateral guidance a height-only
# term cannot give: the corner moves with the sheet, and the tool has to arrive
# at it in xy as well as z.
REACH_K = 12.0
# Kept for the teacher observation and the metrics only; no longer in the
# potential. The fixture stops the tip at ~1.65 mm.
DESCEND_SPAN = 0.012
DESCEND_K = 120.0

# Where the tip sits once the virtual fixture has stopped it -- the height a
# lift starts from. Measured, not chosen; see DESCEND_SPAN above.
LIFT_REF = 0.00165
# Height at which the lift term saturates. Deliberately ABOVE the success
# criterion's SUCCESS_MIN_TIP_Z (10 mm), by the same 4 mm kind of margin
# liftpeg puts above its post top. Saturating a lift term short of the height
# the task actually requires is the single most expensive mistake available
# here: on liftpeg a clear height 13 mm too low capped every vision run at 0%
# success for 22 hours while every metric said the lift had succeeded. If
# SUCCESS_MIN_TIP_Z ever moves, this must move with it -- hence the assert.
LIFT_TOP_MARGIN = 0.004

# Gains. The big one goes on the stage that must be traversed monotonically
# over a long span, which for liftpeg was the 32 mm lift and here is the
# reveal: `seen` has to climb from 0 to 1 while the grasp is kept alive.
import os as _os  # noqa: E402

# --- liftpeg-style waypoints, added 2026-09-12 for the image-only arms -------
# Camera-only removes the proprioception that explained 87 % of the scripted
# teacher's action, so the shaping has to carry far more of the load, and it
# has to be PHASE-FREE: nothing may key off a step counter, because a single
# frame cannot see one. These are geometric targets only.
#
# APPROACH_H puts a waypoint directly ABOVE the grasp corner so the tool
# descends onto it instead of swiping in sideways -- the sideways entry is what
# drags the sheet before the jaw ever closes.
APPROACH_H = float(_os.environ.get("TISSUE_APPROACH_H_MM", "12.0")) / 1000.0
APPROACH_GAIN = float(_os.environ.get("TISSUE_APPROACH_GAIN", "1.0"))
# The GATE: liftpeg's grasped_strict only counts a grasp when the tool is AT
# the grip point, which is what made its grasp height track the target to
# within 0.2 mm. Contact alone is not enough -- it is what lets a policy claim
# the lift stage from a bad pose.
GRIP_TOL = float(_os.environ.get("TISSUE_GRIP_TOL_MM", "5.0")) / 1000.0

REACH_GAIN = float(_os.environ.get("TISSUE_REACH_GAIN", "1.0"))
LIFT_GAIN = float(_os.environ.get("TISSUE_LIFT_GAIN", "2.0"))
REVEAL_GAIN = float(_os.environ.get("TISSUE_REVEAL_GAIN", "6.0"))


def _tip_z(env) -> torch.Tensor:
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b
  return (a3b._tool_pos(env) - env.scene.env_origins)[:, 2]


def _quantities(env) -> dict[str, torch.Tensor]:
  """Everything the potential needs, computed once per call."""
  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as T
  from surgical_lab.tasks.retract.mdp import grasp as gr
  from surgical_lab.tasks.retract.mdp import visibility as vis
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

  held = gr.is_held(env)
  seen = vis.visible_fraction(env, T.GOAL_SIDE)
  tip_z = _tip_z(env)

  # Distance from the tool to the grasp corner -- the reach stage's currency,
  # and the analogue of liftpeg's d_grip.
  _anchor = T._anchor_pos(env)
  _tool = a3b._tool_pos(env)
  d_anchor = torch.norm(_tool - _anchor, dim=-1)
  # Waypoint straight above the corner, and the positional gate on the grasp.
  _above = _anchor.clone()
  _above[:, 2] = _above[:, 2] + APPROACH_H
  d_approach = torch.norm(_tool - _above, dim=-1)
  at_grip = d_anchor < GRIP_TOL

  # Descent, measured from the floor rather than from a target height: the
  # fixture decides where "as low as possible" is, and it is not the same
  # number the policy could compute from its start pose.
  above = (tip_z - T.FLOOR_HARD_Z).clamp_min(0.0)
  descend_frac = 1.0 - torch.tanh(DESCEND_K * (above - 0.002).clamp_min(0.0))

  # Height, not displacement. See the module docstring: paying for displacement
  # is what produced grab-drag-release.
  lift_top = T.SUCCESS_MIN_TIP_Z + LIFT_TOP_MARGIN
  assert lift_top > T.SUCCESS_MIN_TIP_Z, (
    "the lift term must saturate ABOVE the height success requires, or the "
    "policy is paid in full for a lift that does not qualify")
  lift_frac = ((tip_z - LIFT_REF) / (lift_top - LIFT_REF)).clamp(0.0, 1.0)
  # The gate into the reveal stage is the success criterion's own height
  # clause, so the potential's top stage and the task's definition of done
  # cannot drift apart.
  lifted = tip_z >= T.SUCCESS_MIN_TIP_Z
  # `held` is the physical grasp; `held_strict` additionally requires the tool
  # to be at the corner, so the lift stage cannot be claimed from a pose that
  # happens to make contact. Straight port of liftpeg's grasped_strict.
  held_strict = held & at_grip

  st = getattr(env, "_tissue_grasp", None)
  if st is None:
    pull_frac = torch.zeros_like(seen)
  else:
    pull = torch.norm(a3b._tool_pos(env) - st.anchor, dim=-1)
    # Reported for the teacher observation and the metrics only. The potential
    # deliberately does NOT read this -- paying for drag in any direction is
    # the exploit this reward exists to avoid.
    pull_frac = (pull / gr.MAX_DRAG).clamp(0.0, 1.0)

  return {"held": held, "seen": seen, "lifted": lifted,
          "lift_frac": lift_frac, "descend_frac": descend_frac,
          "d_anchor": d_anchor, "pull_frac": pull_frac,
          "d_approach": d_approach, "at_grip": at_grip,
          "held_strict": held_strict}


def staged_potential(q: dict[str, torch.Tensor]) -> torch.Tensor:
  """Nested-ceiling staged potential, liftpeg's construction.

  Stage ceilings stay nested as the gains move: each stage starts where the
  previous one tops out, so advancing is always worth more than perfecting the
  stage you are in and raising one gain can never invert the order.

      approach  A * (1 - tanh(K d_above))                    0 .. A
      reach     A + R * (1 - tanh(K d_anchor))               A .. A+R
      lift      A + R + L * lift_frac                        .. A+R+L
      reveal    A + R + L + V * seen                         .. A+R+L+V

  TWO additions for the camera-only arms, 2026-09-12:

  * an APPROACH waypoint directly above the corner, so the tool is paid to
    arrive from above and descend onto the grasp rather than swiping in
    laterally -- the sideways entry drags the sheet before the jaw closes.
  * the lift stage now gates on `held_strict`, which is the physical grasp AND
    the tool being within GRIP_TOL of the corner. Contact alone let a policy
    claim the lift stage from a bad pose. This is what made liftpeg's grasp
    height track its target to within 0.2 mm.

  Everything here is geometric. Nothing keys off a step counter, deliberately:
  a camera-only actor cannot see one, which is the measured reason the history
  vector existed in the first place.
  """
  approach = APPROACH_GAIN * (1.0 - torch.tanh(REACH_K * q["d_approach"]))
  reach = APPROACH_GAIN + REACH_GAIN * (1.0 - torch.tanh(REACH_K * q["d_anchor"]))
  base = APPROACH_GAIN + REACH_GAIN
  lift = base + LIFT_GAIN * q["lift_frac"]
  reveal = (base + LIFT_GAIN) + REVEAL_GAIN * q["seen"]
  held = q["held_strict"]
  phi = torch.where(q["at_grip"], reach, approach)
  phi = torch.where(held, lift, phi)
  return torch.where(held & q["lifted"], reveal, phi)


class tissue_staged:
  """Emits phi(s') - phi(s), the same shaping form liftpeg and a3b use."""

  def __init__(self, cfg, env):
    self._prev = torch.zeros(env.num_envs, device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True

  def __call__(self, env) -> torch.Tensor:
    phi = staged_potential(_quantities(env))
    # A fresh episode must emit 0, not phi - phi_of_the_previous_episode. That
    # bug pays the agent for whatever the last episode happened to end on --
    # and it is live in liftpeg's arm_e_shaped, where the reset branch is a
    # no-op because `prev` has already been overwritten by the time it runs.
    delta = torch.where(self._fresh, torch.zeros_like(phi), phi - self._prev)
    self._prev = phi
    self._fresh = torch.zeros_like(self._fresh)
    return delta


def lift_frac_metric(env) -> torch.Tensor:
  return _quantities(env)["lift_frac"]


def reach_mm_metric(env) -> torch.Tensor:
  """Tool-to-corner distance. The first thing to watch: if this does not fall,
  the reach stage is dead and nothing downstream can happen."""
  return _quantities(env)["d_anchor"] * 1000.0


def stage_metric(env) -> torch.Tensor:
  """0 descend, 1 lift, 2 reveal -- which stage the potential is paying."""
  q = _quantities(env)
  return (q["held"].float() + (q["held"] & q["lifted"]).float())
