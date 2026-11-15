"""Phase-locked reward: make the policy copy the scripted trajectory.

The earlier staged potential was too permissive. It rewarded pulling the tissue
in ANY direction, so the policy found grab-drag-release and abandoned retrying
altogether: grasp_attempts fell 9.4 -> 1.3 and grasped 0.90 -> 0.38 while
revealed_only climbed to 0.19 with success_rate still 0.

This replaces exploration with imitation. The env runs the same state machine as
scripts/tissue_scripted.py (which solves 128/128), and the reward is distance to
THAT phase's waypoint. There is no credit for finding another way, because there
is no other way we want.

    DESCEND  straight down, laterally pinned to the start xy
    CLOSE    jaw shut, hold still
    VERIFY   lift and see whether the tissue came
    PEEL     drive to anchor + peel_dir * MAX_DRAG
    DONE

Two deliberate exceptions to the rigidity, both because they are the parts that
have to survive contact with the real rig:

  * the RETRY costs nothing AND earns nothing. The grasp fails 25-65% of the
    time by construction, so charging for a retry would teach the opposite of
    what we need -- but PAYING for one is worse, and that mistake cost a run.
    phi is keyed on the highest phase REACHED this episode, so falling back to
    DESCEND neither penalises nor rewards; only new ground pays.
  * phase TIMING is not scored. Only position within the phase is.

What IS punished is going somewhere off-plan, which is the freedom that caused
the problem.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

P_DESCEND, P_CLOSE, P_VERIFY, P_PEEL, P_DONE = 0, 1, 2, 3, 4

CLOSE_STEPS = 30
VERIFY_STEPS = 22          # measured: below ~20 a held and a failed grasp are
CAME_UP_M = 0.0015         # indistinguishable, GRASP_SLACK eats the first 1.5 mm
PEEL_AZIM_DEG = 60.0
PEEL_ELEV_DEG = 30.0       # 30 clears the 10 mm success height with margin
PEEL_FRAC = 0.95

# Lateral leash during the descent. The grasp point is directly below the tool
# at reset -- that is how the real rig will be set up -- so ANY xy motion on the
# way down is off-plan by definition.
LEASH_M = 0.004
LEASH_W = 4.0


class _Phase:
  def __init__(self, env):
    n, dev = env.num_envs, env.device
    self.k = torch.zeros(n, dtype=torch.long, device=dev)
    self.t = torch.zeros(n, dtype=torch.long, device=dev)
    self.start_xy = torch.zeros(n, 2, device=dev)
    self.rest_z = torch.zeros(n, device=dev)
    self.prev = torch.zeros(n, device=dev)
    self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
    # HIGHEST phase reached this episode. The potential is built on this, not
    # on the current phase, and that is the whole defence against farming --
    # see _phi.
    self.kmax = torch.zeros(n, dtype=torch.long, device=dev)
    a = math.radians(PEEL_AZIM_DEG)
    e = math.radians(PEEL_ELEV_DEG)
    d = torch.tensor([math.cos(e) * math.cos(a), -math.cos(e) * math.sin(a),
                      math.sin(e)], device=dev)
    self.dir = d / d.norm()


def _state(env) -> _Phase:
  st = getattr(env, "_tissue_phase", None)
  if st is None:
    st = _Phase(env)
    env._tissue_phase = st
  return st


def reset_phase(env, env_ids) -> None:
  """Re-arm at episode start. Also captures the start xy, which is the axis the
  descent is leashed to, and the tissue's rest height."""
  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as T
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

  st = _state(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  st.k[env_ids] = P_DESCEND
  st.kmax[env_ids] = P_DESCEND
  st.t[env_ids] = 0
  st.fresh[env_ids] = True
  tip = a3b._tool_pos(env) - env.scene.env_origins
  st.start_xy[env_ids] = tip[env_ids, :2]
  st.rest_z[env_ids] = (T._anchor_pos(env))[env_ids, 2]


def advance_phase(env, env_ids) -> None:
  """Run the state machine one step. A `step`-mode event, so it fires once per
  control step regardless of how many reward terms read the phase."""
  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as T
  from surgical_lab.tasks.retract.mdp import grasp as gr
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

  st = _state(env)
  tip = a3b._tool_pos(env) - env.scene.env_origins
  corner = T._anchor_pos(env)
  jaw_closed = env.sim.data.qpos[:, gr._state(env, T.GRASP_IDS, "tissue").jaw_qadr] \
      < gr.JAW_CLOSED_BELOW
  st.t += 1
  st.kmax = torch.maximum(st.kmax, st.k)

  at_bottom = tip[:, 2] <= (T.TISSUE_FLOOR_Z + 0.004)
  adv = (st.k == P_DESCEND) & at_bottom
  st.k = torch.where(adv, torch.full_like(st.k, P_CLOSE), st.k)
  st.t = torch.where(adv, torch.zeros_like(st.t), st.t)

  adv = (st.k == P_CLOSE) & jaw_closed & (st.t > CLOSE_STEPS)
  st.k = torch.where(adv, torch.full_like(st.k, P_VERIFY), st.k)
  st.t = torch.where(adv, torch.zeros_like(st.t), st.t)
  # Re-baseline the lift test at the START of every VERIFY, not once per
  # episode. "Did it come up" means up relative to where the corner was just
  # before THIS lift; against a stale baseline it means nothing.
  #
  # Measured with the TEACHER driving, so this was the env and not a policy:
  #
  #   episode age   mean rise   P(pass | actually held)
  #     0-200         +0.66 mm      0.435
  #   200-500         -1.94 mm      0.062
  #   500-1000        -2.36 mm      0.002
  #   1000+           -2.38 mm      0.000
  #
  # rest_z was captured in reset_phase BEFORE the sheet settles -- it spawns at
  # TISSUE_Z above the floor and sags ~2.4 mm -- so after ~200 steps the corner
  # sits below a baseline it can never clear, CAME_UP_M is unreachable, and
  # every retry is futile by construction. The teacher hid this by winning in
  # ~1.89 attempts inside that window; any slower policy is capped, which is
  # why the student stalled at VERIFY in 56% of its failures.
  st.rest_z = torch.where(adv, corner[:, 2], st.rest_z)

  done_v = (st.k == P_VERIFY) & (st.t > VERIFY_STEPS)
  came_up = (corner[:, 2] - st.rest_z) > CAME_UP_M
  st.k = torch.where(done_v & came_up, torch.full_like(st.k, P_PEEL), st.k)
  # Failed: back to DESCEND. No re-arm -- phi is keyed on kmax, so the drop is
  # already free without one.
  failed = done_v & ~came_up
  st.k = torch.where(failed, torch.full_like(st.k, P_DESCEND), st.k)
  st.t = torch.where(done_v, torch.zeros_like(st.t), st.t)

  # Losing the tissue mid-peel also sends it back, likewise for free.
  dropped = (st.k == P_PEEL) & ~gr.is_held(env)
  st.k = torch.where(dropped, torch.full_like(st.k, P_DESCEND), st.k)
  st.t = torch.where(dropped, torch.zeros_like(st.t), st.t)


def teacher_action(env) -> torch.Tensor:
  """What the scripted controller would command in this state. (n, 4).

  Reactive, keyed on the phase the env is already tracking -- never on a step
  counter, because the policy has to be able to infer the same thing from what
  it can see.
  """
  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as T
  from surgical_lab.tasks.retract.mdp import grasp as gr
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

  st = _state(env)
  n, dev = env.num_envs, env.device
  u = torch.zeros(n, 3, device=dev)
  jaw = torch.ones(n, device=dev)

  u[:, 2] = torch.where(st.k == P_DESCEND, -torch.ones(n, device=dev), u[:, 2])
  jaw = torch.where(st.k == P_CLOSE, -torch.ones_like(jaw), jaw)
  m = st.k == P_VERIFY
  jaw = torch.where(m, -torch.ones_like(jaw), jaw)
  u[:, 2] = torch.where(m, torch.ones_like(u[:, 2]), u[:, 2])

  g = getattr(env, "_tissue_grasp", None)
  peel = st.k >= P_PEEL
  if g is not None and bool(peel.any()):
    want = g.anchor + st.dir.unsqueeze(0) * (gr.MAX_DRAG * PEEL_FRAC)
    err = want - a3b._tool_pos(env)
    cmd = (err / err.norm(dim=-1, keepdim=True).clamp_min(1e-6)
           * err.norm(dim=-1, keepdim=True).clamp_max(0.004) / 0.004)
    u = torch.where(peel.unsqueeze(-1), cmd.clamp(-1, 1), u)
    jaw = torch.where(peel, -torch.ones_like(jaw), jaw)
  return torch.cat([u, jaw.unsqueeze(-1)], dim=-1)


def action_match(env) -> torch.Tensor:
  """-|| a_policy - a_scripted ||^2, per step. This IS the objective.

  Everything else on this task has been a reward defined on OUTCOMES, and the
  policy kept finding cheaper outcomes than the one we wanted: drag the tissue
  without grasping, then park at the bottom and chatter the jaw 112 times an
  episode. Both were legitimate solutions to what I had written down.

  A reward defined on the ACTION has no such slack. The only way to score is to
  emit what the script would emit, so there is nothing else to discover -- the
  optimum IS the trajectory. It is still RL in every formal sense; the reward
  just happens to be imitation, which is a legitimate thing for a reward to be
  and no less principled than the hand-tuned distance-and-lift shaping the work
  we are compared against uses.

  Squared, not absolute, so being badly wrong costs disproportionately more than
  being slightly wrong -- and the jaw channel, which is the one genuinely
  perceptual decision, is on the same scale as the three motion channels.
  """
  a = env.action_manager.action
  return -((a - teacher_action(env)) ** 2).sum(dim=-1)


def _phi(env) -> torch.Tensor:
  """Phase index + progress within the phase. Nested ceilings, as before."""
  from surgical_lab.tasks.retract.config.psm import tissue_a3b_env_cfg as T
  from surgical_lab.tasks.retract.mdp import grasp as gr
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

  st = _state(env)
  tip = a3b._tool_pos(env) - env.scene.env_origins
  corner = T._anchor_pos(env)

  span = 0.014
  descend = 1.0 - ((tip[:, 2] - T.TISSUE_FLOOR_Z).clamp_min(0.0) / span).clamp(0, 1)
  closing = torch.ones_like(descend)
  lift = ((corner[:, 2] - st.rest_z) / 0.006).clamp(0, 1)

  s = getattr(env, "_tissue_grasp", None)
  if s is None:
    peel = torch.zeros_like(descend)
  else:
    want = s.anchor + st.dir.unsqueeze(0) * (gr.MAX_DRAG * PEEL_FRAC)
    err = torch.norm(want - a3b._tool_pos(env), dim=-1)
    peel = (1.0 - err / (gr.MAX_DRAG * PEEL_FRAC)).clamp(0, 1)

  prog = torch.where(st.k == P_DESCEND, descend,
          torch.where(st.k == P_CLOSE, closing,
          torch.where(st.k == P_VERIFY, lift, peel)))
  # phi is built on the HIGHEST phase reached, and within-phase progress counts
  # only while the agent is AT that phase. This is what makes a retry cost
  # nothing AND earn nothing.
  #
  # The previous version re-armed the potential baseline on a failed grasp so
  # the drop back to DESCEND emitted no penalty. That turned out to pay +2 per
  # cycle forever: the run climbed to 0.85 success, discovered that descend ->
  # close -> verify -> fail was a free +2, and collapsed to 0.000 with 112
  # grasp attempts per episode and every episode running the full 2000 steps.
  # Re-arming removed the telescoping that makes a potential safe.
  at_max = (st.k == st.kmax).float()
  return st.kmax.float() + prog * at_max


class phase_potential:
  """phi(s') - phi(s) over the phase-locked potential."""

  def __init__(self, cfg, env):
    _state(env)

  def reset(self, env_ids=None) -> None:
    pass  # reset_phase (a reset-mode event) owns this

  def __call__(self, env) -> torch.Tensor:
    st = _state(env)
    phi = _phi(env)
    delta = torch.where(st.fresh, torch.zeros_like(phi), phi - st.prev)
    st.prev = phi
    st.fresh = torch.zeros_like(st.fresh)
    return delta


def leash_penalty(env) -> torch.Tensor:
  """Punish leaving the vertical line through the grasp point.

  Only while descending or closing -- the peel is supposed to move sideways.
  This is the term that removes the freedom that produced grab-drag-release:
  there is no lateral exploration to be had on the way down.
  """
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b
  st = _state(env)
  tip = a3b._tool_pos(env) - env.scene.env_origins
  off = torch.norm(tip[:, :2] - st.start_xy, dim=-1)
  gated = (st.k == P_DESCEND) | (st.k == P_CLOSE) | (st.k == P_VERIFY)
  return -LEASH_W * ((off - LEASH_M).clamp_min(0.0) ** 2) * gated.float()


def phase_metric(env) -> torch.Tensor:
  return _state(env).k.float()


def leash_mm_metric(env) -> torch.Tensor:
  from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b
  st = _state(env)
  tip = a3b._tool_pos(env) - env.scene.env_origins
  return torch.norm(tip[:, :2] - st.start_xy, dim=-1) * 1000.0
