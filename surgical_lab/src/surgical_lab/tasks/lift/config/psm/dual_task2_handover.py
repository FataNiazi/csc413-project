"""Task 2 (mid-air handover), reward v0: the Task 1 staged potential, re-staged.

Arm 1 lifts the ring off the peg, arm 2 takes it from arm 1 in the air, arm 1
lets go, and arm 2 sets the ring down flat on the goal. The arms never touch.

Built on what the scripted handover (scripts/demo_task2.py) showed:

  * the exchange works in this layout, at any ring yaw: arm 1's grip turns the
    ring to roughly the same yaw whatever it started at, so the reset yaw is
    left random, as in Task 1;
  * arm 2's grip is the weak link (it gets the thick side of the ring and
    often only its jaw tips on it), so the exchange stage pays for arm 2's
    STRICT grasp, and arm 1 letting go of a ring arm 2 does not hold is
    charged like Task 1's high release;
  * a weak grip slowly slides, so hovering must not pay: as in Task 1, only
    progress pays (PHI(s') - PHI(s)), and discounting does the rest.

Stages (PHI, return units). Nested: every stage starts above the best value
of the one before it.

  phase 0
    reach 1    0 .. 1    arm 1's tool to a grip point on the ring
    lift 1     1 .. 7    held by arm 1, ring height up to post clearance (38 mm)
    reach 2    8 .. 9    ring held by arm 1 in the air (>= AIR_Z), arm 2's tool
                         to its grip point
    both      10         both arms hold the ring (strict) in the air
  handover: arm 2 holds it alone, the ring still in the air, and arm 1 held it
            (pads on) within the last HAND_WINDOW steps -> phase 1 (base 11)
  phase 1
    regrasp  11 .. 12    ring not held by arm 2 (dropped): arm 2 back to it
    carry    12 .. 15    held by arm 2, ring towards the goal (xy)
    place    15 .. 18    over the goal, ring down
    on goal  19          ring lying flat on the goal
  success: on the goal, flat, arm 2 let go -> +SUCCESS_BONUS, episode ends
  lost:    ring on the floor away from the goal -> -LOST_PENALTY, ends
  contact: any arm-arm contact -> -10, ends (Task 1's)

  release penalty (one-shot, both phases): an arm lets go of a raised ring
  that the other arm is not holding, scaled by the ring's height (0 at
  RELEASE_Z, full at RELEASE_Z + RELEASE_SPAN). Setting it down costs nothing.

The HAND_WINDOW check stops a relay counting as a handover: arm 2 picking the
ring off the goal (or the plate) after arm 1 dropped it is not one.

Every number is an env var (T2_*). Task 1's files are not changed; the grip
geometry is imported from dual_task1_staged.
"""

import os

import torch
from mjlab.envs import mdp as base_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from surgical_lab.tasks.lift.config.psm import dual_task1 as T1
from surgical_lab.tasks.lift.config.psm import dual_task1_staged as S
from surgical_lab.tasks.lift.config.psm import liftpeg_dual as D
from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as LR

_f = T1._f
ARMS = T1.ARMS

EPISODE_S = _f("T2_EPISODE_S", "40")

# --- stage sizes ------------------------------------------------------------------
REACH_K = S.REACH_K
GOAL_K = S.GOAL_K
LIFT_GAIN = _f("T2_LIFT_GAIN", "6")
CARRY_GAIN = _f("T2_CARRY_GAIN", "3")
PLACE_GAIN = _f("T2_PLACE_GAIN", "3")

P0_LIFT = 1.0
P0_REACH2 = P0_LIFT + LIFT_GAIN + 1.0                        # 8
P0_BOTH = P0_REACH2 + 2.0                                    # 10
P1_BASE = P0_BOTH + 1.0                                      # 11
P1_CARRY = P1_BASE + 1.0                                     # 12
P1_PLACE = P1_CARRY + CARRY_GAIN                             # 15
P1_ON_GOAL = P1_PLACE + PLACE_GAIN + 1.0                     # 19

# --- v1 ("spot", task Surgicallab-Dual-Task2-Spot): the exchange happens at a
# fixed point above the goal. v0 had no fixed place, and its arm-2 stage only
# needed the ring >= AIR_Z up: going down could lose the stage, going up cost
# nothing, so arm 1 carried the ring ~20 cm up with arm 2 chasing it (run
# k36zdnep: 0 handovers, ring lost in ~half the episodes, 22% arm contact).
SPOT_Z = _f("T2_SPOT_Z_MM", "45") / 1000.0      # ring origin height over the goal; post top is 36
SPOT_R = _f("T2_SPOT_R_MM", "15") / 1000.0      # "at the spot": arm 2's stages count inside this
SPOT_STD = _f("T2_SPOT_STD_MM", "10") / 1000.0  # fine term of the present stage
PRESENT_GAIN = _f("T2_PRESENT_GAIN", "3")       # 2 coarse + 1 fine
REACH2_GAIN = _f("T2_REACH2_GAIN", "3")         # 2 coarse + 1 fine
REACH2_STD = _f("T2_REACH2_STD_MM", "5") / 1000.0
SPOT_CARRY_GAIN = _f("T2_SPOT_CARRY_GAIN", "1")  # the ring is already over the goal

V1_PRESENT = P0_LIFT + LIFT_GAIN                             # 7
V1_REACH2 = V1_PRESENT + PRESENT_GAIN + 1.0                  # 11
V1_TOUCH2 = V1_REACH2 + REACH2_GAIN + 1.0                    # 15
V1_BOTH = V1_TOUCH2 + 1.0                                    # 16
V1_P1_BASE = V1_BOTH + 1.0                                   # 17
V1_P1_CARRY = V1_P1_BASE + 1.0                               # 18
V1_P1_PLACE = V1_P1_CARRY + SPOT_CARRY_GAIN                  # 19
V1_P1_ON_GOAL = V1_P1_PLACE + PLACE_GAIN + 1.0               # 23

AIR_Z = _f("T2_AIR_Z_MM", "20") / 1000.0       # ring origin (its bottom): clear of plate and goal
HAND_WINDOW = int(_f("T2_HAND_WINDOW", "10"))  # steps since arm 1 last had its pads on
FLAT_DEG = _f("T2_FLAT_DEG", "15")
RELEASE_PEN = _f("T2_RELEASE_PEN", "6")
RELEASE_Z = _f("T2_RELEASE_Z_MM", "6") / 1000.0
RELEASE_SPAN = _f("T2_RELEASE_SPAN_MM", "20") / 1000.0

W_SHAPE = _f("T2_W_SHAPE", "1.0")
SUCCESS_BONUS = _f("T2_SUCCESS_BONUS", "5")
LOST_PENALTY = _f("T2_LOST_PENALTY", "2")

JAW_CLOSED = S.JAW_CLOSED
# "Held" is Task 1's strict grasp (both pads on, jaw closed, tool at a grip
# point) with a looser grip-point tolerance. The scripted handover's arm 2
# grips 4-7 mm above the ideal height (it gets the thick side of the ring) and
# still carries the ring to the goal; Task 1's 3.5 mm / 4 mm counted that grip
# for 3 steps, so the handover never registered and arm 1's release was
# charged as a drop.
GRIP_TOL_XY = _f("T2_GRIP_TOL_XY_MM", "5") / 1000.0
GRIP_TOL_Z = _f("T2_GRIP_TOL_Z_MM", "8") / 1000.0
CLEAR_Z = S.CLEAR_Z
SETTLE_STEPS = S.SETTLE_STEPS


# --- state ---------------------------------------------------------------------------

class _State:
  def __init__(self, n, dev):
    z = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    self.phase1 = z()
    self.cleared1 = z()
    self.both_ever = z()                    # metric: both arms held it in the air
    self.since_pads1 = torch.full((n,), 10 ** 6, dtype=torch.long, device=dev)
    self.prev_contact = torch.zeros(n, 2, dtype=torch.bool, device=dev)
    self.lift1_max = torch.zeros(n, device=dev)
    self.z_hand = torch.zeros(n, device=dev)       # mm, ring height at the handover
    self.release_z = torch.zeros(n, device=dev)    # mm, last penalised release
    self.goal_min = torch.full((n,), 1.0, device=dev)
    self.grip2_min = torch.full((n,), 1.0, device=dev)
    self.spot_min = torch.full((n,), 1.0, device=dev)   # v1: ring's closest to the spot
    self.at_spot_ever = z()
    self.touch2_ever = z()


def _state(env) -> _State:
  s = getattr(env, "_t2_state", None)
  if s is None or s.phase1.shape[0] != env.num_envs:
    s = _State(env.num_envs, env.device)
    env._t2_state = s
  return s


def reset_handover(env, env_ids: torch.Tensor | None = None, spot: bool = False) -> None:
  env._t2_spot = spot
  s = _state(env)
  ids = slice(None) if env_ids is None else env_ids
  s.phase1[ids] = False
  s.cleared1[ids] = False
  s.both_ever[ids] = False
  s.since_pads1[ids] = 10 ** 6
  s.prev_contact[ids] = False
  s.lift1_max[ids] = 0.0
  s.z_hand[ids] = 0.0
  s.release_z[ids] = 0.0
  s.goal_min[ids] = 1.0
  s.grip2_min[ids] = 1.0
  s.spot_min[ids] = 1.0
  s.at_spot_ever[ids] = False
  s.touch2_ever[ids] = False
  env._t2_cache = None


def terms(env) -> dict[str, torch.Tensor]:
  """Everything rewards, observations, terminations and metrics read, once per
  env step. Advances the phase and the latches."""
  key = int(getattr(env, "common_step_counter", -1))
  got = getattr(env, "_t2_cache", None)
  if got is not None and got[0] == key:
    return got[1]

  o = env.scene.env_origins
  tools = torch.stack([D.tool_pos(env, asset_name=a) for a in ARMS], dim=1)      # (n,2,3)
  blk = env.scene["block"]
  ring = blk.data.root_link_pos_w - o
  ring_q = blk.data.root_link_quat_w
  goal = env.scene["goal"].data.root_link_pos_w - o
  peg = env.scene["peg"].data.root_link_pos_w - o
  jaw = torch.stack([env.scene[a].data.joint_pos[:, env.scene[a].joint_names.index("jaw")]
                     for a in ARMS], dim=-1)                                       # (n,2)
  contact = T1._holding(env)                                                       # (n,2) pads on

  d_grip, near, grip_vec = [], [], []
  for k, a in enumerate(ARMS):
    P = S._grip_points(env, ring, ring_q, a)                                       # (n,2,3)
    dv = P - tools[:, k:k + 1]
    dist = torch.linalg.norm(dv, dim=-1)
    best = dist.argmin(dim=1)
    d_grip.append(dist.min(dim=1).values)
    near.append(((torch.linalg.norm(dv[..., :2], dim=-1) < GRIP_TOL_XY)
                 & (dv[..., 2].abs() < GRIP_TOL_Z)).any(dim=1))
    grip_vec.append(dv[torch.arange(dv.shape[0], device=env.device), best])
  d_grip = torch.stack(d_grip, -1)
  near = torch.stack(near, -1)
  held = contact & (jaw < JAW_CLOSED) & near
  h0, h1 = held[:, 0], held[:, 1]

  z = ring[:, 2]
  d_goal_xy = torch.linalg.norm(ring[:, :2] - goal[:, :2], dim=-1)
  on_goal = LR.block_at_goal(env)
  lift_frac = ((z - LR.Z_REST) / LR.LIFT_SPAN).clamp(0.0, 1.0)
  in_air = z >= AIR_Z

  q = ring_q
  tilt = torch.rad2deg(torch.acos((1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1.0, 1.0)))
  flat = tilt < FLAT_DEG

  s = _state(env)
  p1_prev = s.phase1.clone()

  # Releases: an arm's pads come off a raised ring the other arm is not holding.
  let_go = s.prev_contact & ~contact                                               # (n,2)
  rel_frac = ((z - RELEASE_Z) / RELEASE_SPAN).clamp(0.0, 1.0)
  released1 = (~p1_prev) & s.cleared1 & let_go[:, 0] & ~h1
  released2 = p1_prev & let_go[:, 1] & ~contact[:, 0]
  released = released1 | released2
  rel_cost = rel_frac * released.float()
  s.release_z = torch.where(released, z * 1000.0, s.release_z)

  # Handover: arm 2 alone, ring in the air, arm 1's pads on it moments ago.
  handed = (~p1_prev) & h1 & ~contact[:, 0] & in_air & (s.since_pads1 <= HAND_WINDOW)
  s.z_hand = torch.where(handed, z * 1000.0, s.z_hand)
  s.phase1 = s.phase1 | handed
  p1 = s.phase1

  s.since_pads1 = torch.where(contact[:, 0], torch.zeros_like(s.since_pads1), s.since_pads1 + 1)
  s.prev_contact = contact.clone()
  s.cleared1 = s.cleared1 | ((~p1) & h0 & (z >= CLEAR_Z))
  spot = bool(getattr(env, "_t2_spot", False))
  spot_pt = torch.cat([goal[:, :2], torch.full_like(z[:, None], SPOT_Z)], -1)
  d_spot = torch.linalg.norm(ring - spot_pt, dim=-1)
  at_spot = (~p1) & s.cleared1 & h0 & (d_spot < SPOT_R)
  touch2 = at_spot & contact[:, 1]
  both = (~p1) & h0 & h1 & in_air
  if spot:
    both = both & (d_spot < 2.0 * SPOT_R)   # gripping shifts the ring a little
  s.both_ever = s.both_ever | both

  resting = on_goal & flat
  success = p1 & resting & ~contact[:, 1]

  floor = z <= LR.DOWN_Z
  far = d_goal_xy > LR.GOAL_HALF + LR.LOST_MARGIN
  lost = ((s.cleared1 | p1) & floor & ~on_goal & far) | (z < -0.002)

  settled = env.episode_length_buf >= SETTLE_STEPS
  s.lift1_max = torch.where((~p1) & settled,
                            torch.maximum(s.lift1_max, (z - LR.Z_REST) * 1000.0), s.lift1_max)
  s.goal_min = torch.where(p1, torch.minimum(s.goal_min, d_goal_xy), s.goal_min)
  s.grip2_min = torch.minimum(s.grip2_min, d_grip[:, 1])
  s.spot_min = torch.where((~p1) & s.cleared1 & h0, torch.minimum(s.spot_min, d_spot), s.spot_min)
  s.at_spot_ever = s.at_spot_ever | at_spot
  s.touch2_ever = s.touch2_ever | touch2

  # --- the potential ---
  reach0 = 1.0 - torch.tanh(REACH_K * d_grip[:, 0])
  lift0 = P0_LIFT + LIFT_GAIN * lift_frac
  reach2 = P0_REACH2 + 1.0 - torch.tanh(REACH_K * d_grip[:, 1])
  phi0 = torch.where(h0, lift0, reach0)
  phi0 = torch.where(s.cleared1 & h0 & in_air, reach2, phi0)
  phi0 = torch.where(both, torch.full_like(phi0, P0_BOTH), phi0)

  regrasp = P1_BASE + 1.0 - torch.tanh(REACH_K * d_grip[:, 1])
  carry1 = P1_CARRY + CARRY_GAIN * (1.0 - torch.tanh(GOAL_K * d_goal_xy))
  down1 = ((CLEAR_Z - z) / (CLEAR_Z - LR.DOWN_Z)).clamp(0.0, 1.0)
  place1 = P1_PLACE + PLACE_GAIN * down1
  phi1 = torch.where(h1, carry1, regrasp)
  phi1 = torch.where(h1 & (d_goal_xy < LR.GOAL_HALF), place1, phi1)
  phi1 = torch.where(resting, torch.full_like(phi1, P1_ON_GOAL), phi1)

  if spot:
    # v1: arm 1 brings the ring to the spot; arm 2's stages only count there.
    present = (V1_PRESENT + (PRESENT_GAIN - 1.0) * (1.0 - torch.tanh(GOAL_K * d_spot))
               + (1.0 - torch.tanh(d_spot / SPOT_STD)))
    reach2s = (V1_REACH2 + (REACH2_GAIN - 1.0) * (1.0 - torch.tanh(REACH_K * d_grip[:, 1]))
               + (1.0 - torch.tanh(d_grip[:, 1] / REACH2_STD)))
    phi0 = torch.where(h0, lift0, reach0)
    phi0 = torch.where(s.cleared1 & h0, present, phi0)
    phi0 = torch.where(at_spot, reach2s, phi0)
    phi0 = torch.where(touch2, torch.full_like(phi0, V1_TOUCH2), phi0)
    phi0 = torch.where(both, torch.full_like(phi0, V1_BOTH), phi0)

    regrasp = V1_P1_BASE + 1.0 - torch.tanh(REACH_K * d_grip[:, 1])
    carry1 = V1_P1_CARRY + SPOT_CARRY_GAIN * (1.0 - torch.tanh(GOAL_K * d_goal_xy))
    place1 = V1_P1_PLACE + PLACE_GAIN * down1
    phi1 = torch.where(h1, carry1, regrasp)
    phi1 = torch.where(h1 & (d_goal_xy < LR.GOAL_HALF), place1, phi1)
    phi1 = torch.where(resting, torch.full_like(phi1, V1_P1_ON_GOAL), phi1)

  phi = torch.where(p1, phi1, phi0)

  # Observation waypoint: up off the post (v0) or the spot (v1) in phase 0, the
  # goal in phase 1.
  wp0 = spot_pt if spot else torch.cat([peg[:, :2], torch.full_like(z[:, None], CLEAR_Z)], -1)
  target = torch.where(p1[:, None],
                       torch.cat([goal[:, :2], torch.full_like(z[:, None], LR.Z_REST)], -1),
                       wp0)
  out = {
    "tools": tools, "ring": ring, "ring_q": ring_q, "goal": goal, "peg": peg, "jaw": jaw,
    "contact": contact, "held": held, "d_grip": d_grip,
    "grip_vec": torch.stack(grip_vec, 1), "lift_frac": lift_frac,
    "on_goal": on_goal, "phase1": p1, "handed": handed, "both": both,
    "success": success, "lost": lost, "phi": phi, "target": target,
    "cleared": s.cleared1, "in_air": in_air, "d_goal_xy": d_goal_xy,
    "tilt": tilt, "released": released, "rel_cost": rel_cost,
    "d_spot": d_spot, "at_spot": at_spot, "touch2": touch2,
  }
  env._t2_cache = (key, out)
  return out


# --- observation ---------------------------------------------------------------------

def obs_handover(env) -> torch.Tensor:
  """38-D, Task 1's layout: both tools, ring pos + quat, goal, peg, grip point -
  tool per arm, waypoint - ring, phase, held (strict) and pad contact per arm,
  jaw per arm, lift fraction, cleared latch, ring in the air."""
  t = terms(env)
  return torch.cat([
    t["tools"].flatten(1), t["ring"], t["ring_q"], t["goal"], t["peg"],
    t["grip_vec"].flatten(1), t["target"] - t["ring"],
    t["phase1"].float()[:, None], t["held"].float(), t["contact"].float(), t["jaw"],
    t["lift_frac"][:, None], t["cleared"].float()[:, None], t["in_air"].float()[:, None],
  ], dim=-1)


# --- rewards ---------------------------------------------------------------------------

class shaping:
  """PHI(s') - PHI(s). Zero on the first step after a reset."""

  def __init__(self, cfg, env):
    self._prev = torch.zeros(env.num_envs, device=env.device)
    self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids=None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True

  def __call__(self, env) -> torch.Tensor:
    phi = terms(env)["phi"]
    delta = torch.where(self._fresh, torch.zeros_like(phi), phi - self._prev)
    self._prev = phi.clone()
    self._fresh = torch.zeros_like(self._fresh)
    return delta


def r_success(env):
  return terms(env)["success"].float()


def r_release(env):
  return terms(env)["rel_cost"]


def r_lost(env):
  return terms(env)["lost"].float()


def t_success(env):
  return terms(env)["success"]


def t_lost(env):
  return terms(env)["lost"]


# --- metrics -------------------------------------------------------------------------------

def m_key(env, key: str, col: int | None = None):
  v = terms(env)[key]
  return (v if col is None else v[:, col]).float()


def m_state(env, attr: str):
  terms(env)
  return getattr(_state(env), attr).float()


def m_goal_mm(env):
  terms(env)
  return _state(env).goal_min * 1000.0


def m_spot_mm(env):
  terms(env)
  return _state(env).spot_min * 1000.0


def m_grip2_mm(env):
  # Closest so far, tracked here: mjlab's metric reductions are mean/last/max.
  terms(env)
  return _state(env).grip2_min * 1000.0


_RING_GEOMS: dict = {}


def m_ring_friction(env):
  """The ring's sliding friction this episode (domain-randomised per reset), to
  check afterwards whether failures cluster on slippery rings."""
  key = id(env.sim.mj_model)
  if key not in _RING_GEOMS:
    import mujoco

    m = env.sim.mj_model
    ids = [i for i in range(m.ngeom)
           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("block/")]
    _RING_GEOMS[key] = torch.tensor(ids, device=env.device)
  f = env.sim.model.geom_friction
  ids = _RING_GEOMS[key]
  if f.dim() == 3:
    return f[:, ids, 0].mean(dim=1)
  return f[ids, 0].mean().expand(env.num_envs)


def dual_task2_handover_env_cfg(play: bool = False, spot: bool = False):
  """spot=False: v0 (Surgicallab-Dual-Task2-Handover). spot=True: v1, the
  exchange at a fixed point above the goal (Surgicallab-Dual-Task2-Spot)."""
  cfg = D.psm_lift_peg_dual_env_cfg(play=play, reward="e_shaped", state_only=True)
  cfg.episode_length_s = EPISODE_S
  dt = cfg.sim.mujoco.timestep * cfg.decimation

  cfg.events["reset_task2_handover"] = EventTermCfg(mode="reset", func=reset_handover,
                                                        params={"spot": spot})

  obs = cfg.observations["critic"].terms
  obs.pop("state", None)
  obs.pop("tool_pos_2", None)
  obs["task2"] = ObservationTermCfg(func=obs_handover)

  # The reward manager multiplies by dt; dividing keeps PHI and the bonuses in
  # return units.
  cfg.rewards = {
    "shaping": RewardTermCfg(func=shaping, weight=W_SHAPE / dt),
    "success": RewardTermCfg(func=r_success, weight=SUCCESS_BONUS / dt),
    "lost": RewardTermCfg(func=r_lost, weight=-LOST_PENALTY / dt),
  }
  if RELEASE_PEN > 0:
    cfg.rewards["release"] = RewardTermCfg(func=r_release, weight=-RELEASE_PEN / dt)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    **({"nan_state": cfg.terminations["nan_state"]}
       if "nan_state" in cfg.terminations else {}),
    "success": TerminationTermCfg(func=t_success),
    "lost": TerminationTermCfg(func=t_lost),
  }
  M = MetricsTermCfg
  cfg.metrics = {
    # Progress in one number: highest PHI reached (stage table above).
    "phi_max": M(func=m_key, params={"key": "phi"}, reduce="max"),
    "both_rate": M(func=m_state, params={"attr": "both_ever"}, reduce="last"),
    "handoff_rate": M(func=m_key, params={"key": "phase1"}, reduce="max"),
    "success_rate": M(func=m_key, params={"key": "success"}, reduce="max"),
    "lost_rate": M(func=m_key, params={"key": "lost"}, reduce="max"),
    "cleared_1_rate": M(func=m_state, params={"attr": "cleared1"}, reduce="last"),
    "held_1_frac": M(func=m_key, params={"key": "held", "col": 0}, reduce="mean"),
    "held_2_frac": M(func=m_key, params={"key": "held", "col": 1}, reduce="mean"),
    "lift_1_mm": M(func=m_state, params={"attr": "lift1_max"}, reduce="last"),
    "hand_z_mm": M(func=m_state, params={"attr": "z_hand"}, reduce="last"),
    "release_mm": M(func=m_state, params={"attr": "release_z"}, reduce="last"),
    "ring_goal_mm": M(func=m_goal_mm, reduce="last"),       # phase 1 closest; 1000 = never
    "grip_2_mm": M(func=m_grip2_mm, reduce="last"),
    "tilt_deg": M(func=m_key, params={"key": "tilt"}, reduce="last"),
    "ring_friction": M(func=m_ring_friction, reduce="last"),
  }
  if spot:
    cfg.metrics["spot_mm"] = M(func=m_spot_mm, reduce="last")      # 1000 = never cleared
    cfg.metrics["at_spot_rate"] = M(func=m_state, params={"attr": "at_spot_ever"}, reduce="last")
    cfg.metrics["touch_2_rate"] = M(func=m_state, params={"attr": "touch2_ever"}, reduce="last")
  D.add_arm_contact_safety(cfg)
  return cfg
