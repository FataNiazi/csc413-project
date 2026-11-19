"""Task 1, reward v1 ("staged"): the lab's single-arm staged reward, per arm.

Why v0 failed (run 1xcaxun9, 1160 iterations): arm 1 parked its tool on the
ring at ~iteration 300 and stayed there. Two reasons:

  * `reach` paid every step for sitting near the ring, forever.
  * `move` measured the ring against the goal AT TABLE HEIGHT, so lifting the
    ring off the 30 mm post moved it AWAY from the target and lowered the
    reward. The one thing that had to happen first was punished.

v1 uses the structure that already works on the single-arm peg lift
(`liftpeg_shaped.arm_e_shaped`, robosuite's nested stages): a potential PHI
that only goes up as the task progresses, paid as its change each step
(PHI(s') - PHI(s)). Sitting still earns nothing; every stage is worth more than
the best value of the stage before it, so moving on always pays.

  phase 0, arm 1 (lift the ring off the peg, set it on the goal)
    reach    0 .. 1      tool to the jaw-aligned grip point on the ring
    lift     1 .. 7      held, ring height up to post clearance (38 mm)
    carry    7 .. 10     held and cleared, ring towards the goal (xy)
    place   10 .. 13     cleared and over the goal, ring down
  handoff: ring touching the goal, arm 1 let go -> phase 1 (base 14 > 13)
  phase 1, arm 2 (pick it off the goal, put it back on the peg)
    reach   14 .. 15     arm 2's tool to its grip point
    lift    15 .. 21     held by arm 2, ring height up to 38 mm
    carry   21 .. 25     held and lifted, ring over the post (coarse + fine)
    insert  25 .. 28     ring centred over the post (< 4.4 mm), ring down
    seated  29           ring resting on the peg
  success: seated and arm 2 let go -> +SUCCESS_BONUS, episode ends
  lost:    ring dropped on the floor away from goal/peg -> -LOST_PENALTY, ends
  contact: any arm-arm contact -> -10, ends (unchanged)

"Held" is the strict single-arm grasp, per arm: both of THAT arm's pads in
contact with the ring, jaw closed, and the tool at a grip point. Contact alone
can be satisfied by resting the jaw on the ring's rim (the lab hit exactly
that in single-arm training).

PHI is in return units: a full success is worth 29 + SUCCESS_BONUS = 34, the
contact penalty 10. Every number is an env var (T1S_*).
"""

import math
import os

import torch
from mjlab.envs import mdp as base_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from surgical_lab.tasks.lift.config.psm import dual_task1 as T1
from surgical_lab.tasks.lift.config.psm import liftpeg_dual as D
from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as LR
from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS
from surgical_lab.tasks.lift.config.psm import liftpeg_shaped as SH
from surgical_lab.tasks.lift.config.psm.liftpeg_grasp import quat_apply

_f = T1._f
ARMS = T1.ARMS

# --- stage sizes (arm E's, plus phase 1) ---------------------------------------
REACH_K = 12.0
GOAL_K = 6.0
PEG_K = 6.0
LIFT_GAIN = _f("T1S_LIFT_GAIN", "6")
CARRY_GAIN = _f("T1S_CARRY_GAIN", "3")
PLACE_GAIN = _f("T1S_PLACE_GAIN", "3")
ALIGN_GAIN = _f("T1S_ALIGN_GAIN", "1")
ALIGN_STD = _f("T1S_ALIGN_STD_M", "0.004")     # fine term: the bore has 3.4 mm of play
INSERT_GAIN = _f("T1S_INSERT_GAIN", "3")
INSERT_XY = LS.RING_CLEARANCE + 0.001         # centred enough to go down the post

P0_CEIL = 1.0 + LIFT_GAIN + CARRY_GAIN + PLACE_GAIN          # 13
P1_BASE = P0_CEIL + 1.0                                     # 14
P1_LIFT = P1_BASE + 1.0                                     # 15
P1_CARRY = P1_LIFT + LIFT_GAIN                              # 21
P1_INSERT = P1_CARRY + CARRY_GAIN + ALIGN_GAIN              # 25
P1_SEATED = P1_INSERT + INSERT_GAIN + 1.0                   # 29

W_SHAPE = _f("T1S_W_SHAPE", "1.0")
SUCCESS_BONUS = _f("T1S_SUCCESS_BONUS", "5")
LOST_PENALTY = _f("T1S_LOST_PENALTY", "2")

GRIP_TOL_XY = SH.GRIP_TOL_XY
GRIP_TOL_Z = SH.GRIP_TOL_Z
JAW_CLOSED = SH.JAW_CLOSED
CLEAR_Z = LR.CLEAR_Z
SETTLE_STEPS = LR.SETTLE_STEPS


# --- per-arm grip geometry -------------------------------------------------------

def _jaw_axis(env, arm: str) -> torch.Tensor:
  """World-xy closing axis of one arm's jaw, from its own pad geoms (fixed:
  tool orientation is not actuated). liftpeg_shaped._jaw_axis pools every pad
  in the scene, which with two arms averages them together."""
  cache = getattr(env, "_t1s_jaw_axis", None)
  if cache is None:
    cache = env._t1s_jaw_axis = {}
  if arm not in cache:
    import mujoco

    m = env.sim.mj_model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    lg = [i for i, n in enumerate(names) if n.startswith(f"{arm}/lg") and "_collision" in n]
    rg = [i for i, n in enumerate(names) if n.startswith(f"{arm}/rg") and "_collision" in n]
    gx = env.sim.data.geom_xpos[0]
    v = (gx[torch.tensor(rg, device=env.device)].mean(0)
         - gx[torch.tensor(lg, device=env.device)].mean(0))[:2]
    cache[arm] = v / torch.norm(v)
  return cache[arm]


def _grip_points(env, ring, ring_q, arm: str) -> torch.Tensor:
  """(n, 2, 3): the two places the ring wall crosses this arm's jaw axis, at
  mid-wall and the teacher's grip height, rotated by the ring's FULL
  orientation (a held ring hangs tilted). Same construction as the scripted
  demo's grasp_point, vectorised."""
  tab, bore = SH._radius_table(env)
  ax = _jaw_axis(env, arm)
  jaw_ang = float(torch.atan2(ax[1], ax[0]))
  q = ring_q
  yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  K = tab.shape[0]
  out = []
  for flip in (0.0, math.pi):
    la = jaw_ang + flip - yaw
    idx = ((la % (2 * math.pi)) / (2 * math.pi) * K).long() % K
    r = bore + (tab[idx] - bore).clamp_min(1e-4) * SH.GRIP_FRAC
    local = torch.stack([r * torch.cos(la), r * torch.sin(la),
                         torch.full_like(la, SH.GRASP_LOCAL_Z)], dim=-1)
    out.append(ring + quat_apply(q, local))
  return torch.stack(out, dim=1)


# --- state ---------------------------------------------------------------------------

class _State:
  def __init__(self, n, dev, peg_goal_d):
    z = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    self.phase1 = z()
    self.cleared1 = z()
    self.cleared2 = z()
    self.lift1_max = torch.zeros(n, device=dev)
    self.z2_max = torch.zeros(n, device=dev)
    self.peg_min = torch.full((n,), peg_goal_d, device=dev)
    self.peg_goal_d = peg_goal_d


def _state(env) -> _State:
  s = getattr(env, "_t1s_state", None)
  if s is None or s.phase1.shape[0] != env.num_envs:
    s = _State(env.num_envs, env.device, 0.054)
    env._t1s_state = s
  return s


def reset_staged(env, env_ids: torch.Tensor | None = None) -> None:
  s = _state(env)
  ids = slice(None) if env_ids is None else env_ids
  s.phase1[ids] = False
  s.cleared1[ids] = False
  s.cleared2[ids] = False
  s.lift1_max[ids] = 0.0
  s.z2_max[ids] = 0.0
  s.peg_min[ids] = s.peg_goal_d
  env._t1s_cache = None


def terms(env) -> dict[str, torch.Tensor]:
  """Everything rewards, observations, terminations and metrics read, once per
  env step. Advances the phase and the `cleared` latches."""
  key = int(getattr(env, "common_step_counter", -1))
  got = getattr(env, "_t1s_cache", None)
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
  contact = T1._holding(env)                                                       # (n,2)

  d_grip, near, grip_vec = [], [], []
  for k, a in enumerate(ARMS):
    P = _grip_points(env, ring, ring_q, a)                                         # (n,2,3)
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

  z = ring[:, 2]
  d_goal_xy = torch.linalg.norm(ring[:, :2] - goal[:, :2], dim=-1)
  d_peg_xy = torch.linalg.norm(ring[:, :2] - peg[:, :2], dim=-1)
  on_goal = LR.block_at_goal(env)
  on_peg = (z < LR.Z_REST + T1.ON_PEG_DZ) & (d_peg_xy < T1.ON_PEG_XY)
  lift_frac = ((z - LR.Z_REST) / LR.LIFT_SPAN).clamp(0.0, 1.0)

  s = _state(env)
  handed = (~s.phase1) & on_goal & ~contact[:, 0]
  s.phase1 = s.phase1 | handed
  p1 = s.phase1
  s.cleared1 = s.cleared1 | ((~p1) & (z >= CLEAR_Z))
  s.cleared2 = s.cleared2 | (p1 & held[:, 1] & (z >= CLEAR_Z))
  success = p1 & on_peg & ~contact[:, 1]

  floor = z <= LR.DOWN_Z
  lost = (((~p1) & s.cleared1 & floor & ~on_goal
           & (d_goal_xy > LR.GOAL_HALF + LR.LOST_MARGIN))
          | (p1 & s.cleared2 & floor & ~on_goal)
          | (z < -0.002))

  settled = env.episode_length_buf >= SETTLE_STEPS
  s.lift1_max = torch.where((~p1) & settled,
                            torch.maximum(s.lift1_max, (z - LR.Z_REST) * 1000.0), s.lift1_max)
  s.z2_max = torch.where(p1, torch.maximum(s.z2_max, z * 1000.0), s.z2_max)
  s.peg_min = torch.where(p1, torch.minimum(s.peg_min, d_peg_xy), s.peg_min)

  # --- the potential ---
  h0, h1 = held[:, 0], held[:, 1]
  reach0 = 1.0 - torch.tanh(REACH_K * d_grip[:, 0])
  lift0 = 1.0 + LIFT_GAIN * lift_frac
  carry0 = 1.0 + LIFT_GAIN + CARRY_GAIN * (1.0 - torch.tanh(GOAL_K * d_goal_xy))
  down0 = ((CLEAR_Z - z) / (CLEAR_Z - LR.DOWN_Z)).clamp(0.0, 1.0)
  place0 = 1.0 + LIFT_GAIN + CARRY_GAIN + PLACE_GAIN * down0
  phi0 = torch.where(h0, lift0, reach0)
  phi0 = torch.where(s.cleared1 & h0, carry0, phi0)
  phi0 = torch.where(s.cleared1 & (d_goal_xy < LR.GOAL_HALF), place0, phi0)

  reach1 = P1_BASE + 1.0 - torch.tanh(REACH_K * d_grip[:, 1])
  lift1 = P1_LIFT + LIFT_GAIN * lift_frac
  carry1 = (P1_CARRY + CARRY_GAIN * (1.0 - torch.tanh(PEG_K * d_peg_xy))
            + ALIGN_GAIN * (1.0 - torch.tanh(d_peg_xy / ALIGN_STD)))
  down1 = ((CLEAR_Z - z) / (CLEAR_Z - LR.Z_REST)).clamp(0.0, 1.0)
  insert1 = P1_INSERT + INSERT_GAIN * down1
  phi1 = torch.where(h1, lift1, reach1)
  phi1 = torch.where(s.cleared2 & h1, carry1, phi1)
  phi1 = torch.where(s.cleared2 & (d_peg_xy < INSERT_XY), insert1, phi1)
  phi1 = torch.where(s.cleared2 & on_peg, torch.full_like(phi1, P1_SEATED), phi1)

  phi = torch.where(p1, phi1, phi0)

  target = torch.where(p1[:, None],
                       torch.cat([peg[:, :2], torch.full_like(z[:, None], CLEAR_Z)], -1),
                       torch.cat([goal[:, :2], torch.full_like(z[:, None], LR.Z_REST)], -1))
  out = {
    "tools": tools, "ring": ring, "ring_q": ring_q, "goal": goal, "peg": peg, "jaw": jaw,
    "contact": contact, "held": held, "d_grip": d_grip,
    "grip_vec": torch.stack(grip_vec, 1), "lift_frac": lift_frac,
    "on_goal": on_goal, "on_peg": on_peg, "phase1": p1, "handed": handed,
    "success": success, "lost": lost, "phi": phi, "target": target,
    "cleared": torch.stack([s.cleared1, s.cleared2], -1),
    "d_goal_xy": d_goal_xy, "d_peg_xy": d_peg_xy,
  }
  env._t1s_cache = (key, out)
  return out


# --- observation ---------------------------------------------------------------------

def obs_staged(env) -> torch.Tensor:
  """38-D: both tools, ring pos + quat, goal, peg, grip point - tool per arm,
  current waypoint - ring, phase, held (strict) and pad contact per arm, jaw
  per arm, lift fraction, cleared latches."""
  t = terms(env)
  return torch.cat([
    t["tools"].flatten(1), t["ring"], t["ring_q"], t["goal"], t["peg"],
    t["grip_vec"].flatten(1), t["target"] - t["ring"],
    t["phase1"].float()[:, None], t["held"].float(), t["contact"].float(), t["jaw"],
    t["lift_frac"][:, None], t["cleared"].float(),
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


def m_lift1_mm(env):
  terms(env)
  return _state(env).lift1_max


def m_z2_mm(env):
  terms(env)
  return _state(env).z2_max


def m_peg_mm(env):
  terms(env)
  return _state(env).peg_min * 1000.0


def m_grip1_mm(env):
  return terms(env)["d_grip"][:, 0] * 1000.0


def m_ring_goal_mm(env):
  return terms(env)["d_goal_xy"] * 1000.0


def dual_task1_staged_env_cfg(play: bool = False):
  cfg = D.psm_lift_peg_dual_env_cfg(play=play, reward="e_shaped", state_only=True)
  cfg.episode_length_s = T1.EPISODE_S
  dt = cfg.sim.mujoco.timestep * cfg.decimation

  cfg.events["reset_task1_staged"] = EventTermCfg(mode="reset", func=reset_staged)

  obs = cfg.observations["critic"].terms
  obs.pop("state", None)
  obs.pop("tool_pos_2", None)
  obs["task1"] = ObservationTermCfg(func=obs_staged)

  # The reward manager multiplies by dt; dividing here keeps PHI and the
  # bonuses in return units.
  cfg.rewards = {
    "shaping": RewardTermCfg(func=shaping, weight=W_SHAPE / dt),
    "success": RewardTermCfg(func=r_success, weight=SUCCESS_BONUS / dt),
    "lost": RewardTermCfg(func=r_lost, weight=-LOST_PENALTY / dt),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    **({"nan_state": cfg.terminations["nan_state"]}
       if "nan_state" in cfg.terminations else {}),
    "success": TerminationTermCfg(func=t_success),
    "lost": TerminationTermCfg(func=t_lost),
  }
  M = MetricsTermCfg
  cfg.metrics = {
    # Progress in one number: highest PHI reached (stage table in the docstring).
    "phi_max": M(func=m_key, params={"key": "phi"}, reduce="max"),
    "handoff_rate": M(func=m_key, params={"key": "phase1"}, reduce="max"),
    "success_rate": M(func=m_key, params={"key": "success"}, reduce="max"),
    "lost_rate": M(func=m_key, params={"key": "lost"}, reduce="max"),
    "held_1_frac": M(func=m_key, params={"key": "held", "col": 0}, reduce="mean"),
    "held_2_frac": M(func=m_key, params={"key": "held", "col": 1}, reduce="mean"),
    "contact_1_frac": M(func=m_key, params={"key": "contact", "col": 0}, reduce="mean"),
    "cleared_1_rate": M(func=m_key, params={"key": "cleared", "col": 0}, reduce="last"),
    "cleared_2_rate": M(func=m_key, params={"key": "cleared", "col": 1}, reduce="last"),
    "lift_1_mm": M(func=m_lift1_mm, reduce="last"),       # 32 = off the post
    "ring_z_2_mm": M(func=m_z2_mm, reduce="last"),        # phase 1, absolute; 38 = clears post
    "ring_goal_mm": M(func=m_ring_goal_mm, reduce="min"),
    "ring_peg_2_mm": M(func=m_peg_mm, reduce="last"),     # phase 1 closest; 54 = never got there
    "grip_1_mm": M(func=m_grip1_mm, reduce="min"),
  }
  D.add_arm_contact_safety(cfg)
  return cfg
