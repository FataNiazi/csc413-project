"""Task 1 on the two-PSM scene, with the most naive reward that describes it.

Task 1 (the relay):
  phase 0  arm 1 lifts the ring off the peg, sets it on the red goal, lets go;
  phase 1  arm 2 picks it up off the goal and puts it back on the peg, lets go.
  Throughout: the two arms must never touch.

Reward v0 ("naive") -- deliberately the simplest thing that points at the task,
as a baseline to see how far plain PPO gets before any task-specific shaping:

  reach    1 - tanh(|ring - tool_active| / 2 cm)       active arm = arm 1 in phase 0,
                                                       arm 2 in phase 1
  move     1 - tanh(|ring - target| / 5 cm)            target = goal in phase 0,
                                                       resting on the peg in phase 1
  handoff  +5 once, when phase 0 completes             (ring touching the goal, arm 1 not holding it)
  success  +10 once, episode ends                      (phase 1, ring resting on the peg,
                                                        arm 2 not holding it)
  contact  -10, episode ends                           any arm-arm contact (whole arms)

No grasp, lift, carry-height or release shaping: the policy has to discover on
its own that the ring must be grasped and lifted over the 30 mm post. Every
number above is an env var (T1_*), so variants need no code change.

"Holding" is per arm (both of THAT arm's jaw pads in contact with the ring).
The single-arm helper in liftpeg_grasp pools every jaw geom in the scene, so
with two arms it would count arm 1's left jaw + arm 2's right jaw as a grasp.
"""

import os

import torch
from mjlab.envs import mdp as base_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from surgical_lab.tasks.lift.config.psm import liftpeg_dual as D
from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as LR
from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS


def _f(name, default):
  return float(os.environ.get(name, default))


EPISODE_S = _f("T1_EPISODE_S", "40")      # 2000 steps; the scripted demo needs ~1500
REACH_STD = _f("T1_REACH_STD_M", "0.02")
MOVE_STD = _f("T1_MOVE_STD_M", "0.05")
W_REACH = _f("T1_W_REACH", "1.0")
W_MOVE = _f("T1_W_MOVE", "2.0")
HANDOFF_BONUS = _f("T1_HANDOFF_BONUS", "5")
SUCCESS_BONUS = _f("T1_SUCCESS_BONUS", "10")
# "Resting on the peg": low, and the bore centred on the post.
ON_PEG_DZ = _f("T1_ON_PEG_DZ_M", "0.003")
ON_PEG_XY = LS.RING_CLEARANCE + 0.0005

ARMS = ("robot", "robot2")


# --- per-arm grasp ------------------------------------------------------------

_JAWS: dict = {}


def _jaw_sets(env):
  key = id(env.sim.mj_model)
  if key not in _JAWS:
    import mujoco

    m = env.sim.mj_model
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    dev = env.device

    def ids(pred):
      return torch.tensor([i for i, n in enumerate(names) if pred(n)], dtype=torch.long,
                          device=dev)

    block = ids(lambda n: n.startswith("block/") and "bar" in n)
    sets = {}
    for arm in ARMS:
      sets[arm] = (ids(lambda n, a=arm: n.startswith(f"{a}/lg") and n.endswith("_collision")),
                   ids(lambda n, a=arm: n.startswith(f"{a}/rg") and n.endswith("_collision")))
    _JAWS[key] = (sets, block)
  return _JAWS[key]


def _holding(env) -> torch.Tensor:
  """(num_envs, 2) bool: both jaw pads of arm k in penetrating contact with the ring."""
  sets, block = _jaw_sets(env)
  out = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
  d = env.sim.data
  n = int(d.nacon[0])
  if n == 0:
    return out
  g = d.contact.geom[:n]
  g1, g2 = g[:, 0].long(), g[:, 1].long()
  wid = d.contact.worldid[:n].long()
  active = d.contact.dist[:n] < 0.0
  on_ring1, on_ring2 = torch.isin(g2, block), torch.isin(g1, block)
  for k, arm in enumerate(ARMS):
    left, right = sets[arm]
    hl = ((torch.isin(g1, left) & on_ring1) | (torch.isin(g2, left) & on_ring2)) & active
    hr = ((torch.isin(g1, right) & on_ring1) | (torch.isin(g2, right) & on_ring2)) & active
    wl = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    wr = torch.zeros_like(wl)
    wl[wid[hl]] = True
    wr[wid[hr]] = True
    out[:, k] = wl & wr
  return out


# --- state, computed once per step -------------------------------------------

class _State:
  def __init__(self, n, dev):
    self.phase = torch.zeros(n, dtype=torch.long, device=dev)
    self.just_handed = torch.zeros(n, dtype=torch.bool, device=dev)
    self.ring_max_z = torch.zeros(n, device=dev)


def _state(env) -> _State:
  s = getattr(env, "_t1_state", None)
  if s is None or s.phase.shape[0] != env.num_envs:
    s = _State(env.num_envs, env.device)
    env._t1_state = s
  return s


def reset_task1(env, env_ids: torch.Tensor | None = None) -> None:
  s = _state(env)
  ids = slice(None) if env_ids is None else env_ids
  s.phase[ids] = 0
  s.just_handed[ids] = False
  s.ring_max_z[ids] = 0.0
  env._t1_cache = None


def terms(env) -> dict[str, torch.Tensor]:
  """Everything the reward, observation and metrics read. Also advances the
  phase latch, once per env step (keyed on the step counter)."""
  key = int(getattr(env, "common_step_counter", -1))
  got = getattr(env, "_t1_cache", None)
  if got is not None and got[0] == key:
    return got[1]

  o = env.scene.env_origins
  tools = torch.stack([D.tool_pos(env, asset_name=a) for a in ARMS], dim=1)     # (n,2,3)
  blk = env.scene["block"]
  ring = blk.data.root_link_pos_w - o
  ring_q = blk.data.root_link_quat_w
  goal = env.scene["goal"].data.root_link_pos_w - o
  peg = env.scene["peg"].data.root_link_pos_w - o
  hold = _holding(env)
  on_goal = LR.block_at_goal(env)
  d_peg_xy = torch.linalg.norm(ring[:, :2] - peg[:, :2], dim=-1)
  on_peg = (ring[:, 2] < LR.Z_REST + ON_PEG_DZ) & (d_peg_xy < ON_PEG_XY)

  s = _state(env)
  # phase 0 -> 1: ring delivered to the goal and arm 1 has let go of it.
  handed = (s.phase == 0) & on_goal & ~hold[:, 0]
  s.just_handed = handed
  s.phase = torch.where(handed, torch.ones_like(s.phase), s.phase)
  s.ring_max_z = torch.maximum(s.ring_max_z, ring[:, 2])
  phase1 = s.phase == 1
  success = phase1 & on_peg & ~hold[:, 1]

  target = torch.where(phase1[:, None],
                       torch.cat([peg[:, :2], torch.full_like(peg[:, :1], LR.Z_REST)], -1),
                       torch.cat([goal[:, :2], torch.full_like(goal[:, :1], LR.Z_REST)], -1))
  active_tool = torch.where(phase1[:, None], tools[:, 1], tools[:, 0])
  out = {
    "tools": tools, "ring": ring, "ring_q": ring_q, "goal": goal, "peg": peg,
    "hold": hold, "on_goal": on_goal, "on_peg": on_peg, "phase1": phase1,
    "handed": handed, "success": success, "target": target,
    "d_reach": torch.linalg.norm(ring - active_tool, dim=-1),
    "d_move": torch.linalg.norm(target - ring, dim=-1),
    "d_peg_xy": d_peg_xy,
  }
  env._t1_cache = (key, out)
  return out


# --- observation ---------------------------------------------------------------

def obs_task1(env) -> torch.Tensor:
  """33-D: both tools, ring pos + quat, goal, peg, ring - each tool, target -
  ring, phase, per-arm holding, per-arm jaw opening."""
  t = terms(env)
  jaws = []
  for a in ARMS:
    r = env.scene[a]
    jaws.append(r.data.joint_pos[:, r.joint_names.index("jaw")])
  return torch.cat([
    t["tools"].flatten(1), t["ring"], t["ring_q"], t["goal"], t["peg"],
    (t["ring"][:, None] - t["tools"]).flatten(1), t["target"] - t["ring"],
    t["phase1"].float()[:, None], t["hold"].float(), torch.stack(jaws, -1),
  ], dim=-1)


# --- rewards, terminations, metrics -----------------------------------------

def r_reach(env):
  return 1.0 - torch.tanh(terms(env)["d_reach"] / REACH_STD)


def r_move(env):
  return 1.0 - torch.tanh(terms(env)["d_move"] / MOVE_STD)


def r_handoff(env):
  return terms(env)["handed"].float()


def t_success(env):
  return terms(env)["success"]


def r_success(env):
  return terms(env)["success"].float()


def m_phase1(env):
  return terms(env)["phase1"].float()


def m_success(env):
  return terms(env)["success"].float()


def m_hold(env, arm: int):
  return terms(env)["hold"][:, arm].float()


def m_lift_mm(env):
  return (_state(env).ring_max_z - LR.Z_REST) * 1000.0


def m_ring_goal_mm(env):
  t = terms(env)
  return torch.linalg.norm(t["ring"][:, :2] - t["goal"][:, :2], dim=-1) * 1000.0


def dual_task1_env_cfg(play: bool = False):
  cfg = D.psm_lift_peg_dual_env_cfg(play=play, reward="e_shaped", state_only=True)
  cfg.episode_length_s = EPISODE_S
  dt = cfg.sim.mujoco.timestep * cfg.decimation

  cfg.events["reset_task1"] = EventTermCfg(mode="reset", func=reset_task1)

  terms_ = cfg.observations["critic"].terms
  terms_.pop("state", None)
  terms_.pop("tool_pos_2", None)
  terms_["task1"] = ObservationTermCfg(func=obs_task1)

  cfg.rewards = {
    "reach": RewardTermCfg(func=r_reach, weight=W_REACH),
    "move": RewardTermCfg(func=r_move, weight=W_MOVE),
    # One-shot events: the reward manager multiplies by dt, so divide here to
    # pay the bonus in return units.
    "handoff": RewardTermCfg(func=r_handoff, weight=HANDOFF_BONUS / dt),
    "success": RewardTermCfg(func=r_success, weight=SUCCESS_BONUS / dt),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    **({"nan_state": cfg.terminations["nan_state"]}
       if "nan_state" in cfg.terminations else {}),
    "success": TerminationTermCfg(func=t_success),
  }
  cfg.metrics = {
    # max over the episode: did it ever get there.
    "handoff_rate": MetricsTermCfg(func=m_phase1, reduce="max"),
    "success_rate": MetricsTermCfg(func=m_success, reduce="max"),
    "held_1_frac": MetricsTermCfg(func=m_hold, params={"arm": 0}, reduce="mean"),
    "held_2_frac": MetricsTermCfg(func=m_hold, params={"arm": 1}, reduce="mean"),
    "lift_mm": MetricsTermCfg(func=m_lift_mm, reduce="last"),
    "ring_goal_mm": MetricsTermCfg(func=m_ring_goal_mm, reduce="min"),
  }
  D.add_arm_contact_safety(cfg)
  return cfg
