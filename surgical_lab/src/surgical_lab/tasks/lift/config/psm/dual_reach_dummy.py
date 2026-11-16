"""Dummy objective for the two-PSM scene: each arm reaches its own random target.

This is a pipeline check, not a research task. It answers three questions
before anyone spends GPU hours on the real bimanual reward:

  1. Does PPO learn at all in the dual scene (reward goes up, distance goes down)?
  2. Do both arms' action slices reach the policy correctly (both distances fall)?
  3. Does training log to wandb, save checkpoints and resume?

Each reset samples one target per arm in a box around that arm's nominal tool
home (arm 1 above the peg, arm 2 40 mm to +x; both at ~69 mm). The boxes are
+-RANGE_XY in x/y and +-RANGE_Z in z and do not overlap, so the arms never
need to cross. The ring and peg are still in the scene but play no part; the
targets sit 20+ mm above the post top, so the jaws should not touch them.

Expected behaviour: dist_*_mm drops from ~15 mm to a few mm and reached_rate
climbs toward 1 within a few hundred iterations. If it does not, something in
the env/action pipeline is wrong, and that is worth knowing before tuning the
real reward.
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
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

ARMS = ("robot", "robot2")
HOME_Z = 0.069
CENTERS = (
  (a3b.ANCHOR_X, a3b.ANCHOR_Y, HOME_Z),
  (D.PSM2_TOOL_XY[0], D.PSM2_TOOL_XY[1], HOME_Z),
)
RANGE_XY = float(os.environ.get("DUMMY_RANGE_XY_M", "0.015"))
RANGE_Z = float(os.environ.get("DUMMY_RANGE_Z_M", "0.010"))
REACH_TOL = float(os.environ.get("DUMMY_REACH_TOL_M", "0.003"))
EPISODE_S = float(os.environ.get("DUMMY_EPISODE_S", "4.0"))


def _targets(env) -> torch.Tensor:
  """(num_envs, 2, 3) targets in the env frame, created on first use."""
  t = getattr(env, "_dummy_targets", None)
  if t is None:
    c = torch.tensor(CENTERS, device=env.device, dtype=torch.float32)
    t = c.unsqueeze(0).repeat(env.num_envs, 1, 1)
    env._dummy_targets = t
  return t


def resample_targets(env, env_ids: torch.Tensor | None = None) -> None:
  t = _targets(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  c = torch.tensor(CENTERS, device=env.device, dtype=torch.float32)
  half = torch.tensor((RANGE_XY, RANGE_XY, RANGE_Z), device=env.device)
  u = torch.rand(n, 2, 3, device=env.device) * 2.0 - 1.0
  t[env_ids] = c.unsqueeze(0) + u * half


def _tools(env) -> torch.Tensor:
  """(num_envs, 2, 3) tool midpoints in the env frame."""
  return torch.stack([D.tool_pos(env, asset_name=a) for a in ARMS], dim=1)


def _dist(env) -> torch.Tensor:
  """(num_envs, 2) tool-to-target distance per arm, metres."""
  return torch.linalg.norm(_targets(env) - _tools(env), dim=-1)


# --- observations -----------------------------------------------------------

def obs_reach_state(env) -> torch.Tensor:
  """Both tools, both targets and both target-minus-tool vectors (18-D)."""
  tools, tgt = _tools(env), _targets(env)
  return torch.cat(
    [tools.flatten(1), tgt.flatten(1), (tgt - tools).flatten(1)], dim=-1
  )


# --- rewards ----------------------------------------------------------------

def reach_reward(env, arm: int, std: float) -> torch.Tensor:
  return 1.0 - torch.tanh(_dist(env)[:, arm] / std)


# --- metrics ----------------------------------------------------------------

def m_dist_mm(env, arm: int) -> torch.Tensor:
  return _dist(env)[:, arm] * 1000.0


def m_reached(env) -> torch.Tensor:
  return (_dist(env) < REACH_TOL).all(dim=-1).float()


def dual_reach_dummy_env_cfg(play: bool = False):
  cfg = D.psm_lift_peg_dual_env_cfg(play=play, reward="e_shaped", state_only=True)
  cfg.episode_length_s = EPISODE_S

  cfg.events["dummy_targets"] = EventTermCfg(mode="reset", func=resample_targets)

  # Actor reads the critic group (see the state runner cfg), so this is what
  # both networks see: the reach state, both arms' joints, and the last action.
  terms = cfg.observations["critic"].terms
  terms.pop("state", None)
  terms.pop("tool_pos_2", None)
  terms["reach"] = ObservationTermCfg(func=obs_reach_state)

  cfg.rewards = {
    "reach_1": RewardTermCfg(func=reach_reward, weight=1.0,
                             params={"arm": 0, "std": 0.01}),
    "reach_2": RewardTermCfg(func=reach_reward, weight=1.0,
                             params={"arm": 1, "std": 0.01}),
    "fine_1": RewardTermCfg(func=reach_reward, weight=0.5,
                            params={"arm": 0, "std": 0.002}),
    "fine_2": RewardTermCfg(func=reach_reward, weight=0.5,
                            params={"arm": 1, "std": 0.002}),
    "action_rate": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.005),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    **({"nan_state": cfg.terminations["nan_state"]}
       if "nan_state" in cfg.terminations else {}),
  }
  cfg.metrics = {
    "dist_1_mm": MetricsTermCfg(func=m_dist_mm, params={"arm": 0}, reduce="last"),
    "dist_2_mm": MetricsTermCfg(func=m_dist_mm, params={"arm": 1}, reduce="last"),
    "reached_rate": MetricsTermCfg(func=m_reached, reduce="last"),
  }
  return cfg
