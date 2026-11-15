"""With LIFTPEG_PROPRIO / LIFTPEG_FORCE_PENALTY unset, liftpeg is unchanged.

The three 2026-09 experiments (proprioception, proprio+DR, force minimisation)
each add terms to an env cfg that four registered tasks and every archived run
share. rsl-rl sizes the actor's 1-D input from the actor group's width and W&B
logs one row per reward and metric term, so a term that leaks in when its flag
is off does not fail loudly -- it silently makes a new run incomparable to the
old ones, which is the exact mistake `docs/liftpeg_reward_ablation.md` was
written to stop repeating.

This test pins the flags-off shape of the cfg so that cannot happen quietly.
It is a config-level test on purpose: building the full env needs a GPU and a
mesh directory, and the property under test is entirely in the cfg.
"""

import os

import pytest

# The baseline, captured from the cfg as it stood before the experiment arms
# were added (commit 'liftpeg: proprioception, force minimisation...').
ACTOR_TERMS = ["zeros"]
CRITIC_TERMS = ["state", "joint_pos", "actions"]
REWARD_TERMS = ["shaping", "success", "lost"]
METRIC_TERMS = [
  "success_rate", "reach_mm", "grasp_rate", "held_frac", "grip_mm",
  "nan_rows", "phased_rate", "lift_mm", "clear_rate", "goal_mm",
]


@pytest.fixture(scope="module")
def cfg():
  for flag in ("LIFTPEG_PROPRIO", "LIFTPEG_PROPRIO_DR", "LIFTPEG_FORCE_PENALTY"):
    os.environ.pop(flag, None)
  os.environ.setdefault("A3B_UNITY_LAYOUT", "1")
  import surgical_lab.tasks.lift.config.psm.liftpeg_a3b_env_cfg as LC

  return LC.psm_lift_peg_a3b_env_cfg(reward="e_shaped")


def test_actor_is_still_the_zeros_branch(cfg):
  """The actor stays camera-only. If this fails, `scripts/bake_zeros_input.py`
  has silently stopped applying to the exported ONNX."""
  assert list(cfg.observations["actor"].terms) == ACTOR_TERMS
  assert cfg.observations["actor"].terms["zeros"].params["dim"] == 3
  assert cfg.observations["actor"].terms["zeros"].noise is None
  assert cfg.observations["actor"].terms["zeros"].delay_max_lag == 0


def test_critic_group_unchanged(cfg):
  assert list(cfg.observations["critic"].terms) == CRITIC_TERMS


def test_reward_terms_unchanged(cfg):
  assert list(cfg.rewards) == REWARD_TERMS


def test_metric_terms_unchanged(cfg):
  assert list(cfg.metrics) == METRIC_TERMS


def test_no_experiment_events(cfg):
  assert "reset_home" not in cfg.events


def test_flags_default_off():
  from surgical_lab.tasks.lift.config.psm import liftpeg_force as LF
  from surgical_lab.tasks.lift.config.psm import liftpeg_proprio as LP

  assert LP.PROPRIO is False
  assert LP.PROPRIO_DR is False
  assert LF.FORCE_PENALTY == 0.0
