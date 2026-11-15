"""The Cartesian ONNX metadata is a hardware deployment contract: a real PSM driven
with the wrong scale, rate, joint order or DH constants will not reproduce sim
behaviour. These tests pin the fields a deployment must have."""

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import surgical_lab.tasks  # noqa: F401  (registers tasks)
from surgical_lab.rl.exporter_utils import get_cartesian_metadata

TASK = "Surgicallab-Push-Block-PSM-Cartesian"

REQUIRED_KEYS = (
  "policy_type",
  "control_dt",
  "action_dim",
  "action_pos_scale",
  "action_max_target_lead",
  "target_integration",
  "arm_joint_names",
  "ik_solver",
  "fk_solver",
  "l_rcc",
  "l_tool",
  "l_yaw2ctrlpnt",
  "l_pitch2yaw",
  "observation_names",
  "joint_pos_lower_limits",
  "joint_pos_upper_limits",
)


@pytest.fixture(scope="module")
def env():
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device=device)
  yield env
  env.close()


@pytest.fixture(scope="module")
def metadata(env):
  return get_cartesian_metadata(env, run_path="test")


def test_required_keys_present(metadata):
  missing = [k for k in REQUIRED_KEYS if k not in metadata]
  assert not missing, f"metadata missing deployment fields: {missing}"


def test_joint_order_matches_ik_chain(metadata):
  """FK/IK index joints positionally, so this order is load-bearing."""
  assert metadata["arm_joint_names"] == [
    "outer_yaw",
    "outer_pitch",
    "main_insertion",
    "tool_roll",
    "tool_pitch",
    "tool_yaw",
  ]


def test_action_contract_matches_env(env, metadata):
  term = env.action_manager.get_term("cartesian_delta")
  assert metadata["action_dim"] == term.action_dim
  assert metadata["action_pos_scale"] == term.cfg.pos_scale
  assert metadata["action_max_target_lead"] == term.cfg.max_target_lead
  # The delta is per control step, not per physics step. If these ever disagree, the
  # action term is integrating in the decimation loop again.
  assert metadata["control_dt"] == pytest.approx(
    env.cfg.sim.mujoco.timestep * env.cfg.decimation
  )


def test_kinematic_constants_match_solver(metadata):
  """The exported constants must be the ones the trained FK/IK actually used."""
  import inspect

  from surgical_lab.robots.psm.psm_kinematics import compute_psm_fk

  defaults = inspect.signature(compute_psm_fk).parameters
  for name in ("l_rcc", "l_tool", "l_yaw2ctrlpnt", "l_pitch2yaw"):
    assert metadata[name] == pytest.approx(defaults[name].default)


def test_observation_layout_matches_policy_input(env, metadata):
  """Observation names must be in the order the policy's input vector is built."""
  assert metadata["observation_names"] == env.observation_manager.active_terms["actor"]
  assert len(metadata["observation_terms_scale"]) == len(metadata["observation_names"])
