"""Tests for BinaryJointPositionAction (gripper control)."""

from unittest.mock import Mock

import mujoco
import pytest
import torch
from conftest import get_test_device
from mjlab.actuator.actuator import TransmissionType
from mjlab.actuator.builtin_actuator import BuiltinMotorActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim import Simulation, SimulationCfg

from surgical_lab.mdp import BinaryJointPositionActionCfg

NUM_ENVS = 4

GRIPPER_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.1 0.05" mass="5.0"/>
      <body name="hand" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="hand_geom" type="box" size="0.05 0.05 0.02" mass="0.5"/>
        <body name="left_finger" pos="0 -0.03 0.02">
          <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
          <geom name="lfp" type="box" size="0.01 0.005 0.02" mass="0.1"/>
        </body>
        <body name="right_finger" pos="0 0.03 0.02">
          <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
          <geom name="rfp" type="box" size="0.01 0.005 0.02" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def device():
  return get_test_device()


def _make_env(device: str):
  """Create a mock env with a gripper entity."""
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(GRIPPER_XML),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinMotorActuatorCfg(
          target_names_expr=("joint1", "finger_joint1", "finger_joint2"),
          transmission_type=TransmissionType.JOINT,
          effort_limit=10.0,
        ),
      ),
    ),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=NUM_ENVS, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)

  env = Mock(spec=ManagerBasedRlEnv)
  env.num_envs = NUM_ENVS
  env.device = device
  env.scene = {"robot": entity}
  return env, entity


def test_action_dim_is_one(device):
  """Binary gripper action should have action_dim=1."""
  env, _ = _make_env(device)
  cfg = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("finger_joint1",),
    open_command_expr={"finger_joint1": 0.04},
    close_command_expr={"finger_joint1": 0.0},
  )
  action = cfg.build(env)
  assert action.action_dim == 1


def test_positive_action_opens(device):
  """Positive action value should map to open command."""
  env, entity = _make_env(device)
  cfg = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("finger_joint1",),
    open_command_expr={"finger_joint1": 0.04},
    close_command_expr={"finger_joint1": 0.0},
  )
  action = cfg.build(env)

  action.process_actions(torch.ones(NUM_ENVS, 1, device=device))
  action.apply_actions()

  # finger_joint1 target should be 0.04 (open)
  joint_ids = action._joint_ids
  target = entity.data.joint_pos_target[:, joint_ids]
  assert torch.allclose(target, torch.full_like(target, 0.04))


def test_negative_action_closes(device):
  """Negative action value should map to close command."""
  env, entity = _make_env(device)
  cfg = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("finger_joint1",),
    open_command_expr={"finger_joint1": 0.04},
    close_command_expr={"finger_joint1": 0.0},
  )
  action = cfg.build(env)

  action.process_actions(torch.full((NUM_ENVS, 1), -1.0, device=device))
  action.apply_actions()

  joint_ids = action._joint_ids
  target = entity.data.joint_pos_target[:, joint_ids]
  assert torch.allclose(target, torch.zeros_like(target))


def test_zero_action_opens(device):
  """Zero action (>= 0) should map to open command."""
  env, entity = _make_env(device)
  cfg = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("finger_joint1",),
    open_command_expr={"finger_joint1": 0.04},
    close_command_expr={"finger_joint1": 0.0},
  )
  action = cfg.build(env)

  action.process_actions(torch.zeros(NUM_ENVS, 1, device=device))
  action.apply_actions()

  joint_ids = action._joint_ids
  target = entity.data.joint_pos_target[:, joint_ids]
  assert torch.allclose(target, torch.full_like(target, 0.04))


def test_mixed_actions(device):
  """Different envs can have different open/close states."""
  env, entity = _make_env(device)
  cfg = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("finger_joint1",),
    open_command_expr={"finger_joint1": 0.04},
    close_command_expr={"finger_joint1": 0.0},
  )
  action = cfg.build(env)

  # Envs 0,2 open; envs 1,3 close
  actions = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]], device=device)
  action.process_actions(actions)
  action.apply_actions()

  joint_ids = action._joint_ids
  target = entity.data.joint_pos_target[:, joint_ids]
  expected = torch.tensor([[0.04], [0.0], [0.04], [0.0]], device=device)
  assert torch.allclose(target, expected)


def test_reset_sets_open(device):
  """Reset should set gripper to open position."""
  env, entity = _make_env(device)
  cfg = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("finger_joint1",),
    open_command_expr={"finger_joint1": 0.04},
    close_command_expr={"finger_joint1": 0.0},
  )
  action = cfg.build(env)

  # Close first
  action.process_actions(torch.full((NUM_ENVS, 1), -1.0, device=device))
  action.apply_actions()

  # Reset envs 0 and 2
  reset_ids = torch.tensor([0, 2], device=device)
  action.reset(reset_ids)

  # Raw actions should be zeroed for reset envs
  assert torch.all(action.raw_action[reset_ids] == 0.0)

  # Processed actions for reset envs should be open
  assert torch.allclose(
    action._processed_actions[reset_ids],
    torch.full((2, 1), 0.04, device=device),
  )

  # Non-reset envs should still be close
  non_reset = torch.tensor([1, 3], device=device)
  assert torch.allclose(
    action._processed_actions[non_reset],
    torch.zeros(2, 1, device=device),
  )


def test_multiple_joints(device):
  """Binary action should work with multiple finger joints."""
  env, entity = _make_env(device)
  cfg = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("finger_joint1", "finger_joint2"),
    open_command_expr={"finger_joint1": 0.04, "finger_joint2": 0.04},
    close_command_expr={"finger_joint1": 0.0, "finger_joint2": 0.0},
  )
  action = cfg.build(env)

  # Still action_dim=1 (binary), but controls 2 joints
  assert action.action_dim == 1
  assert action._num_joints == 2

  action.process_actions(torch.ones(NUM_ENVS, 1, device=device))
  action.apply_actions()

  joint_ids = action._joint_ids
  target = entity.data.joint_pos_target[:, joint_ids]
  expected = torch.full((NUM_ENVS, 2), 0.04, device=device)
  assert torch.allclose(target, expected)
