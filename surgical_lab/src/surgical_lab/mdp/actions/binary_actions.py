"""Binary joint action terms for gripper control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BinaryJointPositionActionCfg(ActionTermCfg):
  """Configuration for binary joint position action term.

  This action term maps a binary action (positive/negative or 1/0) to either
  an "open" or "close" joint configuration. Useful for gripper control where
  the gripper is either fully open or fully closed.

  Convention:
    - Open action: positive values (>= 0)
    - Close action: negative values (< 0)
  """

  joint_names: Sequence[str]
  """Joint names or regex expressions for the action."""
  open_command_expr: dict[str, float]
  """Joint positions when open (dict of regex pattern to value)."""
  close_command_expr: dict[str, float]
  """Joint positions when closed (dict of regex pattern to value)."""

  def build(self, env: ManagerBasedRlEnv) -> BinaryJointPositionAction:
    return BinaryJointPositionAction(self, env)


class BinaryJointPositionAction(ActionTerm):
  """Base class for binary joint actions.

  This action term maps a binary action to the *open* or *close* joint
  configurations. These configurations are specified through the config object.
  If the input action is a float vector, the action is considered binary based
  on the sign of the action values.

  Convention:
    - Open action: positive values (>= 0)
    - Close action: negative values (< 0)

  This is useful for gripper actions where the gripper is either open or closed.
  """

  cfg: BinaryJointPositionActionCfg
  _entity: Entity

  def __init__(self, cfg: BinaryJointPositionActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)

    # Resolve joints by name patterns
    joint_ids, joint_names = self._entity.find_joints(cfg.joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names
    self._num_joints = len(joint_ids)

    # Create tensors for raw and processed actions
    # Binary action has dimension 1
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    self._processed_actions = torch.zeros(
      self.num_envs, self._num_joints, device=self.device
    )

    # Parse open command
    self._open_command = torch.zeros(self._num_joints, device=self.device)
    index_list, name_list, value_list = resolve_matching_names_values(
      cfg.open_command_expr, self._joint_names
    )
    if len(index_list) != self._num_joints:
      raise ValueError(
        f"Could not resolve all joints for open_command_expr. "
        f"Missing: {set(self._joint_names) - set(name_list)}"
      )
    self._open_command[index_list] = torch.tensor(value_list, device=self.device)

    # Parse close command
    self._close_command = torch.zeros(self._num_joints, device=self.device)
    index_list, name_list, value_list = resolve_matching_names_values(
      cfg.close_command_expr, self._joint_names
    )
    if len(index_list) != self._num_joints:
      raise ValueError(
        f"Could not resolve all joints for close_command_expr. "
        f"Missing: {set(self._joint_names) - set(name_list)}"
      )
    self._close_command[index_list] = torch.tensor(value_list, device=self.device)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    # Store raw actions
    self._raw_actions[:] = actions
    # Compute binary mask: negative values -> close, non-negative -> open
    binary_mask = actions < 0
    # Compute command based on mask
    self._processed_actions = torch.where(
      binary_mask, self._close_command, self._open_command
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = self._open_command
    # Write the open target to the entity immediately so the delayed actuator's
    # post-reset backfill picks up 0.04 (open) instead of the 0.0 left by
    # clear_state().
    self._entity.set_joint_position_target(
      self._processed_actions, joint_ids=self._joint_ids
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(
      self._processed_actions, joint_ids=self._joint_ids
    )
