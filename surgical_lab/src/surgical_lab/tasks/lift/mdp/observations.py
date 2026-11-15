"""Observation functions for the lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_to_object_site_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  object_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end effector to a site on the object in base frame.

  Unlike ``ee_to_object_distance`` which targets the object root (body center),
  this targets a specific site on the object, useful for hollow objects where
  the center is not a valid grasp location.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_site_pos_w = obj.data.site_pos_w[:, object_cfg.site_ids].squeeze(1)
  distance_vec_w = obj_site_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b
