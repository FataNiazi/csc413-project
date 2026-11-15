"""Reading deformable (flex) state inside mjlab.

mjlab entities expose rigid-body state, but a flexcomp's vertices live in the raw
mujoco_warp Data as ``flexvert_xpos`` (nworld, nflexvert, 3). ``env.sim.data`` bridges that
to torch, so we read the cloth's grasp corner straight out of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def flex_vertex_pos_w(env: ManagerBasedRlEnv, vertex_id: int) -> torch.Tensor:
  """World position of one flex vertex, per env. Shape (num_envs, 3)."""
  verts = env.sim.data.flexvert_xpos  # (nworld, nflexvert, 3)
  return verts[:, vertex_id, :]


def flex_vertex_pos_all(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World positions of every flex vertex. Shape (num_envs, nflexvert, 3)."""
  return env.sim.data.flexvert_xpos


def corner_index(count: int, ix: int, iy: int) -> int:
  """Vertex index of grid cell (ix, iy) for a `count` x `count` flexcomp grid.

  flexcomp emits vertices with x as the outer loop and y as the inner one, which we
  confirmed by reading back body positions from a compiled towel.
  """
  return ix * count + iy
