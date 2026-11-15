"""Is the goal square actually visible from the camera?

Success for the retraction task is "the whole red square can be seen", so the
test is a literal line-of-sight one: shoot a ray from the camera to each of a
grid of sample points covering the square, and succeed only when NONE of them
is blocked by a tissue triangle.

Chosen over a segmentation render because it is exact and deterministic -- no
pixel thresholds, no resolution dependence, and no colour heuristics. (Colour
heuristics have already produced three false readings on this project: shadowed
pink tissue reads as "red goal" to any reasonable RGB test.)

The one thing it does NOT model is occlusion by anything other than the tissue.
The arm can hide the square from the camera and this will still report visible.
That is a deliberate limit of the cheap test: the arm is thin, it is above the
square only briefly, and paying for a full segmentation pass every step to
catch it is not worth it. If it ever matters, the fix is to add the tool's
capsules as occluders here rather than to switch methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Sample grid across the square, corners included. 5x5 is what the whole-square
# criterion needs: the corners are the last thing a peeling flap uncovers, so a
# grid that omits them would call success early.
SAMPLES = 5
# A ray is blocked only strictly between camera and target; these keep the
# endpoints themselves from counting as hits.
_EPS_NEAR, _EPS_FAR = 1e-4, 1e-4


def _tri_indices(env: ManagerBasedRlEnv) -> torch.Tensor:
  idx = getattr(env, "_tissue_tri_idx", None)
  if idx is None:
    m = env.sim.mj_model
    idx = torch.tensor(
      m.flex_elem.reshape(-1, 3).astype("int64"), device=env.device
    )
    env._tissue_tri_idx = idx
  return idx


def _sample_points(env: ManagerBasedRlEnv, side: float) -> torch.Tensor:
  """(n, SAMPLES**2, 3) world points covering the goal square."""
  goal = env.scene["goal"].data.root_link_pose_w[:, :3]        # (n,3)
  g = getattr(env, "_tissue_goal_grid", None)
  if g is None:
    t = torch.linspace(-0.5, 0.5, SAMPLES, device=env.device) * side
    gx, gy = torch.meshgrid(t, t, indexing="ij")
    g = torch.stack([gx.reshape(-1), gy.reshape(-1),
                     torch.zeros(SAMPLES ** 2, device=env.device)], dim=-1)
    env._tissue_goal_grid = g
  return goal.unsqueeze(1) + g.unsqueeze(0)


def goal_fully_visible(env: ManagerBasedRlEnv, side: float) -> torch.Tensor:
  """(n,) bool: no tissue triangle blocks the view of ANY sample point."""
  return visible_fraction(env, side) >= 1.0


def visible_fraction(env: ManagerBasedRlEnv, side: float) -> torch.Tensor:
  """(n,) in [0,1]: share of sample points with clear line of sight.

  The shaped counterpart to the boolean test -- it gives the reward something
  to climb instead of a single step at the very end of the peel.

  CACHED per simulation step. This is a batched Moeller-Trumbore over
  (env, sample, triangle) = (n, 25, 128), and it is asked for at least five
  times a step: two metrics, the success termination, and twice inside the
  reward. Recomputing it each time multiplied the cost of the whole env for no
  new information -- nothing moves between those calls.
  """
  key = getattr(env, "common_step_counter", None)
  cache = getattr(env, "_tissue_vis_cache", None)
  if cache is not None and key is not None and cache[0] == key:
    return cache[1]
  out = _visible_fraction_uncached(env, side)
  if key is not None:
    env._tissue_vis_cache = (key, out)
  return out


def _visible_fraction_uncached(env: ManagerBasedRlEnv, side: float) -> torch.Tensor:
  from surgical_lab.tasks.retract.mdp.cloth import flex_vertex_pos_all

  cam = env.scene["liftcam"].data.root_link_pose_w[:, :3]
  pts = _sample_points(env, side)
  verts = flex_vertex_pos_all(env)
  tri = _tri_indices(env)

  v0 = verts[:, tri[:, 0], :]
  v1 = verts[:, tri[:, 1], :]
  v2 = verts[:, tri[:, 2], :]
  K = pts.shape[1]
  e1 = (v1 - v0).unsqueeze(1).expand(-1, K, -1, -1)
  e2 = (v2 - v0).unsqueeze(1).expand(-1, K, -1, -1)
  v0 = v0.unsqueeze(1)
  o = cam.unsqueeze(1).unsqueeze(2)
  d = (pts - cam.unsqueeze(1)).unsqueeze(2).expand_as(e1)

  h = torch.linalg.cross(d, e2, dim=-1)
  a = (e1 * h).sum(-1)
  ok = a.abs() > 1e-12
  f = torch.where(ok, 1.0 / torch.where(ok, a, torch.ones_like(a)),
                  torch.zeros_like(a))
  s = o - v0
  u = f * (s * h).sum(-1)
  q = torch.linalg.cross(s.expand_as(e1), e1, dim=-1)
  v = f * (d * q).sum(-1)
  t = f * (e2 * q).sum(-1)
  hit = (ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
         & (t > _EPS_NEAR) & (t < 1.0 - _EPS_FAR))
  return (~hit.any(dim=-1)).float().mean(dim=-1)
