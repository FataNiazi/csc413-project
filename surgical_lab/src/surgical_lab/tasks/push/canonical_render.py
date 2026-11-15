"""CANONICAL (flat, unlit) rendering for the pushblock task.

Two-domain sim-to-sim setup
---------------------------
REALISTIC  = the existing a3b RGB path: mujoco_warp's shaded raytracer with
             textures, shadows, a point light, and the full colour / light /
             camera domain randomisation. Untouched by this module.
CANONICAL  = the SAME camera, SAME resolution, SAME physics state, but the
             image is built from MuJoCo's **segmentation buffer** instead of
             the shaded colour buffer: every pixel is looked up by the geom it
             hit and painted one constant RGB triple per semantic class.

Why the segmentation buffer and not a hand-authored "unlit flat material":
  * Flat BY CONSTRUCTION. mujoco_warp's shader (render.py) always adds a
    diffuse + specular + ambient term and, with use_shadows, a shadow ray. A
    hand-authored material can only *approximate* flatness (emission ~1,
    specular 0, shininess 0); the segmentation buffer never touches the
    shading code at all, so there is provably zero lighting, shadow, texture
    or specular signal in the output.
  * IMMUNE to the visual DR for free. The a3b env randomises geom_rgba,
    light_diffuse, light_pos and light_dir every reset. Segmentation is
    written from the ray/BVH hit (render.py:711-715) before any of that is
    read, so the canonical image is invariant to all colour and lighting DR
    with no need to strip DR terms out of the event manager (which would
    perturb the RNG stream and de-sync the two domains).
  * PERFECTLY PAIRED. mujoco_warp fills rgb_data, depth_data and seg_data in
    ONE kernel launch (render.py `render()`), so the canonical and realistic
    images for a frame come from the identical geom_xpos/cam_xpos, not from
    two renders of a re-simulated state.

Verified non-invasive: with and without ``"segmentation"`` in the camera's
``data_types`` the RGB buffer is bit-identical (same seed, 5 steps, 4 envs,
max|diff| == 0). Segmentation only allocates a second output buffer.

Camera-pose DR is deliberately LEFT ON in the canonical domain: it is a
geometric randomisation, it moves both domains identically, and switching it
off would break the pixel correspondence of a pair.

Usage
-----
Register-free (paired dumping, see scripts/verify_two_domains.py)::

    cfg = psm_push_wide_a3b_env_cfg()
    add_segmentation_to_camera(cfg)          # realistic RGB stays untouched
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    ...
    real = manipulation_mdp.camera_rgb(env, "topdown_cam")   # (B,3,H,W) [0,1]
    canon = canonical_rgb(env, "topdown_cam")                # (B,3,H,W) [0,1]

As a task (canonical pixels feed the policy)::

    Surgicallab-Push-Wide-A3B-Canonical   # registered in config/psm/__init__.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

__all__ = [
  "CANONICAL_COLORS",
  "CANONICAL_CAMERA_NAME",
  "add_segmentation_to_camera",
  "classify_geom",
  "build_canonical_lut",
  "canonical_rgb",
  "canonical_rgb_u8",
  "psm_push_wide_a3b_canonical_env_cfg",
]

CANONICAL_CAMERA_NAME = "topdown_cam"

_MJ_OBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)

# One constant RGB triple per semantic class. Chosen maximally separable so a
# nearest-colour decode of a saved PNG is exact, and so a human can read the
# frame at 84x84. "unknown" is magenta on purpose: an unmapped geom must be
# obvious in the dump rather than silently blending into the floor.
CANONICAL_COLORS: dict[str, tuple[int, int, int]] = {
  "background": (0, 0, 0),  # ray missed everything (skybox)
  "floor": (40, 40, 40),  # terrain plane
  "goal": (0, 255, 0),  # goal bar
  "block": (255, 0, 0),  # the pushed block
  "tool": (0, 96, 255),  # the PSM / tooltip
  "unknown": (255, 0, 255),
}

# Attribute name used to memoise the per-model LUT on the env instance.
_LUT_ATTR = "_canonical_render_lut"


def classify_geom(geom_name: str) -> str:
  """Map a compiled MuJoCo geom name to a canonical class key.

  The a3b scene compiles to (see the geom dump in the module docstring test):
    ``terrain``, ``robot/<link>_visual_<i>`` + ``robot/*_collision``,
    ``block/block_geom``, ``goal/goal_geom``.
  """
  if geom_name == "terrain" or geom_name.startswith("terrain"):
    return "floor"
  if geom_name.startswith("goal/"):
    return "goal"
  if geom_name.startswith("block/"):
    return "block"
  if geom_name.startswith("robot/"):
    return "tool"
  return "unknown"


def add_segmentation_to_camera(
  cfg: "ManagerBasedRlEnvCfg",
  sensor_name: str = CANONICAL_CAMERA_NAME,
) -> "ManagerBasedRlEnvCfg":
  """Opt the named camera sensor into segmentation output, in place.

  Purely additive: the RGB buffer is byte-identical with and without this
  (mujoco_warp writes rgb_data and seg_data from the same kernel into separate
  arrays). Idempotent.
  """
  sensors = cfg.scene.sensors or ()
  for s in sensors:
    if getattr(s, "name", None) != sensor_name:
      continue
    types = tuple(getattr(s, "data_types", ()))
    if "segmentation" not in types:
      s.data_types = types + ("segmentation",)
    return cfg
  names = [getattr(s, "name", "?") for s in sensors]
  raise ValueError(
    f"No camera sensor named '{sensor_name}' on this env cfg. Found: {names}"
  )


def build_canonical_lut(env: "ManagerBasedRlEnv") -> torch.Tensor:
  """(ngeom + 1, 3) float32 colour table in [0, 1], row 0 = background.

  Row ``geom_id + 1`` holds the colour of that geom's class. Built once per
  env; geom ids are stable for the life of the compiled model (mujoco_warp
  keeps ONE model with N worlds), so DR that expands per-world model fields
  does not invalidate it.
  """
  mj_model = env.sim.mj_model
  ngeom = mj_model.ngeom
  lut = torch.empty((ngeom + 1, 3), dtype=torch.float32)
  lut[0] = torch.tensor(CANONICAL_COLORS["background"], dtype=torch.float32)
  for gid in range(ngeom):
    key = classify_geom(mj_model.geom(gid).name)
    lut[gid + 1] = torch.tensor(CANONICAL_COLORS[key], dtype=torch.float32)
  return (lut / 255.0).to(env.device)


def _get_lut(env: "ManagerBasedRlEnv") -> torch.Tensor:
  lut = getattr(env, _LUT_ATTR, None)
  if lut is None:
    lut = build_canonical_lut(env)
    setattr(env, _LUT_ATTR, lut)
  return lut


def canonical_rgb(
  env: "ManagerBasedRlEnv",
  sensor_name: str = CANONICAL_CAMERA_NAME,
  stretch: bool = False,
) -> torch.Tensor:
  """Flat unlit canonical image, (B, 3, H, W) float32 in [0, 1].

  Args:
    env: the running env (its camera sensor must have segmentation enabled --
      call :func:`add_segmentation_to_camera` on the cfg before construction).
    sensor_name: camera sensor to read.
    stretch: apply the same per-frame min-max stretch the realistic
      deployment path uses (``_camera_rgb_normalized``). OFF by default: the
      whole point of the canonical domain is constant colours, and the stretch
      would make a class's value depend on which other classes are in frame.
  """
  sensor = env.scene[sensor_name]
  seg = sensor.data.segmentation  # (B, H, W, 2) int32
  if seg is None:
    raise RuntimeError(
      f"Camera '{sensor_name}' has no segmentation buffer. Add 'segmentation' to "
      "its data_types before building the env, e.g. "
      "canonical_render.add_segmentation_to_camera(cfg)."
    )
  ids = seg[..., 0].long()  # geom id, -1 where the ray missed
  types = seg[..., 1]  # mjOBJ_GEOM (5) on a hit, -1 on a miss
  # Row 0 is the background; every geom hit is row id+1. A non-geom hit (flex)
  # cannot happen in this scene but is folded into background rather than
  # indexing a wrong row.
  idx = torch.where(types == _MJ_OBJ_GEOM, ids + 1, torch.zeros_like(ids))
  img = _get_lut(env)[idx]  # (B, H, W, 3) float in [0,1]
  img = img.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
  if stretch:
    flat = img.flatten(1)
    lo = flat.min(dim=1).values[:, None, None, None]
    hi = flat.max(dim=1).values[:, None, None, None]
    img = (img - lo) / (hi - lo).clamp(min=1e-6)
  return img


def canonical_rgb_u8(
  env: "ManagerBasedRlEnv",
  sensor_name: str = CANONICAL_CAMERA_NAME,
) -> torch.Tensor:
  """Canonical image as (B, H, W, 3) uint8 -- same layout as ``sensor.data.rgb``."""
  img = canonical_rgb(env, sensor_name)  # (B,3,H,W) [0,1]
  return (img * 255.0).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# Config hook


def psm_push_wide_a3b_canonical_env_cfg(
  play: bool = False,
  stretch: bool = False,
) -> "ManagerBasedRlEnvCfg":
  """The a3b pushblock env with the CANONICAL domain feeding the policy.

  Identical to :func:`psm_push_wide_a3b_env_cfg` in every physical respect --
  same robot, block, goal, camera, DR, rewards, terminations. The ONLY change
  is that the ``camera`` observation group returns the flat segmentation-derived
  image instead of the shaded RGB one.
  """
  from surgical_lab.tasks.push.config.psm.a3b_env_cfg import (
    psm_push_wide_a3b_env_cfg,
  )

  cfg = psm_push_wide_a3b_env_cfg(play=play)
  add_segmentation_to_camera(cfg)
  term = cfg.observations["camera"].terms["topdown_rgb"]
  term.func = canonical_rgb
  term.params = {"sensor_name": CANONICAL_CAMERA_NAME, "stretch": stretch}
  return cfg
