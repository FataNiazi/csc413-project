"""Video of the Surgicallab-Dual-Reach-Dummy task, with its targets drawn in.

Drives both arms with a SCRIPTED proportional controller (not a trained policy)
so the video shows what a solved episode looks like, and overlays what the
policy is told:

  - translucent box per arm: the region that arm's target is sampled from
  - solid sphere: the current target (re-sampled every 4 s episode)
  - faint sphere around it: the 3 mm "reached" tolerance
  - line tool -> target: the target-minus-tool vector in the observation

Run from surgical_lab/ (CPU is fine):

  set -a; . scripts/a3_liftpeg_env.sh; set +a
  MUJOCO_GL=egl uv run python scripts/demo_dummy_reach.py   # OUT=... EPISODES=3
"""

import os

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from PIL import Image, ImageDraw, ImageFont

import surgical_lab.tasks  # noqa: F401
from surgical_lab.tasks.lift.config.psm import dual_reach_dummy as DR

DEV = os.environ.get("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
OUT = os.environ.get("OUT", "renders/dual_reach_dummy.mp4")
EPISODES = int(os.environ.get("EPISODES", "3"))
H, W = 720, 960
FRAME_EVERY = 2
SEED = int(os.environ.get("SEED", "3"))
torch.manual_seed(SEED)

cfg = load_env_cfg("Surgicallab-Dual-Reach-Dummy")
cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
env.reset()
ORIG = env.scene.env_origins[0].cpu().numpy()
EP_STEPS = int(round(cfg.episode_length_s / env.step_dt))

m = env.sim.mj_model
d = mujoco.MjData(m)


def sync():
  d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  d.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
  if m.nmocap:
    d.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
    d.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
  mujoco.mj_forward(m, d)


def tools():
  return DR._tools(env)[0].cpu().numpy()


def targets():
  return DR._targets(env)[0].cpu().numpy()


# Action -> tool motion per arm, measured (arm 2 is mounted rotated 180 deg).
def calibrate():
  J = []
  for arm in range(2):
    M = np.zeros((3, 3))
    for k in range(3):
      moved = []
      for s in (1.0, -1.0):
        env.reset()
        for _ in range(10):
          env.step(torch.zeros(1, 8, device=DEV))
        p0 = tools()[arm]
        a = torch.zeros(1, 8, device=DEV)
        a[0, 4 * arm + k] = s
        for _ in range(5):
          env.step(a)
        moved.append((tools()[arm] - p0) / 5)
      M[:, k] = (moved[0] - moved[1]) / 2
    J.append(np.linalg.pinv(M))
  return J


JINV = calibrate()

# --- overlay ---------------------------------------------------------------
COL = {0: (0.1, 0.8, 1.0), 1: (1.0, 0.3, 0.9)}  # arm 1 cyan, arm 2 magenta
COL8 = {k: tuple(int(255 * c) for c in v) for k, v in COL.items()}
CENTERS = np.array(DR.CENTERS)
HALF = np.array((DR.RANGE_XY, DR.RANGE_XY, DR.RANGE_Z))
EYE = np.eye(3).flatten()


def add_geom(scn, gtype, size, pos, rgba):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), EYE,
                      np.asarray(rgba, np.float32))
  scn.ngeom += 1


def add_line(scn, a, b, rgba, width=0.0006):
  if scn.ngeom >= scn.maxgeom:
    return
  g = scn.geoms[scn.ngeom]
  mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), EYE,
                      np.asarray(rgba, np.float32))
  mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                       np.asarray(a, float), np.asarray(b, float))
  scn.ngeom += 1


os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
import imageio.v2 as iio  # noqa: E402

writer = iio.get_writer(OUT, fps=round(1.0 / (env.step_dt * FRAME_EVERY)),
                        codec="libx264", quality=8, macro_block_size=8)
renderer = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.lookat[:] = ORIG + np.array([-0.020, 0.012, 0.060])
cam.distance, cam.azimuth, cam.elevation = 0.30, 250.0, -22.0
FONT = ImageFont.load_default(size=22)
SMALL = ImageFont.load_default(size=18)


def grab(ep, t, reached):
  sync()
  renderer.update_scene(d, cam)
  scn = renderer.scene
  tl, tg = tools(), targets()
  for arm in range(2):
    c = COL[arm]
    add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, HALF, ORIG + CENTERS[arm], (*c, 0.10))
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (DR.REACH_TOL, 0, 0), ORIG + tg[arm],
             (*c, 0.25))
    add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.0012, 0, 0), ORIG + tg[arm], (*c, 1.0))
    add_line(scn, ORIG + tl[arm], ORIG + tg[arm], (*c, 0.9))
  img = Image.fromarray(renderer.render())
  dr = ImageDraw.Draw(img)
  dr.rectangle((0, 0, W, 40), fill=(0, 0, 0))
  dr.text((12, 8), f"Dual-Reach-Dummy   episode {ep + 1}/{EPISODES}   "
          f"t={t * env.step_dt:4.1f}s",
          fill=(255, 255, 255), font=FONT)
  dr.text((W - 360, 10), "scripted controller, not a trained policy",
          fill=(170, 170, 170), font=SMALL)
  dr.rectangle((0, H - 118, W, H), fill=(0, 0, 0))
  for arm in range(2):
    dist = np.linalg.norm(tg[arm] - tl[arm]) * 1000
    ok = dist < DR.REACH_TOL * 1000
    tgt_mm = ", ".join(f"{v:5.1f}" for v in tg[arm] * 1000)
    dr.text((12, H - 112 + 28 * arm),
            f"arm {arm + 1}  target ({tgt_mm}) mm   distance {dist:5.1f} mm"
            + ("   REACHED" if ok else ""),
            fill=COL8[arm], font=FONT)
  dr.text((12, H - 52),
          "box = region this arm's target is sampled from   |   "
          "sphere = target, faint shell = 3 mm tolerance",
          fill=(200, 200, 200), font=SMALL)
  dr.text((12, H - 28),
          "line = target - tool, the vector the policy observes   |   "
          "positions in mm, env frame",
          fill=(200, 200, 200), font=SMALL)
  if reached:
    dr.text((W - 200, H - 112), "BOTH REACHED", fill=(120, 255, 120), font=FONT)
  writer.append_data(np.asarray(img))


env.reset()
GAIN = 0.6
for ep in range(EPISODES):
  for t in range(EP_STEPS):
    a = torch.zeros(1, 8, device=DEV)
    if t >= 25:  # hold still for half a second so the new targets are visible
      err = targets() - tools()
      for arm in range(2):
        u = np.clip(GAIN * JINV[arm] @ err[arm], -1, 1)
        a[0, 4 * arm:4 * arm + 3] = torch.tensor(u, device=DEV)
    reached = bool(DR.m_reached(env)[0])
    if t % FRAME_EVERY == 0:
      grab(ep, t, reached)
    env.step(a)  # the last step of an episode times out and resamples targets
writer.close()
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)")
