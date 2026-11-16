"""Does a second PSM break the LiftPeg scene? Smoke test for liftpeg_dual.py.

Builds `Surgicallab-Liftpeg-Dual-State-PoC` with ONE env and checks, in order:

  1. build:        scene compiles, action dim is 8 (xyz+jaw per arm), poses.
  2. independence: commanding one arm moves that arm and not the other.
  3. calibration:  each arm's action -> world-motion matrix, measured, since
                   arm 2 is mounted rotated and its base axes are not world's.
  4. reach:        each arm, alone, servoes its tool over the peg and the goal.
  5. contact:      arm 2 is driven into arm 1; with DUAL_PSM_ARM_CONTACT=1 the
                   tools must stop short, with =0 they pass through.
  6. soak:         random actions for a few hundred steps, no NaN, resets fine.

Run from the repo root (CPU is fine; DEV=cuda:0 on a GPU box):

  set -a; . scripts/a3_liftpeg_env.sh; set +a
  uv run python scripts/probe_dual_psm.py            # SNAP=1 also saves PNGs
"""

import os

# Written against the original facing-arms layout (arm 2 rotated 180 deg, tools
# 40 mm apart); the task default is now the lab's side-by-side cart.
os.environ.setdefault("DUAL_PSM_LAYOUT", "facing")
import time

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import surgical_lab.tasks  # noqa: F401

DEV = os.environ.get("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
SNAP = os.environ.get("SNAP", "0") == "1"
OUT = os.environ.get("OUT", "renders/dual_psm")

cfg = load_env_cfg("Surgicallab-Liftpeg-Dual-State-PoC")
cfg.scene.num_envs = 1
t0 = time.time()
env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
env.reset()
print(f"[build] OK in {time.time() - t0:.1f}s on {DEV}")

am = env.action_manager
dim = am.total_action_dim
print(f"[build] actions {dict(zip(am.active_terms, am.action_term_dim, strict=True))}  total={dim}")
assert dim == 8, dim
arms = ("robot", "robot2")
R = {n: env.scene[n] for n in arms}
ORIG = env.scene.env_origins[0]
# Slices of the flat action vector, per arm: (xyz, jaw).
SL = {"robot": (slice(0, 3), 3), "robot2": (slice(4, 7), 7)}


def tool(n):
  r = R[n]
  return (r.data.site_pos_w[0, r.site_names.index("tool_midpoint")] - ORIG).cpu().numpy()


def mm(v):
  return np.array2string(np.asarray(v) * 1000, precision=1, suppress_small=True)


def step(a):
  _, _, term, trunc, _ = env.step(torch.as_tensor(a, dtype=torch.float32, device=DEV)[None])
  return bool(term[0] or trunc[0])


def hold(jaw=(1.0, 1.0)):
  a = np.zeros(dim)
  a[SL["robot"][1]], a[SL["robot2"][1]] = jaw
  return a


def reset(settle=10):
  # Every reset is followed by a ~5-step settle of up to 6 mm, mostly in z, in
  # the SINGLE-arm scene too (the home-pose write vs the action's held target).
  # Measure from after it, or it reads as one arm dragging the other.
  env.reset()
  for _ in range(settle):
    step(hold())


for n in arms:
  print(f"[build] {n:7s} base {mm((R[n].data.root_link_pos_w[0] - ORIG).cpu())} mm  "
        f"tool {mm(tool(n))} mm")
for n in ("peg", "block", "goal"):
  print(f"[build] {n:7s} at {mm((env.scene[n].data.root_link_pos_w[0] - ORIG).cpu())} mm")
print(f"[build] tool separation {np.linalg.norm(tool('robot') - tool('robot2')) * 1000:.1f} mm")

# --- CPU mirror of the warp state: contacts and offscreen renders -----------
m = env.sim.mj_model
d = mujoco.MjData(m)


def sync():
  d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  d.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
  if m.nmocap:
    d.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
    d.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
  mujoco.mj_forward(m, d)


def geom_owner(g):
  name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
  return name.split("/")[0]


def arm_arm_contacts():
  sync()
  out = []
  for c in d.contact[: d.ncon]:
    o1, o2 = geom_owner(c.geom1), geom_owner(c.geom2)
    if {o1, o2} == set(arms):
      out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                  mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), c.dist))
  return out


def snap(tag):
  if not SNAP:
    return
  sync()
  try:
    with mujoco.Renderer(m, 480, 640) as r:
      cam = mujoco.MjvCamera()
      cam.lookat[:] = (ORIG.cpu().numpy() + np.array([0.0, 0.02, 0.06]))
      cam.distance, cam.azimuth, cam.elevation = 0.45, 135.0, -30.0
      r.update_scene(d, cam)
      img = r.render()
    import imageio.v3 as iio
    os.makedirs(OUT, exist_ok=True)
    iio.imwrite(f"{OUT}/{tag}.png", img)
    print(f"[snap] {OUT}/{tag}.png")
  except Exception as e:  # no GL on this box: report, do not fail the probe
    print(f"[snap] skipped ({type(e).__name__}: {e}); try MUJOCO_GL=egl or osmesa")


snap("0_home")

# --- 2. independence ---------------------------------------------------------
N = 30
for mover in arms:
  reset()
  p0 = {n: tool(n) for n in arms}
  for _ in range(N):
    a = hold()
    a[SL[mover][0]] = (1.0, 0.0, 0.0)
    step(a)
  dp = {n: tool(n) - p0[n] for n in arms}
  other = arms[1 - arms.index(mover)]
  ok = np.linalg.norm(dp[mover]) > 5e-3 and np.linalg.norm(dp[other]) < 1e-3
  print(f"[indep] drive {mover:7s} +x{N}: {mover} moved {mm(dp[mover])} mm, "
        f"{other} moved {mm(dp[other])} mm -> {'OK' if ok else 'FAIL'}")

reset()
for _ in range(10):
  step(hold(jaw=(1.0, -1.0)))
jaw = {n: float(R[n].data.joint_pos[0, R[n].joint_names.index("jaw")]) for n in arms}
print(f"[indep] jaw open/close split: robot={jaw['robot']:.3f} (open) "
      f"robot2={jaw['robot2']:.3f} (closed) -> "
      f"{'OK' if jaw['robot'] > jaw['robot2'] + 0.1 else 'FAIL'}")

# --- 3. calibration: J[:, k] = world tool motion per unit of action axis k ---
J = {}
for n in arms:
  cols = []
  for k in range(3):
    # +1 then -1 from the same start: the difference cancels any residual drift.
    moved = []
    for sgn in (1.0, -1.0):
      reset()
      p0 = tool(n)
      for _ in range(10):
        a = hold()
        a[SL[n][0].start + k] = sgn
        step(a)
      moved.append(tool(n) - p0)
    cols.append((moved[0] - moved[1]) / 20)
  J[n] = np.stack(cols, axis=1)
  print(f"[calib] {n:7s} mm of world motion per step, per action axis (cols=a0,a1,a2):")
  for row, ax in zip(J[n] * 1000, "xyz", strict=True):
    print(f"          world {ax}: {np.array2string(row, precision=3, suppress_small=True)}")


# --- 4. reach ------------------------------------------------------------------
def servo(n, target, steps=400, tol=1.5e-3):
  """P-servo one arm's tool to `target`. Also returns the arm-arm contact pairs
  seen on the way, the deepest penetration among them (m, negative), and how
  far the OTHER (holding) arm's tool was pushed from where it started (m)."""
  Jinv = np.linalg.pinv(J[n])
  other = arms[1 - arms.index(n)]
  p_other = tool(other)
  pairs, deepest, shoved = set(), 0.0, 0.0
  for i in range(steps):  # noqa: B007 -- i is the step count returned
    shoved = max(shoved, np.linalg.norm(tool(other) - p_other))
    for g1, g2, dist in arm_arm_contacts():
      pairs.add(tuple(sorted((g1, g2))))
      deepest = min(deepest, dist)
    err = target - tool(n)
    if np.linalg.norm(err) < tol:
      break
    a = hold()
    a[SL[n][0]] = np.clip(Jinv @ err, -1.0, 1.0)
    if step(a):
      break
  return i, target - tool(n), pairs, deepest, shoved


peg_xy = (env.scene["peg"].data.root_link_pos_w[0] - ORIG).cpu().numpy()[:2]
goal_xy = (env.scene["goal"].data.root_link_pos_w[0] - ORIG).cpu().numpy()[:2]
targets = {
  "above peg (z=45mm)": np.array([*peg_xy, 0.045]),
  "above goal (z=30mm)": np.array([*goal_xy, 0.030]),
}
for n in arms:
  for name, tgt in targets.items():
    reset()
    k, err, pairs, deepest, shoved = servo(n, tgt)
    print(f"[reach] {n:7s} -> {name:20s} {'OK  ' if np.linalg.norm(err) < 1.5e-3 else 'MISS'} "
          f"err {mm(err)} mm in {k} steps; arm-arm contact pairs on the way: "
          f"{len(pairs)} (deepest {deepest * 1000:.2f} mm); other arm pushed "
          f"{shoved * 1000:.1f} mm")
    for pr in sorted(pairs):
      print(f"          {pr[0]} <-> {pr[1]}")
    snap(f"reach_{n}_{name.split()[1]}")

# --- 5. arm-arm contact: drive arm 2's tool onto arm 1's ----------------------
reset()
target = tool("robot")
first, min_sep = None, 1e9
for i in range(200):
  err = target - tool("robot2")
  a = hold()
  a[SL["robot2"][0]] = np.clip(np.linalg.pinv(J["robot2"]) @ err, -1.0, 1.0)
  step(a)
  min_sep = min(min_sep, np.linalg.norm(tool("robot") - tool("robot2")))
  if first is None and arm_arm_contacts():
    first = (i, arm_arm_contacts()[:3])
print(f"[contact] arm contact {'ON' if os.environ.get('DUAL_PSM_ARM_CONTACT', '1') == '1' else 'OFF'}: "
      f"closest tool-tool distance {min_sep * 1000:.1f} mm; "
      f"first arm-arm contact: {first if first else 'none'}")
snap("contact")

# --- 6. soak -------------------------------------------------------------------
reset()
resets, t0 = 0, time.time()
for i in range(300):
  a = np.random.uniform(-1, 1, dim)
  resets += step(a)
  q = env.sim.data.qpos[0]
  if not torch.isfinite(q).all():
    raise SystemExit(f"[soak] NaN in qpos at step {i}")
print(f"[soak] 300 random steps OK, {resets} episode ends, "
      f"{(time.time() - t0) / 300 * 1000:.0f} ms/step")
env.close()
