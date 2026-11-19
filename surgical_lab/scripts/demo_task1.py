"""Scripted Task 1 on the lab (side-by-side) two-PSM layout, recorded to mp4.

  arm 2: park out of the way (its home is right above the goal square);
  arm 1: unthread the ring off the peg, carry it to the red goal, set it down,
         let go, and back out of the way;
  arm 2: pick the ring up off the goal, lift it clear of the post, carry it
         over the peg and lower it back onto the post (re-insert), let go.

All without the arms touching. Arm-arm contact is checked two ways: by the
task's own contact sensor (whole arms, including the upper-link boxes), and by
the probe below, which also flags any arm part touching the peg, ring or
floor (other than the jaws gripping the ring).

Grasping reuses the single-arm scripted teacher's geometry, per arm (see
demo_dual_handoff.py, which this is adapted from). Arm 2's shaft is tilted
~39 deg but its wrist is levelled (DUAL_PSM2_WRIST_LEVEL), so both arms grip
with vertical jaws.

Run from surgical_lab/ (CPU is fine, a few minutes):

  set -a; . scripts/a3_liftpeg_env.sh; set +a
  MUJOCO_GL=egl uv run python scripts/demo_task1.py      # OUT=... SEED=...
"""

import copy
import math
import os

os.environ.setdefault("DUAL_PSM_LAYOUT", "side")
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from PIL import Image, ImageDraw, ImageFont

import surgical_lab.tasks  # noqa: F401
from surgical_lab.tasks.lift.config.psm import liftpeg_rewards as LR
from surgical_lab.tasks.lift.config.psm import liftpeg_dual as D
from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS
from surgical_lab.tasks.lift.config.psm import liftpeg_teacher as T

DEV = os.environ.get("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
assert D.LAYOUT == "side", "this demo is for the side-by-side layout"
OUT = os.environ.get("OUT", "renders/dual_psm/task1.mp4")
SEED = int(os.environ.get("SEED", "0"))
DEBUG = os.environ.get("DEBUG", "0") == "1"
FRAME_EVERY = int(os.environ.get("FRAME_EVERY", "2"))   # 50 Hz control -> 25 fps
W, H = 960, 720

torch.manual_seed(SEED)
np.random.seed(SEED)
# DEMO_TASK=Surgicallab-Dual-Task1-Naive runs the script inside the Task 1 env
# and reports what its reward saw (phase switch, success, return): a check that
# the reward fires on a known-good episode.
TASK = os.environ.get("DEMO_TASK", "Surgicallab-Liftpeg-Dual-State-PoC")
DROP = os.environ.get("DEMO_DROP", "0") == "1"
cfg = load_env_cfg(TASK)
cfg.scene.num_envs = 1
if "arm_contact_rate" not in cfg.metrics:
  D.add_arm_contact_safety(cfg)   # the whole-arm contact sensor (penalty unused here)
cfg.terminations = {}
env = ManagerBasedRlEnv(cfg=cfg, device=DEV)
env.reset()

ARMS = ("robot", "robot2")
SL = {"robot": (slice(0, 3), 3), "robot2": (slice(4, 7), 7)}
ORIG = env.scene.env_origins[0]
jaw_open = {"robot": True, "robot2": True}
m = env.sim.mj_model
d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]


def pos(entity, site=None):
  e = env.scene[entity]
  p = e.data.root_link_pos_w[0] if site is None else e.data.site_pos_w[0, e.site_names.index(site)]
  return (p - ORIG).cpu().numpy()


def tool(arm):
  return pos(arm, "tool_midpoint")


def ring():
  return pos("block")


def sync():
  d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  d.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
  if m.nmocap:
    d.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
    d.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
  mujoco.mj_forward(m, d)


def holding(arm):
  """Both of this arm's jaws touching the ring."""
  sync()
  touch = set()
  for c in d.contact[: d.ncon]:
    a, b = names[c.geom1], names[c.geom2]
    for jaw, other in ((a, b), (b, a)):
      if other.startswith("block/") and jaw.startswith(f"{arm}/"):
        side = jaw.split("/")[1][:2]
        if side in ("lg", "rg"):
          touch.add(side)
  return touch == {"lg", "rg"}


# --- collision flagging (see module docstring) -----------------------------------
mp = copy.deepcopy(m)
dp = mujoco.MjData(mp)
_ALL = 0xFF


def _owner(g):
  return names[g].split("/")[0] if "/" in names[g] else "terrain"


ARM_GEOMS = [i for i, n in enumerate(names)
             if _owner(i) in ARMS and n.endswith("_collision")]
for i in ARM_GEOMS:
  mp.geom_contype[i] = mp.geom_conaffinity[i] = _ALL

_PART = {"main_insertion_link": "shaft", "tool_roll_capsule": "wrist (roll)",
         "tool_pitch_capsule": "wrist (pitch)"}
_WORLD = {"block": "ring", "peg": "peg", "goal": "goal", "terrain": "floor"}


def _label(g):
  o, n = _owner(g), names[g].split("/")[-1]
  if o in ARMS:
    arm = "arm 1" if o == "robot" else "arm 2"
    if n.endswith("_box_collision"):
      return f"{arm} {n[: -len('_box_collision')].replace('_', ' ')}"
    part = "jaw" if n[:2] in ("lg", "rg") else _PART.get(n.replace("_collision", ""), n)
    return f"{arm} {part}"
  if o == "peg":
    return "peg post" if "post" in n else "peg plate"
  return _WORLD.get(o, o)


def collisions():
  """{label: deepest penetration (m, <= 0)} for every unexpected overlap now."""
  dp.qpos[:] = d.qpos
  dp.mocap_pos[:] = d.mocap_pos
  dp.mocap_quat[:] = d.mocap_quat
  mujoco.mj_kinematics(mp, dp)
  mujoco.mj_collision(mp, dp)
  out = {}
  for c in dp.contact[: dp.ncon]:
    o1, o2 = _owner(c.geom1), _owner(c.geom2)
    if (o1 in ARMS) == (o2 in ARMS) and (o1 not in ARMS or o1 == o2):
      continue  # world-world, or an arm against itself
    l1, l2 = sorted((_label(c.geom1), _label(c.geom2)))
    if l1.endswith(" jaw") and l2 == "ring":
      continue  # the grasp
    key = f"{l1} <-> {l2}"
    out[key] = min(out.get(key, 0.0), c.dist)
  return out


# label -> list of [first step, last step, deepest m, phase caption at start]
events, active = {}, {}


def track():
  sync()
  now = collisions()
  for k, pen in now.items():
    if k in active:
      ev = active[k]
      ev[1], ev[2] = n_step, min(ev[2], pen)
    else:
      ev = active[k] = [n_step, n_step, pen, caption]
      events.setdefault(k, []).append(ev)
  for k in list(active):
    if k not in now:
      del active[k]
  return now


# --- per-arm geometry ----------------------------------------------------------
def jaw_axis(arm):
  """World-xy closing axis of this arm's jaw (fixed: orientation is not actuated)."""
  gx = env.sim.data.geom_xpos[0].cpu().numpy()
  lg = [i for i, n in enumerate(names) if n.startswith(f"{arm}/lg") and "_collision" in n]
  rg = [i for i, n in enumerate(names) if n.startswith(f"{arm}/rg") and "_collision" in n]
  a = (gx[rg].mean(0) - gx[lg].mean(0))[:2]
  return a / np.linalg.norm(a)


def base_rot(arm):
  """World->base rotation of the arm's Cartesian action frame (yaw only)."""
  q = env.scene[arm].data.root_link_quat_w[0].cpu().numpy()
  yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# The teacher's ring-boundary table, borrowed without its single-arm __init__.
_rt = SimpleNamespace(dev=env.device)


def grasp_point(arm, rank=0, side=None):
  """liftpeg_teacher.grasp_target for one arm: the thinner of the two places the
  ring wall crosses this arm's jaw axis (`rank` 1 takes the other one), or with
  `side` (world xy direction) the one lying furthest that way from the centre.

  The point is placed in the ring's frame and rotated by its FULL orientation,
  not just its yaw: a ring held at one wall hangs tilted, and the mid-air
  handover grips it there."""
  q = env.scene["block"].data.root_link_quat_w[0].cpu().numpy().astype(np.float64)
  yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, q)
  rot = rot.reshape(3, 3)
  ax = jaw_axis(arm)
  o = ring()
  cands = []
  for flip in (0.0, math.pi):
    la = math.atan2(ax[1], ax[0]) + flip - yaw          # same direction, ring frame
    outer = float(T.LiftpegTeacher._outer_radius(_rt, torch.tensor([la], device=env.device))[0])
    thick = max(outer - _rt._bore, 1e-4)
    r = _rt._bore + thick * T.GRIP_FRAC
    local = np.array([r * math.cos(la), r * math.sin(la), T.GRASP_LOCAL_Z])
    cands.append((thick, o + rot @ local))
  if side is not None:
    cands.sort(key=lambda c: -float(np.dot(c[1][:2] - o[:2], side)))
  else:
    cands.sort(key=lambda c: c[0])
  return cands[rank % 2][1]


# --- recording -------------------------------------------------------------------
import imageio.v2 as iio  # noqa: E402

# Frames go straight to the encoder: a 1500-step run at 960x720 is ~1.5 GB if
# they are kept in RAM until the end.
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
writer = iio.get_writer(OUT, fps=round(1.0 / (env.step_dt * FRAME_EVERY)),
                        codec="libx264", quality=8, macro_block_size=8)
n_frames, caption, n_step, n_arm_contact = 0, "", 0, 0
ret = 0.0
t1_success_seen = [False]
if TASK == "Surgicallab-Dual-Task1-Naive":
  from surgical_lab.tasks.lift.config.psm import dual_task1 as T1
elif TASK == "Surgicallab-Dual-Task1-Staged":
  from surgical_lab.tasks.lift.config.psm import dual_task1_staged as T1
else:
  T1 = None
_seen = {"held": None, "phi_band": -1, "lost": False}
renderer = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.lookat[:] = ORIG.cpu().numpy() + np.array([-0.025, 0.012, 0.055])
cam.distance, cam.azimuth, cam.elevation = 0.34, 250.0, -22.0
FONT = ImageFont.load_default(size=22)


def grab():
  global n_frames
  sync()
  renderer.update_scene(d, cam)
  img = Image.fromarray(renderer.render())
  dr = ImageDraw.Draw(img)
  dr.rectangle((0, 0, W, 40), fill=(0, 0, 0))
  dr.text((12, 8), caption, fill=(255, 255, 255), font=FONT)
  r = ring() * 1000
  dr.text((W - 330, 8), f"t={n_step * env.step_dt:5.1f}s  ring z={r[2]:5.1f} mm",
          fill=(200, 220, 255), font=FONT)
  dr.rectangle((0, H - 40, W, H), fill=(150, 0, 0) if active else (0, 0, 0))
  if active:  # arm part touching the scene (jaw grazing the post etc.)
    dr.text((12, H - 32), "contact: " + ";  ".join(sorted(active)), fill=(255, 255, 255),
            font=FONT)
  dr.text((W - 290, H - 32), f"arm-arm contacts: {n_arm_contact}",
          fill=(120, 255, 120) if n_arm_contact == 0 else (255, 120, 120), font=FONT)
  writer.append_data(np.asarray(img))
  n_frames += 1


LEAD_MAX = 0.004
TERM = {"robot": "cartesian_delta", "robot2": "cartesian_delta_2"}


def lead(arm):
  """How far the arm's integrated Cartesian target leads its actual pose (base
  frame). This task runs with max_target_lead=None, like the real arm, so the
  target is a bare integrator: commanding `error / pos_scale` against a blocked
  tool winds it up without limit (measured: 180 mm by the time a 500-step
  stalled descent turned into a lift). step() therefore caps the lead at
  LEAD_MAX itself -- the env's own anti-windup, applied by the script instead
  of by changing the task."""
  t = env.action_manager.get_term(TERM[arm])
  q = env.scene[arm].data.joint_pos[:, t._joint_ids]
  return (t._ee_target_tf[0, :3, 3] - t._fk_fn(q)[0, :3, 3]).cpu().numpy()


def step(active=None, err=None):
  global n_step
  a = np.zeros(8)
  for arm in ARMS:
    a[SL[arm][1]] = 1.0 if jaw_open[arm] else -1.0
  if active is not None:
    ld = lead(active)
    new = ld + np.clip(base_rot(active) @ err / T.POS_SCALE, -1.0, 1.0) * T.POS_SCALE
    n = np.linalg.norm(new)
    if n > LEAD_MAX:
      new *= LEAD_MAX / n
    a[SL[active][0]] = (new - ld) / T.POS_SCALE
  _, rew, *_ = env.step(torch.as_tensor(a, dtype=torch.float32, device=env.device)[None])
  global ret
  ret += float(rew[0])
  n_step += 1
  if T1 is not None:
    t = T1.terms(env)
    if bool(t["handed"][0]):
      print(f"        [reward] phase 0 -> 1 (handoff bonus) at step {n_step}", flush=True)
    if bool(t["success"][0]) and not t1_success_seen[0]:
      t1_success_seen[0] = True
      print(f"        [reward] SUCCESS fired at step {n_step}", flush=True)
    if "phi" in t:
      held = tuple(bool(x) for x in t["held"][0])
      if held != _seen["held"]:
        _seen["held"] = held
        print(f"        [reward] held (strict) arm1={held[0]} arm2={held[1]} at step {n_step}, "
              f"phi {float(t['phi'][0]):.2f}", flush=True)
      band = int(float(t["phi"][0]))
      if band > _seen["phi_band"]:
        _seen["phi_band"] = band
        print(f"        [reward] phi reached {float(t['phi'][0]):.2f} at step {n_step}", flush=True)
      if "released" in t and bool(t["released"][0]):
        print(f"        [reward] arm 1 released at ring z {float(t['ring'][0, 2]) * 1000:.1f} mm, "
              f"release cost {float(t['rel_cost'][0]):.2f}, tilt {float(t['tilt'][0]):.1f} deg",
              flush=True)
      if bool(t["handed"][0]):
        print(f"        [reward] handoff: ring tilt {float(t['tilt'][0]):.1f} deg", flush=True)
      if bool(t["lost"][0]) and not _seen["lost"]:
        _seen["lost"] = True
        print(f"        [reward] LOST fired at step {n_step}", flush=True)
  global n_arm_contact
  n_arm_contact += int(D.arm_arm_contact(env)[0])
  track()
  if n_step % FRAME_EVERY == 0:
    grab()


def say(msg):
  global caption
  caption = msg
  print(f"[{n_step:5d}] {msg}", flush=True)


def goto(arm, target, xy_tol=None, z_tol=None, max_steps=500, target_fn=None):
  """Servo `arm`'s tool to `target` (or target_fn() each step). Arrival is
  checked per axis group, like the teacher's phase gates: xy within xy_tol AND
  |dz| within z_tol, either one skipped when None. True on arrival."""
  for i in range(max_steps):
    tgt = target_fn() if target_fn else target
    err = tgt - tool(arm)
    if DEBUG and i % 20 == 0:
      print(f"        {arm} i={i} tool {np.round(tool(arm) * 1000, 1)} tgt {np.round(tgt * 1000, 1)} "
            f"ring {np.round(ring() * 1000, 1)}", flush=True)
    if ((xy_tol is None or np.linalg.norm(err[:2]) < xy_tol)
        and (z_tol is None or abs(err[2]) < z_tol)):
      return True
    step(arm, err)
  print(f"        ! {arm} did not reach {np.round(tgt * 1000, 1)} mm "
        f"(err {np.round((tgt - tool(arm)) * 1000, 1)} mm)")
  return False


def wait(n):
  for _ in range(n):
    step()


def pick(arm, rise, label, tries=4, choose=None):
  """Approach, descend, close, lift by `rise` (tool z, m). True if held.
  `choose(k)` gives the grasp point for attempt k (default: thinnest wall first)."""
  choose = choose or (lambda k: grasp_point(arm, k))
  for k in range(tries):
    jaw_open[arm] = True
    say(f"{label}: approach ring" + (f" (retry {k})" if k else ""))
    goto(arm, None, xy_tol=T.XY_TOL, z_tol=2e-3,
         target_fn=lambda k=k: np.array([*choose(k)[:2], T.APPROACH_Z]))
    say(f"{label}: descend")
    # Capped like the teacher's RETRY_STEPS: a descent that has not converged
    # by then is resting on the wall and should try the other side instead.
    # Looser than the teacher's 1.0 / 0.8 mm gates: one env here, not 256, and
    # the jaw settles on the wall 1-1.5 mm off the computed point, straddling it.
    if not goto(arm, None, xy_tol=2.0e-3, z_tol=1.5e-3, max_steps=T.RETRY_STEPS,
                target_fn=lambda k=k: choose(k)):
      say(f"{label}: descent blocked, trying the other side")
      goto(arm, np.array([*tool(arm)[:2], T.APPROACH_Z]), z_tol=2e-3)
      continue
    say(f"{label}: close jaw")
    jaw_open[arm] = False
    held = choose(k)  # latched: from here the tool follows waypoints
    for _ in range(T.CLOSE_STEPS):
      step(arm, held - tool(arm))
    say(f"{label}: lift")
    z0 = ring()[2]
    goto(arm, np.array([*held[:2], held[2] + rise]), z_tol=T.Z_TOL)
    if holding(arm) and ring()[2] > z0 + 0.5 * rise:
      return True
    say(f"{label}: grasp slipped, reopening")
    jaw_open[arm] = True
    goto(arm, np.array([*tool(arm)[:2], T.APPROACH_Z]), z_tol=2e-3)
    wait(30)
  return False


def _over(arm, xy, z):
  """Tool target that puts the held ring's CENTRE over xy, live (the tool grips
  the wall, ~7 mm off centre)."""
  return lambda: np.array([*(np.asarray(xy) + tool(arm)[:2] - ring()[:2]), z])


def carry(arm, xy, label):
  """Carry the held ring at its current height until its centre is over xy."""
  say(f"{label}: carry")
  return goto(arm, None, xy_tol=T.GOAL_XY_TOL, max_steps=800,
              target_fn=_over(arm, xy, tool(arm)[2]))


def place(arm, xy, label, by_ring=False):
  """Carry the held ring so its CENTRE lands on xy, set it down, release.

  by_ring: lower until the RING is at its measured release height, rather than
  the tool at PLACE_Z -- PLACE_Z assumes the teacher's grip height on the ring,
  which a ring that has slipped in the jaw (as after the mid-air handover) no
  longer has."""
  def over(z):
    return _over(arm, xy, z)

  carry(arm, xy, label)
  say(f"{label}: set down")
  if by_ring:
    goto(arm, None, z_tol=T.Z_TOL, target_fn=lambda: _over(
      arm, xy, tool(arm)[2] - (ring()[2] - (T.PLACE_Z - T.GRASP_LOCAL_Z)))())
  else:
    goto(arm, None, z_tol=T.Z_TOL, target_fn=over(T.PLACE_Z))
  say(f"{label}: release")
  jaw_open[arm] = True
  wait(T.RELEASE_STEPS)
  say(f"{label}: back off")
  goto(arm, np.array([*tool(arm)[:2], T.APPROACH_Z]), z_tol=2e-3)


# --- task 1 ----------------------------------------------------------------------
# Parking spots, env frame. Arm 2 sits on the -x side with its shaft rising
# toward -x, so it parks further out that way; arm 1 parks on the +x side.
PARK2 = np.array([-0.085, 0.012, 0.100])
PARK1 = np.array([0.035, 0.012, 0.100])


def peg_xy():
  return pos("peg")[:2]


def on_peg():
  """Ring resting down on the post: low, and its bore centred on the post."""
  off = np.linalg.norm(ring()[:2] - peg_xy())
  return bool(ring()[2] < LR.Z_REST + 0.003 and off < LS.RING_CLEARANCE + 0.0005), off


def reinsert(arm, label):
  """Carry the held ring over the post and lower its bore down onto it."""
  # The ring's underside has to clear the 36 mm post top on the way over.
  lift = max(0.0, LR.CLEAR_Z + 0.006 - ring()[2])
  if lift > 0:
    say(f"{label}: raise the ring clear of the post")
    goto(arm, tool(arm) + np.array([0.0, 0.0, lift]), z_tol=T.Z_TOL)
  say(f"{label}: carry over the peg")
  goto(arm, None, xy_tol=0.6e-3, max_steps=900, target_fn=_over(arm, peg_xy(), tool(arm)[2]))
  say(f"{label}: lower onto the post")
  # Live in xy (keep the bore centred on the post) and by RING height in z.
  goto(arm, None, z_tol=0.8e-3, max_steps=900, target_fn=lambda: _over(
    arm, peg_xy(), tool(arm)[2] - (ring()[2] - LR.Z_REST))())
  say(f"{label}: release")
  jaw_open[arm] = True
  wait(T.RELEASE_STEPS)
  say(f"{label}: back off")
  goto(arm, np.array([*tool(arm)[:2], T.APPROACH_Z]), z_tol=2e-3)


def task1():
  off0 = np.linalg.norm(ring()[:2] - peg_xy())
  print(f"        ring starts {off0 * 1000:.1f} mm off the post centre")
  say("Arm 2: park out of the way")
  goto("robot2", PARK2, xy_tol=2e-3, z_tol=2e-3, max_steps=600)

  goal = pos("goal")
  ok1 = pick("robot", T.LIFT_RISE, "Arm 1")
  if ok1 and DROP:
    # Negative check for the reward: let go over the goal from carry height.
    carry("robot", goal[:2], "Arm 1")
    say("Arm 1: DROP from carry height (DEMO_DROP=1)")
    jaw_open["robot"] = True
    wait(150)
    on_goal = bool(LR.block_at_goal(env)[0])
    print(f"        ring at {np.round(ring() * 1000, 1)} mm; on goal: {on_goal}")
    return f"DROP check: ring on goal {on_goal} (arm 2 skipped)"
  if ok1:
    place("robot", goal[:2], "Arm 1")
  on_goal = bool(LR.block_at_goal(env)[0])
  print(f"        ring at {np.round(ring() * 1000, 1)} mm; on goal: {on_goal}")
  say("Arm 1: move out of the way")
  goto("robot", PARK1, xy_tol=2e-3, z_tol=2e-3, max_steps=600)

  ok2 = ok3 = False
  off = float("nan")
  if ok1:
    ok2 = pick("robot2", T.LIFT_RISE, "Arm 2")
    if ok2:
      reinsert("robot2", "Arm 2")
      ok3, off = on_peg()
  say(f"done: ring {'back on the peg' if ok3 else 'NOT on the peg'} "
      f"({off * 1000:.1f} mm off centre)")
  goto("robot2", PARK2, xy_tol=3e-3, z_tol=3e-3, max_steps=300)
  wait(40)
  return (f"arm1 lift+place on goal: {'OK' if ok1 and on_goal else 'FAIL'} | "
          f"arm2 pick off goal: {'OK' if ok2 else 'FAIL'} | "
          f"arm2 re-insert on peg: {'OK' if ok3 else 'FAIL'}")


t0 = time.time()
for _ in range(10):  # the post-reset settle (see probe_dual_psm.py)
  step()
say("start")
wait(15)
result = task1()
print(f"[result] {result} | {n_step} steps, {time.time() - t0:.0f}s wall")
print(f"[arm-arm] contact steps reported by the task's sensor: {n_arm_contact}")
if T1 is not None:
  print(f"[reward] {TASK}: return {ret:.2f}; episode sums "
        + ", ".join(f"{k} {float(v[0]):.2f}" for k, v in env.reward_manager._episode_sums.items()))

print(f"[collisions] {sum(len(v) for v in events.values())} contact episode(s) "
      f"over {len(events)} pair(s):" if events else "[collisions] none")
for k, evs in sorted(events.items(), key=lambda kv: kv[1][0][0]):
  steps = sum(e[1] - e[0] + 1 for e in evs)
  worst = min(e[2] for e in evs)
  print(f"  {k:36s} {len(evs):3d}x  {steps:5d} steps  deepest {-worst * 1000:5.2f} mm  "
        f"first at step {evs[0][0]} ({evs[0][3]})")

writer.close()
print(f"[video] {OUT}: {n_frames} frames")
env.close()
