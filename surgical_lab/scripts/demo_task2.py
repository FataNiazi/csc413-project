"""Scripted Task 2 (mid-air handover) on the lab (side-by-side) layout, to mp4.

  arm 2: park out of the way;
  arm 1: unthread the ring off the peg, gripping the wall on arm 1's side, and
         hold it up at the handover point;
  arm 2: come down onto the opposite wall and close, so both arms hold it;
  arm 1: let go and back away (the ring stays in the air, in arm 2's jaw);
  arm 2: carry the ring to the red goal, set it down flat, let go.

A feasibility check for training Task 2 in this layout, and the reference the
Task 2 reward is checked against (DEMO_TASK=Surgicallab-Dual-Task2-Handover).
Adapted from demo_task1.py (Task 1 is untouched) and the old facing-layout
mid-air demo in demo_dual_handoff.py.

The ring is a rounded triangle and both arms' jaws close along (nearly) the
same world line, so the two grips sit on opposite walls along that line. At
some ring yaws one of those walls is a corner too thick to straddle.
DEMO_YAW=best (default) turns the ring to the yaw where both walls are thin,
as the old demo did; DEMO_YAW=keep leaves the reset yaw (random, like training).

Run from surgical_lab/ (CPU is fine, a few minutes):

  set -a; . scripts/a3_liftpeg_env.sh; set +a
  MUJOCO_GL=egl uv run python scripts/demo_task2.py      # OUT=... SEED=... DEMO_YAW=keep
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
OUT = os.environ.get("OUT", "renders/dual_psm/task2.mp4")
DEMO_YAW = os.environ.get("DEMO_YAW", "best")
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
if TASK == "Surgicallab-Dual-Task2-Handover":
  from surgical_lab.tasks.lift.config.psm import dual_task2_handover as T1
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


LEAD_MAX = float(os.environ.get("DEMO_LEAD_MAX", "0.004"))
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


class Dropped(Exception):
  pass


def _over(arm, xy, z):
  """Tool target that puts the held ring's CENTRE over xy, live (the tool grips
  the wall, ~7 mm off centre)."""
  def target():
    off = tool(arm) - ring()
    if np.linalg.norm(off[:2]) > 0.020 or off[2] > 0.030:
      raise Dropped(f"ring left {arm}'s jaw: tool - ring {np.round(off * 1000, 1)} mm")
    return np.array([*(np.asarray(xy) + off[:2]), z])
  return target


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


# --- task 2 -------------------------------------------------------------------------
PARK2 = np.array([-0.085, 0.012, 0.100])
HOME1 = np.array([0.040, 0.012, 0.100])
# Where arm 1 holds the ring up for the exchange (ring centre, env frame). Between
# the peg and the goal, at a height that keeps the ring clear of the post.
HANDOFF_XY = np.array([float(x) for x in os.environ.get("DEMO_HANDOFF_XY", "-0.020,0.010").split(",")])
YAW_OFF = math.radians(float(os.environ.get("DEMO_YAW_OFF", "0")))   # turn the ring this far past the best yaw
HANDOFF_RISE = float(os.environ.get("DEMO_HANDOFF_RISE", "0.045"))   # tool lift at pickup


def handover_yaw():
  """Turn the ring to the yaw where BOTH walls on the shared jaw line are thin
  enough to grip (see demo_dual_handoff.handover_yaw)."""
  th = np.linspace(0.0, 2 * math.pi, 720, endpoint=False)
  out = T.LiftpegTeacher._outer_radius(_rt, torch.tensor(th, dtype=torch.float32,
                                                         device=env.device)).cpu().numpy()
  thick = out - _rt._bore
  worst = np.maximum(thick, np.roll(thick, -360))
  i = int(np.argmin(worst))
  ax = jaw_axis("robot")
  yaw = math.atan2(ax[1], ax[0]) - th[i] + YAW_OFF
  blk = env.scene["block"]
  st = torch.zeros(1, 13, device=env.device)
  st[0, :3] = blk.data.root_link_pos_w[0]
  st[0, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  blk.write_root_state_to_sim(st)
  print(f"        ring turned to yaw {math.degrees(yaw):.1f} deg: walls on the jaw line "
        f"{thick[i] * 1000:.2f} / {np.roll(thick, -360)[i] * 1000:.2f} mm")


def walls_on_jaw_line():
  """Wall thickness at the two places the ring crosses arm 1's jaw line, now."""
  q = env.scene["block"].data.root_link_quat_w[0].cpu().numpy()
  yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
  ax = jaw_axis("robot")
  out = []
  for flip in (0.0, math.pi):
    la = math.atan2(ax[1], ax[0]) + flip - yaw
    o = float(T.LiftpegTeacher._outer_radius(_rt, torch.tensor([la], device=env.device))[0])
    out.append(o - _rt._bore)
  return out


TRACE = os.environ.get("DEMO_TRACE", "0") == "1"
G2_DZ = float(os.environ.get("DEMO_G2_DZ", "0"))              # arm 2 grips this much lower (m, < 0)
G2_LIVE = os.environ.get("DEMO_G2_LIVE", "0") == "1"          # arm 2 follows the ring as it tips under it
STOP_HOLD = os.environ.get("DEMO_STOP", "") == "hold"         # stop after the hold check
FAST = os.environ.get("DEMO_FAST", "0") == "1"                # place straight after the release
SHARE_LIFT = float(os.environ.get("DEMO_SHARE_LIFT", "0"))    # arm 2 lifts this much before arm 1 lets go


def _pads(arm):
  sync()
  touch = {}
  for c in d.contact[: d.ncon]:
    a, b = names[c.geom1], names[c.geom2]
    for jaw, other in ((a, b), (b, a)):
      if other.startswith("block/") and jaw.startswith(f"{arm}/"):
        side = jaw.split("/")[1][:3]
        touch[side] = min(touch.get(side, 1.0), c.dist)
  return {k: round(v * 1000, 2) for k, v in sorted(touch.items())}


def tilt_deg():
  q = env.scene["block"].data.root_link_quat_w[0].cpu().numpy()
  return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))


def yaw_report(tag):
  """Ring yaw, and the wall each arm's side of the jaw line crosses."""
  q = env.scene["block"].data.root_link_quat_w[0].cpu().numpy()
  yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
  w = walls_on_jaw_line()        # [along +axis, along -axis]; the axis points to -x (arm 2's side)
  if jaw_axis("robot")[0] > 0:
    w = w[::-1]
  print(f"        [yaw] {tag}: ring yaw {yaw:.1f} deg, tilt {tilt_deg():.1f}, wall at -x (arm 2 side) "
        f"{w[0] * 1000:.2f} mm, at +x (arm 1 side) {w[1] * 1000:.2f} mm", flush=True)


def _trace(tag, held2):
  q = env.scene["block"].data.root_link_quat_w[0].cpu().numpy()
  tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
  jw = {a: float(env.scene[a].data.joint_pos[0, env.scene[a].joint_names.index("jaw")]) for a in ARMS}
  print(f"        [{tag}] ring {np.round(ring() * 1000, 1)} tilt {tilt:.1f}  tool2-held2 "
        f"{np.round((tool('robot2') - held2) * 1000, 1)}  tool2-ring {np.round((tool('robot2') - ring()) * 1000, 1)}"
        f"  jaw {jw['robot']:.2f}/{jw['robot2']:.2f}  pads1 {_pads('robot')}  pads2 {_pads('robot2')}", flush=True)


def task2():
  a1, a2 = jaw_axis("robot"), jaw_axis("robot2")
  print(f"        jaw closing axes: arm 1 {np.round(a1, 3)}, arm 2 {np.round(a2, 3)} "
        f"({math.degrees(math.acos(abs(float(np.dot(a1, a2))))):.1f} deg apart)")
  if DEMO_YAW == "best":
    say("set-up: turn the ring so both walls on the jaw line are grippable")
    handover_yaw()
    wait(10)
  w = walls_on_jaw_line()
  print(f"        walls on the jaw line: {w[0] * 1000:.2f} / {w[1] * 1000:.2f} mm")
  say("Arm 2: park out of the way")
  goto("robot2", PARK2, xy_tol=2e-3, z_tol=2e-3, max_steps=600)
  yaw_report("before arm 1 grips")

  # Arm 1 takes the wall on ITS side (+x, away from arm 2), so arm 2, coming in
  # from -x, takes the near wall and never reaches across arm 1's jaw.
  away = np.array([1.0, 0.0])
  ok1 = pick("robot", HANDOFF_RISE, "Arm 1",
             choose=lambda k: grasp_point("robot", side=away if k % 2 == 0 else -away))
  ok_both = ok_hand = ok_goal = False
  flat = float("nan")
  if ok1:
    yaw_report("after arm 1 grips and lifts")
    print(f"        arm 1 is on the {'+x' if tool('robot')[0] > ring()[0] else '-x (arm 2 side!)'} wall")
    say("Arm 1: carry to the handover point")
    try:
      carry("robot", HANDOFF_XY, "Arm 1")
    except Dropped as e:
      print(f"        ! DROPPED: {e}")
    z_hand = ring()[2]
    print(f"        ring at the handover point: {np.round(ring() * 1000, 1)} mm, tilt {tilt_deg():.1f} deg")

    def g2():  # the wall on the far side of the ring from arm 1's tool
      d = ring()[:2] - tool("robot")[:2]
      return grasp_point("robot2", side=d / (np.linalg.norm(d) + 1e-9))

    jaw_open["robot2"] = True
    if TRACE:
      _trace("arm 1 at the handover point", g2())
    say("Arm 2: approach from above")
    goto("robot2", None, xy_tol=T.XY_TOL, z_tol=2e-3, max_steps=700,
         target_fn=lambda: g2() + np.array([0.0, 0.0, 0.025]))
    if TRACE:
      _trace("before arm 2 descends", g2())
    say("Arm 2: descend onto the ring")
    held2 = g2() + np.array([0.0, 0.0, G2_DZ])   # latched: a live target would chase the hanging ring down
    live2 = lambda: g2() + np.array([0.0, 0.0, G2_DZ])
    goto("robot2", held2, xy_tol=2.0e-3, z_tol=1.5e-3, max_steps=T.RETRY_STEPS,
         target_fn=live2 if G2_LIVE else None)
    if G2_LIVE:
      held2 = live2()
    if TRACE:
      _trace("arm 2 down, jaw open", held2)
    say("Arm 2: close jaw (both arms holding)")
    jaw_open["robot2"] = False
    for _ in range(T.CLOSE_STEPS):
      step("robot2", (live2() if G2_LIVE else held2) - tool("robot2"))
    ok_both = holding("robot") and holding("robot2")
    yaw_report("both holding")
    print(f"        both jaws on the ring: {ok_both}")
    if TRACE:
      _trace("after arm 2 closed", held2)
    if SHARE_LIFT > 0:
      say("Arm 2: take the weight")
      tgt = tool("robot2") + np.array([0.0, 0.0, SHARE_LIFT])
      for _ in range(40):
        step("robot2", tgt - tool("robot2"))
      if TRACE:
        _trace("after arm 2 took the weight", held2)

    say("Arm 1: release")
    jaw_open["robot"] = True
    for i in range(T.RELEASE_STEPS):
      step()
      if TRACE and i % 4 == 0:
        _trace(f"release +{i}", held2)
    say("Arm 1: back away")
    goto("robot", tool("robot") + np.array([0.0, 0.0, 0.025]), z_tol=2e-3)
    if not FAST:
      goto("robot", HOME1, xy_tol=2e-3, z_tol=2e-3, max_steps=600)
    yaw_report("arm 2 alone")
    # Hold still for 2 s: the ring must stay in arm 2's jaw, not creep out of it.
    say("Arm 2: hold (checking the ring does not slip)")
    g0 = tool("robot2")[2] - ring()[2]
    for i in range(10 if FAST else 100):
      step()
      if TRACE and i % 20 == 0:
        _trace(f"hold +{i}", held2)
    slip = (tool("robot2")[2] - ring()[2]) - g0
    ok_hand = bool(holding("robot2") and ring()[2] > 0.020 and abs(slip) < 0.001)
    print(f"        slip during the hold: {slip * 1000:.2f} mm")
    print(f"        arm 2 holding alone: {ok_hand}; ring z {ring()[2] * 1000:.1f} mm "
          f"(was {z_hand * 1000:.1f} at the handover); contact flag {holding('robot2')}; tilt {tilt_deg():.1f} deg")
    if ok_hand and not STOP_HOLD:
      if ring()[2] < LR.CLEAR_Z + 0.002:
        say("Arm 2: raise the ring clear of the post")
        goto("robot2", tool("robot2") + np.array([0.0, 0.0, LR.CLEAR_Z + 0.004 - ring()[2]]),
             z_tol=T.Z_TOL)
      try:
        place("robot2", pos("goal")[:2], "Arm 2", by_ring=True)
      except Dropped as e:
        print(f"        ! DROPPED: {e}")
      ok_goal = bool(LR.block_at_goal(env)[0])
      q = env.scene["block"].data.root_link_quat_w[0].cpu().numpy()
      flat = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[1] ** 2 + q[2] ** 2)))))
  final = ring()
  say(f"done: ring at ({final[0] * 1000:.0f}, {final[1] * 1000:.0f}) mm; on the goal: {ok_goal}")
  goto("robot2", PARK2, xy_tol=3e-3, z_tol=3e-3, max_steps=300)
  wait(40)
  return (f"arm1 pick: {'OK' if ok1 else 'FAIL'} | both holding: {'OK' if ok_both else 'FAIL'} | "
          f"mid-air handover: {'OK' if ok_hand else 'FAIL'} | arm2 place on goal: "
          f"{'OK' if ok_goal else 'FAIL'} (tilt {flat:.1f} deg)")


t0 = time.time()
for _ in range(10):  # the post-reset settle (see probe_dual_psm.py)
  step()
say("start")
wait(15)
result = task2()
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
