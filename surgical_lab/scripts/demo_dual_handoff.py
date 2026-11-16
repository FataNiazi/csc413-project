"""Scripted two-PSM demos on the dual LiftPeg scene, recorded to mp4.

DEMO=relay (default):
  arm 1: unthread the ring off the peg, carry it to the red goal, set it down,
         let go, and back out of the way;
  arm 2: pick the ring up off the goal and carry it to a second spot
         (DUAL_DROP_XY, default 40 mm to +x of the peg), set it down, let go.

DEMO=midair -- a handover with no table in between:
  arm 1: unthread the ring off the peg (gripping the side facing AWAY from
         arm 2) and hold it up at the handoff point (DUAL_HANDOFF_XY);
  arm 2: come down onto the opposite side of the ring and close, so both arms
         hold it; arm 1 then lets go and backs away;
  arm 2: carry the ring to the red goal and set it down.

Grasping reuses the single-arm scripted teacher's geometry (liftpeg_teacher.py:
grip where the ring wall crosses the jaw's closing axis, at its measured grip
height), done per arm -- the teacher itself reads `robot` only and pools every
jaw geom in the scene, so it cannot drive arm 2. Each arm's world-frame error is
rotated into that arm's own base frame before it becomes an action, since arm 2
is mounted rotated.

Collisions are FLAGGED, not just simulated. The PSM collision set only lets the
jaws touch the world -- shafts and wrist links have no world contact at all --
so the sim's own contact list would never show, say, a shaft through the post.
Each step the state is mirrored into a PROBE copy of the model in which every
arm collision geom collides with everything, and any overlap other than a jaw
on the ring (the grasp itself) is flagged: drawn in red on the video while it
lasts, and summarised at the end. The physics is not changed.

Terminations are dropped: the ring reaching the goal is `success` in the task,
which would reset the env halfway through the demo.

Run from the repo root (CPU is fine, ~5 min; MUJOCO_GL=egl for the renderer):

  set -a; . scripts/a3_liftpeg_env.sh; set +a
  MUJOCO_GL=egl uv run python scripts/demo_dual_handoff.py
  DEMO=midair MUJOCO_GL=egl uv run python scripts/demo_dual_handoff.py
"""

import copy
import math
import os

# Written against the original facing-arms layout (arm 2 rotated 180 deg, tools
# 40 mm apart); the task default is now the lab's side-by-side cart.
os.environ.setdefault("DUAL_PSM_LAYOUT", "facing")
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
from surgical_lab.tasks.lift.config.psm import liftpeg_teacher as T

DEV = os.environ.get("DEV", "cuda:0" if torch.cuda.is_available() else "cpu")
DEMO = os.environ.get("DEMO", "relay")
assert DEMO in ("relay", "midair"), DEMO
OUT = os.environ.get("OUT", f"renders/dual_psm/{'handoff' if DEMO == 'relay' else 'handoff_midair'}.mp4")
SEED = int(os.environ.get("SEED", "0"))
HANDOFF_XY = tuple(float(x) for x in os.environ.get(
  "DUAL_HANDOFF_XY", "0.015,0.013").split(","))
DROP_XY = tuple(float(x) for x in os.environ.get("DUAL_DROP_XY", "0.040,0.012").split(","))
DEBUG = os.environ.get("DEBUG", "0") == "1"
FRAME_EVERY = int(os.environ.get("FRAME_EVERY", "2"))   # 50 Hz control -> 25 fps
W, H = 960, 720

torch.manual_seed(SEED)
np.random.seed(SEED)
cfg = load_env_cfg("Surgicallab-Liftpeg-Dual-State-PoC")
cfg.scene.num_envs = 1
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
n_frames, caption, n_step = 0, "", 0
renderer = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.lookat[:] = ORIG.cpu().numpy() + np.array([0.0, 0.010, 0.045])
cam.distance, cam.azimuth, cam.elevation = 0.36, 130.0, -25.0
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
  if active:
    dr.rectangle((0, H - 40, W, H), fill=(150, 0, 0))
    dr.text((12, H - 32), "COLLISION: " + ";  ".join(sorted(active)), fill=(255, 255, 255),
            font=FONT)
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
  env.step(torch.as_tensor(a, dtype=torch.float32, device=env.device)[None])
  n_step += 1
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


# --- the demos -------------------------------------------------------------------
HOME1 = np.array([-0.010, 0.060, 0.090])   # arm 1 parked out of arm 2's way


def relay():
  goal = pos("goal")
  ok1 = pick("robot", T.LIFT_RISE, "Arm 1")
  if ok1:
    place("robot", goal[:2], "Arm 1")
  on_goal = bool(LR.block_at_goal(env)[0])
  print(f"        ring at {np.round(ring() * 1000, 1)} mm; on goal: {on_goal}")

  say("Arm 1: move out of the way")
  goto("robot", HOME1, xy_tol=2e-3, z_tol=2e-3, max_steps=600)

  ok2 = False
  if ok1:
    # Same rise as arm 1, not a token lift: the carry to the drop spot passes over
    # the peg, and a ring carried below the 36 mm post top snags on it.
    ok2 = pick("robot2", T.LIFT_RISE, "Arm 2")
    if ok2:
      place("robot2", DROP_XY, "Arm 2")
  final = ring()
  drop_err = np.linalg.norm(final[:2] - np.asarray(DROP_XY))
  say(f"done: ring at ({final[0] * 1000:.0f}, {final[1] * 1000:.0f}) mm, "
      f"{drop_err * 1000:.1f} mm from the drop spot")
  wait(40)
  return (f"arm1 pick+place: {'OK' if ok1 and on_goal else 'FAIL'} | "
          f"arm2 pick+place: {'OK' if ok2 and drop_err < 0.006 else 'FAIL'}")


def handover_yaw():
  """Turn the ring on its post to the yaw where BOTH walls on the shared jaw line
  are thin enough to grip.

  The open jaw spans ~5 mm (pad centres 4.7-7.5 mm apart) and the ring wall runs
  1.2 mm at a face to 5.4 mm at a corner. Both arms close along the same world
  line, so they grip opposite walls along it -- and for a triangle, a face on
  one side means a corner on the other. At a random yaw one arm therefore meets
  a corner it cannot straddle (the first mid-air run: arm 2 came down on top of
  the wall and pushed the ring out of arm 1's jaws). At the best yaw both walls
  are 2.33 mm. This is demo staging, not something the task guarantees."""
  th = np.linspace(0.0, 2 * math.pi, 720, endpoint=False)
  out = T.LiftpegTeacher._outer_radius(_rt, torch.tensor(th, dtype=torch.float32,
                                                         device=env.device)).cpu().numpy()
  thick = out - _rt._bore
  worst = np.maximum(thick, np.roll(thick, -360))
  i = int(np.argmin(worst))
  ax = jaw_axis("robot")
  yaw = math.atan2(ax[1], ax[0]) - th[i]
  blk = env.scene["block"]
  st = torch.zeros(1, 13, device=env.device)
  st[0, :3] = blk.data.root_link_pos_w[0]
  st[0, 3:7] = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  blk.write_root_state_to_sim(st)
  print(f"        ring turned to yaw {math.degrees(yaw):.1f} deg: walls on the jaw line "
        f"{thick[i] * 1000:.2f} / {np.roll(thick, -360)[i] * 1000:.2f} mm")


def midair():
  say("set-up: turn the ring so both sides are grippable")
  handover_yaw()
  wait(10)
  # Both jaws close along (nearly) the same world line, since arm 2 is arm 1
  # turned 180 deg, so the two grips sit on OPPOSITE walls along that line, ~13 mm
  # apart. Arm 1 takes the wall facing away from arm 2 (-x) so arm 2's shaft,
  # coming in from +x, never has to reach across arm 1's jaw.
  away = np.array([-1.0, 0.0])
  ok1 = pick("robot", T.LIFT_RISE, "Arm 1",
             choose=lambda k: grasp_point("robot", side=away if k % 2 == 0 else -away))
  ok_hand = ok_goal = False
  if ok1:
    carry("robot", HANDOFF_XY, "Arm 1: to the handoff point")
    z_hand = ring()[2]

    def g2():  # the wall on the far side of the ring from arm 1's tool
      d = ring()[:2] - tool("robot")[:2]
      return grasp_point("robot2", side=d / (np.linalg.norm(d) + 1e-9))

    jaw_open["robot2"] = True
    say("Arm 2: approach from above")
    goto("robot2", None, xy_tol=T.XY_TOL, z_tol=2e-3, max_steps=600,
         target_fn=lambda: g2() + np.array([0.0, 0.0, 0.025]))
    say("Arm 2: descend onto the ring")
    # LATCHED, unlike the pick off the post: the ring hangs from arm 1, so a
    # live target follows it down as soon as arm 2's jaw touches it, and arm 2
    # ends up pushing it out of arm 1's grip.
    held2 = g2()
    goto("robot2", held2, xy_tol=2.0e-3, z_tol=1.5e-3, max_steps=T.RETRY_STEPS)
    say("Arm 2: close jaw (both arms holding)")
    jaw_open["robot2"] = False
    for _ in range(T.CLOSE_STEPS):
      step("robot2", held2 - tool("robot2"))
    both = holding("robot") and holding("robot2")
    print(f"        both jaws on the ring: {both}")

    say("Arm 1: release")
    jaw_open["robot"] = True
    wait(T.RELEASE_STEPS)
    say("Arm 1: back away")
    goto("robot", tool("robot") + np.array([0.0, 0.0, 0.025]), z_tol=2e-3)
    goto("robot", HOME1, xy_tol=2e-3, z_tol=2e-3, max_steps=600)
    # "Holding" = the ring is still in the air, at arm 2's jaw. Not the jaw
    # contact flag: it needs both pads touching on the same step, which the
    # teacher measured as noisy (0.73 during a healthy lift), and it read False
    # on a handover that visibly held. The ring does slip ~7 mm in the jaw.
    near = np.linalg.norm(ring()[:2] - tool("robot2")[:2]) < 0.015
    ok_hand = bool(near and ring()[2] > 0.020)
    print(f"        arm 2 holding alone: {ok_hand}; ring z {ring()[2] * 1000:.1f} mm "
          f"(was {z_hand * 1000:.1f} at the handoff); contact flag {holding('robot2')}")
    if ok_hand:
      # The slip leaves the ring's underside below the 36 mm post top, and the
      # way to the goal passes over the peg: raise it back above first.
      say("Arm 2: raise the ring clear of the post")
      goto("robot2", tool("robot2") + np.array([0.0, 0.0, max(0.0, LR.CLEAR_Z + 0.004 - ring()[2])]),
           z_tol=T.Z_TOL)
      place("robot2", pos("goal")[:2], "Arm 2", by_ring=True)
      ok_goal = bool(LR.block_at_goal(env)[0])
  final = ring()
  say(f"done: ring at ({final[0] * 1000:.0f}, {final[1] * 1000:.0f}) mm; "
      f"on the goal: {ok_goal}")
  wait(40)
  return (f"arm1 pick: {'OK' if ok1 else 'FAIL'} | mid-air handoff: "
          f"{'OK' if ok_hand else 'FAIL'} | arm2 place on goal: {'OK' if ok_goal else 'FAIL'}")


t0 = time.time()
for _ in range(10):  # the post-reset settle (see probe_dual_psm.py)
  step()
say("start")
wait(15)
result = relay() if DEMO == "relay" else midair()
print(f"[result] {result} | {n_step} steps, {time.time() - t0:.0f}s wall")

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
