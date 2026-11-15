# Surgical Lab: two-PSM peg/ring scene

Simulation for the CSC413 project: two dVRK Patient Side Manipulators (PSMs) in
[mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp, around the peg-ring
task (lift the green ring off the yellow peg, pass it between the arms).

This is a trimmed snapshot of the MedCVR lab's internal `surgical_lab` code: the
package, the vendored `mujoco_warp`, and the scripts for the dual-arm task. It
builds on the lab's single-arm peg-lift task and its reward functions.

## Setup

Linux. An NVIDIA GPU (driver >= 570 / CUDA 12.8) for training; CPU works for
building the scene and 1-env checks, just slowly (~0.15 s per step).

```bash
# uv, if you do not have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

cd surgical_lab
uv sync          # creates .venv here; installs mjlab at the pinned commit and
                 # builds mujoco_warp from third_party/ (do not pip install them)
```

Every command below is run from `surgical_lab/` after loading the scene's
layout variables:

```bash
set -a; . scripts/a3_liftpeg_env.sh; set +a
```

## The dual-arm task

`Surgicallab-Liftpeg-Dual-State-PoC`
(`src/surgical_lab/tasks/lift/config/psm/liftpeg_dual.py`):

- The single-arm `Surgicallab-Liftpeg-State-Shaped` task with a second PSM,
  `robot2`, mounted facing the first (rotated 180 deg). Placement:
  `DUAL_PSM2_TOOL_XY`, `DUAL_PSM2_YAW_DEG`.
- Actions, 8-D: `[dx, dy, dz, jaw]` for arm 1, then the same for arm 2.
  Position steps of 0.5 mm per 50 Hz step, in each arm's own base frame; the
  jaw is binary (>= 0 opens). Tool orientation is fixed for now.
- Arm-arm contact on by default (`DUAL_PSM_ARM_CONTACT=0` turns it off).
- State-based: no wall, no wrist-camera bracket.
- **Reward, success and termination still score arm 1 only.** Arm 2's terms
  are the next thing to build.

## Scripts

| Script | What it does |
|---|---|
| `scripts/probe_dual_psm.py` | Smoke test: builds 1 env and checks the arms move independently, both reach the peg and goal, arm-arm contact, and a random-action soak. `SNAP=1 MUJOCO_GL=egl` also saves PNGs. |
| `scripts/demo_dual_handoff.py` | Scripted demos recorded to mp4 (`MUJOCO_GL=egl`). `DEMO=relay` (default): arm 1 places the ring on the goal, arm 2 carries it on. `DEMO=midair`: arm 1 lifts it, arm 2 takes it in mid-air and places it. Flags every unexpected collision. |

```bash
uv run python scripts/probe_dual_psm.py
MUJOCO_GL=egl uv run python scripts/demo_dual_handoff.py
DEMO=midair MUJOCO_GL=egl uv run python scripts/demo_dual_handoff.py
```

Training uses mjlab's CLI, e.g. on a GPU machine:

```bash
uv run train Surgicallab-Liftpeg-Dual-State-PoC --env.scene.num-envs 1024
```

(`--agent zero`/`random` with `uv run play <task> --viewer viser` opens a 3D
browser viewer at http://localhost:8080 without a trained policy.)

## Things to know

- The Cartesian target is an integrator with no anti-windup
  (`max_target_lead=None`, matching the real arm): commanding motion against a
  blocked tool winds the target up without limit.
- Only the jaw and wrist-pitch geoms touch the world; shafts and wrist-roll
  links pass through the peg, ring and floor in the physics. The demo script's
  collision check covers this; training does not yet.
- Every reset is followed by a ~5-step settle of up to 6 mm.
- `uv run python -c "import surgical_lab.tasks; from mjlab.tasks.registry import list_tasks; print(list_tasks())"`
  lists all registered tasks (the older single-arm ones are included).
