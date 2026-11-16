# Surgical Lab: two-PSM peg/ring scene

Simulation for the CSC413 project: two dVRK Patient Side Manipulators (PSMs) in
[mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp, around the peg-ring
task (lift the green ring off the yellow peg, pass it between the arms).

This is a trimmed snapshot of the MedCVR lab's internal `surgical_lab` code: the
package, the vendored `mujoco_warp`, and the scripts for the dual-arm task. It
builds on the lab's single-arm peg-lift task and its reward functions.

**New to remote GPUs, uv, wandb or tmux?** Start with
[`docs/env_setup/`](docs/env_setup/README.md): a step-by-step guide from SSH-ing
into a DH lab machine to watching your first training run in wandb, plus fixes
for every problem we've hit.

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
  `robot2`, mounted like the lab cart: both PSMs face the same way, side by
  side, PSM2 250 mm to PSM1's right (seen from the front) with its shaft tilted
  ~39 deg so the tools meet over the peg. Placement: `DUAL_PSM_LAYOUT`
  (`side`, default, or the old `facing`), `DUAL_PSM2_BASE_DX`,
  `DUAL_PSM2_TOOL_XY`. Arm 2's wrist is levelled so its jaws point straight
  down like arm 1's (`DUAL_PSM2_WRIST_LEVEL=1`). In `side`, every arm link has collision geometry
  against the other arm (boxes fitted to the upper links).
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
| `scripts/demo_task1.py` | Scripted Task 1 on the lab layout, recorded to mp4 (`MUJOCO_GL=egl`): arm 1 lifts the ring off the peg and sets it on the goal, arm 2 picks it up and re-inserts it on the peg. Counts arm-arm contacts (task sensor) and flags any arm part touching the scene. |
| `scripts/demo_dual_handoff.py` | (Old `facing` layout, set automatically.) Scripted demos recorded to mp4 (`MUJOCO_GL=egl`). `DEMO=relay` (default): arm 1 places the ring on the goal, arm 2 carries it on. `DEMO=midair`: arm 1 lifts it, arm 2 takes it in mid-air and places it. Flags every unexpected collision. |

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

### Pipeline check: `Surgicallab-Dual-Reach-Dummy`

Step-by-step for the team (DH lab machines, wandb, tmux, resuming, viewing the
policy): [`docs/training_dummy_reach_dh.md`](docs/training_dummy_reach_dh.md).

A throwaway objective for checking the training setup before the real
bimanual reward exists: each arm reaches its own random target (a 30 x 30 x
20 mm box around its home tool position, resampled every 4 s episode). Expect
`Episode_Metrics/dist_1_mm` and `dist_2_mm` to fall from ~15 mm to a few mm,
and `reached_rate` (both arms within 3 mm) to climb, within a few hundred
iterations. See `src/surgical_lab/tasks/lift/config/psm/dual_reach_dummy.py`.

```bash
# once per machine: wandb login   (or export WANDB_API_KEY=...; never commit it)
export WANDB_ENTITY=hardik-narkhede-university-of-toronto    # log to the team, not your personal account
tmux new -s train
MUJOCO_GL=egl uv run train Surgicallab-Dual-Reach-Dummy --env.scene.num-envs 1024 \
  --agent.run-name ppo-dummyreach-s42 --video True --video-length 200 --video-interval 2400
```

Any contact between the two arms costs `DUAL_ARM_CONTACT_PENALTY` (default 10,
about one whole episode's best return) and ends the episode
(`DUAL_ARM_CONTACT_TERMINATE=1`); `Episode_Metrics/arm_contact_rate` is the
fraction of episodes that had one. It covers the whole arm: instrument,
parallelogram, carriage and base.

It logs to the wandb project `csc413-dual-psm`. `--video True` uploads a clip of
env 0 to wandb every ~100 iterations, filmed from the front with the two targets
drawn as spheres (dual tasks set this view in `liftpeg_dual._add_video_view`). Checkpoints land in
`logs/rsl_rl/dual_reach_dummy/<timestamp>_<run-name>/`: the newest
`model_<iter>.pt` (every 100 iterations) plus `best.pt`. To continue after a
crash or reboot, add `--agent.resume True` (it loads the newest run's newest
checkpoint). On CPU it runs (`--gpu-ids None`), but at ~55 s per iteration
with 16 envs, so use it only as a smoke test.

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
