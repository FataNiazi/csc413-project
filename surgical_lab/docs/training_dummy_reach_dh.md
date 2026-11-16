# Training a PPO policy on the dummy reach task (DH lab machines)

This is the first training run for the project. It is a **pipeline check, not a
result**: the task (`Surgicallab-Dual-Reach-Dummy`) just asks each arm to move
its tool to its own random target, so if PPO cannot learn it, something in the
setup is broken. Once it passes, we know the sim, PPO, wandb logging and
checkpointing all work on the DH machines, and we can move on to the real
Task 1 reward.

**The run passes if**, within a few hundred iterations, in wandb:

| Metric | Start | Should reach |
|---|---|---|
| `Episode_Metrics/dist_1_mm`, `dist_2_mm` (tool-to-target distance at episode end) | ~15 mm | a few mm |
| `Episode_Metrics/reached_rate` (both arms within 3 mm) | 0 | climbing toward 1 |
| `Episode_Metrics/arm_contact_rate` (episodes where the arms touched) | some | ~0 |
| `Train/mean_reward` | ~0.2 | rising, then flat |

If the distances stay flat at ~15 mm for 200+ iterations, stop and post in the
group chat; do not keep tuning.

What the task looks like (scripted, not trained): `dual_reach_dummy.mp4` in the
shared folder, or render it yourself with `scripts/demo_dummy_reach.py`.

> Not yet verified on a DH machine: this guide was tested end to end on a
> CPU-only laptop (4 training iterations, local logging). The GPU run and
> wandb upload are what this first run is checking.

**First time on a DH machine?** Do the [environment setup
guide](env_setup/README.md) first (SSH, uv, wandb, tmux explained step by
step). This page is the short version plus what's specific to this task.

---

## 0. Ground rules

- **One person per machine.** Before launching, run `nvidia-smi`: if a python
  process is already using the GPU, pick another `dh2020pcXX`.
- **Long runs at night / off-hours.** These are shared lab desktops; someone
  may sit down at the machine during the day. The dummy run is short (see
  below) and fine any time.
- **Always run inside `tmux`**, so the run survives your SSH connection
  dropping.
- **Never commit the wandb API key** (or any key). `wandb login` stores it in
  `~/.netrc`, outside the repo.

## 1. One-time setup on a DH machine

SSH in the way you normally do for course work (machine names look like
`dh2020pc03`), then:

```bash
nvidia-smi                 # expect an NVIDIA GPU (RTX 4080 on dh2020pc03), driver >= 570
df -h ~ ; quota -s         # you need ~8 GB free for the Python env + cache
```

If your home quota is too small for that, put uv's cache on local disk for
every command below:

```bash
export UV_CACHE_DIR=/tmp/$USER/uv
```

Install uv (user-local, no sudo) and get the code:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc           # or open a new shell, so `uv` is on PATH

git clone https://github.com/FataNiazi/csc413-project.git
cd csc413-project
git checkout dev-hardik    # until PR #1 is merged into main
cd surgical_lab
uv sync                    # creates .venv here; first time takes a few minutes
```

Check that the GPU is visible from the env:

```bash
uv run python -c "import torch, warp; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); warp.init()"
```

You want `True NVIDIA GeForce RTX ...` and warp listing a `cuda:0` device.

## 2. wandb (once per machine)

```bash
uv run wandb login                    # paste your key from https://wandb.ai/authorize
export WANDB_ENTITY=hardik-narkhede-university-of-toronto        # our team, so runs land there, not in your personal account
```

Put the `export WANDB_ENTITY=...` line in your `~/.bashrc` so you don't forget
it. Runs go to the project **`csc413-dual-psm`** inside the team (https://wandb.ai/hardik-narkhede-university-of-toronto/csc413-dual-psm once the first run exists).

No internet on the machine? Use `export WANDB_MODE=offline`, then later
`uv run wandb sync logs/rsl_rl/dual_reach_dummy/<run dir>/wandb/offline-run-*`
from a machine that has it.

## 3. Smoke test (2 minutes)

From `csc413-project/surgical_lab`:

```bash
set -a; . scripts/a3_liftpeg_env.sh; set +a     # scene settings: always source this first
uv run train Surgicallab-Dual-Reach-Dummy \
  --env.scene.num-envs 64 --agent.max-iterations 3 \
  --agent.run-name smoke-<your-initials>
```

It should print a few `Learning iteration` blocks and a wandb run link. Delete
the smoke run in wandb afterwards if you like.

## 4. The real run

```bash
tmux new -s train
cd ~/csc413-project/surgical_lab
set -a; . scripts/a3_liftpeg_env.sh; set +a
MUJOCO_GL=egl uv run train Surgicallab-Dual-Reach-Dummy \
  --env.scene.num-envs 1024 \
  --agent.seed 42 \
  --agent.run-name ppo-dummyreach-s42-<your-initials> \
  --video True --video-length 200 --video-interval 2400
```

`--video True` records a 4 s clip (one episode) of env 0 every 2400 env
steps (~100 iterations) and uploads it to wandb, in the run's **video** panel.
It shows both arms from the front, with each arm's target as a sphere (cyan =
arm 1, magenta = arm 2, sized to the 3 mm "reached" tolerance), so you can see
whether the policy is actually reaching and keeping the arms apart. Rendering
only happens while a clip is recording, so it barely slows training.
`MUJOCO_GL=egl` is required for rendering over SSH (no display).

- Detach with `Ctrl-b` then `d`; reattach later with `tmux attach -t train`.
- Default length is 500 iterations. Speed on the 4080 is not measured yet;
  expect tens of minutes, not hours. Note the `Iteration time` it prints and
  post it, it tells us how long real runs will take.
- `--env.scene.num-envs`: 1024 is a safe start. If GPU memory (`nvidia-smi`)
  is well under the 16 GB, 2048 or 4096 trains faster per wall-clock; if you
  get CUDA out-of-memory, halve it.
- Run name convention: `<method>-<task>-s<seed>-<initials>`, so runs from
  different people and seeds are easy to compare in wandb.

What PPO is learning from (all in
`src/surgical_lab/tasks/lift/config/psm/dual_reach_dummy.py`):

- **Observation** (actor and critic): both tool positions, both targets, both
  target-minus-tool vectors, both arms' joint angles, the last action.
- **Action** (8-D): `[dx, dy, dz, jaw]` for arm 1, then for arm 2. Position
  deltas, 25 mm/s max.
- **Reward**: closeness to target per arm (coarse + fine term), a small
  penalty for jerky actions, and **-10 plus episode end for any arm-arm
  contact** (whole arm, not just the tips).
- **Episode**: 4 s (200 steps); targets resample each episode.

## 5. If it gets interrupted

Checkpoints are saved every 100 iterations to
`logs/rsl_rl/dual_reach_dummy/<timestamp>_<run-name>/` (`model_<iter>.pt` =
newest, `best.pt` = best mean reward). To continue the newest run:

```bash
uv run train Surgicallab-Dual-Reach-Dummy --env.scene.num-envs 1024 --agent.resume True
```

## 6. Looking at the trained policy

Option A, a video file (headless; not yet tried on a DH machine):

```bash
MUJOCO_GL=egl uv run play Surgicallab-Dual-Reach-Dummy --agent trained \
  --checkpoint-file logs/rsl_rl/dual_reach_dummy/<run dir>/best.pt \
  --num-envs 1 --video True --video-length 400
```

Option B, the live 3D viewer in your own browser over an SSH tunnel. On your
laptop:

```bash
ssh -L 8080:localhost:8080 <you>@<dh machine>
```

then on the DH machine run the same `play` command with `--viewer viser`
instead of `--video True`, and open http://localhost:8080 on your laptop.

Either way, the policy should send both tools to their targets and stop there,
with the arms never touching.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Wrong machine (no NVIDIA GPU) or a broken `uv sync`; rerun `uv sync` and recheck. |
| `CUDA out of memory` | Someone else is on the GPU (`nvidia-smi`), or lower `--env.scene.num-envs`. |
| `Disk quota exceeded` during `uv sync` | `export UV_CACHE_DIR=/tmp/$USER/uv`, `rm -rf .venv`, `uv sync` again. |
| Run not showing in the team on wandb | `WANDB_ENTITY` not set in that shell. |
| `arm_contact_rate` stays high | The arms keep touching. Post it; it is a reward-design question, not a bug to patch locally. |

## 8. What to post when you're done

- Machine name, GPU, `num-envs`, and the `Iteration time` it printed.
- The wandb run link.
- Pass / fail against the table at the top.
