# Environment setup: start here

Never trained a policy on a remote GPU before? This folder takes you from zero
to a training run you can watch in wandb, in order. Each page is short, and
explains *why* as well as *what*, because most of these tools are new.

| # | Page | What you get out of it | Time |
|---|---|---|---|
| 1 | [DH machines and SSH](01_dh_machines_and_ssh.md) | A shell on a lab GPU machine, and the house rules for sharing them | 5 min |
| 2 | [Install uv and the code](02_install_uv_and_code.md) | The repo and its Python environment on that machine | 10-15 min (mostly downloading) |
| 3 | [wandb](03_wandb.md) | Your runs logging to the team's shared dashboard | 5 min |
| 4 | [tmux and launching runs](04_tmux_and_running.md) | A run that keeps going after you log out | 5 min |
| 5 | [Monitoring, videos and checkpoints](05_monitoring_and_videos.md) | Knowing whether a run worked, and seeing what the policy does | - |
| 6 | [Troubleshooting](06_troubleshooting.md) | Fixes for every problem we have actually hit | - |

Then do the first real run: [training the dummy reach policy](../training_dummy_reach_dh.md).

## The whole thing on one screen

Once you have done pages 1-3 once, every run after that is just:

```bash
ssh <utorid>@dh2020pcXX.utm.utoronto.ca      # 1. log in to a free GPU machine
nvidia-smi                                   #    check nobody else is using the GPU
tmux new -s train                            # 4. FIRST start tmux; everything below runs inside it
cd ~/Documents/CSC413/csc413-project/surgical_lab
git pull && uv sync                          # 2. latest code + env (fast if nothing changed)
set -a; . scripts/a3_liftpeg_env.sh; set +a  #    scene settings, every time
MUJOCO_GL=egl uv run train <Task-Name> --env.scene.num-envs 1024 \
  --agent.seed 42 --agent.run-name <method>-<task>-s42-<initials> --video True
# then Ctrl-b, let go, d  -> detached; watch it in wandb (page 5)
```

## Glossary

| Word | Meaning here |
|---|---|
| **DH machines** | The Deerfield Hall lab desktops (`dh2020pcXX`). Each has an NVIDIA GPU (dh2020pc03: RTX 4080, 16 GB). Shared by everyone in the department. |
| **SSH** | Logging in to another computer's terminal over the network. |
| **uv** | The tool that installs this project's exact Python packages into a private folder (`.venv`). Like pip + venv, but fast and reproducible. |
| **wandb** (Weights & Biases) | A website that records every training run's numbers, settings and videos, so the whole team can see and compare them. |
| **tmux** | Keeps a terminal session alive on the remote machine after you disconnect. |
| **env / envs** | A copy of the simulated scene. Training runs 1024+ of them in parallel on the GPU. (Not the same as a Python *environment*, `.venv`.) |
| **Iteration** | One round of PPO: every env takes 24 steps, then the networks are updated once. |
| **Checkpoint** | A saved copy of the policy's weights (`model_<iter>.pt`, `best.pt`). |
| **Task** | A registered scene + reward, e.g. `Surgicallab-Dual-Reach-Dummy`. |
