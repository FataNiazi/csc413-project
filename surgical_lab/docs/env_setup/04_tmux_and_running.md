# 4. tmux and launching runs

## Why tmux

A program started over SSH dies when the connection drops (closed laptop,
flaky Wi-Fi). `tmux` runs a terminal session **on the DH machine** that keeps
going after you disconnect; you reconnect to it later.

(Clusters use a job scheduler, `sbatch`, for this. The DH machines are plain
desktops with no scheduler: you start the run yourself, inside tmux.)

## The order matters

1. **Start tmux first**: `tmux new -s train`. A fresh shell opens, with a
   **green status bar at the bottom** (session name on the left, e.g.
   `[train]`). That bar means "you are inside tmux".
2. **Inside it**, `cd` to the repo and launch training.
3. **Detach** once iterations are printing: press **`Ctrl-b`, let go, then
   `d`**. The bar disappears, you're back in your normal shell, and the run
   keeps going. You can now log out.
4. **Later**: SSH back in and `tmux attach -t train` to see the output again.

Anything started *inside* tmux survives logout; anything started outside it is
tied to your SSH connection.

| Do this | To |
|---|---|
| `tmux new -s train` | Start a session named `train` |
| `Ctrl-b`, let go, `d` | **Detach** (run keeps going) |
| `tmux attach -t train` | Reattach |
| `tmux ls` | List your sessions (after a correct detach: `train: 1 windows ...`) |
| `exit` (inside the session) | Close the session when the run is done |

**Careful: `Ctrl-d` is not detach.** `Ctrl-d` means "end of input": it closes
the shell inside tmux, which ends the whole session (you'll see `[exited]`). If
a run was going in it, that run stops too.

## Launching a run

```bash
tmux new -s train
# --- everything below runs inside tmux ---
cd ~/Documents/CSC413/csc413-project/surgical_lab
set -a; . scripts/a3_liftpeg_env.sh; set +a
MUJOCO_GL=egl uv run train Surgicallab-Dual-Reach-Dummy \
  --env.scene.num-envs 1024 --agent.seed 42 \
  --agent.run-name ppo-dummyreach-s42-<initials> \
  --video True --video-length 400 --video-interval 2400
# then Ctrl-b, let go, d
```

Piece by piece:

- `set -a; . scripts/a3_liftpeg_env.sh; set +a`: loads the scene's layout
  settings (goal position, camera, ...) as environment variables. **Needed every
  time**, from `surgical_lab/` (it's a relative path).
- `MUJOCO_GL=egl`: lets the video renderer work without a screen (over SSH).
  Only matters with `--video True`.
- `uv run train <Task>`: mjlab's training command (PPO from rsl-rl).
- `--env.scene.num-envs 1024`: parallel copies of the scene. More is faster
  per wall-clock until the GPU is full; halve it if you get "CUDA out of
  memory".
- `--agent.seed`: different seeds give independent runs of the same setup.
- `--agent.run-name`: the name in wandb (see the naming rule on page 3).
- `--video True ...`: films env 0 for 400 steps (two 4 s episodes, so at least one whole one; clips start mid-episode) every 2400 steps and uploads
  the clip to wandb.
- Other useful flags: `--agent.max-iterations N` (length; default 500 for the
  dummy task), `--agent.resume True` (continue the newest run).
  `uv run train <Task> --help` lists everything.

## What to expect when it starts

1. **First launch on a machine: a minute or two of `Module ... (compiled)`
   lines**, with nothing else happening. That's Warp compiling the simulator
   for the GPU, not a hang. Later launches say `(cached)` and take seconds.
2. Tables describing the task (events, actions, observations, rewards).
3. A `wandb: View run at https://wandb.ai/...` link: open it.
4. `Learning iteration X/500` blocks, one per iteration, with the numbers.
5. At the end, a wandb summary and your prompt back. (A short run finishing
   and giving you your terminal back is normal.)

**Always do a 3-iteration smoke test first** on a new machine or a new task:
add `--env.scene.num-envs 64 --agent.max-iterations 3`. If that prints three
iterations and a wandb link, the real run will work. It finishes in seconds,
so no tmux needed.

Next: [monitoring, videos and checkpoints](05_monitoring_and_videos.md).
