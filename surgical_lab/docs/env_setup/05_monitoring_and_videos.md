# 5. Monitoring, videos and checkpoints

## In wandb

Open the run link the terminal printed (or the project dashboard). Panels are
grouped by prefix; the ones that matter:

| Group | What's in it | Read it as |
|---|---|---|
| **Episode_Metrics** | Task-specific measurements, e.g. `dist_1_mm`, `reached_rate`, `arm_contact_rate` | **Did it work?** Look here first. |
| **Train** | `mean_reward`, `mean_episode_length` | Rising then flat = learning then converged |
| **Episode_Reward** | Each reward term separately | Which part of the reward the policy is earning or paying |
| **Episode_Termination** | Why episodes ended (`time_out`, `arm_contact`, ...) | Lots of early terminations = something going wrong |
| **Loss** | PPO's internal losses | Only look if something looks broken |
| **video** | Clips from `--video True` | What the policy actually does |
| **System** tab | GPU use and memory, hostname | How hard the GPU is working; which machine ran it |

Each task's guide lists what "passing" looks like (e.g. the table at the top of
the [dummy reach guide](../training_dummy_reach_dh.md)).

A smoke test (3 iterations) only gives ~3 points per chart: short flat lines
are expected there.

## In the terminal

Each `Learning iteration` block prints the same numbers, plus
**`Iteration time`** (seconds per iteration) and an **ETA**. Post the iteration
time with your results: it tells us how long bigger runs will take.

`nvidia-smi` (in another SSH session) shows GPU memory: if your run uses much
less than 16 GB you can raise `--env.scene.num-envs` next time.

## Checkpoints

Saved to `logs/rsl_rl/<experiment>/<date>_<run-name>/`:

- `model_<iter>.pt`: the newest (saved every 100 iterations; older ones are
  pruned). This is what `--agent.resume True` continues from.
- `best.pt`: the highest mean reward so far.

They're also uploaded to the wandb run (**Files** / artifacts).

## Watching a trained policy

As a video file (no screen needed):

```bash
MUJOCO_GL=egl uv run play <Task> --agent trained \
  --checkpoint-file logs/rsl_rl/<experiment>/<run dir>/best.pt \
  --num-envs 1 --video True --video-length 400
```

The mp4 goes to that run's `videos/play/` folder. Copy it to your laptop
(run this **on your laptop**):

```bash
scp <utorid>@dh2020pc03.utm.utoronto.ca:~/Documents/CSC413/csc413-project/surgical_lab/logs/rsl_rl/<experiment>/<run dir>/videos/play/*.mp4 .
```

Or live in your browser: SSH in with a tunnel,
`ssh -L 8080:localhost:8080 <utorid>@dh2020pc03.utm.utoronto.ca`, run the same
`play` command with `--viewer viser` instead of `--video True`, and open
http://localhost:8080 on your laptop.

Next: [troubleshooting](06_troubleshooting.md).
