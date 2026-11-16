# 6. Troubleshooting

Every row here is something we actually ran into.

| Symptom | Cause | Fix |
|---|---|---|
| `Command 'uv' not found ... snap install astral-uv` | uv isn't installed on that machine | Don't use snap (needs admin). `curl -LsSf https://astral.sh/uv/install.sh \| sh` then `source $HOME/.local/bin/env` ([page 2](02_install_uv_and_code.md)) |
| `'xterm-ghostty': unknown terminal type` (or `clear` / `tmux` refuse to work) | Your laptop's terminal type isn't known on the DH machines | `echo 'export TERM=xterm-256color' >> ~/.bashrc && source ~/.bashrc` |
| `warning: Failed to hardlink files; falling back to full copy` | Home folder is on a network drive | Harmless. Silence it with `export UV_LINK_MODE=copy` in `~/.bashrc` |
| `Disk quota exceeded` during `uv sync` | Home folder quota | `export UV_CACHE_DIR=/tmp/$USER/uv`, `rm -rf .venv`, `uv sync` again |
| Training prints only `Module ... (compiled)` lines for a minute or two | First launch compiles the GPU code | Wait. Next launch says `(cached)` and starts in seconds. To check it's alive: `top -u $USER` in a second SSH session shows python busy |
| `torch.cuda.is_available()` prints `False` | No NVIDIA GPU on that machine, or a broken install | `nvidia-smi`; if there's a GPU, `rm -rf .venv && uv sync` |
| `CUDA out of memory` | Someone else on the GPU, or too many envs | Check `nvidia-smi`; halve `--env.scene.num-envs` |
| `tmux new -s train` prints `[exited]`, or the run stopped when you tried to leave | `Ctrl-d` closes the shell (and the session) | Detach is **`Ctrl-b`, let go, `d`** ([page 4](04_tmux_and_running.md)) |
| Run shows up in your personal wandb, not the team's | `WANDB_ENTITY` not set in that shell | `echo $WANDB_ENTITY`; add it to `~/.bashrc` ([page 3](03_wandb.md)) |
| `wandb login` shows the wrong account | An old key in `~/.netrc` | `uv run wandb login --relogin` |
| Run fails with a scene/goal-position error, or the scene looks wrong | Forgot the scene settings | `set -a; . scripts/a3_liftpeg_env.sh; set +a` from `surgical_lab/`, every time |
| EGL / OpenGL error with `--video True` | Rendering without a screen | Make sure `MUJOCO_GL=egl` is on the command. If it still fails, train without `--video` and record afterwards with `play` ([page 5](05_monitoring_and_videos.md)) |
| `git pull` says local changes would be overwritten | You edited files on the DH machine | `git stash` (keeps your edits aside), then `git pull` |
| Inviting someone to a wandb team says they'll also join another organization | That team lives inside that org | Use our team, which is outside any lab org ([page 3](03_wandb.md)) |

Still stuck: post the **exact command** and the **last ~30 lines of output**
in the group chat (never with an API key in it).
