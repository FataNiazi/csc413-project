# 2. Install uv and the code

## Install uv (once per machine, no admin rights needed)

`uv` installs Python packages. The DH machines don't have it, and the message
suggesting `snap install astral-uv` needs admin rights, so install it just for
yourself:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env      # makes `uv` available in this shell
uv --version                     # e.g. uv 0.13.8
```

New shells pick it up automatically from now on. (If `curl` fails, use
`python3 -m pip install --user uv` and `export PATH="$HOME/.local/bin:$PATH"`.)

## Get the code

```bash
mkdir -p ~/Documents/CSC413 && cd ~/Documents/CSC413
git clone https://github.com/FataNiazi/csc413-project.git
cd csc413-project
git checkout dev-hardik          # where the sim lives until PR #1 is merged into main
cd surgical_lab
```

Later, to get teammates' updates: `git pull` (from inside the repo).

## Create the Python environment: `uv sync`

```bash
uv sync
```

What it does: reads `pyproject.toml` (which packages the project needs) and
`uv.lock` (the exact version of each one), creates a private `.venv/` folder
here, and installs exactly those versions into it: PyTorch with CUDA, the
simulator (mjlab, MuJoCo Warp from `third_party/`), the PPO library (rsl-rl)
and wandb. Nothing is installed system-wide, so it can't break anything else on
the shared machine, and everyone gets identical versions.

- First time: downloads several GB and takes ~5-10 min on the DH network drive.
- Re-run it after any `git pull` that changes `pyproject.toml` or `uv.lock`.
  Running it when nothing changed is harmless and takes seconds.
- To run anything inside the environment, prefix it with `uv run`
  (e.g. `uv run train ...`, `uv run python script.py`).

A warning `Failed to hardlink files; falling back to full copy` is harmless
(your home folder is on a network drive). It goes away with
`UV_LINK_MODE=copy` from page 1.

## Check that the GPU is visible

```bash
uv run python -c "import torch, warp; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); warp.init()"
```

You want `True NVIDIA GeForce RTX ...` and, in Warp's list, a `"cuda:0"` device.
If you get `False`, see [troubleshooting](06_troubleshooting.md).

Next: [wandb](03_wandb.md).
