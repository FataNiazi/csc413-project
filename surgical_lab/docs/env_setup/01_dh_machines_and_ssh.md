# 1. DH machines and SSH

## Why the DH machines

Training needs an NVIDIA GPU (the simulator, MuJoCo Warp, only runs on
NVIDIA). Our supervisor asked us not to use lab (MedCVR) compute, so every
group in the course has the same resources: the Deerfield Hall lab machines,
which we can all log in to with our UTORid.

## Logging in

From a terminal on your laptop (macOS/Linux terminal, or PowerShell / WSL on
Windows):

```bash
ssh <utorid>@dh2020pc03.utm.utoronto.ca
```

Use your UTORid password. Other machines follow the same pattern
(`dh2020pc01`, `dh2020pc02`, ...). Warnings about `des3-cbc-sha1` at login are
harmless.

## Check the machine before you use it

```bash
nvidia-smi
```

- You should see an NVIDIA GPU (e.g. `NVIDIA GeForce RTX 4080 ... 16376MiB`).
- Look at the **Processes** table at the bottom. `Xorg` / `gnome-shell` are just
  the desktop and are fine. **If there is a `python` process using GBs of
  memory, someone else is training: pick another machine.**

## House rules

- **One person training per machine.** Agree in the group chat who is on which
  `dh2020pcXX`.
- **Long runs at night or off-hours.** These are shared desktops; someone may
  sit down at one during the day (and could even reboot it). Short runs like
  the dummy task (~5 min) are fine any time.
- **Always launch training inside `tmux`** (page 4), so a dropped Wi-Fi
  connection does not kill your run.
- **Never put passwords or API keys in the repo, the group chat, or a
  screenshot.**

## One-time shell fixes

Run these once per machine (they are saved in `~/.bashrc`, which runs every
time you log in):

```bash
# Some laptop terminals (e.g. Ghostty, kitty) send a terminal type the DH
# machines don't know ("'xterm-ghostty': unknown terminal type"), which breaks
# `clear` and `tmux`. Harmless if yours is fine.
echo 'export TERM=xterm-256color' >> ~/.bashrc

# Your home folder is on a network drive; this silences a harmless uv warning.
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc

source ~/.bashrc
```

Next: [install uv and the code](02_install_uv_and_code.md).
