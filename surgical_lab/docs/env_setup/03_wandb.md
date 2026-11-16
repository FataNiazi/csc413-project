# 3. wandb (Weights & Biases)

## What it is and why we use it

Every training run sends its numbers (reward, distances, success rate, ...),
its exact settings, GPU usage and videos to wandb.ai. Everyone on the team sees
everyone's runs on one dashboard, so we can compare methods and seeds without
passing log files around.

- Team: **`hardik-narkhede-university-of-toronto`**
- Project: **`csc413-dual-psm`** (created automatically by the first run)
- Dashboard: https://wandb.ai/hardik-narkhede-university-of-toronto/csc413-dual-psm

## Join the team (once)

1. Hardik sends an invite to the email you gave him. Accept it (make a wandb
   account with that email if you don't have one).
2. Check: in wandb's left sidebar, under **Teams**, you should see
   `hardik-narkhede-university-of-toronto`.

Why a separate team: a wandb team lives inside an *organization*. Hardik's main
account belongs to the MedCVR lab's organization, and inviting you to a team
there would have added you to the lab's org too. So the team was made on a
fresh account with a UofT email, outside any lab.

## Log in on the DH machine (once per machine)

1. In the browser, logged in to **your own** wandb account, open
   https://wandb.ai/authorize (or your name, top right, then **API keys**) and copy
   the key.
2. On the DH machine, from `surgical_lab/`:

```bash
uv run wandb login          # paste the key; nothing shows as you paste, that's normal
```

It saves the key to `~/.netrc`, so you never type it again on that machine.
**Your key is like a password: use only your own, never share it, never
commit it.** Wrong account? `uv run wandb login --relogin`.

## Send your runs to the team, not your personal space

```bash
echo 'export WANDB_ENTITY=hardik-narkhede-university-of-toronto' >> ~/.bashrc
source ~/.bashrc
echo $WANDB_ENTITY          # should print the team name
```

Without this, runs land in your personal wandb space where nobody else can see
them. (`export` alone only lasts for the current terminal; putting it in
`~/.bashrc` makes every new shell and tmux session have it. Only keep one
`WANDB_ENTITY` line in there.)

## No internet on the machine?

```bash
export WANDB_MODE=offline   # logs to disk only
# later, from a machine with internet:
uv run wandb sync logs/rsl_rl/<experiment>/<run dir>/wandb/offline-run-*
```

## Naming runs

`--agent.run-name <method>-<task>-s<seed>-<initials>`, e.g.
`ppo-dummyreach-s42-hn`. Runs from different people and seeds are then easy to
filter and compare. Delete smoke-test runs from the dashboard afterwards so
they don't clutter the charts.

Next: [tmux and launching runs](04_tmux_and_running.md).
