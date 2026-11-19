"""Evaluate a trained Task 1 policy: success, completion time, and which arm
parts touch when the two arms collide.

Runs the policy's mean action (no exploration noise) on the training task's
own env (same episode length, terminations and reward), for an equal number of
episodes per env so short (successful) episodes are not over-counted. Every
episode is labelled by what ended it:

  arm_contact > lost > success > time_out   (if two fire on the same step, the
                                             earlier one in this list counts)

and every arm-arm contact is broken down by the pair of links that touched
(e.g. `main_insertion_link x main_insertion_link` = shaft on shaft), read from
the solver's contact list at the moment the contact termination fires, before
the env resets.

On a GPU machine, from surgical_lab/ (after `set -a; . scripts/a3_liftpeg_env.sh; set +a`):

  uv run python scripts/eval_task1.py \\
    --checkpoint logs/rsl_rl/dual_task1_staged/<run dir>/model_5999.pt \\
    --num-envs 256 --episodes 1024 --out eval_task1.json
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import math
import os
from dataclasses import asdict

import torch

PRIORITY = ("nan_state", "arm_contact", "lost", "success", "time_out")
ARM_PREFIX = {"robot/": 1, "robot2/": 2}


def parse():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--task", default="Surgicallab-Dual-Task1-Staged")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--episodes", type=int, default=1024,
                  help="total; rounded up to a whole number per env")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", default=None, help="optional JSON file for the numbers")
  return ap.parse_args()


class ContactFlagger:
  """Names the link pairs behind each arm-arm contact."""

  def __init__(self, env):
    import mujoco

    m = env.sim.mj_model
    arm, label = [0] * m.ngeom, [""] * m.ngeom
    for i in range(m.ngeom):
      g = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
      for pre, k in ARM_PREFIX.items():
        if g.startswith(pre):
          arm[i] = k
          body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or ""
          label[i] = body.split("/", 1)[-1]
    self.arm = torch.tensor(arm, device=env.device)
    self.label = label

  def pairs(self, env, env_mask: torch.Tensor) -> dict[int, list[tuple[str, str, float]]]:
    """{env id: [(arm 1 link, arm 2 link, penetration mm), ...]} for envs in env_mask."""
    d = env.sim.data
    n = int(d.nacon[0])
    out: dict[int, list] = {}
    if n == 0:
      return out
    g = d.contact.geom[:n].long()
    wid = d.contact.worldid[:n].long()
    dist = d.contact.dist[:n]
    a1, a2 = self.arm[g[:, 0]], self.arm[g[:, 1]]
    cross = ((a1 == 1) & (a2 == 2)) | ((a1 == 2) & (a2 == 1))
    cross &= env_mask[wid]
    for k in cross.nonzero().flatten().tolist():
      i, j = int(g[k, 0]), int(g[k, 1])
      if int(a1[k]) == 2:
        i, j = j, i
      out.setdefault(int(wid[k]), []).append(
        (self.label[i], self.label[j], -float(dist[k]) * 1000.0))
    return out


def main():
  args = parse()
  os.environ.setdefault("MUJOCO_GL", "egl")
  import surgical_lab.tasks  # noqa: F401  (registers the tasks)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task)

  # Record which terminations fire on each step, and the contact pairs at the
  # moment arm_contact fires (the env resets right after, inside step()).
  fired: dict[str, torch.Tensor] = {}
  flagger_box: list[ContactFlagger] = []
  contact_now: dict[int, list] = {}
  staged = None
  if args.task == "Surgicallab-Dual-Task1-Staged":
    from surgical_lab.tasks.lift.config.psm import dual_task1_staged as staged

  def wrap(name, fn):
    @functools.wraps(fn)
    def g(env, *a, **kw):
      out = fn(env, *a, **kw)
      fired[name] = out.clone()
      if name == "arm_contact" and bool(out.any()):
        contact_now.update(flagger_box[0].pairs(env, out))
      if staged is not None and "_phase1" not in fired:
        fired["_phase1"] = staged.terms(env)["phase1"].clone()
      return out
    return g

  for name, term in cfg.terminations.items():
    if callable(term.func) and not isinstance(term.func, type):
      term.func = wrap(name, term.func)

  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  flagger_box.append(ContactFlagger(env))
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(venv, asdict(agent_cfg), device=args.device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=args.device)
  policy = runner.get_inference_policy(device=args.device)

  N = args.num_envs
  per_env = math.ceil(args.episodes / N)
  dt = env.step_dt
  steps = torch.zeros(N, dtype=torch.long, device=args.device)
  done_count = torch.zeros(N, dtype=torch.long, device=args.device)
  episodes = []        # (reason, seconds, handed_off)
  pair_count = collections.Counter()
  pair_depth = collections.defaultdict(float)
  contact_examples = []

  obs, _ = venv.reset()
  while int((done_count < per_env).sum()) > 0:
    fired.clear()
    contact_now.clear()
    with torch.inference_mode():
      act = policy(obs)
    obs, _, dones, _ = venv.step(act)
    steps += 1
    ids = dones.nonzero().flatten().tolist()
    for i in ids:
      if int(done_count[i]) < per_env:
        reason = next((r for r in PRIORITY if r in fired and bool(fired[r][i])), "other")
        handed = bool(fired["_phase1"][i]) if "_phase1" in fired else None
        episodes.append((reason, float(steps[i]) * dt, handed))
        if reason == "arm_contact":
          prs = contact_now.get(i, [])
          for a, b, depth in set((a, b, round(dd, 2)) for a, b, dd in prs):
            pair_count[(a, b)] += 1
            pair_depth[(a, b)] = max(pair_depth[(a, b)], depth)
          if len(contact_examples) < 20:
            contact_examples.append({"t_s": round(float(steps[i]) * dt, 2),
                                     "handed_off": handed,
                                     "pairs": sorted({f"{a} x {b}" for a, b, _ in prs})})
      done_count[i] += 1
      steps[i] = 0
    if len(episodes) and len(episodes) % 256 == 0 and ids:
      print(f"  {len(episodes)}/{per_env * N} episodes", flush=True)

  n = len(episodes)
  by = collections.Counter(r for r, _, _ in episodes)
  t_succ = sorted(t for r, t, _ in episodes if r == "success")
  handed = [h for _, _, h in episodes if h is not None]

  def pct(q):
    return t_succ[min(len(t_succ) - 1, int(q * len(t_succ)))] if t_succ else float("nan")

  res = {
    "task": args.task, "checkpoint": args.checkpoint, "episodes": n,
    "policy": "deterministic (mean action)",
    "rates": {r: by.get(r, 0) / n for r in PRIORITY + ("other",)},
    "handoff_rate": (sum(handed) / len(handed)) if handed else None,
    "success_time_s": {"mean": sum(t_succ) / len(t_succ) if t_succ else None,
                       "median": pct(0.5), "p10": pct(0.1), "p90": pct(0.9)},
    "arm_contact_pairs": [
      {"arm1_link": a, "arm2_link": b, "episodes": c, "max_penetration_mm": round(pair_depth[(a, b)], 2)}
      for (a, b), c in pair_count.most_common()],
    "arm_contact_examples": contact_examples,
  }

  print(f"\n=== {args.task}: {n} episodes ({per_env} per env x {N} envs), deterministic policy ===")
  print(f"checkpoint: {args.checkpoint}")
  for r in PRIORITY + ("other",):
    if by.get(r, 0):
      print(f"  {r:12s} {by[r] / n:6.3f}  ({by[r]} episodes)")
  if res["handoff_rate"] is not None:
    print(f"  handoff      {res['handoff_rate']:6.3f}  (ring flat on the goal, arm 1 let go)")
  if t_succ:
    s = res["success_time_s"]
    print(f"  completion time on success: mean {s['mean']:.2f} s, median {s['median']:.2f}, "
          f"p10 {s['p10']:.2f}, p90 {s['p90']:.2f}")
  if pair_count:
    print("  arm-arm contacts by link pair (episodes; deepest penetration):")
    for p in res["arm_contact_pairs"]:
      print(f"    arm 1 {p['arm1_link']:22s} x arm 2 {p['arm2_link']:22s} "
            f"{p['episodes']:4d}   {p['max_penetration_mm']:.2f} mm")
  elif by.get("arm_contact", 0):
    print("  arm-arm contact fired but no contact pair was found (sensor margin?)")
  else:
    print("  no arm-arm contact")
  if args.out:
    with open(args.out, "w") as f:
      json.dump(res, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
