import json
import os
import re
import shutil
import statistics

import torch
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils.wandb_log_writer import WandbLogWriter

from surgical_lab.mdp.recorders import PolicyCamVideoRecorder
from surgical_lab.rl.exporter_utils import get_cartesian_metadata, get_diffik_metadata

VIDEO_RECORDER_TERM = "policy_cam_video"


def _store_config_allow_change(self, env_cfg, train_cfg) -> None:
  """rsl-rl's WandbLogWriter.store_config, with allow_val_change=True.

  A warm restart re-attaches to the SAME wandb run (WANDB_RUN_ID + WANDB_RESUME=allow
  in train_liftpeg_view_fir.sbatch) so the curve stays continuous. Its train_cfg
  legitimately differs from the one stored at link 1 (resume False->True, load_run
  '.*'->'<run dir>'), and wandb refuses that by default with ConfigError -- every
  chain link 2 died on it (fir jobs 58090698, 58224620, 58225183, 58225415), each
  spawning the next link, i.e. a crash loop. store_config runs inside
  OnPolicyRunner.__init__, before a subclass can intervene, hence the class patch.
  """
  from dataclasses import asdict

  wandb.config.update({"train_cfg": train_cfg}, allow_val_change=True)
  try:
    wandb.config.update({"env_cfg": env_cfg.to_dict()}, allow_val_change=True)  # type: ignore[union-attr]
  except Exception:  # noqa: BLE001
    wandb.config.update({"env_cfg": asdict(env_cfg)}, allow_val_change=True)  # type: ignore[arg-type]
  _store_task_knobs()


# Task knobs live in module-level constants read from the environment, and none
# of them reached wandb. Searching 120 runs for which MOUNT_SPIN_DEG each used
# returned nothing, because the answer was only ever in the run NAME and in the
# launcher's --export string. Two things get recorded here:
#
#   dr_env       every LIFTPEG_/A3B_/PSM_ variable actually set, verbatim. This
#                is what the launcher asked for.
#   dr_resolved  the constants as the modules resolved them, defaults included.
#                This is what the run actually trained with, and it is the half
#                that was missing -- an unset knob left no trace at all, so a
#                run that took a default was indistinguishable from one that
#                never had the knob.
#
# Read from sys.modules rather than importing: if the task never loaded these
# modules there is nothing to record, and no unrelated task pays an import.
_KNOB_PREFIXES = ("LIFTPEG_", "A3B_", "PSM_", "SURGICAL_LAB_", "TISSUE_")
_KNOB_MODULES = (
  "surgical_lab.tasks.lift.config.psm.liftpeg_a3_dr",
  "surgical_lab.tasks.lift.config.psm.liftpeg_scene",
)


def _jsonable(v):
  """Keep scalars and flat containers of scalars; drop anything else."""
  if isinstance(v, (bool, int, float, str)) or v is None:
    return v
  if isinstance(v, (tuple, list)):
    out = [_jsonable(x) for x in v]
    return out if all(x is not None or x is None for x in out) else None
  if isinstance(v, dict):
    return {str(k): _jsonable(x) for k, x in v.items()}
  return None


def _store_task_knobs() -> None:
  import sys

  try:
    env = {k: v for k, v in os.environ.items() if k.startswith(_KNOB_PREFIXES)}
    if env:
      wandb.config.update({"dr_env": dict(sorted(env.items()))},
                          allow_val_change=True)

    resolved = {}
    for name in _KNOB_MODULES:
      mod = sys.modules.get(name)
      if mod is None:
        continue
      short = name.rsplit(".", 1)[-1]
      for attr in dir(mod):
        if not attr.isupper() or attr.startswith("_"):
          continue
        val = _jsonable(getattr(mod, attr, None))
        if val is not None:
          resolved[f"{short}.{attr}"] = val
    if resolved:
      wandb.config.update({"dr_resolved": dict(sorted(resolved.items()))},
                          allow_val_change=True)
  except Exception as exc:  # noqa: BLE001
    # Never let bookkeeping take down a training run.
    print(f"[runner] could not record task knobs: {exc}", flush=True)


WandbLogWriter.store_config = _store_config_allow_change  # type: ignore[method-assign]

# Checkpoint retention. The lab PC is a shared desktop that gets rebooted into
# Windows without warning, so runs are interrupted routinely and every
# checkpoint write is a potential power-cut window.
KEEP_LAST_N = int(os.environ.get("SURGICAL_LAB_KEEP_LAST_N", "1"))
# What best.pt is ranked by. mean_reward was the wrong yardstick: on liftpeg the
# shaped potential keeps paying for reaching and carrying, so a policy that
# grasps well and never completes can out-score one that transfers the ring.
# Concretely, the 2026-08-22 vision run peaked at success ~0.88 around
# iteration 7500 and decayed to 0.000 by 18000 as the action std ran away --
# and best.pt tracked mean_reward to iteration 7100, which was lucky rather
# than correct. Rank by the thing we actually want.
SUCCESS_METRIC_KEY = "Episode_Metrics/success_rate"
# Deliberately does NOT match mjlab's load_checkpoint regex ("model_.*.pt").
# mjlab picks the checkpoint to resume from by sorting names under
# f"{name:0>15}", and "model_best.pt" would sort after every "model_<iter>.pt",
# silently hijacking every resume. A name outside the regex cannot.
BEST_CHECKPOINT_NAME = "best.pt"
BEST_METRIC_NAME = "best_metric.json"
CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt\Z")

# --- Behaviour-cloning auxiliary -------------------------------------------
#
# On tissue retraction the reward IS -||a - a_scripted||^2, and PPO is a
# catastrophically bad way to optimise that: it samples a ~ N(mu, sigma),
# folds the result into a 24-step GAE return, fits a value function to a
# 2000-step episode, and nudges mu toward whichever noise happened to score
# better. That is a zeroth-order estimate of a gradient available in closed
# form. Measured on run z0qufrma: imitate went -0.17 (iter 0) -> -14.2 (iter
# 80) while the action std collapsed 0.49 -> 0.12, so the search direction was
# wrong AND the search radius vanished.
#
# The fix is to add the exact gradient alongside the PPO one -- the standard
# demo-augmented policy-gradient setup (Rajeswaran et al., DAPG), and the same
# BC-regularised PPO used throughout sim-to-real manipulation. This stays a
# PPO run: the surrogate, the advantages and the reward are all untouched. The
# auxiliary only supplies the term PPO was trying to discover by sampling.
#
# Off by default. Only tasks that publish a "teacher" observation group get it.
BC_COEF = float(os.environ.get("SURGICAL_LAB_BC_COEF", "1.0"))
BC_GROUP = "teacher"


class SurgicalLabOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
    super().__init__(env, train_cfg, log_dir, device)

    # rsl-rl logs scalars as wandb.log(..., step=it), so videos must be logged with the
    # same iteration or they land on a different x-axis. `it` is only in scope inside
    # the logger call, so wrap it rather than reimplement learn().
    # The logger CLEARS ep_extras at the end of log(), and save() runs after
    # log(), so the success rate has to be read on the way in or it is gone.
    self._last_success: float | None = None
    inner_log = self.logger.log

    def log_wrapper(it: int, *args, **kwargs):
      self._capture_success()
      inner_log(it, *args, **kwargs)
      if self._video_recorder is not None:
        self._log_videos(it)

    self._video_recorder = self._find_video_recorder()
    self.logger.log = log_wrapper

    self._bc_loss: float | None = None
    self._bc_sum, self._bc_n = 0.0, 0
    self._install_bc_auxiliary()

  def _install_bc_auxiliary(self) -> None:
    """Add the exact regression gradient to PPO's own gradient, one optimiser.

    The first version of this ran a SECOND Adam over the actor after
    `alg.update()` returned. That does not work, and the reason is worth
    keeping: two optimisers with independent momentum state pull the same
    weights in opposite directions, and each one's effective step is set by its
    own learning rate, so BC was permanently capped at 1e-3 while PPO's
    adaptive schedule ran free. Measured: bc loss sat at 0.18 for 120
    iterations against a floor of 0.045, and `imitate` fell -0.14 -> -20.8,
    i.e. exactly the pure-PPO trajectory. PPO was simply winning.

    Injecting the BC gradient just before `optimizer.step()` accumulates it
    into the SAME .grad buffers PPO has already populated, so one Adam sees one
    summed loss -- which is what "auxiliary loss" is supposed to mean.

    That 0.045 is the measured floor: bucketing the teacher action by the
    actor's own `home_delta` leaves only 0.045 of within-bucket variance out of
    0.3476 total, so the observation explains 87% of the target before the
    camera contributes anything. The cap was never perceptual.
    """
    obs = self.env.get_observations()
    if BC_GROUP not in obs.keys() or BC_COEF <= 0.0:
      return
    alg = self.alg
    actor = alg.actor
    cur: dict = {"batch": None}

    inner_gen = alg.storage.mini_batch_generator

    def gen_capture(*a, **k):
      for batch in inner_gen(*a, **k):
        cur["batch"] = batch
        yield batch

    inner_step = alg.optimizer.step

    def step_with_bc(*a, **k):
      batch = cur["batch"]
      if batch is not None:
        actor(batch.observations, masks=batch.masks,
              hidden_state=batch.hidden_states[0], stochastic_output=True)
        mu = actor.output_distribution_params[0]
        loss = BC_COEF * torch.nn.functional.mse_loss(
          mu, batch.observations[BC_GROUP]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), alg.max_grad_norm)
        self._bc_sum += loss.item() / max(BC_COEF, 1e-8)
        self._bc_n += 1
      return inner_step(*a, **k)

    inner_update = alg.update

    def update_with_bc() -> dict[str, float]:
      self._bc_sum, self._bc_n = 0.0, 0
      stats = inner_update()
      self._bc_loss = self._bc_sum / max(self._bc_n, 1)
      stats["bc"] = self._bc_loss
      return stats

    alg.storage.mini_batch_generator = gen_capture
    alg.optimizer.step = step_with_bc
    alg.update = update_with_bc
    print(f"[INFO] BC auxiliary active (coef {BC_COEF}, group '{BC_GROUP}', "
          f"summed into PPO's optimiser)")

  def _capture_success(self) -> None:
    """Mean of this iteration's per-episode success_rate, if the task has one.

    ep_extras is a list of per-step dicts; entries only carry the key on steps
    where an episode actually ended, so averaging over the whole list would
    divide by the wrong denominator.
    """
    extras = getattr(self.logger, "ep_extras", None)
    if not extras:
      return
    vals = []
    for info in extras:
      v = info.get(SUCCESS_METRIC_KEY)
      if v is None:
        continue
      vals.append(float(v.item() if hasattr(v, "item") else v))
    if vals:
      self._last_success = statistics.mean(vals)

  def _find_video_recorder(self) -> PolicyCamVideoRecorder | None:
    manager = getattr(self.env.unwrapped, "recorder_manager", None)
    if manager is None or VIDEO_RECORDER_TERM not in manager:
      return None
    term = manager.get_term(VIDEO_RECORDER_TERM)
    return term if isinstance(term, PolicyCamVideoRecorder) else None

  def _log_videos(self, it: int) -> None:
    recorder = self._video_recorder
    if recorder is None or not recorder.pending or wandb.run is None:
      return

    video_dir = os.path.join(self.logger.log_dir or ".", "videos")
    os.makedirs(video_dir, exist_ok=True)

    payload: dict = {}
    for tag, (episode_return, clip) in recorder.pending.items():
      path = os.path.join(video_dir, f"{tag}_{it:05d}.mp4")
      payload[f"eval/{tag}_trajectory"] = recorder.to_wandb_video(clip, path)
      payload[f"eval/{tag}_return"] = episode_return
      payload[f"eval/{tag}_length"] = len(clip)
    recorder.pending = {}

    wandb.log(payload, step=it)
    print(
      f"[INFO] Logged policy-view videos at iteration {it}: "
      + "  ".join(
        f"{tag}={payload[f'eval/{tag}_return']:.2f}"
        for tag in ("worst", "median", "best")
        if f"eval/{tag}_return" in payload
      )
    )

  @staticmethod
  def _fsync_dir(dirpath: str) -> None:
    """Durably record the rename itself, not just the file contents."""
    fd = os.open(dirpath or ".", os.O_DIRECTORY)
    try:
      os.fsync(fd)
    finally:
      os.close(fd)

  def _save_atomic(self, path: str, infos) -> None:
    """Write the checkpoint via a temp file + os.replace.

    mjlab's save() does a bare torch.save(path). If the machine dies mid-write
    -- and on this box it does -- that leaves a truncated checkpoint AND has
    already destroyed the previous good one, because the write happens in
    place. Writing to a sibling temp file and renaming means the old
    checkpoint stays intact until a complete new one exists; os.replace is
    atomic on POSIX.
    """
    tmp = f"{path}.tmp"
    upload = self.cfg.get("upload_model", False)
    # Suppress the W&B upload for the temp path; re-run it below on the real
    # path so the artifact name is right.
    self.cfg["upload_model"] = False
    try:
      super().save(tmp, infos)
      with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
      os.replace(tmp, path)
      self._fsync_dir(os.path.dirname(path))
    finally:
      self.cfg["upload_model"] = upload
      if os.path.exists(tmp):
        os.remove(tmp)
    if upload:
      self.logger.save_model(path, self.current_learning_iteration)

  def _mean_reward(self) -> float | None:
    buf = getattr(self.logger, "rewbuffer", None)
    if not buf:
      return None
    return statistics.mean(buf)

  def _update_best(self, path: str) -> None:
    """Keep a copy of the best-scoring checkpoint, ranked by SUCCESS RATE.

    Last-N alone is not enough -- with KEEP_LAST_N=1 it is the only thing
    standing between a late collapse and losing the good policy entirely. The
    2026-08-22 vision run is the worked example: peak success ~0.88 near
    iteration 7500, 0.000 by 18000, and every model_<iter>.pt from the peak
    long since pruned.

    Falls back to mean episode return for tasks that expose no success_rate
    metric, so this stays usable outside liftpeg.
    """
    metric = self._last_success
    kind = "success_rate"
    if metric is None:
      metric = self._mean_reward()
      kind = "mean_reward"
    if metric is None:
      # Happens on a save taken before any episode has terminated, so there is
      # nothing to rank the checkpoint by. Not fatal, but silence here would
      # look identical to "best tracking is broken".
      buf = getattr(self.logger, "rewbuffer", None)
      print(
        "[WARN] best-checkpoint tracking skipped: no success_rate this "
        "iteration and the reward buffer is "
        f"{'missing' if buf is None else 'empty'} at this save."
      )
      return
    run_dir = os.path.dirname(path)
    meta_path = os.path.join(run_dir, BEST_METRIC_NAME)
    previous = None
    if os.path.exists(meta_path):
      try:
        with open(meta_path) as f:
          meta = json.load(f)
        # Never compare across yardsticks: a resumed run whose metadata was
        # written under mean_reward would otherwise have a reward value
        # compared against a success rate, and 0.07 < 0.76 would read as an
        # improvement (or block one) for no meaningful reason.
        previous = meta["metric"] if meta.get("kind") == kind else None
      except (OSError, ValueError, KeyError):
        previous = None  # Corrupt metadata should not block a new best.
    if previous is not None and metric <= previous:
      return

    best = os.path.join(run_dir, BEST_CHECKPOINT_NAME)
    tmp = f"{best}.tmp"
    shutil.copyfile(path, tmp)
    os.replace(tmp, best)
    meta_tmp = f"{meta_path}.tmp"
    with open(meta_tmp, "w") as f:
      json.dump(
        {
          "kind": kind,
          "metric": metric,
          "iteration": self.current_learning_iteration,
          "source": os.path.basename(path),
        },
        f,
      )
    os.replace(meta_tmp, meta_path)
    self._fsync_dir(run_dir)
    print(f"[INFO] New best checkpoint: {kind}={metric:.4f} -> {best}")

  def _prune_checkpoints(self, path: str) -> None:
    """Keep the newest KEEP_LAST_N model_<iter>.pt. best.pt is never touched.

    KEEP_LAST_N defaults to 1: the newest is what a resume needs, and best.pt
    covers the "run degraded, go back" case that the other four were doing
    badly. Raise it with SURGICAL_LAB_KEEP_LAST_N when bisecting a run.
    """
    if KEEP_LAST_N <= 0:
      return
    run_dir = os.path.dirname(path)
    found = []
    for name in os.listdir(run_dir):
      m = CHECKPOINT_RE.match(name)
      if m:
        found.append((int(m.group(1)), name))
    found.sort()
    for _, name in found[:-KEEP_LAST_N]:
      try:
        os.remove(os.path.join(run_dir, name))
        print(f"[INFO] Pruned old checkpoint: {name}")
      except OSError as e:
        print(f"[WARN] Could not prune {name}: {e}")

  def save(self, path: str, infos=None):
    self._save_atomic(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    # The ONNX export is a convenience artifact; the .pt checkpoint above is
    # the ground truth. The Alpha3 actor's AdaptiveAvgPool2d(10->12) is not
    # expressible by the torchscript ONNX exporter (output size not a factor
    # of input size -- ABLATIONS.md issue #4), and a checkpoint save must not
    # kill a training run over a deployment artifact. Deployment export for
    # that model needs the exact matmul formulation instead (adaptive avg-pool
    # is a fixed linear map, so it CAN be exported exactly -- just not by this
    # code path).
    try:
      self.export_policy_to_onnx(policy_path, filename)
      # rsl-rl sets logger_type to "WandbLogWriter", never "wandb", so comparing
      # against "wandb" is always false and silently skips the upload. Ask wandb
      # directly.
      run_name: str = wandb.run.name if wandb.run else "local"  # type: ignore[assignment]
      onnx_path = os.path.join(policy_path, filename)
      metadata = self._get_metadata(run_name)
      attach_metadata_to_onnx(onnx_path, metadata)
      if wandb.run:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    except Exception as e:  # noqa: BLE001
      print(f"[WARN] ONNX export failed ({type(e).__name__}); checkpoint saved, "
            f"continuing training: {str(e).splitlines()[0][:160]}")
    # Retention runs last: only prune once a complete new checkpoint (and its
    # best-copy, if it won) is safely on disk.
    self._update_best(path)
    self._prune_checkpoints(path)

  def _get_metadata(self, run_name: str) -> dict:
    """Pick the metadata exporter matching the task's action space.

    mjlab's get_base_metadata only describes joint-position policies; a Cartesian
    task has no 'joint_pos' term and would raise KeyError on every checkpoint.
    """
    env = self.env.unwrapped
    active_actions = env.action_manager.active_terms
    if "joint_pos" in active_actions:
      return get_base_metadata(env, run_name)
    if "cartesian_delta" in active_actions:
      return get_cartesian_metadata(env, run_name)
    if "ee_delta" in active_actions:
      return get_diffik_metadata(env, run_name)
    raise ValueError(
      f"No metadata exporter for action terms {active_actions}. Deploying a policy "
      "without its action contract is unsafe; add an exporter for this action type."
    )
