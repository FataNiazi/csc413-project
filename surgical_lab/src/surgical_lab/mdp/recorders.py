"""Recorder terms for surgical_lab.

The video recorder logs exactly what the policy sees: the camera sensor feeding the
observation, not a third-person view of the scene. mjlab's built-in ``--video`` flag
renders the viewer for env 0 on a step trigger, which answers a different question.

The key design choice is that no extra rollouts are run. The training envs are already
stepping the policy and already rendering their cameras every step, and each reset
re-samples the domain randomization, so a handful of tracked envs *are* a set of
independent randomized rollouts of the current policy. We ride along on them and pay
only for the frame copies, instead of paying to simulate a separate eval env.

The trade-off: tracked episodes come from the training distribution, so actions carry
the policy's exploration noise and the episodes compared in one flush completed at
slightly different times (within the last interval). This shows what the policy is
actually doing during training, not a clean deterministic evaluation of a single
snapshot of the weights.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.sensor import CameraSensor


class PolicyCamVideoRecorder(RecorderTerm):
  """Buffer camera frames for a few training envs and rank their episodes by return."""

  def __init__(self, cfg: PolicyCamVideoRecorderCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.cfg: PolicyCamVideoRecorderCfg = cfg

    # Two frame sources. A raw SENSOR is right when the policy observes that
    # sensor unchanged. obs_func is for views the policy does NOT see raw: the
    # wrist is rendered wide and fisheye-warped down, and the side-by-side arm
    # is two sensors concatenated, so filming either sensor would show
    # something the actor never receives -- which is indistinguishable from a
    # miswired run when you watch the clip.
    self._obs_func = cfg.obs_func
    if self._obs_func is None:
      sensor = env.scene[cfg.sensor_name]
      assert isinstance(sensor, CameraSensor)
      self._sensor = sensor
      h, w = sensor.cfg.height, sensor.cfg.width
    else:
      self._sensor = None
      h = w = None  # allocated lazily; the obs shape is only known once it runs

    self._k = min(cfg.num_track, env.num_envs)

    # Frames stay on GPU; only completed episodes are copied to host.
    self._frames = (
      None if h is None else torch.zeros(
        (self._k, cfg.max_frames, h, w, 3), dtype=torch.uint8, device=env.device
      )
    )
    self._t = torch.zeros(self._k, dtype=torch.long, device=env.device)
    self._returns = torch.zeros(self._k, device=env.device)

    self._episodes: deque[tuple[float, np.ndarray]] = deque(maxlen=cfg.max_episodes)
    self._last_flush = time.time()
    self._started = time.time()
    self._did_first = False
    self.pending: dict[str, tuple[float, np.ndarray]] = {}

  def _rgb(self) -> torch.Tensor:
    if self._obs_func is not None:
      # Observation terms are (B, 3, H, W) float in [0, 1]; the buffer holds
      # (B, H, W, 3) uint8, the same layout CameraSensor.data.rgb gives.
      x = self._obs_func(self._env)[: self._k]
      return (x.permute(0, 2, 3, 1).clamp(0, 1) * 255).to(torch.uint8)
    rgb = self._sensor.data.rgb
    assert rgb is not None, f"Camera '{self.cfg.sensor_name}' has no RGB data"
    return rgb[: self._k]

  def _write_frames(self, env_ids: torch.Tensor) -> None:
    """Append the current frame for env_ids, dropping frames past the cap."""
    if self._frames is None:
      f = self._rgb()
      self._frames = torch.zeros(
        (self._k, self.cfg.max_frames, *f.shape[1:]),
        dtype=torch.uint8, device=f.device,
      )
    room = self._t[env_ids] < self.cfg.max_frames
    ids = env_ids[room]
    if ids.numel():
      self._frames[ids, self._t[ids]] = self._rgb()[ids]
      self._t[ids] += 1

  def record_post_step(self) -> None:
    tracked = torch.arange(self._k, device=self._env.device)
    self._write_frames(tracked)

    # Envs that reset this step had their terminal reward banked in record_pre_reset;
    # adding it again here would credit it to the new episode.
    live = tracked[~self._env.reset_buf[: self._k]]
    if live.numel():
      self._returns[live] += self._env.reward_buf[live]

    if (not self._did_first
        and time.time() - self._started >= self.cfg.first_flush_s):
      self._did_first = True
      self._flush_partial()
    elif time.time() - self._last_flush >= self.cfg.interval_s:
      self._flush()

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    ids = env_ids[env_ids < self._k]
    if not ids.numel():
      return

    # Still pre-reset, so this is the terminal frame and the terminal reward.
    self._returns[ids] += self._env.reward_buf[ids]
    self._write_frames(ids)

    for i in ids.tolist():
      n = int(self._t[i])
      if n >= self.cfg.min_frames:  # see min_frames: stubs win "best" otherwise
        clip = self._frames[i, :n].cpu().numpy()
        self._episodes.append((float(self._returns[i]), clip))

    self._t[ids] = 0
    self._returns[ids] = 0.0

  def _flush_partial(self) -> None:
    """First flush only: rank the episodes IN PROGRESS rather than waiting.

    Snapshots each tracked env's frames so far without disturbing its counters,
    so recording continues uninterrupted and the episode is still banked
    normally when it ends.
    """
    self._last_flush = time.time()
    live = [(float(self._returns[i]), i) for i in range(self._k)
            if int(self._t[i]) >= self.cfg.min_frames]
    if len(live) < 3:
      self._did_first = False   # not enough yet; try again next step
      return
    live.sort()
    pick = {"worst": live[0][1], "median": live[len(live) // 2][1],
            "best": live[-1][1]}
    self.pending = {
      tag: (float(self._returns[i]),
            self._frames[i, : int(self._t[i])].cpu().numpy())
      for tag, i in pick.items()
    }

  def _flush(self) -> None:
    """Rank buffered episodes and stage best/median/worst for the runner to log."""
    self._last_flush = time.time()
    if len(self._episodes) < 3:
      return  # Nothing meaningful to rank yet.

    ranked = sorted(self._episodes, key=lambda e: e[0])
    self.pending = {
      "worst": ranked[0],
      "median": ranked[len(ranked) // 2],
      "best": ranked[-1],
    }
    self._episodes.clear()

  def to_wandb_video(self, clip: np.ndarray, path: str):
    """Encode (T, H, W, C) uint8 to mp4 and wrap it for W&B.

    We encode to a file rather than handing wandb.Video the raw array, because that
    path requires moviepy, whose pillow/decorator pins conflict with this project's
    dependency tree.
    """
    import imageio_ffmpeg
    import wandb

    if self.cfg.upscale > 1:
      clip = clip.repeat(self.cfg.upscale, axis=1).repeat(self.cfg.upscale, axis=2)
    clip = np.ascontiguousarray(clip)

    _, h, w, _ = clip.shape
    writer = imageio_ffmpeg.write_frames(
      path,
      size=(w, h),
      fps=self.cfg.fps,
      macro_block_size=1,  # 32x32 upscaled is not a multiple of 16; do not pad.
      quality=8,
    )
    writer.send(None)
    for frame in clip:
      writer.send(frame.tobytes())
    writer.close()

    return wandb.Video(path, format="mp4")


@dataclass(kw_only=True)
class PolicyCamVideoRecorderCfg(RecorderTermCfg):
  """Log best/median/worst policy-view rollouts to W&B on a wall-clock interval."""

  func: type[RecorderTerm] = PolicyCamVideoRecorder

  sensor_name: str = "front_cam"
  """Camera sensor whose frames the policy observes. Ignored when obs_func is set."""
  obs_func: Callable[..., torch.Tensor] | None = None
  """Observation term to film instead of a sensor, for views the policy does not
  see raw (fisheye-warped wrist, side-by-side concatenation). Takes the env and
  returns (B, 3, H, W) in [0, 1]."""
  num_track: int = 20
  """Number of training envs to record. These are the rollouts we rank by return."""
  interval_s: float = 600.0
  """Wall-clock seconds between flushes. Wall clock, not steps, because the throughput
  of a vision run is not known ahead of time."""
  fps: int = 50
  """Playback rate. The control loop runs at 1/step_dt = 50 Hz, so this is real time."""
  max_frames: int = 2400
  """Frame cap per episode. Must be >= episode_length_s / step_dt or clips are
  silently truncated -- this was 1200 with a comment claiming an episode was
  1000 steps, which stopped being true when the physics timestep halved and
  episodes became 2000 steps."""
  min_frames: int = 50
  """Episodes shorter than this are NOT ranked. Ranking is by episode RETURN, and
  when returns hover near zero or negative a stub episode accumulates the least
  negative total and therefore wins "best" -- so the best-trajectory video ends
  up being the shortest clip rather than the most successful one. Observed
  exactly that: a 6-frame clip logged as eval/best_trajectory while median and
  worst were full episodes."""
  upscale: int = 4
  """Nearest-neighbour upscale for legibility. 32x32 is unreadable in the W&B UI; the
  pixels are unchanged, just bigger."""
  first_flush_s: float = 45.0
  """Wall-clock seconds before the FIRST flush, which is allowed to use partial
  episodes. Without this the first video cannot appear until three episodes have
  COMPLETED, and an episode is 2000 steps -- about 7 minutes of wall clock at
  2048 envs. Waiting 10 minutes to find out whether a run is doing anything
  sensible is the wrong default; a mid-episode clip answers that in one."""
  max_episodes: int = 32
  """Cap on buffered completed episodes, so a slow flush cannot grow without bound."""
