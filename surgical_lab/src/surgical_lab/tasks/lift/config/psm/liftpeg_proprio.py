"""Proprioception for liftpeg, as DISPLACEMENT FROM THE EPISODE'S HOME POSE.

Why this exists
---------------
On hardware the wrist-camera Alpha3 policy nearly does the task and then
collides with the peg SHAFT: the picture tells it roughly where the ring is,
but not precisely enough, and it has no idea where its own tool is. Four cheap
numbers the robot already publishes -- tool-tip x/y/z and the jaw angle -- close
that gap.

Why DISPLACEMENT and not absolute position
------------------------------------------
The sim-to-real frame offset on this rig is large and documented, and it is very
nearly CONSTANT:

  * `docs/ik_calibration_report.md`: analytical FK disagrees with the MuJoCo
    model by 174.99 mm (median over 512 settled configurations, spread only
    1.3 mm). Fitting ONE rigid transform takes the residual to 0.82 mm, i.e.
    99.5% of the error is a single translation.
  * `docs/tissue_s2r_alignment.md` §7: liftpeg's home height is 60.4 mm in sim
    against 32.8 mm measured on the real rig (12 episodes, 30.4-34.8 mm) -- a
    1.8x absolute disagreement on the one axis the task cares about most.

An ABSOLUTE tool position therefore hands the policy a number that is wrong by
more than the whole workspace on hardware. A displacement from the pose the arm
started the episode at cancels any constant frame offset exactly, and leaves
only the (much smaller) scale and drift terms -- which is what `PROPRIO_DR`
below randomises.

The home pose is LATCHED per env at reset rather than taken from a constant,
because `randomize_home_joints` draws a different home every episode. It is
captured LAZILY -- the reset event only marks the env pending, and the first
observation call of the new episode fills it in -- because `.data` is stale
inside a reset event (the sim has not been forwarded yet), so a value read
there describes the PREVIOUS episode. Capturing at the first observation also
makes the definition self-consistent by construction: the first observation of
every episode is exactly zero.

Everything here is OFF unless `LIFTPEG_PROPRIO=1`. With the flag off the actor
keeps its constant-zeros branch and every existing task is untouched.

DEPLOYMENT NOTE: turning this on REPLACES the `_zeros_obs` term, so the
exported ONNX gains a genuine second input and `scripts/bake_zeros_input.py`
no longer applies. The robot-side node has to feed it.
"""

from __future__ import annotations

import os

import torch

from surgical_lab.tasks.lift.config.psm.liftpeg_grasp import _jaw_cmd
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b

# --- flags ----------------------------------------------------------------

PROPRIO = os.environ.get("LIFTPEG_PROPRIO", "0") == "1"
"""Experiment 1: give the actor (dx, dy, dz, djaw) from the home pose."""

PROPRIO_DR = os.environ.get("LIFTPEG_PROPRIO_DR", "0") == "1"
"""Experiment 3: domain-randomise those four numbers. Implies PROPRIO."""


# --- randomisation magnitudes ---------------------------------------------
#
# The dominant real-world error on a DISPLACEMENT is not per-step sensor noise.
# The dVRK's joint encoders are on the motors, behind cable transmissions, so
# what actually corrupts an integrated tool-tip displacement is a PER-EPISODE
# SYSTEMATIC: a gain error (cable stretch, pulley radius, FK calibration) and a
# slow offset (backlash take-up, thermal drift). A policy can average zero-mean
# per-step noise out over a 2000-step episode; it cannot average out a constant.
# Training with per-step noise ALONE would therefore give false confidence.
#
# So both are modelled, and the per-episode terms are the larger of the two.
#
# The bias magnitude is sized off the measured kinematic residual: 0.82 mm
# median after the rigid-transform fit, 4.07 mm end-to-end IK->sim placement
# with the offset applied (docs/ik_calibration_report.md). +-1.5 mm sits
# between the two -- generous against the fitted model, conservative against
# the raw pipeline.
PROPRIO_SCALE_DR = float(os.environ.get("LIFTPEG_PROPRIO_SCALE_DR", "0.05"))
"""Per-episode multiplicative gain error on dx/dy/dz, uniform in 1 +- this."""

PROPRIO_BIAS_XYZ = float(os.environ.get("LIFTPEG_PROPRIO_BIAS_XYZ", "0.0015"))
"""Per-episode additive bias on dx/dy/dz, metres, uniform in +- this."""

PROPRIO_BIAS_JAW = float(os.environ.get("LIFTPEG_PROPRIO_BIAS_JAW", "0.02"))
"""Per-episode additive bias on djaw, radians, uniform in +- this."""

# The jaw angle is measured directly at the joint and is close to binary, so it
# gets a much smaller share of both the systematic and the per-step term, and
# NO gain error at all -- there is no integration to accumulate one.
PROPRIO_STEP_SIGMA_XYZ = float(os.environ.get("LIFTPEG_PROPRIO_STEP_XYZ", "0.0002"))
"""Per-step Gaussian sigma on dx/dy/dz, metres."""

PROPRIO_STEP_SIGMA_JAW = float(os.environ.get("LIFTPEG_PROPRIO_STEP_JAW", "0.01"))
"""Per-step Gaussian sigma on djaw, radians."""

# Observation latency, in CONTROL steps of 20 ms. The real loop reads an
# encoder, publishes over ROS and runs inference before the number reaches the
# policy; 0-2 steps is 0-40 ms, which brackets what that chain costs. Applied
# through mjlab's native DelayBuffer (`delay_min_lag` / `delay_max_lag`), so it
# costs one ring buffer and no code of ours.
PROPRIO_DELAY_MAX_LAG = int(os.environ.get("LIFTPEG_PROPRIO_DELAY", "2"))


# --- per-episode latch ----------------------------------------------------


class _Home:
  """Per-env home pose, plus the per-episode DR draws that ride on it.

  `pending` is the whole trick. A reset event cannot read the home pose
  (`.data` there still describes the previous episode), so the event only
  raises this flag and the next observation call -- which runs against a
  forwarded sim -- does the capture.
  """

  def __init__(self, env):
    n, dev = env.num_envs, env.device
    self.pos = torch.zeros(n, 3, device=dev)
    self.jaw = torch.zeros(n, device=dev)
    # Every env starts pending: the very first observation defines home.
    self.pending = torch.ones(n, dtype=torch.bool, device=dev)
    self.scale = torch.ones(n, 4, device=dev)
    self.bias = torch.zeros(n, 4, device=dev)

  def draw(self, env, ids) -> None:
    """Draw this episode's gain and offset errors. No `.data` read -- safe in
    a reset event."""
    if not PROPRIO_DR:
      return
    k = self.scale[ids].shape[0]
    dev = self.scale.device
    u = lambda a: (torch.rand(k, device=dev) * 2.0 - 1.0) * a  # noqa: E731
    s = torch.ones(k, 4, device=dev)
    s[:, 0] = 1.0 + u(PROPRIO_SCALE_DR)
    s[:, 1] = 1.0 + u(PROPRIO_SCALE_DR)
    s[:, 2] = 1.0 + u(PROPRIO_SCALE_DR)
    # index 3 (jaw) keeps gain 1.0: it is measured, not integrated.
    self.scale[ids] = s
    b = torch.zeros(k, 4, device=dev)
    b[:, 0] = u(PROPRIO_BIAS_XYZ)
    b[:, 1] = u(PROPRIO_BIAS_XYZ)
    b[:, 2] = u(PROPRIO_BIAS_XYZ)
    b[:, 3] = u(PROPRIO_BIAS_JAW)
    self.bias[ids] = b


def home(env) -> _Home:
  got = getattr(env, "_liftpeg_home", None)
  if got is None:
    got = _Home(env)
    env._liftpeg_home = got
  return got


def reset_home(env, env_ids=None) -> None:
  """Reset event: mark the home pose stale and redraw the episode's DR.

  Registered as an explicit `EventTermCfg(mode="reset", ...)` -- the same
  pattern `LR.reset_latches` uses -- so the clear happens at a known point
  rather than wherever a manager happens to run first.
  """
  h = home(env)
  ids = slice(None) if env_ids is None else env_ids
  h.pending[ids] = True
  h.draw(env, ids)


def _raw(env):
  """(tool position in env frame, jaw command). Both read from LIVE data."""
  return a3b._tool_pos(env), _jaw_cmd(env)


def proprio_obs(env) -> torch.Tensor:
  """(dx, dy, dz, djaw) from this episode's home pose. (num_envs, 4).

  Under `PROPRIO_DR` the per-episode gain and offset are applied HERE, before
  mjlab's own pipeline adds the per-step Gaussian and the latency -- which is
  the right order: the systematic corrupts the signal, the sensor noise is
  added on top of the corrupted signal.
  """
  h = home(env)
  tool, jaw = _raw(env)
  if bool(h.pending.any()):
    m = h.pending
    h.pos[m] = tool[m]
    h.jaw[m] = jaw[m]
    h.pending[m] = False
  d = torch.cat([tool - h.pos, (jaw - h.jaw).unsqueeze(-1)], dim=-1)
  if PROPRIO_DR:
    d = d * h.scale + h.bias
  return d


# --- metric ---------------------------------------------------------------


def m_proprio_dz_mm(env) -> torch.Tensor:
  """Vertical displacement from home, mm.

  Logged because the failure mode this whole experiment has to rule out is an
  observation that is silently constant -- a term wired to the wrong asset, or
  a home latch that is re-captured every step, reads exactly 0.000 forever and
  looks identical to a healthy run everywhere else. A run whose `prop_dz_mm`
  is flat at zero is not testing proprioception.
  """
  return proprio_obs(env)[:, 2].abs() * 1000.0
