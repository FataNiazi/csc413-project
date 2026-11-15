"""RL configs specific to tissue retraction.

The CNN arms deliberately have no config here -- they reuse liftpeg's verbatim
(see this package's `__init__`, which explains why). This module exists for the
one arm that cannot: Alpha3, whose actor needs an encoder checkpoint and a
token geometry, and both of those are tissue's own.
"""

from __future__ import annotations

import os

from surgical_lab.tasks.lift.config.psm.rl_cfg import (
  psm_lift_peg_a3b_ppo_runner_cfg,
)

# Where the frozen MAE comes from. No default that points at a liftpeg encoder:
# an encoder pretrained on the peg scene would load without complaint (same
# geometry, same channels) and quietly give the tissue policy an eye trained on
# a different world. Missing is better than wrong, so the default is a tissue
# path that will simply not exist until one is pretrained.
TISSUE_ALPHA3_ENCODER_CKPT = os.environ.get(
  "TISSUE_ALPHA3_ENCODER",
  "encoders/alpha3_tissue_single/checkpoint.pt",
)


def psm_tissue_alpha3_ppo_runner_cfg(low_entropy: bool = True):
  """The tissue staged-reward vision recipe with a frozen MAE as the eye.

  Single-variable against `Surgicallab-Tissue-Staged-Vision`: identical env,
  identical reward, identical PPO schedule -- only the actor's vision encoder
  differs (frozen MAE instead of the SpatialSoftmax CNN). Modelled on
  `psm_lift_peg_alpha3_ppo_runner_cfg`, and the two should stay in step.
  """
  cfg = psm_lift_peg_a3b_ppo_runner_cfg(arm="a_staged", low_entropy=low_entropy)
  # BoundedMAEPretrainedStd, NOT the bare MAEPretrainedModel. The bare class
  # drops the mu/sigma soft bounds, and without them the action std runs away
  # past the action clip -- measured on liftpeg at 2.44 -> 5.08 over 1500
  # iterations with success pinned at 0.000, and the tissue STATE arm did the
  # same thing (1.07 -> 2.60, success 1.00 -> 0.00) for exactly this reason:
  # it uses a plain MLP with no sigma bound.
  cfg.actor.class_name = "surgical_lab.rl.bounded_actor:BoundedMAEPretrainedStd"
  from surgical_lab.rl.bounded_actor import BoundedMAEPretrainedStd as _B
  assert cfg.algorithm.entropy_coef == _B._ENTROPY_COEF, (
    f"entropy_coef {cfg.algorithm.entropy_coef} does not match the bounded "
    f"actor's declared {_B._ENTROPY_COEF}; the bounds penalty would be "
    f"misweighted against the entropy bonus")

  # Geometry must match pretraining EXACTLY, and the side-by-side view is a
  # different geometry: 84x168 gives 12x24 = 288 tokens where the square views
  # give 12x12 = 144. Read from the SAME env var the env cfg reads, so the
  # actor cannot silently disagree with the observation it is handed. (A
  # mismatch would still be caught -- MAEEncoder.load_pretrained refuses a
  # pos_embedding of the wrong length -- but failing here names the cause.)
  view = os.environ.get("TISSUE_CAM_VIEW", "side")
  cfg.actor.cnn_cfg = {
    # dim 256, depth 4, heads 4, mlp 512 -- the Alpha3 encoder shape.
    "features_dim": 256,
    "encoder_dim": 256,
    "patch_grid": (12, 24) if view == "sbs" else 12,
    "depth": 4,
    "heads": 4,
    "mlp_dim": 512,
    "frame_stack": 1,
    "freeze": True,
    "checkpoint_path": TISSUE_ALPHA3_ENCODER_CKPT,
  }
  # The VIEW is in the experiment name, unlike liftpeg's. Two Alpha3 arms are
  # launched here at once and their observations are different SHAPES, so a
  # `train_auto`-style resume that picked up the sibling's checkpoint would not
  # even fail cleanly -- it would load a 144-token encoder against a 288-token
  # observation. Separate directories make that impossible.
  cfg.experiment_name = f"psm_tissue_alpha3_{view}"
  return cfg
