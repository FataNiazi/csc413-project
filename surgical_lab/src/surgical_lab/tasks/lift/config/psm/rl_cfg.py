from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def psm_lift_peg_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="psm_lift_peg",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=2_000,
  )


def psm_lift_peg_vision_ppo_runner_cfg(
) -> RslRlOnPolicyRunnerCfg:
  cnn_cfg = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
  }
  class_name = "mjlab.rl.spatial_softmax:SpatialSoftmaxCNNModel"
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      cnn_cfg=cnn_cfg,
      class_name=class_name,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="psm_lift_peg_vision",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def psm_lift_peg_a3b_ppo_runner_cfg(arm: str = "unity", low_entropy: bool = False):
  """The a3b PPO recipe, retargeted to lift-peg.

  Inherited wholesale from pushblock so the two tasks stay comparable: camera
  only actor with the SpatialSoftmax CNN, privileged MLP critic, fixed 3e-4 LR,
  entropy 0.02, action clip 1.0. Only the experiment name changes -- the action
  dimension (4 instead of 2) is read from the env, not the agent cfg.

  Every reward arm gets its own `experiment_name`, so the four run directories
  never collide and `scripts/train_auto.sh`'s resume-by-default probe stays
  per-arm rather than picking up a sibling's checkpoints.
  """
  from surgical_lab.tasks.push.config.psm.rl_cfg import (
    psm_push_wide_a3b_ppo_runner_cfg,
  )

  cfg = psm_push_wide_a3b_ppo_runner_cfg()
  name = "psm_lift_peg_a3b" if arm == "unity" else f"psm_lift_peg_a3b_{arm}"
  if low_entropy:
    # v7's single most anomalous number was Policy/mean_std = 157.13 against an
    # action clip of 1.0 -- the policy was pure saturated noise. A Gaussian's
    # entropy is sum(log std), so with no advantage signal to oppose it the
    # entropy bonus grows log std without bound. Pushblock shows the same drift
    # (std 10.35) and still succeeds, because bang-bang pushing tolerates it;
    # grasping a 4.4 mm post at 0.5 mm/step does not.
    #
    # This arm holds the reward fixed (arm A's) and turns the entropy term
    # down instead, so the ablation covers "the reward was wrong" AND "the
    # exploration schedule was wrong" rather than assuming the first.
    cfg.algorithm.entropy_coef = 0.005
    cfg.actor.distribution_cfg["init_std"] = 0.5
    name += "_lowent"
  cfg.experiment_name = name
  cfg.max_iterations = 6000
  return cfg


def psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy: bool = False):
  """Privileged-state teacher: the a3b recipe with the camera removed.

  The actor reads `_critic_state` (tool, block, block-tool, goal-block) instead
  of 84x84 pixels, so both actor and critic are plain MLPs. Everything else --
  task, DR, action space, reward, PPO hyperparameters -- is arm A's, so a
  difference against arm A is attributable to the observation and nothing else.

  That is the point: liftpeg is now known to be FEASIBLE (the scripted teacher
  in liftpeg_teacher.py grasps at 100% of ring yaws and clears the post 56%),
  so the open question is why RL cannot learn it. This separates "perception is
  the bottleneck" from "exploration / the binary jaw is the bottleneck".
  """
  cfg = psm_lift_peg_a3b_ppo_runner_cfg(arm="a_staged", low_entropy=low_entropy)
  cfg.actor.cnn_cfg = None
  cfg.actor.class_name = "MLPModel"
  cfg.critic.cnn_cfg = None
  cfg.critic.class_name = "MLPModel"
  cfg.obs_groups = {"actor": ("critic",), "critic": ("critic",)}
  # Distinct experiment names are load-bearing, not cosmetic: train_auto.sh
  # resumes from the newest checkpoint under logs/rsl_rl/<experiment_name>, so
  # a shared name would make the low-entropy run silently CONTINUE the 0.02 run
  # instead of starting fresh -- and the comparison would be meaningless.
  cfg.experiment_name = (
    "psm_lift_peg_a3b_state_lowent" if low_entropy else "psm_lift_peg_a3b_state"
  )
  return cfg


# ---------------------------------------------------------------------------
# Alpha3: the a3b recipe with a frozen SimOrReal MAE as the actor's eye.
#
# Same single-variable change Alpha3 made to a3b on pushblock -- ONLY the
# actor's vision encoder differs, so liftpeg-Alpha3 vs liftpeg-a3b stays a
# clean comparison. The encoder comes from scripts/pretrain_mae.py.
# ---------------------------------------------------------------------------

import os as _os

LIFTPEG_ALPHA3_ENCODER_CKPT = _os.environ.get(
  "LIFTPEG_ALPHA3_ENCODER",
  "encoders/alpha3_liftpeg_single/checkpoint.pt",
)


def psm_lift_peg_alpha3_ppo_runner_cfg(arm: str = "e_shaped",
                                       low_entropy: bool = True):
  cfg = psm_lift_peg_a3b_ppo_runner_cfg(arm=arm, low_entropy=low_entropy)
  # BoundedMAEPretrainedStd, NOT the bare MAEPretrainedModel. The bare class
  # drops the mu/sigma soft bounds that BoundedActorCNNStd applies on the a3b
  # vision arm, and without them the action std runs away past the action clip
  # (measured 2.44 -> 5.08 over 1500 iterations, success pinned at 0.000).
  # The mixin's penalty is scaled by BOUNDS_COEF / _ENTROPY_COEF, so assert the
  # class's declared coef matches the schedule this cfg actually sets.
  cfg.actor.class_name = (
    "surgical_lab.rl.bounded_actor:BoundedMAEPretrainedStd"
  )
  from surgical_lab.rl.bounded_actor import BoundedMAEPretrainedStd as _B
  assert cfg.algorithm.entropy_coef == _B._ENTROPY_COEF, (
    f"entropy_coef {cfg.algorithm.entropy_coef} does not match the bounded "
    f"actor's declared {_B._ENTROPY_COEF}; the bounds penalty would be "
    f"misweighted against the entropy bonus")
  # Geometry must match pretraining exactly, and the side-by-side arm is a
  # different geometry: 84x168 gives 12x24 = 288 tokens where the single-camera
  # arms give 12x12 = 144. Derive it from the SAME env var the env cfg reads, so
  # the actor cannot silently disagree with the observation it is handed.
  # (A mismatch would still be caught -- MAEEncoder.load_pretrained refuses a
  # pos_embedding whose length differs -- but failing here names the cause.)
  _view = _os.environ.get("LIFTPEG_CAM_VIEW", "side")
  _patch_grid = (12, 24) if _view == "sbs" else 12
  cfg.actor.cnn_cfg = {
    # dim 256, depth 4, heads 4, mlp 512.
    "features_dim": 256,
    "encoder_dim": 256,
    "patch_grid": _patch_grid,
    "depth": 4,
    "heads": 4,
    "mlp_dim": 512,
    "frame_stack": 1,
    "freeze": True,
    "checkpoint_path": LIFTPEG_ALPHA3_ENCODER_CKPT,
  }
  cfg.experiment_name = "psm_liftpeg_alpha3"
  return cfg
