from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def psm_reach_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  def _model() -> RslRlModelCfg:
    return RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    )

  return RslRlOnPolicyRunnerCfg(
    actor=_model(),
    critic=_model(),
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
    experiment_name="psm_reach",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=500,
  )


def psm_reach_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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

  def _model() -> RslRlModelCfg:
    return RslRlModelCfg(
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
    )

  return RslRlOnPolicyRunnerCfg(
    actor=_model(),
    critic=_model(),
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
    experiment_name="psm_reach_vision",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=2_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )
