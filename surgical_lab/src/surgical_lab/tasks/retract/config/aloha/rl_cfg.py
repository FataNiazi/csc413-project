from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _algorithm() -> RslRlPpoAlgorithmCfg:
  return RslRlPpoAlgorithmCfg(
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
  )


def aloha_retract_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """State-based."""
  model = RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  return RslRlOnPolicyRunnerCfg(
    actor=model,
    critic=model,
    algorithm=_algorithm(),
    experiment_name="aloha_retract",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def aloha_retract_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Image-based: spatial-softmax CNN over the camera, same PPO."""
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
  model = RslRlModelCfg(
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
    actor=model,
    critic=model,
    algorithm=_algorithm(),
    experiment_name="aloha_retract_vision",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
    obs_groups={"actor": ("actor", "camera"), "critic": ("critic", "camera")},
  )
