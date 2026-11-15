import os

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def psm_push_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name="psm_push_block",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1_000,
  )


def psm_push_block_vision_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
    experiment_name="psm_push_block_vision",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=3_000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic", "camera"),
    },
  )


def psm_push_direct_s2r_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Camera-only actor for direct sim2real transfer.

  The deployed medrct_onnx node can feed images natively but cannot assemble a
  proprio vector (no rl_inputs support in the C++), so the actor consumes ONLY
  the camera group -- the exported ONNX then has a single image input the node
  already knows how to bind. The critic keeps privileged state (asymmetric AC).
  """
  cfg = psm_push_block_vision_ppo_runner_cfg()
  # The actor's 1D branch is a constant-zeros term (see _zeros_obs in env_cfg):
  # rsl-rl needs a non-empty 1D group, but the zeros carry no information, so
  # the actor is effectively camera-only and export bakes the zeros into the
  # graph, leaving a single camera input for the robot-side node.
  cfg.obs_groups = {
    "actor": ("actor", "camera"),
    "critic": ("critic", "camera"),
  }
  cfg.experiment_name = "psm_push_direct_s2r"
  return cfg


def psm_push_wide_a3b_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for the wide-goal push task, built on the known-good mjlab baseline.

  This deliberately does NOT replicate the Unity/SB3 a3b recipe. The only
  things carried over from Unity are the deployment I/O contract (84x84 RGB in,
  2-D continuous out) and general architectural conviction; the hyperparameters
  come from `psm_push_direct_s2r_ppo_runner_cfg`, which is the recipe that
  actually trained here -- SpatialSoftmax CNN, adaptive LR with a KL target,
  camera-only actor and privileged critic.

  One change from that baseline: a third conv layer. The 2-layer stack was
  sized for 32x32 input and leaves a 21x21 map on 84x84; three stride-2 layers
  bring it to 11x11 with a wider receptive field, which is the standard shape
  for 84x84 pixels.
  """
  cfg = psm_push_direct_s2r_ppo_runner_cfg()
  cfg.actor.cnn_cfg = {
    "output_channels": [16, 32, 64],
    "kernel_size": [5, 3, 3],
    "stride": [2, 2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
  }
  # Asymmetric actor-critic: the critic is a plain MLP over privileged state
  # (see _critic_state in a3b_env_cfg) with NO camera and NO CNN. The actor
  # keeps the camera-only contract the deployed ONNX node requires.
  #
  # This is what psm_cart_s2r_8192 -- the run this recipe is based on -- did.
  # Carrying the camera into the critic was a leftover from the abandoned
  # SB3-a3b replication, and it cost twice: a noisier value function, and a
  # second full 84x84 CNN forward+backward in every one of the 5 PPO epochs.
  cfg.critic.cnn_cfg = None
  cfg.critic.class_name = "MLPModel"
  cfg.obs_groups = {
    "actor": ("actor", "camera"),
    "critic": ("critic",),
  }
  # The baseline's num_mini_batches=4 was sized for 32x32 pixels; at 84x84 an
  # image is 6.9x larger and 4 minibatches OOMs. Measured on a 2g.20gb MIG
  # (4 iterations each, real train CLI):
  #     2048 envs / 24 mb  31.8 s/it  1546 steps/s
  #     2048 envs / 12 mb  26.5 s/it  1855 steps/s
  #     2048 envs /  8 mb  OOM
  #     4096 envs / 24 mb  OOM
  #     4096 envs / 48 mb  52.9 s/it  1858 steps/s
  #     8192 envs / 24 mb  OOM
  # The run is PPO-UPDATE bound, not rollout bound: raw env stepping alone is
  # ~63k steps/s at 84x84, so >95% of an iteration is the update. Throughput is
  # therefore flat in env count -- 4096 is chosen for per-update diversity under
  # DR, not for speed.
  #
  # 48 came from that table and is now stale twice over. It was measured on a
  # 20gb slice (job 52155919 runs on 3g.40gb and used only 13.8/40.4 GB), and it
  # was measured with TWO CNNs; the critic's is now gone. The rollout image
  # buffer (4096*24*3*84*84 fp32 = 8.3 GB) dominates and is independent of the
  # minibatch count, so the freed memory goes straight into larger minibatches.
  # 12 quadruples the per-step batch to 8192 samples -- still well under the
  # 49,152 the known-good psm_cart_s2r_8192 run used at 4 minibatches -- and
  # cuts optimizer/launch overhead, which is where the measured ~20% of FP32
  # peak was going (5.9 TFLOP/s achieved vs ~30 TFLOP/s for 60/132 SMs).
  cfg.algorithm.num_mini_batches = 12

  # --- exploration / LR: match Unity's a3b_resnet_e2e.yaml -------------------
  # That config is the ONLY one we have seen solve this task (28M steps). Job
  # 52194268 ran 253M steps under the settings below-as-they-were and never
  # left a ~20% success plateau, with action std collapsing 0.98 -> 0.05 inside
  # 150 iterations and staying there for 2400 more.
  #
  #   Unity            ours (before)      effect
  #   ent_coef 0.02    entropy_coef 0.005   4x weaker -> std collapse
  #   lr 3e-4 const    1e-3 adaptive(KL)    3.3x hotter, plus a schedule Unity
  #                                         did not have
  #
  # A high LR with weak entropy regularization is the textbook route to exactly
  # the collapse we measured, so both move together.
  cfg.algorithm.entropy_coef = 0.02
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.schedule = "fixed"

  # --- bound the action ------------------------------------------------------
  # This was None, and so was the env's max_target_lead, so NOTHING bounded |a|.
  # Job 52259180's policy exploited that: measured on real observations it emits
  # |a| ~ 30, i.e. ~15 mm per step at pos_scale=0.0005, lunging the tool across
  # the whole +-58 mm workspace. Success became bimodal -- within ~8 steps or
  # never -- and the deterministic policy scored 0.17 while the training log
  # read 0.99 (see docs/a3b_push_findings.md section 4).
  #
  # ML-Agents clips to [-1, 1], which is why the Unity policy this replicates
  # never showed the problem (scripts/add_action_clip.py documents the same
  # issue biting a previous export). 1.0 makes the per-step displacement
  # pos_scale * 1.0 = 0.5 mm, which is the motion the task was designed around
  # (Unity 0.01 u * 0.05 m/u).
  cfg.clip_actions = 1.0

  cfg.experiment_name = "psm_push_wide_a3b"
  cfg.max_iterations = 10_000
  return cfg


# Repo-relative; resolved against the repo root by MAEEncoder.load_pretrained.
# checkpoint.pt is the FINAL save from scripts/pretrain_mae.py -- deliberately,
# because the original ablation fleet consumed the last checkpoint, not a
# best-by-val selection. checkpoint_best.pt also exists on disk; picking it
# would be a (mild) improvement over Alpha3, which is exactly what we are not
# doing yet. Override via A3_ENCODER_CKPT (e.g. a stamped early-stop snapshot).
ALPHA3_ENCODER_CKPT = os.environ.get(
  "A3_ENCODER_CKPT", "encoders/alpha3_mjlab/checkpoint.pt"
)


def psm_push_wide_dextrah_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """DextrAH-style privileged TEACHER: state-in, action-out, no camera.

  Stage 1 of the DextrAH-RGB recipe (teacher-student distillation): a policy
  trained on privileged state, later distilled into an RGB student by online
  DAgger. The actor reads the same privileged vector the a3b critic already
  uses (_critic_state); both nets are plain MLPs. Everything else -- env, DR,
  rewards, PPO -- stays the a3b recipe, so teacher-vs-student comparisons stay
  single-variable.

  Note the deployed-contract caveat: this policy is NOT deployable on the
  robot (it reads sim-only state); its ONNX convenience export is meaningless.
  The deliverable is the student.
  """
  cfg = psm_push_wide_a3b_ppo_runner_cfg()
  cfg.actor.cnn_cfg = None
  # Bounds-regularised actor (rl_games bounds_loss_coef=0.005 equivalent),
  # keeping |mu| near [-1,1] so the distillation loss downstream regresses
  # control signal rather than clipped-away magnitude. The implementation
  # rides the entropy term and assumes algorithm.entropy_coef == 0.02 (set in
  # the a3b base cfg); see bounded_actor.py.
  cfg.actor.class_name = "surgical_lab.rl.bounded_actor:BoundedActorMLP"
  cfg.obs_groups = {
    "actor": ("critic",),   # privileged state for the acting policy
    "critic": ("critic",),
  }
  # dextrah's teacher trains at entropy_coef 0.002 (their yaml), 10x below the
  # a3b recipe. With mu bounded near +-1 this is load-bearing: at 0.02 the
  # entropy bonus holds sigma ~1.8 against |mu|~1.1 (rollout SNR 0.6) and the
  # teacher collapses to all-timeouts -- observed twice. The unbounded teacher
  # only tolerated 0.02 because |mu|~15 dwarfed sigma. bounded_actor.py's
  # ENTROPY_COEF must match this value.
  cfg.algorithm.entropy_coef = 0.002
  # No images anywhere in the PPO update -- the 84x84-driven minibatch split
  # is unnecessary. 4 matches the known-good state-based baseline.
  cfg.algorithm.num_mini_batches = 4
  cfg.experiment_name = "psm_push_wide_dextrah_teacher"
  return cfg


def psm_push_wide_alpha3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Alpha3: the a3b recipe with a frozen SimOrReal MAE encoder as the actor's eye.

  Alpha3 and a3b differ in exactly one place -- the actor's vision encoder --
  and that is preserved here. Everything else (task, DR, PPO, asymmetric
  critic, action clip, the 84x84 -> 2D deployment contract) is inherited from
  `psm_push_wide_a3b_ppo_runner_cfg` unchanged, so mjlab-Alpha3 vs mjlab-a3b is
  the same single-variable comparison Unity's Alpha3 vs a3b was.

  The encoder checkpoint comes from `scripts/pretrain_mae.py`. `freeze=True` is
  Alpha3; the unfrozen variant is muse's a4a arm.
  """
  cfg = psm_push_wide_a3b_ppo_runner_cfg()
  cfg.actor.class_name = "surgical_lab.rl.mae_pretrained:MAEPretrainedModel"
  cfg.actor.cnn_cfg = {
    # Geometry must match pretraining: 84/7 = 12x12 = 144 tokens, dim 256,
    # depth 4, heads 4, mlp 512 (the job-45685709 launch config).
    "features_dim": 256,
    "encoder_dim": 256,
    "patch_grid": 12,
    "depth": 4,
    "heads": 4,
    "mlp_dim": 512,
    "frame_stack": 1,
    "freeze": True,
    "checkpoint_path": ALPHA3_ENCODER_CKPT,
  }
  # net_arch [256, 256] -- Alpha3's head. Same as the a3b baseline's.
  cfg.experiment_name = "psm_push_wide_alpha3"
  return cfg
