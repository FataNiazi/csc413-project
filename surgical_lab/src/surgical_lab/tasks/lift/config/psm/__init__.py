from mjlab.tasks.registry import register_mjlab_task

from surgical_lab.rl.runner import SurgicalLabOnPolicyRunner

from .env_cfg import (
  psm_lift_needle_env_cfg,
  psm_lift_needle_vision_env_cfg,
  psm_lift_peg_env_cfg,
  psm_lift_peg_vision_env_cfg,
)
from .rl_cfg import psm_lift_peg_ppo_runner_cfg, psm_lift_peg_vision_ppo_runner_cfg


register_mjlab_task(
  task_id="Surgicallab-Lift-Peg-PSM",
  env_cfg=psm_lift_peg_env_cfg(),
  play_env_cfg=psm_lift_peg_env_cfg(play=True),
  rl_cfg=psm_lift_peg_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)


register_mjlab_task(
  task_id="Surgicallab-Lift-Peg-PSM-Rgb",
  env_cfg=psm_lift_peg_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=psm_lift_peg_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=psm_lift_peg_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)


register_mjlab_task(
  task_id="Surgicallab-Lift-Needle-PSM",
  env_cfg=psm_lift_needle_env_cfg(),
  play_env_cfg=psm_lift_needle_env_cfg(play=True),
  rl_cfg=psm_lift_peg_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)


register_mjlab_task(
  task_id="Surgicallab-Lift-Needle-PSM-Rgb",
  env_cfg=psm_lift_needle_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=psm_lift_needle_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=psm_lift_peg_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)


from .liftpeg_a3b_env_cfg import psm_lift_peg_a3b_env_cfg  # noqa: E402
from .rl_cfg import psm_lift_peg_a3b_ppo_runner_cfg  # noqa: E402

# Lift-peg on the a3b scene: the pushblock recipe (camera-only actor with a
# SpatialSoftmax CNN, privileged MLP critic, same PPO/DR) on a 3-D lift, with a
# 45-degree side camera because height is invisible from directly overhead.
register_mjlab_task(
  task_id="Surgicallab-Liftpeg-A3B",
  env_cfg=psm_lift_peg_a3b_env_cfg(),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(play=True),
  rl_cfg=psm_lift_peg_a3b_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Reward ablation. v7 (the arm above, Unity's reward ported faithfully) never
# succeeded in 6000 iterations: its potential saturates 2.5 mm into a 32 mm
# lift, and with no advantage signal the entropy bonus drove the action std to
# 157. A-C vary ONLY the shaping term -- same scene, home pose, DR, camera,
# action space and success test -- so that comparison is clean. D re-runs A's
# reward with the entropy schedule turned down, so the ablation covers the
# second hypothesis too rather than assuming the first. Rationale for each arm
# is in `liftpeg_rewards.py` and `rl_cfg.py`.
for _suffix, _arm, _lowent in (
  ("A", "a_staged", False),       # robosuite nested-ceiling potential
  ("B", "b_unity_fixed", False),  # Unity's terms, the two broken ones replaced
  ("C", "c_dense", False),        # absolute staged reward, no success cutoff
  ("D", "a_staged", True),        # A again, with the entropy runaway shut down
):
  register_mjlab_task(
    task_id=f"Surgicallab-Liftpeg-A3B-{_suffix}",
    env_cfg=psm_lift_peg_a3b_env_cfg(reward=_arm),
    play_env_cfg=psm_lift_peg_a3b_env_cfg(reward=_arm, play=True),
    rl_cfg=psm_lift_peg_a3b_ppo_runner_cfg(arm=_arm, low_entropy=_lowent),
    runner_cls=SurgicalLabOnPolicyRunner,
  )


# Privileged-state teacher. Same scene/reward/actions as arm A, camera removed.
from .rl_cfg import psm_lift_peg_a3b_state_ppo_runner_cfg  # noqa: E402

register_mjlab_task(
  task_id="Surgicallab-Liftpeg-State",
  env_cfg=psm_lift_peg_a3b_env_cfg(reward="a_staged", state_only=True),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(reward="a_staged", state_only=True,
                                        play=True),
  rl_cfg=psm_lift_peg_a3b_state_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Heavily shaped arm. Reward and observation both built around the jaw-aligned
# grip point that the scripted teacher needed to reach 97.7%, with a grasp gate
# that a rim-rest cannot satisfy and a lift stage large enough that lifting is
# worth attempting. See liftpeg_shaped.py for what each choice is a response to.
_shaped_state_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
# Distinct experiment name or train_auto.sh resumes the OLD lowent teacher --
# its resume probe keys on logs/rsl_rl/<experiment_name>.
_shaped_state_rl.experiment_name = "psm_lift_peg_shaped_state"
register_mjlab_task(
  task_id="Surgicallab-Liftpeg-State-Shaped",
  env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped", state_only=True),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped", state_only=True,
                                        play=True),
  rl_cfg=_shaped_state_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

# The same reward on the CAMERA env -- the vision arm.
_shaped_vis_rl = psm_lift_peg_a3b_ppo_runner_cfg(arm="a_staged", low_entropy=True)
_shaped_vis_rl.experiment_name = "psm_lift_peg_shaped_vision"
register_mjlab_task(
  task_id="Surgicallab-Liftpeg-A3B-Shaped",
  env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped"),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped", play=True),
  rl_cfg=_shaped_vis_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)


# Same teacher with the entropy schedule turned down. Measured at iteration
# ~1700 of the v9 ablation: the action std is still climbing on ALL FOUR arms,
# including D, which already runs entropy_coef 0.005 / init_std 0.5 -- so that
# setting only DELAYS the runaway rather than stopping it (D at 2.88 against an
# action clip of 1.0, and rising). Running the teacher at 0.02 alone therefore
# risks a failure that cannot be attributed: observation or exploration? This
# arm pairs with the one above so a plateau in both points at the observation,
# and a plateau in only the first points at exploration.
register_mjlab_task(
  task_id="Surgicallab-Liftpeg-State-Lowent",
  env_cfg=psm_lift_peg_a3b_env_cfg(reward="a_staged", state_only=True),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(reward="a_staged", state_only=True,
                                        play=True),
  rl_cfg=psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True),
  runner_cls=SurgicalLabOnPolicyRunner,
)


# Same shaped reward and same camera as -Shaped, with the actor's policy MEAN
# held inside the action clip. The only difference from the arm above is the
# actor class -- one variable, so a difference is attributable to the bounds
# penalty and nothing else.
#
# Why: -Shaped plateaued at success 0.000 for 4000 iterations while grasping
# perfectly well (held 34.6% against the state teacher's 58.5%). Rolling out
# its PRE-clip mean showed dz at -14.6 with 95.7% of samples outside [-1, 1]:
# the descend-to-reach phase drove the z head far past the clip, and once
# there every sample maps to -1 and the head stops receiving gradient. It
# holds the ring and keeps commanding down (-0.26 um/step). See
# `surgical_lab/rl/bounded_actor.py` for the measurement, and
# `scripts/probe_liftpeg_jawcmd.py` to reproduce it.
#
# Ruled out first, so this is not a guess: the lift IS visible at 84 px (a
# 32 mm lift moves the ring 15.3 px, 18% of the frame -- see
# scripts/probe_liftpeg_visibility.py), and the actor does NOT need to see its
# own jaw (it latches it as cleanly as the state policy, 0.10 switches per
# 100 held steps).
_shaped_vis_bounded_rl = psm_lift_peg_a3b_ppo_runner_cfg(
  arm="a_staged", low_entropy=True
)
# BoundedActorCNN._ENTROPY_COEF must equal algorithm.entropy_coef -- PPO
# applies the penalty as entropy_coef * (BOUNDS_COEF / _ENTROPY_COEF), so a
# mismatch silently rescales it. low_entropy=True sets 0.005; the class
# declares 0.005.
assert _shaped_vis_bounded_rl.algorithm.entropy_coef == 0.005
_shaped_vis_bounded_rl.actor.class_name = (
  "surgical_lab.rl.bounded_actor:BoundedActorCNN"
)
_shaped_vis_bounded_rl.experiment_name = "psm_liftpeg_shaped_vision_bounded"
register_mjlab_task(
  task_id="Surgicallab-Liftpeg-A3B-Shaped-Bounded",
  env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped"),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped", play=True),
  rl_cfg=_shaped_vis_bounded_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)


# The bounded arm again, with the action STD bounded as well as the mean.
#
# -Bounded fixed the mu blowup and reached success ~0.88 (logged) / 0.76
# (measured) around iteration 7500, then decayed monotonically to 0.000 by
# 18000 as the action std grew 1.6 -> 13.3. Same failure one parameter over:
# the entropy bonus pushes log std up with constant force, and the surrogate
# loss stops opposing it once std exceeds the action clip, because past that
# point nearly every sample clips to +-1. See BoundedActorCNNStd.
#
# Kept as a SEPARATE task rather than changing -Bounded in place, so the run
# that produced the 0.000 -> 0.76 result stays reproducible.
_shaped_vis_std_rl = psm_lift_peg_a3b_ppo_runner_cfg(arm="a_staged", low_entropy=True)
assert _shaped_vis_std_rl.algorithm.entropy_coef == 0.005
_shaped_vis_std_rl.actor.class_name = (
  "surgical_lab.rl.bounded_actor:BoundedActorCNNStd"
)
_shaped_vis_std_rl.experiment_name = "psm_liftpeg_shaped_vision_bounded_std"
register_mjlab_task(
  task_id="Surgicallab-Liftpeg-A3B-Shaped-Bounded-Std",
  env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped"),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped", play=True),
  rl_cfg=_shaped_vis_std_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)


# Alpha3 liftpeg: identical env to the A3B shaped-bounded-std arm; the ONLY
# change is the actor's vision encoder (frozen MAE instead of the CNN).
from .rl_cfg import psm_lift_peg_alpha3_ppo_runner_cfg  # noqa: E402

register_mjlab_task(
  task_id="Surgicallab-Liftpeg-Alpha3",
  env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped"),
  play_env_cfg=psm_lift_peg_a3b_env_cfg(reward="e_shaped", play=True),
  rl_cfg=psm_lift_peg_alpha3_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)


# Two-PSM proof of concept: the shaped state task with a second arm added.
# Reward and success still score arm 1 only -- see liftpeg_dual.py.
from .liftpeg_dual import psm_lift_peg_dual_env_cfg  # noqa: E402

_dual_state_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_dual_state_rl.experiment_name = "psm_lift_peg_dual_state_poc"
register_mjlab_task(
  task_id="Surgicallab-Liftpeg-Dual-State-PoC",
  env_cfg=psm_lift_peg_dual_env_cfg(),
  play_env_cfg=psm_lift_peg_dual_env_cfg(play=True),
  rl_cfg=_dual_state_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Dummy objective on the two-PSM scene: each arm reaches its own random target.
# A pipeline check (PPO learns, both action slices work, wandb + checkpoints),
# not a research task. See dual_reach_dummy.py.
from .dual_reach_dummy import dual_reach_dummy_env_cfg  # noqa: E402

_dual_reach_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_dual_reach_rl.experiment_name = "dual_reach_dummy"
_dual_reach_rl.max_iterations = 500
_dual_reach_rl.wandb_project = "csc413-dual-psm"
register_mjlab_task(
  task_id="Surgicallab-Dual-Reach-Dummy",
  env_cfg=dual_reach_dummy_env_cfg(),
  play_env_cfg=dual_reach_dummy_env_cfg(play=True),
  rl_cfg=_dual_reach_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Task 1 (relay: arm 1 peg -> goal, arm 2 goal -> peg) with the naive v0 reward.
# See dual_task1.py.
from .dual_task1 import dual_task1_env_cfg  # noqa: E402

_dual_task1_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_dual_task1_rl.experiment_name = "dual_task1_naive"
_dual_task1_rl.max_iterations = 6000
_dual_task1_rl.wandb_project = "csc413-dual-psm"
register_mjlab_task(
  task_id="Surgicallab-Dual-Task1-Naive",
  env_cfg=dual_task1_env_cfg(),
  play_env_cfg=dual_task1_env_cfg(play=True),
  rl_cfg=_dual_task1_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Task 1 with reward v1: the single-arm staged potential, per arm and phase.
# See dual_task1_staged.py.
from .dual_task1_staged import dual_task1_staged_env_cfg  # noqa: E402

_dual_task1s_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_dual_task1s_rl.experiment_name = "dual_task1_staged"
_dual_task1s_rl.max_iterations = 6000
_dual_task1s_rl.wandb_project = "csc413-dual-psm"
# std bounded at ~1.2 (v0's unbounded std drifted to 2.8); must match entropy 0.005.
assert _dual_task1s_rl.algorithm.entropy_coef == 0.005
_dual_task1s_rl.actor.class_name = "surgical_lab.rl.bounded_actor:BoundedActorMLPStdLowEnt"
register_mjlab_task(
  task_id="Surgicallab-Dual-Task1-Staged",
  env_cfg=dual_task1_staged_env_cfg(),
  play_env_cfg=dual_task1_staged_env_cfg(play=True),
  rl_cfg=_dual_task1s_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Task 2 (mid-air handover), reward v0: Task 1's staged potential re-staged for
# the exchange in the air. See dual_task2_handover.py. Task 1 is unchanged.
from .dual_task2_handover import dual_task2_handover_env_cfg  # noqa: E402

_dual_task2_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_dual_task2_rl.experiment_name = "dual_task2_handover"
_dual_task2_rl.max_iterations = 6000
_dual_task2_rl.wandb_project = "csc413-dual-psm"
assert _dual_task2_rl.algorithm.entropy_coef == 0.005
_dual_task2_rl.actor.class_name = "surgical_lab.rl.bounded_actor:BoundedActorMLPStdLowEnt"
register_mjlab_task(
  task_id="Surgicallab-Dual-Task2-Handover",
  env_cfg=dual_task2_handover_env_cfg(),
  play_env_cfg=dual_task2_handover_env_cfg(play=True),
  rl_cfg=_dual_task2_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Task 2, reward v1: the exchange at a fixed spot above the goal (v0 had no
# fixed place and arm 1 carried the ring away upwards). Same file, spot=True.
_dual_task2s_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_dual_task2s_rl.experiment_name = "dual_task2_spot"
_dual_task2s_rl.max_iterations = 6000
_dual_task2s_rl.wandb_project = "csc413-dual-psm"
assert _dual_task2s_rl.algorithm.entropy_coef == 0.005
_dual_task2s_rl.actor.class_name = "surgical_lab.rl.bounded_actor:BoundedActorMLPStdLowEnt"
register_mjlab_task(
  task_id="Surgicallab-Dual-Task2-Spot",
  env_cfg=dual_task2_handover_env_cfg(spot=True),
  play_env_cfg=dual_task2_handover_env_cfg(play=True, spot=True),
  rl_cfg=_dual_task2s_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Task 2, reinsert variant: hand over at the spot above the goal (enforced),
# then arm 2 puts the ring back on the peg. Same file, reinsert=True.
_dual_task2r_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_dual_task2r_rl.experiment_name = "dual_task2_reinsert"
_dual_task2r_rl.max_iterations = 6000
_dual_task2r_rl.wandb_project = "csc413-dual-psm"
assert _dual_task2r_rl.algorithm.entropy_coef == 0.005
_dual_task2r_rl.actor.class_name = "surgical_lab.rl.bounded_actor:BoundedActorMLPStdLowEnt"
register_mjlab_task(
  task_id="Surgicallab-Dual-Task2-Reinsert",
  env_cfg=dual_task2_handover_env_cfg(reinsert=True),
  play_env_cfg=dual_task2_handover_env_cfg(play=True, reinsert=True),
  rl_cfg=_dual_task2r_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)
