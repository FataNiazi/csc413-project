"""Task registration for tissue retraction on the PSM.

The RL config is liftpeg's, unchanged apart from the experiment name. That is
deliberate rather than lazy: liftpeg's `-Bounded-Std` arm is the one that took
the vision policy from 0.000 to 0.9551, and both of its fixes are about the
action distribution rather than anything task-specific --

  * BoundedActorCNNStd soft-bounds mu AND sigma. Without the sigma bound a run
    walked std from 1.6 to 13.3 and destroyed itself: the entropy term stops
    opposing growth once std exceeds the action clip, because past that point
    nearly every sample clips to +-1 anyway.
  * entropy_coef 0.005 rather than 0.02.

Both apply verbatim here -- same 4-D action, same clip, same actor -- so
starting from a config already known to train a vision policy on this rig
removes one large variable from the first tissue run.
"""

from mjlab.tasks.registry import register_mjlab_task

from surgical_lab.rl.runner import SurgicalLabOnPolicyRunner
from surgical_lab.tasks.lift.config.psm.rl_cfg import (
  psm_lift_peg_a3b_ppo_runner_cfg,
  psm_lift_peg_a3b_state_ppo_runner_cfg,
)
from surgical_lab.tasks.retract.config.psm.tissue_a3b_env_cfg import (
  psm_lift_tissue_a3b_env_cfg,
)

_tissue_rl = psm_lift_peg_a3b_ppo_runner_cfg(arm="a_staged", low_entropy=True)
assert _tissue_rl.algorithm.entropy_coef == 0.005
_tissue_rl.actor.class_name = "surgical_lab.rl.bounded_actor:BoundedActorCNNStd"
_tissue_rl.experiment_name = "psm_tissue_vis84"

register_mjlab_task(
  task_id="Surgicallab-Tissue-A3B-Vision",
  env_cfg=psm_lift_tissue_a3b_env_cfg(),
  play_env_cfg=psm_lift_tissue_a3b_env_cfg(play=True),
  rl_cfg=_tissue_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)


# --- the TEACHER arm ------------------------------------------------------
# Run this BEFORE the vision arm. The lesson from liftpeg: a vision run that
# plateaus at 0.000 cannot tell you whether the reward is unlearnable or the
# perception is -- and it costs GPU-days to find out. The teacher reads the
# privileged state the reward is computed from, so if IT cannot learn the task,
# the reward is wrong and no amount of pixels will fix it.
_tissue_state_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_tissue_state_rl.experiment_name = "psm_tissue_state"

register_mjlab_task(
  task_id="Surgicallab-Tissue-State",
  env_cfg=psm_lift_tissue_a3b_env_cfg(state_only=True),
  play_env_cfg=psm_lift_tissue_a3b_env_cfg(state_only=True, play=True),
  rl_cfg=_tissue_state_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)


# --- the STAGED-REWARD arms ----------------------------------------------
# liftpeg's potential shaping instead of imitation, per
# `surgical_lab.tasks.retract.mdp.rewards_tissue`. Registered as separate task
# ids rather than a flag on the existing ones so every run already logged
# against psm_tissue_vis84 stays comparable.
#
# Run the STATE arm first, for the reason spelled out above: if the teacher
# cannot learn the task from the privileged quantities the reward is computed
# from, the reward is wrong, and no amount of pixels will fix it. That is
# cheap to find out here and expensive to find out from a vision run.
_tissue_staged_state_rl = psm_lift_peg_a3b_state_ppo_runner_cfg(low_entropy=True)
_tissue_staged_state_rl.experiment_name = "psm_tissue_staged_state"

register_mjlab_task(
  task_id="Surgicallab-Tissue-Staged-State",
  env_cfg=psm_lift_tissue_a3b_env_cfg(state_only=True, reward="staged"),
  play_env_cfg=psm_lift_tissue_a3b_env_cfg(
    state_only=True, reward="staged", play=True),
  rl_cfg=_tissue_staged_state_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)

_tissue_staged_rl = psm_lift_peg_a3b_ppo_runner_cfg(arm="a_staged", low_entropy=True)
assert _tissue_staged_rl.algorithm.entropy_coef == 0.005
_tissue_staged_rl.actor.class_name = (
  "surgical_lab.rl.bounded_actor:BoundedActorCNNStd")
_tissue_staged_rl.experiment_name = "psm_tissue_staged_vis84"

register_mjlab_task(
  task_id="Surgicallab-Tissue-Staged-Vision",
  env_cfg=psm_lift_tissue_a3b_env_cfg(reward="staged"),
  play_env_cfg=psm_lift_tissue_a3b_env_cfg(reward="staged", play=True),
  rl_cfg=_tissue_staged_rl,
  runner_cls=SurgicalLabOnPolicyRunner,
)


# --- the ALPHA3 arm -------------------------------------------------------
# Identical env to the staged-reward vision arm above; the ONLY change is the
# actor's vision encoder (a frozen SimOrReal MAE instead of the SpatialSoftmax
# CNN), so Alpha3-vs-Staged-Vision stays a clean single-variable comparison.
# The encoder comes from scripts/pretrain_mae.py via TISSUE_ALPHA3_ENCODER.
#
# ONE task id for both camera views, as on liftpeg: the view is an env var
# (TISSUE_CAM_VIEW), read by the env cfg and by the runner cfg together, so a
# second registration could only disagree with the first.
from .rl_cfg import psm_tissue_alpha3_ppo_runner_cfg  # noqa: E402

register_mjlab_task(
  task_id="Surgicallab-Tissue-Staged-Alpha3",
  env_cfg=psm_lift_tissue_a3b_env_cfg(reward="staged"),
  play_env_cfg=psm_lift_tissue_a3b_env_cfg(reward="staged", play=True),
  rl_cfg=psm_tissue_alpha3_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)
