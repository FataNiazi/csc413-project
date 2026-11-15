from mjlab.tasks.registry import register_mjlab_task

from surgical_lab.rl.runner import SurgicalLabOnPolicyRunner

from .env_cfg import (
  psm_push_block_vision_env_cfg,
  psm_push_cartesian_env_cfg,
  psm_push_cartesian_vision_env_cfg,
  psm_push_direct_s2r_env_cfg,
  psm_push_env_cfg,
)
from .rl_cfg import (
  psm_push_block_vision_ppo_runner_cfg,
  psm_push_direct_s2r_ppo_runner_cfg,
  psm_push_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Surgicallab-Push-Block-PSM",
  env_cfg=psm_push_env_cfg(),
  play_env_cfg=psm_push_env_cfg(play=True),
  rl_cfg=psm_push_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Surgicallab-Push-Block-PSM-Cartesian",
  env_cfg=psm_push_cartesian_env_cfg(),
  play_env_cfg=psm_push_cartesian_env_cfg(play=True),
  rl_cfg=psm_push_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Surgicallab-Push-Block-PSM-Rgb",
  env_cfg=psm_push_block_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=psm_push_block_vision_env_cfg(cam_type="rgb", sim2real=False, play=True),
  rl_cfg=psm_push_block_vision_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Surgicallab-Push-Block-PSM-Sim2Real",
  env_cfg=psm_push_block_vision_env_cfg(cam_type="rgb", sim2real=True),
  play_env_cfg=psm_push_block_vision_env_cfg(cam_type="rgb", sim2real=True, play=True),
  rl_cfg=psm_push_block_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Surgicallab-Push-Block-PSM-Cartesian-Sim2Real",
  env_cfg=psm_push_cartesian_vision_env_cfg(cam_type="rgb", sim2real=True),
  play_env_cfg=psm_push_cartesian_vision_env_cfg(
    cam_type="rgb", sim2real=True, play=True
  ),
  rl_cfg=psm_push_block_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Surgicallab-Push-Block-PSM-Direct-S2R",
  env_cfg=psm_push_direct_s2r_env_cfg(),
  play_env_cfg=psm_push_direct_s2r_env_cfg(play=True),
  rl_cfg=psm_push_direct_s2r_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

from .a3b_env_cfg import psm_push_wide_a3b_env_cfg  # noqa: E402
from .rl_cfg import (  # noqa: E402
  psm_push_wide_a3b_ppo_runner_cfg,
  psm_push_wide_alpha3_ppo_runner_cfg,
  psm_push_wide_dextrah_teacher_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Surgicallab-Push-Wide-A3B",
  env_cfg=psm_push_wide_a3b_env_cfg(),
  play_env_cfg=psm_push_wide_a3b_env_cfg(play=True),
  rl_cfg=psm_push_wide_a3b_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

# Alpha3: identical env to a3b -- the ONLY change is the actor's vision
# encoder (frozen SimOrReal MAE instead of a from-scratch CNN), matching how
# Unity's Alpha3 differed from Unity's a3b.
register_mjlab_task(
  task_id="Surgicallab-Push-Wide-Alpha3",
  env_cfg=psm_push_wide_a3b_env_cfg(),
  play_env_cfg=psm_push_wide_a3b_env_cfg(play=True),
  rl_cfg=psm_push_wide_alpha3_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

# DextrAH stage 1: privileged-state teacher on the same env. Not deployable;
# exists to be distilled into an RGB student (stage 2).
register_mjlab_task(
  task_id="Surgicallab-Push-Wide-Dextrah-Teacher",
  env_cfg=psm_push_wide_a3b_env_cfg(),
  play_env_cfg=psm_push_wide_a3b_env_cfg(play=True),
  rl_cfg=psm_push_wide_dextrah_teacher_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)


# --- two-domain sim-to-sim study -------------------------------------------
# CANONICAL domain: same env, same camera, same physics, but the policy sees a
# flat unlit segmentation-derived image (one constant colour per scene element)
# instead of the shaded/DR'd RGB. Purely additive -- Surgicallab-Push-Wide-A3B
# and -Alpha3 above are untouched. See tasks/push/canonical_render.py.
from surgical_lab.tasks.push.canonical_render import (  # noqa: E402
  psm_push_wide_a3b_canonical_env_cfg,
)

register_mjlab_task(
  task_id="Surgicallab-Push-Wide-A3B-Canonical",
  env_cfg=psm_push_wide_a3b_canonical_env_cfg(),
  play_env_cfg=psm_push_wide_a3b_canonical_env_cfg(play=True),
  rl_cfg=psm_push_wide_a3b_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)
