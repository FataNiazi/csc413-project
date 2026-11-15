from mjlab.tasks.registry import register_mjlab_task

from surgical_lab.rl.runner import SurgicalLabOnPolicyRunner

from .env_cfg import (
  psm_reach_cartesian_env_cfg,
  psm_reach_env_cfg,
  psm_reach_vision_env_cfg,
)
from .rl_cfg import psm_reach_ppo_runner_cfg, psm_reach_vision_ppo_runner_cfg

register_mjlab_task(
  task_id="Surgicallab-Reach-PSM",
  env_cfg=psm_reach_env_cfg(),
  play_env_cfg=psm_reach_env_cfg(play=True),
  rl_cfg=psm_reach_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Surgicallab-Reach-PSM-Cartesian",
  env_cfg=psm_reach_cartesian_env_cfg(),
  play_env_cfg=psm_reach_cartesian_env_cfg(play=True),
  rl_cfg=psm_reach_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Surgicallab-Reach-PSM-Rgb",
  env_cfg=psm_reach_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=psm_reach_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=psm_reach_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Surgicallab-Reach-PSM-Sim2Real",
  env_cfg=psm_reach_vision_env_cfg(cam_type="rgb", sim2real=True),
  play_env_cfg=psm_reach_vision_env_cfg(cam_type="rgb", sim2real=True, play=True),
  rl_cfg=psm_reach_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)
