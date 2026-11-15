from mjlab.tasks.registry import register_mjlab_task

from surgical_lab.rl.runner import SurgicalLabOnPolicyRunner

from .env_cfg import psm_cut_env_cfg, psm_cut_vision_env_cfg
from .rl_cfg import psm_cut_ppo_runner_cfg, psm_cut_vision_ppo_runner_cfg

register_mjlab_task(
  task_id="Surgicallab-Cut-PSM",
  env_cfg=psm_cut_env_cfg(),
  play_env_cfg=psm_cut_env_cfg(play=True),
  rl_cfg=psm_cut_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Surgicallab-Cut-PSM-Rgb",
  env_cfg=psm_cut_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=psm_cut_vision_env_cfg(cam_type="rgb", sim2real=False, play=True),
  rl_cfg=psm_cut_vision_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Surgicallab-Cut-PSM-Sim2Real",
  env_cfg=psm_cut_vision_env_cfg(cam_type="rgb", sim2real=True),
  play_env_cfg=psm_cut_vision_env_cfg(cam_type="rgb", sim2real=True, play=True),
  rl_cfg=psm_cut_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)
