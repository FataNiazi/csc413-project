from mjlab.tasks.registry import register_mjlab_task

from surgical_lab.rl.runner import SurgicalLabOnPolicyRunner

from .env_cfg import aloha_retract_env_cfg
from .rl_cfg import aloha_retract_ppo_runner_cfg, aloha_retract_vision_ppo_runner_cfg

register_mjlab_task(
  task_id="Surgicallab-Retract-Tissue-Aloha",
  env_cfg=aloha_retract_env_cfg(vision=False),
  play_env_cfg=aloha_retract_env_cfg(vision=False, play=True),
  rl_cfg=aloha_retract_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)

register_mjlab_task(
  task_id="Surgicallab-Retract-Tissue-Aloha-Rgb",
  env_cfg=aloha_retract_env_cfg(vision=True),
  play_env_cfg=aloha_retract_env_cfg(vision=True, play=True),
  rl_cfg=aloha_retract_vision_ppo_runner_cfg(),
  runner_cls=SurgicalLabOnPolicyRunner,
)
