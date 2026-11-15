"""Base configuration for the reach task: drive the PSM tool tip to a sampled target pose.

This is the surgical Reach benchmark from Orbit-Surgical (Isaac-Lab), rebuilt on mjlab. The
robot-agnostic pieces live here; the PSM specifics (robot asset, action, target ranges) are
set in config/psm/env_cfg.py, exactly as our push/lift tasks are structured.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from surgical_lab.tasks.reach import mdp as reach_mdp
from surgical_lab.tasks.reach.mdp import ReachCommandCfg


def make_reach_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base reach task configuration (robot set per-config)."""

  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # EE -> target vector in the base frame (what the policy steers to zero).
    "ee_to_target": ObservationTermCfg(
      func=manipulation_mdp.ee_to_object_distance,
      params={
        "object_name": "target",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # site set per-robot
      },
      noise=Unoise(n_min=-0.001, n_max=0.001),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {**actor_terms}

  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {}  # set per-robot

  commands: dict[str, CommandTermCfg] = {
    "ee_pose": ReachCommandCfg(
      entity_name="robot",
      target_name="target",
      resampling_time_range=(4.0, 4.0),
      debug_vis=True,
    )
  }

  events = {
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  rewards = {
    "position_tracking": RewardTermCfg(
      func=reach_mdp.position_tracking,
      weight=1.0,
      params={
        "command_name": "ee_pose",
        "std": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # site set per-robot
      },
    ),
    "position_tracking_fine": RewardTermCfg(
      func=reach_mdp.position_tracking,
      weight=0.5,
      params={
        "command_name": "ee_pose",
        "std": 0.01,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={
        "max_vel": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  curriculum = {
    "action_rate_weight": CurriculumTermCfg(
      func=manipulation_mdp.reward_curriculum,
      params={
        "reward_name": "action_rate_l2",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 150 * 24, "weight": -0.05},
        ],
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=1.5,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=32,
      njmax=512,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=8.0,
  )
