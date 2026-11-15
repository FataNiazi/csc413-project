"""Tissue retraction with a single ALOHA arm, under Cartesian (differential IK) control.

Grasp the tissue near a marked corner and drag that corner to a marked endpoint.
"""

from typing import Any, Literal

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from surgical_lab.mdp.actions import BinaryJointPositionActionCfg
from surgical_lab.robots.aloha.aloha_constants import (
  ARM_ACTUATORS,
  EE_SITE,
  FINGER_JOINTS,
  GRIPPER_CLOSED,
  GRIPPER_OPEN,
  get_aloha_entity_cfg,
)
from surgical_lab.tasks.retract import scene as tissue_scene
from surgical_lab.tasks.retract.mdp import cloth as cloth_mdp
from surgical_lab.tasks.retract.mdp import observations as retract_obs
from surgical_lab.tasks.retract.mdp import rewards as retract_rew

CORNER_ID = cloth_mdp.corner_index(
  tissue_scene.CLOTH_COUNT, tissue_scene.CORNER_IX, tissue_scene.CORNER_IY
)
GOAL = tissue_scene.goal_pos()

SUCCESS_THRESHOLD = 0.03  # 3 cm of the endpoint counts as retracted


def psm_free_scene() -> SceneCfg:
  return SceneCfg(
    terrain=None,  # a flex body must never touch a plane; the table box is the ground
    num_envs=1,
    env_spacing=2.0,
    entities={
      "robot": get_aloha_entity_cfg(),
      "table": EntityCfg(spec_fn=tissue_scene.get_table_spec),
      "tissue": EntityCfg(spec_fn=tissue_scene.get_tissue_spec),
      "markers": EntityCfg(spec_fn=tissue_scene.get_marker_spec),
    },
  )


def _actions() -> dict[str, ActionTermCfg]:
  return {
    # Cartesian: 3-D end-effector position deltas, resolved by mjlab's damped
    # least-squares differential IK.
    "ee_delta": DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=ARM_ACTUATORS,
      frame_type="site",
      frame_name=EE_SITE,
      use_relative_mode=True,
      delta_pos_scale=0.02,
      position_weight=1.0,
      orientation_weight=0.0,  # position-only -> 3-D action
      joint_limit_weight=1.0,
      damping=0.05,
      max_dq=0.2,
    ),
    "gripper": BinaryJointPositionActionCfg(
      entity_name="robot",
      joint_names=FINGER_JOINTS,
      open_command_expr={".*_finger": GRIPPER_OPEN},
      close_command_expr={".*_finger": GRIPPER_CLOSED},
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  robot_ee = SceneEntityCfg("robot", site_names=(EE_SITE,), joint_names=FINGER_JOINTS)
  return {
    "retract": RewardTermCfg(
      func=retract_rew.staged_retract_reward,
      weight=1.0,
      params={
        "vertex_id": CORNER_ID,
        "goal_pos": GOAL,
        "reaching_std": 0.06,
        "dragging_std": 0.08,
        "asset_cfg": robot_ee,
      },
    ),
    "grasp": RewardTermCfg(
      func=retract_rew.grasp_reward,
      weight=0.5,
      params={
        "vertex_id": CORNER_ID,
        "distance_threshold": 0.03,
        "closed_threshold": 0.5 * (GRIPPER_OPEN + GRIPPER_CLOSED),
        "asset_cfg": robot_ee,
      },
    ),
    "lift": RewardTermCfg(
      func=retract_rew.tissue_lifted_off_table,
      weight=0.3,
      params={"vertex_id": CORNER_ID, "height": GOAL[2]},
    ),
    "goal": RewardTermCfg(
      func=retract_rew.goal_reward,
      weight=8.0,
      params={
        "vertex_id": CORNER_ID,
        "goal_pos": GOAL,
        "threshold": SUCCESS_THRESHOLD,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_vel_hinge": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_hinge_penalty,
      weight=-0.01,
      params={
        "max_vel": 1.0,
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_ACTUATORS),
      },
    ),
  }


def _observations(vision: bool) -> dict[str, ObservationGroupCfg]:
  robot_arm = SceneEntityCfg("robot", joint_names=ARM_ACTUATORS)
  robot_ee = SceneEntityCfg("robot", site_names=(EE_SITE,))
  robot_fingers = SceneEntityCfg("robot", joint_names=FINGER_JOINTS)

  proprio: dict[str, ObservationTermCfg] = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"asset_cfg": robot_arm},
    ),
    "gripper": ObservationTermCfg(
      func=retract_obs.gripper_state, params={"asset_cfg": robot_fingers}
    ),
    "ee_pos": ObservationTermCfg(
      func=retract_obs.ee_pos,
      noise=Unoise(n_min=-0.002, n_max=0.002),
      params={"asset_cfg": robot_ee},
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  privileged: dict[str, ObservationTermCfg] = {
    "ee_to_corner": ObservationTermCfg(
      func=retract_obs.ee_to_corner,
      noise=Unoise(n_min=-0.002, n_max=0.002),
      params={"vertex_id": CORNER_ID, "asset_cfg": robot_ee},
    ),
    "corner_to_goal": ObservationTermCfg(
      func=retract_obs.corner_to_goal,
      noise=Unoise(n_min=-0.002, n_max=0.002),
      params={"vertex_id": CORNER_ID, "goal_pos": GOAL},
    ),
    "corner_pos": ObservationTermCfg(
      func=retract_obs.corner_pos, params={"vertex_id": CORNER_ID}
    ),
  }

  # The critic always sees the cloth state; the actor only does in the state-based task.
  actor_terms = dict(proprio) if vision else {**proprio, **privileged}
  critic_terms = {**proprio, **privileged}

  groups = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }
  return groups


def aloha_retract_env_cfg(
  vision: bool = False,
  play: bool = False,
  cam_type: Literal["rgb"] = "rgb",
) -> ManagerBasedRlEnvCfg:
  scene = psm_free_scene()

  sensors: tuple = ()
  observations = _observations(vision)
  if vision:
    cam = CameraSensorCfg(
      name="front_cam",
      camera_name="table/overhead_cam",
      height=32,
      width=32,
      data_types=(cam_type,),
      enabled_geom_groups=(0, 3),
      use_shadows=False,
      use_textures=True,
    )
    sensors = (cam,)
    params: dict[str, Any] = {"sensor_name": cam.name}
    observations["camera"] = ObservationGroupCfg(
      terms={
        f"front_cam_{cam_type}": ObservationTermCfg(
          func=manipulation_mdp.camera_rgb, params=params
        )
      },
      enable_corruption=False,
      concatenate_terms=True,
    )
  scene.sensors = sensors

  events = {
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.02, 0.02),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_ACTUATORS),
      },
    ),
  }

  terminations = {"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)}

  cfg = ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=_actions(),
    events=events,
    rewards=_rewards(),
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=1.2,
      elevation=-20.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      # Cloth needs three orders of magnitude more contact budget than the rigid tasks
      # (the push task uses 32 / 512). Under-budgeting silently drops contacts and the
      # grasp then fails with no error.
      # Per-env budgets. Cloth needs orders of magnitude more than the rigid tasks (push
      # uses 32/512), but these are multiplied by num_envs, so oversizing them will OOM
      # a 20GB MIG slice instantly. Sized for a 400-vertex sheet plus the arm.
      nconmax=2_000,
      njmax=4_000,
      mujoco=MujocoCfg(
        # Exactly the aloha cloth benchmark's solver settings. mjlab's defaults
        # (newton / implicitfast) send this cloth to NaN.
        timestep=0.002,
        integrator="euler",
        solver="cg",
        jacobian="sparse",
        cone="pyramidal",
        iterations=50,
        ls_iterations=20,
      ),
    ),
    decimation=5,  # 100 Hz control
    episode_length_s=10.0,
  )

  if play:
    cfg.episode_length_s = 1e9
    cfg.observations["actor"].enable_corruption = False

  return cfg
