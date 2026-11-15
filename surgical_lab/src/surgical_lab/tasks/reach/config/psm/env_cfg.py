"""PSM reach task: drive the dVRK tool tip to a randomly sampled target position."""

from typing import Any, Literal

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp

from surgical_lab.mdp import CartesianDeltaActionCfg
from surgical_lab.mdp.recorders import PolicyCamVideoRecorderCfg
from surgical_lab.robots.psm.psm_constants import get_psm_robot_cfg
from surgical_lab.robots.psm.psm_kinematics import compute_psm_fk, compute_psm_ik
from surgical_lab.tasks.reach.mdp import ReachCommandCfg
from surgical_lab.tasks.reach.reach_env_cfg import make_reach_env_cfg

PSM_MAIN_JOINTS = (
  "outer_yaw",
  "outer_pitch",
  "main_insertion",
  "tool_roll",
  "tool_pitch",
  "tool_yaw",
)
EE_SITE = "tool_midpoint"  # proven to exist + be tracked in the push task


def _psm_ik_fn(
  t_ee_0: torch.Tensor, q_last_joint: torch.Tensor, q_actual: torch.Tensor
) -> torch.Tensor:
  del q_last_joint, q_actual
  return compute_psm_ik(t_ee_0)


def get_target_spec(radius: float = 0.004) -> mujoco.MjSpec:
  """A small green sphere mocap marker for the reach target."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="target", mocap=True)
  body.add_geom(
    name="target_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    rgba=(0.0, 0.8, 0.0, 0.6),
    group=0,
    contype=0,
    conaffinity=0,
  )
  return spec


def psm_reach_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_reach_env_cfg()

  cfg.scene.entities = {
    "robot": get_psm_robot_cfg(tool_type="lnd"),
    "target": EntityCfg(spec_fn=get_target_spec),
  }

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=PSM_MAIN_JOINTS,
      scale={"main_insertion": 0.05, "(outer_|tool_).*": 0.5},
      use_default_offset=True,
    ),
  }

  # Point the command, obs, and rewards at the PSM tool site.
  ee_pose = cfg.commands["ee_pose"]
  assert isinstance(ee_pose, ReachCommandCfg)
  ee_pose.ee_site_name = EE_SITE
  # Base-frame offsets covering the tool's reachable workspace. The PSM tool tip rests at
  # roughly base + [0, -0.10, 0.03] and reaches down/forward from there.
  ee_pose.target_pose_range = ReachCommandCfg.TargetPoseRangeCfg(
    x=(-0.05, 0.05),
    y=(-0.13, -0.03),
    z=(0.0, 0.08),
  )

  cfg.observations["actor"].terms["ee_to_target"].params["asset_cfg"].site_names = (
    EE_SITE,
  )
  for name in ("position_tracking", "position_tracking_fine"):
    cfg.rewards[name].params["asset_cfg"].site_names = (EE_SITE,)

  cfg.rewards["joint_vel_hinge"].params["asset_cfg"].joint_names = PSM_MAIN_JOINTS

  cfg.viewer.body_name = "tool_pitch_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.commands["ee_pose"].resampling_time_range = (4.0, 4.0)

  return cfg


def _use_cartesian_action(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.actions.pop("joint_pos")
  cfg.actions["cartesian_delta"] = CartesianDeltaActionCfg(
    entity_name="robot",
    actuator_names=PSM_MAIN_JOINTS,
    position_only=True,
    fk_fn=compute_psm_fk,
    ik_fn=_psm_ik_fn,
    pos_scale=0.02,
    rot_scale=0.5,
  )


def psm_reach_cartesian_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = psm_reach_env_cfg(play=play)
  _use_cartesian_action(cfg)
  return cfg


def psm_reach_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb",
  sim2real: bool = False,
  play: bool = False,
  video_interval_s: float = 600.0,
) -> ManagerBasedRlEnvCfg:
  """Vision-based reach: the policy sees the on-board camera instead of the privileged
  EE->target vector, so it must locate the green target in the image and servo to it."""
  cfg = psm_reach_env_cfg(play=play)

  cam_cfg = CameraSensorCfg(
    name="front_cam",
    camera_name="robot/front_cam",
    height=32,
    width=32,
    data_types=(cam_type,),
    # sim2real renders the "real" visual geoms (0,1,2); otherwise the fast (0,3) set.
    enabled_geom_groups=(0, 1, 2) if sim2real else (0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
  if cam_type == "depth":
    param_kwargs["cutoff_distance"] = 0.5
    func = manipulation_mdp.camera_depth
  else:
    func = manipulation_mdp.camera_rgb

  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      f"front_cam_{cam_type}": ObservationTermCfg(func=func, params=param_kwargs),
    },
    enable_corruption=False,
    concatenate_terms=True,
  )

  # Actor loses the privileged target vector (must read it from the image); the critic
  # keeps it (asymmetric actor-critic), same as our push vision task.
  cfg.observations["actor"].terms.pop("ee_to_target")

  # Log best/median/worst policy-view videos to W&B during training.
  if not play and cam_type == "rgb":
    cfg.recorders["policy_cam_video"] = PolicyCamVideoRecorderCfg(
      sensor_name="front_cam",
      num_track=20,
      interval_s=video_interval_s,
      fps=int(round(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation))),
    )

  if sim2real:
    cfg.scene.terrain.materials = ()
    cfg.scene.terrain.textures = ()
    cfg.events["floor_color"] = EventTermCfg(
      mode="reset",
      func=dr.geom_rgba,
      params={
        "asset_cfg": SceneEntityCfg("terrain"),
        "operation": "abs",
        "distribution": "uniform",
        "ranges": {0: (0.0, 0.4), 1: (0.6, 0.8), 2: (0.95, 1.0), 3: (1.0, 1.0)},
      },
    )
    cfg.events["camera_pos"] = EventTermCfg(
      mode="reset",
      func=dr.body_pos,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("front_cam_frame",)),
        "axes": [0, 1, 2],
        "ranges": (-0.005, 0.005),
      },
    )
    cfg.events["camera_fov"] = EventTermCfg(
      mode="reset",
      func=dr.cam_fovy,
      params={
        "asset_cfg": SceneEntityCfg("robot", camera_names=("front_cam",)),
        "ranges": (50, 65),
      },
    )

  return cfg
