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
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from surgical_lab.mdp import CartesianDeltaActionCfg
from surgical_lab.mdp.recorders import PolicyCamVideoRecorderCfg
from surgical_lab.robots.psm.psm_constants import get_psm_robot_cfg
from surgical_lab.robots.psm.psm_kinematics import compute_psm_fk, compute_psm_ik
from surgical_lab.tasks.push.mdp import PushCommandCfg
from surgical_lab.tasks.push.push_env_cfg import make_push_env_cfg

PSM_MAIN_JOINTS = (
  "outer_yaw",
  "outer_pitch",
  "main_insertion",
  "tool_roll",
  "tool_pitch",
  "tool_yaw",
)


def _psm_ik_fn(
  t_ee_0: torch.Tensor, q_last_joint: torch.Tensor, q_actual: torch.Tensor
) -> torch.Tensor:
  """Adapts the PSM's fully-determined analytical IK to the CartesianDeltaAction
  interface, which also passes the redundant joint and current joint state used
  by 7-DOF arms for branch selection (unneeded for the PSM's 6-DOF chain)."""
  del q_last_joint, q_actual
  return compute_psm_ik(t_ee_0)


def get_block_spec(block_size: float = 0.0075, mass: float = 0.5) -> mujoco.MjSpec:
  """Create a 1.5cm block (half-size = 0.0075m)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="block")
  body.add_freejoint(name="block_joint")
  body.add_geom(
    name="block_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(block_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec


def get_target_spec(target_size: float = 0.025) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="target", mocap=True)
  body.add_geom(
    name="target_geom",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(target_size,) * 3,
    rgba=(0.0, 0.8, 0.0, 1.0),
    group=0,
  )
  return spec


def psm_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_push_env_cfg()

  cfg.scene.entities = {
    "robot": get_psm_robot_cfg(tool_type="lnd"),
    "block": EntityCfg(spec_fn=get_block_spec),
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

  assert cfg.commands is not None
  push_command = cfg.commands["push_command"]
  assert isinstance(push_command, PushCommandCfg)
  push_command.object_pose_range = PushCommandCfg.ObjectPoseRangeCfg(
    x=(-0.035, 0.035),
    y=(0.04, 0.15),
    z=(0.0075, 0.0075),
    yaw=(-3.14, 3.14),
  )
  push_command.target_pose_range = PushCommandCfg.TargetPoseRangeCfg(
    x=(-0.035, 0.035),
    y=(0.06, 0.13),
    z=(0.001, 0.001),
    yaw=(-3.14, 3.14),
  )

  cfg.observations["actor"].terms["ee_to_block"].params["asset_cfg"].site_names = (
    "tool_midpoint",
  )
  cfg.rewards["push"].params["asset_cfg"].site_names = ("tool_midpoint",)

  cfg.rewards["joint_pos_limits"].params["asset_cfg"].joint_names = PSM_MAIN_JOINTS
  cfg.rewards["joint_vel_hinge"].params["asset_cfg"].joint_names = PSM_MAIN_JOINTS

  joint_pos_obs = ObservationTermCfg(
    func=manipulation_mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.001, n_max=0.001),
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=PSM_MAIN_JOINTS)},
  )
  cfg.observations["actor"].terms["joint_pos"] = joint_pos_obs
  cfg.observations["critic"].terms["joint_pos"] = joint_pos_obs

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "tool_pitch_link"

  cfg.viewer.body_name = "tool_pitch_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    assert cfg.commands is not None
    cfg.commands["push_command"].resampling_time_range = (4.0, 4.0)

  return cfg


def _use_cartesian_action(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace joint-position control with end-effector delta control."""
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


def psm_push_cartesian_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = psm_push_env_cfg(play=play)
  _use_cartesian_action(cfg)
  return cfg


def psm_push_cartesian_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb",
  sim2real: bool = False,
  play: bool = False,
  video_interval_s: float = 600.0,
) -> ManagerBasedRlEnvCfg:
  """Vision-based push with Cartesian control (and, with sim2real, visual DR)."""
  cfg = psm_push_block_vision_env_cfg(
    cam_type=cam_type, sim2real=sim2real, play=play
  )
  _use_cartesian_action(cfg)

  if not play and cam_type == "rgb":
    cfg.recorders["policy_cam_video"] = PolicyCamVideoRecorderCfg(
      sensor_name="front_cam",
      num_track=20,
      interval_s=video_interval_s,
      fps=int(round(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation))),
    )

  return cfg


def psm_push_block_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb",
  sim2real: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = psm_push_env_cfg(play=play)

  cam_cfg = CameraSensorCfg(
    name="front_cam",
    camera_name="robot/front_cam",
    height=32,
    width=32,
    data_types=(cam_type,),
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

  camera_obs = ObservationGroupCfg(
    terms={
      f"front_cam_{cam_type}": ObservationTermCfg(func=func, params=param_kwargs),
    },
    enable_corruption=False,
    concatenate_terms=True,
  )
  cfg.observations["camera"] = camera_obs

  # Pop privileged info from actor observations.
  cfg.observations["actor"].terms.pop("ee_to_block")
  cfg.observations["actor"].terms.pop("block_to_goal")

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
    cfg.events["floor_light"] = EventTermCfg(
      mode="reset",
      func=dr.light_dir,
      params={
        "asset_cfg": SceneEntityCfg("terrain"),
        "ranges": {0: (0.2, 0.2), 1: (0.2, 0.2), 2: (-0.8, -1.2)},
      },
    )
    cfg.events["block_color"] = EventTermCfg(
      mode="reset",
      func=dr.geom_rgba,
      params={
        "asset_cfg": SceneEntityCfg("block", geom_names=(".*")),
        "ranges": {0: (0.6, 1.0), 1: (0.0, 0.1), 2: (0.0, 0.1), 3: (1.0, 1.0)},
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
    cfg.events["camera_quat"] = EventTermCfg(
      mode="reset",
      func=dr.body_quat,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("front_cam_frame",)),
        "roll_range": (-0.01, 0.01),
        "pitch_range": (-0.01, 0.01),
        "yaw_range": (-0.01, 0.01),
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


def _zeros_obs(env, dim: int = 3):
  """Constant-zeros observation.

  The rsl-rl MLP head requires a non-empty 1D observation branch (a pure
  camera-only actor fails with `torch.cat(): expected a non-empty list of
  Tensors`), but the real-robot ONNX node cannot assemble a proprio vector. A
  constant term carries no information -- the actor is effectively camera-only
  -- and at export the zeros are baked into the graph so the deployed model has
  a single camera input.
  """
  import torch as _torch

  return _torch.zeros(env.num_envs, dim, device=env.device)


def psm_push_direct_s2r_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Vision push trained to transfer DIRECTLY onto the dVRK increment pipeline.

  The deployed control chain is: medrct_onnx runs the policy once per camera
  frame (~30 Hz RealSense) and publishes a Cartesian position increment in the
  robot BASE frame with no anti-windup; dvrk_planning's IncrementTeleopController
  integrates it into an absolute target and solves IK. This cfg makes training
  match that chain exactly, so no translation layer is needed at deployment:

  - action deltas integrate in the BASE frame (not the end-effector frame),
  - no max_target_lead clamp (the real pipeline has none),
  - control dt 0.035 s (~28.6 Hz) to match camera-frame-driven inference,
  - actor is CAMERA-ONLY (see rl_cfg obs_groups): the C++ node feeds images
    natively but has no way to assemble a proprio vector, so the deployed input
    is exactly the training input. The critic keeps privileged state.
  - full sim2real visual DR (floor/light/block color, camera pose/fov).

  Deployment yaml must set position_speed_multiplier == pos_scale (0.02) and
  px/py/pz continuous actions at indices 0/1/2.
  """
  cfg = psm_push_cartesian_vision_env_cfg(cam_type="rgb", sim2real=True, play=play)

  action = cfg.actions["cartesian_delta"]
  assert isinstance(action, CartesianDeltaActionCfg)
  action.frame = "base"
  action.max_target_lead = None

  # Camera-only actor in effect: replace all actor proprio with constant zeros
  # (see _zeros_obs). Critic keeps its own privileged group untouched.
  cfg.observations["actor"].terms = {
    "zeros": ObservationTermCfg(func=_zeros_obs, params={"dim": 3}),
  }

  # ~30 Hz control to match camera-frame-driven inference (RealSense).
  cfg.decimation = 7

  if not play:
    rec = cfg.recorders.get("policy_cam_video")
    if rec is not None:
      rec.fps = int(round(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation)))

  return cfg
