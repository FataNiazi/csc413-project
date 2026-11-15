from typing import Any, Literal

import mujoco
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

from surgical_lab import SURGICAL_LAB_SRC_PATH
from surgical_lab.robots.psm.psm_constants import get_psm_robot_cfg
from surgical_lab.tasks.cut.cut_env_cfg import make_cut_env_cfg

ROPE_MOUNT_POS = (0.0, 0.01, 0.006)
ROPE_MOUNT_SCALE = 0.001
ROPE_MOUNT_LEFT_PIN = (-0.03, 0.00275, 0.044)
ROPE_MOUNT_RIGHT_PIN = (0.03, 0.00275, 0.044)
ROPE_MOUNT_POLE_COLLISION_BOTTOM_Z = -0.004
ROPE_MOUNT_POLE_COLLISION_TOP_Z = 0.041
ROPE_MOUNT_POLE_COLLISION_RADIUS = 0.00125


def _world_pos(
  pos: tuple[float, float, float],
  offset: tuple[float, float, float],
) -> tuple[float, float, float]:
  return (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])


ROPE_MOUNT_LEFT_PIN_WORLD = _world_pos(ROPE_MOUNT_POS, ROPE_MOUNT_LEFT_PIN)
ROPE_MOUNT_RIGHT_PIN_WORLD = _world_pos(ROPE_MOUNT_POS, ROPE_MOUNT_RIGHT_PIN)


def get_rope_mount_spec(scale: float = ROPE_MOUNT_SCALE) -> mujoco.MjSpec:
  """Create the static rope mount mesh."""
  mesh_path = (
    SURGICAL_LAB_SRC_PATH / "robots" / "psm" / "xmls" / "assets" / "rope_mount_v10.obj"
  )

  spec = mujoco.MjSpec()
  spec.assets[mesh_path.name] = mesh_path.read_bytes()
  spec.add_mesh(
    name=mesh_path.name,
    file=mesh_path.name,
    scale=(scale, scale, scale),
  )

  body = spec.worldbody.add_body(name="rope_mount", pos=ROPE_MOUNT_POS)
  body.add_geom(
    name="rope_mount_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh_path.name,
    rgba=(0.92941, 0.83922, 0.16863, 1.0),
    group=0,
    contype=0,
    conaffinity=0,
    quat=(0.0, 0.0, 0.0, 1.0),
  )
  for side_name, pin in (
    ("left", ROPE_MOUNT_LEFT_PIN),
    ("right", ROPE_MOUNT_RIGHT_PIN),
  ):
    body.add_geom(
      name=f"rope_mount_{side_name}_pole_collision",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=(
        pin[0],
        pin[1],
        ROPE_MOUNT_POLE_COLLISION_BOTTOM_Z,
        pin[0],
        pin[1],
        ROPE_MOUNT_POLE_COLLISION_TOP_Z,
      ),
      size=(ROPE_MOUNT_POLE_COLLISION_RADIUS,),
      rgba=(0.95, 0.95, 0.1, 1.0),
      group=3,
      contype=1,
      conaffinity=1,
      condim=4,
      friction=(1.0, 0.005, 0.0001),
    )
  body.add_site(
    name="left_rope_pin",
    pos=ROPE_MOUNT_LEFT_PIN,
    size=(0.0015,),
    rgba=(0.1, 0.4, 1.0, 1.0),
    group=0,
  )
  body.add_site(
    name="right_rope_pin",
    pos=ROPE_MOUNT_RIGHT_PIN,
    size=(0.0015,),
    rgba=(0.1, 0.4, 1.0, 1.0),
    group=0,
  )
  return spec


def get_rope_spec() -> mujoco.MjSpec:
  """Create a rope pinned between the mount poles."""
  spec = mujoco.MjSpec.from_string(
    """
    <mujoco>
      <worldbody>
        <body name="ropes" pos="0 0 0">
          <flexcomp type="grid" count="9 1 1" spacing="0.0075 0.004 0.004"
                    pos="0 0.01275 0.05" radius="0.0015"
                    rgba="0.95 0.95 0.95 1" name="rope1" dim="1"
                    mass="0.0002">
            <edge equality="true"/>
            <contact condim="3" selfcollide="none" conaffinity="0" contype="0"/>
            <pin id="0"/>
            <pin id="8"/>
          </flexcomp>
        </body>
      </worldbody>
    </mujoco>
  """
  )
  if spec.flexes:
    flex = spec.flexes[0]
    flex.damping = 0.002
    flex.edgedamping = 0.005
    flex.edgestiffness = 0.05
  # MuJoCo flexcomp emits anonymous joints; mjlab resolves entity joints by name.
  for joint_index, joint in enumerate(spec.joints):
    if not joint.name:
      joint.name = f"rope1_flex_joint_{joint_index}"
    joint.damping[:] = 0.002
    joint.armature = 1e-7
  return spec


def psm_cut_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_cut_env_cfg()

  cfg.scene.entities = {
    "robot": get_psm_robot_cfg(tool_type="lnd"),
    "rope_mount": EntityCfg(spec_fn=get_rope_mount_spec),
    "rope": EntityCfg(spec_fn=get_rope_spec),
  }

  for obs_group in cfg.observations.values():
    obs_group.terms.pop("ee_to_object", None)
    obs_group.terms.pop("object_to_goal", None)

  cfg.commands = {}

  for event_name in (
    "object_friction_slide",
    "object_friction_spin",
    "object_friction_roll",
  ):
    cfg.events.pop(event_name, None)

  cfg.rewards.pop("cut", None)
  cfg.rewards.pop("goal", None)

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(
        "outer_yaw",
        "outer_pitch",
        "main_insertion",
        "tool_roll",
        "tool_pitch",
        "tool_yaw",
      ),
      scale={"main_insertion": 0.01, "(outer_|tool_).*": 0.25},
      use_default_offset=True,
    ),
  }

  main_joints = (
    "outer_yaw",
    "outer_pitch",
    "main_insertion",
    "tool_roll",
    "tool_pitch",
    "tool_yaw",
  )
  cfg.rewards["joint_pos_limits"].params["asset_cfg"].joint_names = main_joints
  cfg.rewards["joint_vel_hinge"].params["asset_cfg"].joint_names = main_joints

  joint_pos_obs = ObservationTermCfg(
    func=manipulation_mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.001, n_max=0.001),
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=main_joints)},
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

  return cfg


def psm_cut_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb",
  sim2real: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = psm_cut_env_cfg(play=play)

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
  cfg.observations["actor"].terms.pop("ee_to_object", None)
  cfg.observations["actor"].terms.pop("object_to_goal", None)

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
    cfg.events["rope_mount_color"] = EventTermCfg(
      mode="reset",
      func=dr.geom_rgba,
      params={
        "asset_cfg": SceneEntityCfg("rope_mount", geom_names=("rope_mount_visual",)),
        "ranges": {
          0: (0.45, 0.75),
          1: (0.45, 0.75),
          2: (0.45, 0.75),
          3: (1.0, 1.0),
        },
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
