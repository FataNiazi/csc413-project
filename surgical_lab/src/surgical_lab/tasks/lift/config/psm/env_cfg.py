import math
from typing import Any, Literal

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from surgical_lab import SURGICAL_LAB_SRC_PATH
from surgical_lab.mdp.actions import BinaryJointPositionActionCfg
from surgical_lab.mdp.recorders import PolicyCamVideoRecorderCfg
from surgical_lab.robots.psm.psm_constants import (
  get_psm_robot_cfg,
)
from surgical_lab.tasks.lift import mdp as lift_mdp
from surgical_lab.tasks.lift.lift_env_cfg import make_lift_env_cfg


def get_triangle_spec(scale: float = 0.01, mass: float = 0.05) -> mujoco.MjSpec:
  mesh_path = (
    SURGICAL_LAB_SRC_PATH / "robots" / "psm" / "xmls" / "assets" / "triangle.obj"
  )

  spec = mujoco.MjSpec()
  spec.add_mesh(name="triangle.obj", file=str(mesh_path), scale=(scale, scale, scale))

  body = spec.worldbody.add_body(name="triangle")
  body.add_freejoint(name="triangle_joint")
  body.mass = mass

  body.add_geom(
    name="triangle_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="triangle.obj",
    group=0,
    rgba=(0.8, 0.2, 0.2, 1.0),
    contype=0,
    conaffinity=0,
  )

  body.add_geom(
    name="triangle_c0",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.002, 0.01, 0.0075),
    pos=(-0.0055, -0.003, 0.0075),
    quat=(0.965093, 0, 0, 0.261909),
    group=3,
    rgba=(0.8, 0.2, 0.2, 1.0),
    contype=1,
    conaffinity=0,
  )
  body.add_geom(
    name="triangle_c1",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.002, 0.01, 0.0075),
    pos=(0.0055, -0.003, 0.0075),
    quat=(0.965093, 0, 0, -0.261909),
    group=3,
    rgba=(0.8, 0.2, 0.2, 1.0),
    contype=1,
    conaffinity=0,
  )
  body.add_geom(
    name="triangle_c2",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.002, 0.01, 0.0075),
    pos=(0, 0.0065, 0.0075),
    quat=(0.703845, 0, 0, 0.710353),
    group=3,
    rgba=(0.8, 0.2, 0.2, 1.0),
    contype=1,
    conaffinity=0,
  )

  body.add_site(
    name="grasp_site",
    size=(0.0005,),
    pos=(0, 0.0085, 0.0075),
    rgba=(0, 1, 0, 1),
    group=0,
  )

  return spec


def get_needle_spec(
  radius: float = 0.02, collision_radius: float = 0.002, density: float = 7800.0
) -> mujoco.MjSpec:
  """A 40mm surgical needle: a 180-deg arc lying flat in the XY plane.

  The mesh (needle_40mm.obj) is authored in metres, so no rescale. Its centerline
  is a semicircle of `radius` centred at the body origin, spanning theta in
  [90, 270] deg (the -x half-plane).

  Getting a thin, curved, light object to grasp stably took three coupled fixes,
  each addressing a distinct failure mode found by tracing the blowup:

  1. Freejoint ARMATURE (the root cause). A thin arc has one tiny principal
     inertia; a jaw impulse about it spins the needle to hundreds of rad/s and
     the free-flight integration then diverges on its own (gyroscopic qacc ->
     NaN) with no contact even present. Armature adds to the mass-matrix diagonal
     so an impulse yields a far smaller delta-omega and the spin integrates
     stably, while barely affecting translation (liftability).
  2. BOX colliders, not capsules. The PSM fingertips are capsules; a needle
     capsule near-parallel to a jaw capsule has an ill-defined contact normal
     (-> NaN). Boxes tangent to the arc (like the stable Lift-Peg colliders)
     give a well-defined normal.
  3. Enlarged contact buffers (see _swap_in_needle): 12 boxes overflow the peg's
     nconmax.

  The visual is the true thin swept-rod mesh; the collider is inflated to
  ~`collision_radius` so the jaws can seat on it. Mass/inertia come from the
  boxes at surgical-steel density.
  """
  mesh_path = (
    SURGICAL_LAB_SRC_PATH / "robots" / "psm" / "xmls" / "assets" / "needle_40mm.obj"
  )

  spec = mujoco.MjSpec()
  spec.add_mesh(name="needle_40mm.obj", file=str(mesh_path), scale=(1.0, 1.0, 1.0))

  body = spec.worldbody.add_body(name="needle")
  jnt = body.add_freejoint(name="needle_joint")
  jnt.armature = 1e-4  # see fix (1) above; ~170x the smallest inertia axis

  # Body origin at the grasped apex, NOT the arc's geometric center. The lift
  # command measures success/height from the object's root_link (body origin);
  # if that sits at the arc center it is 20mm of empty air from the grasped
  # material and ~13mm from the COM, so hoisting the needle barely moves the root
  # -> the lift reward is nearly flat and success is essentially unreachable
  # (observed: 0% success while the peg hits 85%). Shifting the arc by +radius in
  # x puts the apex (theta=180) at the origin, so root height == the height of the
  # point the jaws hold. This is a pure metric fix -- MuJoCo computes inertia
  # about the COM regardless, so physics/stability are unchanged.
  apex_shift = radius

  # Visual: the true swept-rod mesh (surgical-steel grey), no collision. Zero
  # density so all mass/inertia come from the collision boxes below. The mesh is
  # authored about the arc center, so offset it by the same apex shift to align
  # with the shifted colliders.
  body.add_geom(
    name="needle_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="needle_40mm.obj",
    pos=(apex_shift, 0.0, 0.0),
    group=0,
    rgba=(0.75, 0.75, 0.78, 1.0),
    contype=0,
    conaffinity=0,
    density=0.0,
  )

  # Collision: a chain of boxes tangent to the arc chord (fix (2) above). A thin
  # swept rod is unreliable as a single mesh collider, so approximate it
  # segment-wise. solref timeconst is softened to 8x dt for a little margin.
  n_seg = 12
  theta0, theta1 = math.pi / 2.0, 3.0 * math.pi / 2.0

  def arc_pt(t: float) -> tuple[float, float, float]:
    return (radius * math.cos(t) + apex_shift, radius * math.sin(t), 0.0)

  for i in range(n_seg):
    a = theta0 + (theta1 - theta0) * i / n_seg
    b = theta0 + (theta1 - theta0) * (i + 1) / n_seg
    p, q = arc_pt(a), arc_pt(b)
    mx, my = 0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])
    half_len = 0.5 * math.hypot(q[0] - p[0], q[1] - p[1])
    yaw = math.atan2(q[1] - p[1], q[0] - p[0])
    body.add_geom(
      name=f"needle_c{i}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=(mx, my, 0.0),
      quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)),
      size=(half_len + collision_radius, collision_radius, collision_radius),
      group=3,
      rgba=(0.75, 0.75, 0.78, 1.0),
      contype=1,
      conaffinity=0,
      density=density,
      solref=(0.04, 1.0),
    )

  # Grasp target: apex of the arc (theta = 180 deg) where the jaws close.
  gx, gy, gz = arc_pt(math.pi)
  body.add_site(
    name="grasp_site",
    size=(0.0005,),
    pos=(gx, gy, gz),
    rgba=(0, 1, 0, 1),
    group=0,
  )

  return spec


def psm_lift_peg_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_env_cfg()

  cfg.scene.entities = {
    "robot": get_psm_robot_cfg(tool_type="lnd"),
    "block": EntityCfg(spec_fn=get_triangle_spec),
  }

  psm_actuator_names = (
    "outer_yaw",
    "outer_pitch",
    "main_insertion",
    "tool_roll",
    "tool_pitch",
    "tool_yaw",
  )

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=psm_actuator_names,
      scale={"main_insertion": 0.05, "(outer_|tool_).*": 0.5},
      use_default_offset=True,
    ),
    "jaw_pos": BinaryJointPositionActionCfg(
      entity_name="robot",
      joint_names=("jaw",),
      open_command_expr={"jaw": 0.5},
      close_command_expr={"jaw": 0.0},
    ),
  }

  joint_pos_obs = ObservationTermCfg(
    func=manipulation_mdp.joint_pos_rel,
    noise=Unoise(n_min=-0.01, n_max=0.01),
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=psm_actuator_names)},
  )
  cfg.observations["actor"].terms["joint_pos"] = joint_pos_obs
  cfg.observations["critic"].terms["joint_pos"] = joint_pos_obs

  # Everything in the surgical robotics scene is scaled down so adjust the noise accordingly.
  cfg.observations["actor"].terms["ee_to_block"].noise = Unoise(
    n_min=-0.001, n_max=0.001
  )
  cfg.observations["critic"].terms["ee_to_block"].noise = Unoise(
    n_min=-0.001, n_max=0.001
  )
  cfg.observations["actor"].terms["block_to_goal"].noise = Unoise(
    n_min=-0.001, n_max=0.001
  )
  cfg.observations["critic"].terms["block_to_goal"].noise = Unoise(
    n_min=-0.001, n_max=0.001
  )

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_height"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.success_threshold = 0.01
  lift_command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(-0.05, 0.05),
    y=(0.0, 0.1),
    z=(0.02, 0.02),
    yaw=(-3.14, 3.14),
  )
  lift_command.difficulty = "dynamic"
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(0.0, 0.0),
    y=(0.02, 0.02),
    z=(0.06, 0.06),
  )

  cfg.observations["actor"].terms["ee_to_block"].params["asset_cfg"].site_names = (
    "tool_midpoint",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("tool_midpoint",)
  cfg.rewards["lift"].params["reaching_std"] = 0.1
  cfg.rewards["lift"].params["bringing_std"] = 0.1
  cfg.rewards["lift_precise"].params["std"] = 0.05

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
  # TODO: What should be the max vel for the main joints?
  cfg.rewards["joint_vel_hinge"].params["max_vel"] = 0.3

  grasp_site_cfg = SceneEntityCfg("block", site_names=("grasp_site",))

  # Replace the default ee_to_block observation with one targeting the grasp site.
  ee_to_grasp_site = ObservationTermCfg(
    func=lift_mdp.ee_to_object_site_distance,
    params={
      "object_name": "block",
      "object_cfg": grasp_site_cfg,
      "asset_cfg": SceneEntityCfg("robot", site_names=("tool_midpoint",)),
    },
    noise=Unoise(n_min=-0.001, n_max=0.001),
  )
  cfg.observations["actor"].terms["ee_to_block"] = ee_to_grasp_site
  cfg.observations["critic"].terms["ee_to_block"] = ee_to_grasp_site

  cfg.rewards["grasp"] = RewardTermCfg(
    func=lift_mdp.grasp_reward,
    weight=0.5,
    params={
      "object_name": "block",
      "reaching_std": 0.01,
      "ee_cfg": SceneEntityCfg("robot", site_names=("tool_midpoint",)),
      "jaw_cfg": SceneEntityCfg("robot", joint_names=("jaw",)),
      "object_cfg": grasp_site_cfg,
    },
  )
  cfg.rewards["open_jaw_lift"] = RewardTermCfg(
    func=lift_mdp.open_jaw_lift_penalty,
    weight=-3.0,
    params={
      "object_name": "block",
      "lift_threshold": 0.025,
      "jaw_cfg": SceneEntityCfg("robot", joint_names=("jaw",)),
    },
  )
  cfg.rewards["penetration"] = RewardTermCfg(
    func=lift_mdp.penetration_penalty,
    weight=-2.0,
    params={
      "object_name": "block",
      "interior_radius": 0.005,
      "ee_cfg": SceneEntityCfg("robot", site_names=("tool_midpoint",)),
    },
  )

  fingertip_geoms = r"^(lg|rg|tool_pitch_capsule).*_collision$"

  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
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
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def psm_lift_peg_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  sim2real: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = psm_lift_peg_env_cfg(play=play)

  camera_names = ["robot/wrist_camera"]
  cam_kwargs = {
    "robot/wrist_camera": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore
      **shared_cam_kwargs,  # type: ignore
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  # Pop privileged info from actor observations.
  cfg.observations["actor"].terms.pop("ee_to_block")
  cfg.observations["actor"].terms.pop("block_to_goal")

  # Log best/median/worst policy-view (wrist camera) videos to W&B during training.
  if not play and cam_type == "rgb":
    cfg.recorders["policy_cam_video"] = PolicyCamVideoRecorderCfg(
      sensor_name="wrist_camera",
      num_track=20,
      interval_s=600.0,
      fps=int(round(1.0 / (cfg.sim.mujoco.timestep * cfg.decimation))),
    )

  if sim2real:
    # Remove joint velocity observation for sim2real
    # as it is not reliable on the real robot
    cfg.observations["actor"].terms.pop("joint_vel")

  return cfg


def _swap_in_needle(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Replace the peg block with the needle and size up the contact buffers.

  The needle's box-chain collider (12 boxes) produces far more simultaneous
  contacts than the peg's 3 boxes -- a flat needle resting on the ground alone
  overflows the peg's nconmax=32, and a jaw squeeze adds more -- so enlarge the
  per-world contact (nconmax) and constraint (njmax) buffers for this object."""
  cfg.scene.entities["block"] = EntityCfg(spec_fn=get_needle_spec)
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 768

  # Lower the lift goal for the needle: 0.06 asks for a ~7.8cm hoist off the
  # floor, a long haul for a sparse success signal on an object this fiddly to
  # grasp (0% success after 16k iters). 0.03 (goal root ~5cm, ~4.8cm of lift)
  # keeps the task meaningful while making first success far easier to reach.
  # The peg keeps 0.06 -- it already solves that at ~85%.
  assert cfg.commands is not None
  lift_command = cfg.commands["lift_height"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(0.0, 0.0),
    y=(0.02, 0.02),
    z=(0.03, 0.03),
  )
  return cfg


def psm_lift_needle_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Lift a surgical needle. Identical to Lift-Peg but for the object geometry:
  every reward/observation keys on the "block" entity and its "grasp_site" by
  name, so swapping the spec is the only change."""
  return _swap_in_needle(psm_lift_peg_env_cfg(play=play))


def psm_lift_needle_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  sim2real: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _swap_in_needle(
    psm_lift_peg_vision_env_cfg(cam_type=cam_type, sim2real=sim2real, play=play)
  )
