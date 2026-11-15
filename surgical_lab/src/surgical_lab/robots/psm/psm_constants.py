"""dVRK Patient Side Manipulator (PSM) constants."""

from pathlib import Path
from typing import Literal

import os

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
)
from mjlab.utils.spec_config import CollisionCfg

from surgical_lab import SURGICAL_LAB_SRC_PATH

##
# MJCF and assets.
##

PSM_ROOT_XML: Path = SURGICAL_LAB_SRC_PATH / "robots" / "psm" / "xmls"

assert PSM_ROOT_XML.exists()

PSM_LND_XML: Path = PSM_ROOT_XML / "psm_lnd.xml"

PSM_RTS_XML: Path = PSM_ROOT_XML / "psm_rts.xml"


def get_spec(tool_type: Literal["lnd", "rts"]) -> mujoco.MjSpec:
  if tool_type == "lnd":
    spec = mujoco.MjSpec.from_file(str(PSM_LND_XML))
  elif tool_type == "rts":
    spec = mujoco.MjSpec.from_file(str(PSM_RTS_XML))
  else:
    raise ValueError(f"Invalid tool type: {tool_type}")

  # The printed wrist-camera bracket is OFF by default. It is a real 37.5 mm
  # part on an 8.8 mm shaft, so it occludes a 45-degree overhead camera badly:
  # measured tissue pixels visible in the 84 px policy view fall from ~390 with
  # no bracket to 152 with it mounted 26 mm up the shaft. Any task using the
  # overhead camera should not pay that unless it is actually using the wrist
  # camera too. Set PSM_WRIST_MOUNT=1 to include it.
  if os.environ.get("PSM_WRIST_MOUNT", "0") != "1":
    for g in list(spec.geoms):
      if g.name == "wrist_camera_mount":
        spec.delete(g)

  # Shaft diameter. THREE FIGURES DISAGREE and none of them has been reconciled:
  # the XML models a 0.0042 capsule radius (8.4 mm), its own bracket comment
  # says the shaft is 8.80 mm, and the bench measures 10 mm. Overridable rather
  # than edited, because psm_lnd.xml is shared with liftpeg and pushblock and
  # this is not the branch to change their instrument on.
  _d = os.environ.get("PSM_SHAFT_DIAMETER", "")
  if _d:
    r = float(_d) / 2000.0
    for g in list(spec.geoms):
      if g.name in ("main_insertion_link", "main_insertion_link_collision"):
        g.size[0] = r
  return spec


##
# Actuator config.
##

# dVRK PSM Motor Constants.
# Based on Maxon RE025 specifications and dVRK documentation.
#
# Groups:
# 1. Large Motors (J1 Outer Yaw, J2 Outer Pitch): Maxon RE025-Twin, ~56.5:1 gear ratio.
# 2. Insertion Motor (J3): Maxon RE025, linear drive.
# 3. Tool Motors (J4-J7): Maxon RE025, ~11.71:1 gear ratio (plus cable transmission).

# Inertia (Armature) values [kg*m^2] for rotary, [kg] for linear
# Estimated reflected inertia
ARMATURE_BIG = 0.009  # J1, J2 (High gear ratio 56.5^2 * J_motor)
ARMATURE_TOOL = 0.0005  # J4-J7 (Lower gear ratio 11.71^2 * J_motor)
ARMATURE_INSERTION = 0.5  # J3 (Effective mass of insertion carriage)

BIG_MOTOR = ElectricActuator(
  reflected_inertia=ARMATURE_BIG,
  velocity_limit=2.0,
  effort_limit=5.0,
)

INSERTION_MOTOR = ElectricActuator(
  reflected_inertia=ARMATURE_INSERTION,
  velocity_limit=0.25,
  effort_limit=15.0,
)

TOOL_MOTOR = ElectricActuator(
  reflected_inertia=ARMATURE_TOOL,
  velocity_limit=5.0,
  effort_limit=0.5,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0  # Overdamped

STIFFNESS_BIG = ARMATURE_BIG * NATURAL_FREQ**2
DAMPING_BIG = 2.0 * DAMPING_RATIO * ARMATURE_BIG * NATURAL_FREQ

STIFFNESS_INSERTION = ARMATURE_INSERTION * NATURAL_FREQ**2
DAMPING_INSERTION = 2.0 * DAMPING_RATIO * ARMATURE_INSERTION * NATURAL_FREQ

STIFFNESS_TOOL = ARMATURE_TOOL * NATURAL_FREQ**2
DAMPING_TOOL = 2.0 * DAMPING_RATIO * ARMATURE_TOOL * NATURAL_FREQ

ACTUATOR_OUTER = BuiltinPositionActuatorCfg(
  target_names_expr=("outer_yaw", "outer_pitch"),
  stiffness=STIFFNESS_BIG,
  damping=DAMPING_BIG,
  effort_limit=BIG_MOTOR.effort_limit,
  armature=BIG_MOTOR.reflected_inertia,
)

ACTUATOR_INSERTION = BuiltinPositionActuatorCfg(
  target_names_expr=("main_insertion",),
  stiffness=STIFFNESS_INSERTION,
  damping=DAMPING_INSERTION,
  effort_limit=INSERTION_MOTOR.effort_limit,
  armature=INSERTION_MOTOR.reflected_inertia,
)

ACTUATOR_TOOL = BuiltinPositionActuatorCfg(
  target_names_expr=("tool_roll", "tool_pitch", "tool_yaw", "jaw"),
  stiffness=STIFFNESS_TOOL,
  damping=DAMPING_TOOL,
  effort_limit=TOOL_MOTOR.effort_limit,
  armature=TOOL_MOTOR.reflected_inertia,
)

##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0.1, 0),
  joint_pos={
    "outer_yaw": 0.0,
    "outer_pitch": 0.0,
    "main_insertion": 0.10,
    "tool_roll": 0.0,
    "tool_pitch": 0.0,
    "tool_yaw": 0.0,
    "jaw": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_GRIPPER_GEOM_REGEX = r"^(lg|rg|tool_pitch_capsule).*_collision$"

# Gripper only collision config
GRIPPER_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={
    _GRIPPER_GEOM_REGEX: 0,
    ".*_collision": 0,
  },
  conaffinity={
    _GRIPPER_GEOM_REGEX: 1,
    ".*_collision": 0,
  },
  condim={
    _GRIPPER_GEOM_REGEX: 6,
    ".*_collision": 3,
  },
  friction={
    _GRIPPER_GEOM_REGEX: (1, 5e-3, 5e-4),
    ".*_collision.*": (0.6,),
  },
  solref={
    _GRIPPER_GEOM_REGEX: (0.01, 1),
  },
  priority={
    _GRIPPER_GEOM_REGEX: 1,
  },
)

##
# Final config.
##

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ACTUATOR_OUTER,
    ACTUATOR_INSERTION,
    ACTUATOR_TOOL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_psm_robot_cfg(tool_type: Literal["lnd", "rts"] = "lnd") -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(GRIPPER_ONLY_COLLISION,),
    spec_fn=lambda: get_spec(tool_type),
    articulation=ARTICULATION,
  )


PSM_ACTION_SCALE: dict[str, float] = {}
for a in ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    # PSM_ACTION_SCALE[n] = 0.25 * e / s
    PSM_ACTION_SCALE[n] = 0.5  # Hardcoded for now

PSM_ACTION_SCALE["main_insertion"] = 0.05  # Hardcoded for now

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_psm_robot_cfg(tool_type="rts"))

  viewer.launch(robot.spec.compile())
