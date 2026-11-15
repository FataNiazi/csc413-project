"""Single-arm ALOHA (ViperX 300s) entity.

The ALOHA MJCF is not vendored in this repo; it comes from mujoco_menagerie at the ref
pinned in benchmarks/aloha/__init__.py. Point ALOHA_ASSET_DIR at a directory containing
aloha.xml and its assets/ (see docs/cloth_simulation_report.md for the fetch recipe).

We use ALOHA rather than the PSM for cloth tasks for a physical reason: MuJoCo Warp's flex
collision cannot feel geometry thinner than roughly 3x the cloth's vertex spacing, and the
ALOHA finger's collision AABB (2.3 x 5.5 cm) clears that bar, while the PSM's millimetre
jaw does not.
"""

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ALOHA_ASSET_DIR = Path(
  os.environ.get(
    "SURGICAL_LAB_ALOHA_DIR", "/scratch/anoopreh/mjbench_assets/aloha_cloth_upstream"
  )
)

ARM = "left"  # which of the two ALOHA arms to keep

# mjlab namespaces an entity's elements and strips the MJCF's own "left/" prefix, so
# inside the entity these are bare names.
ARM_ACTUATORS = (
  "waist",
  "shoulder",
  "elbow",
  "forearm_roll",
  "wrist_angle",
  "wrist_rotate",
)
GRIPPER_ACTUATOR = "gripper"
FINGER_JOINTS = ("left_finger", "right_finger")
EE_SITE = "gripper"

# Gripper actuator ctrlrange is [0.002, 0.037] (finger rail displacement, metres).
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.002


def get_aloha_arm_spec() -> mujoco.MjSpec:
  """Load ALOHA and delete everything except one arm."""
  xml = ALOHA_ASSET_DIR / "aloha.xml"
  if not xml.exists():
    raise FileNotFoundError(
      f"ALOHA MJCF not found at {xml}. Set SURGICAL_LAB_ALOHA_DIR, or fetch the assets "
      "(see docs/cloth_simulation_report.md)."
    )
  spec = mujoco.MjSpec.from_file(str(xml))

  other = f"{'right' if ARM == 'left' else 'left'}/"
  # Delete the other arm. Order matters: keyframes carry a ctrl vector sized for both
  # arms and actuators/equalities reference bodies, so they all go before the bodies.
  for key in list(spec.keys):
    spec.delete(key)
  # The robot MJCF carries lights that target arm bodies; the scene supplies its own.
  for light in list(spec.lights):
    spec.delete(light)
  # mjlab builds its own <position> actuators from the EntityCfg, so drop the MJCF's.
  for act in list(spec.actuators):
    spec.delete(act)
  for eq in [e for e in spec.equalities if e.name and e.name.startswith(other)]:
    spec.delete(eq)
  for body in [b for b in spec.bodies if b.name.startswith(other) and b.parent.name == "world"]:
    spec.delete(body)

  return spec


# Gains read off the ALOHA MJCF's own position actuators.
ARM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=ARM_ACTUATORS,
  stiffness=100.0,
  damping=10.0,
  effort_limit=50.0,
)
GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=FINGER_JOINTS,
  stiffness=2000.0,
  damping=124.0,
  effort_limit=100.0,
)

HOME = EntityCfg.InitialStateCfg(
  joint_pos={
    "waist": 0.0,
    "shoulder": 0.5,
    "elbow": -0.4,
    "forearm_roll": 0.0,
    "wrist_angle": 0.6,
    "wrist_rotate": 0.0,
    ".*_finger": GRIPPER_OPEN,
  },
)


ARTICULATION = EntityArticulationInfoCfg(
  actuators=(ARM_ACTUATOR_CFG, GRIPPER_ACTUATOR_CFG),
)


def get_aloha_entity_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME,
    spec_fn=get_aloha_arm_spec,
    articulation=ARTICULATION,
  )
