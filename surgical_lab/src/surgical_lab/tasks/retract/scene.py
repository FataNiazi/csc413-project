"""Scene pieces for tissue retraction: the tissue, the table, and the markers.

Everything here obeys the rules in docs/cloth_simulation_report.md:

  * the tissue rests on a **box**, never a plane -- flex on a plane detonates in warp
  * vertices stay above ~0.05 g -- lighter ones explode
  * the gripper that must grasp it is 2.3 x 5.5 cm, comfortably above the ~3x-vertex-spacing
    floor below which flex cannot feel geometry at all
  * flex parameters are the aloha towel's, which is the only cloth setup we have seen work
"""

import mujoco

# The aloha towel's recipe, at a smaller grid (400 verts instead of 900) to keep the RL
# throughput sane. Spacing stays at 14 mm: the ALOHA finger must remain >> spacing.
CLOTH_COUNT = 10
CLOTH_SPACING = 0.014  # -> 12.6 cm square. Spacing stays 14mm: the ALOHA
# finger (2.3 x 5.5 cm) must stay well above the ~3x-spacing floor for flex to feel it.
CLOTH_RADIUS = 0.004  # collision radius (independent of stiffness thickness)
CLOTH_MASS = 0.03  # 0.30 g/vertex, well above the ~0.05 g explosion floor
CLOTH_THICKNESS = 1e-2  # bending stiffness thickness
CLOTH_YOUNG = 3e3
CLOTH_Z = 0.005  # just above the table top

TABLE_TOP_Z = 0.0
TABLE_HALF = (0.61, 0.37, 0.1)

# Grasp corner (max x, min y) and where it must be dragged to.
CORNER_IX, CORNER_IY = CLOTH_COUNT - 1, 0
DRAG_OFFSET = (0.0, 0.14, 0.05)  # pull the corner across the table and lift it clear


def cloth_extent() -> float:
  return CLOTH_SPACING * (CLOTH_COUNT - 1)


def corner_xy() -> tuple[float, float]:
  half = cloth_extent() / 2
  return (+half, -half)


def goal_pos() -> tuple[float, float, float]:
  cx, cy = corner_xy()
  return (cx + DRAG_OFFSET[0], cy + DRAG_OFFSET[1], CLOTH_Z + DRAG_OFFSET[2])


def get_table_spec() -> mujoco.MjSpec:
  """The table (a BOX: a flex body resting on a plane explodes), plus the task camera."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name="table_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, TABLE_TOP_Z - TABLE_HALF[2]),
    size=TABLE_HALF,
    rgba=(0.55, 0.55, 0.6, 1.0),
    contype=1,
    conaffinity=1,
    condim=3,
    friction=(1.0, 0.005, 0.0001),
  )
  # Overhead view of the tissue, for the image-based task.
  body.add_camera(
    name="overhead_cam",
    pos=(0.05, 0.0, 0.55),
    quat=(0.0, 0.0, 0.0, 1.0),
    fovy=58.0,
  )
  return spec


def get_tissue_spec() -> mujoco.MjSpec:
  """The tissue: a 2-D flex sheet, using the aloha towel's parameters."""
  cx, cy = 0.0, 0.0
  spec = mujoco.MjSpec.from_string(f"""
  <mujoco>
    <worldbody>
      <body name="tissue_root">
        <flexcomp type="grid" name="tissue" dim="2"
                  count="{CLOTH_COUNT} {CLOTH_COUNT} 1"
                  spacing="{CLOTH_SPACING} {CLOTH_SPACING} {CLOTH_SPACING}"
                  pos="{cx} {cy} {CLOTH_Z}"
                  radius="{CLOTH_RADIUS}" mass="{CLOTH_MASS}"
                  rgba="0.94 0.48 0.62 1">
          <edge equality="true"/>
          <contact selfcollide="none" condim="3"/>
          <elasticity young="{CLOTH_YOUNG}" poisson="0" thickness="{CLOTH_THICKNESS}"
                      damping="1e-3" elastic2d="bend"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  """)
  # flexcomp emits anonymous joints; mjlab resolves entity joints by name.
  for i, joint in enumerate(spec.joints):
    if not joint.name:
      joint.name = f"tissue_flex_joint_{i}"
  return spec


def get_marker_spec() -> mujoco.MjSpec:
  """Visual markers: blue = the corner to grasp, green = where to drag it. No collision."""
  spec = mujoco.MjSpec()
  cx, cy = corner_xy()
  gx, gy, gz = goal_pos()

  body = spec.worldbody.add_body(name="markers")
  body.add_geom(
    name="grasp_marker",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(cx, cy, CLOTH_Z),
    size=(0.008, 0, 0),
    rgba=(0.1, 0.4, 1.0, 0.9),
    contype=0,
    conaffinity=0,
    group=0,
  )
  body.add_geom(
    name="goal_marker",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=(gx, gy, gz),
    size=(0.012, 0, 0),
    rgba=(0.1, 0.9, 0.2, 0.9),
    contype=0,
    conaffinity=0,
    group=0,
  )
  return spec
