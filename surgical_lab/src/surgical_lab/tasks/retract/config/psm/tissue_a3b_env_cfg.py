"""Tissue-lift on the a3b scene: 3 corners pinned, lift the free one straight up.

Same composition trick as liftpeg — start from `psm_push_wide_a3b_env_cfg()` so
the PSM, Unity home pose, blue floor and DR are shared, then swap the object,
the camera and the task terms.

Layout: a square flex sheet lying on the blue floor with **three corners
pinned**, so the only thing that can move is the fourth. The task is to grasp
that free corner and raise it to a fixed height — a claw machine, same success
definition as liftpeg.

## READ THIS BEFORE TRAINING IT

**This scene cannot be grasped by the dVRK as-is, and that is a physics
limitation, not a tuning problem.** From `docs/cloth_simulation_report.md` (Rule
3) and confirmed in `docs/RETRACTION_HANDOFF.md`: MuJoCo Warp's flex collision
cannot feel geometry thinner than roughly 3x the cloth's vertex spacing. At the
spacing below that threshold is ~2 cm; the PSM's jaw is a couple of millimetres
across, so the jaw passes straight through the sheet. The earlier tissue runs
used ALOHA (2.3 x 5.5 cm finger AABB) precisely for this reason.

Three ways out, none of which I picked for you (see PLAN.md):
  (a) fatten the jaw's COLLISION geoms to ~2-3 cm, visuals unchanged
  (b) much finer cloth -- but Rule 2 says finer needs more mass or it explodes
  (c) virtual attach: weld the corner to the jaw when close and closed

So this module builds and renders the scene, and defines the task terms, but
nothing here has been trained. Pick a route first.
"""

from __future__ import annotations

import math
import os

# --- camera view switch -----------------------------------------------------
# side | wrist | sbs.
#
#   side   the bench camera alone. THE DEFAULT, and byte-identical to what this
#          task has always done -- every existing tissue run stays comparable.
#   wrist  the tool-mounted fisheye alone, rendered wide and warped down to
#          84x84.
#   sbs    front | wrist concatenated on WIDTH -> 84x168, the layout the
#          side-by-side MAE is pretrained on.
#
# The wrist views need the printed camera bracket in the robot spec, and
# psm_constants reads PSM_WRIST_MOUNT when the spec is BUILT -- which happens
# inside psm_lift_tissue_a3b_env_cfg() below, long after this module is
# imported. Setting it here, at module scope and before anything touches the
# robot, is what makes the mount actually appear; setting it inside the builder
# would work too but only by accident of ordering, and setting it from the
# caller has already been forgotten once.
TISSUE_CAM_VIEW = os.environ.get("TISSUE_CAM_VIEW", "side")
if TISSUE_CAM_VIEW not in ("side", "wrist", "sbs"):
  raise ValueError(
    f"TISSUE_CAM_VIEW must be side|wrist|sbs, got {TISSUE_CAM_VIEW!r}")
if TISSUE_CAM_VIEW in ("wrist", "sbs"):
  os.environ["PSM_WRIST_MOUNT"] = "1"

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg

from surgical_lab.tasks.lift.config.psm import liftpeg_a3b_env_cfg as lp
from surgical_lab.tasks.lift.config.psm import liftpeg_scene as LS
from surgical_lab.tasks.push.config.psm import a3b_env_cfg as a3b
from surgical_lab.tasks.retract.mdp import grasp as gr
from surgical_lab.tasks.retract.mdp import rewards_scripted as rs
from surgical_lab.tasks.retract.mdp import rewards_tissue as rt
from surgical_lab.tasks.retract.mdp import visibility as vis
from surgical_lab.tasks.retract.mdp.cloth import corner_index, flex_vertex_pos_w

# Sized to the PSM's workspace, not ALOHA's table: a 12.6 cm sheet would put the
# free corner far outside anything this arm can reach from the Unity home pose.
# Grid resolution. 8 quads per side cannot represent a smooth drape -- the sheet
# folds along straight creases where real tissue tents -- and there is headroom
# to refine it: mass per vertex must stay above ~0.05 g or mujoco_warp's flex
# explodes (docs/tissue_scene_findings.md, Bug B), and at TISSUE_MASS = 0.02 kg
#
#      9x9   81 verts   0.247 g/vertex
#     13x13 169 verts   0.118 g/vertex
#     17x17 289 verts   0.069 g/vertex   <- still ~1.4x the floor
#     21x21 441 verts   0.045 g/vertex   <- below it
#
# so 17 is the finest grid available without also raising the mass.
#
# MEASURED, and the answer is no. 17x17 does reduce the faceting -- the p95
# shading gradient inside the sheet falls from 0.0552 to 0.0214, a 61% cut in
# the sharpest steps, which is what reads as "jank" -- but it costs
#
#      9x9   9370 env-steps/s (256 envs)
#     17x17   769 env-steps/s     -> 12.2x slower for 3.6x the vertices
#
# superlinear, as tissue_scene_findings.md warned ("cost grows ~9x from 1x to
# 4x resolution, which matters at 4096 envs"). Not worth it for training.
#
# The larger remaining gap is TEXTURE, not resolution, and it is free: the real
# sheet shows 80 distinct shading levels and 17.5% of its interior pixels on a
# shading step, against sim's 37 and 5.1%. Real silicone has wrinkles and sheen;
# sim's tissue is flat-shaded. A material would close more of the perceptual gap
# than quadrupling the vertex count, at no runtime cost.
TISSUE_COUNT = int(os.environ.get("TISSUE_COUNT", "9"))
# The sheet is RECTANGULAR on the rig -- sim modelled it square -- and it is
# BIGGER than the first bench figures suggested.
#
# The drawing gave 50 x 55 mm, which has a 74.3 mm diagonal. Measuring the
# diagonal directly gives 80 mm, so those were the VISIBLE tan under the tape
# rather than the sheet itself. Keeping the drawing's 50:55 ratio and scaling it
# to an 80 mm diagonal gives 53.8 x 59.2.
#
# The direct diagonal is the one to trust: the rig's visible tan spans 79.3 mm
# in frame, which is that diagonal almost exactly -- so from the scene camera
# the tape hides essentially none of the sheet, and the fitted frame width was
# right after all. The sheet was simply 8% small.
TISSUE_SIDE_X = float(os.environ.get("TISSUE_SIDE_X", "0.0538"))
TISSUE_SIDE_Y = float(os.environ.get("TISSUE_SIDE_Y", "0.0592"))
TISSUE_SIDE = TISSUE_SIDE_X          # kept for anything still reading one side
# The flexcomp gets the same COUNT on both axes and different spacings, so the
# grid stays square in index space -- corner_index, patch_ids and pinned_ids all
# assume that -- while the sheet itself is rectangular in metres.
TISSUE_SPACING_X = TISSUE_SIDE_X / (TISSUE_COUNT - 1)
TISSUE_SPACING_Y = TISSUE_SIDE_Y / (TISSUE_COUNT - 1)
TISSUE_SPACING = TISSUE_SPACING_X    # legacy alias
# Vertex collision radius. With the floor implemented as a joint limit (see
# below) the flex never collides with anything, so this is now purely cosmetic:
# MuJoCo renders a flex sheet inflated by `radius`, so 3 mm at 6.25 mm spacing
# drew the sheet as a lumpy union of near-touching spheres. Small = smooth.
TISSUE_RADIUS = 0.0008
TISSUE_MASS = 0.02          # 0.25 g/vertex, above the ~0.05 g explosion floor
# None selects MuJoCo's inextensible <edge equality> instead of membrane
# elasticity. Measured during a peel, with the drag clamped at 22 mm:
#     young 3e3   max edge stretch 3.84x
#     edge        max edge stretch 1.57x
# Unbounded (no drag clamp) it was 8.05x and the sheet tore open, showing the
# floor through a hole. Real tissue folds rather than stretching, and an
# inextensible sheet is the closer model; the two options are mutually
# exclusive in MuJoCo ("flex constraints and elasticity cannot both be
# present"), so this is the half worth keeping.
# None selects MuJoCo's inextensible <edge equality>; a number selects membrane
# elasticity instead. They are mutually exclusive ("flex constraints and
# elasticity cannot both be present").
#
# None makes the sheet behave like PAPER -- it creases and folds but cannot
# stretch at all. The comment that chose it argued "real tissue folds rather
# than stretching", which is the assumption worth testing: the rig's silicone
# visibly stretches under the jaw.
TISSUE_YOUNG = (None if os.environ.get("TISSUE_YOUNG", "") in ("", "none", "None")
                else float(os.environ["TISSUE_YOUNG"]))
# --- appearance, fitted against data/Sep2 --------------------------------
# See docs/tissue_s2r_alignment.md for the measurements. Two things matter and
# they are coupled, which is why neither is a standalone constant:
#
#  1. The scene renders at 0.610x albedo at its BRIGHTEST (nominal diffuse is
#     the top of the light DR's range). The real tissue's red is 0.836, so the
#     required albedo would be 1.37 -- impossible. The light has to come up
#     before any colour is matchable; see TISSUE_LIGHT_DR.
#  2. The sheet was flat pink (0.94, 0.48, 0.62) against a real tissue that
#     measures (0.836, 0.630, 0.435) -- tan. There is no appearance DR on the
#     tissue at all (the shared terrain/light DR does not touch it), so this
#     was a FIXED wrong colour, not a distribution missing its centre.
#
# The light is scaled HERE and not in LS.LIGHT_DIFFUSE, which liftpeg shares
# (liftpeg_a3b_env_cfg.py) and which has a deployed policy behind it.
# Sets the light on the SPEC, which only survives when the light DR is off
# (play, or replay_real_episode.py --pin-light). During training a3b's
# light_intensity event overwrites it every reset -- see TISSUE_LIGHT_DR, which
# is the knob that actually acts.
TISSUE_LIGHT_SCALE = float(os.environ.get("TISSUE_LIGHT_SCALE", "1.8"))
# a3b's light_intensity writes light_diffuse ABSOLUTELY as U(0.15, 1.0). With
# the light pinned and measured against data/Sep2:
#
#     sim at nominal diffuse (= the TOP of that range)   render factor 0.610
#     the real bench                                     render factor 0.901
#
# The real rig is BRIGHTER than the brightest draw the DR can make, so no
# episode ever looked like it -- the whole range sits below the target. Fitting
# factor against diffuse (1.0 -> 0.610, 1.8 -> 0.890) puts the real bench at an
# equivalent diffuse of ~1.8. Re-centred to bracket that, keeping Unity's ~7x
# dynamic range (2.0 / 0.3 = 6.7x) rather than narrowing the DR.
TISSUE_LIGHT_DR = tuple(
  float(v) for v in os.environ.get("TISSUE_LIGHT_DR", "0.3,2.0").split(",")
)
# Fitted to REAL Sep2/e0's tissue, (0.825, 0.641, 0.429) rendered.
#
# e14 was used first, on the reasoning that sim's own render already sat closest
# to it, and its FLOOR is still what the floor nominal below is fitted to. But
# e0 is the episode this work has been eyeballed against throughout, and by eye
# its tissue is the better match -- the e14 fit renders the sheet with markedly
# less blue (0.305 against 0.429), which reads more saturated and darker than
# the bench does.
#
# The two episodes' tissue really does differ that much: blue 0.429 in e0
# against 0.304 in e14, which is most of the bench's episode-to-episode spread
# in one channel. The nominal can only sit on one of them.
#
# Superseded reasoning, kept because the argument still stands for the floor:
#
# The bench's colour varies a lot between episodes (floor saturation runs 0.27
# to 0.85), and averaging a bimodal spread gives a condition that never
# occurred. Better for the nominal to BE a lighting condition the rig actually
# produced, with the DR spanning the rest. e14 is the episode sim's own render
# already sat closest to, so it is the least arbitrary choice.
#
#     e14 tissue   0.839, 0.587, 0.304   floor  0.278, 0.597, 0.682
#
# An earlier (0.93, 0.70, 0.48) came from e0 alone, and rendered a sheet pinker
# than the bench: 0.429 blue against e14's 0.304.
TISSUE_RGB = tuple(
  float(v) for v in os.environ.get("TISSUE_RGB", "0.700,0.551,0.384").split(",")
)
# Floor nominal, tissue-local: LS.FLOOR_RGB is Unity ground truth and shared
# with liftpeg, so it is overridden on the assembled spec rather than edited.
# e14's floor divided by the floor's own render factor.
TISSUE_FLOOR_RGB = tuple(
  float(v) for v in os.environ.get("TISSUE_FLOOR_RGB", "0.249,0.550,0.655").split(","))
# --- floor colour DR --------------------------------------------------------
# a3b draws the floor from R(0, 0.4) G(0.6, 0.8) B(0.95, 1.0). Measured across
# the 30 Sep2 episodes and divided by the floor's own render factor, the real
# bench implies an albedo of R 0.06-0.35, G 0.61-0.83, **B 0.64-0.91**.
#
# The blue ranges are DISJOINT: the DR never draws below 0.95 and the rig never
# reaches above 0.91, so no episode of training has ever shown the policy a
# floor the colour of the one it will be deployed onto.
#
# The width matters as much as the centre. The bench's floor hue is stable
# (187-198 deg across all 30) but its SATURATION runs 0.27 to 0.85 -- a 3x
# swing, which is what "washed out in some episodes" is. The R range carries
# that axis, so it stays wide.
# Centred on e14 -- the nominal -- and wide enough to still cover the whole
# observed bench spread (R 0.06-0.35, G 0.61-0.83, B 0.64-0.91 in albedo).
TISSUE_FLOOR_RGB_DR = (
  tuple(float(v) for v in os.environ.get("TISSUE_FLOOR_R", "0.06,0.52").split(",")),
  tuple(float(v) for v in os.environ.get("TISSUE_FLOOR_G", "0.61,0.92").split(",")),
  tuple(float(v) for v in os.environ.get("TISSUE_FLOOR_B", "0.64,0.97").split(",")),
)


def set_tissue_floor_colour(spec) -> None:
  """Point the terrain at this task's floor nominal, on the assembled spec."""
  for g in spec.geoms:
    if g.name == "terrain":
      g.rgba = tuple(TISSUE_FLOOR_RGB) + (1.0,)
# NOTE: there is still no appearance DR on the tissue, only this fixed colour.
# The shared terrain/light DR does not reach it, and dr.geom_rgba cannot: the
# sheet is a flexcomp, not a geom. Randomising it needs a flex-aware term, and
# until that exists the sheet is the one object in the scene whose colour the
# policy sees exactly one of.
TISSUE_THICKNESS = 1e-2
# Spawn height. MUST move together with TISSUE_FLOOR_Z: the floor is a joint
# limit expressed RELATIVE to this (range = [TISSUE_FLOOR_Z - TISSUE_Z, 1]),
# so raising the floor alone puts the limit ABOVE the spawn point and every
# vertex starts in violation -- the sheet is then held up by 81 violated
# constraints instead of resting on a floor, which reads as the membrane
# tenting and springing. Raise BOTH to lift the sheet.
TISSUE_Z = float(os.environ.get("TISSUE_Z", "0.004"))
# World z below which no tissue vertex may go -- the floor, as a constraint
# rather than a contact. See _limit_vertex_drop().
# 3 mm, not 1.5. Two reasons, both measured:
#  - the joint limit is SOFT. It is violated by a consistent ~0.9 mm at every
#    setting tried (1.5 / 3.0 / 4.0 mm), because the constraint solver trades
#    limit violation against the inextensible-edge constraints.
#  - the sheet must never reach the goal square's own plane (z=0). When it did,
#    rays from the camera to a goal sample point terminated exactly AT the
#    tissue, the intersection landed at t~=1.0, and the visibility test's
#    end-point epsilon discarded it as "not blocking" -- reporting the goal
#    fully visible with the sheet lying flat on top of it. That fired the
#    success termination on ~every other step: 1600 false successes over 200
#    steps at floor=1.5 mm, and 0 at floor=3.0 mm.
TISSUE_FLOOR_Z = float(os.environ.get("TISSUE_FLOOR_Z", "0.003"))

# The sheet is centred on the peg spawn point, so the free corner lands roughly
# where liftpeg's peg does and the same camera framing works for both.
# The sheet is placed so its FREE CORNER -- the only part that matters -- lands
# near where liftpeg's peg sat, i.e. near the middle of the frame. Centring the
# SHEET there instead (the obvious choice, and what this used to do) puts the
# grasp corner 35 mm off-centre diagonally, which pushed it to the very edge of
# the image: measured normalised frame coordinate |y| = 0.908 where 1.0 is the
# border, a handful of pixels at 84x84. In frame, but not usefully visible.
#
#   shift  corner |x|  corner |y|  whole sheet in frame  tool->corner
#    0.0      0.738       0.908           94.1 %            37.1 mm
#    0.5      0.437       0.880           99.6 %            20.9 mm
#    0.8      0.266       0.747          100.0 %            13.0 mm   <- chosen
#    1.0      0.216       0.732           99.2 %            10.8 mm
#
# 0.8 rather than 1.0 because it is the only one that frames the WHOLE sheet.
# It also puts the corner 13 mm from the tool's home tip instead of 37 mm,
# which makes the approach far less fiddly to drive by hand.
# The free corner sits DIRECTLY BELOW the tool's home tip, so the robot begins
# the task by descending straight down -- no navigating to the grasp point.
# That is how the real rig will be set up, and it removes a whole approach
# phase from the task.
#
# Measured home tip with the DR off: (0.01, 12.56, 15.19) mm. The corner sits
# at tissue_centre + (-half, +half), so the centre is the tip xy mirrored.
HOME_TIP_XY = (0.0, 0.01256)
# Where the grasp corner actually sits relative to the home tip, measured over
# all 30 episodes of data/Sep2 as the tool's LATERAL travel from home to the
# first jaw close:
#
#     dx   mean  +2.4 mm  sd 4.8
#     dy   mean +24.7 mm  sd 2.3
#     |xy| mean  25.3 mm  sd 2.4   (range 20.8-29.5)
#
# The comment this replaces asserted the opposite -- that the free corner sits
# DIRECTLY BELOW the home tip, "how the real rig will be set up". It is not how
# the rig was set up: the real tool travels ~25 mm sideways on its way down,
# every episode, and dy's sd of 2.3 mm says that is the scene rather than
# operator scatter. With the corner underneath the tip, sim deleted an approach
# phase the rig actually performs.
#
# The PSM base frame and the sim world agree 1:1 here -- replaying Sep2/e0
# through --drive eef reproduces its real (-3.9, +26.7) mm as (-4.0, +26.5) --
# so these go in unrotated.
GRASP_OFFSET_XY = (0.0024, 0.0247)
# Rigid shift of the whole sheet, metres, world XY. Exists so the measured
# jaw-to-corner gap at jaw closure can be driven to zero without disturbing
# GRASP_OFFSET_XY, which is a FITTED approach geometry and means something
# else. Default 0: across 30 Sep2 episodes the lateral gap has mean +0.09 mm
# against sd 4.87, so there is no fleet-wide bias for a constant to remove --
# this is for per-episode correction and for testing single-episode fits.
TISSUE_XY_OFFSET = tuple(
  float(v) for v in os.environ.get("TISSUE_XY_OFFSET", "0,0").split(","))
TISSUE_CX = (HOME_TIP_XY[0] + GRASP_OFFSET_XY[0] + TISSUE_SIDE_X / 2.0
             + TISSUE_XY_OFFSET[0])
TISSUE_CY = (HOME_TIP_XY[1] + GRASP_OFFSET_XY[1] - TISSUE_SIDE_Y / 2.0
             + TISSUE_XY_OFFSET[1])
LIFT_TARGET_Z = 0.040
LIFT_TOL = lp.LIFT_TOL

# Two ADJACENT edges are taped down, which pins three corners and leaves
# exactly one free -- the real-rig equivalent of running two strips of tape in
# an L. Point-pinning three corners (the old scheme) leaves the edges between
# them free to lift, so the sheet could be peeled from the wrong side.
#
# The free corner faces the CAMERA (+y), so the flap peels toward the lens and
# the revealed goal square is not occluded by the tissue that just uncovered it.
_LAST = TISSUE_COUNT - 1
FREE_CORNER = (0, _LAST)                  # -x, +y : near side, camera-facing
TAPED_EDGE_IX = _LAST                     # the +x edge
TAPED_EDGE_IY = 0                         # the -y (far) edge
# Push-block green desaturated to the bench's. The HUE was already right --
# 119 deg rendered against the real tape's 119 -- it was only ever too vivid:
# saturation 0.68 rendered against 0.40 measured across the 30 Sep2 episodes.
# So this keeps the hue and drops the saturation, rather than picking a new
# colour.
TAPE_RGB = tuple(
  float(v) for v in os.environ.get("TISSUE_TAPE_RGB", "0.348,0.683,0.358").split(","))
# Yaw of the whole sheet about its FREE CORNER, degrees.
#
# The real tissue is taped down corner-on -- a diamond -- and sim had it
# edge-on. Measured on data/Sep2/e0 as the mask's width profile from the top of
# the sheet to the bottom (1.0 = widest row):
#
#     REAL   0.04 0.29 0.53 0.82 1.00 0.90 0.79 0.55 0.37 0.21   <- diamond
#     SIM    0.68 0.71 0.74 0.76 0.79 0.84 0.88 0.96 0.98 1.00   <- edge-on
#
# A bench camera swung 45 deg in azimuth produces the SAME picture, so the side
# view cannot tell the two apart. The WRIST camera can: it is bolted to the
# tool and tool_roll moves 1.2 deg across an episode, so its frame is the
# robot's and it is blind to where the bench camera sits. The real wrist view
# reads 0.13 0.30 0.50 0.68 0.86 1.00 0.87 0.60 0.43 0.22, fully inside the
# frame -- a diamond in the ROBOT's frame. So it is the sheet, not the camera.
#
# -45 and not +45: at yaw 0 the free->taped diagonal runs (+x, -y); rotating it
# by -45 sends it to (0, -1), i.e. straight away from the camera, which puts
# the taped corner at the TOP of the image and the free corner nearest -- the
# real layout.
TISSUE_YAW_DEG = float(os.environ.get("TISSUE_YAW_DEG", "-45.0"))
# --- home height ----------------------------------------------------------
# `main_insertion` for default_joint_pos, the same override liftpeg makes in
# its HOME_JOINTS. Without it this task silently inherits a3b's default of
# 0.10, whose lower_tool comment sources the resulting tip height to "Unity's
# home height, read off a PLAY dump" -- Unity's convention, never the rig's.
#
# What that inheritance cost, measured with every randomiser pinned:
#
#                  sim home tip   target top   sim approach   real approach
#     liftpeg          73.88 mm      13.50 mm       60.4 mm        32.8 mm
#     tissue (was)     13.55 mm       4.80 mm        8.8 mm        48.3 mm
#
# The real arm homes at the same z for BOTH tasks (eef_position z0 = -101.8
# mm); the 15.5 mm between the two real approaches is just the peg's grasp
# point standing that far above the tissue surface. Neither sim task matched
# its rig -- liftpeg starts 1.8x too high, tissue started 5.5x too low -- so
# there was never a depth calibration to inherit, only a camera one.
#
# Why it mattered more here than for liftpeg: the task is "descend until you
# cannot, then close", and HOME_INSERT_JITTER spread the sim descent over
# 4.8-12.8 mm. The real 48.3 mm was not off-centre, it was outside the
# training distribution, so a policy would close ~39 mm early in free space.
#
# Solved in the assembled env, not derived: the shaft is vertical to 0.004 deg
# so dz/d(insertion) measures -1.0000 m/m exactly, and 0.271034 total puts the
# tip at 53.90 mm = 49.1 mm above the sheet's top face -- the mean descent over
# all 30 Sep2 episodes (sd 1.8 mm), not one episode's 48.3. Tip xy moves 0.03 mm,
# so HOME_TIP_XY and everything placed from it still hold.
TISSUE_HOME_INSERTION = float(
  os.environ.get("TISSUE_HOME_INSERTION", "0.059971")   # + lower_tool's 0.211063
)
# 11 mm strip, 5 mm of it OVERHANGING the sheet and ~6 mm lying over it.
# Fixed by two bench facts: the tissue is 50 mm but 55 mm across INCLUDING the
# tape, so the tape adds exactly 5 mm to the footprint; and the green covers
# 32.3% of the sheet in data/Sep2 against sim's 23.0%. Solving
# (2*50*W - W^2) / 55^2 = 0.323 gives W = 11.0 mm; rendering it measured 27.7%
# against the rig's 32.3%, so the strip was widened INWARD to 12.4 mm -- the
# overhang is what fixes the footprint, so widening does not disturb the 55 mm.
#
# Sim previously laid an 8 mm strip ENTIRELY on the sheet -- no overhang at all,
# so the footprint stayed 50 mm and the tape covered too little.
TAPE_HALF_W = float(os.environ.get("TISSUE_TAPE_HALF_W", "0.005"))  # 10 mm, measured
# 72/67.5 mm, not the bench's 80/75. With the strips anchored at the corner they
# meet at -- which is what the rig does, one laid over the other -- the measured
# lengths overshoot: green spans 0.821 of frame against the rig's 0.720. 72 mm
# lands on 0.702. The measured lengths presumably include tape that is turned
# down onto the table edge and not visible from above.
TAPE_LEN_X = float(os.environ.get("TISSUE_TAPE_LEN_X", "0.072"))    # 80 mm, measured
TAPE_LEN_Y = float(os.environ.get("TISSUE_TAPE_LEN_Y", "0.0675"))    # 75 mm, measured
TAPE_OVERHANG = float(os.environ.get("TISSUE_TAPE_OVERHANG", "0.007"))  # past the sheet
TAPE_HALF_T = 0.0002                      # 0.4 mm thick strip
# DERIVED, not hardcoded. MuJoCo draws a flex sheet inflated by its vertex
# radius, so the tissue's visible top surface is TISSUE_Z + TISSUE_RADIUS --
# not TISSUE_Z. A fixed 0.4 mm offset was written when the radius was 3 mm and
# buried the tape inside the sheet once the radius shrank to 0.8 mm, which is
# the pink showing through the green. Sit the strip clear of that surface.
TAPE_Z = TISSUE_Z + TISSUE_RADIUS + TAPE_HALF_T + 0.00015


def pinned_ids() -> tuple[int, ...]:
  """Every vertex on the two taped edges."""
  ids = {corner_index(TISSUE_COUNT, TAPED_EDGE_IX, iy) for iy in range(TISSUE_COUNT)}
  ids |= {corner_index(TISSUE_COUNT, ix, TAPED_EDGE_IY) for ix in range(TISSUE_COUNT)}
  return tuple(sorted(ids))


def free_corner_id() -> int:
  return corner_index(TISSUE_COUNT, *FREE_CORNER)


def vertex_xy(ix: int, iy: int) -> tuple[float, float]:
  """Env-local xy of grid cell (ix, iy). Verified against compiled extents."""
  h = (TISSUE_COUNT - 1) / 2.0
  return (TISSUE_CX + (ix - h) * TISSUE_SPACING_X,
          TISSUE_CY + (iy - h) * TISSUE_SPACING_Y)


# The goal square hides under the flap: offset diagonally inboard from the free
# corner so that revealing ALL of it needs a real peel, not a twitch.
_fx, _fy = vertex_xy(*FREE_CORNER)
# 12 mm, not liftpeg's 20.6 mm. Measured: a 20.6 mm square under a 50 mm sheet
# cannot be fully uncovered by folding one corner -- the best any grip achieves
# is 0.60, because the two taped edges hold the material over the square in
# place. 12 mm with a 3x3 grip reaches 1.00. Sheet size and reach are unchanged
# in exchange; see the sizing table in the commit that introduced this.
# 20 mm, measured. An earlier 12 mm was fitted so the reveal metric behaved
# under a 50 mm sheet with a 3x3 grip; the rig's square is 20 x 20 and the
# metric should follow the rig, not the other way round.
GOAL_SIDE = float(os.environ.get("TISSUE_GOAL_SIDE", "0.020"))
# 15 mm inboard, not 12: the square is 20.6 mm across and jitters +-3 mm, so
# at 12 mm its corner reached x=-26.3 against a sheet edge at -25 and the red
# poked out from under the tissue at reset -- the episode began already solved.
# Required: |offset| >= half_square + jitter = 10.3 + 3 = 13.3 mm.
# half the square + 4 mm, so a +-3 mm jitter still leaves it fully covered at
# reset. Any larger and the fold cannot reach it; any smaller and the episode
# starts already solved.
# half the square + jitter + a real margin. The margin is not cosmetic: the
# camera looks along a 45-degree line, so a goal corner sitting only ~1 mm
# inside the sheet edge is visible PAST the edge even though it is
# geometrically underneath. Measured at offset 10 mm / jitter 3 mm: 20% of the
# square already visible at reset. Jitter is halved here for the same reason.
GOAL_JITTER = 0.0015
_GOAL_OFF = GOAL_SIDE / 2.0 + GOAL_JITTER + 0.004
_GOAL_CENTER_UNROT = (_fx + _GOAL_OFF, _fy - _GOAL_OFF)


def _yaw_about_free_corner(x: float, y: float) -> tuple[float, float]:
  """Rotate a world xy by TISSUE_YAW_DEG about the FREE CORNER.

  The free corner is the pivot because it is the one point the task is built
  around: at yaw 0 it sits exactly under the home tool tip, and the whole
  reachability argument for shrinking the sheet depends on that. Rotating about
  the sheet's CENTRE instead would slide the grasp corner out from under the
  tool and quietly break the 13 mm approach the camera framing was fitted to.
  """
  t = math.radians(TISSUE_YAW_DEG)
  c, sn = math.cos(t), math.sin(t)
  dx, dy = x - _fx, y - _fy
  return (_fx + c * dx - sn * dy, _fy + sn * dx + c * dy)


# The tape geoms and the flexcomp all carry ABSOLUTE world coordinates and all
# sit inside `tissue_root`, so the sheet is rotated by giving that body a yaw
# and the offset that turns "rotate about the world origin" into "rotate about
# the free corner": pos = P - R.P, which maps a child at c to P + R.(c - P).
# Nothing inside the body has to change.
_ROOT_POS = (
  _fx - (math.cos(math.radians(TISSUE_YAW_DEG)) * _fx
         - math.sin(math.radians(TISSUE_YAW_DEG)) * _fy),
  _fy - (math.sin(math.radians(TISSUE_YAW_DEG)) * _fx
         + math.cos(math.radians(TISSUE_YAW_DEG)) * _fy),
)
# The goal square is a SEPARATE entity, placed by mocap in world coordinates,
# so it does not ride the body rotation and has to be rotated by hand.
GOAL_CENTER = _yaw_about_free_corner(*_GOAL_CENTER_UNROT)
# The patch of vertices the jaw takes with it.
GRASP_IDS = gr.patch_ids(TISSUE_COUNT, FREE_CORNER)


def get_tissue_goal_spec() -> mujoco.MjSpec:
  """The red goal square, at OUR size rather than liftpeg's.

  Visual only: nothing may rest on it, and success is a line-of-sight test, so
  it never needs to generate a contact.
  """
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal", mocap=True)
  # Yawed with the sheet. place_tissue_cam writes this body's pose with an
  # IDENTITY quaternion, so without a rotation on the geom itself the square
  # stayed axis-aligned while the tissue above it was turned -45 deg -- it read
  # as an upright square under a diamond, where the rig shows both as diamonds.
  _t = math.radians(TISSUE_YAW_DEG) / 2.0
  g = body.add_geom(
    name="goal_square", type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.5 * GOAL_SIDE, 0.5 * GOAL_SIDE, 0.0002),
    pos=(0.0, 0.0, 0.0002), quat=(math.cos(_t), 0.0, 0.0, math.sin(_t)),
    contype=0, conaffinity=0, group=2,
  )
  g.rgba = LS.GOAL_SQUARE_RGB + (1.0,)
  return spec


def get_tissue_spec() -> mujoco.MjSpec:
  pins = "".join(f'<pin id="{i}"/>' for i in pinned_ids())
  # TISSUE_YOUNG=None selects MuJoCo's inextensible-edge constraint instead of
  # membrane elasticity. The two are mutually exclusive ("flex constraints and
  # elasticity cannot both be present").
  _elastic = (
    '<edge equality="true"/>' if TISSUE_YOUNG is None else
    f'<elasticity young="{TISSUE_YOUNG}" poisson="0" '
    f'thickness="{TISSUE_THICKNESS}" damping="1e-3"/>')
  # Green tape, visual only (contype 0): it marks where the sheet is held down
  # so the policy can see the constraint, exactly as the real rig will look.
  _ex, _ = vertex_xy(TAPED_EDGE_IX, 0)
  _, _ey = vertex_xy(0, TAPED_EDGE_IY)
  # Strips sit fully INBOARD of the sheet edge and stop at the corners. Real
  # tape laps onto the table, but a single flat strip cannot do that without
  # floating 4 mm above the floor beyond the edge, which reads far worse at
  # 84 px than tape that simply ends where the tissue does.
  # Real tape lengths, from the bench drawing: the strips are LONGER than the
  # sheet's sides -- 80 and 75 mm against 50 and 55 -- because they lap onto the
  # table to hold the tissue down. Sim ran them at exactly the sheet's side, so
  # they stopped at the corners where the real ones carry on past.
  _span_x = TAPE_LEN_X / 2.0
  _span_y = TAPE_LEN_Y / 2.0
  # Offset so the strip OVERHANGS the edge by TAPE_OVERHANG: its centre sits
  # (half-width - overhang) inside the sheet, which is 0.5 mm at these numbers.
  # The old form used the full half-width, putting the tape wholly on the sheet.
  _inset = TAPE_HALF_W - TAPE_OVERHANG
  _ex = _ex - _inset * (1 if TAPED_EDGE_IX == _LAST else -1)
  _ey = _ey + _inset * (1 if TAPED_EDGE_IY == 0 else -1)
  # Both strips START at the corner where they meet and run outward from it, as
  # they do on the rig: the two lengths are measured from that corner. Centring
  # them on the sheet instead -- which is what this did -- pushed the excess out
  # at BOTH ends, so each strip overshot the shared corner by half its overhang
  # and the join read as a cross rather than a right angle.
  _corner_x, _corner_y = vertex_xy(TAPED_EDGE_IX, TAPED_EDGE_IY)
  _tape_x_cy = _corner_y + _span_y - TAPE_HALF_W
  _tape_y_cx = _corner_x - _span_x + TAPE_HALF_W
  _tape = (
    f'<geom name="tape_x" type="box" contype="0" conaffinity="0" group="2" '
    f'size="{TAPE_HALF_W} {_span_y} {TAPE_HALF_T}" '
    f'pos="{_ex} {_tape_x_cy} {TAPE_Z}" '
    f'rgba="{TAPE_RGB[0]} {TAPE_RGB[1]} {TAPE_RGB[2]} 1"/>'
    f'<geom name="tape_y" type="box" contype="0" conaffinity="0" group="2" '
    f'size="{_span_x} {TAPE_HALF_W} {TAPE_HALF_T}" '
    f'pos="{_tape_y_cx} {_ey} {TAPE_Z}" '
    f'rgba="{TAPE_RGB[0]} {TAPE_RGB[1]} {TAPE_RGB[2]} 1"/>'
  )
  # flexcomp's `pos` IS the grid centre. The original code added +half on each
  # axis, which put the sheet 25 mm off the spawn point in both x and y.
  # Verified by reading the compiled vertex extents back, not by inspection.
  half = TISSUE_SIDE_X / 2.0
  spec = mujoco.MjSpec.from_string(f"""
  <mujoco>
    <worldbody>
      <body name="tissue_root" pos="{_ROOT_POS[0]} {_ROOT_POS[1]} 0" euler="0 0 {TISSUE_YAW_DEG}">
        {_tape}
        <flexcomp type="grid" name="tissue" dim="2"
                  count="{TISSUE_COUNT} {TISSUE_COUNT} 1"
                  spacing="{TISSUE_SPACING_X} {TISSUE_SPACING_Y} {TISSUE_SPACING_X}"
                  pos="{TISSUE_CX} {TISSUE_CY} {TISSUE_Z}"
                  radius="{TISSUE_RADIUS}" mass="{TISSUE_MASS}"
                  rgba="{TISSUE_RGB[0]} {TISSUE_RGB[1]} {TISSUE_RGB[2]} 1">
          {pins}
          <!-- No <edge equality>: MuJoCo rejects "flex constraints and
               elasticity (young) cannot both be present". Elasticity is the
               better half to keep -- it gives a real stretch response rather
               than hard distance constraints. -->
          <contact selfcollide="none" condim="3" contype="0" conaffinity="0"/>
          <!-- No elastic2d="bend": MuJoCo rejects pins combined with bending
               ("pins are not supported for bending"). Pinning the three
               corners is the task definition, so bending stiffness is what
               gives way; the sheet is a membrane, floppier than the ALOHA
               towel but still a surface. -->
          {_elastic}
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  """)
  # flexcomp emits anonymous joints; mjlab resolves entity joints by name.
  #
  # The z slide joint of every free vertex also gets a LOWER LIMIT, and that
  # limit is the floor. This is not a stylistic choice:
  #
  #   mujoco_warp does not replicate flex-vs-geom collision correctly across
  #   worlds. With a sheet resting on a box, contiguous BLOCKS of worlds get no
  #   flex contacts at all and their sheets free-fall forever -- 8/16, 31/64,
  #   always starting at world 3, identical indices every run, unaffected by
  #   nconmax/njmax, timestep, floor thickness, spawn gap, elasticity or DR.
  #   Reproduced in ~40 lines of pure mujoco_warp with no mjlab, no robot and
  #   no randomisation (scratch: minrepro.py). One world is always fine, which
  #   is why every earlier single-env check passed.
  #
  # A joint limit goes through the ordinary constraint solver instead, which
  # IS replicated correctly: 0/256 worlds fall, at any radius. For a flat sheet
  # on a flat floor the two are physically equivalent.
  for i, joint in enumerate(spec.joints):
    if not joint.name:
      joint.name = f"tissue_flex_joint_{i}"
    if abs(joint.axis[2]) > 0.5:
      joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
      joint.range = [TISSUE_FLOOR_Z - TISSUE_Z, 1.0]
  return spec


def _anchor_pos(env) -> torch.Tensor:
  """The grasp corner, env-local."""
  return flex_vertex_pos_w(env, GRASP_IDS[0]) - env.scene.env_origins


# The tip must be at least this high at the moment of success. Measured: a peel
# at 30 degrees elevation reveals the square while still holding, with the tip
# at 13.3 mm (p5 13.30), so 10 mm leaves real margin. 45 and 60 degrees give
# 17.7 and 21.0 mm.
SUCCESS_MIN_TIP_Z = float(os.environ.get("TISSUE_SUCCESS_MIN_TIP_Z", "0.010"))


def tissue_revealed(env) -> torch.Tensor:
  """Success: the whole goal square visible, WHILE holding the tissue, WITH the
  tool lifted clear.

  Visibility alone was not enough, and the failure was instructive: a state
  teacher hit 100% by dragging the corner aside without ever completing a
  grasp. Once the flap has been moved the tissue does not spring back -- it is
  inextensible and pinned, so it stays folded -- and the square therefore stays
  revealed long after any grasp ended. The scripted controller was exploiting
  the same thing without anyone noticing: it held the tissue at only 13.2% of
  its own successes.

  So the criterion now also requires the grasp to be live and the tool to be
  up, which forces the behaviour we actually want to transfer: grasp, lift,
  retract. This is a deliberate narrowing rather than a bug fix -- "the square
  is visible" really was satisfied.

  It costs nothing in feasibility: revealing the square WHILE still holding it
  is achievable on 100% of grasps at every peel elevation tried.
  """
  return (vis.goal_fully_visible(env, GOAL_SIDE)
          & gr.is_held(env)
          & (_tip_z(env) >= SUCCESS_MIN_TIP_Z))


def revealed_only_metric(env) -> torch.Tensor:
  """Visibility WITHOUT the grasp/height requirement.

  Logged alongside success so the gap between them is visible: if this climbs
  while success_rate does not, the policy has rediscovered the drag-and-let-go
  shortcut and is being correctly refused.
  """
  return vis.goal_fully_visible(env, GOAL_SIDE).float()


def tip_z_metric(env) -> torch.Tensor:
  return _tip_z(env) * 1000.0


def success_metric(env) -> torch.Tensor:
  return tissue_revealed(env).float()


def visible_metric(env) -> torch.Tensor:
  return vis.visible_fraction(env, GOAL_SIDE)


def grasp_metric(env) -> torch.Tensor:
  return gr.is_held(env).float()


def attempts_metric(env) -> torch.Tensor:
  return gr.grasp_attempts(env)


def success_bonus(env) -> torch.Tensor:
  return tissue_revealed(env).float() * 2.0


def retract_shaping(env) -> torch.Tensor:
  """PLACEHOLDER shaping: reach the corner, hold it, uncover the square.

  Deliberately generic. The real reward for this task is meant to be fitted to
  recorded demonstrations -- the whole design is "shape onto the trajectory a
  human drove" rather than invent one -- so this exists only so the env is
  trainable and watchable before those demos exist. Do not tune it; replace it.
  """
  tool = a3b._tool_pos(env)
  anchor = _anchor_pos(env)
  reach_d = torch.norm(anchor - tool, dim=-1)
  reach = torch.exp(-reach_d / 0.02)
  held = gr.is_held(env).float()
  seen = vis.visible_fraction(env, GOAL_SIDE)
  return 0.5 * reach + 1.0 * held + 3.0 * seen


def capture_home(env, env_ids: torch.Tensor | None = None) -> None:
  """Record the tool pose at episode start, AFTER the home jitter.

  The proprioceptive observation is displacement FROM here, not absolute pose.
  That is the form the real robot can actually supply -- it always knows how far
  it has moved from where it was homed -- and it is invariant to the sim/real
  offset in absolute tool position, which is the thing we cannot reproduce.
  """
  st = _home_store(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  # Only FLAG here. Entity `.data` is stale inside reset events, so reading the
  # pose now records the PREVIOUS episode's -- measured dz of -313 mm after 30
  # idle steps, against a tool that had not moved at all. The actual capture
  # happens on the first observation after the reset, where the data is live.
  st["pending"][env_ids] = True


def _home_store(env):
  st = getattr(env, "_tissue_home", None)
  if st is None:
    st = {"tip": torch.zeros(env.num_envs, 3, device=env.device),
          "jaw": torch.zeros(env.num_envs, device=env.device),
          "pending": torch.ones(env.num_envs, dtype=torch.bool,
                                device=env.device)}
    env._tissue_home = st
  return st


def _jaw_angle(env) -> torch.Tensor:
  robot = env.scene["robot"]
  return robot.data.joint_pos[:, robot.joint_names.index("jaw")]


def _target_lead(env) -> torch.Tensor:
  """|commanded target - achieved tool position|, metres.

  The sim twin of the dVRK's PID tracking error. The Classic PSM has no force
  sensor -- it faults when the commanded setpoint diverges from the measured
  joint past tolerance, so this, not contact force, is the quantity that
  predicts an auto-stop on hardware.
  """
  term = env.action_manager.get_term("cartesian_delta")
  p_target = term._ee_target_tf[..., :3, 3]
  q_arm = term._entity.data.joint_pos[:, term._joint_ids]
  p_actual = term._fk_fn(q_arm)[..., :3, 3]
  return torch.norm(p_target - p_actual, dim=-1)


def m_target_lead_mm(env) -> torch.Tensor:
  return _target_lead(env) * 1000.0


def target_lead_fault(env) -> torch.Tensor:
  """Terminate: the arm is commanding into something it cannot move through.

  20 mm by default here against liftpeg's 40 mm -- tissue is a thin sheet with
  nothing to push into, so any sustained lead is already wrong.
  """
  if TISSUE_TARGET_LEAD_FAULT <= 0.0:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return _target_lead(env) > TISSUE_TARGET_LEAD_FAULT


def home_delta(env) -> torch.Tensor:
  """(n, 4): how far the tool has moved from its homed pose -- dx, dy, dz, djaw.

  Replaces the 3-D constant-zeros placeholder the actor used to carry. That
  placeholder existed only because rsl-rl needs a non-empty 1-D branch and the
  deployed ONNX node could not assemble a proprio vector; displacement from home
  is a vector it CAN assemble, so the actor gets real information instead of
  padding.
  """
  st = _home_store(env)
  tip = a3b._tool_pos(env) - env.scene.env_origins
  jaw = _jaw_angle(env)
  if bool(st["pending"].any()):
    m = st["pending"]
    st["tip"][m] = tip[m]
    st["jaw"][m] = jaw[m]
    st["pending"] = torch.zeros_like(st["pending"])
  return torch.cat([tip - st["tip"], (jaw - st["jaw"]).unsqueeze(-1)], dim=-1)


def _tissue_state(env) -> torch.Tensor:
  """Privileged state for the TEACHER's actor, not just the critic.

  Richer than _critic_state because an actor has to act on it: it carries the
  grasp flag, how far the tissue has been dragged, how much of the square is
  showing and how high the tip is -- i.e. everything the staged potential is
  computed from. If the teacher cannot learn the task from exactly the
  quantities the reward is built out of, the reward is wrong, and that is the
  whole point of running it first.
  """
  from surgical_lab.tasks.retract.mdp import rewards_tissue as _rt
  tool = a3b._tool_pos(env)
  anchor = _anchor_pos(env)
  goal = env.scene["goal"].data.root_link_pos_w - env.scene.env_origins
  q = _rt._quantities(env)
  return torch.cat(
    [tool, anchor, anchor - tool, goal[:, :2],
     q["held"].float().unsqueeze(-1),
     q["seen"].unsqueeze(-1),
     q["pull_frac"].unsqueeze(-1),
     q["descend_frac"].unsqueeze(-1),
     _tip_z(env).unsqueeze(-1),
     home_delta(env)],
    dim=-1,
  )


def _phase_onehot(env) -> torch.Tensor:
  """(n, 5) one-hot of the scripted controller's current phase. Diagnostic."""
  k = rs._state(env).k
  return torch.nn.functional.one_hot(k, num_classes=5).float()


def _critic_state(env) -> torch.Tensor:
  tool = a3b._tool_pos(env)
  anchor = _anchor_pos(env)
  goal = env.scene["goal"].data.root_link_pos_w - env.scene.env_origins
  return torch.cat(
    [tool, anchor, anchor - tool, goal[:, :2],
     gr.is_held(env).float().unsqueeze(-1),
     vis.visible_fraction(env, GOAL_SIDE).unsqueeze(-1)],
    dim=-1,
  )


# How many redraws before giving up and using the un-jittered nominal pose.
CAM_TRIES = int(os.environ.get("TISSUE_CAM_TRIES", "24"))
# --- side camera, fitted to the rig -----------------------------------------
# Both of these are tissue-local because LS.CAM_FOVY and LS.CAM_PITCH_DEG are
# read at import and shared with liftpeg.
#
# FOVY. LS ships 55, reasoned from the OAK-1's native VFOV. The rig disagrees:
# with the sheet measured on the bench at 50 mm (55 mm including its tape) and
# imaging corner-on, its 77.8 mm diagonal spans 0.717 of the real frame, so the
# real camera frames 108.5 mm. At the nominal 155.6 mm standoff that is
# **38.4 deg**, not 55. Independently, replay_real_episode.py's own --liftcam-fov
# help already said "the real OAK crop is far narrower (~40 deg from its
# calibrated fx)" -- so liftpeg carried two mutually inconsistent numbers and
# the measurement agrees with the calibrated-fx one.
#
# Sheet width in frame: 0.494 at fovy 55, **0.740 at 38.4**, against the rig's
# 0.731.
#
# DISTANCE is a separate axis from field of view and the sheet alone cannot see
# it: close+wide and far+narrow frame the same 108.5 mm. What separates them is
# PERSPECTIVE on something at a different depth -- the printed bracket, ~60 mm
# up the shaft. Same physical part in both domains, so its size relative to the
# sheet is a pure distance measurement:
#
#     bracket width / sheet width     REAL 0.296     SIM at 155.6 mm 0.557
#
# so the rig's camera stands further off than liftpeg's nominal. Moving along
# the SAME ray keeps the aim, so only the standoff and the fovy move together.
# MEASURED AT THE BENCH, and it supersedes liftpeg's nominal 155.6 mm.
# Three numbers taken together, which close on each other:
#     lens height above the tissue plane   150 mm (+-2)
#     horizontal to the NEAR tissue edge    90 mm
#     horizontal to the tissue CENTRE      135 mm
# -> diagonal to the near edge 174.9 mm against a measured ~170, and to the
#    centre 201.8 mm.
#
# An earlier bench figure of "156 mm to the tissue centre" is superseded: it is
# very close to the 174.9 mm diagonal to the NEAR EDGE, which is the more
# natural thing to put a tape measure on, and the three-number set is
# self-consistent where a single diagonal cannot be checked.
TISSUE_CAM_DIST = float(os.environ.get("TISSUE_CAM_DIST", "0.2018"))
# The frame width the rig shows, held fixed as the distance changes.
# 0.1009, not the 0.1085 the diagonal predicts. The predicted value left the sim
# sheet at 0.667 of frame against the rig's 0.717 -- measured at FRAME 0, where
# the tool is still high and occludes almost nothing, and stable to +-0.007
# across a 155.6 / 250 / 350 mm distance sweep, so it is the framing and not
# noise. Closed empirically rather than explained; the residual is most likely
# the sheet's yaw not being exactly 45 deg, so its horizontal extent is a little
# under the full diagonal.
#
# Re-fitted 0.1091 -> 0.1414 when the tape took its measured 80/75 mm lengths.
# The strips now run well past the sheet, as they do on the rig, so the visible
# footprint is set by the TAPE and not by the tissue -- 0.942 of frame at the
# old width against the rig's 0.727.
#
# Earlier note, still true of its own change:
# Re-fitted 0.1009 -> 0.1091 when the tape gained its 5 mm overhang: that grew
# the sheet+tape footprint from 50 to 55 mm, so the same frame had to widen by
# the same 8% to keep the measured width on the rig's 0.717.
TISSUE_FRAME_M = float(os.environ.get("TISSUE_FRAME_M", "0.1414"))
TISSUE_CAM_FOVY = float(os.environ.get(
  "TISSUE_CAM_FOVY",
  str(2.0 * math.degrees(math.atan(TISSUE_FRAME_M / (2.0 * TISSUE_CAM_DIST)))),
))
# PITCH, from the sheet's ASPECT. A flat sheet's height-to-width in frame is set
# by how obliquely the camera views the floor, and distance barely touches it --
# so aspect isolates the angle from the standoff, which the sheet's width alone
# cannot do.
#
#     SIM pitch 42.5   aspect 0.709
#     SIM pitch 46.0          0.753
#     SIM pitch 50.5          0.798
#
# FIT AGAINST THE WHOLE FLEET, not one episode. Over all 30 Sep2 episodes at
# frame 0 the real aspect is median 0.823, IQR 0.806-0.833 -- tight. Sweeping
# after the tape change gives 52.7 -> 0.786, 56.0 -> 0.810, 59.0 -> 0.836, so
# 0.823 lands at 57.4 deg.
#
# An earlier value of 42.5 here was fitted to e0 alone, whose aspect is 0.700 --
# the 13th percentile of the fleet and one of its outliers. The fleet's own
# spread (sd 0.077 against an IQR of 0.027) is carried by a handful of episodes
# where the sheet is not lying flat at reset, and e0 is one of them. Sheet WIDTH
# and centroid are stable across all 30 (sd 0.012 and 0.006), so the camera does
# not move; only the tissue's drape varies.
# From the bench, not from the footage: atan(150 / 135) = 48.0 deg. The earlier
# 57.4 was fitted to the sheet's aspect over 30 episodes and is superseded --
# aspect is a weak lever on pitch, and the tape measure is not.
TISSUE_CAM_PITCH_DEG = float(os.environ.get("TISSUE_CAM_PITCH", "48.0"))
# Camera position: the aim point plus the standoff, back along the same ray, so
# the pitch above still describes the direction whatever the distance.
# NB the NOMINAL centre, deliberately excluding TISSUE_XY_OFFSET. The camera is
# bolted to the bench and the sheet is placed by hand, so the aim must not
# follow the sheet -- with it coupled, offsetting the tissue slid the camera
# the same way and the sheet stayed put in frame, which is exactly the motion
# we are trying to see.
_CAM_AIM = _yaw_about_free_corner(
  TISSUE_CX - TISSUE_XY_OFFSET[0], TISSUE_CY - TISSUE_XY_OFFSET[1])
_cp = math.cos(math.radians(TISSUE_CAM_PITCH_DEG))
_sp = math.sin(math.radians(TISSUE_CAM_PITCH_DEG))
TISSUE_CAM_POS = (
  _CAM_AIM[0],
  _CAM_AIM[1] + TISSUE_CAM_DIST * _cp,
  TISSUE_CAM_DIST * _sp,
)


def get_tissue_cam_spec() -> mujoco.MjSpec:
  """liftpeg's camera entity at THIS task's field of view.

  NOTE the fovy has to be baked into the spec. Writing `sim.model.cam_fovy`
  after the env is built does nothing: the camera sensor derives its ray
  directions from fovy when it is constructed (`_RAY_FIELDS` in mjlab's
  sensor_context) and never revisits them, so the override prints a confident
  message and changes no pixels.
  """
  spec = LS.get_cam_spec()
  for cam in spec.cameras:
    cam.fovy = TISSUE_CAM_FOVY
  return spec


# --- wrist camera -----------------------------------------------------------
# The real wrist unit is a fisheye, so a raw pinhole render is the wrong
# picture: render WIDE and resample through the fitted warp.
#
# EVERY constant and the grid itself are IMPORTED from liftpeg, not restated.
# That module's own docstring records two separate occasions where a copy of
# this calibration drifted from the original (the fit's -5.317 deg roll was
# silently dropped from both the RL observation and the dump script, because
# each held its own copy of the parameters). One definition, three callers:
# liftpeg's observation, this observation, and the MAE dump scripts.
TISSUE_CAM_RES = 84
_WRIST_GRID: dict[str, tuple] = {}


def _wrist_grid(env):
  """liftpeg's fisheye grid at THIS task's output resolution, cached per device.

  Pinned to TISSUE_CAM_RES rather than liftpeg's CAM_RES: that one follows
  LIFTPEG_CAM_RES, and a tissue run that happened to inherit a liftpeg env var
  would silently build the grid for a different output size than the sensor
  the observation is fed into.
  """
  key = str(env.device)
  if key not in _WRIST_GRID:
    _WRIST_GRID[key] = lp.build_wrist_grid(
      env.device, (TISSUE_CAM_RES, TISSUE_CAM_RES))
  return _WRIST_GRID[key]


def _camera_wrist(env) -> torch.Tensor:
  """Wrist view, fisheye-warped, as (B, 3, 84, 84).

  Normalisation is liftpeg's `_stretch` -- a per-frame whole-image min-max,
  mirroring the deployment's `cv::normalize(..., NORM_MINMAX)`. NOT /255: the
  deployed image_processor applies the stretch, so training on a fixed scale
  would be a silent input-distribution shift.
  """
  from mjlab.tasks.manipulation import mdp as manipulation_mdp

  from surgical_lab.utils import fisheye as fe

  grid, valid = _wrist_grid(env)
  raw = manipulation_mdp.camera_rgb(env, "wristcam_cam")  # (B, 3, SRC, SRC)
  return lp._stretch(fe.apply(raw, grid, valid))


def _camera_sbs(env) -> torch.Tensor:
  """Front | wrist concatenated on WIDTH -> (B, 3, 84, 168).

  Front on the LEFT, matching scripts/make_sidebyside_dataset.py: the
  positional embedding is the only thing telling the encoder which half is
  which, so the order is part of the calibration.

  Each half is stretched on its OWN min/max rather than jointly -- the rig is
  two independent sensors each running its own cv::normalize, and a joint
  stretch would couple their exposures in a way the hardware never does.
  """
  return torch.cat(
    [a3b._camera_rgb_normalized(env, "liftcam_cam"), _camera_wrist(env)], dim=-1
  )


def _bake_wrist_fovy(spec) -> None:
  """Write the fisheye SOURCE fov into the spec's wrist camera.

  THIS IS NOT OPTIONAL and it cannot be done later. psm_lnd.xml carries a
  placeholder `fovy="90"` (its own comment says so), while the fisheye grid is
  built for a `lp.WRIST_RENDER_FOV` = 135 deg source render. Writing
  `sim.model.cam_fovy` after the env is built changes nothing: mjlab's camera
  sensor derives its ray directions from fovy at CONSTRUCTION and never
  revisits them, so the override prints a confident message and moves no
  pixels. The warp then reads a 90 deg image as if it spanned 135, magnifying
  everything by tan(67.5)/tan(45) = 2.41x -- which is why the arm, not the
  scene, filled the wrist frame. Same bake scripts/replay_real_episode.py
  does, for the same reason.
  """
  for cam in spec.cameras:
    if cam.name.endswith("wrist_camera"):
      cam.fovy = lp.WRIST_RENDER_FOV


# Stricter than liftpeg's 0.92. "Inside the image" is not the same as "usefully
# visible": a grasp point two pixels from the border is technically in frame
# and useless to a policy that has to aim at it.
CAM_FRAME_MARGIN = 0.85

# Light home-pose scatter, OmniReset style. A quarter of liftpeg's, because
# there the tool had to find a peg it could be well away from; here it starts
# directly above the grasp point and only needs enough spread that the policy
# cannot treat "straight down N mm" as a constant.
# Yaw and pitch jitter are ZERO. They are what moved the tool in x and y, and
# the task is now "descend straight down until you cannot" -- a start pose that
# is laterally displaced makes that instruction wrong rather than robust. Only
# the insertion (vertical) is randomised, which is the axis that genuinely
# cannot be reproduced from sim.
HOME_YAW_JITTER = float(os.environ.get("TISSUE_HOME_YAW_JIT", "0.0"))
HOME_PITCH_JITTER = float(os.environ.get("TISSUE_HOME_PITCH_JIT", "0.0"))
# Insertion jitter is DELIBERATELY the loose one, and it is a safety feature
# rather than variety for its own sake. It sets how far the tool must descend
# before it reaches the tissue. If that distance is fixed, the cheapest policy
# is "go down N mm" open loop -- and N does not transfer, because the real
# tool-tip-to-floor distance cannot be reproduced from the sim. Randomising it
# per episode makes the open-loop solution fail in training, so the only thing
# left to condition on is the image. This is the standard episode-wise
# table-height randomisation from the sim-to-real literature.
HOME_INSERT_JITTER = float(os.environ.get("TISSUE_HOME_INSERT_JIT", "0.004"))

# --- floor safety ---------------------------------------------------------
# The tool must never push into the floor: there are no force sensors, it can
# damage the instrument, and on the real arm it trips the autostop. The tissue
# surface sits at ~TISSUE_FLOOR_Z, so there is only a few mm between "low
# enough to grasp" and "too low".
FLOOR_HARD_Z = float(os.environ.get("TISSUE_FLOOR_HARD_Z", "0.0"))

# --- the virtual fixture --------------------------------------------------
# A hard floor on the EE target's z, in the controller's own (FK / robot base)
# frame. This is the mechanism that makes the descent safe AND trivial:
#
#   * the tool physically cannot be driven below it, whatever the policy asks
#     for, so floor contact is prevented rather than discouraged;
#   * the tip bottoms out at the SAME height regardless of where the episode
#     started -- measured spread 0.00 mm over 128 envs across the full home
#     randomisation -- so "hold down until it stops" is a correct strategy and
#     the starting height, table height and tool-to-floor distance stop
#     mattering entirely;
#   * it removes the requirement that the policy perceive its own height,
#     which it cannot do well: 0.344 px per mm at 84 px, one pixel = 2.9 mm.
#
# The number is in CONTROLLER units, not world units, which is deliberate --
# on the real arm you get it the same way: jog the tool down until it just
# grazes the floor, read off the controller's z, enter it before inference.
# Calibrated here so the tip bottoms ~2.5 mm above the floor, i.e. right at the
# tissue surface (the sheet's grasp corner rests at 2.63 mm):
#
#     min_target_z   -0.3300  -0.3275  -0.3250  -0.3200
#     tip bottoms     1.60     ~2.5     3.19     8.19   mm
TISSUE_MIN_TARGET_Z = float(os.environ.get("TISSUE_MIN_TARGET_Z", "-0.3272"))
# Camera-only actor: no proprioception, no action history. See the note where
# the actor terms are built.
TISSUE_IMAGE_ONLY = os.environ.get("TISSUE_IMAGE_ONLY", "0") == "1"
# The dVRK has NO force sensor -- it faults when the PID's commanded setpoint
# diverges from the measured joint past its tolerance. Liftpeg guards that with
# a 40 mm termination; tissue is a lighter task with a thin sheet and nothing
# to push into, so 20 mm is the default here. Metres. 0 disables.
TISSUE_TARGET_LEAD_FAULT = float(
  os.environ.get("TISSUE_TARGET_LEAD_FAULT", "0.020"))
FLOOR_WARN_Z = float(os.environ.get("TISSUE_FLOOR_WARN_Z", "0.0015"))


def _tip_z(env) -> torch.Tensor:
  return (a3b._tool_pos(env) - env.scene.env_origins)[:, 2]


def floor_violation(env) -> torch.Tensor:
  """Terminate: the tip went below the floor.

  A termination, not just a penalty. A penalty makes contact expensive and the
  policy trades it off against task reward; a termination makes it fatal, which
  is the behaviour we need when the real consequence is an autostop and a
  re-home. Same shape as the virtual-fixture violations used in surgical safe
  RL, where breaching the fixture ends or resets the episode rather than
  costing a few points.
  """
  return _tip_z(env) < FLOOR_HARD_Z


def floor_penalty(env) -> torch.Tensor:
  """Graded cost for entering the last 1.5 mm above the floor.

  The termination alone is a cliff with no gradient leading up to it, so the
  policy only learns about the floor by falling off. This slopes the last
  stretch so descending too eagerly is discouraged BEFORE it becomes fatal.
  """
  over = (FLOOR_WARN_Z - _tip_z(env)).clamp_min(0.0) / FLOOR_WARN_Z
  return -(over ** 2)


def floor_margin_metric(env) -> torch.Tensor:
  """Lowest the tip got, in mm above the floor. The number to watch."""
  return _tip_z(env) * 1000.0


def randomize_home_joints(env, env_ids: torch.Tensor | None = None) -> None:
  """Scatter the tool's start pose about the nominal home.

  Reuses liftpeg's implementation with this task's (much smaller) jitters --
  including its two hard-won details: it reads default_joint_pos rather than
  the live joint_pos (entity .data is stale inside reset events, so reading
  live jitters about wherever the LAST episode ended), and it re-adds
  LOWER_TOOL_OFFSET because default_joint_pos predates the lower_tool event.
  """
  import surgical_lab.tasks.lift.config.psm.liftpeg_a3b_env_cfg as _lp
  _y, _p, _i = _lp.HOME_YAW_JITTER, _lp.HOME_PITCH_JITTER, _lp.HOME_INSERT_JITTER
  _lp.HOME_YAW_JITTER = HOME_YAW_JITTER
  _lp.HOME_PITCH_JITTER = HOME_PITCH_JITTER
  _lp.HOME_INSERT_JITTER = HOME_INSERT_JITTER
  try:
    _lp.randomize_home_joints(env, env_ids)
  finally:
    _lp.HOME_YAW_JITTER, _lp.HOME_PITCH_JITTER, _lp.HOME_INSERT_JITTER = _y, _p, _i


def _must_see_points(env, n, gpos):
  """Everything the camera has to have in frame at reset, (n, k, 3) env-local.

  Computed ANALYTICALLY rather than read from the sim: entity `.data` is stale
  inside reset events, and the tissue has no spawn randomisation anyway, so its
  rest positions are known exactly from the grid.
  """
  dev = env.device
  # Both the spawn height AND the settled height. The sheet sags from
  # TISSUE_Z onto the (soft) floor limit within a few steps, and testing only
  # the spawn height left the patch out of frame in 1.8% of episodes once it
  # had settled -- measured 100.00% in frame at spawn positions vs 98.24% at
  # settled ones. A point set spanning both is right wherever it ends up.
  ij = [(i // TISSUE_COUNT, i % TISSUE_COUNT) for i in GRASP_IDS]
  _settled = TISSUE_FLOOR_Z - 0.0015      # allow for the soft-limit violation
  patch = torch.tensor(
    [[*vertex_xy(ix, iy), z] for ix, iy in ij for z in (TISSUE_Z, _settled)],
    device=dev, dtype=torch.float32)
  patch = patch.unsqueeze(0).expand(n, -1, -1)
  gh = 0.5 * GOAL_SIDE
  gc = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]],
                    device=dev, dtype=torch.float32) * gh
  goal_pts = torch.cat(
    [gpos[:, :2].unsqueeze(1) + gc.unsqueeze(0),
     torch.zeros((n, 4, 1), device=dev)], dim=-1)
  tip = LS._tip_box_corners(dev).unsqueeze(0).expand(n, -1, -1)
  return torch.cat([patch, goal_pts, tip], dim=1)


def _true_cam_pose(env, env_ids, body_pos, body_quat):
  """Compose the mocap body pose with the camera's PER-WORLD local offset.

  The camera is not the body. a3b's `camera_pos` / `camera_quat` DR events
  jitter the camera element's local pose per world (pos: x +-, y 0..8 mm,
  z +-2 mm; quat: pitch/yaw +-3 deg), and those live in sim.model.cam_pos /
  cam_quat with a leading world axis -- NOT in the CPU model, where they read
  as identity. Framing checks done against the body pose alone are therefore
  wrong by 8.6 mm on average and up to 18.6 mm, which is enough to put a point
  on the wrong side of the image border.
  """
  from mjlab.utils.lab_api.math import quat_apply, quat_mul
  m = env.sim.mj_model
  ci = next(i for i in range(m.ncam)
            if "liftcam/" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) or ""))
  lp = env.sim.model.cam_pos[env_ids, ci]
  lq = env.sim.model.cam_quat[env_ids, ci]
  return body_pos + quat_apply(body_quat, lp), quat_mul(body_quat, lq)


def place_tissue_cam(env, env_ids: torch.Tensor | None) -> None:
  """Place the camera with liftpeg's own DR, then REJECT poses that lose sight
  of the grasp point.

  camera_pose_batch is the pre-hemisphere sampler: nominal 45-degree pose plus
  position/pitch/yaw/roll/radial jitter. The hemisphere sampler is deliberately
  NOT used here -- this task is meant to match the real rig one-for-one.

  The rejection is a safety constraint, not a nicety. Measured over 1024 resets
  with the raw sampler, the grasp corner was OUTSIDE the frame in 20% of
  episodes (and the whole sheet in 27%), reaching 1.22 in normalised frame
  coordinates where 1.0 is the edge. A policy that cannot see the corner has to
  guess where it is, and a guess about where to close the jaw near the floor is
  exactly the failure we cannot afford on the real arm. Same reasoning as the
  tool-tip visibility constraint on liftpeg.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  origins = env.scene.env_origins[env_ids]
  ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(n, 4)

  # Goal square under the flap, drawn BEFORE the camera because the visibility
  # test needs to know where it is. Jittered so the policy cannot memorise one
  # pixel location, but only a little: it has to stay fully covered by the
  # tissue at reset, or the episode starts already solved.
  gj = (torch.rand((n, 2), device=env.device) * 2.0 - 1.0) * GOAL_JITTER
  gpos = torch.cat([
    torch.tensor(GOAL_CENTER, device=env.device).unsqueeze(0).expand(n, 2) + gj,
    torch.full((n, 1), 0.0, device=env.device)], dim=-1)
  env.scene["goal"].write_mocap_pose_to_sim(
    torch.cat([origins + gpos, ident], dim=-1), env_ids=env_ids
  )

  pts = _must_see_points(env, n, gpos)
  fovy = torch.full((n,), TISSUE_CAM_FOVY, device=env.device)
  def _ok(c, q, pt, fv, ids):
    tp, tq = _true_cam_pose(env, ids, c, q)
    return LS.points_in_frame(tp, tq, pt, fv, margin=CAM_FRAME_MARGIN)

  cam, quat = LS.camera_pose_batch(
    n, env.device, base_pos=TISSUE_CAM_POS, base_pitch_deg=TISSUE_CAM_PITCH_DEG)
  ok = _ok(cam, quat, pts, fovy, env_ids)
  for _ in range(CAM_TRIES):
    if bool(ok.all()):
      break
    bad = (~ok).nonzero(as_tuple=True)[0]
    c2, q2 = LS.camera_pose_batch(
      len(bad), env.device, base_pos=TISSUE_CAM_POS, base_pitch_deg=TISSUE_CAM_PITCH_DEG)
    cam[bad], quat[bad] = c2, q2
    ok[bad] = _ok(cam[bad], quat[bad], pts[bad], fovy[bad], env_ids[bad])
  if not bool(ok.all()):
    # Fall back to the un-jittered nominal rig, which frames everything by
    # construction. Better a slightly less varied episode than a blind one.
    bad = (~ok).nonzero(as_tuple=True)[0]
    c0, q0 = LS.camera_pose_batch(
      len(bad), env.device, base_pos=TISSUE_CAM_POS, base_pitch_deg=TISSUE_CAM_PITCH_DEG, jitter=0.0)
    cam[bad], quat[bad] = c0, q0
  env._tissue_cam_reject = int((~ok).sum())

  env.scene["liftcam"].write_mocap_pose_to_sim(
    torch.cat([origins + cam, quat], dim=-1), env_ids=env_ids
  )

  key = torch.tensor(LS.LIGHT_POS, device=env.device).unsqueeze(0).expand(n, 3)
  env.scene["keylight"].write_mocap_pose_to_sim(
    torch.cat([origins + key, ident], dim=-1), env_ids=env_ids
  )


# --- force-based grasp ------------------------------------------------------
# "weld" is the historical kinematic path: update_grasp writes the held patch's
# qpos every control step. A position write transmits NO FORCE, so the rest of
# the sheet is never loaded and never moves -- measured at exactly 11.1% of an
# 81-vertex grid lifted, i.e. the 3x3 patch and nothing else, identically under
# both constitutive models.
#
# "connect" instead toggles a real equality constraint between the jaw and the
# free corner's vertex body. The solver enforces it with a force, which
# propagates through the mesh, and holds the corner AT the jaw rather than
# 5-10 mm from it.
GRASP_MODE = os.environ.get("TISSUE_GRASP_MODE", "weld")
# Anchor on body1, in tool_gripper1_link's own frame.
#
# NOT psm_lnd.xml's tool_tip site pos of (0, -0.0102, 0) -- that site hangs off
# a different body, and using it put the anchor 3.72 mm from the tool position
# every other part of this task uses (a3b._tool_pos). Solved in the assembled
# model instead: express _tool_pos in body1's frame and read it off. The small
# x term is because _tool_pos is the midpoint of the two jaw bodies while the
# constraint attaches to gripper1 alone.
GRASP_EQ_ANCHOR = (0.00032, -0.00649, 0.0)
GRASP_EQ_NAME = "tissue_grasp"
# Constraint softness. The default (0.02, 1) leaves the corner a measured
# 5.74 mm from the jaw while held -- steady-state violation, the solver trading
# the constraint off against the membrane's pull. 0.010 halves that to 2.65 mm
# and 0.005 is identical, because a solref timeconst cannot usefully go below
# ~2x the 5 ms physics timestep. The residual 2.65 mm is roughly the jaw's own
# half-width, so the tissue sits between the jaws rather than at a point, which
# is what it should do.
GRASP_EQ_SOLREF = tuple(
  float(x) for x in os.environ.get("TISSUE_EQ_SOLREF", "0.010,1.0").split(","))


def _vertex_body_index(v: int) -> int:
  """flexcomp names each free vertex's body by its OWN vertex index.

  This used to compute "the count of unpinned vertices before this one, plus
  one", verified on the free corner (vertex 8) -- which is the one vertex where
  the two rules agree, because only vertex 0 is pinned before it. Re-measured
  across the whole grid by comparing body xpos against flexvert_xpos: the
  identity is exact for all 64 free vertices (worst error 0.0000 mm) while the
  counting rule is wrong for 56 of them, by up to 52 mm, and names seven bodies
  that do not exist at all.

  Nothing had noticed because the only caller that walks the whole grid is the
  nearest-vertex grasp, which is default-off -- so TISSUE_GRASP_NEAREST was
  attaching its equality to a vertex up to 52 mm from the one the jaw pinched.
  """
  return v


def grasp_eq_name(v: int) -> str:
  """Name of the per-vertex grasp equality for FLEX VERTEX `v`.

  Keyed on the vertex rather than its position in some list, so grasp.py can
  look the equality up by name and no ordering convention has to be kept in
  sync between the two modules.
  """
  return f"{GRASP_EQ_NAME}_v{v}"


def add_grasp_equality(spec) -> None:
  """INACTIVE connects between the jaw and the graspable tissue vertices.

  Added on the assembled scene spec, which is the only place both bodies exist
  -- the robot and the tissue are separate entity specs, and the flexcomp's
  per-vertex bodies only appear once they are merged.

  ONE equality, against the free corner, when the grasp targets that corner.

  One equality PER FREE VERTEX when TISSUE_GRASP_NEAREST is on, and that is not
  an optimisation -- it is the only way the nearest-vertex grasp can be correct
  in a BATCHED sim. mujoco_warp lays eq_obj2id out as a (neq,) field shared by
  every world, so "retarget the equality at the vertex this env pinched" writes
  the same target into all 1024 envs: whichever env grasps first decides what
  every other env drags. eq_active, by contrast, IS per-world (nworld, neq). So
  the per-env target is expressed by SELECTION -- give every graspable vertex
  its own permanently-compiled, normally-inactive equality and let each env
  switch on the single one it is holding with.

  Cheap because they are inactive: 64 free vertices on the 9x9 grid, and an
  equality that is off contributes no constraint rows to the solve.
  """
  def _add(name: str, vbody: int) -> None:
    eq = spec.add_equality()
    eq.name = name
    eq.type = mujoco.mjtEq.mjEQ_CONNECT
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = "robot/tool_gripper1_link"
    eq.name2 = f"tissue/tissue_{vbody}"
    eq.data[:3] = GRASP_EQ_ANCHOR
    eq.active = False

  if not gr.GRASP_NEAREST:
    _add(GRASP_EQ_NAME, _vertex_body_index(free_corner_id()))
    return
  pin = set(pinned_ids())
  for v in range(TISSUE_COUNT * TISSUE_COUNT):
    if v not in pin:                 # pinned vertices have no body to weld to
      _add(grasp_eq_name(v), _vertex_body_index(v))


def get_tissue_keylight_spec() -> mujoco.MjSpec:
  """liftpeg's keylight, scaled up for this scene only.

  The shared spec renders the workspace at 0.62x albedo, which puts the real
  tissue's colour out of reach at ANY rgba (see TISSUE_LIGHT_SCALE). Wrapping
  rather than editing LS.LIGHT_DIFFUSE keeps liftpeg -- which shares that
  constant and has a deployed policy -- untouched.
  """
  spec = LS.get_keylight_spec()
  for light in spec.lights:
    # NOT clamped at 1.0: LIGHT_DIFFUSE is already ~(1.0, 0.99, 0.94), so a
    # unit clamp would silently turn a 1.8x scale into 1.07x. light_diffuse is
    # not a normalised colour; --light-scale in replay_real_episode.py clamps
    # at 4.0 and this matches it.
    light.diffuse = tuple(
      min(4.0, c * TISSUE_LIGHT_SCALE) for c in light.diffuse
    )
  return spec


def psm_lift_tissue_a3b_env_cfg(
  play: bool = False, state_only: bool = False, reward: str = "imitate"
) -> ManagerBasedRlEnvCfg:
  """`reward` selects the shaping arm; everything else is identical.

  * ``imitate`` (default) -- phase-locked distance to the scripted trajectory's
    waypoint. What the distillation arm uses; keeps every existing run
    reproducible.
  * ``staged`` -- liftpeg's nested-ceiling potential, see
    `surgical_lab.tasks.retract.mdp.rewards_tissue`. Task shaping rather than
    imitation, so the policy is free to find its own route to the same end.
  """
  cfg = a3b.psm_push_wide_a3b_env_cfg(play=play)

  # mjlab spreads envs across the world on a grid and offsets every entity by
  # its env origin -- but a flexcomp's vertices are NOT offset. Every world's
  # sheet therefore sits at the same absolute position while each env's robot
  # and camera sit 0.5 m away, so the tissue is out of frame in every env.
  # Collapsing the spacing costs nothing (mujoco_warp worlds are independent
  # physics instances; the spread is only for visualisation) and puts the sheet
  # back under the tool. Any flex task on this scaffolding needs this line.
  cfg.scene.env_spacing = 0.0

  ents = dict(cfg.scene.entities)
  for gone in ("topdown", "block"):
    ents.pop(gone, None)
  ents["tissue"] = EntityCfg(spec_fn=get_tissue_spec)
  ents["liftcam"] = EntityCfg(spec_fn=get_tissue_cam_spec)
  ents["keylight"] = EntityCfg(spec_fn=get_tissue_keylight_spec)
  ents["goal"] = EntityCfg(spec_fn=get_tissue_goal_spec)
  # Home height, the same override liftpeg makes. MERGE, never replace: a3b's
  # dict carries entries we must not drop.
  ents["robot"].init_state.joint_pos = {
    **ents["robot"].init_state.joint_pos,
    "main_insertion": TISSUE_HOME_INSERTION,
  }
  cfg.scene.entities = ents
  if GRASP_MODE == "connect":
    _prev_spec_fn = cfg.scene.spec_fn

    def _with_grasp_eq(spec):
      if _prev_spec_fn is not None:
        _prev_spec_fn(spec)
      add_grasp_equality(spec)

    cfg.scene.spec_fn = _with_grasp_eq

  _prev_fn = cfg.scene.spec_fn

  def _with_floor_colour(spec):
    if _prev_fn is not None:
      _prev_fn(spec)
    set_tissue_floor_colour(spec)

  cfg.scene.spec_fn = _with_floor_colour

  if TISSUE_CAM_VIEW in ("wrist", "sbs"):
    # Bake the wrist camera's SOURCE fov into the spec. See _bake_wrist_fovy:
    # doing this after the env is built is a no-op and leaves the warp reading
    # a 90 deg render as a 135 deg one, a 2.41x magnification.
    _prev_wrist_fn = cfg.scene.spec_fn

    def _with_wrist_fovy(spec):
      if _prev_wrist_fn is not None:
        _prev_wrist_fn(spec)
      _bake_wrist_fovy(spec)

    cfg.scene.spec_fn = _with_wrist_fovy

  from surgical_lab.mdp.actions import BinaryJointPositionActionCfg

  cfg.actions["cartesian_delta"].position_axes = (0, 1, 2)
  cfg.actions["cartesian_delta"].min_target_z = TISSUE_MIN_TARGET_Z
  cfg.actions["jaw_pos"] = BinaryJointPositionActionCfg(
    entity_name="robot",
    joint_names=("jaw",),
    open_command_expr={"jaw": 0.5},
    close_command_expr={"jaw": 0.0},
  )

  cfg.observations["critic"].terms["state"] = ObservationTermCfg(func=_critic_state)
  # The actor's 3-D zeros placeholder becomes real proprioception.
  # 32 steps of history, not 1. The teacher's action is a discontinuous
  # function of a phase driven by STEP COUNTERS (CLOSE_STEPS=30,
  # VERIFY_STEPS=22), and a single frame cannot see a counter. Measured with
  # the teacher itself driving and pure supervised regression: per-dim error
  # was [0.000, 0.001, 0.152, 0.036] -- all of it in u_z and jaw, the two
  # channels that step at a phase change -- and the loss oscillated 0.02 <->
  # 0.38 in time with the transitions rather than converging. djaw separates
  # CLOSE from VERIFY in steady state but not at the instant it flips.
  # 32 >= max(CLOSE_STEPS, VERIFY_STEPS) + margin, so the window always spans
  # the transition the policy has to detect.
  # Memory, switchable so the ablation varies ONE thing.
  #
  #   TISSUE_HIST      steps of home_delta (dx,dy,dz,djaw) history
  #   TISSUE_ACT_HIST  steps of last-action history
  #
  # Why any of this: home DR is vertical-only with 4 mm of insertion jitter, so
  # "descended N mm" does NOT mean "at the bottom" -- the same descent means
  # different things depending on where the episode started. That is deliberate
  # (it is what stops an open-loop solution working), but it makes the bottom
  # knowable ONLY by saturation: I commanded down AND z stopped changing.
  # home_delta history supplies the second half; action history supplies the
  # first. With state history alone, "z is constant" is ambiguous between
  # saturated and not-asking-to-descend.
  #
  # 32 covers CLOSE_STEPS=30 and VERIFY_STEPS=22, the longest thing that has to
  # be counted.
  _hist = int(os.environ.get("TISSUE_HIST", "32"))
  _ahist = int(os.environ.get("TISSUE_ACT_HIST", "32"))
  if TISSUE_IMAGE_ONLY:
    # CAMERA AND NOTHING ELSE. The base a3b config's 3-D constant-zeros term is
    # left in place rather than replaced, which is the same shape every
    # deployed liftpeg policy uses -- one image input once the zeros are baked
    # out at export, and no state for the inference node to reconstruct.
    #
    # Be clear-eyed about what this gives up. The history was not margin: the
    # scripted teacher's action is a discontinuous function of a phase driven
    # by STEP COUNTERS (CLOSE_STEPS=30, VERIFY_STEPS=22) and a single frame
    # cannot see a counter. Regression error was measured at
    # [0.000, 0.001, 0.152, 0.036] -- all of it in u_z and jaw, the two
    # channels that step at a phase change. Separately, bucketing the teacher
    # action by `home_delta` left only 0.045 of within-bucket variance out of
    # 0.3476, so proprioception alone explained 87 % of the target before the
    # camera contributed anything. Removing it removes most of the signal the
    # BC auxiliary was fitting, which is why that auxiliary is turned off with
    # it (SURGICAL_LAB_BC_COEF=0) rather than left to chase an unfittable
    # target and fight the PPO gradient.
    pass
  else:
    cfg.observations["actor"].terms = {
      "home_delta": ObservationTermCfg(
        func=home_delta, **({"history_length": _hist} if _hist > 0 else {})
      ),
    }
    if _ahist > 0:
      cfg.observations["actor"].terms["actions"] = ObservationTermCfg(
        func=a3b.base_mdp_last_action, history_length=_ahist
      )
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if getattr(s, "name", "") != "topdown_cam"
  ) + (
    CameraSensorCfg(
      name="liftcam_cam",
      camera_name="liftcam/liftcam",
      height=84,
      width=84,
      data_types=("rgb",),
      enabled_geom_groups=(0, 1, 2),
      use_shadows=True,
      use_textures=True,
    ),
  )
  cfg.observations["camera"].terms = {
    "liftcam_rgb": ObservationTermCfg(
      func=a3b._camera_rgb_normalized, params={"sensor_name": "liftcam_cam"}
    ),
  }
  if TISSUE_CAM_VIEW in ("wrist", "sbs"):
    # Rendered WIDE (lp.WRIST_SRC px at lp.WRIST_RENDER_FOV, baked above);
    # _camera_wrist resamples that down to 84x84 through the fisheye grid.
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
      CameraSensorCfg(
        name="wristcam_cam",
        camera_name="robot/wrist_camera",
        height=lp.WRIST_SRC,
        width=lp.WRIST_SRC,
        data_types=("rgb",),
        enabled_geom_groups=(0, 1, 2),
        use_shadows=True,
        use_textures=True,
      ),
    )
    # The front sensor stays in the scene under `wrist` too -- the sbs view
    # needs it, and the in-training recorder and the visibility events read it.
    cfg.observations["camera"].terms = {
      "liftcam_rgb": ObservationTermCfg(
        func=_camera_wrist if TISSUE_CAM_VIEW == "wrist" else _camera_sbs
      ),
    }
  # The scripted action for THIS state, carried through the rollout so the BC
  # auxiliary in SurgicalLabOnPolicyRunner can regress on it. Deliberately its
  # own group and deliberately absent from `obs_groups`: rsl-rl stores every
  # group in the rollout but only builds networks over the groups the agent
  # config names, so this reaches the optimiser without reaching the actor.
  # Putting it in "critic" would leak it to the state-teacher arm, where actor
  # and critic share that group and the actor could just copy it.
  # DIAGNOSTIC. The teacher switches CLOSE -> VERIFY on a step counter, and the
  # two look identical in frame (jaw shut, tool at the bottom) while the
  # commanded u_z differs by a full unit. If the BC loss only collapses once
  # the actor is handed the phase directly, the observation -- not the
  # optimiser -- is what is capping it.
  if os.environ.get("SURGICAL_LAB_PHASE_OBS") == "1":
    cfg.observations["actor"].terms["phase_onehot"] = ObservationTermCfg(
      func=_phase_onehot
    )

  cfg.observations["teacher"] = ObservationGroupCfg(
    terms={"teacher_act": ObservationTermCfg(func=rs.teacher_action)},
  )

  cfg.rewards = {
    # PHASE-LOCKED. The earlier staged potential rewarded pulling the tissue in
    # any direction, and the policy duly found grab-drag-release: over 419
    # iterations grasp_attempts fell 9.4 -> 1.3 and grasped 0.90 -> 0.38 while
    # revealed_only rose to 0.19 with success_rate still 0. This scores
    # distance to the scripted trajectory's waypoint for the CURRENT phase, so
    # there is no credit for finding another route.
    # THE objective: match the scripted action at every step. Dominant weight,
    # because everything else is a tiebreaker next to "do what the script does".
    "imitate": RewardTermCfg(func=rs.action_match, weight=10.0),
    # Kept, small, only so there is still a gradient along the phases when the
    # action match is already near-perfect.
    "phase": RewardTermCfg(func=rs.phase_potential, weight=0.5),
    # Leaving the vertical line through the grasp point, while descending.
    # This is the freedom that produced the shortcut.
    "leash": RewardTermCfg(func=rs.leash_penalty, weight=1.0),
    # retry_bonus REMOVED. It paid per rising edge of the jaw closing, gated
    # only on being low -- so the policy parked at the bottom and chattered the
    # jaw 112 times an episode. The phase potential no longer penalises a
    # retry, and a retry is the only route to PEEL, so the incentive is intact
    # without a term that can be farmed.
    "success": RewardTermCfg(func=success_bonus, weight=1.0),
    "floor": RewardTermCfg(func=floor_penalty, weight=2.0),
  }
  if reward == "staged":
    # Task shaping, not imitation -- so the scripted teacher must go.
    #
    # SurgicalLabOnPolicyRunner switches on a BC auxiliary for any task that
    # publishes a "teacher" observation group (SURGICAL_LAB_BC_COEF, default
    # 1.0), and it fires silently: the smoke test on this arm reported
    # "Mean bc loss 0.114" while nothing in the reward mentioned the teacher.
    # Leaving it on would put imitation back into the one arm whose purpose is
    # to find out whether the task is learnable from reward alone -- and a
    # failure would be uninterpretable for exactly the reason the imitation arm
    # was introduced to avoid. liftpeg, whose shaping this copies, publishes no
    # teacher group and therefore never runs the auxiliary.
    cfg.observations.pop("teacher", None)
    cfg.rewards = {
      "shaping": RewardTermCfg(func=rt.tissue_staged, weight=1.0),
      "success": RewardTermCfg(func=success_bonus, weight=1.0),
      "floor": RewardTermCfg(func=floor_penalty, weight=2.0),
    }
  elif reward != "imitate":
    raise ValueError(f"unknown reward arm {reward!r}")
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "success": TerminationTermCfg(func=tissue_revealed),
    "floor_hit": TerminationTermCfg(func=floor_violation),
  }
  if TISSUE_TARGET_LEAD_FAULT > 0.0:
    cfg.terminations["target_lead_fault"] = TerminationTermCfg(
      func=target_lead_fault)
  cfg.metrics = {
    "success_rate": MetricsTermCfg(func=success_metric, reduce="max"),
    "visible_frac": MetricsTermCfg(func=visible_metric, reduce="max"),
    "grasped": MetricsTermCfg(func=grasp_metric, reduce="max"),
    "grasp_attempts": MetricsTermCfg(func=attempts_metric, reduce="last"),
    "floor_margin_mm": MetricsTermCfg(func=floor_margin_metric, reduce="min"),
    "revealed_only": MetricsTermCfg(func=revealed_only_metric, reduce="max"),
    "tip_z_mm": MetricsTermCfg(func=tip_z_metric, reduce="max"),
    "phase": MetricsTermCfg(func=rs.phase_metric, reduce="max"),
    "leash_mm": MetricsTermCfg(func=rs.leash_mm_metric, reduce="max"),
  }
  # ALWAYS logged, even when the fault is off, so a run that would trip the
  # real arm cannot look identical in sim to one that would not. Registered
  # AFTER the dict literal above, which REPLACES cfg.metrics -- putting it
  # beside the termination dropped it silently, exactly as it did on liftpeg.
  cfg.metrics["target_lead_mm"] = MetricsTermCfg(
    func=m_target_lead_mm, reduce="max")
  if reward == "staged":
    # Which stage the potential is actually paying, and how far the lift got.
    # liftpeg's lesson: without a metric for the stage gate, a run that stalls
    # one stage short looks identical to one that never started.
    cfg.metrics["stage"] = MetricsTermCfg(func=rt.stage_metric, reduce="max")
    cfg.metrics["lift_frac"] = MetricsTermCfg(
      func=rt.lift_frac_metric, reduce="max")

  events = {}
  for k, v in cfg.events.items():
    if "block" in k or "goal" in k or k == "place_fixtures":
      continue
    acfg = (v.params or {}).get("asset_cfg")
    if acfg is not None and getattr(acfg, "name", "") == "topdown":
      acfg.name = "liftcam"
      if getattr(acfg, "light_names", None):
        # the camera entity carries no light; liftpeg puts it on `keylight`
        acfg.light_names = ("keylight",)
        acfg.name = "keylight"
      if getattr(acfg, "camera_names", None):
        acfg.camera_names = ("liftcam",)
    events[k] = v
  # Re-centre the floor colour DR on the bench, tissue-locally: a3b's range is
  # shared with pushblock and liftpeg.
  if "floor_color" in events:
    _fc = events["floor_color"]
    events["floor_color"] = EventTermCfg(
      mode=_fc.mode, func=_fc.func,
      params={**_fc.params,
              "ranges": {0: TISSUE_FLOOR_RGB_DR[0], 1: TISSUE_FLOOR_RGB_DR[1],
                         2: TISSUE_FLOOR_RGB_DR[2], 3: (1.0, 1.0)}},
    )
  # Re-centre the light DR so it can actually reach the real bench. Rebuilt
  # rather than mutated in place: `v` is a3b's own term and mutating its params
  # would reach back into the shared config.
  if "light_intensity" in events:
    _li = events["light_intensity"]
    events["light_intensity"] = EventTermCfg(
      mode=_li.mode,
      func=_li.func,
      params={**_li.params,
              "ranges": {0: TISSUE_LIGHT_DR, 1: TISSUE_LIGHT_DR,
                         2: TISSUE_LIGHT_DR}},
    )
  # No tissue respawn event: three pinned corners fix the sheet in place, so
  # there is nothing to randomise about its pose. Appearance DR still applies
  # through the shared terrain/light terms.
  # AFTER lower_tool (which sets the nominal insertion) and BEFORE the camera
  # placement, whose visibility test uses the home tip box.
  events["home_jitter"] = EventTermCfg(mode="reset", func=randomize_home_joints)
  events["place_tissue_cam"] = EventTermCfg(mode="reset", func=place_tissue_cam)
  # Rules-based grasp with a stochastic failure rate. `step` mode so the held
  # vertex is re-driven to the tool tip every control step.
  events["capture_home"] = EventTermCfg(mode="reset", func=capture_home)
  events["reset_phase"] = EventTermCfg(mode="reset", func=rs.reset_phase)
  events["advance_phase"] = EventTermCfg(mode="step", func=rs.advance_phase)
  events["reset_grasp"] = EventTermCfg(
    mode="reset", func=gr.reset_grasp, params={"vertex_ids": GRASP_IDS})
  events["update_grasp"] = EventTermCfg(
    mode="step", func=gr.update_grasp, params={"vertex_ids": GRASP_IDS})
  cfg.events = events

  # Cloth needs orders of magnitude more contact budget than the rigid tasks
  # (pushblock runs 32/512). Under-budgeting silently DROPS contacts and the
  # grasp then fails with no error, so this is sized for an 81-vertex sheet
  # plus the arm. These are per-env and multiply by num_envs -- oversizing
  # will OOM the card.
  cfg.sim.nconmax = 600
  cfg.sim.njmax = 1_200

  rec = getattr(cfg, "recorders", None)
  if rec and "policy_cam_video" in rec:
    # Film what the ACTOR receives, not a fixed sensor. Under `wrist` the
    # policy sees a fisheye-warped 84x84 and under `sbs` a concatenation of two
    # sensors; filming liftcam_cam in either case shows a view the actor is
    # never given, which reads as a miswired run when you watch the clip.
    if TISSUE_CAM_VIEW == "wrist":
      rec["policy_cam_video"].obs_func = _camera_wrist
    elif TISSUE_CAM_VIEW == "sbs":
      rec["policy_cam_video"].obs_func = _camera_sbs
    else:
      rec["policy_cam_video"].sensor_name = "liftcam_cam"

  if state_only:
    # Teacher arm: the actor reads privileged state instead of pixels. The RL
    # config maps BOTH actor and critic onto the "critic" group, so swapping
    # that group's state term is all it takes.
    #
    # The camera SENSOR and the recorder stay, deliberately. A state run is
    # otherwise blind to us -- the metrics say what happened but not how, and
    # on this project the real bugs (the tissue tearing, the grasp tell, the
    # camera-frame error) were all found by looking at frames rather than at
    # numbers. The render cost is worth paying: the teacher's advantage over
    # the vision arm is sample efficiency, not wall-clock per iteration.
    cfg.observations.pop("camera", None)
    cfg.observations["critic"].terms["state"] = ObservationTermCfg(
      func=_tissue_state)

  return cfg
