export LIFTPEG_A3_DR=1
export PSM_WRIST_MOUNT=1
export A3B_UNITY_LAYOUT=1
export A3B_UNITY_HOME=1
# Calibrated base pose. The editor raised the floor 18 mm under a fixed camera,
# which is the same geometry as lowering the camera 18 mm over a floor at 0.
export LIFTPEG_CAM_POS="0.00619,0.1175,0.14135"
export LIFTPEG_CAM_FOVY="40.1"
export LIFTPEG_CAM_PITCH="48.9"
export LIFTPEG_CAM_FOVY_DR="36.0,44.0"
# Visibility-verified: 8192/8192 tool tips in frame, worst |y| 0.952.
# pitch 45 not the calibrated 48.9 -- see the report; 48.9 asymptotes at ~1.00
# even with the jitter shrunk to nothing, because the NOMINAL sits at the edge.
export LIFTPEG_CAM_PITCH="45.0"
export LIFTPEG_CAM_FOVY_DR="38.0,44.0"
export LIFTPEG_CAM_PITCH_JITTER="3.0"
export LIFTPEG_CAM_DIST_SCALE="0.95,1.08"
# Black barrier wall, placed by hand in the scene editor against the real
# footage (band width 0 px mid-episode, peaking ~25 px near f300).
export LIFTPEG_A3_WALL=1
export LIFTPEG_A3_WALL_POS="-0.140,0.010,0.060"
# Calibrated goal offset: 3.5 mm -x, 11 mm +y of the nominal square.
export LIFTPEG_GOAL_CENTER="-0.0485,0.0237"

# Mesh assets. liftpeg_scene.py defaults to the copies in
# src/surgical_lab/robots/psm/xmls/assets, so LIFTPEG_MESH_DIR is left unset
# unless the caller set it. If a dir without triangle.obj is given, the scene
# spec fails to compile AT IMPORT TIME, mjlab's task importer swallows the
# exception, and every Surgicallab-Liftpeg-* task silently disappears from the
# registry -- the job then dies claiming the task does not exist.
