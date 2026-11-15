# third_party/mujoco_warp

Full write-up (symptom, root cause, proof, affected runs): `docs/mujoco_warp_mesh_bbox_bug.md`.
Regression check: `scripts/check_mesh_bbox_render.py` (FORCE_BUG=1 re-instates the bug).

Vendored copy of upstream `mujoco_warp` at tag `v3.10.0.1` (byte-identical to the
PyPI 3.10.0.1 wheel we were running; tests and `test_data/` dropped), plus ONE
cherry-picked line from upstream commit 7c97281 ("Improve renderer", #1602,
2026-08-31) in `mujoco_warp/_src/bvh.py::build_mesh_bvh`:

    -  half = 0.5 * (pmax - pmin)
    +  half = np.maximum(np.abs(pmin), np.abs(pmax))

Why: MuJoCo recentres mesh vertices on the mesh CoM, so a mesh's AABB is not
symmetric about the geom origin. The old line sized the per-geom bounding box
about the AABB centre while `bvh.py::_compute_bvh_bounds` places it on the geom
origin, so e.g. the top 14.5 mm of the LiftPeg post sat outside its own box. The
ray tracer then skipped the peg for rays that already had a nearer hit on the
ring wall behind it -> the green ring was drawn OVER the yellow post whenever
the ring rose above ~21.5 mm (tilted by a jaw / lifted). Visible in both the
wrist and the front camera. Pixel-exact vs MuJoCo's OpenGL renderer after the fix.

Why not pin upstream git: the fixing commit requires mujoco>=3.12 and is 121
commits (23k lines, renderer rewrite included) past 3.10.0.1, while the pinned
mjlab requires mujoco~=3.10 / mujoco-warp~=3.10. Drop this directory and go back
to the upstream source once mjlab supports a mujoco_warp release that contains
7c97281 (v3.12.0 does NOT).

Version is `3.10.0.1+bvhfix` so mjlab's `mujoco-warp~=3.10.0,>=3.10.0.1` still resolves.
