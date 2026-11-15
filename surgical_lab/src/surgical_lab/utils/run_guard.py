"""Guard against evaluating a checkpoint in a different scene than it trained in.

`A3B_UNITY_LAYOUT` is read at import time in `a3b_env_cfg` and **defaults to 0**,
so any script that forgets to export it silently builds the pre-Unity scene:
different goal centre (0.1202 m vs 0.2088 m), different block size, different
PSM base. Nothing errors -- the policy just quietly scores ~0.44 instead of
~0.99, which reads as a broken policy rather than a broken harness.

mjlab writes the resolved env config to `<run>/params/env.yaml`, and the arm's
`main_insertion` reset position differs between the two layouts (0.211063 vs
0.04394), so that file is a reliable fingerprint of which scene a checkpoint
trained in. Compare it against the scene we just built and fail loudly.
"""

from __future__ import annotations

import re
from pathlib import Path

# Reset value of main_insertion under each layout, from a3b_env_cfg's
# "position_range" event term.
_UNITY_INSERTION = 0.211063
_LEGACY_INSERTION = 0.04394
_TOL = 1e-6


def _recorded_insertions(env_yaml: Path) -> list[float]:
  """Every scalar appearing as the low end of a `position_range` tuple."""
  text = env_yaml.read_text(errors="replace")
  return [
    float(m)
    for m in re.findall(
      r"position_range: !!python/tuple\s*\n\s*-\s*([-+0-9.eE]+)", text
    )
  ]


def assert_layout_matches_checkpoint(checkpoint: str | Path) -> None:
  """Raise if the current A3B_UNITY_LAYOUT disagrees with the checkpoint's run.

  Silently returns when the run directory has no `params/env.yaml` (e.g. a
  checkpoint copied out of its run), since a missing fingerprint is not
  evidence of a mismatch.
  """
  from surgical_lab.tasks.push.config.psm.a3b_env_cfg import UNITY_LAYOUT

  env_yaml = Path(checkpoint).resolve().parent / "params" / "env.yaml"
  if not env_yaml.exists():
    print(
      f"[run_guard] no {env_yaml} -- cannot verify the checkpoint's scene "
      f"layout; proceeding with UNITY_LAYOUT={int(UNITY_LAYOUT)}"
    )
    return

  found = _recorded_insertions(env_yaml)
  want = _UNITY_INSERTION if UNITY_LAYOUT else _LEGACY_INSERTION
  other = _LEGACY_INSERTION if UNITY_LAYOUT else _UNITY_INSERTION

  if any(abs(v - want) < _TOL for v in found):
    print(f"[run_guard] scene layout matches checkpoint (UNITY_LAYOUT={int(UNITY_LAYOUT)})")
    return

  if any(abs(v - other) < _TOL for v in found):
    trained_flag = 0 if UNITY_LAYOUT else 1
    raise SystemExit(
      f"\n[run_guard] SCENE MISMATCH\n"
      f"  this process built  : A3B_UNITY_LAYOUT={int(UNITY_LAYOUT)} "
      f"(main_insertion={want})\n"
      f"  checkpoint trained  : A3B_UNITY_LAYOUT={trained_flag} "
      f"(main_insertion={other})\n"
      f"  from {env_yaml}\n\n"
      f"  Re-run with A3B_UNITY_LAYOUT={trained_flag}. Evaluating across "
      f"layouts does not error, it just reports a wrong number.\n"
    )

  print(
    f"[run_guard] {env_yaml} records main_insertion values {found}, which match "
    f"neither known layout; cannot verify. Proceeding."
  )
