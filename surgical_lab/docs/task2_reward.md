# Task 2 reward: mid-air handover (v0, untrained)

The first reward for Task 2, and the scripted check it is built on. **Status:
v0, not trained yet.** It is our best first guess from what the scripted
handover showed. Expect to revise it once a training run shows where it gets
stuck. Code: `src/surgical_lab/tasks/lift/config/psm/dual_task2_handover.py`,
task `Surgicallab-Dual-Task2-Handover`. The structure follows Task 1's staged
reward ([`task1_reward.md`](task1_reward.md)); Task 1's files are unchanged.

## Contents

1. [The task](#1-the-task)
2. [Is it possible? The scripted check](#2-is-it-possible-the-scripted-check)
3. [Reward v0](#3-reward-v0)
4. [How it differs from Task 1](#4-how-it-differs-from-task-1)
5. [Checks so far](#5-checks-so-far)
6. [Metrics](#6-metrics)
7. [Running and changing it](#7-running-and-changing-it)
8. [Open questions](#8-open-questions)

## 1. The task

Same scene, arms and actions as Task 1 (lab-cart layout, 8-D position-only
actions, the wrists do not rotate).

1. **Arm 1** lifts the ring off the peg.
2. **Arm 2** grips the ring while arm 1 still holds it, in the air.
3. **Arm 1** lets go. The ring stays in the air, held by arm 2 only.
4. **Arm 2** sets the ring down flat on the goal and lets go.

The arms must never touch (same contact sensor, -10 and the episode ends).

Training setup: identical to Task 1 (`task1_reward.md` section 2: PPO
settings, bounded-std actor, 1024 envs, 6000 iterations, 40 s episodes). The
only change to the observation is in the 38-D task state: one "ring off the
post" latch plus a "ring in the air" flag, instead of Task 1's two latches.

## 2. Is it possible? The scripted check

`scripts/demo_task2.py` does the handover with hand-written moves (no
learning), to check that the physics allows it before spending GPU time.

| What we tried | Result |
|---|---|
| Arm 2 grips at the computed height, then a 2 s hold | Ring slides out of arm 2's jaw and drops |
| Arm 2 grips 4 mm / 6 mm lower | Descent is blocked 2-4 mm short; held for 2 s, dropped during set-down |
| Arm 2 follows the tipping ring down / pushes harder / grips 2 mm higher | All dropped |
| **Arm 2 places straight after arm 1 lets go** (`DEMO_FAST=1`) | **Works: ring flat on the goal, no arm contact** |

Then the working version on more seeds and ring angles:

| | Flat on the goal | Landed tipped | Failed |
|---|---|---|---|
| Seed 0, ring angle 0, +-7, +-14, +-25 deg from the best | 6 | 1 | 0 |
| Seeds 1 and 2, best angle | 0 | 0 | 2 |

What this showed:

- **The handover is possible** in this layout.
- **Ring angle does not matter much.** The jaws close along one fixed line,
  so the two arms grip opposite walls. Arm 1's grip turns the ring to about
  the same angle whatever it started at (150-157 deg from starts across
  +-14 deg). Training therefore keeps Task 1's fully random reset angle.
- **Arm 2's grip is the weak point.** After arm 1's grip, arm 2 faces the
  thick part of the ring (~4.8 mm vs 1.3 mm for arm 1). It grips 4-7 mm higher
  than the ideal point, often with only the jaw tips, and the ring slowly
  slides out. Hovering loses it; placing quickly works.
- **The seed matters more than the angle.** The seed sets the ring's
  randomised friction (0.4-1.4) and mass (0.7-1.4x). Our guess is that
  slippery rings fail, but this has not been checked; the reward logs
  friction (`ring_friction`) so training can answer it.

## 3. Reward v0

Each step pays the change in a progress score PHI (as in Task 1): progress
pays, standing still pays nothing, and every stage starts above the best
value of the stage before it.

### 3.1 In words

1. **Arm 1 Jaw approaches the Ring** (grip point on the wall).
2. **Arm 1 holds the Ring** and lifts it until it clears the Peg's post.
3. With the Ring held up in the air by arm 1, **Arm 2 Jaw approaches the
   Ring**.
4. **Both arms hold the Ring** in the air.
5. **Handover:** arm 1 lets go while arm 2 holds the Ring, still in the air.
   The phase switches to arm 2.
6. Arm 2 carries the **Ring towards the Goal**.
7. Over the Goal, arm 2 **lowers the Ring**.
8. Arm 2 lets go with the **Ring lying flat on the Goal**: success, the episode
   ends.

At any point:
- **Letting go of a raised Ring the other arm is not holding** costs up to -6,
  scaled by the Ring's height (setting it down costs nothing).
- **Dropping the Ring on the floor** away from the Goal costs -2 and ends the
  episode.
- **Arm 1 touching Arm 2** costs -10 and ends the episode.

### 3.2 Stages

| Phase | Stage | PHI | When |
|---|---|---|---|
| 0 | reach 1 | 0 .. 1 | arm 1's tool approaching its grip point |
| 0 | lift 1 | 1 .. 7 | arm 1 holding, ring rising to 38 mm (clear of the post) |
| 0 | reach 2 | 8 .. 9 | ring held by arm 1, cleared and in the air (>= 20 mm); arm 2's tool approaching its grip point |
| 0 | both | 10 | both arms holding, ring in the air |
| | **handover** | -> phase 1 | arm 2 holding, arm 1's pads off, ring in the air, arm 1's pads on within the last 10 steps |
| 1 | regrasp | 11 .. 12 | arm 2 not holding (dropped): its tool back towards the ring |
| 1 | carry | 12 .. 15 | arm 2 holding, ring towards the goal |
| 1 | place | 15 .. 18 | over the goal, ring coming down |
| 1 | on goal | 19 | ring lying flat on the goal |
| | **success** | +5, ends | on the goal, flat, arm 2's pads off |

Best possible return: about 19 + 5 = 24 (PHI starts at ~0.4).

The 10-step window stops a relay from counting as a handover: if arm 1 drops
the ring and arm 2 picks it up off the goal or plate later, that is not a
handover.

### 3.3 Symbols

As in `task1_reward.md` section 6.2, with these changes:

| Symbol (code variable) | State-information quantity | Definition |
|---|---|---|
| `held[:, k]` | **Arm k Holding Ring** | `contact & (jaw < 0.25) & near grip point`, with **5 mm xy, 8 mm z** (Task 1: 3.5 mm, 4 mm) |
| `in_air` | Ring position (height) | `z >= AIR_Z` = 20 mm: well clear of the plate (5.99 mm) and the goal (0.4 mm) |
| `cleared1` | | latch: phase 0, arm 1 holding, `z >= CLEAR_Z` (38 mm) |
| `since_pads1` | | steps since arm 1's pads were last on the ring |
| `both` | Arm 1 Holding Ring and Arm 2 Holding Ring | `~phase1 & held1 & held2 & in_air` |
| `phase1` | | latch: set by the handover |

### 3.4 In the code's variables

```python
# --- per-step quantities (terms() in dual_task2_handover.py) ---
held     = contact & (jaw < JAW_CLOSED) & near_grip                    # near: 5 mm xy, 8 mm z
in_air   = z >= AIR_Z                                                  # 20 mm
handed   = ~phase1 & held[:, 1] & ~contact[:, 0] & in_air & (since_pads1 <= HAND_WINDOW)
phase1  |= handed                                                      # latched
cleared1 |= ~phase1 & held[:, 0] & (z >= CLEAR_Z)
both     = ~phase1 & held[:, 0] & held[:, 1] & in_air
success  = phase1 & on_goal & flat & ~contact[:, 1]                    # flat: tilt < 15 deg
lost     = ((cleared1 | phase1) & (z <= DOWN_Z) & ~on_goal
            & (d_goal_xy > GOAL_HALF + LOST_MARGIN)) | (z < -0.002)

# one-shot release cost: pads come off a raised ring the other arm is not holding
let_go    = prev_contact & ~contact                                    # per arm
released  = (~phase1 & cleared1 & let_go[:, 0] & ~held[:, 1]) \
          | (phase1 & let_go[:, 1] & ~contact[:, 0])
rel_cost  = clamp((z - RELEASE_Z) / RELEASE_SPAN, 0, 1) * released     # 6 mm, 20 mm

# --- phase 0 potential ---
reach0 = 1 - tanh(12 * d_grip[:, 0])                                   # 0 .. 1
lift0  = 1 + 6 * lift_frac                                             # 1 .. 7
reach2 = 8 + 1 - tanh(12 * d_grip[:, 1])                               # 8 .. 9
phi0 = where(held[:, 0], lift0, reach0)
phi0 = where(cleared1 & held[:, 0] & in_air, reach2, phi0)
phi0 = where(both, 10, phi0)

# --- phase 1 potential ---
regrasp = 11 + 1 - tanh(12 * d_grip[:, 1])                             # 11 .. 12
carry1  = 12 + 3 * (1 - tanh(6 * d_goal_xy))                           # 12 .. 15
place1  = 15 + 3 * clamp((CLEAR_Z - z) / (CLEAR_Z - DOWN_Z), 0, 1)    # 15 .. 18
phi1 = where(held[:, 1], carry1, regrasp)
phi1 = where(held[:, 1] & (d_goal_xy < GOAL_HALF), place1, phi1)
phi1 = where(on_goal & flat, 19, phi1)

phi = where(phase1, phi1, phi0)

# --- reward per step (return units) ---
r = (phi - phi_prev) + 5 * success - 2 * lost - 6 * rel_cost - 10 * arm_arm_contact
```

Constants as in Task 1: `GOAL_HALF` 10.3 mm, `LOST_MARGIN` 15 mm, `DOWN_Z`
4 mm, `lift_frac = clamp((z - 5.99 mm) / 32 mm, 0, 1)`.

### 3.5 In state-information terms

Kernel widths: `tanh(12 d)` falls off over ~83 mm, `tanh(6 d)` over ~167 mm.

- **PHI0** (phase 0):
  - not Arm 1 Holding Ring: `1 - tanh(Arm 1 Jaw to Ring / 83 mm)`
  - Arm 1 Holding Ring: `1 + 6 * lift fraction of Ring position`
  - Arm 1 Holding Ring, Ring off the post and in the air:
    `9 - tanh(Arm 2 Jaw to Ring / 83 mm)`
  - Arm 1 Holding Ring and Arm 2 Holding Ring, in the air: `10`
- **Handover:** Arm 2 Holding Ring, Arm 1's pads off it, Ring in the air,
  Arm 1 was holding it 0.2 s ago at most.
- **PHI1** (phase 1):
  - not Arm 2 Holding Ring: `12 - tanh(Arm 2 Jaw to Ring / 83 mm)`
  - Arm 2 Holding Ring: `12 + 3 * (1 - tanh(Ring to Goal / 167 mm))`
  - Arm 2 Holding Ring, Ring over the Goal: `15 + 3 * (how far the Ring has come down from 38 mm to 4 mm)`
  - Ring flat on the Goal: `19`
- **Success:** Ring flat on the Goal and Arm 2's pads off it.

## 4. How it differs from Task 1

| | Task 1 (v1.1) | Task 2 (v0) |
|---|---|---|
| Phase switch | ring flat on the goal, arm 1 let go | arm 2 holding alone, ring in the air, arm 1 held it within 10 steps |
| Middle stages | carry and place (arm 1), then arm 2 picks it off the goal | arm 2 reaches and grips while arm 1 holds it up |
| Holding tolerance | 3.5 mm xy, 4 mm z | 5 mm xy, 8 mm z (arm 2 grips high; see section 5) |
| Release penalty | arm 1 letting go high in phase 0 | either arm letting go of a raised ring the other arm is not holding |
| Final stage | ring seated on the peg | ring flat on the goal |
| Extra metrics | | `both_rate`, `hand_z_mm`, `ring_friction` |

## 5. Checks so far

- **The reward follows the scripted handover.** Running `demo_task2.py`
  inside the task (`DEMO_TASK=Surgicallab-Dual-Task2-Handover`, seed 0,
  `DEMO_FAST=1`): arm 1 grips at step 229 (PHI 1.0), the ring clears the
  post at step 351 (8.2), both hold at 699 (10.0), the handover registers at
  760 (14.4), and success fires at 966 (19.0). No release penalty and no arm
  contact.
- **Bug found by that check:** with Task 1's strict 3.5 mm / 4 mm holding
  tolerance, arm 2's high grip counted as holding for only 3 steps. The
  handover never registered and arm 1's release was charged as a drop
  (return -6.2 for a run that actually worked). Hence the looser tolerance.
- **It trains:** a 2-iteration CPU run (16 envs) runs and logs every metric.
- **Not done yet:** any real training run.

## 6. Metrics

Logged under `Episode_Metrics/` in wandb.

| Metric | Meaning |
|---|---|
| `phi_max` | Highest PHI reached. **The one number to watch:** > 1 gripping, > 8 ring clear of the post, 10 both holding, > 11 handed over, 19 on the goal. |
| `both_rate` | Episodes where both arms held the ring in the air |
| `handoff_rate`, `success_rate`, `lost_rate`, `arm_contact_rate` | Fraction of episodes where each happened |
| `cleared_1_rate` | Arm 1 lifted the ring clear of the post |
| `held_1_frac`, `held_2_frac` | Fraction of the episode each arm was holding |
| `lift_1_mm` | Highest lift above rest in phase 0 |
| `hand_z_mm` | Ring height at the handover |
| `release_mm` | Ring height at the last penalised release (a drop) |
| `ring_goal_mm` | Closest the ring got to the goal centre in phase 1 (1000 = never) |
| `grip_2_mm` | Closest arm 2's tool got to its grip point (logged as the episode average in run `k36zdnep`, before this was fixed) |
| `tilt_deg` | Ring tilt at episode end |
| `ring_friction` | The ring's randomised friction that episode, to see whether failures are the slippery rings |

## 7. Running and changing it

Train (DH machine, inside tmux; do the 3-iteration smoke test first):

```bash
set -a; . scripts/a3_liftpeg_env.sh; set +a
MUJOCO_GL=egl uv run train Surgicallab-Dual-Task2-Handover --env.scene.num-envs 1024 \
  --agent.seed 42 --agent.run-name ppo-task2handover-v0-s42-<initials> \
  --video True --video-length 2000 --video-interval 24000
```

Logs go to `logs/rsl_rl/dual_task2_handover/`. Check a reward change
against the scripted handover before training on it:

```bash
DEMO_TASK=Surgicallab-Dual-Task2-Handover DEMO_FAST=1 DEMO_G2_DZ=-0.004 \
  DEMO_SHARE_LIFT=0.002 MUJOCO_GL=egl uv run python scripts/demo_task2.py
```

Every number is an environment variable:

| Variable | Default | |
|---|---|---|
| `T2_LIFT_GAIN`, `T2_CARRY_GAIN`, `T2_PLACE_GAIN` | 6, 3, 3 | stage sizes |
| `T2_SUCCESS_BONUS`, `T2_LOST_PENALTY` | 5, 2 | event rewards |
| `T2_RELEASE_PEN`, `T2_RELEASE_Z_MM`, `T2_RELEASE_SPAN_MM` | 6, 6, 20 | release penalty |
| `T2_GRIP_TOL_XY_MM`, `T2_GRIP_TOL_Z_MM` | 5, 8 | what counts as holding |
| `T2_AIR_Z_MM` | 20 | ring "in the air" |
| `T2_HAND_WINDOW` | 10 | steps arm 1 may have let go before the handover |
| `T2_FLAT_DEG` | 15 | "flat" on the goal |
| `T2_EPISODE_S` | 40 | episode length |
| `DUAL_ARM_CONTACT_PENALTY`, `DUAL_ARM_CONTACT_TERMINATE` | 10, 1 | arm contact (Task 1's) |

Scripted demo settings (`demo_task2.py`): `DEMO_FAST`, `DEMO_G2_DZ`,
`DEMO_SHARE_LIFT`, `DEMO_YAW_OFF` (degrees past the best ring angle),
`DEMO_YAW=keep` (random angle), `SEED`, `DEMO_TRACE=1`.

## 8. Open questions

- **Will PPO find a firmer grip than the script did?** The script's arm 2
  grip is weak and slides. If PPO cannot do better, `success_rate` will level
  off; `ring_friction` will show whether the failures are the slippery rings.
  If so, narrowing the friction range for Task 2 is an option.
- **Only 9 scripted runs.** They show the handover is possible, not how often
  it works.
- **The looser holding tolerance** could let a jaw resting near the rim count
  as holding (the problem Task 1's strict check was for). Watch the training
  videos.
- **The shaping is `PHI(s') - PHI(s)`, not `gamma * PHI(s') - PHI(s)`**, as in
  Task 1 (see `task1_reward.md` section 10).
