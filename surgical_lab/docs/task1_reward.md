# Task 1 reward: the pure-PPO baseline

How the reward for Task 1 was built, what each version did in training, and
why it changed. The final version (v1.1, task `Surgicallab-Dual-Task1-Staged`)
reaches ~0.93 success and is our pure-PPO baseline. Code:
`src/surgical_lab/tasks/lift/config/psm/dual_task1.py` (v0) and
`dual_task1_staged.py` (v1, v1.1).

## Contents

1. [The task](#1-the-task)
2. [Training setup](#2-training-setup)
3. [v0: the naive reward](#3-v0-the-naive-reward)
4. [v1: the staged reward](#4-v1-the-staged-reward)
5. [v1.1: set it down flat (final)](#5-v11-set-it-down-flat-final)
6. [The final reward, written out](#6-the-final-reward-written-out)
7. [Results](#7-results)
8. [Metrics](#8-metrics)
9. [Reproducing and changing it](#9-reproducing-and-changing-it)
10. [Caveats](#10-caveats)

## 1. The task

Two dVRK PSMs stand side by side, both facing the viewer, as on the lab cart.
PSM2 is 250 mm to PSM1's right, with its shaft tilted ~39 deg so the two
tools meet over the peg.

- **Phase 0 (arm 1):** lift the green ring off the yellow peg, which has a
  30 mm post; set it on the red goal square; let go.
- **Phase 1 (arm 2):** pick the ring up off the goal, carry it back over the
  peg, lower it onto the post; let go.
- **Throughout:** the two arms must never touch, anywhere. That covers the
  base, carriage, parallelogram, instrument shaft, wrist and jaws.

## 2. Training setup

Shared by every version; only the reward and its terminations change.

| | |
|---|---|
| Simulator | mjlab / MuJoCo Warp; 2 ms physics step, 10 physics steps per control step, so **dt = 0.02 s (50 Hz)** |
| Episode | 40 s (2000 steps), or shorter if a termination fires |
| Action (8-D) | per arm: `[dx, dy, dz, jaw]`. Position deltas in the arm's base frame, up to 0.5 mm per step (25 mm/s); the jaw is binary (>= 0 opens). The tool's orientation is held fixed. |
| Observation (58-D, actor and critic) | both arms' joint angles (6 + 6), last action (8), and a 38-D task state (see below) |
| Task state (38-D) | both tool positions (6); ring position (3) and orientation quaternion (4); goal and peg positions (6); per arm, the vector from tool to the nearest grip point on the ring (6); current waypoint minus ring (3); phase flag (1); per arm, holding (strict) and both pads touching (4); jaw openings (2); ring lift fraction (1); "cleared" latches (2). In v0 the 33-D task state had the ring-minus-tool vectors instead of grip points. |
| Policy / value | MLPs 256-256-128, ELU, observation normalisation |
| PPO | rsl-rl. 24 steps per env per iteration, 5 epochs, 12 minibatches, lr 3e-4 (fixed), gamma 0.99, lambda 0.95, clip 0.2, entropy 0.005, initial action std 0.5, actions clipped to +-1 |
| Exploration-noise bound (v1 onwards) | `BoundedActorMLPStdLowEnt`: a soft penalty on action std above 1.0, which settles std at ~1.2. In v0 the std drifted 0.5 -> 2.8 once the reward went flat. |
| Scale | 1024 parallel envs, 6000 iterations (~147M steps), one RTX 4080 (DH lab machine), ~1.9 s per iteration, ~3.3 h |

**mjlab multiplies every reward term by dt.** All values below are in
**return units**: the code divides weights by dt, so "+5" means +5 added to
the episode return once.

### Arm-arm contact (every version)

A contact sensor watches the two arms' whole kinematic trees against each
other. Every link has a collision shape against the other arm (boxes fitted
to the upper links' meshes, capsules on the wrist, the real jaw pads). The
only exception is the few-mm `tool_yaw_link`. Any contact gives **-10 and
ends the episode**.

## 3. v0: the naive reward

Task `Surgicallab-Dual-Task1-Naive`. The simplest thing that describes the
task, as a baseline for how far plain PPO gets without task-specific shaping.

Paid **every step**:

```
reach = 1.0 * (1 - tanh(|ring - tool_active| / 2 cm))   active arm: arm 1 in phase 0, arm 2 in phase 1
move  = 2.0 * (1 - tanh(|ring - target|     / 5 cm))   target: the goal (phase 0), resting on the peg (phase 1)
```

Paid **once**:

| Event | Condition | Reward |
|---|---|---|
| handoff | ring touching the goal, arm 1 not holding it | +5; phase 0 -> 1 |
| success | phase 1, ring resting on the peg (< 3 mm above rest, centred within 3.9 mm), arm 2 not holding it | +10, episode ends |
| contact | any arm-arm contact | -10, episode ends |

"Holding" means both of that arm's jaw pads are in contact with the ring.
This is checked per arm: the single-arm helper pools every jaw in the scene,
so it would count arm 1's left pad plus arm 2's right pad as a grasp.

**Result** (run `1xcaxun9`, stopped at ~1200 iterations): plateaued from
iteration ~300. Arm 1 drove its tool onto the ring on the peg and **parked
there for the whole episode**: reach ~0.85, held 0, lift 0, handoff 0. The
action std drifted 0.5 -> 2.8.

**Why:**
1. `reach` pays every step for sitting near the ring, forever, so parking is
   a stable optimum.
2. `move` measures the ring against the goal **at table height**. Lifting the
   ring straight up off the 30 mm post moves it *away* from that point
   (~54 -> 62 mm), so the one move that has to happen first **lowers** the
   reward.
3. Getting the ring off the post takes ~60 consecutive coordinated steps
   while gripping. Random exploration never finds that with no signal along
   the way.

## 4. v1: the staged reward

Task `Surgicallab-Dual-Task1-Staged`, as of commit `e491282`. It reuses the
structure that already trains the lab's single-arm peg lift
(`liftpeg_shaped.arm_e_shaped`, after robosuite's staged rewards):

- **One progress score, the potential PHI.** Each step pays its *change*:
  `r_t = PHI(s_t+1) - PHI(s_t)`. Sitting still earns nothing, so parking
  stops paying. Over an episode the shaping sums to `PHI(end) - PHI(start)`.
- **Nested stages.** Each stage starts at or above the best value of the
  stage before it, so moving on to the next stage never lowers PHI.

### The stages

`d_grip_k` is arm k's tool distance to its nearest **grip point** (below);
`lift = clamp((z_ring - 5.99 mm) / 32 mm, 0, 1)`, where 32 mm takes the
ring's bottom from resting on the plate to 2 mm above the 36 mm post top
(`CLEAR_Z` = 38 mm).

| PHI | Stage | Active when | Value |
|---|---|---|---|
| 0 - 1 | arm 1 reach | otherwise (phase 0) | `1 - tanh(12 * d_grip_1)` |
| 1 - 7 | arm 1 lift | arm 1 holding | `1 + 6 * lift` |
| 7 - 10 | arm 1 carry | holding, and ring has cleared the post (latched) | `7 + 3 * (1 - tanh(6 * d_ring_goal_xy))` |
| 10 - 13 | arm 1 place | cleared, ring centre over the goal square | `10 + 3 * down`, `down = clamp((38 mm - z) / 34 mm, 0, 1)` |
| **14** | **handoff** | ring touching the goal, arm 1 not holding it | phase 0 -> 1 |
| 14 - 15 | arm 2 reach | otherwise (phase 1) | `14 + 1 - tanh(12 * d_grip_2)` |
| 15 - 21 | arm 2 lift | arm 2 holding | `15 + 6 * lift` |
| 21 - 25 | arm 2 carry | holding, and arm 2 has lifted it past 38 mm (latched) | `21 + 3 * (1 - tanh(6 * d_ring_peg_xy)) + 1 * (1 - tanh(d_ring_peg_xy / 4 mm))` |
| 25 - 28 | arm 2 insert | cleared, ring centred over the post (< 4.4 mm) | `25 + 3 * down`, `down = clamp((38 mm - z) / 32 mm, 0, 1)` |
| 29 | seated | cleared, ring resting on the peg | `29` |

The carry-to-peg stage has the extra fine term (4 mm scale) because the
ring's bore only has ~3.4 mm of play around the post.

### Events (outside the potential)

| Event | Condition | Reward |
|---|---|---|
| success | phase 1, ring resting on the peg, arm 2's pads not both on it | +5, episode ends |
| lost | ring on the floor (z <= 4 mm) away from the goal, after it was lifted, in either phase | -2, episode ends |
| contact | any arm-arm contact | -10, episode ends |

A full success is worth 29 (shaping) + 5 = **34**.

### "Holding" (strict, per arm)

All three must hold:

1. both of **that arm's** jaw pads are in penetrating contact with the ring;
2. that arm's jaw is closed (joint < 0.25);
3. its tool is at a **grip point** (within 3.5 mm in xy and 4 mm in z).

Contact alone is not enough. The lab's single-arm runs learned to rest the
jaw on the ring's rim, which registered as a grasp without gripping
anything.

**Grip points.** The tool's orientation is fixed, so each arm's jaws close
along a fixed world axis, measured per arm from its own pad geometry. A wall
can only be pinched where the ring's radial direction lines up with that
axis. There are two such places, each at mid-wall and 10.3 mm up the ring,
rotated with the ring's full orientation so a tilted ring still works. This
is the same construction the scripted demo uses.

**Result** (run `pfvo7emt`, 6000 iterations): **phase 0 learned by iteration
~750** (handoff 0.96, ring lifted 32 mm, off the post every time). **Phase 1
never:** arm 2 reached and sometimes gripped the ring but never got it above
~12 mm, and success was 0. Action std stayed ~1.1, so the bound worked.

**Why** (from the training videos; the user spotted the first one):

1. **Arm 1 dropped the ring onto the goal from ~25 mm.** It landed tipped,
   often half on the square's edge, and arm 2 could not grip a tipped ring.
   v1 allowed this twice over: the handoff only asked for the ring to
   *touch* the goal, and `place` pays for the ring being low, which a
   dropped ring reaches instantly. Potential shaping telescopes, so it
   cannot tell a drop from a set-down that ends in the same state, and
   dropping is faster.
2. **Arm 2's lift had a dead zone.** `lift` is measured from the ring's
   resting height on the *peg plate* (6 mm), but on the goal square the ring
   sits at 0.4 mm. Arm 2's first ~5.6 mm of lift paid nothing, which is the
   same "the necessary first move is unpaid" trap as v0, for arm 2 only.

## 5. v1.1: set it down flat (final)

Same task, commit `bd4391c`. Three changes to v1:

| Change | Detail |
|---|---|
| **Flat handoff** | The handoff additionally needs the ring lying flat: tilt < 15 deg, computed from its orientation. A tipped ring keeps the episode in phase 0. |
| **Release-height penalty** | When arm 1 lets go of a lifted ring in phase 0 (its pads go from both touching to not), it pays once: `-6 * clamp((z_release - 6 mm) / 20 mm, 0, 1)`. Releasing at 6 mm or below is free; at 26 mm or above it costs the full -6. This is outside the potential, because the potential cannot see it. 6 is larger than the whole place stage (3): dropping saves ~1 s of lowering, and once phase 1 pays, arriving sooner is worth more than that stage. |
| **Arm 2's lift from the handoff height** | `lift_2 = clamp((z - z_handoff) / (38 mm - z_handoff), 0, 1)`, with `z_handoff` latched when the handoff fires. Arm 2 is paid from its first millimetre. |

Checked before training on the scripted Task 1 episode
(`DEMO_TASK=Surgicallab-Dual-Task1-Staged scripts/demo_task1.py`):

- **Normal run:** PHI rises through every stage, 0.4 -> 29; the release at
  0.9 mm costs nothing; the handoff fires at 4.6 deg tilt; success fires; no
  arm contact.
- **`DEMO_DROP=1` run** (let go from carry height): the release at 42.7 mm
  is charged the full -6.

## 6. The final reward, written out

The v1.1 reward in one place, three ways: in plain words, in the state
quantities from our state-information list (`state_information/README.md`,
branch `state_information`), and in the code's own variable names (the
`terms()` function in `dual_task1_staged.py`).

### 6.1 In words

Per step, the reward is the change in a progress score, PHI. PHI says how far
through the relay the episode is:

1. **Arm 1 Jaw approaches the Ring** (grip point on the wall).
2. **Arm 1 holds the Ring** and lifts it straight up, until the Ring clears
   the Peg's post.
3. Arm 1 carries the **Ring towards the Goal**.
4. Over the Goal, arm 1 **lowers the Ring**.
5. Arm 1 lets go with the **Ring lying flat on the Goal**, and the phase
   switches to arm 2. Letting go high costs extra.
6. **Arm 2 Jaw approaches the Ring**.
7. **Arm 2 holds the Ring** and lifts it until it clears the post.
8. Arm 2 carries the **Ring over the Peg**, precisely.
9. Arm 2 **lowers the Ring onto the Peg**.
10. Arm 2 lets go with the **Ring resting on the Peg**: success, the episode
    ends.

At any point:
- **Arm 1 touching Arm 2** (any part) costs -10 and ends the episode.
- **Dropping the Ring on the floor** away from the Goal and Peg costs -2 and
  ends it.

### 6.2 Symbols

| Symbol (code variable) | State-information quantity | Definition |
|---|---|---|
| `tools[:, k]` | Arm k Jaw Position | tool midpoint of arm k (env frame) |
| `ring`, `z = ring[:, 2]` | Ring position | ring origin, which is its *bottom* centre; rests at 5.99 mm on the peg plate, 0.4 mm on the goal |
| `ring_q`, `tilt` | Ring orientation | quaternion; `tilt` = angle between the ring's axis and vertical |
| `goal`, `peg` | Goal position, Peg position | |
| `jaw[:, k]` | (jaw opening) | arm k's jaw joint; < 0.25 = closed |
| *(fixed)* | Arm k Jaw Angle | tool orientation is held constant (position-only actions), so it is not a variable. It sets each arm's jaw closing axis `ax_k`, which defines the grip points. |
| `d_grip[:, k]` | Arm k Jaw to Ring | distance from arm k's tool to the nearest of its two grip points |
| `contact[:, k]` | (part of) Arm k Holding Ring | both of arm k's pads in contact with the ring |
| `held[:, k]` | **Arm k Holding Ring** | `contact & (jaw < 0.25) & near grip point` (3.5 mm xy, 4 mm z) |
| `d_goal_xy` | Ring to Goal | horizontal ring-goal distance |
| `on_goal` | Ring to Goal | ring in solver contact with the goal square |
| `d_peg_xy` | Ring to Peg | horizontal ring-peg distance |
| `on_peg` | Ring to Peg | `z < 5.99 + 3 mm` and `d_peg_xy < 3.89 mm` |
| `arm_arm_contact` | Arm 1 Jaw to Arm 2 Jaw (extended to whole arms) | contact sensor between the two arms' entire link trees |
| `lift_frac` | | `clamp((z - Z_REST) / LIFT_SPAN, 0, 1)`, `Z_REST` = 5.99 mm, `LIFT_SPAN` = 32.0 mm |
| `CLEAR_Z` | | 38 mm: ring bottom 2 mm above the 36 mm post top |
| `phase1`, `cleared1`, `cleared2`, `z_hand` | | episode latches (below) |

### 6.3 In the code's variables

```python
# --- per-step quantities ---
held     = contact & (jaw < JAW_CLOSED) & near_grip                    # (n, 2), per arm
flat     = tilt < FLAT_DEG                                             # 15 deg
handed   = ~phase1 & on_goal & flat & ~contact[:, 0]                   # arm 1 let go, ring flat on goal
phase1  |= handed                                                      # latched; z_hand = z at this step
cleared1 |= ~phase1 & (z >= CLEAR_Z)                                   # ring off the post (phase 0)
cleared2 |= phase1 & held[:, 1] & (z >= CLEAR_Z)                       # arm 2 lifted it past post height
success  = phase1 & on_peg & ~contact[:, 1]
lost     = ((~phase1 & cleared1 & (z <= DOWN_Z) & ~on_goal & (d_goal_xy > GOAL_HALF + LOST_MARGIN))
            | (phase1 & cleared2 & (z <= DOWN_Z) & ~on_goal))
released = ~phase1 & cleared1 & prev_contact1 & ~contact[:, 0]        # arm 1 opening on a lifted ring

# --- phase 0 potential (arm 1) ---
reach0 = 1 - tanh(12 * d_grip[:, 0])                                   # 0 .. 1
lift0  = 1 + 6 * lift_frac                                             # 1 .. 7
carry0 = 7 + 3 * (1 - tanh(6 * d_goal_xy))                             # 7 .. 10
place0 = 10 + 3 * clamp((CLEAR_Z - z) / (CLEAR_Z - DOWN_Z), 0, 1)       # 10 .. 13
phi0 = reach0
phi0 = where(held[:, 0], lift0, phi0)
phi0 = where(cleared1 & held[:, 0], carry0, phi0)
phi0 = where(cleared1 & (d_goal_xy < GOAL_HALF), place0, phi0)

# --- phase 1 potential (arm 2) ---
lift2_frac = clamp((z - z_hand) / (CLEAR_Z - z_hand), 0, 1)
reach1  = 14 + 1 - tanh(12 * d_grip[:, 1])                              # 14 .. 15
lift1   = 15 + 6 * lift2_frac                                           # 15 .. 21
carry1  = 21 + 3 * (1 - tanh(6 * d_peg_xy)) + 1 * (1 - tanh(d_peg_xy / 0.004))   # 21 .. 25
insert1 = 25 + 3 * clamp((CLEAR_Z - z) / (CLEAR_Z - Z_REST), 0, 1)      # 25 .. 28
phi1 = reach1
phi1 = where(held[:, 1], lift1, phi1)
phi1 = where(cleared2 & held[:, 1], carry1, phi1)
phi1 = where(cleared2 & (d_peg_xy < INSERT_XY), insert1, phi1)          # INSERT_XY = 4.39 mm
phi1 = where(cleared2 & on_peg, 29, phi1)

phi = where(phase1, phi1, phi0)

# --- per-step reward, in return units (the code divides weights by dt) ---
r = (phi - phi_prev)                                                   # 0 on the first step of an episode
    + 5 * success
    - 2 * lost
    - 6 * released * clamp((z - 0.006) / 0.020, 0, 1)
    - 10 * arm_arm_contact
done = success | lost | arm_arm_contact | time_out(40 s)
```

### 6.4 In the state-information terms

With `Hold_k` = Arm k Holding Ring, `D(a, b)` = distance, `D_xy` =
horizontal distance, `s(x, w) = 1 - tanh(x / w)` (1 at x = 0, falling to
~0.24 at x = w), and the Ring's height `h`, measured from its resting height
on the peg plate:

**Phase 0 (arm 1), PHI in 0..13:**

```
PHI = s(D(Arm 1 Jaw, Ring grip point), 83 mm)                     if not Hold_1
    = 1 + 6 * h / 32 mm                                             if Hold_1, Ring still on the Peg
    = 7 + 3 * s(D_xy(Ring, Goal), 167 mm)                          if Hold_1, Ring has cleared the Peg
    = 10 + 3 * (how far the Ring has come down)                     if Ring cleared and over the Goal
```

**Handoff:** Ring to Goal = touching, Ring orientation flat (< 15 deg), and
not Hold_1 -> phase 1. If arm 1 let go with the Ring higher than 6 mm, add
`-6 * min(1, (height - 6 mm) / 20 mm)`.

**Phase 1 (arm 2), PHI in 14..29:**

```
PHI = 14 + s(D(Arm 2 Jaw, Ring grip point), 83 mm)                if not Hold_2
    = 15 + 6 * (Ring height gained since the handoff) / (to 38 mm)  if Hold_2
    = 21 + 3 * s(D_xy(Ring, Peg), 167 mm) + s(D_xy(Ring, Peg), 4 mm)
                                                                    if Hold_2, Ring has cleared Peg height
    = 25 + 3 * (how far the Ring has come down the post)            if Ring centred over the Peg (< 4.4 mm)
    = 29                                                            if Ring to Peg = resting on it
```

**Success:** Ring to Peg = resting on it, and not Hold_2 -> +5, done.
**Arm 1 touching Arm 2** (any link) -> -10, done.
**Ring on the floor** away from Goal and Peg -> -2, done.

### 6.5 Coverage of the state-information relationships (Task 1)

| Relationship | Used? | Where |
|---|---|---|
| Arm 1 Jaw to Ring | yes | `d_grip[:, 0]`: phase 0 reach stage, and the "near grip point" part of `held` |
| Arm 1 Jaw Angle to Ring | indirectly | the jaw angle is fixed, so it is not rewarded. It sets the jaw axis, which decides **where** on the ring the grip points are. |
| Arm 1 Jaw to Peg | no | not in the reward. Lifting straight up is what the lift stage pays for, and carry only starts once the ring clears the post. Jaw-post contact is allowed and not penalised. |
| Arm 1 Jaw Angle to Peg | no | fixed |
| Arm 1 Jaw to Goal | indirectly | through Ring to Goal while held (carry, place) |
| Arm 1 Jaw Angle to Goal ("placed so arm 2 can pick it up") | yes, as the **Ring's** orientation | the flat-handoff gate (`tilt < 15 deg`) and the release-height penalty. That was the v1 failure. |
| Arm 2 Jaw to Ring | yes | `d_grip[:, 1]`: phase 1 reach stage, and `held` |
| Arm 2 Jaw Angle to Ring | indirectly | as for arm 1. Arm 2's wrist is levelled so its jaws point straight down. |
| Arm 2 Jaw to Peg | indirectly | through Ring to Peg while held (carry, insert) |
| Arm 2 Jaw Angle to Peg | no | fixed |
| Arm 1 Jaw to Arm 2 Jaw | yes, **extended to the whole arms** | contact sensor over both arms' full link trees: -10 and episode end. It is a contact, not a distance, so there is no shaping for keeping clear. |
| Ring to Goal | yes | `d_goal_xy` (carry), `on_goal` (handoff), the over-goal test (place) |
| Ring to Peg | yes | `d_peg_xy` (carry, insert), `on_peg` (seated, success) |
| Arm k Holding Ring | yes | `held[:, k]`: gates every lift, carry and place stage; `contact[:, k]` decides "let go" |

## 7. Results

Training metrics with exploration noise, one seed (42), averaged over the
last 100 iterations.

| | v0 naive | v1 staged | **v1.1 (final)** |
|---|---|---|---|
| wandb run | `1xcaxun9` | `pfvo7emt` | `xyh5dx7r` |
| iterations | ~1200 (stopped) | 6000 | 6000 |
| arm 1 lifts the ring off the post | no (0 mm) | yes (32.6 mm) | yes |
| handoff | 0 | 0.96 | **0.98** |
| ring height at arm 1's release | n/a | ~25 mm (dropped) | **~4 mm** (set down) |
| arm 2 lifts it over the post | 0 | 0 | **0.93** |
| **success** | 0 | 0 | **0.92-0.94** |
| arm-arm contact | ~0 | 0.03 | **0.01** |
| ring lost | n/a | 0 | 0 |
| episode length | 40 s (timeout) | 40 s (timeout) | **~10.6 s** (done) |
| action std | 2.8, rising | 1.11 | 0.87 |

**v1.1 learning curve:**

| Iteration | Milestone |
|---|---|
| ~880 | phase 0 learned (handoff ~0.9) |
| 1600-3200 | arm 2 learns to lift the ring |
| 3248 | first success |
| ~3600 | success 0.68 |
| ~5200 | success 0.94, flat after that |

The iteration-5083 training video shows the whole relay in ~8 s per episode.

## 8. Metrics

Logged by `Surgicallab-Dual-Task1-Staged` under `Episode_Metrics/` in wandb.

| Metric | Meaning |
|---|---|
| `phi_max` | Highest PHI reached in the episode. **The one number to watch:** >1 gripping, >7 ring off the post, 14 handed off, 29 back on the peg. |
| `handoff_rate`, `success_rate`, `lost_rate`, `arm_contact_rate` | Fraction of episodes where each happened |
| `goal_touch_rate` | Ring reached the goal in phase 0, flat or not. Well above `handoff_rate` means it is landing tipped. |
| `release_1_mm` | Ring height at arm 1's last release (want ~0-6) |
| `tilt_deg` | Ring tilt at episode end |
| `held_1_frac`, `held_2_frac`, `contact_1_frac` | Fraction of the episode each arm was holding (strict), or had both pads on the ring |
| `lift_1_mm` | Highest lift above rest in phase 0, after the reset settle; 32 = off the post |
| `ring_z_2_mm` | Highest absolute ring height in phase 1; 38 clears the post |
| `cleared_1_rate`, `cleared_2_rate` | Ring got past post-clearance height (per phase) |
| `ring_goal_mm` | Closest the ring got to the goal centre (xy) |
| `ring_peg_2_mm` | Closest the ring got to the peg centre in phase 1; 54 = never got there |
| `grip_1_mm` | Closest arm 1's tool got to a grip point |

## 9. Reproducing and changing it

Train (DH machine, inside tmux; setup in [`env_setup/`](env_setup/README.md)):

```bash
set -a; . scripts/a3_liftpeg_env.sh; set +a
MUJOCO_GL=egl uv run train Surgicallab-Dual-Task1-Staged --env.scene.num-envs 1024 \
  --agent.seed 42 --agent.run-name ppo-task1staged-s42-<initials> \
  --video True --video-length 2000 --video-interval 24000
```

Evaluate without exploration noise: success, completion time, and which
links touch on each arm-arm contact.

```bash
uv run python scripts/eval_task1.py \
  --checkpoint logs/rsl_rl/dual_task1_staged/<run dir>/model_5999.pt \
  --num-envs 256 --episodes 1024 --out eval_task1.json
```

Check a reward change against the scripted episode before training on it:

```bash
DEMO_TASK=Surgicallab-Dual-Task1-Staged MUJOCO_GL=egl uv run python scripts/demo_task1.py
DEMO_DROP=1 DEMO_TASK=Surgicallab-Dual-Task1-Staged MUJOCO_GL=egl uv run python scripts/demo_task1.py
```

Every number is an environment variable, so variants need no code change.

| Variable | Default | |
|---|---|---|
| `T1S_LIFT_GAIN`, `T1S_CARRY_GAIN`, `T1S_PLACE_GAIN`, `T1S_ALIGN_GAIN`, `T1S_INSERT_GAIN` | 6, 3, 3, 1, 3 | stage sizes |
| `T1S_SUCCESS_BONUS`, `T1S_LOST_PENALTY` | 5, 2 | event rewards |
| `T1S_FLAT_DEG` | 15 | handoff flatness |
| `T1S_RELEASE_PEN`, `T1S_RELEASE_Z_MM`, `T1S_RELEASE_SPAN_MM` | 6, 6, 20 | release penalty |
| `T1S_LIFT2_FROM_HANDOFF` | 1 | arm 2 lift reference |
| `T1_EPISODE_S` | 40 | episode length |
| `DUAL_ARM_CONTACT_PENALTY`, `DUAL_ARM_CONTACT_TERMINATE` | 10, 1 | arm contact |

`T1S_FLAT_DEG=180 T1S_RELEASE_PEN=0 T1S_LIFT2_FROM_HANDOFF=0` recovers v1.

## 10. Caveats

- **The results above are training metrics:** stochastic actions, one seed.
  The numbers for the report should come from `eval_task1.py` (mean action)
  on more seeds.
- **The shaping is `PHI(s') - PHI(s)`, not the textbook `gamma * PHI(s') -
  PHI(s)`** (Ng et al., 1999). That matches the lab's single-arm reward, but
  with gamma = 0.99 it is not exactly policy-invariant: it adds a small
  pressure towards high-PHI states.
- **The reward uses privileged simulator state** (contacts, ring pose), as a
  state-based baseline should. It is not something a camera-only policy
  could compute.
- **The remaining ~1% arm contacts have not been broken down yet;**
  `eval_task1.py` reports which links touch.
