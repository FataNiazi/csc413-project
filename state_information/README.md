## State Information:

- Peg position (yellow structure with peg in the middle): On the ground in the center of the environment

- Goal position (red square): On the ground aligned with the Peg horizontally

- Ring position and orientation: Placed centered on the Peg, lying flat on its side, with the Peg going through the ring

- Arm 1 Jaw Position: Positioned above the Peg and aligned to the Peg with an offset on one axis

- Arm 2 Jaw Position: Positioned above the Peg and aligned to the Peg with an opposite offset on the same axis compared to Arm 1

- Arm 1 Jaw Angle and Arm 2 Jaw Angle: Initially vertically downwards

- Arm 1 Holding Ring and Arm 2 Holding Ring: Boolean on if Arm 1 or Arm 2 is holding the Ring, mainly used for Task 2

## Relationships: 

### Task 1:

- Arm 1 Jaw to Ring (to allow for pickup of Ring)

- Arm 1 Jaw Angle to Ring (to ensure Ring can be picked up)

- Arm 1 Jaw to Peg (to minimize collision with the Peg when extracting the Ring)

- Arm 1 Jaw Angle to Peg (to minimize collision with the Peg when extracting the Ring)

- Arm 1 Jaw to Goal (move Ring to position)

- Arm 1 Jaw Angle to Goal (to ensure Ring is placed in a way Arm 2 can pick up)

- Arm 2 Jaw to Ring (to allow for pickup of Ring)

- Arm 2 Jaw Angle to Ring (to ensure Ring can be picked up)

<!-- Ignoring Arm 2 Jaw Angle to Goal, as Ring can substitute for it -->

- Arm 2 Jaw to Peg (to return Ring to the Peg and minimize collision with the Peg)

- Arm 2 Jaw Angle to Peg (to minimize collision with the Peg)

- Arm 1 Jaw to Arm 2 Jaw (for avoiding each other)

- Ring to Goal (to tell when ring is successfully on the goal)

- Ring to Peg (to tell when ring is successfully placed back)

### Task 2:

- Arm 1 Jaw to Ring (to allow for pickup of Ring)

- Arm 1 Jaw Angle to Ring (to ensure Ring can be picked up)

- Arm 1 Jaw to Peg (to minimize collision with the Peg when extracting the Ring)

- Arm 1 Jaw Angle to Peg (to minimize collision with the Peg when extracting the Ring)

- Arm 1 Jaw to Arm 2 Jaw (for avoiding each other and transfer)

- Arm 1 Jaw Angle to Arm 2 Jaw Angle (to ensure Arm 2 Jaw can pick up Ring)

- Arm 2 Jaw to Ring (to allow for pickup of Ring)

- Arm 2 Jaw Angle to Ring (to ensure Ring can be picked up)

- Arm 2 Jaw to Goal (to move Ring towards the Goal)

- Arm 2 Jaw Angle to Goal (to place Ring in correct orientation)

- Ring to Goal (to tell when ring is successfully on the goal)

<!-- Ring to Peg is not needed since Arm 1 Jaw already checks Peg, could be used to check collisions? -->