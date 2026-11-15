## State Information:

- Base position: On the ground in the center of the environment

- Goal position (red square center): On the ground with it aligned with the Base, touching the edge of the Base

 - Ring position: Resting in the center above the base, with the Base’s peg right in the middle of the Ring

- Arm 1 Jaw Position: Positioned above the Base and aligned to the Base with an offset on one axis

- Arm 2 Jaw Position: Positioned above the Base and aligned to the Base with an opposite offset on the same axis compared to Arm 1

- Arm 1 and 2 Jaw Angle: Initially vertically downwards

## Relationships: 

### Task 1:

- Arm 1 Jaw to Ring (to allow for pickup of Ring)

- Arm 1 Jaw Angle to Ring (to ensure Ring can be picked up)

- Arm 1 Jaw to Base (to minimize collision with Base's peg when extracting the Ring)

- Arm 1 Jaw Angle to Base (to minimize collision with Base's peg when extracting the Ring)

- Arm 1 Jaw to Goal (move Ring to position)

- Arm 1 Jaw Angle to Goal (to ensure Ring is placed in a way Arm 2 can pick up)

- Arm 2 Jaw to Ring (to allow for pickup of Ring)

- Arm 2 Jaw Angle to Ring (to ensure Ring can be picked up)

<!-- Ignoring Arm 2 Jaw Angle to Goal, as Ring can substitute for it -->

- Arm 2 Jaw to Base (return Ring to Base and minimize collision with Base's peg)

- Arm 2 Jaw Angle to Base (minimize collision with Base's peg)

- Arm 1 Jaw to Arm 2 Jaw (for avoiding each other)

- Ring to Goal (to tell when ring is successfully on the goal)

- Ring to Base (to tell when ring is successfully placed back)

### Task 2:

- Arm 1 Jaw to Ring (to allow for pickup of Ring)

- Arm 1 Jaw Angle to Ring (to ensure Ring can be picked up)

- Arm 1 Jaw to Base (to minimize collision with Base's peg when extracting the Ring)

- Arm 1 Jaw Angle to Base (to minimize collision with Base's peg when extracting the Ring)

- Arm 1 Jaw to Arm 2 Jaw (for avoiding each other and transfer)

- Arm 1 Jaw Angle to Arm 2 Jaw Angle (to ensure Arm 2 Jaw can pick up Ring)

- Arm 2 Jaw to Ring (to allow for pickup of Ring)

- Arm 2 Jaw Angle to Ring (to ensure Ring can be picked up)

- Arm 2 Jaw to Goal (to move Ring towards the Goal)

- Arm 2 Jaw Angle to Goal (to place Ring in correct orientation)

- Ring to Goal (to tell when ring is successfully on the goal)

<!-- Ring to Base is not needed since Arm 1 Jaw already checks Base, could be used to check collisions? -->