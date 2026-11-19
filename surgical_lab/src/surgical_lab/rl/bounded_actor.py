"""Actors with rl_games-style bounds regularisation on the policy mean.

dextrah's teacher trains under rl_games' `bound_loss_type: regularization,
bounds_loss_coef: 0.005` -- a mu^2 penalty that keeps the policy mean inside
[-1, 1]. Their distillation loss regresses raw mu and is well-conditioned
ONLY because of that property. Our rsl_rl teacher had no such penalty: |mu|
ran to ~15.5 against a +-1 clip, so distilling onto it spent ~93% of the
objective on magnitude the clip discards. Measured consequence: student
deterministic success 0.92 -> 0.85 as imitation loss IMPROVED 4.0 -> 0.12.

rsl_rl's PPO assembles `loss = ... - entropy_coef * entropy.mean()` with
`entropy = actor.output_entropy`. This subclass returns

    output_entropy  =  true_entropy - (bounds_coef / entropy_coef) * sum(mu^2)

so PPO's existing entropy term expands to

    - entropy_coef * true_entropy  +  bounds_coef * sum(mu^2)

i.e. exactly the rl_games regularisation term, with no PPO code copied. The
logged "entropy" metric is shifted by the penalty (cosmetic); the KL/adaptive
LR path is untouched (the teacher uses a fixed schedule anyway).

The ratio is what PPO actually applies, so it is only correct for ONE value of
`algorithm.entropy_coef`. Each subclass therefore declares the entropy_coef it
was paired with, and the rl_cfg that selects the class must match it. The cfg
is not visible from the model, so this cannot be asserted -- if entropy_coef
changes in the rl_cfg, change `_ENTROPY_COEF` on the matching class.
"""

from __future__ import annotations

import torch
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel

from surgical_lab.rl.mae_pretrained import MAEPretrainedModel
from rsl_rl.models.mlp_model import MLPModel

BOUNDS_COEF = 0.01    # penalty weight on out-of-bound mu
ENTROPY_COEF = 0.002  # dextrah teacher yaml entropy_coef (10x below a3b's)
SOFT_BOUND = 1.1      # rl_games' soft bound
SIGMA_SOFT_BOUND = 1.0  # std larger than the whole action range is not exploration


class _SoftBoundMixin:
  """Penalise only the OUT-OF-BOUND part of mu (rl_games 'bound' form).

  A plain mu^2 penalty collapsed the pushblock teacher (success rose to 0.50
  by iter ~160, then 0.0 for the rest of the run with every episode timing
  out): the optimal pushblock policy saturates at +-1, so a penalty pulling
  all mu toward zero opposes the optimum itself, and once success collapses
  there is no reward gradient to recover with. The soft-bound form is zero
  inside +-1.1 and quadratic outside -- bang-bang stays legal, the 15x blowup
  does not.
  """

  _BOUNDS_COEF: float = BOUNDS_COEF
  _ENTROPY_COEF: float = ENTROPY_COEF
  # None == bound mu only. A float additionally bounds the STD -- see
  # BoundedActorCNNStd for why that turned out to be necessary too.
  _SIGMA_MAX: float | None = None

  @property
  def output_entropy(self) -> torch.Tensor:
    entropy = super().output_entropy
    mu, sigma = self.output_distribution_params[0], None
    over = (mu.abs() - SOFT_BOUND).clamp_min(0.0)
    penalty = over.pow(2).sum(dim=-1)
    if self._SIGMA_MAX is not None:
      sigma = self.output_distribution_params[1]
      s_over = (sigma - self._SIGMA_MAX).clamp_min(0.0)
      # Broadcast: a scalar std is shared across the batch, so sum only the
      # action dimension and let the add rely on broadcasting rather than
      # assuming a leading batch dim that a "scalar" std does not have.
      penalty = penalty + s_over.pow(2).sum(dim=-1)
    return entropy - (self._BOUNDS_COEF / self._ENTROPY_COEF) * penalty


class BoundedActorMLP(_SoftBoundMixin, MLPModel):
  """The dextrah pushblock teacher's actor. entropy_coef 0.002."""


class BoundedActorMLPStd(BoundedActorMLP):
  """BoundedActorMLP, plus the soft bound on the action STD, at entropy 0.02.

  For the TISSUE state arm, which destroyed itself exactly the way the vision
  arm did before BoundedActorCNNStd existed -- one rung down the ladder, on the
  plain MLP that has no sigma bound at all:

      iter   success   std        iter   success   std
       132     0.000   1.07        267     0.90*   1.88
       200     1.000   ~1.4        367     0.000   2.60
       227     0.983   1.63
      (*oscillating 0.83-1.00 by then)

  Reaching 100% and then losing it in 170 iterations. Same mechanism as the
  CNN case: a Gaussian's entropy is sum(log sigma), so the entropy bonus pushes
  std up with constant force, and the surrogate's opposition vanishes once std
  passes the action clip because nearly every sample then clips to +-1.

  Why entropy 0.02 rather than the 0.005 the other arms use: on tissue it is
  load-bearing, not a leftover. Measured A/B on the staged reward, all with
  TISSUE_REACH_GAIN=6:

      entropy 0.005  ->  std COLLAPSED to 0.31, the policy never descended,
                         stage stayed 0 and grasp_attempts 0.000
      entropy 0.02   ->  success 1.00 by iteration 200

  The task needs the exploration to find a corner 58 mm away; it just needs
  something to stop paying for it forever. That is this bound.

  Equilibrium follows from balancing the two gradients,
  entropy_coef / sigma == 2 * BOUNDS_COEF * (sigma - SIGMA_MAX), which at
  entropy 0.02 and BOUNDS_COEF 0.01 settles sigma at (1+sqrt(5))/2 = 1.62 --
  bounded and stable, but higher than the 1.21 the vision arm settles at
  because the entropy push is 4x stronger. If 1.62 proves too loose (watch for
  success oscillating rather than holding), raise _BOUNDS_COEF: 0.04 moves the
  equilibrium to 1.21 without touching the exploration the reward depends on.
  """

  _SIGMA_MAX = SIGMA_SOFT_BOUND
  _ENTROPY_COEF = 0.02


class BoundedActorCNN(_SoftBoundMixin, SpatialSoftmaxCNNModel):
  """The liftpeg VISION actor. entropy_coef 0.005 (the low-entropy schedule).

  Measured on the shaped vision arm at iteration 4100 (success 0.000), rolling
  out the PRE-clip policy mean against an action clip of 1.0:

      dim   p5 .. p95            |mu| > 1
      dx    -6.5  .. +25.9        83.4 %
      dy   -14.1  ..  +5.8        73.9 %
      dz   -37.8  ..  +1.6        95.7 %   mean -14.6
      jaw  -19.9  .. +20.2        98.6 %

  The state actor on the same reward (success 0.991) sits at dz -3.4 .. +2.3.
  The vision actor is pinned at maximum-DESCEND and cannot recover: past the
  clip every sample maps to -1, so the head's gradient is dead. It grasps as
  well as the state policy (held 34.6% vs 58.5%) and then holds the ring at a
  standstill (-0.26 um/step) because it is still commanding down.

  `low_entropy=True` (entropy_coef 0.02 -> 0.005, init_std 1.0 -> 0.5) was
  enough to stop this on the MLP state actor and was NOT enough here -- the
  blowup is in mu, which the entropy term does not touch at all.
  """

  _ENTROPY_COEF = 0.005


class BoundedActorCNNStd(BoundedActorCNN):
  """BoundedActorCNN, plus the same soft bound on the action STD.

  Bounding mu alone fixed the descend-forever failure and took the vision arm
  from 0.000 to ~0.76, but the run then destroyed itself over the next 12000
  iterations:

      iter   success   std        iter   success   std
      7500     0.885   2.04      13000     0.583   5.73
      9000     0.768   2.63      15000     0.429   8.19
     10500     0.680   3.60      17000     0.062  11.29
     12000     0.539   4.77      18000     0.000  13.31

  Same root cause as the mu blowup, one parameter over. A Gaussian's entropy is
  sum(log sigma), so d(entropy)/d(log sigma) is a CONSTANT 1 -- the bonus
  pushes std up with undiminishing force forever. What normally opposes it is
  the surrogate loss, and that opposition VANISHES once std exceeds the action
  clip: past that point nearly every sample clips to +-1, so changing std
  barely changes the realised action distribution and the policy gradient
  w.r.t. std goes to zero. Constant push against vanishing resistance is
  exponential growth, which is what the numbers above are (~x1.08 per 500
  iterations, over 12000 iterations).

  Lowering entropy_coef only rescales the growth rate; the codebase already
  recorded that it "only DELAYS the runaway". This bounds the quantity itself.
  The penalty reaches equilibrium rather than clamping: at
  2 * BOUNDS_COEF * (sigma - SIGMA_MAX) == ENTROPY_COEF / sigma the two forces
  balance, which for these coefficients settles sigma near 1.2.
  """

  _SIGMA_MAX = SIGMA_SOFT_BOUND


class BoundedMAEPretrainedStd(_SoftBoundMixin, MAEPretrainedModel):
  """Alpha3's frozen-MAE actor with the SAME mu and sigma soft bounds.

  The liftpeg Alpha3 arm was first registered by taking the a3b runner cfg and
  overriding `actor.class_name` to `MAEPretrainedModel`. That silently dropped
  BoundedActorCNNStd, and with it BOTH bounds -- reintroducing a failure this
  module already existed to fix. Measured on that run:

      iter   success   std     entropy
      1700     0.000   2.44      8.85
      2070     0.000   3.06      9.74
      3225     0.000   5.08     11.77

  which is the BoundedActorCNNStd docstring's table again, arrived at from a
  different direction. Grasp rate reached 0.77 and lift_mm never passed 1.7:
  it can close the jaws, but with sigma five times the action clip it cannot
  hold a command steady long enough to lift.

  entropy_coef 0.005 to match the low-entropy schedule the a3b runner cfg
  applies, exactly as BoundedActorCNN declares it -- the mixin's penalty is
  scaled by BOUNDS_COEF / _ENTROPY_COEF, so this constant must track whatever
  `algorithm.entropy_coef` the paired cfg sets or the two forces are misweighted.
  """

  _ENTROPY_COEF = 0.005
  _SIGMA_MAX = SIGMA_SOFT_BOUND


class BoundedActorMLPStdLowEnt(BoundedActorMLP):
  """BoundedActorMLPStd at entropy 0.005 (the low-entropy state recipe).

  For the dual-PSM Task 1 runs: with the naive reward the unbounded MLP's std
  ran 0.5 -> 2.8 once the reward went flat (run 1xcaxun9). Equilibrium
  entropy_coef / sigma == 2 * BOUNDS_COEF * (sigma - 1) puts sigma at ~1.21.
  """

  _SIGMA_MAX = SIGMA_SOFT_BOUND
  _ENTROPY_COEF = 0.005
