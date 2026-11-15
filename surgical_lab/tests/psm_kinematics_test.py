import unittest

import torch

from surgical_lab.robots.psm import psm_kinematics as kin


class TestPsmKinematics(unittest.TestCase):
  def test_fk_zero_shape_and_homogeneous(self) -> None:
    q = torch.zeros(6, dtype=torch.float64)
    t = kin.compute_psm_fk(q)
    self.assertEqual(t.shape, (4, 4))
    expected_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    self.assertTrue(torch.allclose(t[3, :], expected_row, atol=1e-6))

  def test_fk_batches_and_homogeneous_last_row(self) -> None:
    q = torch.stack(
      [torch.linspace(-0.2, 0.2, steps=6), torch.linspace(0.3, -0.3, steps=6)],
      dim=0,
    ).to(torch.float64)
    t = kin.compute_psm_fk(q)
    self.assertEqual(t.shape, (2, 4, 4))
    last_row = t[..., 3, :]
    expected_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=t.dtype, device=t.device)
    self.assertTrue(torch.allclose(last_row, expected_row.expand_as(last_row), atol=1e-6))

  def test_fk_up_to_3_shape(self) -> None:
    q = torch.zeros(3, dtype=torch.float64)
    t = kin.compute_psm_fk_up_to_3(q)
    self.assertEqual(t.shape, (4, 4))

  def test_ik_round_trip_single(self) -> None:
    q_true = torch.tensor([0.1, -0.3, 0.15, 0.4, -0.2, 0.6], dtype=torch.float64)
    t = kin.compute_psm_fk(q_true)
    q_sol = kin.compute_psm_ik(t)
    t_sol = kin.compute_psm_fk(q_sol)
    self.assertEqual(q_sol.shape, q_true.shape)
    self.assertTrue(torch.allclose(q_sol, q_true, atol=1e-4, rtol=1e-4))
    self.assertTrue(torch.allclose(t_sol, t, atol=1e-5, rtol=1e-5))

  def test_ik_round_trip_batch(self) -> None:
    q_true = torch.tensor(
      [
        [0.2, -0.4, 0.18, 0.5, -0.3, 0.8],
        [-0.3, 0.25, 0.12, -0.6, 0.4, -0.5],
      ],
      dtype=torch.float64,
    )
    t = kin.compute_psm_fk(q_true)
    q_sol = kin.compute_psm_ik(t)
    self.assertEqual(q_sol.shape, q_true.shape)
    t_sol = kin.compute_psm_fk(q_sol)
    self.assertTrue(torch.allclose(q_sol, q_true, atol=1e-4, rtol=1e-4))
    self.assertTrue(torch.allclose(t_sol, t, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
  unittest.main()
