"""Batched PyTorch forward and inverse kinematics for the dVRK PSM."""

from __future__ import annotations

import torch

PI = torch.pi
PI_2 = torch.pi / 2
EPS = 1e-8


def _broadcast(*tensors, device, dtype):
  out = []
  for t in tensors:
    out.append(torch.as_tensor(t, device=device, dtype=dtype))
  return torch.broadcast_tensors(*out)


def mat_from_dh_revolute(
  alpha, a, theta, d, offset
) -> torch.Tensor:
  """DH transform for a revolute joint (batched)."""
  theta = torch.as_tensor(theta)
  device = theta.device
  dtype = theta.dtype
  alpha, a, theta, d, offset = _broadcast(
    alpha, a, theta, d, offset, device=device, dtype=dtype
  )
  ca = torch.cos(alpha)
  sa = torch.sin(alpha)
  th = theta + offset
  ct = torch.cos(th)
  st = torch.sin(th)
  zeros = torch.zeros_like(ct)
  ones = torch.ones_like(ct)
  row1 = torch.stack((ct, -st, zeros, a), dim=-1)
  row2 = torch.stack((st * ca, ct * ca, -sa, -d * sa), dim=-1)
  row3 = torch.stack((st * sa, ct * sa, ca, d * ca), dim=-1)
  row4 = torch.stack((zeros, zeros, zeros, ones), dim=-1)
  return torch.stack((row1, row2, row3, row4), dim=-2)


def mat_from_dh_prismatic(
  alpha, a, theta, d, offset
) -> torch.Tensor:
  """DH transform for a prismatic joint (batched)."""
  theta = torch.as_tensor(theta)
  device = theta.device
  dtype = theta.dtype
  alpha, a, theta, d, offset = _broadcast(
    alpha, a, theta, d, offset, device=device, dtype=dtype
  )
  ca = torch.cos(alpha)
  sa = torch.sin(alpha)
  d_eff = d + offset + theta
  zeros = torch.zeros_like(d_eff)
  ones = torch.ones_like(d_eff)
  ct = torch.ones_like(d_eff)
  st = torch.zeros_like(d_eff)
  row1 = torch.stack((ct, -st, zeros, a), dim=-1)
  row2 = torch.stack((st * ca, ct * ca, -sa, -d_eff * sa), dim=-1)
  row3 = torch.stack((st * sa, ct * sa, ca, d_eff * ca), dim=-1)
  row4 = torch.stack((zeros, zeros, zeros, ones), dim=-1)
  return torch.stack((row1, row2, row3, row4), dim=-2)


def _signed_angle(
  normal: torch.Tensor, p_from: torch.Tensor, p_to: torch.Tensor
) -> torch.Tensor:
  """Signed angle from ``p_from`` to ``p_to`` about ``normal`` (brax order)."""
  p_from = p_from / torch.clamp(
    torch.linalg.norm(p_from, dim=-1, keepdim=True), min=EPS
  )
  p_to = p_to / torch.clamp(
    torch.linalg.norm(p_to, dim=-1, keepdim=True), min=EPS
  )
  cross = torch.cross(p_from, p_to, dim=-1)
  s = torch.sum(cross * normal, dim=-1)
  c = torch.sum(p_from * p_to, dim=-1)
  return torch.atan2(s, c)


def compute_psm_fk_up_to_3(
  joint_pos: torch.Tensor, l_rcc: float = 0.4318
) -> torch.Tensor:
  """Forward kinematics through the first three joints of the PSM.

  Args:
    joint_pos: (..., 3) joint values.

  Returns:
    (..., 4, 4) homogeneous transform.
  """
  assert joint_pos.shape[-1] == 3, "Expected 3 joint values."
  zero = torch.zeros_like(joint_pos[..., 0])
  t_1_0 = mat_from_dh_revolute(zero + PI_2, zero, joint_pos[..., 0], zero, zero + PI_2)
  t_2_1 = mat_from_dh_revolute(zero - PI_2, zero, joint_pos[..., 1], zero, zero - PI_2)
  t_3_2 = mat_from_dh_prismatic(zero + PI_2, zero, joint_pos[..., 2], zero, zero - l_rcc)
  return torch.matmul(torch.matmul(t_1_0, t_2_1), t_3_2)


def compute_psm_fk(
  joint_pos: torch.Tensor,
  l_rcc: float = 0.4318,
  l_tool: float = 0.4162,
  l_yaw2ctrlpnt: float = 0.0102,
  l_pitch2yaw: float = 0.0091,
) -> torch.Tensor:
  """Forward kinematics for the PSM.

  Args:
    joint_pos: (..., 6) joint values.

  Returns:
    (..., 4, 4) homogeneous transform of the tool tip in the base frame.
  """
  assert joint_pos.shape[-1] == 6, "Expected 6 joint values."
  zero = torch.zeros_like(joint_pos[..., 0])

  t_7_0 = compute_psm_fk_up_to_3(joint_pos[..., :3], l_rcc=l_rcc)
  t_4_3 = mat_from_dh_revolute(zero, zero, joint_pos[..., 3], zero + l_tool, zero)
  t_5_4 = mat_from_dh_revolute(zero - PI_2, zero, joint_pos[..., 4], zero, zero - PI_2)
  t_6_5 = mat_from_dh_revolute(
    zero - PI_2, zero + l_pitch2yaw, joint_pos[..., 5], zero, zero - PI_2
  )
  t_7_6 = mat_from_dh_revolute(zero - PI_2, zero, zero, zero + l_yaw2ctrlpnt, zero + PI_2)

  t_7_0 = torch.matmul(t_7_0, t_4_3)
  t_7_0 = torch.matmul(t_7_0, t_5_4)
  t_7_0 = torch.matmul(t_7_0, t_6_5)
  t_7_0 = torch.matmul(t_7_0, t_7_6)
  return t_7_0


def compute_psm_ik(
  t_7_0: torch.Tensor,
  l_rcc: float = 0.4318,
  l_tool: float = 0.4162,
  l_yaw2ctrlpnt: float = 0.0102,
  l_pitch2yaw: float = 0.0091,
) -> torch.Tensor:
  """Analytical inverse kinematics for the PSM (batched).

  Args:
    t_7_0: (..., 4, 4) tool tip transform in the base frame.

  Returns:
    (..., 6) joint values that achieve the requested pose.
  """
  assert t_7_0.shape[-2:] == (4, 4), "Expected (..., 4, 4) input."
  device = t_7_0.device
  dtype = t_7_0.dtype
  batch = t_7_0.shape[:-2]

  l_tool2rcm_offset = l_rcc - l_tool

  # T_Pinch_7: translation (0, 0, -l_yaw2ctrlpnt).
  t_pinch_7 = torch.eye(4, device=device, dtype=dtype).expand(*batch, 4, 4).clone()
  t_pinch_7[..., 2, 3] = -l_yaw2ctrlpnt

  t_pinch_0 = torch.matmul(t_7_0, t_pinch_7)

  r_pinch_0 = t_pinch_0[..., :3, :3]
  # Inverse of rotation = transpose.
  r_0_pinch = r_pinch_0.transpose(-1, -2)
  p_pinch_local = torch.einsum("...ij,...j->...i", r_0_pinch, t_pinch_0[..., :3, 3])
  n_palm_pinch = -p_pinch_local
  n_palm_pinch[..., 0] = 0.0
  n_palm_pinch = n_palm_pinch / torch.clamp(
    torch.linalg.norm(n_palm_pinch, dim=-1, keepdim=True), min=EPS
  )

  t_palm_pinch = torch.eye(4, device=device, dtype=dtype).expand(*batch, 4, 4).clone()
  t_palm_pinch[..., :3, 3] = n_palm_pinch * l_pitch2yaw

  t_palm_0 = torch.matmul(torch.matmul(t_7_0, t_pinch_7), t_palm_pinch)

  palm_pos = t_palm_0[..., :3, 3]
  insertion_depth = torch.linalg.norm(palm_pos, dim=-1)
  xz_diagonal = torch.sqrt(palm_pos[..., 0] ** 2 + palm_pos[..., 2] ** 2)

  j1 = torch.atan2(palm_pos[..., 0], -palm_pos[..., 2])
  j2 = -torch.atan2(palm_pos[..., 1], xz_diagonal)
  j3 = insertion_depth + l_tool2rcm_offset

  # j4
  pinch_pos = t_pinch_0[..., :3, 3]
  cross_palmlink_x7_0 = torch.cross(
    t_7_0[..., :3, 0], pinch_pos - palm_pos, dim=-1
  )
  q123 = torch.stack((j1, j2, j3), dim=-1)
  t_3_0 = compute_psm_fk_up_to_3(q123, l_rcc=l_rcc)
  t_3_0_ry = t_3_0[..., :3, 1]
  t_3_0_rz = t_3_0[..., :3, 2]
  j4 = _signed_angle(-t_3_0_rz, cross_palmlink_x7_0, t_3_0_ry)

  # j5
  zero = torch.zeros_like(j4)
  t_4_3 = mat_from_dh_revolute(zero, zero, j4, zero + l_tool, zero)
  t_4_0 = torch.matmul(t_3_0, t_4_3)
  t_4_0_ry = t_4_0[..., :3, 1]
  t_4_0_rz = t_4_0[..., :3, 2]
  j5 = _signed_angle(-t_4_0_ry, pinch_pos - palm_pos, t_4_0_rz)

  # j6
  t_5_4 = mat_from_dh_revolute(zero - PI_2, zero, j5, zero, zero - PI_2)
  t_5_0 = torch.matmul(t_4_0, t_5_4)
  t_7_0_rz = t_7_0[..., :3, 2]
  t_5_0_rx = t_5_0[..., :3, 0]
  t_5_0_ry = t_5_0[..., :3, 1]
  j6 = _signed_angle(-t_5_0_ry, t_7_0_rz, t_5_0_rx)

  return torch.stack((j1, j2, j3, j4, j5, j6), dim=-1)
