"""DextrAH-RGB student network, adapted to the pushblock camera contract.

Faithful to dextrah_lab's default distillation pipeline as *implemented*
(a2c_stereo_transformer.py / stereo_encoder.py, mono-adapted), not as the
paper prose describes it:

  * ImageNet-pretrained torchvision ResNet-18, trained end-to-end, ImageNet
    per-channel normalization applied inside the module.
  * image embedding -> Linear(512,128) -> GELU -> Linear(128,64), mirroring
    their out_layer shape (theirs eats a stereo-fusion CLS token; we are
    monocular 84x84, so the trunk's pooled features stand in).
  * [img_embed(64), prev_action] -> LSTM(512, 1 layer, LayerNorm) ->
    MLP[512,512,256] ELU -> mu; sigma is a global learned parameter
    (fixed_sigma=True semantics, matching rsl_rl's Gaussian).
  * aux head: cat([lstm_out, pre-RNN features]) -> MLP[512,256] ELU ->
    Linear(256,3), predicting the TRUE block position (env-local frame),
    exactly their object_pos auxiliary (aux_coeff 1.0, plain L2).
  * No object state, no joint state in: their student's proprio is dominated
    by arm joints (which our deployed contract never had) and every velocity
    input is multiplied by zero in their checkout anyway -- vision + memory
    carry the task, which is the property being tested.

Deployment: recurrent ONNX with (h, c) as explicit inputs/outputs; the dVRK
node holds the two tensors between steps and feeds them back.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DextrahStudent(nn.Module):
  IMG_EMBED = 64
  LSTM_HIDDEN = 512

  def __init__(self, num_actions: int = 2, aux_dim: int = 3,
               sigma_init: float = 1.0, backbone: str = "resnet18"):
    super().__init__()
    self.backbone_kind = backbone
    self.register_buffer(
      "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    )
    self.register_buffer(
      "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    )
    if backbone == "resnet18":
      from torchvision.models import ResNet18_Weights, resnet18

      self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
      self.backbone.fc = nn.Identity()
      feat_dim = 512
    elif backbone == "cnn_ss":
      # a3b's conv stack + spatial softmax. Their ResNet-18 is sized for
      # 320x240; at our 84x84 its stem (7x7 s2 + maxpool) discards 4x before
      # any features exist, leaving a ~3x3 grid, and the student localises the
      # block only to ~7 mm (block width is 20 mm). Spatial softmax returns
      # per-channel (x, y) expectations, which is a positional code by
      # construction -- and this exact stack reaches 0.99 on this task under
      # direct RL, so it isolates vision from the imitation objective.
      self.backbone = nn.Sequential(
        nn.Conv2d(3, 16, 5, 2, 2), nn.ELU(),
        nn.Conv2d(16, 32, 3, 2, 1), nn.ELU(),
        nn.Conv2d(32, 64, 3, 2, 1), nn.ELU(),
      )
      feat_dim = 64 * 2  # (x, y) per channel
    else:
      raise ValueError(f"unknown backbone {backbone!r}")
    self.img_proj = nn.Sequential(
      nn.Linear(feat_dim, 128), nn.GELU(), nn.Linear(128, self.IMG_EMBED)
    )
    in_dim = self.IMG_EMBED + num_actions  # embedding + previous action
    self.pre_ln = nn.LayerNorm(in_dim)     # their rnn layer_norm: True
    self.lstm = nn.LSTM(in_dim, self.LSTM_HIDDEN, num_layers=1, batch_first=False)
    self.mlp = nn.Sequential(
      nn.Linear(self.LSTM_HIDDEN, 512), nn.ELU(),
      nn.Linear(512, 512), nn.ELU(),
      nn.Linear(512, 256), nn.ELU(),
    )
    self.mu = nn.Linear(256, num_actions)
    # log-sigma, global. Under --sigma-mode fixed there is no sigma term in the
    # loss, so this receives no gradient and holds its init: the rollout gets a
    # deliberate exploration scale instead of inheriting the teacher's 4.55,
    # which would swamp a clipped (+-1) mu target.
    self.sigma = nn.Parameter(
      torch.full((num_actions,), float(math.log(sigma_init)))
    )
    self.aux_mlp = nn.Sequential(
      nn.Linear(self.LSTM_HIDDEN + in_dim, 512), nn.ELU(),
      nn.Linear(512, 256), nn.ELU(),
    )
    self.aux_head = nn.Linear(256, aux_dim)

  def initial_state(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.zeros(1, batch, self.LSTM_HIDDEN, device=device)
    return z, z.clone()

  def forward(
    self,
    image: torch.Tensor,          # [B, 3, H, W] in [0, 1]
    prev_action: torch.Tensor,    # [B, num_actions]
    h: torch.Tensor,
    c: torch.Tensor,
  ):
    x = (image - self.img_mean) / self.img_std
    emb = self.img_proj(self.backbone(x))            # [B, 64]
    feat = torch.cat([emb, prev_action], dim=-1)     # [B, 66] pre-RNN features
    z = self.pre_ln(feat).unsqueeze(0)               # [1, B, 66] one step
    out, (h2, c2) = self.lstm(z, (h, c))
    out = out.squeeze(0)                             # [B, 512]
    trunk = self.mlp(out)
    mu = self.mu(trunk)
    sigma = self.sigma.exp().expand_as(mu)
    aux = self.aux_head(self.aux_mlp(torch.cat([out, feat], dim=-1)))
    return mu, sigma, aux, h2, c2
