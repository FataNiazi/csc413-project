"""Two-camera dual-output MAE: front + wrist in, four reconstructions out.

The single-camera model (`dual_output_mae.DualOutputMAE`) takes one image and
emits two reconstructions, one per domain head, with domain routing living
entirely in the training loop. This keeps that structure exactly and adds a
second camera.

What changes:

* **A learned camera embedding**, one row per view, added to every token of
  that view -- the same device the thesis used for its sim/real domain
  embedding, applied to a different factor. Both views share the tokenizer and
  the spatial positional embedding, because they are the same kind of sensor
  looking at the same grid; only the camera row differs.

* **Independent masks per view.** Same ratio, different random pattern, so the
  model must reason across views rather than inpaint each one on its own.

* **Both views concatenated into one token sequence** before the encoder, so
  attention is genuinely cross-camera rather than two towers with a late fuse.

* **Each domain head reconstructs BOTH views** (hence four reconstructions in
  total), and the loss scores the head owning the input's domain over both.

Why a camera embedding rather than pasting the two images side by side: with a
learned row per view there is nothing spatially special about "left" or
"right", so the model cannot pick up a spurious left/right geometry that does
not exist in the rig.

Why there is still NO domain embedding: embed the factor the model should
CONDITION on, not the factor it should be INVARIANT to. Which camera produced a
token is a real, knowable property the model should exploit. Whether the token
came from sim or real is precisely the nuisance the bottleneck exists to
squeeze out, and handing the encoder a flag for it would work against that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from surgical_lab.encoders.dual_output_mae import EarlyCNN, Transformer, VSTEncoder

N_VIEWS = 2  # front, wrist


class MultiCamDualOutputMAE(nn.Module):
  def __init__(
    self,
    *,
    encoder: VSTEncoder,
    decoder_dim: int = 256,
    decoder_depth: int = 3,
    decoder_heads: int = 4,
    decoder_dim_head: int = 64,
    masking_ratio: float = 0.75,
    mean_pool_bottleneck: bool = True,
    per_view_pool: bool = False,
    single_head: bool = False,
    loss_on_masked_only: bool = False,
  ):
    super().__init__()
    self.encoder = encoder
    self.masking_ratio = masking_ratio
    self.mean_pool_bottleneck = mean_pool_bottleneck
    # With a single pooled vector the decoder must reconstruct BOTH views from
    # one 256-d code -- roughly 165:1 compression against the single-camera
    # arm's 83:1, which is the most likely reason multi-cam reconstructions
    # stay blurry while single-cam's are sharp. Pooling PER VIEW gives each its
    # own code and doubles the per-view capacity. Adds no parameters, so a
    # checkpoint trained under either setting loads into the other.
    self.per_view_pool = per_view_pool
    self.single_head = single_head
    self.loss_on_masked_only = loss_on_masked_only
    self.decoder_dim = decoder_dim

    encoder_dim = encoder.dim
    self.encoder_dim = encoder_dim
    self.Hp = encoder.image_height // encoder.image_patch_height
    self.Wp = encoder.image_width // encoder.image_patch_width
    self.num_patches = self.Hp * self.Wp

    img_ch = encoder.image_channels
    self.early_conv = EarlyCNN(img_ch, encoder_dim)
    self.pool_to_patches = nn.AdaptiveAvgPool2d((self.Hp, self.Wp))
    self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim))
    # One row per camera, added to every token of that view.
    self.cam_embedding = nn.Parameter(torch.randn(N_VIEWS, encoder_dim) * 0.02)
    self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

    self.mask_token = nn.Parameter(torch.randn(decoder_dim))
    self.decoder_pos_embedding = nn.Parameter(
      torch.randn(1, self.num_patches, decoder_dim))
    # The decoder needs to know which view a slot belongs to as well -- with the
    # mean-pool bottleneck the only thing distinguishing the two halves of the
    # sequence is these embeddings.
    self.decoder_cam_embedding = nn.Parameter(
      torch.randn(N_VIEWS, decoder_dim) * 0.02)
    self.decoder = Transformer(
      decoder_dim, decoder_depth, decoder_heads, decoder_dim_head,
      decoder_dim * 4, dropout=0.0)

    patch_dim = img_ch * encoder.image_patch_height * encoder.image_patch_width
    self.to_pixels_img1 = nn.Sequential(
      nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, patch_dim))
    if not single_head:
      self.to_pixels_img2 = nn.Sequential(
        nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, patch_dim))

  # -- helpers --------------------------------------------------------------
  def image_to_patch(self, img):
    ph, pw = self.encoder.image_patch_height, self.encoder.image_patch_width
    return rearrange(img, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)

  def _tokens_from_image(self, img):
    feat = self.pool_to_patches(self.early_conv(img))
    return feat.flatten(2).transpose(1, 2)

  def embed(self, views):
    """Per-view tokens with the spatial and camera embeddings applied."""
    return [self._tokens_from_image(v) + self.pos_embedding
            + self.cam_embedding[i][None, None, :]
            for i, v in enumerate(views)]

  # -- forward --------------------------------------------------------------
  def forward(self, views, masking_ratio: float | None = None) -> dict:
    """`views` is a list [front, wrist] of [B, C, H, W]."""
    assert len(views) == N_VIEWS, f"expected {N_VIEWS} views, got {len(views)}"
    if masking_ratio is None:
      masking_ratio = self.masking_ratio
    B = views[0].shape[0]
    device = views[0].device
    N = self.num_patches
    num_masked = int(masking_ratio * N)
    batch_range = torch.arange(B, device=device)[:, None]

    patches = [self.image_to_patch(v) for v in views]
    tokens = self.embed(views)

    # A DIFFERENT permutation per view, so the visible patches of one view do
    # not line up with the visible patches of the other.
    masked_idx, unmasked_idx, vis = [], [], []
    for i in range(N_VIEWS):
      perm = torch.rand(B, N, device=device).argsort(dim=-1)
      mi, ui = perm[:, :num_masked], perm[:, num_masked:]
      masked_idx.append(mi)
      unmasked_idx.append(ui)
      vis.append(tokens[i][batch_range, ui])

    encoded = self.encoder.transformer(torch.cat(vis, dim=1))

    if self.mean_pool_bottleneck:
      if self.per_view_pool:
        # encoded is the two views' visible tokens concatenated in order, so
        # the split is exactly at the per-view visible count.
        k = unmasked_idx[0].shape[1]
        parts = []
        for i in range(N_VIEWS):
          g = self.enc_to_dec(encoded[:, i * k:(i + 1) * k].mean(dim=1))
          parts.append(g.unsqueeze(1).expand(-1, N, -1))
        dec_tokens = torch.cat(parts, dim=1).clone()
      else:
        dec_global = self.enc_to_dec(encoded.mean(dim=1))
        dec_tokens = dec_global.unsqueeze(1).expand(-1, N_VIEWS * N, -1).clone()
    else:
      dec_unmasked = self.enc_to_dec(encoded)
      k = unmasked_idx[0].shape[1]
      parts = []
      for i in range(N_VIEWS):
        t = torch.zeros(B, N, self.decoder_dim, device=device,
                        dtype=dec_unmasked.dtype)
        t[batch_range, unmasked_idx[i]] = dec_unmasked[:, i * k:(i + 1) * k]
        t[batch_range, masked_idx[i]] = self.mask_token.to(t.dtype)
        parts.append(t)
      dec_tokens = torch.cat(parts, dim=1)

    dec_pos = torch.cat(
      [self.decoder_pos_embedding + self.decoder_cam_embedding[i][None, None, :]
       for i in range(N_VIEWS)], dim=1)
    decoded = self.decoder(dec_tokens + dec_pos)

    pred1 = self.to_pixels_img1(decoded)
    pred2 = pred1 if self.single_head else self.to_pixels_img2(decoded)
    return {
      "pred_img1": pred1,
      "pred_img2": pred2,
      "target": torch.cat(patches, dim=1),
      "masked_indices": masked_idx,
      "unmasked_indices": unmasked_idx,
      "num_patches": N,
    }

  def compute_loss(self, output: dict, use_img1: bool) -> torch.Tensor:
    """MSE on the head owning this domain, over BOTH views."""
    pred = output["pred_img1"] if (use_img1 or self.single_head) \
        else output["pred_img2"]
    target = output["target"]
    if not self.loss_on_masked_only:
      return F.mse_loss(pred, target)
    N = output["num_patches"]
    idx = torch.cat([output["masked_indices"][i] + i * N
                     for i in range(N_VIEWS)], dim=1)
    g = idx.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
    return F.mse_loss(pred.gather(1, g), target.gather(1, g))


def build_multicam_mae(*, image_size: int = 84, patch_size: int = 7,
                       dim: int = 256, depth: int = 4, heads: int = 4,
                       mlp_dim: int = 512, decoder_dim: int = 256,
                       decoder_depth: int = 3, decoder_heads: int = 4,
                       masking_ratio: float = 0.75,
                       mean_pool_bottleneck: bool = True,
                       per_view_pool: bool = False,
                       single_head: bool = False,
                       loss_on_masked_only: bool = False):
  enc = VSTEncoder(image_size=image_size, image_patch_size=patch_size, dim=dim,
                   depth=depth, heads=heads, mlp_dim=mlp_dim)
  return MultiCamDualOutputMAE(
    encoder=enc, decoder_dim=decoder_dim, decoder_depth=decoder_depth,
    decoder_heads=decoder_heads, masking_ratio=masking_ratio,
    mean_pool_bottleneck=mean_pool_bottleneck, per_view_pool=per_view_pool,
    single_head=single_head,
    loss_on_masked_only=loss_on_masked_only)
