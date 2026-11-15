"""Alpha3's SimOrReal MAE, ported from multimodal_rl.

Source of truth: `multimodal_rl` @ `train-sim-or-real-encoder`,
`extra/pretrain_vtmae_dual_output.py` (the `vst` path) plus `sear/models/
pretrain_models.py::VST`. The launch config is that repo's HEAD commit,
"tests.sh: launch config used to train the Alpha3 encoder (job 45685709)",
which runs `cc_train.py` -> `extra/pretrain_vtmae_dual_output_sweeps.sh`:

    --patch_size 7 --dim 256 --depth 4 --heads 4 --mlp_dim 512
    --decoder_dim 256 --decoder_depth 3 --decoder_heads 4
    --masking_ratio 0.75 --mean_pool_bottleneck
    --batch_size 2048 --encoder_lr 1e-4 --decoder_lr 3e-4

Only the `vst` arm is ported -- that is Alpha3. The ResNet/timm arms (E3a/E5a)
are ablations and are omitted; `single_head` (E1) and `loss_on_masked_only`
(E6) are kept because they are one-line switches on this same graph.

Three properties are deliberately faithful rather than correct, because the
Unity-trained Alpha3 they came from reaches ~90-100% on the real robot:

  * The encoder sees only the 25% visible tokens during pretraining, but
    `MAEEncoder.forward` (the RL path) runs it over all 144 unmasked -- a 4x
    change in sequence length the transformer never saw in pretraining.
  * The reconstruction loss covers ALL patches, not just the masked ones, so
    this is closer to pooled-latent autoencoding with dropout-by-masking than
    to a textbook MAE.
  * Everything passes through a single mean-pooled 256-d vector which is then
    tiled to all 144 decoder positions, so every positional detail the decoder
    uses comes from `decoder_pos_embedding` alone.

Module and parameter names mirror the original so a checkpoint from either
repo loads into the other.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vit_pytorch.vit import Transformer


class EarlyCNN(nn.Module):
  """Conv tokenizer. Verbatim from multimodal_rl's `EarlyCNN`.

  Three stride-2 convs: 84 -> 42 -> 21 -> 10, so an 84x84 input leaves a 10x10
  map that `AdaptiveAvgPool2d(12)` then UPSAMPLES to the 12x12 patch grid.
  That mismatch is Alpha3's, kept on purpose.
  """

  def __init__(self, in_channels: int, encoder_dim: int):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, encoder_dim // 8, 4, stride=2, padding=1)
    self.conv2 = nn.Conv2d(encoder_dim // 8, encoder_dim // 4, 4, stride=2, padding=1)
    self.conv3 = nn.Conv2d(encoder_dim // 4, encoder_dim // 2, 4, stride=2, padding=1)
    self.conv4 = nn.Conv2d(encoder_dim // 2, encoder_dim, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    return self.conv4(x)


class VSTEncoder(nn.Module):
  """The ViT trunk, wrapped so state-dict keys read `encoder.transformer.*`.

  multimodal_rl's `VST` also builds linear patch-embedding paths and a
  `pos_embedding` sized for image+segmentation tokens. `DualOutputMAE` uses
  none of them -- it tokenizes with EarlyCNN and owns its own pos_embedding --
  so they are dropped here. Loading an original checkpoint with strict=False
  simply reports those as unexpected.
  """

  def __init__(
    self,
    *,
    image_size: int | tuple[int, int],
    image_patch_size: int,
    dim: int,
    depth: int,
    heads: int,
    mlp_dim: int,
    image_channels: int = 3,
    dim_head: int = 64,
    dropout: float = 0.0,
  ):
    super().__init__()
    # int for a square image, (H, W) for a rectangle. The rectangle case is
    # what lets two 84x84 views be trained as one 168x84 side-by-side image on
    # this otherwise unmodified pipeline -- the control for whether the
    # multicam module's camera embeddings earn their complexity.
    h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
    for name, v in (("height", h), ("width", w)):
      if v % image_patch_size != 0:
        raise ValueError(
          f"image {name} {v} is not divisible by patch size {image_patch_size}"
        )
    self.image_height, self.image_width = h, w
    self.image_patch_height = self.image_patch_width = image_patch_size
    self.image_channels = image_channels
    self.dim = dim
    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


class DualOutputMAE(nn.Module):
  """Single image in, two reconstructions out; the loop picks which head to score.

  There is no domain embedding and no domain flag into `forward`: both heads are
  always computed, and "domain routing" is entirely a property of the training
  loop, which feeds sim on even steps and scores head 1, real on odd steps and
  scores head 2.
  """

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
    single_head: bool = False,
    loss_on_masked_only: bool = False,
  ):
    super().__init__()
    self.encoder = encoder
    self.masking_ratio = masking_ratio
    self.mean_pool_bottleneck = mean_pool_bottleneck
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
    self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

    self.mask_token = nn.Parameter(torch.randn(decoder_dim))
    self.decoder_pos_embedding = nn.Parameter(
      torch.randn(1, self.num_patches, decoder_dim)
    )
    self.decoder = Transformer(
      decoder_dim, decoder_depth, decoder_heads, decoder_dim_head,
      decoder_dim * 4, dropout=0.0,
    )

    patch_dim = img_ch * encoder.image_patch_height * encoder.image_patch_width
    self.to_pixels_img1 = nn.Sequential(
      nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, patch_dim)
    )
    if not single_head:
      self.to_pixels_img2 = nn.Sequential(
        nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, patch_dim)
      )

  def image_to_patch(self, img: torch.Tensor) -> torch.Tensor:
    ph = self.encoder.image_patch_height
    pw = self.encoder.image_patch_width
    return rearrange(img, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=ph, pw=pw)

  def _tokens_from_image(self, img: torch.Tensor) -> torch.Tensor:
    feat = self.early_conv(img)            # [B, D, H', W']
    feat = self.pool_to_patches(feat)      # [B, D, Hp, Wp]
    return feat.flatten(2).transpose(1, 2)  # [B, N, D]

  def forward(self, img: torch.Tensor, masking_ratio: float | None = None) -> dict:
    if masking_ratio is None:
      masking_ratio = self.masking_ratio
    B = img.shape[0]
    device = img.device

    patches = self.image_to_patch(img)
    N = patches.shape[1]

    num_masked = int(masking_ratio * N)
    batch_range = torch.arange(B, device=device)[:, None]
    rand_indices = torch.rand(B, N, device=device).argsort(dim=-1)
    masked_indices = rand_indices[:, :num_masked]
    unmasked_indices = rand_indices[:, num_masked:]

    tokens = self._tokens_from_image(img) + self.pos_embedding
    # The encoder only ever sees the visible tokens.
    enc_tokens = tokens[batch_range, unmasked_indices]
    encoded = self.encoder.transformer(enc_tokens)

    if self.mean_pool_bottleneck:
      pooled = encoded.mean(dim=1)                      # [B, D]
      dec_global = self.enc_to_dec(pooled)              # [B, decoder_dim]
      dec_tokens = dec_global.unsqueeze(1).expand(-1, N, -1).clone()
    else:
      dec_unmasked = self.enc_to_dec(encoded)
      # dtype must follow the source: under autocast enc_to_dec emits half and
      # an index_put into a float32 buffer raises.
      dec_tokens = torch.zeros(
        B, N, self.decoder_dim, device=device, dtype=dec_unmasked.dtype
      )
      dec_tokens[batch_range, unmasked_indices] = dec_unmasked
      dec_tokens[batch_range, masked_indices] = self.mask_token.to(dec_tokens.dtype)

    decoded = self.decoder(dec_tokens + self.decoder_pos_embedding)

    pred1 = self.to_pixels_img1(decoded)
    pred2 = pred1 if self.single_head else self.to_pixels_img2(decoded)
    return {
      "pred_img1": pred1,
      "pred_img2": pred2,
      "target": patches,
      "masked_indices": masked_indices,
      "unmasked_indices": unmasked_indices,
    }

  def compute_loss(self, output: dict, use_img1: bool) -> torch.Tensor:
    """MSE against the input, on the head that owns this domain.

    Alpha3 scores every patch, including the ones the encoder could see.
    `loss_on_masked_only` (E6) restricts it to the masked patches, which is what
    a textbook MAE does.
    """
    pred = output["pred_img1"] if (use_img1 or self.single_head) else output["pred_img2"]
    target = output["target"]
    if not self.loss_on_masked_only:
      return F.mse_loss(pred, target)
    idx = output["masked_indices"]
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
    return F.mse_loss(pred.gather(1, gather_idx), target.gather(1, gather_idx))


def build_alpha3_mae(
  *,
  image_size: int | tuple[int, int] = 84,
  patch_size: int = 7,
  dim: int = 256,
  depth: int = 4,
  heads: int = 4,
  mlp_dim: int = 512,
  decoder_dim: int = 256,
  decoder_depth: int = 3,
  decoder_heads: int = 4,
  masking_ratio: float = 0.75,
  single_head: bool = False,
  loss_on_masked_only: bool = False,
) -> DualOutputMAE:
  """Alpha3's exact configuration, from the job-45685709 launch script."""
  encoder = VSTEncoder(
    image_size=image_size,
    image_patch_size=patch_size,
    dim=dim,
    depth=depth,
    heads=heads,
    mlp_dim=mlp_dim,
  )
  return DualOutputMAE(
    encoder=encoder,
    decoder_dim=decoder_dim,
    decoder_depth=decoder_depth,
    decoder_heads=decoder_heads,
    masking_ratio=masking_ratio,
    mean_pool_bottleneck=True,
    single_head=single_head,
    loss_on_masked_only=loss_on_masked_only,
  )
