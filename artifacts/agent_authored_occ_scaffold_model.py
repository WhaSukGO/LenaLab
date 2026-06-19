"""
model.py – Neural network for multi-camera 3D occupancy prediction.

Architecture overview
─────────────────────
Encoder (per-camera, no pretrained weights)
  • Input : imgs  [B*6, 3, 128, 352]
  • 7×7 stem + three stride-2 residual stages → stride-8 map [B*6, 256, 16, 44]
  • Two heads from that backbone:
      depth_logits [B*6, D=41, 16, 44]  (soft depth distribution for Lift-Splat)
      context      [B*6, C=128, 16, 44] (lifted into 3D voxel grid by occ_core)

OccHead (processes 3D voxel grid → per-voxel occupancy logits)
  • Input  : vox [B, C=128, X=200, Y=200, Z=12]
  • Flatten Z into channels → [B, C*Z=1536, 200, 200]
  • 1×1 bottleneck → 256 BEV channels
  • 4× stride-1 residual blocks (256 ch, 3×3 conv)
  • 1×1 output conv → Z=12 per-(x,y) logits
  • Permute → [B, 200, 200, 12]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from occ_core import DEPTH_BINS

# ── Shared hyper-parameters ────────────────────────────────────────────────────
C_CTX = 128        # context channels (must match encoder ↔ head)
_D    = DEPTH_BINS  # 41
_Z    = 12          # voxel Z slices (fixed by occ_core grid)


# ── Building blocks ────────────────────────────────────────────────────────────

class ConvBnReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock2d(nn.Module):
    """Standard pre-act-style residual block with optional stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.shortcut(x), inplace=True)


# ── Encoder ────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Per-camera feature extractor (no pretrained weights).

    Input  : imgs  [B*6, 3, 128, 352]
    Output : (depth_logits [B*6, D, 16, 44],
               context      [B*6, C, 16, 44])
    """

    def __init__(self, C: int = C_CTX, D: int = _D):
        super().__init__()
        self.C = C

        # Stem: stride-1, 3 → 32  →  [32, 128, 352]
        self.stem = ConvBnReLU(3, 32, k=7, s=1, p=3)

        # Stage 1: stride-2  →  [64, 64, 176]
        self.stage1 = nn.Sequential(
            ResBlock2d(32, 64, stride=2),
            ResBlock2d(64, 64),
            ResBlock2d(64, 64),
        )

        # Stage 2: stride-4  →  [128, 32, 88]
        self.stage2 = nn.Sequential(
            ResBlock2d(64, 128, stride=2),
            ResBlock2d(128, 128),
            ResBlock2d(128, 128),
        )

        # Stage 3: stride-8  →  [256, 16, 44]
        self.stage3 = nn.Sequential(
            ResBlock2d(128, 256, stride=2),
            ResBlock2d(256, 256),
            ResBlock2d(256, 256),
        )

        # Depth head: two conv layers for richer depth features
        self.depth_head = nn.Sequential(
            ConvBnReLU(256, 256, k=3, s=1, p=1),
            ConvBnReLU(256, 128, k=3, s=1, p=1),
            nn.Conv2d(128, D, 1),
        )

        # Context head: encode semantic features into C channels
        self.ctx_head = nn.Sequential(
            ConvBnReLU(256, 256, k=3, s=1, p=1),
            ConvBnReLU(256, C,   k=3, s=1, p=1),
        )

    def forward(self, imgs: torch.Tensor):
        x = self.stem(imgs)    # [B*6,  32, 128, 352]
        x = self.stage1(x)     # [B*6,  64,  64, 176]
        x = self.stage2(x)     # [B*6, 128,  32,  88]
        x = self.stage3(x)     # [B*6, 256,  16,  44]
        return self.depth_head(x), self.ctx_head(x)


# ── Occupancy head ─────────────────────────────────────────────────────────────

class OccHead(nn.Module):
    """Decode a 3-D voxel feature grid into per-voxel occupancy logits.

    Input  : vox  [B, C=128, X=200, Y=200, Z=12]
    Output :      [B, X=200, Y=200, Z=12]

    Strategy: treat (C, Z) jointly as a fat BEV channel dimension, then apply
    efficient 2-D BEV residual processing.  The 1×1 bottleneck mixes all Z slices
    with all C features before spatial reasoning begins.
    """

    def __init__(self, C: int = C_CTX, Z: int = _Z):
        super().__init__()
        in_ch = C * Z  # 128 × 12 = 1536

        # 1×1 bottleneck: collapse (C, Z) → 256 BEV channels
        self.compress = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Four residual BEV blocks at full 200×200 resolution
        self.blocks = nn.Sequential(
            ResBlock2d(256, 256),
            ResBlock2d(256, 256),
            ResBlock2d(256, 256),
            ResBlock2d(256, 256),
        )

        # Final 1×1: produce one logit per Z slice at each (x, y)
        self.out_conv = nn.Conv2d(256, Z, 1)

    def forward(self, vox: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z = vox.shape
        # [B, C, X, Y, Z] → [B, C, Z, X, Y] → [B, C*Z, X, Y]
        x = vox.permute(0, 1, 4, 2, 3).reshape(B, C * Z, X, Y)
        x = self.compress(x)      # [B, 256, X, Y]
        x = self.blocks(x)        # [B, 256, X, Y]
        x = self.out_conv(x)      # [B, Z,   X, Y]
        return x.permute(0, 2, 3, 1)  # [B, X, Y, Z]


# ── Factory functions (called by occ_core.py) ─────────────────────────────────

def build_encoder() -> nn.Module:
    """Return a freshly initialised Encoder (no pretrained weights)."""
    return Encoder(C=C_CTX, D=_D)


def build_occ_head() -> nn.Module:
    """Return a freshly initialised OccHead matching the encoder's C."""
    return OccHead(C=C_CTX, Z=_Z)
