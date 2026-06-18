"""
model.py — Multi-camera BEV vehicle-occupancy network.
Exposes: build_encoder(), build_bev_head()

Encoder: [B*6, 3, 128, 352] -> depth_logits [B*6, D, 16, 44]
                              -> context     [B*6, C, 16, 44]
BEV head: [B, C, 200, 200]  -> occupancy_logits [B, 200, 200]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from bev_core import DEPTH_BINS

# ── Hyper-parameters ─────────────────────────────────────────────────────────
D = DEPTH_BINS   # 41 depth bins  (fixed by scaffold)
C = 128          # context channels (our choice)
# ─────────────────────────────────────────────────────────────────────────────


# ── Primitives ────────────────────────────────────────────────────────────────
class ConvBnAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, act=True):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, k, s, p, bias=False),
                  nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """Standard pre-activation residual block (bottleneck optional)."""
    def __init__(self, c, mid=None):
        super().__init__()
        mid = mid or c
        self.conv1 = ConvBnAct(c, mid, k=3, s=1, p=1)
        self.conv2 = ConvBnAct(mid, c, k=3, s=1, p=1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class DownBlock(nn.Module):
    """Strided downsample + N residual blocks."""
    def __init__(self, cin, cout, n_res=2):
        super().__init__()
        self.down = ConvBnAct(cin, cout, k=3, s=2, p=1)
        self.res  = nn.Sequential(*[ResBlock(cout) for _ in range(n_res)])

    def forward(self, x):
        return self.res(self.down(x))


# ── Encoder ───────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Input : [B*N, 3, 128, 352]
    Output: depth_logits [B*N, D, 16, 44]
            context      [B*N, C, 16, 44]

    Backbone: stem(3→32) + 3× DownBlock (/8 total) → 16×44 @ 256-ch
    """
    def __init__(self):
        super().__init__()
        # Stem: keeps full resolution for first refinement
        self.stem = nn.Sequential(
            ConvBnAct(3, 32, k=3, s=1, p=1),
            ConvBnAct(32, 32, k=3, s=1, p=1),
        )                                        # [B*N, 32, 128, 352]

        # Three stride-2 stages → /8 → 16×44
        self.stage1 = DownBlock(32,  64,  n_res=2)   # 64×176
        self.stage2 = DownBlock(64,  128, n_res=2)   # 32×88
        self.stage3 = DownBlock(128, 256, n_res=3)   # 16×44

        # ── Depth head (WHERE in depth) ──────────────────────────────────
        # Predicts D unnormalised log-probs; softmax applied by lift_splat.
        self.depth_head = nn.Sequential(
            ConvBnAct(256, 256, k=3, s=1, p=1),
            ConvBnAct(256, 256, k=3, s=1, p=1),
            nn.Conv2d(256, D, kernel_size=1),
        )

        # ── Context head (WHAT the feature describes) ────────────────────
        self.ctx_head = nn.Sequential(
            ConvBnAct(256, 256, k=3, s=1, p=1),
            nn.Conv2d(256, C, kernel_size=1),
        )

    def forward(self, x):                        # x: [B*N, 3, H, W]
        f = self.stem(x)
        f = self.stage1(f)
        f = self.stage2(f)
        f = self.stage3(f)
        return self.depth_head(f), self.ctx_head(f)


# ── BEV Head ──────────────────────────────────────────────────────────────────
class BEVHead(nn.Module):
    """
    Input : pooled BEV feature [B, C, 200, 200]
    Output: occupancy logits   [B, 200, 200]

    A residual decoder that refines the sparse pooled BEV features.
    No spatial size change needed (input is already 200×200).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Widen channels for a richer representation
            ConvBnAct(C, 256, k=3, s=1, p=1),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            # Compress
            ConvBnAct(256, 128, k=3, s=1, p=1),
            ResBlock(128),
            ResBlock(128),
            # Final 1×1 to scalar occupancy logit
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, bev):                      # [B, C, 200, 200]
        return self.net(bev).squeeze(1)          # [B, 200, 200]


# ── Public factory functions (called by bev_core.py) ─────────────────────────
def build_encoder() -> nn.Module:
    return Encoder()


def build_bev_head() -> nn.Module:
    return BEVHead()
