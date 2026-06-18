"""Reference `model.py` for the BEV scaffold — the known-good network that fills the agent's slot.
Used to calibrate the scaffold (does the locked core + this network reproduce the reference IoU?).
The agent authors a file with this same interface; everything else (geometry, augmentation, training,
calibration) is locked in bev_core.py.

Interface:
  build_encoder() -> nn.Module: forward(imgs[B*N,3,H,W]) -> (depth_logits[B*N,D,h,w], context[B*N,C,h,w])
  build_bev_head() -> nn.Module: forward(bev[B,C,X,Y]) -> occupancy_logits[B,X,Y]
"""
import torch.nn as nn
import torchvision
from bev_core import DEPTH_BINS                       # D is fixed by the locked geometry

C = 64                                                # context channels (your choice; head must match)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        bb = torchvision.models.resnet18(weights=None)     # from scratch (sandbox: no network)
        self.trunk = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                   bb.layer1, bb.layer2, bb.layer3)    # -> /16, 256 ch
        self.head = nn.Conv2d(256, DEPTH_BINS + C, 1)

    def forward(self, imgs):
        x = self.head(self.trunk(imgs))
        return x[:, :DEPTH_BINS], x[:, DEPTH_BINS:]    # (depth_logits, context)


class BEVHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 1, 1))

    def forward(self, bev):
        return self.net(bev).squeeze(1)


def build_encoder():
    return Encoder()


def build_bev_head():
    return BEVHead()
