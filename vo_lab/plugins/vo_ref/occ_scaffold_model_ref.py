"""Reference model.py for the occupancy scaffold — the known-good network filling the agent's slot.
Interface: build_encoder() (imgs[B*N,3,H,W] -> depth_logits[B*N,D,h,w], context[B*N,C,h,w]) and
build_occ_head() (vox[B,C,X,Y,Z] -> logits[B,X,Y,Z]). Geometry/aug/training are locked in occ_core.py."""
import torch.nn as nn
import torchvision
from occ_core import DEPTH_BINS

C = 32


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        bb = torchvision.models.resnet18(weights=None)
        self.trunk = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2, bb.layer3)
        self.head = nn.Conv2d(256, DEPTH_BINS + C, 1)

    def forward(self, imgs):
        x = self.head(self.trunk(imgs))
        return x[:, :DEPTH_BINS], x[:, DEPTH_BINS:]


class OccHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(C, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(True),
                                 nn.Conv3d(32, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(True),
                                 nn.Conv3d(32, 1, 1))

    def forward(self, vox):
        return self.net(vox).squeeze(1)


def build_encoder():
    return Encoder()


def build_occ_head():
    return OccHead()
