"""Lift-Splat-Shoot BEV model (the reference baseline for the LenaLab BEV domain).

Surround cameras -> per-pixel (depth distribution x context) -> lift to a camera frustum of 3D
points -> splat (voxel-pool) into the ego BEV grid -> conv head -> vehicle-occupancy logits.
Geometry uses the real scaled intrinsics + cam->ego extrinsics from the data adapter.

This module is importable: `from bev_lss import LiftSplat, BevData`. The grader loads a checkpoint
and runs `model(imgs, intrins, cam2ego)`; it never trains. Reference trainer in __main__.
"""
import os, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------- data ---------
class BevData(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npz")))
        # ImageNet normalization (backbone is pretrained)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        imgs = torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0   # (6,3,H,W)
        imgs = (imgs - self.mean) / self.std
        return (imgs, torch.from_numpy(d["intrins"]).float(),
                torch.from_numpy(d["cam2ego"]).float(),
                torch.from_numpy(d["bev"]).float())


# --------- model ---------
class LiftSplat(nn.Module):
    def __init__(self, H=128, W=352, grid=(200, 200), xb=(-50., 50.), yb=(-50., 50.),
                 res=0.5, dbound=(4.0, 45.0, 1.0), C=64, downsample=16):
        super().__init__()
        self.H, self.W, self.ds, self.C = H, W, downsample, C
        self.XG, self.YG = grid
        self.xb, self.yb, self.res = xb, yb, res
        self.dbins = torch.arange(*dbound)
        self.D = len(self.dbins)
        _pre = os.environ.get("SCRATCH", "0") != "1"   # SCRATCH=1 -> from-scratch (no pretrained download)
        bb = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT if _pre else None)
        self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                      bb.layer1, bb.layer2, bb.layer3)   # -> /16, 256 ch
        self.depthnet = nn.Conv2d(256, self.D + C, 1)
        self.bevenc = nn.Sequential(
            nn.Conv2d(C, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1))
        self.register_buffer("frustum", self._frustum())                # (D,fH,fW,3) u,v,d

    def _frustum(self):
        fH, fW = self.H // self.ds, self.W // self.ds
        ds = self.dbins.view(-1, 1, 1).expand(self.D, fH, fW)
        xs = torch.linspace(0, self.W - 1, fW).view(1, 1, fW).expand(self.D, fH, fW)
        ys = torch.linspace(0, self.H - 1, fH).view(1, fH, 1).expand(self.D, fH, fW)
        return torch.stack((xs, ys, ds), -1)                            # (D,fH,fW,3)

    def geometry(self, intrins, cam2ego):
        """frustum pixel/depth -> ego-frame 3D points. returns (B,N,D,fH,fW,3)."""
        B, N = intrins.shape[:2]
        fr = self.frustum.to(intrins.device)
        # (u,v,d) -> camera ray point (d*x, d*y, d)
        pts = torch.cat((fr[..., :2] * fr[..., 2:3], fr[..., 2:3]), -1)  # (D,fH,fW,3)
        pts = pts.view(1, 1, *pts.shape).expand(B, N, *pts.shape).unsqueeze(-1)  # (...,3,1)
        Kinv = torch.inverse(intrins).view(B, N, 1, 1, 1, 3, 3)
        cam = (Kinv @ pts).squeeze(-1)                                   # camera coords
        R = cam2ego[..., :3, :3].view(B, N, 1, 1, 1, 3, 3)
        t = cam2ego[..., :3, 3].view(B, N, 1, 1, 1, 3)
        return (R @ cam.unsqueeze(-1)).squeeze(-1) + t                   # ego coords

    def voxel_pool(self, geom, feat):
        """geom (B,N,D,fH,fW,3) ego pts, feat (B,N,D,fH,fW,C) -> BEV (B,C,XG,YG)."""
        B = geom.shape[0]
        gx = ((geom[..., 0] - self.xb[0]) / self.res).long()            # x fwd -> row
        gy = ((geom[..., 1] - self.yb[0]) / self.res).long()            # y left -> col
        keep = (gx >= 0) & (gx < self.XG) & (gy >= 0) & (gy < self.YG)
        out = geom.new_zeros(B, self.C, self.XG * self.YG)
        feat = feat.reshape(B, -1, self.C)                              # (B,P,C)
        gx, gy, keep = gx.reshape(B, -1), gy.reshape(B, -1), keep.reshape(B, -1)
        for b in range(B):                                              # per-batch scatter-add
            k = keep[b]
            idx = (gx[b, k] * self.YG + gy[b, k])
            out[b].index_add_(1, idx, feat[b, k].t())
        return out.view(B, self.C, self.XG, self.YG)

    def forward(self, imgs, intrins, cam2ego):
        B, N = imgs.shape[:2]
        x = self.backbone(imgs.view(B * N, *imgs.shape[2:]))            # (B*N,256,fH,fW)
        x = self.depthnet(x)
        depth = x[:, :self.D].softmax(1)                               # (B*N,D,fH,fW)
        ctx = x[:, self.D:]                                            # (B*N,C,fH,fW)
        feat = depth.unsqueeze(2) * ctx.unsqueeze(1)                    # outer product -> lift
        fH, fW = feat.shape[-2:]
        feat = feat.view(B, N, self.D, self.C, fH, fW).permute(0, 1, 2, 4, 5, 3)
        geom = self.geometry(intrins, cam2ego)
        bev = self.voxel_pool(geom, feat)
        return self.bevenc(bev).squeeze(1)                             # (B,XG,YG) logits


def build_model():
    """Entry point the harness-owned grader calls to instantiate the solver's model."""
    return LiftSplat()


def iou(logits, gt, thr=0.0):
    pred = (logits > thr)
    g = gt > 0.5
    inter = (pred & g).sum().item()
    union = (pred | g).sum().item()
    return inter / union if union > 0 else float("nan")


# --------- reference trainer ---------
if __name__ == "__main__":
    import sys
    root = sys.argv[1]
    out = sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    tr = torch.utils.data.DataLoader(BevData(root, "train"), batch_size=4, shuffle=True,
                                     num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(BevData(root, "val"), batch_size=4, num_workers=4)
    model = LiftSplat().to(dev)
    # pos_weight: vehicles are a small fraction of the grid
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=epochs, steps_per_epoch=len(tr))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=dev))
    best = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, K, c2e, bev in tr:
            imgs, K, c2e, bev = imgs.to(dev), K.to(dev), c2e.to(dev), bev.to(dev)
            logits = model(imgs, K, c2e)
            loss = lossf(logits, bev)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval(); ious = []
        with torch.no_grad():
            for imgs, K, c2e, bev in va:
                imgs, K, c2e, bev = imgs.to(dev), K.to(dev), c2e.to(dev), bev.to(dev)
                logits = model(imgs, K, c2e)
                for b in range(imgs.shape[0]):
                    ious.append(iou(logits[b], bev[b]))
        miou = float(np.nanmean(ious))
        if miou > best:
            best = miou
            torch.save(model.state_dict(), out)
        print(f"  epoch {ep:02d}  loss {loss.item():.3f}  val_IoU {miou:.4f}  (best {best:.4f})", flush=True)
    print(f"REF_DONE seed={seed} best_val_IoU={best:.4f} -> {out}", flush=True)
