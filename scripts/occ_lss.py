"""Lift-Splat-to-3D occupancy reference + variance trainer (the occupancy-domain baseline).

Surround cameras -> per-pixel (depth distribution x context) -> lift to a camera frustum of 3D
points -> voxel-pool into the ego 3D grid [X,Y,Z] -> 3D-conv head -> per-voxel occupancy logits.
Extends scripts/bev_lss.py from a 2D BEV grid to a 3D voxel grid (the only structural change is
pooling on (gx,gy,gz) and a 3D head). SCRATCH=1 -> from-scratch backbone (no pretrained download).

usage: python occ_lss.py <occ_root> <out.pt> [epochs] [seed]
"""
import os, glob, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

H, W, N = 128, 352, 6
XB = YB = (-50.0, 50.0); ZB = (-2.0, 4.0); RES = 0.5
XG = int((XB[1] - XB[0]) / RES); YG = int((YB[1] - YB[0]) / RES); ZG = int((ZB[1] - ZB[0]) / RES)
DBOUND = (4.0, 45.0, 1.0); C = 32; DS = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class OccData(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        imgs = torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0
        imgs = (imgs - MEAN) / STD
        return (imgs, torch.from_numpy(d["intrins"]).float(),
                torch.from_numpy(d["cam2ego"]).float(), torch.from_numpy(d["occ"]).float())


class LiftSplat3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.dbins = torch.arange(*DBOUND); self.D = len(self.dbins)
        _pre = os.environ.get("SCRATCH", "0") != "1"
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if _pre else None)
        self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2, bb.layer3)
        self.depthnet = nn.Conv2d(256, self.D + C, 1)
        self.head = nn.Sequential(
            nn.Conv3d(C, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32, 1, 1))
        self.register_buffer("frustum", self._frustum())

    def _frustum(self):
        fH, fW = H // DS, W // DS
        ds = self.dbins.view(-1, 1, 1).expand(self.D, fH, fW)
        xs = torch.linspace(0, W - 1, fW).view(1, 1, fW).expand(self.D, fH, fW)
        ys = torch.linspace(0, H - 1, fH).view(1, fH, 1).expand(self.D, fH, fW)
        return torch.stack((xs, ys, ds), -1)

    def geometry(self, intrins, cam2ego):
        B, n = intrins.shape[:2]
        fr = self.frustum.to(intrins.device)
        pts = torch.cat((fr[..., :2] * fr[..., 2:3], fr[..., 2:3]), -1)
        pts = pts.view(1, 1, *pts.shape).expand(B, n, *pts.shape).unsqueeze(-1)
        Kinv = torch.inverse(intrins).view(B, n, 1, 1, 1, 3, 3)
        cam = (Kinv @ pts).squeeze(-1)
        R = cam2ego[..., :3, :3].view(B, n, 1, 1, 1, 3, 3)
        t = cam2ego[..., :3, 3].view(B, n, 1, 1, 1, 3)
        return (R @ cam.unsqueeze(-1)).squeeze(-1) + t

    def voxel_pool(self, geom, feat):
        B = geom.shape[0]
        gx = ((geom[..., 0] - XB[0]) / RES).long()
        gy = ((geom[..., 1] - YB[0]) / RES).long()
        gz = ((geom[..., 2] - ZB[0]) / RES).long()
        keep = (gx >= 0) & (gx < XG) & (gy >= 0) & (gy < YG) & (gz >= 0) & (gz < ZG)
        out = geom.new_zeros(B, C, XG * YG * ZG)
        feat = feat.reshape(B, -1, C)
        gx, gy, gz, keep = gx.reshape(B, -1), gy.reshape(B, -1), gz.reshape(B, -1), keep.reshape(B, -1)
        for b in range(B):
            k = keep[b]
            idx = (gx[b, k] * YG + gy[b, k]) * ZG + gz[b, k]
            out[b].index_add_(1, idx, feat[b, k].t())
        return out.view(B, C, XG, YG, ZG)

    def forward(self, imgs, intrins, cam2ego):
        B, n = imgs.shape[:2]
        x = self.depthnet(self.backbone(imgs.view(B * n, *imgs.shape[2:])))
        depth = x[:, :self.D].softmax(1); ctx = x[:, self.D:]
        feat = depth.unsqueeze(2) * ctx.unsqueeze(1)
        fH, fW = feat.shape[-2:]
        feat = feat.view(B, n, self.D, C, fH, fW).permute(0, 1, 2, 4, 5, 3)
        vox = self.voxel_pool(self.geometry(intrins, cam2ego), feat)
        return self.head(vox).squeeze(1)                       # (B, XG, YG, ZG) logits


def build_model():
    return LiftSplat3D()


def iou(logits, gt, thr=0.0):
    p = logits > thr; g = gt > 0.5
    inter = (p & g).sum().item(); union = (p | g).sum().item()
    return inter / union if union > 0 else float("nan")


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    tr = torch.utils.data.DataLoader(OccData(root, "train"), batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(OccData(root, "val"), batch_size=4, num_workers=4)
    model = LiftSplat3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=epochs, steps_per_epoch=len(tr))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=dev))   # voxels are very sparse
    best = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, K, c2e, occ in tr:
            imgs, K, c2e, occ = imgs.to(dev), K.to(dev), c2e.to(dev), occ.to(dev)
            loss = lossf(model(imgs, K, c2e), occ)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval(); ious = []
        with torch.no_grad():
            for imgs, K, c2e, occ in va:
                imgs, K, c2e, occ = imgs.to(dev), K.to(dev), c2e.to(dev), occ.to(dev)
                logits = model(imgs, K, c2e)
                for b in range(imgs.shape[0]):
                    ious.append(iou(logits[b], occ[b]))
        miou = float(np.nanmean(ious))
        if miou > best:
            best = miou; torch.save(model.state_dict(), out)
        print(f"  epoch {ep:02d} loss {loss.item():.3f} val_IoU {miou:.4f} (best {best:.4f})", flush=True)
    print(f"OCC_DONE seed={seed} best_val_IoU={best:.4f} -> {out}", flush=True)
