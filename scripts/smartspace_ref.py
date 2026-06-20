"""IPM (Inverse Perspective Mapping) BEV floor-occupancy reference + variance trainer.

The geometry-appropriate baseline for STATIC overhead cameras + floor occupancy (Phase-2 finding: the
ego-vehicle lift-splat scatters its learned-depth frustum thinly across a big warehouse -- only ~3% of
points land in-grid and it learns no spatial signal). Instead, for each floor cell (X,Y) at a few height
planes z, project the world point to each camera via P = K @ inv(cam2world), bilinearly sample that
camera's feature map (grid_sample), masked-mean over the cameras that see the cell, concat the planes,
and a 2D conv head predicts per-cell occupancy. Dense, correct floor coverage. SCRATCH=1 -> from scratch.

The projection is constant per (static) scene, so it is built from the batch's shared calibration.
usage: python smartspace_ref.py <smartspace_occ_root> <out.pt> [epochs] [seed]
"""
import os, glob, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

H, W = 128, 352
RES = 0.5
ZPLANES = [0.0, 0.8, 1.6]                                   # height planes (m): floor, mid-body, head
C = 64; DS = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
# grid SIZE fixed for the head/batching; world bounds are per-sample. Defaults match Warehouse_000.
XG, YG = (int(v) for v in os.environ.get("SMARTSPACE_GRID", "206,203").split(","))


class SmartSpaceData(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.files = sorted(glob.glob(os.path.join(root, split, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        imgs = torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0
        imgs = (imgs - MEAN) / STD
        return (imgs, torch.from_numpy(d["intrins"]).float(),
                torch.from_numpy(d["cam_proj"]).float(),
                torch.from_numpy(d["grid_bounds"]).float(),
                torch.from_numpy(d["bev"]).float())


class IPMBev(nn.Module):
    def __init__(self):
        super().__init__()
        _pre = os.environ.get("SCRATCH", "0") != "1"
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if _pre else None)
        self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2, bb.layer3)
        self.reduce = nn.Conv2d(256, C, 1)
        self.head = nn.Sequential(
            nn.Conv2d(C * len(ZPLANES), 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 1, 1))

    def ipm_grid(self, cam_proj, gb):
        """Per height plane: normalized sampling grid (N,XG,YG,2) + validity mask (N,XG,YG).
        Projects floor cells straight through cam_proj (verified world->image); static cameras ->
        built from the scene's shared calibration."""
        N = cam_proj.shape[0]; dev = cam_proj.device
        xs = gb[0] + (torch.arange(XG, device=dev) + 0.5) * RES
        ys = gb[2] + (torch.arange(YG, device=dev) + 0.5) * RES
        X, Y = torch.meshgrid(xs, ys, indexing="ij")        # (XG,YG)
        grids, masks = [], []
        for z in ZPLANES:
            pts = torch.stack([X, Y, torch.full_like(X, z), torch.ones_like(X)], -1)   # (XG,YG,4)
            uv = torch.einsum("nij,xyj->nxyi", cam_proj, pts)                          # (N,XG,YG,3)
            zc = uv[..., 2].clamp(min=1e-6)
            gu = (uv[..., 0] / zc) / (W - 1) * 2 - 1
            gv = (uv[..., 1] / zc) / (H - 1) * 2 - 1
            valid = (uv[..., 2] > 0.1) & (gu.abs() <= 1) & (gv.abs() <= 1)
            grids.append(torch.stack([gu, gv], -1)); masks.append(valid.float())
        return torch.stack(grids), torch.stack(masks)       # (P,N,XG,YG,2), (P,N,XG,YG)

    def forward(self, imgs, intrins, cam_proj, grid_bounds):
        B, N = imgs.shape[:2]
        f = self.reduce(self.backbone(imgs.view(B * N, *imgs.shape[2:])))  # (B*N,C,fH,fW)
        f = f.view(B, N, C, *f.shape[-2:])
        grids, masks = self.ipm_grid(cam_proj[0], grid_bounds[0])  # scene-shared calib
        P = grids.shape[0]
        planes = []
        for p in range(P):
            gg = grids[p].unsqueeze(0).expand(B, N, XG, YG, 2).reshape(B * N, XG, YG, 2)
            samp = F.grid_sample(f.reshape(B * N, C, *f.shape[-2:]), gg,
                                 align_corners=False, padding_mode="zeros").view(B, N, C, XG, YG)
            mk = masks[p].view(1, N, 1, XG, YG)
            planes.append((samp * mk).sum(1) / mk.sum(1).clamp(min=1.0))   # masked mean over cameras
        return self.head(torch.cat(planes, 1)).squeeze(1)   # (B, XG, YG) logits


def build_model():
    return IPMBev()


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
    bs = int(os.environ.get("SMARTSPACE_BS", "4"))
    tr = torch.utils.data.DataLoader(SmartSpaceData(root, "train"), batch_size=bs, shuffle=True,
                                     num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(SmartSpaceData(root, "val"), batch_size=bs, num_workers=4)
    model = IPMBev().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=epochs, steps_per_epoch=len(tr))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30.0, device=dev))
    best = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, K, cp, gb, bev in tr:
            imgs, K, cp, gb, bev = imgs.to(dev), K.to(dev), cp.to(dev), gb.to(dev), bev.to(dev)
            loss = lossf(model(imgs, K, cp, gb), bev)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval(); ious = []
        with torch.no_grad():
            for imgs, K, cp, gb, bev in va:
                imgs, K, cp, gb, bev = imgs.to(dev), K.to(dev), cp.to(dev), gb.to(dev), bev.to(dev)
                logits = model(imgs, K, cp, gb)
                for b in range(imgs.shape[0]):
                    ious.append(iou(logits[b], bev[b]))
        miou = float(np.nanmean(ious))
        if miou > best:
            best = miou; torch.save(model.state_dict(), out)
        print(f"  epoch {ep:02d} loss {loss.item():.3f} val_IoU {miou:.4f} (best {best:.4f})", flush=True)
    print(f"SMARTSPACE_DONE seed={seed} best_val_IoU={best:.4f} -> {out}", flush=True)
