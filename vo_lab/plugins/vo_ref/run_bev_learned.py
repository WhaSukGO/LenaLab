"""Reference LEARNED BEV model (the known-good baseline + bar for the BEV Track-B domain).

Self-contained main.py for the sandbox: reads $LAB_DATA/train/<token>.npz (6 surround cams +
intrinsics + cam->ego extrinsics + bev occupancy GT), trains a Lift-Splat-Shoot network FROM
SCRATCH (no pretrained weights -- the sandbox has no network), then reads
$LAB_DATA/test_input/<token>.npz (cams + calib, NO GT), predicts a 200x200 binary vehicle-
occupancy mask, and writes $LAB_ARTIFACTS/pred_<token>.npy. Graded by held-out IoU (eval_bev.py).

Geometry uses the real scaled intrinsics + cam->ego extrinsics. Fixed task grid: ego frame,
100m x 100m @ 0.5m -> 200x200, x forward / y left, depth bins 4..45m. Pure torch + numpy.
"""
import os, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, N = 128, 352, 6
XG = YG = 200
XB = YB = (-50.0, 50.0)
RES = 0.5
DBOUND = (4.0, 45.0, 1.0)
C = 64
DS = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load(npz, with_gt):
    d = np.load(npz)
    imgs = torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0
    imgs = (imgs - MEAN) / STD
    out = [imgs, torch.from_numpy(d["intrins"]).float(), torch.from_numpy(d["cam2ego"]).float()]
    if with_gt:
        out.append(torch.from_numpy(d["bev"]).float())
    return out


class LiftSplat(nn.Module):
    def __init__(self):
        super().__init__()
        self.dbins = torch.arange(*DBOUND)
        self.D = len(self.dbins)
        bb = torchvision.models.resnet18(weights=None)             # FROM SCRATCH (no download)
        self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                                      bb.layer1, bb.layer2, bb.layer3)  # -> /16, 256 ch
        self.depthnet = nn.Conv2d(256, self.D + C, 1)
        self.bevenc = nn.Sequential(
            nn.Conv2d(C, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 1, 1))
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
        keep = (gx >= 0) & (gx < XG) & (gy >= 0) & (gy < YG)
        out = geom.new_zeros(B, C, XG * YG)
        feat = feat.reshape(B, -1, C)
        gx, gy, keep = gx.reshape(B, -1), gy.reshape(B, -1), keep.reshape(B, -1)
        for b in range(B):
            k = keep[b]
            out[b].index_add_(1, gx[b, k] * YG + gy[b, k], feat[b, k].t())
        return out.view(B, C, XG, YG)

    def forward(self, imgs, intrins, cam2ego):
        B, n = imgs.shape[:2]
        x = self.depthnet(self.backbone(imgs.view(B * n, *imgs.shape[2:])))
        depth = x[:, :self.D].softmax(1)
        ctx = x[:, self.D:]
        feat = depth.unsqueeze(2) * ctx.unsqueeze(1)
        fH, fW = feat.shape[-2:]
        feat = feat.view(B, n, self.D, C, fH, fW).permute(0, 1, 2, 4, 5, 3)
        bev = self.voxel_pool(self.geometry(intrins, cam2ego), feat)
        return self.bevenc(bev).squeeze(1)


def main():
    data = os.environ["LAB_DATA"]
    art = os.environ["LAB_ARTIFACTS"]
    os.makedirs(art, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    train_files = sorted(glob.glob(os.path.join(data, "train", "*.npz")))
    test_files = sorted(glob.glob(os.path.join(data, "test_input", "*.npz")))
    print(f"train={len(train_files)} test={len(test_files)} dev={DEV}", flush=True)

    model = LiftSplat().to(DEV)
    EP, BS = 24, 4
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    steps = (len(train_files) + BS - 1) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EP, steps_per_epoch=steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=DEV))
    for ep in range(EP):
        model.train()
        order = np.random.permutation(len(train_files))
        for i in range(0, len(order), BS):
            batch = [load(train_files[j], True) for j in order[i:i + BS]]
            imgs = torch.stack([b[0] for b in batch]).to(DEV)
            K = torch.stack([b[1] for b in batch]).to(DEV)
            c2e = torch.stack([b[2] for b in batch]).to(DEV)
            gt = torch.stack([b[3] for b in batch]).to(DEV)
            loss = lossf(model(imgs, K, c2e), gt)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"  epoch {ep:02d} loss {loss.item():.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        for f in test_files:
            tok = os.path.splitext(os.path.basename(f))[0]
            imgs, K, c2e = load(f, False)
            logits = model(imgs.unsqueeze(0).to(DEV), K.unsqueeze(0).to(DEV), c2e.unsqueeze(0).to(DEV))[0]
            pred = (logits > 0.0).cpu().numpy().astype(np.uint8)        # binary occupancy
            np.save(os.path.join(art, f"pred_{tok}.npy"), pred)
    print(f"wrote {len(test_files)} predictions", flush=True)


if __name__ == "__main__":
    main()
