"""Reference LEARNED 3D-occupancy model (the occupancy Track-B baseline + bar). Self-contained
main.py for the sandbox: reads $LAB_DATA/train/<token>.npz (6 cams + calib + occ voxel GT), trains
a Lift-Splat-to-3D net FROM SCRATCH, then for each $LAB_DATA/test_input/<token>.npz predicts a
XG*YG*ZG binary voxel mask -> $LAB_ARTIFACTS/pred_<token>.npy. Graded by held-out voxel IoU."""
import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, N = 128, 352, 6
XB = YB = (-50.0, 50.0); ZB = (-2.0, 4.0); RES = 0.5
XG = int((XB[1] - XB[0]) / RES); YG = int((YB[1] - YB[0]) / RES); ZG = int((ZB[1] - ZB[0]) / RES)
DBOUND = (4.0, 45.0, 1.0); C = 32; DS = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load(npz, gt):
    d = np.load(npz)
    imgs = (torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0 - MEAN) / STD
    out = [imgs, torch.from_numpy(d["intrins"]).float(), torch.from_numpy(d["cam2ego"]).float()]
    if gt:
        out.append(torch.from_numpy(d["occ"]).float())
    return out


class LiftSplat3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.dbins = torch.arange(*DBOUND); self.D = len(self.dbins)
        bb = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2, bb.layer3)
        self.depthnet = nn.Conv2d(256, self.D + C, 1)
        self.head = nn.Sequential(nn.Conv3d(C, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(True),
                                  nn.Conv3d(32, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(True), nn.Conv3d(32, 1, 1))
        self.register_buffer("frustum", self._frustum())

    def _frustum(self):
        fH, fW = H // DS, W // DS
        ds = self.dbins.view(-1, 1, 1).expand(self.D, fH, fW)
        xs = torch.linspace(0, W - 1, fW).view(1, 1, fW).expand(self.D, fH, fW)
        ys = torch.linspace(0, H - 1, fH).view(1, fH, 1).expand(self.D, fH, fW)
        return torch.stack((xs, ys, ds), -1)

    def geometry(self, K, c2e):
        B, n = K.shape[:2]; fr = self.frustum.to(K.device)
        pts = torch.cat((fr[..., :2] * fr[..., 2:3], fr[..., 2:3]), -1)
        pts = pts.view(1, 1, *pts.shape).expand(B, n, *pts.shape).unsqueeze(-1)
        cam = (torch.inverse(K).view(B, n, 1, 1, 1, 3, 3) @ pts).squeeze(-1)
        R = c2e[..., :3, :3].view(B, n, 1, 1, 1, 3, 3); t = c2e[..., :3, 3].view(B, n, 1, 1, 1, 3)
        return (R @ cam.unsqueeze(-1)).squeeze(-1) + t

    def voxel_pool(self, geom, feat):
        B = geom.shape[0]
        gx = ((geom[..., 0] - XB[0]) / RES).long(); gy = ((geom[..., 1] - YB[0]) / RES).long()
        gz = ((geom[..., 2] - ZB[0]) / RES).long()
        keep = (gx >= 0) & (gx < XG) & (gy >= 0) & (gy < YG) & (gz >= 0) & (gz < ZG)
        out = geom.new_zeros(B, C, XG * YG * ZG); feat = feat.reshape(B, -1, C)
        gx, gy, gz, keep = gx.reshape(B, -1), gy.reshape(B, -1), gz.reshape(B, -1), keep.reshape(B, -1)
        for b in range(B):
            k = keep[b]
            out[b].index_add_(1, (gx[b, k] * YG + gy[b, k]) * ZG + gz[b, k], feat[b, k].t())
        return out.view(B, C, XG, YG, ZG)

    def forward(self, imgs, K, c2e):
        B, n = imgs.shape[:2]
        x = self.depthnet(self.backbone(imgs.view(B * n, *imgs.shape[2:])))
        depth = x[:, :self.D].softmax(1); ctx = x[:, self.D:]
        feat = (depth.unsqueeze(2) * ctx.unsqueeze(1))
        fH, fW = feat.shape[-2:]
        feat = feat.view(B, n, self.D, C, fH, fW).permute(0, 1, 2, 4, 5, 3)
        return self.head(self.voxel_pool(self.geometry(K, c2e), feat)).squeeze(1)


def build_model():
    return LiftSplat3D()


def main():
    data = os.environ["LAB_DATA"]; art = os.environ["LAB_ARTIFACTS"]; os.makedirs(art, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    train = sorted(glob.glob(os.path.join(data, "train", "*.npz")))
    test = sorted(glob.glob(os.path.join(data, "test_input", "*.npz")))
    print(f"train={len(train)} test={len(test)} dev={DEV} grid={XG}x{YG}x{ZG}", flush=True)
    m = LiftSplat3D().to(DEV); EP, BS = 24, 4
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EP, steps_per_epoch=(len(train) + BS - 1) // BS)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=DEV))
    for ep in range(EP):
        m.train(); order = np.random.permutation(len(train))
        for i in range(0, len(order), BS):
            b = [load(train[j], True) for j in order[i:i + BS]]
            imgs = torch.stack([x[0] for x in b]).to(DEV); K = torch.stack([x[1] for x in b]).to(DEV)
            c2e = torch.stack([x[2] for x in b]).to(DEV); occ = torch.stack([x[3] for x in b]).to(DEV)
            loss = lossf(m(imgs, K, c2e), occ)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"  epoch {ep:02d} loss {loss.item():.3f}", flush=True)
    m.eval()
    with torch.no_grad():
        for f in test:
            tok = os.path.splitext(os.path.basename(f))[0]
            imgs, K, c2e = load(f, False)
            logits = m(imgs.unsqueeze(0).to(DEV), K.unsqueeze(0).to(DEV), c2e.unsqueeze(0).to(DEV))[0]
            np.save(os.path.join(art, f"pred_{tok}.npy"), (logits > 0.0).cpu().numpy().astype(np.uint8))
    print(f"wrote {len(test)} predictions", flush=True)


if __name__ == "__main__":
    main()
