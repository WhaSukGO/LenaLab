"""Reference LEARNED smart-space floor-occupancy model (the Track-B baseline + bar). Self-contained
main.py for the sandbox: reads $LAB_DATA/train/<token>.npz (N static cams + cam_proj + bev GT), trains
an IPM net FROM SCRATCH, then for each $LAB_DATA/test_input/<token>.npz predicts an XG*YG binary floor
mask -> $LAB_ARTIFACTS/pred_<token>.npy. Graded by held-out floor IoU (per-space self-verification)."""
import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W = 128, 352
RES = 0.5
ZPLANES = [0.0, 0.8, 1.6]
C = 64; DS = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
XG, YG = (int(v) for v in os.environ.get("SMARTSPACE_GRID", "206,203").split(","))


def load(npz, gt):
    d = np.load(npz)
    imgs = (torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0 - MEAN) / STD
    out = [imgs, torch.from_numpy(d["cam_proj"]).float(), torch.from_numpy(d["grid_bounds"]).float()]
    if gt:
        out.append(torch.from_numpy(d["bev"]).float())
    return out


class IPMBev(nn.Module):
    def __init__(self):
        super().__init__()
        bb = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2, bb.layer3)
        self.reduce = nn.Conv2d(256, C, 1)
        self.head = nn.Sequential(nn.Conv2d(C * len(ZPLANES), 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, 1, 1))

    def ipm_grid(self, cam_proj, gb):
        N = cam_proj.shape[0]; dev = cam_proj.device
        xs = gb[0] + (torch.arange(XG, device=dev) + 0.5) * RES
        ys = gb[2] + (torch.arange(YG, device=dev) + 0.5) * RES
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        grids, masks = [], []
        for z in ZPLANES:
            pts = torch.stack([X, Y, torch.full_like(X, z), torch.ones_like(X)], -1)
            uv = torch.einsum("nij,xyj->nxyi", cam_proj, pts)
            zc = uv[..., 2].clamp(min=1e-6)
            gu = (uv[..., 0] / zc) / (W - 1) * 2 - 1; gv = (uv[..., 1] / zc) / (H - 1) * 2 - 1
            valid = (uv[..., 2] > 0.1) & (gu.abs() <= 1) & (gv.abs() <= 1)
            grids.append(torch.stack([gu, gv], -1)); masks.append(valid.float())
        return torch.stack(grids), torch.stack(masks)

    def forward(self, imgs, cam_proj, grid_bounds):
        B, N = imgs.shape[:2]
        f = self.reduce(self.backbone(imgs.view(B * N, *imgs.shape[2:])))
        f = f.view(B, N, C, *f.shape[-2:])
        grids, masks = self.ipm_grid(cam_proj[0], grid_bounds[0])
        planes = []
        for p in range(grids.shape[0]):
            gg = grids[p].unsqueeze(0).expand(B, N, XG, YG, 2).reshape(B * N, XG, YG, 2)
            samp = F.grid_sample(f.reshape(B * N, C, *f.shape[-2:]), gg, align_corners=False,
                                 padding_mode="zeros").view(B, N, C, XG, YG)
            mk = masks[p].view(1, N, 1, XG, YG)
            planes.append((samp * mk).sum(1) / mk.sum(1).clamp(min=1.0))
        return self.head(torch.cat(planes, 1)).squeeze(1)


def build_model():
    return IPMBev()


def main():
    data = os.environ["LAB_DATA"]; art = os.environ["LAB_ARTIFACTS"]; os.makedirs(art, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    train = sorted(glob.glob(os.path.join(data, "train", "*.npz")))
    test = sorted(glob.glob(os.path.join(data, "test_input", "*.npz")))
    print(f"train={len(train)} test={len(test)} dev={DEV} grid={XG}x{YG}", flush=True)
    m = IPMBev().to(DEV); EP, BS = 24, 4
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=EP, steps_per_epoch=(len(train) + BS - 1) // BS)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30.0, device=DEV))
    for ep in range(EP):
        m.train(); order = np.random.permutation(len(train))
        for i in range(0, len(order), BS):
            b = [load(train[j], True) for j in order[i:i + BS]]
            imgs = torch.stack([x[0] for x in b]).to(DEV); cp = torch.stack([x[1] for x in b]).to(DEV)
            gb = torch.stack([x[2] for x in b]).to(DEV); bev = torch.stack([x[3] for x in b]).to(DEV)
            loss = lossf(m(imgs, cp, gb), bev)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"  epoch {ep:02d} loss {loss.item():.3f}", flush=True)
    m.eval()
    with torch.no_grad():
        for f in test:
            tok = os.path.splitext(os.path.basename(f))[0]
            imgs, cp, gb = load(f, False)
            logits = m(imgs.unsqueeze(0).to(DEV), cp.unsqueeze(0).to(DEV), gb.unsqueeze(0).to(DEV))[0]
            np.save(os.path.join(art, f"pred_{tok}.npy"), (logits > 0.0).cpu().numpy().astype(np.uint8))
    print(f"wrote {len(test)} predictions", flush=True)


if __name__ == "__main__":
    main()
