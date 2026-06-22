"""Scene-agnostic IPM reference for CROSS-SPACE smart-space occupancy: train on several warehouses,
test on an UNSEEN one. Camera-count-agnostic (per-sample cam_proj + cam_valid mask, masked-mean over
real cameras only) and world-frame-agnostic (per-sample grid_bounds on a canonical 224x224 grid).

SAVES a checkpoint you can reuse or train more from:
  <out>.pt        best model state_dict  (for inference: build_model(); load_state_dict)
  <out>.full.pt   {model, optimizer, scheduler, epoch, best, config}  (for --resume / train-more)

usage: python smartspace_xspace_ref.py <train_dirs_comma_sep> <val_dir> <out.pt> [epochs] [seed] [resume.full.pt]
"""
import sys, os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

H, W = 128, 352
RES = 0.5; XG = YG = 224
ZPLANES = [0.0, 0.8, 1.6]; C = 64; DS = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SceneData(torch.utils.data.Dataset):
    def __init__(self, dirs):
        self.files = []
        for d in dirs:
            self.files += sorted(glob.glob(os.path.join(d, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        imgs = torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0
        imgs = (imgs - MEAN) / STD
        return (imgs, torch.from_numpy(d["cam_proj"]).float(), torch.from_numpy(d["cam_valid"]).float(),
                torch.from_numpy(d["grid_bounds"]).float(), torch.from_numpy(d["bev"]).float())


class IPMBevX(nn.Module):
    def __init__(self):
        super().__init__()
        _pre = os.environ.get("SCRATCH", "0") != "1"
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if _pre else None)
        self.backbone = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1, bb.layer2, bb.layer3)
        self.reduce = nn.Conv2d(256, C, 1)
        self.head = nn.Sequential(nn.Conv2d(C * len(ZPLANES), 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
                                  nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Conv2d(128, 1, 1))

    def ipm_grid(self, cam_proj, gb):                 # cam_proj (N,3,4), gb (4,) -> (P,N,XG,YG,2),(P,N,XG,YG)
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
            grids.append(torch.stack([gu, gv], -1))
            masks.append(((uv[..., 2] > 0.1) & (gu.abs() <= 1) & (gv.abs() <= 1)).float())
        return torch.stack(grids), torch.stack(masks)

    def forward(self, imgs, cam_proj, cam_valid, grid_bounds):
        B, N = imgs.shape[:2]
        f = self.reduce(self.backbone(imgs.view(B * N, *imgs.shape[2:]))).view(B, N, C, -1, W // DS)
        fH = f.shape[3]; f = f.view(B, N, C, fH, W // DS)
        outs = []
        for b in range(B):                            # per-sample geometry (scenes differ in a batch)
            grids, masks = self.ipm_grid(cam_proj[b], grid_bounds[b])
            masks = masks * cam_valid[b].view(1, N, 1, 1)   # ignore padded cameras
            planes = []
            for p in range(grids.shape[0]):
                samp = F.grid_sample(f[b], grids[p], align_corners=False, padding_mode="zeros")  # (N,C,XG,YG)
                mk = masks[p].view(N, 1, XG, YG)
                planes.append((samp * mk).sum(0) / mk.sum(0).clamp(min=1.0))
            outs.append(self.head(torch.cat(planes, 0).unsqueeze(0)))
        return torch.cat(outs, 0).squeeze(1)          # (B,XG,YG)


def build_model():
    return IPMBevX()


def iou(logits, gt, thr=0.0):
    p = logits > thr; g = gt > 0.5
    u = (p | g).sum().item()
    return (p & g).sum().item() / u if u else float("nan")


def evaluate(model, loader, dev):
    model.eval(); ious = []
    with torch.no_grad():
        for imgs, cp, cv, gb, bev in loader:
            imgs, cp, cv, gb, bev = imgs.to(dev), cp.to(dev), cv.to(dev), gb.to(dev), bev.to(dev)
            lo = model(imgs, cp, cv, gb)
            for b in range(imgs.shape[0]):
                ious.append(iou(lo[b], bev[b]))
    return float(np.nanmean(ious))


if __name__ == "__main__":
    train_dirs = sys.argv[1].split(","); val_dir = sys.argv[2]; out = sys.argv[3]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    resume = sys.argv[6] if len(sys.argv) > 6 else None
    torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
    bs = int(os.environ.get("SMARTSPACE_BS", "2"))
    tr = torch.utils.data.DataLoader(SceneData(train_dirs), batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(SceneData([val_dir]), batch_size=bs, num_workers=4)
    model = IPMBevX().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=epochs, steps_per_epoch=max(len(tr), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(40.0, device=dev))
    start_ep, best = 0, 0.0
    if resume and os.path.exists(resume):
        st = torch.load(resume, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"]); start_ep = st["epoch"] + 1; best = st["best"]
        print(f"resumed from {resume} @ epoch {start_ep} (best {best:.4f})", flush=True)
    print(f"train {len(tr.dataset)} ({len(train_dirs)} scenes) | val {len(va.dataset)} (unseen) | dev {dev}", flush=True)
    for ep in range(start_ep, epochs):
        model.train()
        for imgs, cp, cv, gb, bev in tr:
            imgs, cp, cv, gb, bev = imgs.to(dev), cp.to(dev), cv.to(dev), gb.to(dev), bev.to(dev)
            loss = lossf(model(imgs, cp, cv, gb), bev)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        miou = evaluate(model, va, dev)
        if miou > best:
            best = miou; torch.save(model.state_dict(), out)            # best weights (reuse/inference)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": ep, "best": best, "config": {"XG": XG, "YG": YG, "C": C, "ZPLANES": ZPLANES}},
                   out + ".full.pt")                                    # resume / train-more
        print(f"  epoch {ep:02d} loss {loss.item():.3f} val_IoU(unseen) {miou:.4f} (best {best:.4f})", flush=True)
    print(f"XSPACE_DONE seed={seed} best_unseen_IoU={best:.4f} -> {out} (+ {out}.full.pt)", flush=True)
