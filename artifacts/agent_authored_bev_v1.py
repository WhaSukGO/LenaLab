#!/usr/bin/env python3
"""
Multi-camera Bird's-Eye-View vehicle occupancy — Lift-Splat-Shoot.

Improvements over v1:
  * Horizontal-flip surround-camera augmentation (correct camera swap + extrinsic update)
  * Dropout2d in BEV head for regularisation
  * Threshold calibration via training-set IoU sweep after training
  * 60 epochs with OneCycleLR
"""

import os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models

# ─── Paths & device ───────────────────────────────────────────────────────────
LAB_DATA      = os.environ.get("LAB_DATA",      "/data")
LAB_ARTIFACTS = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(LAB_ARTIFACTS, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── BEV grid ─────────────────────────────────────────────────────────────────
BEV_XMIN = BEV_YMIN = -50.0   # metres
BEV_RES  = 0.5                 # metres/cell
BEV_H = BEV_W = 200            # rows = x-axis (forward), cols = y-axis (left)

# ─── Depth discretisation ─────────────────────────────────────────────────────
D_MIN, D_MAX, D_BINS = 1.0, 50.0, 41

# ─── Hyperparameters ──────────────────────────────────────────────────────────
CTX_C      = 64      # feature channels per frustum point
BATCH      = 4
LR         = 2e-3
EPOCHS     = 60
POS_WEIGHT = 15.0    # BCE pos_weight (vehicles ~2 % of BEV)

# Horizontal-flip: surround camera re-ordering
# [FL=0, F=1, FR=2, BL=3, B=4, BR=5] -> [FR=2, F=1, FL=0, BR=5, B=4, BL=3]
_HFLIP_CAM_ORDER = [2, 1, 0, 5, 4, 3]

# 4×4 matrices to flip camera-x and ego-y
# E_new = flip_ego @ E @ flip_cam   (cam→ego extrinsic update)
_FLIP_CAM = torch.diag(torch.tensor([-1., 1., 1., 1.]))   # x-flip in camera frame
_FLIP_EGO = torch.diag(torch.tensor([ 1.,-1., 1., 1.]))   # y-flip in ego frame


def hflip_sample(imgs, K, E, bev=None):
    """
    Apply horizontal flip to one surround-camera sample.
    imgs : (6, 3, H, W)
    K    : (6, 3, 3)
    E    : (6, 4, 4)
    bev  : (200, 200) or None
    """
    W = imgs.shape[-1]

    # 1) Swap left/right cameras
    imgs = imgs[_HFLIP_CAM_ORDER]
    K    = K[_HFLIP_CAM_ORDER].clone()
    E    = E[_HFLIP_CAM_ORDER].clone()

    # 2) Flip each image left-right
    imgs = torch.flip(imgs, dims=[-1])

    # 3) Update intrinsics: cx -> W - cx  (horizontal principal point)
    K[:, 0, 2] = W - K[:, 0, 2]

    # 4) Update extrinsics: E_new = flip_ego @ E @ flip_cam
    fc = _FLIP_CAM.to(E)           # (4,4)
    fe = _FLIP_EGO.to(E)           # (4,4)
    E  = (fe @ E.view(-1, 4, 4) @ fc).view(6, 4, 4)

    # 5) Flip BEV y-axis (columns)
    if bev is not None:
        bev = torch.flip(bev, dims=[-1])

    return imgs, K, E, bev


# ─── Dataset ──────────────────────────────────────────────────────────────────
class BEVSet(Dataset):
    def __init__(self, root, train=True):
        self.fs    = sorted(glob.glob(os.path.join(root, "*.npz")))
        self.train = train

    def __len__(self): return len(self.fs)

    def __getitem__(self, i):
        d    = np.load(self.fs[i])
        imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        K    = torch.from_numpy(d["intrins"])
        E    = torch.from_numpy(d["cam2ego"])
        tok  = os.path.splitext(os.path.basename(self.fs[i]))[0]

        if self.train:
            bev = torch.from_numpy(d["bev"].astype(np.float32))
            # Random horizontal flip (50 %)
            if torch.rand(1).item() > 0.5:
                imgs, K, E, bev = hflip_sample(imgs, K, E, bev)
            return imgs, K, E, bev, tok
        return imgs, K, E, tok


# ─── Model ────────────────────────────────────────────────────────────────────
class BEVNet(nn.Module):
    """Lift-Splat-Shoot BEV occupancy network."""

    def __init__(self):
        super().__init__()
        self.register_buffer("dvals", torch.linspace(D_MIN, D_MAX, D_BINS))

        # ResNet-18 up to layer2  →  (B*N, 128, H/8, W/8)
        r = models.resnet18(weights=None)
        self.enc = nn.Sequential(
            r.conv1, r.bn1, r.relu, r.maxpool,
            r.layer1, r.layer2,
        )

        # Depth distribution + context
        self.dhead = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, D_BINS + CTX_C, 1),
        )

        # BEV segmentation head with dropout
        self.bhead = nn.Sequential(
            nn.Conv2d(CTX_C, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(0.1),
            nn.Conv2d(256,   256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(0.1),
            nn.Conv2d(256,   128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,     1, 1),
        )

    def forward(self, imgs, K, E):
        B, N, C, iH, iW = imgs.shape

        # ImageNet normalisation
        mn = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        sd = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        imgs = (imgs - mn) / sd

        # Backbone
        x = self.enc(imgs.view(B * N, C, iH, iW))   # (B*N, 128, fH, fW)
        _, _, fH, fW = x.shape

        # Depth distribution + context
        dc    = self.dhead(x)
        dprob = F.softmax(dc[:, :D_BINS], dim=1)    # (B*N, D, fH, fW)
        ctx   = dc[:, D_BINS:]                       # (B*N, C, fH, fW)

        # Frustum features  (B*N, D, C, fH, fW)
        frust = dprob.unsqueeze(2) * ctx.unsqueeze(1)

        # Pixel centres in original image coords
        sX = iW / fW;  sY = iH / fH
        us = (torch.arange(fW, device=x.device, dtype=torch.float32) + 0.5) * sX
        vs = (torch.arange(fH, device=x.device, dtype=torch.float32) + 0.5) * sY
        vg, ug = torch.meshgrid(vs, us, indexing="ij")

        ones = torch.ones_like(ug)
        pix  = torch.stack([ug, vg, ones], -1).view(1, 1, fH, fW, 3, 1) \
                   .expand(B, N, -1, -1, -1, -1)

        # Camera rays: K^{-1} * pix
        Kinv = torch.linalg.inv(K).view(B, N, 1, 1, 3, 3)
        rays = (Kinv @ pix).squeeze(-1).unsqueeze(2)      # (B,N,1,fH,fW,3)

        # Scale by depth bins
        d    = self.dvals.view(1, 1, D_BINS, 1, 1, 1)
        pts  = rays * d                                   # (B,N,D,fH,fW,3)

        # cam -> ego
        R = E[:, :, :3, :3].view(B, N, 1, 1, 1, 3, 3)
        t = E[:, :, :3,  3].view(B, N, 1, 1, 1, 3)
        pts_ego = (R @ pts.unsqueeze(-1)).squeeze(-1) + t  # (B,N,D,fH,fW,3)

        # BEV scatter
        ix = ((pts_ego[..., 0] - BEV_XMIN) / BEV_RES).long()
        iy = ((pts_ego[..., 1] - BEV_YMIN) / BEV_RES).long()
        z_ok = (pts_ego[..., 2] > -3.0) & (pts_ego[..., 2] < 3.0)
        vm   = (ix >= 0) & (ix < BEV_H) & (iy >= 0) & (iy < BEV_W) & z_ok

        bidx = torch.arange(B, device=x.device).view(B,1,1,1,1) \
                    .expand(B, N, D_BINS, fH, fW)
        vm1  = vm.reshape(-1)
        bb   = bidx.reshape(-1)[vm1]
        ii   = ix.reshape(-1)[vm1]
        jj   = iy.reshape(-1)[vm1]
        flat = bb * (BEV_H * BEV_W) + ii * BEV_W + jj

        ff = frust.view(B, N, D_BINS, CTX_C, fH, fW) \
                  .permute(0,1,2,4,5,3).contiguous()
        fv = ff.reshape(-1, CTX_C)[vm1]

        pool = torch.zeros(B * BEV_H * BEV_W, CTX_C,
                           device=x.device, dtype=fv.dtype)
        pool.index_add_(0, flat, fv)
        pool = pool.view(B, BEV_H, BEV_W, CTX_C) \
                   .permute(0, 3, 1, 2).contiguous()

        return self.bhead(pool.float()).squeeze(1)   # (B,200,200)


# ─── Losses ───────────────────────────────────────────────────────────────────
def dice_loss(logits, targets, eps=1e-5):
    p     = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(-1, -2))
    denom = p.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    print(f"Device : {DEVICE}")
    ds = BEVSet(os.path.join(LAB_DATA, "train"), train=True)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2,
                    pin_memory=False, drop_last=True)
    print(f"Train  : {len(ds)} samples, {len(dl)} batches/epoch")

    model   = BEVNet().to(DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Params : {nparams/1e6:.1f}M")

    opt    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
                 opt, max_lr=LR,
                 steps_per_epoch=len(dl), epochs=EPOCHS,
                 pct_start=0.05, anneal_strategy="cos")
    scaler = GradScaler()
    pw     = torch.tensor([POS_WEIGHT], device=DEVICE)
    bce    = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_loss = 1e9
    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        t0 = time.time()
        for imgs, K, E, bev, _ in dl:
            imgs = imgs.to(DEVICE); K = K.to(DEVICE)
            E    = E.to(DEVICE);   bev = bev.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            with autocast():
                logit = model(imgs, K, E)
                loss  = 0.5 * bce(logit, bev) + 0.5 * dice_loss(logit, bev)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            sched.step()
            ep_loss += loss.item()

        avg = ep_loss / len(dl)
        tag = " *" if avg < best_loss else ""
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), "/tmp/bev_best.pt")
        print(f"[{ep+1:3d}/{EPOCHS}] loss={avg:.4f}  {time.time()-t0:.1f}s{tag}")

    print(f"Best training loss: {best_loss:.4f}")
    return model


# ─── Threshold calibration on training set ────────────────────────────────────
def calibrate_threshold(model):
    """Sweep thresholds on train set; return the one maximising mean IoU."""
    model.eval()
    val_ds = BEVSet(os.path.join(LAB_DATA, "train"), train=False)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    all_probs, all_gts = [], []

    with torch.no_grad():
        for imgs, K, E, tok in val_dl:
            imgs = imgs.to(DEVICE); K = K.to(DEVICE); E = E.to(DEVICE)
            logit = model(imgs, K, E)
            prob  = torch.sigmoid(logit).squeeze(0).cpu().numpy()
            # reload GT from file
            fpath = os.path.join(LAB_DATA, "train", tok[0] + ".npz")
            gt    = np.load(fpath)["bev"].astype(np.float32)
            all_probs.append(prob)
            all_gts.append(gt)

    best_thr, best_miou = 0.40, -1.0
    for t in candidates:
        ious = []
        for prob, gt in zip(all_probs, all_gts):
            pred  = (prob > t).astype(np.float32)
            inter = (pred * gt).sum()
            union = ((pred + gt) > 0).sum()
            ious.append(inter / union if union > 0 else 0.0)
        miou = float(np.mean(ious))
        print(f"  thr={t:.2f}: mIoU={miou:.4f}")
        if miou > best_miou:
            best_miou, best_thr = miou, t

    print(f"  => best threshold = {best_thr}  (train mIoU = {best_miou:.4f})")
    return best_thr


# ─── Inference ────────────────────────────────────────────────────────────────
def infer(model, thr):
    model.eval()
    ds = BEVSet(os.path.join(LAB_DATA, "test_input"), train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Test   : {len(ds)} samples  threshold={thr}")

    with torch.no_grad():
        for imgs, K, E, toks in dl:
            imgs = imgs.to(DEVICE); K = K.to(DEVICE); E = E.to(DEVICE)
            logit = model(imgs, K, E)
            prob  = torch.sigmoid(logit).squeeze(0).cpu().numpy()
            pred  = (prob > thr).astype(np.uint8)
            np.save(os.path.join(LAB_ARTIFACTS, f"pred_{toks[0]}.npy"), pred)

    print("Inference done. Artifacts written.")


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = train()
    model.load_state_dict(torch.load("/tmp/bev_best.pt",
                                     map_location=DEVICE, weights_only=False))

    print("\nCalibrating threshold on training set...")
    best_thr = calibrate_threshold(model)

    infer(model, best_thr)
