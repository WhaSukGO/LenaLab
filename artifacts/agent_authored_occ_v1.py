#!/usr/bin/env python3
"""
Multi-camera 3D vehicle-occupancy prediction.

Architecture:
  - ResNet-18 + FPN (layer2@stride8 + layer3@stride16 upsampled) → 64 ch
  - Inverse-projection: voxel centres → cameras (FP32, avoids FP16 overflow)
  - Vectorised grid_sample, mean-pool across cameras
  - 3D conv head (HEAD_C=64) → logits

Loss:  weighted BCE (pos_weight=80) + soft-IoU
Train: stage-1 on 90% data → find threshold; stage-2 fine-tune on 100%
"""

import os, glob, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# ── constants ────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VX, VY, VZ       = 200, 200, 12
XMIN, YMIN, ZMIN = -50., -50., -2.
VSIZE            = 0.5
N_CAM            = 6
IMG_H, IMG_W     = 128, 352
FEAT_C           = 64
HEAD_C           = 64
POS_WEIGHT       = 80.0
BATCH_SIZE       = 2
N_EPOCHS         = 40        # stage-1 epochs (on 90% data)
FINETUNE_EPOCHS  = 8         # stage-2 epochs (on 100% data)
LR_MAX           = 2e-3

# ── dataset ──────────────────────────────────────────────────────────────────
class OccDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(f"{data_dir}/*.npz"))

    def __len__(self):  return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        imgs    = torch.from_numpy(d['imgs']).float() / 255.0
        imgs    = imgs.permute(0, 3, 1, 2)               # (6,3,H,W)
        intrins = torch.from_numpy(d['intrins'])
        cam2ego = torch.from_numpy(d['cam2ego'])
        occ     = torch.from_numpy(d['occ']).float()     # (200,200,12)
        token   = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return imgs, intrins, cam2ego, occ, token


# ── FPN image backbone ────────────────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """ResNet-18 with FPN: layer2 (stride 8, 128ch) + layer3 (stride 16, 256ch)
    upsampled to stride 8, then projected to FEAT_C channels."""

    def __init__(self, out_c=FEAT_C):
        super().__init__()
        r = models.resnet18(weights=None)
        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1   # stride 4,  64ch
        self.layer2 = r.layer2   # stride 8, 128ch
        self.layer3 = r.layer3   # stride 16, 256ch
        self.neck   = nn.Sequential(
            nn.Conv2d(128 + 256, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c3u = F.interpolate(c3, size=c2.shape[-2:],
                            mode='bilinear', align_corners=False)
        return self.neck(torch.cat([c2, c3u], dim=1))   # (B, out_c, H/8, W/8)


# ── voxel grid ───────────────────────────────────────────────────────────────
def build_voxel_coords() -> torch.Tensor:
    ix, iy, iz = torch.meshgrid(
        torch.arange(VX), torch.arange(VY), torch.arange(VZ), indexing='ij')
    cx = XMIN + VSIZE * (ix.float() + 0.5)
    cy = YMIN + VSIZE * (iy.float() + 0.5)
    cz = ZMIN + VSIZE * (iz.float() + 0.5)
    return torch.stack([cx, cy, cz], dim=-1).reshape(-1, 3)


# ── projection grids (FP32) ───────────────────────────────────────────────────
@torch.no_grad()
def compute_grids(vox_coords, intrins, cam2ego, H, W):
    """
    Returns:
      grids  (B, N, N_vox, 2) normalised uv ∈ [-1,1]  float32
      valids (B, N, N_vox)    bool (in-front & in-image)
    """
    B, N   = cam2ego.shape[:2]
    N_vox  = vox_coords.shape[0]
    device = cam2ego.device
    v      = vox_coords.float()          # (N_vox, 3) FP32

    grids  = torch.zeros(B, N, N_vox, 2, device=device)
    valids = torch.zeros(B, N, N_vox, dtype=torch.bool, device=device)

    for b in range(B):
        for n in range(N):
            R     = cam2ego[b, n, :3, :3].float()
            t     = cam2ego[b, n, :3,  3].float()
            K     = intrins[b, n].float()

            pts   = (v - t.unsqueeze(0)) @ R          # (N_vox, 3)
            z     = pts[:, 2]
            safe  = z.clamp(min=0.1)                  # FP32, no overflow

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            u  = (pts[:, 0] / safe * fx + cx)
            vv = (pts[:, 1] / safe * fy + cy)

            u_n  = (2.0 * u  / (W - 1) - 1.0).clamp(-1, 1)
            vv_n = (2.0 * vv / (H - 1) - 1.0).clamp(-1, 1)

            in_img = (u_n > -1) & (u_n < 1) & (vv_n > -1) & (vv_n < 1)
            valid  = (z > 0.1) & in_img

            grids[b, n, :, 0] = u_n
            grids[b, n, :, 1] = vv_n
            valids[b, n]       = valid

    return grids, valids


# ── main model ───────────────────────────────────────────────────────────────
class OccNet(nn.Module):
    def __init__(self, feat_c=FEAT_C, head_c=HEAD_C):
        super().__init__()
        self.feat_c  = feat_c
        self.encoder = ImageEncoder(out_c=feat_c)
        self.register_buffer('vox_coords', build_voxel_coords())

        in_c = feat_c + 4                # image feats + xyz + visibility
        self.head = nn.Sequential(
            nn.Conv3d(in_c,    head_c,      3, padding=1, bias=False),
            nn.BatchNorm3d(head_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_c,  head_c,      3, padding=1, bias=False),
            nn.BatchNorm3d(head_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_c,  head_c // 2, 3, padding=1, bias=False),
            nn.BatchNorm3d(head_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(head_c // 2, 1,       1),
        )

    def forward(self, imgs, intrins, cam2ego):
        B, N, C, H, W = imgs.shape
        device = imgs.device
        N_vox  = VX * VY * VZ

        # 1. Projection grids (FP32, no grad)
        grids, valids = compute_grids(self.vox_coords, intrins, cam2ego, H, W)

        # 2. Image features
        feats = self.encoder(imgs.reshape(B * N, C, H, W))     # (B*N, FC, Hf, Wf)
        FC, Hf, Wf = feats.shape[1], feats.shape[2], feats.shape[3]

        # 3. Vectorised grid_sample
        sampled = F.grid_sample(
            feats,                                              # (B*N, FC, Hf, Wf)
            grids.reshape(B * N, N_vox, 1, 2),                 # (B*N, N_vox, 1, 2)
            mode='bilinear', padding_mode='zeros', align_corners=True,
        ).squeeze(-1)                                           # (B*N, FC, N_vox)
        sampled = sampled.reshape(B, N, FC, N_vox)             # (B, N, FC, N_vox)

        # 4. Mean-pool across cameras
        val_f    = valids.float().unsqueeze(2)                  # (B, N, 1, N_vox)
        vox_feat = (sampled * val_f).sum(1)                    # (B, FC, N_vox)
        vox_cnt  = val_f.sum(1)                                # (B, 1, N_vox)
        vox_feat = (vox_feat / vox_cnt.clamp(min=1)).reshape(B, FC, VX, VY, VZ)
        vis      = (vox_cnt / N).reshape(B, 1, VX, VY, VZ)

        # 5. Coordinate channels
        gx = torch.linspace(-1, 1, VX, device=device)
        gy = torch.linspace(-1, 1, VY, device=device)
        gz = torch.linspace(-1, 1, VZ, device=device)
        cx3, cy3, cz3 = torch.meshgrid(gx, gy, gz, indexing='ij')
        coord = torch.stack([cx3, cy3, cz3], 0).unsqueeze(0).expand(B, -1, -1, -1, -1)

        feat3d = torch.cat([vox_feat, coord, vis], dim=1)     # (B, FC+4, VX, VY, VZ)

        # 6. 3D conv head
        return self.head(feat3d).squeeze(1)                    # (B, VX, VY, VZ)


# ── loss ─────────────────────────────────────────────────────────────────────
def compute_loss(pred, target, pos_weight=POS_WEIGHT):
    pw   = torch.tensor([pos_weight], device=pred.device)
    bce  = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
    p    = torch.sigmoid(pred)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = (p + target - p * target).sum(dim=(1, 2, 3))
    siou  = (inter / union.clamp(min=1e-6)).mean()
    return bce + (1.0 - siou)


# ── per-sample voxel IoU ──────────────────────────────────────────────────────
@torch.no_grad()
def voxel_iou(pred_bin, target):
    ious = []
    for i in range(pred_bin.shape[0]):
        p = pred_bin[i].float()
        g = target[i].to(pred_bin.device).float()
        inter = (p * g).sum()
        union = ((p + g) > 0).float().sum()
        ious.append((inter / union.clamp(min=1)).item())
    return ious


# ── threshold search ─────────────────────────────────────────────────────────
def find_threshold(model, val_loader):
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for imgs, intrins, cam2ego, occ, _ in val_loader:
            prob = torch.sigmoid(model(
                imgs.to(DEVICE), intrins.to(DEVICE), cam2ego.to(DEVICE)
            )).cpu()
            for i in range(prob.shape[0]):
                probs.append(prob[i].numpy())
                gts.append(occ[i].numpy())

    best_thr, best_miou = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, 0.05):
        ious = []
        for p, g in zip(probs, gts):
            pb    = (p > thr).astype(np.float32)
            inter = (pb * g).sum()
            union = ((pb + g) > 0).sum()
            ious.append(inter / max(union, 1))
        miou = float(np.mean(ious))
        if miou > best_miou:
            best_miou, best_thr = miou, float(thr)

    print(f"  best thr={best_thr:.2f}  val_mIoU={best_miou:.4f}")
    return best_thr


# ── one training phase ────────────────────────────────────────────────────────
def run_phase(model, train_loader, n_epochs, lr_max, label=""):
    opt   = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr_max,
        epochs=n_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, div_factor=10, final_div_factor=100,
    )
    for epoch in range(1, n_epochs + 1):
        model.train()
        tot_loss = 0.0
        t0       = time.time()
        for imgs, intrins, cam2ego, occ, _ in train_loader:
            imgs    = imgs.to(DEVICE)
            intrins = intrins.to(DEVICE)
            cam2ego = cam2ego.to(DEVICE)
            occ     = occ.to(DEVICE)
            pred    = model(imgs, intrins, cam2ego)
            loss    = compute_loss(pred, occ)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
        avg = tot_loss / len(train_loader)
        print(f"  [{label}] ep {epoch:3d}/{n_epochs}  loss={avg:.4f}  ({time.time()-t0:.0f}s)")


# ── full training ─────────────────────────────────────────────────────────────
def train_model():
    print(f"Device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_ds = OccDataset('/data/train')
    n      = len(all_ds)
    n_val  = max(4, n // 10)
    n_tr   = n - n_val
    g = torch.Generator().manual_seed(42)
    tr_ds, val_ds = torch.utils.data.random_split(all_ds, [n_tr, n_val], generator=g)

    tr_ld  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=1, pin_memory=True)
    full_ld = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    model = OccNet().to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_par:.2f}M")

    # ── Stage 1: train on 90%, track best val IoU ──────────────────────────
    print(f"\n=== Stage 1: {N_EPOCHS} epochs on {n_tr} samples ===")
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR_MAX,
        epochs=N_EPOCHS, steps_per_epoch=len(tr_ld),
        pct_start=0.1, div_factor=10, final_div_factor=100,
    )

    best_iou, best_ckpt = -1.0, None

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        t0       = time.time()
        for imgs, intrins, cam2ego, occ, _ in tr_ld:
            imgs    = imgs.to(DEVICE)
            intrins = intrins.to(DEVICE)
            cam2ego = cam2ego.to(DEVICE)
            occ     = occ.to(DEVICE)
            pred    = model(imgs, intrins, cam2ego)
            loss    = compute_loss(pred, occ)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()

        avg_loss = tot_loss / len(tr_ld)
        elapsed  = time.time() - t0

        if epoch % 5 == 0 or epoch == N_EPOCHS:
            model.eval()
            all_ious = []
            with torch.no_grad():
                for imgs, intrins, cam2ego, occ, _ in val_ld:
                    imgs, intrins = imgs.to(DEVICE), intrins.to(DEVICE)
                    cam2ego = cam2ego.to(DEVICE)
                    pred    = model(imgs, intrins, cam2ego)
                    pb      = torch.sigmoid(pred) > 0.4
                    all_ious += voxel_iou(pb, occ)
            miou = float(np.mean(all_ious))
            mark = " ★" if miou > best_iou else ""
            print(f"Ep {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  "
                  f"val_mIoU={miou:.4f}{mark}  ({elapsed:.0f}s)")
            if miou > best_iou:
                best_iou  = miou
                best_ckpt = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"Ep {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  ({elapsed:.0f}s)")

    print(f"\nBest stage-1 val mIoU: {best_iou:.4f}")
    if best_ckpt:
        model.load_state_dict(best_ckpt)

    threshold = find_threshold(model, val_ld)

    # ── Stage 2: fine-tune on ALL data ────────────────────────────────────
    if FINETUNE_EPOCHS > 0:
        print(f"\n=== Stage 2: {FINETUNE_EPOCHS} fine-tune epochs on {n} samples ===")
        run_phase(model, full_ld, FINETUNE_EPOCHS, LR_MAX * 0.1, "FT")

    return model, threshold


# ── inference ─────────────────────────────────────────────────────────────────
def predict_all(model, threshold=0.4):
    test_dir = '/data/test_input'
    out_dir  = '/artifacts'
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(f"{test_dir}/*.npz"))
    print(f"\nPredicting {len(files)} samples (thr={threshold:.2f}) …")
    model.eval()

    with torch.no_grad():
        for fpath in files:
            d       = np.load(fpath)
            imgs    = torch.from_numpy(d['imgs']).float() / 255.
            imgs    = imgs.permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)
            intrins = torch.from_numpy(d['intrins']).unsqueeze(0).to(DEVICE)
            cam2ego = torch.from_numpy(d['cam2ego']).unsqueeze(0).to(DEVICE)

            logits = model(imgs, intrins, cam2ego)            # (1, VX, VY, VZ)
            mask   = (torch.sigmoid(logits[0]) > threshold).cpu().numpy().astype(np.uint8)

            token = os.path.splitext(os.path.basename(fpath))[0]
            np.save(os.path.join(out_dir, f"pred_{token}.npy"), mask)

    print("Predictions written.")


# ── entry ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model, thr = train_model()
    predict_all(model, threshold=thr)
