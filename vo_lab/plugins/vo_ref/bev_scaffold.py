"""LOCKED BEV scaffold core — HARNESS-OWNED, seeded into the agent's workspace (the agent must
NOT edit this). It owns the parts the n=3 free-form runs proved fragile: the Lift-Splat GEOMETRY
(frustum -> ego -> voxel-pool), the CORRECT horizontal-flip augmentation (camera swap + extrinsic
+ ego-y update), the training loop, and threshold calibration. The agent authors ONLY `model.py`
(the learnable network) behind a fixed interface — isolating *network design* from the
augmentation/validation choices that sank run 2.

Interface the agent's model.py MUST provide:
  build_encoder() -> nn.Module with forward(imgs[B*N,3,H,W]) -> (depth_logits[B*N,D,h,w],
                     context[B*N,C,h,w])   # D is fixed (=DEPTH_BINS); C and the /downsample are yours
  build_bev_head() -> nn.Module with forward(bev[B,C,X,Y]) -> occupancy_logits[B,X,Y]

This file: reads $LAB_DATA/train + test_input, trains, writes $LAB_ARTIFACTS/pred_<token>.npy.
Run with: python3 bev_core.py   (entry_command). Pure torch + numpy.
"""
import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, N = 128, 352, 6
XG = YG = 200
XB = YB = (-50.0, 50.0)
RES = 0.5
DBOUND = (4.0, 45.0, 1.0)
DEPTH_BINS = len(torch.arange(*DBOUND))          # = D (fixed; the agent's depth head must match)
EPOCHS = int(os.environ.get("BEV_EPOCHS", "24"))     # override for fast smoke tests
BATCH, LR, POS_WEIGHT = 4, 2e-3, 8.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FLIP_CAM_ORDER = [2, 1, 0, 5, 4, 3]              # [FL,F,FR,BL,B,BR] -> mirror left<->right
_DBINS = torch.arange(*DBOUND)


# ---------- data ----------
def _norm(imgs):
    return (imgs.float() / 255.0 - MEAN) / STD


def _load(npz, with_gt):
    d = np.load(npz)
    imgs = torch.from_numpy(d["imgs"]).permute(0, 3, 1, 2)              # (6,3,H,W) uint8
    out = [imgs, torch.from_numpy(d["intrins"]).float(), torch.from_numpy(d["cam2ego"]).float()]
    if with_gt:
        out.append(torch.from_numpy(d["bev"]).float())
    return out


# ---------- LOCKED augmentation (the part run 2 got wrong) ----------
def _flip(imgs, intrins, cam2ego, bev):
    """Horizontal flip, done correctly: mirror pixels, SWAP left/right cameras, flip the
    intrinsic principal point, negate the ego-y of the extrinsics, and y-flip the BEV target."""
    imgs = torch.flip(imgs, dims=[-1])[FLIP_CAM_ORDER]
    intrins = intrins[FLIP_CAM_ORDER].clone()
    intrins[:, 0, 2] = (W - 1) - intrins[:, 0, 2]                       # cx -> W-1-cx
    c2e = cam2ego[FLIP_CAM_ORDER].clone()
    fy = torch.diag(torch.tensor([1., -1., 1.]))                       # ego y-flip
    c2e[:, :3, :3] = fy @ c2e[:, :3, :3]
    c2e[:, :3, 3] = (fy @ c2e[:, :3, 3].unsqueeze(-1)).squeeze(-1)
    cf = torch.diag(torch.tensor([-1., 1., 1.]))                       # camera x-flip (mirror)
    c2e[:, :3, :3] = c2e[:, :3, :3] @ cf
    bev = torch.flip(bev, dims=[1])                                     # y is dim-1 (cols)
    return imgs, intrins, c2e, bev


# ---------- LOCKED Lift-Splat geometry ----------
def _frustum(h, w):
    ds = _DBINS.view(-1, 1, 1).expand(DEPTH_BINS, h, w)
    xs = torch.linspace(0, W - 1, w).view(1, 1, w).expand(DEPTH_BINS, h, w)
    ys = torch.linspace(0, H - 1, h).view(1, h, 1).expand(DEPTH_BINS, h, w)
    return torch.stack((xs, ys, ds), -1)                               # (D,h,w,3)


def lift_splat(depth_logits, context, intrins, cam2ego):
    """(depth_logits[B*N,D,h,w], context[B*N,C,h,w]) + calib -> BEV feature grid [B,C,XG,YG]."""
    BN, D, h, w = depth_logits.shape
    C = context.shape[1]
    B = BN // N
    depth = depth_logits.softmax(1)
    feat = (depth.unsqueeze(2) * context.unsqueeze(1))                 # (B*N,D,C,h,w)
    feat = feat.view(B, N, D, C, h, w).permute(0, 1, 2, 4, 5, 3)       # (B,N,D,h,w,C)
    fr = _frustum(h, w).to(depth_logits.device)
    pts = torch.cat((fr[..., :2] * fr[..., 2:3], fr[..., 2:3]), -1)
    pts = pts.view(1, 1, D, h, w, 3).expand(B, N, D, h, w, 3).unsqueeze(-1)
    Kinv = torch.inverse(intrins).view(B, N, 1, 1, 1, 3, 3)
    cam = (Kinv @ pts).squeeze(-1)
    R = cam2ego[..., :3, :3].view(B, N, 1, 1, 1, 3, 3)
    t = cam2ego[..., :3, 3].view(B, N, 1, 1, 1, 3)
    geom = (R @ cam.unsqueeze(-1)).squeeze(-1) + t                     # ego coords (B,N,D,h,w,3)
    gx = ((geom[..., 0] - XB[0]) / RES).long()
    gy = ((geom[..., 1] - YB[0]) / RES).long()
    keep = (gx >= 0) & (gx < XG) & (gy >= 0) & (gy < YG)
    out = context.new_zeros(B, C, XG * YG)
    feat = feat.reshape(B, -1, C)
    gx, gy, keep = gx.reshape(B, -1), gy.reshape(B, -1), keep.reshape(B, -1)
    for b in range(B):
        k = keep[b]
        out[b].index_add_(1, gx[b, k] * YG + gy[b, k], feat[b, k].t())
    return out.view(B, C, XG, YG)


# ---------- train + infer (LOCKED) ----------
def main():
    import model                                                       # the agent's only file
    data = os.environ["LAB_DATA"]; art = os.environ["LAB_ARTIFACTS"]; os.makedirs(art, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    train_files = sorted(glob.glob(os.path.join(data, "train", "*.npz")))
    test_files = sorted(glob.glob(os.path.join(data, "test_input", "*.npz")))
    print(f"[scaffold] train={len(train_files)} test={len(test_files)} dev={DEV} D={DEPTH_BINS}", flush=True)

    enc = model.build_encoder().to(DEV)
    head = model.build_bev_head().to(DEV)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, LR, weight_decay=1e-4)
    steps = (len(train_files) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCHS, steps_per_epoch=steps)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=DEV))

    def fwd(imgs, K, c2e):
        B, n = imgs.shape[:2]
        x = _norm(imgs).reshape(B * n, *imgs.shape[2:]).to(DEV)        # [B,N,3,H,W] -> [B*N,3,H,W]
        d, ctx = enc(x)
        return head(lift_splat(d, ctx, K.to(DEV), c2e.to(DEV)))

    for ep in range(EPOCHS):
        enc.train(); head.train()
        order = np.random.permutation(len(train_files))
        for i in range(0, len(order), BATCH):
            batch = [_load(train_files[j], True) for j in order[i:i + BATCH]]
            imgs, K, c2e, gt = [], [], [], []
            for im, k, c, b in batch:
                if np.random.rand() < 0.5:                             # LOCKED flip aug
                    im, k, c, b = _flip(im, k, c, b)
                imgs.append(im); K.append(k); c2e.append(c); gt.append(b)
            logits = fwd(torch.stack(imgs), torch.stack(K), torch.stack(c2e))
            loss = lossf(logits, torch.stack(gt).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"[scaffold] epoch {ep:02d} loss {loss.item():.3f}", flush=True)

    # LOCKED threshold calibration on the TRAIN set (no held-out peeking)
    enc.eval(); head.eval()
    best_t, best_iou = 0.0, -1.0
    with torch.no_grad():
        logit_list, gt_list = [], []
        for f in train_files[:120]:
            imgs, K, c2e, gt = _load(f, True)
            logit_list.append(fwd(imgs.unsqueeze(0), K.unsqueeze(0), c2e.unsqueeze(0))[0].cpu())
            gt_list.append(gt.bool())
        for t in np.linspace(-2, 2, 21):
            inter = union = 0
            for lg, g in zip(logit_list, gt_list):
                p = lg > t
                inter += (p & g).sum().item(); union += (p | g).sum().item()
            iou = inter / union if union else 0
            if iou > best_iou:
                best_iou, best_t = iou, float(t)
    print(f"[scaffold] calibrated threshold {best_t:.2f} (train IoU {best_iou:.3f})", flush=True)

    with torch.no_grad():
        for f in test_files:
            tok = os.path.splitext(os.path.basename(f))[0]
            imgs, K, c2e = _load(f, False)
            logits = fwd(imgs.unsqueeze(0), K.unsqueeze(0), c2e.unsqueeze(0))[0]
            np.save(os.path.join(art, f"pred_{tok}.npy"), (logits > best_t).cpu().numpy().astype(np.uint8))
    print(f"[scaffold] wrote {len(test_files)} predictions", flush=True)


if __name__ == "__main__":
    main()
