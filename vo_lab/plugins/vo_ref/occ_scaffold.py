"""LOCKED 3D-occupancy scaffold core — HARNESS-OWNED, seeded as `occ_core.py` (agent must NOT edit).
Owns the fragile parts: the Lift-Splat-to-3D geometry, the correct surround flip augmentation
(camera swap + extrinsic + ego-y + occ-y-flip), the training loop, and threshold calibration. The
agent authors ONLY `model.py` (the learnable network), isolating network design from the
geometry/augmentation choices that drive the free-form variance.

Interface the agent's model.py MUST provide:
  build_encoder() -> nn.Module: forward(imgs[B*6,3,H,W]) -> (depth_logits[B*6,D,h,w], context[B*6,C,h,w])
                    D fixed (=DEPTH_BINS, import it); C and the downsample h,w are yours
  build_occ_head() -> nn.Module: forward(vox[B,C,X,Y,Z]) -> occupancy_logits[B,X,Y,Z]

Run with: python3 occ_core.py   (entry_command). Pure torch + numpy.
"""
import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, W, N = 128, 352, 6
XB = YB = (-50.0, 50.0); ZB = (-2.0, 4.0); RES = 0.5
XG = int((XB[1] - XB[0]) / RES); YG = int((YB[1] - YB[0]) / RES); ZG = int((ZB[1] - ZB[0]) / RES)
DBOUND = (4.0, 45.0, 1.0); DEPTH_BINS = len(torch.arange(*DBOUND))
EPOCHS = int(os.environ.get("OCC_EPOCHS", "24")); BATCH, LR, POS_WEIGHT = 4, 2e-3, 20.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
FLIP_CAM_ORDER = [2, 1, 0, 5, 4, 3]
_DBINS = torch.arange(*DBOUND)


def _norm(imgs):
    return (imgs.float() / 255.0 - MEAN) / STD


def _load(npz, with_gt):
    d = np.load(npz)
    imgs = torch.from_numpy(d["imgs"]).permute(0, 3, 1, 2)
    out = [imgs, torch.from_numpy(d["intrins"]).float(), torch.from_numpy(d["cam2ego"]).float()]
    if with_gt:
        out.append(torch.from_numpy(d["occ"]).float())
    return out


def _flip(imgs, intrins, cam2ego, occ):
    """Correct horizontal flip for a surround rig + 3D occupancy target."""
    imgs = torch.flip(imgs, dims=[-1])[FLIP_CAM_ORDER]
    intrins = intrins[FLIP_CAM_ORDER].clone(); intrins[:, 0, 2] = (W - 1) - intrins[:, 0, 2]
    c2e = cam2ego[FLIP_CAM_ORDER].clone()
    fy = torch.diag(torch.tensor([1., -1., 1.]))
    c2e[:, :3, :3] = fy @ c2e[:, :3, :3]; c2e[:, :3, 3] = (fy @ c2e[:, :3, 3].unsqueeze(-1)).squeeze(-1)
    cf = torch.diag(torch.tensor([-1., 1., 1.])); c2e[:, :3, :3] = c2e[:, :3, :3] @ cf
    occ = torch.flip(occ, dims=[1])                                    # ego-y -> grid dim 1
    return imgs, intrins, c2e, occ


def _frustum(h, w):
    ds = _DBINS.view(-1, 1, 1).expand(DEPTH_BINS, h, w)
    xs = torch.linspace(0, W - 1, w).view(1, 1, w).expand(DEPTH_BINS, h, w)
    ys = torch.linspace(0, H - 1, h).view(1, h, 1).expand(DEPTH_BINS, h, w)
    return torch.stack((xs, ys, ds), -1)


def lift_splat_3d(depth_logits, context, intrins, cam2ego):
    """(depth_logits[B*N,D,h,w], context[B*N,C,h,w]) + calib -> 3D feature grid [B,C,XG,YG,ZG]."""
    BN, D, h, w = depth_logits.shape; C = context.shape[1]; B = BN // N
    depth = depth_logits.softmax(1)
    feat = (depth.unsqueeze(2) * context.unsqueeze(1)).view(B, N, D, C, h, w).permute(0, 1, 2, 4, 5, 3)
    fr = _frustum(h, w).to(depth_logits.device)
    pts = torch.cat((fr[..., :2] * fr[..., 2:3], fr[..., 2:3]), -1)
    pts = pts.view(1, 1, D, h, w, 3).expand(B, N, D, h, w, 3).unsqueeze(-1)
    cam = (torch.inverse(intrins).view(B, N, 1, 1, 1, 3, 3) @ pts).squeeze(-1)
    R = cam2ego[..., :3, :3].view(B, N, 1, 1, 1, 3, 3); t = cam2ego[..., :3, 3].view(B, N, 1, 1, 1, 3)
    geom = (R @ cam.unsqueeze(-1)).squeeze(-1) + t
    gx = ((geom[..., 0] - XB[0]) / RES).long(); gy = ((geom[..., 1] - YB[0]) / RES).long()
    gz = ((geom[..., 2] - ZB[0]) / RES).long()
    keep = (gx >= 0) & (gx < XG) & (gy >= 0) & (gy < YG) & (gz >= 0) & (gz < ZG)
    out = context.new_zeros(B, C, XG * YG * ZG); feat = feat.reshape(B, -1, C)
    gx, gy, gz, keep = gx.reshape(B, -1), gy.reshape(B, -1), gz.reshape(B, -1), keep.reshape(B, -1)
    for b in range(B):
        k = keep[b]
        out[b].index_add_(1, (gx[b, k] * YG + gy[b, k]) * ZG + gz[b, k], feat[b, k].t())
    return out.view(B, C, XG, YG, ZG)


def main():
    import model
    data = os.environ["LAB_DATA"]; art = os.environ["LAB_ARTIFACTS"]; os.makedirs(art, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    train = sorted(glob.glob(os.path.join(data, "train", "*.npz")))
    test = sorted(glob.glob(os.path.join(data, "test_input", "*.npz")))
    print(f"[occ-scaffold] train={len(train)} test={len(test)} dev={DEV} D={DEPTH_BINS}", flush=True)
    enc = model.build_encoder().to(DEV); head = model.build_occ_head().to(DEV)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, epochs=EPOCHS, steps_per_epoch=(len(train) + BATCH - 1) // BATCH)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=DEV))

    def fwd(imgs, K, c2e):
        d, ctx = enc(_norm(imgs).reshape(imgs.shape[0] * N, *imgs.shape[2:]).to(DEV))
        return head(lift_splat_3d(d, ctx, K.to(DEV), c2e.to(DEV)))

    for ep in range(EPOCHS):
        enc.train(); head.train(); order = np.random.permutation(len(train))
        for i in range(0, len(order), BATCH):
            batch = [_load(train[j], True) for j in order[i:i + BATCH]]
            imgs, K, c2e, gt = [], [], [], []
            for im, k, c, o in batch:
                if np.random.rand() < 0.5:
                    im, k, c, o = _flip(im, k, c, o)
                imgs.append(im); K.append(k); c2e.append(c); gt.append(o)
            loss = lossf(fwd(torch.stack(imgs), torch.stack(K), torch.stack(c2e)), torch.stack(gt).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print(f"[occ-scaffold] epoch {ep:02d} loss {loss.item():.3f}", flush=True)

    enc.eval(); head.eval()
    best_t, best_iou = 0.0, -1.0
    with torch.no_grad():
        lg, gg = [], []
        for f in train[:120]:
            imgs, K, c2e, occ = _load(f, True)
            lg.append(fwd(imgs.unsqueeze(0), K.unsqueeze(0), c2e.unsqueeze(0))[0].cpu()); gg.append(occ.bool())
        for t in np.linspace(-2, 2, 21):
            inter = union = 0
            for l, g in zip(lg, gg):
                p = l > t; inter += (p & g).sum().item(); union += (p | g).sum().item()
            iou = inter / union if union else 0
            if iou > best_iou:
                best_iou, best_t = iou, float(t)
    print(f"[occ-scaffold] calibrated threshold {best_t:.2f} (train IoU {best_iou:.3f})", flush=True)
    with torch.no_grad():
        for f in test:
            tok = os.path.splitext(os.path.basename(f))[0]
            imgs, K, c2e = _load(f, False)
            logits = fwd(imgs.unsqueeze(0), K.unsqueeze(0), c2e.unsqueeze(0))[0]
            np.save(os.path.join(art, f"pred_{tok}.npy"), (logits > best_t).cpu().numpy().astype(np.uint8))
    print(f"[occ-scaffold] wrote {len(test)} predictions", flush=True)


if __name__ == "__main__":
    main()
