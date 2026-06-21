"""
Multi-camera floor-occupancy via IPM + temporal background subtraction.

Key design choices:
  • static cameras → precompute projection grids once
  • temporal-median background → 3-ch difference image directly highlights agents
  • 6-ch input (3 rgb + 3 bg-diff) per camera
  • bilinear grid_sample onto BEV at 4 height planes, fuse (mean+max) across cams
  • focal loss  alpha=0.97, gamma=2 for ~0.4% positive rate
  • adaptive top-K inference: predict exactly K cells per frame, where K is drawn
    from the per-frame probability-rank (so we match the expected ~218 occupied cells
    regardless of absolute probability scale on unseen frames)
"""
import os, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LAB_DATA      = os.environ['LAB_DATA']
LAB_ARTIFACTS = os.environ['LAB_ARTIFACTS']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

NUM_CAMS = 19
IMG_H, IMG_W = 128, 352
XG, YG   = 206, 203
CELL_SIZE = 0.5
HEIGHTS  = [0.0, 0.5, 1.0, 1.5]
FEAT_CH  = 64
HEAD_CH  = 128

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1,3,1,1)
IMG_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1,3,1,1)


# ──────────────────────────────────────────────────────────────────
# BACKGROUND MODEL   (temporal median over training frames)
# ──────────────────────────────────────────────────────────────────

def compute_background(train_dir):
    files = sorted(glob.glob(os.path.join(train_dir, '*.npz')))
    print(f"  Background model: {len(files)} frames …")
    stacks = []
    for f in files:
        d = np.load(f)
        img = d['imgs'].astype(np.float32) / 255.0    # (N,H,W,3)
        stacks.append(np.transpose(img, (0,3,1,2)))    # (N,3,H,W)
    all_imgs = np.stack(stacks, 0)                     # (T,N,3,H,W)
    bg = np.median(all_imgs, 0).astype(np.float32)     # (N,3,H,W)
    print(f"  Done. bg={bg.shape}")
    return bg


# ──────────────────────────────────────────────────────────────────
# PROJECTION GRIDS  (static)
# ──────────────────────────────────────────────────────────────────

def build_sample_grids(cam_proj_np, grid_bounds_np):
    x0, x1, y0, y1 = grid_bounds_np
    xs = x0 + CELL_SIZE * (np.arange(XG, dtype=np.float32) + 0.5)
    ys = y0 + CELL_SIZE * (np.arange(YG, dtype=np.float32) + 0.5)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    ones   = np.ones_like(xx)
    gs, ms = [], []
    for z in HEIGHTS:
        zz  = np.full_like(xx, z)
        pts = np.stack([xx, yy, zz, ones], 0).reshape(4,-1)
        uvw = cam_proj_np @ pts                               # (N,3,M)
        w   = uvw[:,2,:]
        sw  = np.where(w>1e-6, w, 1.0)
        u, v = uvw[:,0,:]/sw, uvw[:,1,:]/sw
        u_n = (2*u/(IMG_W-1)-1).reshape(NUM_CAMS,XG,YG).astype(np.float32)
        v_n = (2*v/(IMG_H-1)-1).reshape(NUM_CAMS,XG,YG).astype(np.float32)
        valid = ((w>1e-6)&(u>=0)&(u<=IMG_W-1)&(v>=0)&(v<=IMG_H-1)
                ).reshape(NUM_CAMS,XG,YG).astype(np.float32)
        gs.append(np.stack([u_n,v_n],-1))
        ms.append(valid)
    return (torch.from_numpy(np.stack(gs,0)),
            torch.from_numpy(np.stack(ms,0)))


# ──────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────

class FloorDataset(Dataset):
    def __init__(self, data_dir, bg_np, is_train=True):
        self.files    = sorted(glob.glob(os.path.join(data_dir,'*.npz')))
        self.bg       = bg_np   # (N,3,H,W) float32
        self.is_train = is_train

    def __len__(self):  return len(self.files)

    def __getitem__(self, idx):
        d    = np.load(self.files[idx])
        imgs = np.transpose(d['imgs'].astype(np.float32)/255., (0,3,1,2))  # (N,3,H,W)
        diff = imgs - self.bg
        imgs_t = torch.from_numpy(imgs)
        diff_t = torch.from_numpy(diff)
        inp = torch.cat([(imgs_t-IMG_MEAN)/IMG_STD, diff_t/0.15], 1)  # (N,6,H,W)
        if self.is_train:
            return inp, torch.from_numpy(d['bev'].astype(np.float32))
        return inp, os.path.splitext(os.path.basename(self.files[idx]))[0]


# ──────────────────────────────────────────────────────────────────
# NETWORK
# ──────────────────────────────────────────────────────────────────

class CBR(nn.Sequential):
    def __init__(self, ci, co, k=3, s=1, p=1, d=1):
        super().__init__(
            nn.Conv2d(ci,co,k,stride=s,padding=p,dilation=d,bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True))

class CameraBackbone(nn.Module):
    """(B*N,6,128,352)→(B*N,FEAT_CH,32,88)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            CBR(6,32,s=2), CBR(32,48), CBR(48,64,s=2), CBR(64,64), CBR(64,FEAT_CH))
    def forward(self,x): return self.net(x)

class BEVLifter(nn.Module):
    def __init__(self, grids, masks):
        super().__init__()
        self.register_buffer('grids', grids)   # (nh,N,XG,YG,2)
        self.register_buffer('masks', masks)    # (nh,N,XG,YG)
        self.nh = grids.shape[0]

    def forward(self, feats):
        B,N,C,Hf,Wf = feats.shape
        ff = feats.reshape(B*N,C,Hf,Wf)
        out = []
        for hi in range(self.nh):
            g  = self.grids[hi]
            m  = self.masks[hi]
            ge = g.unsqueeze(0).expand(B,-1,-1,-1,-1).reshape(B*N,XG,YG,2)
            s  = F.grid_sample(ff,ge,mode='bilinear',padding_mode='zeros',align_corners=True)
            s  = s.reshape(B,N,C,XG,YG) * m.unsqueeze(0).unsqueeze(2)
            nv = m.unsqueeze(0).unsqueeze(2).sum(1,keepdim=True).clamp(min=1)
            out += [s.sum(1)/nv.squeeze(1), s.max(1)[0]]
        return torch.cat(out,1)   # (B,C*nh*2,XG,YG)

class BEVHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            CBR(in_ch,HEAD_CH), nn.Dropout2d(0.10),
            CBR(HEAD_CH,HEAD_CH,p=2,d=2), nn.Dropout2d(0.10),
            CBR(HEAD_CH,64), nn.Conv2d(64,1,1))
    def forward(self,x): return self.net(x).squeeze(1)

class FloorOccModel(nn.Module):
    def __init__(self, grids, masks):
        super().__init__()
        self.backbone = CameraBackbone()
        self.lifter   = BEVLifter(grids, masks)
        self.head     = BEVHead(FEAT_CH*len(HEIGHTS)*2)
    def forward(self, imgs):
        B,N,C,H,W = imgs.shape
        f = self.backbone(imgs.reshape(B*N,C,H,W))
        f = f.reshape(B,N,FEAT_CH,f.shape[-2],f.shape[-1])
        return self.head(self.lifter(f))


# ──────────────────────────────────────────────────────────────────
# FOCAL LOSS
# ──────────────────────────────────────────────────────────────────

def focal_loss(logits, targets, alpha=0.97, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.sigmoid(logits)*targets + (1-torch.sigmoid(logits))*(1-targets)
    a_t = alpha*targets + (1-alpha)*(1-targets)
    return (a_t*(1-p_t)**gamma*bce).mean()


# ──────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────

def train():
    t0 = time.time()
    BUDGET = 3100

    print("Building background…")
    bg_np = compute_background(os.path.join(LAB_DATA,'train'))

    ref = np.load(sorted(glob.glob(os.path.join(LAB_DATA,'train','*.npz')))[0])
    print("Building grids…")
    grids, masks = build_sample_grids(ref['cam_proj'], ref['grid_bounds'])
    grids, masks = grids.to(DEVICE), masks.to(DEVICE)

    ds  = FloorDataset(os.path.join(LAB_DATA,'train'), bg_np, is_train=True)
    ldl = DataLoader(ds, batch_size=4, shuffle=True,
                     num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Train: {len(ds)} samples | {len(ldl)} batches")

    model  = FloorOccModel(grids, masks).to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    EPOCHS = 160
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, steps_per_epoch=len(ldl),
        epochs=EPOCHS, pct_start=0.03)
    scaler = torch.amp.GradScaler('cuda')

    best_loss, best_state = float('inf'), None

    for epoch in range(EPOCHS):
        if time.time()-t0 > BUDGET:
            print(f"Budget at epoch {epoch}.")
            break
        model.train()
        tot = 0.0
        for imgs, bevs in ldl:
            imgs = imgs.to(DEVICE, non_blocking=True)
            bevs = bevs.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss = focal_loss(model(imgs), bevs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()

        avg = tot/len(ldl)
        lr  = sched.get_last_lr()[0]
        elap = time.time()-t0

        iou_str = ""
        if (epoch+1) % 25 == 0:
            model.eval()
            ii = uu = 0.0
            with torch.no_grad():
                for iv,bv in DataLoader(ds, batch_size=4, num_workers=4):
                    iv = iv.to(DEVICE)
                    bv = bv.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        lg = model(iv)
                    p  = (torch.sigmoid(lg) > 0.5).float()
                    ii += (p*bv).sum().item()
                    uu += (p+bv-p*bv).sum().item()
            if uu > 0:
                iou_str = f"  iou={ii/uu:.4f}"

        print(f"E{epoch+1:4d}  loss={avg:.5f}  lr={lr:.2e}  t={elap:.0f}s{iou_str}")
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # ── collect training-set per-frame GT cell count stats ──
    gt_counts = np.array([np.load(f)['bev'].sum()
                          for f in sorted(glob.glob(os.path.join(LAB_DATA,'train','*.npz')))])
    gt_mean = float(gt_counts.mean())
    gt_std  = float(gt_counts.std())
    print(f"Training GT cells: mean={gt_mean:.1f}  std={gt_std:.1f}")

    # ── also find a fixed threshold via IoU calibration (fallback) ──
    print("Calibrating threshold…")
    model.eval()
    all_p, all_b = [], []
    with torch.no_grad():
        for iv,bv in DataLoader(ds, batch_size=4, num_workers=4):
            iv = iv.to(DEVICE)
            with torch.amp.autocast('cuda'):
                lg = model(iv)
            all_p.append(torch.sigmoid(lg).cpu())
            all_b.append(bv)
    all_p = torch.cat(all_p,0).numpy()
    all_b = torch.cat(all_b,0).numpy()

    best_t, best_iou = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        pr = (all_p>t).astype(np.float32)
        i  = (pr*all_b).sum(); u = (pr+all_b-pr*all_b).sum()
        if u>0 and i/u>best_iou:
            best_iou, best_t = i/u, float(t)
    print(f"Best fixed thr={best_t:.2f}  iou={best_iou:.4f}")

    return model, bg_np, best_t, gt_mean, gt_std


# ──────────────────────────────────────────────────────────────────
# INFERENCE  (adaptive top-K + fixed-threshold combined)
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, bg_np, fixed_thr, gt_mean, gt_std):
    """
    Strategy:
      1. Compute probability map for each test frame.
      2. Use fixed threshold to get confident predictions.
      3. Additionally pad using rank-based (top-K) selection to reach
         the expected count K ~ gt_mean.  Only add cells above p=0.2
         to avoid noisy additions.
      4. Final pred = union of (confident ∪ top-K-padded).
    This handles the training/test probability-scale shift.
    """
    model.eval()
    bg    = torch.from_numpy(bg_np)
    files = sorted(glob.glob(os.path.join(LAB_DATA,'test_input','*.npz')))
    os.makedirs(LAB_ARTIFACTS, exist_ok=True)

    K_target = int(round(gt_mean))   # ~219

    pred_counts = []
    for fpath in files:
        d     = np.load(fpath)
        token = os.path.splitext(os.path.basename(fpath))[0]

        imgs = np.transpose(d['imgs'].astype(np.float32)/255., (0,3,1,2))
        imgs_t = torch.from_numpy(imgs)
        diff_t = imgs_t - bg
        inp = torch.cat([(imgs_t-IMG_MEAN)/IMG_STD, diff_t/0.15],1).unsqueeze(0).to(DEVICE)

        with torch.amp.autocast('cuda'):
            logits = model(inp)
        probs = torch.sigmoid(logits[0]).cpu().numpy()   # (XG, YG)

        flat = probs.ravel()   # (XG*YG,)

        # fixed-threshold predictions
        conf_mask = (flat >= fixed_thr)

        # top-K padded predictions (only cells with p >= 0.20)
        sortidx   = np.argsort(flat)[::-1]             # descending
        reasonable= sortidx[flat[sortidx] >= 0.20]     # cells above noise
        topk      = reasonable[:K_target]              # top-K reasonable cells

        # union of both strategies
        combined  = np.zeros(XG*YG, dtype=bool)
        combined[conf_mask]  = True
        combined[topk]       = True

        pred = combined.reshape(XG, YG).astype(np.uint8)
        pred_counts.append(pred.sum())
        np.save(os.path.join(LAB_ARTIFACTS, f'pred_{token}.npy'), pred)

    n = len(files)
    print(f"Saved {n} preds  "
          f"mean_cells={np.mean(pred_counts):.1f}±{np.std(pred_counts):.1f}  "
          f"(target≈{K_target})")


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== Training ===")
    model, bg_np, fixed_thr, gt_mean, gt_std = train()

    print("=== Inference ===")
    predict(model, bg_np, fixed_thr, gt_mean, gt_std)

    print("Done.")
