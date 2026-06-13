#!/usr/bin/env python3
"""
Learned Monocular Visual Odometry
  - Preloads all images + flows into RAM → zero I/O during training
  - 8-channel input: [RGB_t, RGB_{t+1}, flow_upsamp] at 96×320
  - Compact custom CNN trained from scratch on GPU (no pretrained weights needed)
  - Accumulates predicted relative poses into test trajectories
"""

import os, glob, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─── Config ───────────────────────────────────────────────────────────────────
LAB_DATA      = os.getenv('LAB_DATA',      '/data')
LAB_ARTIFACTS = os.getenv('LAB_ARTIFACTS', '/artifacts')

IMG_H, IMG_W   = 96,  320          # Half-res for speed
FLOW_H, FLOW_W = 48,  160          # Flow at quarter-res (~1 ms)
BATCH_SIZE     = 32
NUM_EPOCHS     = 120
LR_MAX         = 3e-4
LR_MIN         = 5e-6
WEIGHT_ROT     = 50.0
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

torch.manual_seed(42); np.random.seed(42)

# ─── Pose utilities ───────────────────────────────────────────────────────────
def load_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) == 12:
                T = np.eye(4, dtype=np.float64)
                T[:3] = np.array(v, dtype=np.float64).reshape(3, 4)
                poses.append(T)
    return poses

def rel_pose(T0, T1):
    return np.linalg.inv(T0) @ T1   # T_{i+1} = T_i @ T_rel

def mat_to_6dof(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3].astype(np.float64))
    return np.concatenate([rvec.ravel(), T[:3, 3]]).astype(np.float32)

def dof6_to_mat(v):
    R, _ = cv2.Rodrigues(np.asarray(v[:3], np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R;  T[:3, 3] = v[3:]
    return T

# ─── Image helpers ────────────────────────────────────────────────────────────
def bgr_to_rgb_norm(bgr):
    """BGR full-res → (3, IMG_H, IMG_W) float32 ImageNet-normalised."""
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)

def bgr_to_gray_small(bgr):
    """BGR full-res → (FLOW_H, FLOW_W) uint8 grayscale."""
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (FLOW_W, FLOW_H))

def flow_from_grays(g0, g1):
    """(FLOW_H, FLOW_W) uint8 × 2 → (2, FLOW_H, FLOW_W) float32 normalised."""
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow[..., 0] /= (FLOW_W / 2.0)
    flow[..., 1] /= (FLOW_H / 2.0)
    return flow.transpose(2, 0, 1).astype(np.float32)

def build_input(rgb0, rgb1, flow_small):
    """→ (8, IMG_H, IMG_W) float32"""
    fu = np.stack([
        cv2.resize(flow_small[0], (IMG_W, IMG_H)) * (IMG_W / FLOW_W),
        cv2.resize(flow_small[1], (IMG_W, IMG_H)) * (IMG_H / FLOW_H),
    ])
    return np.concatenate([rgb0, rgb1, fu], axis=0)

# ─── Dataset: fully preloaded ──────────────────────────────────────────────────
class VODataset(Dataset):
    """Preloads all images + flows into RAM; __getitem__ is pure numpy (zero I/O)."""

    def __init__(self, seq_dirs, augment=False):
        self.augment = augment
        t0 = time.time()

        # ── Step 1: collect unique image paths per sequence ──
        path_to_idx = {}       # path → index in self.rgb_imgs / self.gray_imgs
        rgb_list    = []       # (3, H, W) float32
        gray_list   = []       # (FLOW_H, FLOW_W) uint8

        pairs_info = []        # (i0, i1, pose6)
        flows_list = []        # (2, FLOW_H, FLOW_W) float32

        for d in seq_dirs:
            P    = load_poses(os.path.join(d, 'poses.txt'))
            imgs = sorted(glob.glob(os.path.join(d, 'left_*.png')))
            N    = min(len(P), len(imgs))

            # Preload images for this sequence
            for i in range(N):
                if imgs[i] not in path_to_idx:
                    bgr = cv2.imread(imgs[i])
                    path_to_idx[imgs[i]] = len(rgb_list)
                    rgb_list.append(bgr_to_rgb_norm(bgr))
                    gray_list.append(bgr_to_gray_small(bgr))

            # Build pairs with precomputed flow
            for i in range(N - 1):
                i0   = path_to_idx[imgs[i]]
                i1   = path_to_idx[imgs[i + 1]]
                flow = flow_from_grays(gray_list[i0], gray_list[i1])
                fi   = len(flows_list)
                flows_list.append(flow.astype(np.float16))
                pose6 = mat_to_6dof(rel_pose(P[i], P[i + 1]))
                pairs_info.append((i0, i1, fi, pose6))

        # ── Stack into contiguous arrays ──
        self.rgb   = np.stack(rgb_list)    # (N_imgs, 3, H, W)
        self.flows = np.stack(flows_list)  # (N_pairs, 2, fH, fW)
        self.idx0  = np.array([p[0] for p in pairs_info], dtype=np.int32)
        self.idx1  = np.array([p[1] for p in pairs_info], dtype=np.int32)
        self.fidx  = np.array([p[2] for p in pairs_info], dtype=np.int32)
        self.poses = np.stack([p[3] for p in pairs_info]).astype(np.float32)

        ram = (self.rgb.nbytes + self.flows.nbytes) / 1e6
        print(f'  Preloaded: {len(pairs_info)} pairs  |  RAM: {ram:.0f} MB  |  '
              f'{time.time()-t0:.1f}s')

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        rgb0 = self.rgb[self.idx0[idx]].copy()
        rgb1 = self.rgb[self.idx1[idx]].copy()
        flow = self.flows[self.fidx[idx]].astype(np.float32)
        pose = self.poses[idx].copy()

        # ── Augmentation ──
        if self.augment and np.random.rand() < 0.5:
            b    = np.float32(np.random.uniform(0.8, 1.2))
            rgb0 *= b;  rgb1 *= b

        if self.augment and np.random.rand() < 0.5:
            rgb0 = rgb0[:, :, ::-1].copy()
            rgb1 = rgb1[:, :, ::-1].copy()
            flow = np.stack([-flow[0, :, ::-1], flow[1, :, ::-1]]).copy()
            pose = pose.copy()
            pose[[1, 2, 3]] *= -1   # ry, rz, tx

        x = build_input(rgb0, rgb1, flow).astype(np.float32)
        return torch.as_tensor(x), torch.as_tensor(pose)

# ─── Model ────────────────────────────────────────────────────────────────────
class ConvBN(nn.Module):
    def __init__(self, i, o, k, s=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, k, stride=s, padding=k//2, bias=False),
            nn.BatchNorm2d(o), nn.LeakyReLU(0.1, inplace=True))
    def forward(self, x): return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.a(x) + x)

class PoseNet(nn.Module):
    """8-channel [RGB0,RGB1,flow] @ 96×320 → 6-DoF pose."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBN(8, 64, 7, 2),    # → 48×160
            ResBlock(64),
            ConvBN(64, 128, 5, 2),  # → 24×80
            ResBlock(128),
            ConvBN(128, 256, 3, 2), # → 12×40
            ResBlock(256),
            ConvBN(256, 256, 3, 2), # → 6×20
            ResBlock(256),
            ConvBN(256, 512, 3, 2), # → 3×10
            ConvBN(512, 512, 3, 1), # → 3×10
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 6),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x): return self.head(self.enc(x))

# ─── Training ─────────────────────────────────────────────────────────────────
def train():
    print(f'Device: {DEVICE}')
    seq_dirs = sorted(glob.glob(os.path.join(LAB_DATA, 'train', 'seq_*')))
    ds = VODataset(seq_dirs, augment=True)

    # num_workers=0 → no forked processes → no /dev/shm limit issue
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, pin_memory=True, drop_last=True)

    model = PoseNet().to(DEVICE)
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')

    opt    = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=1e-4)
    warmup = optim.lr_scheduler.LinearLR(
                 opt, start_factor=0.05, end_factor=1.0,
                 total_iters=5 * len(dl))
    cosine = CosineAnnealingLR(opt, T_max=max(1, NUM_EPOCHS - 5), eta_min=LR_MIN)

    t0 = time.time()
    for ep in range(NUM_EPOCHS):
        model.train()
        ep_loss = rot_l = trn_l = 0.0

        for x, y in dl:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            p    = model(x)
            rl   = WEIGHT_ROT * nn.functional.smooth_l1_loss(p[:, :3], y[:, :3], beta=0.005)
            tl   =              nn.functional.smooth_l1_loss(p[:, 3:], y[:, 3:], beta=0.10)
            loss = rl + tl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ep < 5:
                warmup.step()
            ep_loss += loss.item(); rot_l += rl.item(); trn_l += tl.item()

        if ep >= 5:
            cosine.step()

        if ep == 0 or (ep + 1) % 20 == 0:
            n = len(dl)
            print(f'  ep {ep+1:3}/{NUM_EPOCHS}  '
                  f'loss={ep_loss/n:.5f}  rot={rot_l/n:.5f}  trn={trn_l/n:.5f}  '
                  f'lr={opt.param_groups[0]["lr"]:.2e}  t={time.time()-t0:.0f}s')

    print(f'Training done in {time.time()-t0:.0f}s')
    return model

# ─── Inference ────────────────────────────────────────────────────────────────
def infer(model):
    model.eval()
    os.makedirs(LAB_ARTIFACTS, exist_ok=True)

    for d in sorted(glob.glob(os.path.join(LAB_DATA, 'test_input', 'seq_*'))):
        seq  = os.path.basename(d)
        s    = seq.split('_')[-1]
        imgs = sorted(glob.glob(os.path.join(d, 'left_*.png')))
        N    = len(imgs)
        print(f'  Inferring {seq}: {N} frames')

        T    = np.eye(4, dtype=np.float64)
        traj = [T[:3, 3].copy()]

        # Build inference inputs in batches
        BSIZ = 64
        for b0 in range(0, N - 1, BSIZ):
            b1    = min(b0 + BSIZ, N - 1)
            batch = []
            for i in range(b0, b1):
                bgr0  = cv2.imread(imgs[i])
                bgr1  = cv2.imread(imgs[i + 1])
                rgb0  = bgr_to_rgb_norm(bgr0)
                rgb1  = bgr_to_rgb_norm(bgr1)
                flow  = flow_from_grays(bgr_to_gray_small(bgr0), bgr_to_gray_small(bgr1))
                batch.append(build_input(rgb0, rgb1, flow))
            inp = np.stack(batch).astype(np.float32)
            with torch.no_grad():
                preds = model(torch.as_tensor(inp, device=DEVICE)).cpu().numpy()
            for p6 in preds:
                T = T @ dof6_to_mat(p6)
                traj.append(T[:3, 3].copy())

        out = os.path.join(LAB_ARTIFACTS, f'traj_{s}.txt')
        np.savetxt(out, np.array(traj), fmt='%.6f')
        print(f'    -> {out}  ({len(traj)} lines)')

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = train()
    infer(model)
    print('Done.')
