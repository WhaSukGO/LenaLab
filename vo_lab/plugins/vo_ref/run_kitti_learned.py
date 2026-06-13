"""Reference LEARNED monocular VO (pure PyTorch, trains on the GPU) — the learned-research
baseline. No custom CUDA extensions: a small CNN regresses the 6-DoF relative pose between
consecutive frames, trained supervised on a TRAIN sequence's GT, then accumulated into a
trajectory on each held-out TEST sequence.

Reads $LAB_DATA:
  train/      left_%06d.png + poses.txt (KITTI 3x4 cam->world per frame) + intrinsics.txt
  test_input/seq_<s>/  left_%06d.png + intrinsics.txt   (NO labels)
Writes $LAB_ARTIFACTS/traj_<s>.txt — camera centres per frame for each test sequence.

The trainer runs on the GPU as a harness JOB (gpu_lease + CUDA image): wall-clock, not tokens.
Env knobs (the recipe may tune): LAB_EPOCHS, LAB_LR, LAB_BATCH. LEARNED_SMOKE=1 -> 1 epoch."""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

H, W = 128, 384          # network input size


def _load_gray(path, size=(W, H)):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, size).astype(np.float32) / 255.0
    return im


def _rel_pose(Ti, Tj):
    """Relative transform cam_j -> cam_i for 3x4 cam->world Ti, Tj. Returns (aa[3], t[3])."""
    Ai = np.eye(4); Ai[:3] = Ti.reshape(3, 4)
    Aj = np.eye(4); Aj[:3] = Tj.reshape(3, 4)
    R = Ai[:3, :3]; t = Ai[:3, 3]
    inv = np.eye(4); inv[:3, :3] = R.T; inv[:3, 3] = -R.T @ t
    rel = inv @ Aj
    aa, _ = cv2.Rodrigues(rel[:3, :3])
    return aa.ravel().astype(np.float32), rel[:3, 3].astype(np.float32)


def main() -> int:
    import torch
    import torch.nn as nn

    data = Path(os.environ["LAB_DATA"])
    art = Path(os.environ["LAB_ARTIFACTS"]); art.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1 if os.environ.get("LEARNED_SMOKE") == "1" else int(os.environ.get("LAB_EPOCHS", 30))
    lr = float(os.environ.get("LAB_LR", 1e-4))
    batch = int(os.environ.get("LAB_BATCH", 32))
    print(f"[learned-vo] device={dev} epochs={epochs} lr={lr} batch={batch}", flush=True)

    # --- build training pairs from each train sequence (pairs within a seq only) ---
    seqdirs = sorted((data / "train").glob("seq_*"))
    X, Yr, Yt = [], [], []
    for sd in seqdirs:
        frames = sorted(sd.glob("left_*.png"))
        poses = np.loadtxt(sd / "poses.txt")
        n = min(len(frames), len(poses))
        imgs = np.stack([_load_gray(frames[i]) for i in range(n)])  # (n,H,W)
        for i in range(n - 1):
            X.append(np.stack([imgs[i], imgs[i + 1]]))              # (2,H,W)
            aa, t = _rel_pose(poses[i], poses[i + 1])
            Yr.append(aa); Yt.append(t)
    X = torch.tensor(np.stack(X)); Yr = torch.tensor(np.stack(Yr)); Yt = torch.tensor(np.stack(Yt))
    print(f"[learned-vo] {len(X)} training pairs from {len(seqdirs)} sequences", flush=True)

    # --- small CNN: 2 stacked frames -> 6-DoF (axis-angle 3 + translation 3) ---
    def conv(a, b, s): return nn.Sequential(nn.Conv2d(a, b, 3, s, 1), nn.BatchNorm2d(b), nn.ReLU())
    net = nn.Sequential(
        conv(2, 16, 2), conv(16, 32, 2), conv(32, 64, 2), conv(64, 128, 2), conv(128, 128, 2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6),
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xd, Yrd, Ytd = X.to(dev), Yr.to(dev), Yt.to(dev)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xd), device=dev)
        tot = 0.0
        for k in range(0, len(Xd), batch):
            b = perm[k:k + batch]
            pred = net(Xd[b])
            # weight rotation higher (radians are small vs metric translation)
            loss = 100.0 * ((pred[:, :3] - Yrd[b]) ** 2).mean() + ((pred[:, 3:] - Ytd[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"[learned-vo] epoch {ep:3d}  loss {tot/len(Xd):.5f}", flush=True)

    # --- inference: accumulate predicted relative poses per test sequence ---
    net.eval()
    for seqdir in sorted((data / "test_input").glob("seq_*")):
        tf = sorted(seqdir.glob("left_*.png"))
        ti = np.stack([_load_gray(f) for f in tf])
        Tw = np.eye(4); centres = [Tw[:3, 3].copy()]
        with torch.no_grad():
            for i in range(len(tf) - 1):
                x = torch.tensor(np.stack([ti[i], ti[i + 1]])[None]).to(dev)
                out = net(x)[0].cpu().numpy()
                Rr, _ = cv2.Rodrigues(out[:3]); rel = np.eye(4); rel[:3, :3] = Rr; rel[:3, 3] = out[3:]
                Tw = Tw @ rel
                centres.append(Tw[:3, 3].copy())
        out_path = art / f"traj_{seqdir.name.replace('seq_', '')}.txt"
        np.savetxt(out_path, np.array(centres), fmt="%.6f")
        print(f"[learned-vo] wrote {out_path.name} ({len(centres)} frames)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
