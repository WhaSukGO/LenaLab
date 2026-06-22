"""Step-2 variant of the scene-agnostic IPM reference WITH domain randomization (DR) — the data-vs-
architecture discriminator. SAME architecture as smartspace_xspace_ref.py; only the training-time data
distribution changes:
  - photometric jitter: random brightness/contrast/gamma per camera (simulate lighting variation)
  - DropView: randomly zero a fraction of the valid cameras each step (force rig-invariant fusion)
If unseen IoU rises materially over the no-DR run (0.046) with the architecture unchanged, the bottleneck
is DATA/DOMAIN diversity, not architecture. If it stays flat, architecture is implicated.

usage: python smartspace_xspace_ref_dr.py <train_dirs_csv> <val_dir> <out.pt> [epochs] [seed]
env: SMARTSPACE_BS, DROPVIEW (default 0.3), JITTER (default 1)
"""
import sys, os, glob
import numpy as np
import torch
import torch.nn as nn

# reuse the EXACT model from the base reference (no architecture change)
import importlib.util as _u
_b = os.path.join(os.path.dirname(__file__), "smartspace_xspace_ref.py")
_sp = _u.spec_from_file_location("xref", _b); xref = _u.module_from_spec(_sp); _sp.loader.exec_module(xref)
IPMBevX, evaluate, MEAN, STD = xref.IPMBevX, xref.evaluate, xref.MEAN, xref.STD


class DRSceneData(torch.utils.data.Dataset):
    """Same npz contract as xref.SceneData, plus train-time photometric jitter."""
    def __init__(self, dirs, jitter=True):
        self.files = []
        for d in dirs:
            self.files += sorted(glob.glob(os.path.join(d, "*.npz")))
        self.jitter = jitter

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        imgs = torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0  # (N,3,H,W) in [0,1]
        if self.jitter:
            N = imgs.shape[0]
            b = (0.7 + 0.6 * torch.rand(N, 1, 1, 1))          # brightness  x0.7..1.3
            c = (0.7 + 0.6 * torch.rand(N, 1, 1, 1))          # contrast
            g = (0.7 + 0.6 * torch.rand(N, 1, 1, 1))          # gamma
            mean = imgs.mean(dim=(2, 3), keepdim=True)
            imgs = ((imgs.clamp(1e-4) ** g) * b)
            imgs = (imgs - mean) * c + mean
            imgs = imgs.clamp(0, 1)
        imgs = (imgs - MEAN) / STD
        return (imgs, torch.from_numpy(d["cam_proj"]).float(), torch.from_numpy(d["cam_valid"]).float(),
                torch.from_numpy(d["grid_bounds"]).float(), torch.from_numpy(d["bev"]).float())


if __name__ == "__main__":
    train_dirs = sys.argv[1].split(","); val_dir = sys.argv[2]; out = sys.argv[3]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 18
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    dropview = float(os.environ.get("DROPVIEW", "0.3"))
    jitter = os.environ.get("JITTER", "1") != "0"
    torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
    bs = int(os.environ.get("SMARTSPACE_BS", "4"))
    tr = torch.utils.data.DataLoader(DRSceneData(train_dirs, jitter), batch_size=bs, shuffle=True,
                                     num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(xref.SceneData([val_dir]), batch_size=bs, num_workers=4)
    model = IPMBevX().to(dev)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, epochs=epochs, steps_per_epoch=max(len(tr), 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(40.0, device=dev))
    print(f"DR train {len(tr.dataset)} ({len(train_dirs)} scenes) | val {len(va.dataset)} (unseen) | "
          f"dropview={dropview} jitter={jitter}", flush=True)
    best = 0.0
    for ep in range(epochs):
        model.train()
        for imgs, cp, cv, gb, bev in tr:
            imgs, cp, cv, gb, bev = imgs.to(dev), cp.to(dev), cv.to(dev), gb.to(dev), bev.to(dev)
            if dropview > 0:                                   # DropView: drop some valid cams this step
                keep = (torch.rand_like(cv) > dropview).float()
                cvd = cv * keep
                cvd = torch.where(cvd.sum(1, keepdim=True) > 0, cvd, cv)   # never drop ALL cams
            else:
                cvd = cv
            loss = lossf(model(imgs, cp, cvd, gb), bev)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        miou = evaluate(model, va, dev)
        if miou > best:
            best = miou; torch.save(model.state_dict(), out)
        print(f"  epoch {ep:02d} loss {loss.item():.3f} val_IoU(unseen) {miou:.4f} (best {best:.4f})", flush=True)
    print(f"DR_DONE seed={seed} best_unseen_IoU={best:.4f} (no-DR baseline was 0.046) -> {out}", flush=True)
