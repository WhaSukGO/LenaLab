"""HARNESS-OWNED BEV grader. The solver provides a model definition + trained weights; this script
owns the held-out val split, the ground-truth BEV, the IoU metric, and the decision threshold.

The solver never sees val GT and cannot change the metric: the grader re-loads GT from the
harness cache and computes IoU itself. Tampering with the metric here earns nothing because the
harness runs THIS file, not the solver's copy.

usage: python grade_bev.py <solver_model.py> <checkpoint.pt> <data_root> [bar]
  <solver_model.py> must expose `build_model()` -> nn.Module taking (imgs, intrins, cam2ego)
                    and returning per-cell logits of shape (B, XG, YG).
prints: per-scene IoU, mean IoU, and VERIFIED/REJECTED vs bar.
"""
import sys, os, glob, json, importlib.util
import numpy as np
import torch

SOLVER, CKPT, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
BAR = float(sys.argv[4]) if len(sys.argv) > 4 else None
dev = "cuda" if torch.cuda.is_available() else "cpu"

# --- load the solver's model definition (code only; GT/metric stay harness-owned) ---
spec = importlib.util.spec_from_file_location("solver", SOLVER)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
model = mod.build_model().to(dev).eval()
model.load_state_dict(torch.load(CKPT, map_location=dev))

# --- harness-owned held-out data + GT + metric ---
meta = json.load(open(os.path.join(ROOT, "meta.json")))
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
files = sorted(glob.glob(os.path.join(ROOT, "val", "*.npz")))
inter_t = union_t = 0
per = []
with torch.no_grad():
    for f in files:
        d = np.load(f)
        imgs = torch.from_numpy(d["imgs"]).float().permute(0, 3, 1, 2) / 255.0
        imgs = ((imgs - mean) / std).unsqueeze(0).to(dev)
        K = torch.from_numpy(d["intrins"]).float().unsqueeze(0).to(dev)
        c2e = torch.from_numpy(d["cam2ego"]).float().unsqueeze(0).to(dev)
        gt = torch.from_numpy(d["bev"]).bool().to(dev)
        logits = model(imgs, K, c2e)[0]
        pred = logits > 0.0
        inter = (pred & gt).sum().item(); union = (pred | gt).sum().item()
        inter_t += inter; union_t += union
        per.append(inter / union if union > 0 else float("nan"))

mean_iou = float(np.nanmean(per))           # per-sample mean IoU (primary metric)
global_iou = inter_t / union_t              # dataset-level IoU (secondary)
print(f"held-out val: {len(files)} samples over scenes {meta['val_scenes']}")
print(f"  per-sample mean IoU = {mean_iou:.4f}")
print(f"  global (pooled) IoU = {global_iou:.4f}")
if BAR is not None:
    ok = mean_iou >= BAR
    print(f"  bar = {BAR:.4f}  ->  {'VERIFIED' if ok else 'REJECTED'}")
    sys.exit(0 if ok else 1)
