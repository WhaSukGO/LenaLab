"""PRODUCTION per-space floor-occupancy model. Builds on the agent's winning recipe (IPM + temporal
background subtraction + 4-plane mean/max fusion + focal loss) but upgrades the backbone to an
ImageNet-PRETRAINED ResNet18 (the lever the offline sandbox couldn't use), finer features (stride-8),
val-IoU model selection, longer training. Per-space = the deployable unit.

Trains on <cache>/train, evaluates held-out IoU on <cache>/val (later time, same warehouse).
Saves: <out>.pt (model state_dict), <out>.bg.npy (background), <out>.meta.json (thr, val IoU).

usage: python smartspace_prod.py <cache_dir> <out_prefix> [epochs] [seed]
env: PROD_BS (default 4), PROD_BACKBONE_LAYER (2=stride8 default | 3=stride16)
"""
import sys, os, glob, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMG_H, IMG_W = 128, 352
XG, YG = 206, 203
CELL = 0.5
HEIGHTS = [0.0, 0.5, 1.0, 1.5]
FEAT_CH = 64
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYER = int(os.environ.get("PROD_BACKBONE_LAYER", "2"))


def compute_background(train_dir):
    files = sorted(glob.glob(os.path.join(train_dir, "*.npz")))
    acc = None  # running median via sampling to bound memory
    stack = []
    for f in files:
        d = np.load(f); stack.append(np.transpose(d["imgs"].astype(np.float32) / 255., (0, 3, 1, 2)))
    bg = np.median(np.stack(stack, 0), 0).astype(np.float32)  # (N,3,H,W)
    return bg


def build_sample_grids(cam_proj, grid_bounds, ncam):
    x0, x1, y0, y1 = grid_bounds
    xs = x0 + CELL * (np.arange(XG, dtype=np.float32) + 0.5)
    ys = y0 + CELL * (np.arange(YG, dtype=np.float32) + 0.5)
    xx, yy = np.meshgrid(xs, ys, indexing="ij"); ones = np.ones_like(xx)
    gs, ms = [], []
    for z in HEIGHTS:
        pts = np.stack([xx, yy, np.full_like(xx, z), ones], 0).reshape(4, -1)
        uvw = cam_proj @ pts; w = uvw[:, 2, :]; sw = np.where(w > 1e-6, w, 1.0)
        u, v = uvw[:, 0, :] / sw, uvw[:, 1, :] / sw
        u_n = (2 * u / (IMG_W - 1) - 1).reshape(ncam, XG, YG).astype(np.float32)
        v_n = (2 * v / (IMG_H - 1) - 1).reshape(ncam, XG, YG).astype(np.float32)
        valid = ((w > 1e-6) & (u >= 0) & (u <= IMG_W - 1) & (v >= 0) & (v <= IMG_H - 1)).reshape(ncam, XG, YG).astype(np.float32)
        gs.append(np.stack([u_n, v_n], -1)); ms.append(valid)
    return torch.from_numpy(np.stack(gs, 0)), torch.from_numpy(np.stack(ms, 0))


class FloorDS(torch.utils.data.Dataset):
    def __init__(self, d, bg, train=True):
        self.files = sorted(glob.glob(os.path.join(d, "*.npz"))); self.bg = bg; self.train = train

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        imgs = np.transpose(d["imgs"].astype(np.float32) / 255., (0, 3, 1, 2))
        diff = imgs - self.bg
        it = torch.from_numpy(imgs); dt = torch.from_numpy(diff)
        if self.train:                                # light photometric aug (per-space robustness)
            b = 0.85 + 0.3 * torch.rand(it.shape[0], 1, 1, 1); it = (it * b).clamp(0, 1)
        inp = torch.cat([(it - MEAN) / STD, dt / 0.15], 1)   # (N,6,H,W)
        bev = torch.from_numpy(d["bev"].astype(np.float32)) if "bev" in d.files else torch.zeros(XG, YG)
        return inp, bev


class Backbone(nn.Module):
    """6-ch in -> pretrained ResNet18 (conv1 inflated) -> FEAT_CH at stride 8 (LAYER=2) or 16 (LAYER=3)."""
    def __init__(self):
        super().__init__()
        bb = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        w = bb.conv1.weight.data                                  # (64,3,7,7)
        conv1 = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            conv1.weight[:, :3] = w; conv1.weight[:, 3:] = w * 0.5   # bg-diff inits from rgb filters
        layers = [conv1, bb.bn1, bb.relu, bb.maxpool, bb.layer1]; outc = 64   # LAYER=1 -> stride 4 (fine)
        if LAYER >= 2:
            layers.append(bb.layer2); outc = 128                             # stride 8
        if LAYER >= 3:
            layers.append(bb.layer3); outc = 256                            # stride 16
        self.net = nn.Sequential(*layers)
        self.reduce = nn.Conv2d(outc, FEAT_CH, 1)

    def forward(self, x): return self.reduce(self.net(x))


class Lifter(nn.Module):
    def __init__(self, grids, masks):
        super().__init__(); self.register_buffer("g", grids); self.register_buffer("m", masks)
        self.nh = grids.shape[0]

    def forward(self, f):
        B, N, C, Hf, Wf = f.shape; ff = f.reshape(B * N, C, Hf, Wf); out = []
        for hi in range(self.nh):
            ge = self.g[hi].unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * N, XG, YG, 2)
            s = F.grid_sample(ff, ge, mode="bilinear", padding_mode="zeros", align_corners=True).reshape(B, N, C, XG, YG)
            mk = self.m[hi].unsqueeze(0).unsqueeze(2)
            nv = mk.sum(1).clamp(min=1)
            out += [(s * mk).sum(1) / nv, (s * mk + (mk - 1) * 1e4).max(1)[0].clamp(min=-1e3)]
        return torch.cat(out, 1)


class ProdModel(nn.Module):
    def __init__(self, grids, masks):
        super().__init__()
        self.bb = Backbone(); self.lift = Lifter(grids, masks)
        self.head = nn.Sequential(
            nn.Conv2d(FEAT_CH * len(HEIGHTS) * 2, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 3, 1, 2, dilation=2), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(0.1),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 1, 1))

    def forward(self, imgs):
        B, N, C, H, W = imgs.shape
        f = self.bb(imgs.reshape(B * N, C, H, W))
        f = f.reshape(B, N, FEAT_CH, f.shape[-2], f.shape[-1])
        return self.head(self.lift(f)).squeeze(1)


def focal(logits, t, a=0.97, g=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits); pt = p * t + (1 - p) * (1 - t); at = a * t + (1 - a) * (1 - t)
    return (at * (1 - pt) ** g * bce).mean()


@torch.no_grad()
def eval_iou(model, loader, dev, thr=0.5):
    model.eval(); probs, gts = [], []
    for inp, bev in loader:
        lg = model(inp.to(dev)); probs.append(torch.sigmoid(lg).cpu()); gts.append(bev)
    P = torch.cat(probs).numpy(); G = torch.cat(gts).numpy()
    best = (0.5, 0.0)
    for t in np.arange(0.05, 0.96, 0.05):
        pr = P > t; i = (pr * (G > 0.5)).sum(); u = (pr | (G > 0.5)).sum()
        iou = i / u if u else 0.0
        if iou > best[1]: best = (float(t), float(iou))
    return best  # (thr, iou)


if __name__ == "__main__":
    cache, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    torch.manual_seed(seed); np.random.seed(seed); dev = "cuda"
    bs = int(os.environ.get("PROD_BS", "4"))
    print(f"backbone=ResNet18(pretrained,layer{LAYER}) bs={bs} epochs={epochs}", flush=True)
    bg = compute_background(os.path.join(cache, "train"))
    ref = np.load(sorted(glob.glob(os.path.join(cache, "train", "*.npz")))[0])
    ncam = ref["cam_proj"].shape[0]
    grids, masks = build_sample_grids(ref["cam_proj"], ref["grid_bounds"], ncam)
    grids, masks = grids.to(dev), masks.to(dev)
    tr = torch.utils.data.DataLoader(FloorDS(os.path.join(cache, "train"), bg, True), batch_size=bs, shuffle=True, num_workers=4, drop_last=True)
    va = torch.utils.data.DataLoader(FloorDS(os.path.join(cache, "val"), bg, False), batch_size=bs, num_workers=4)
    model = ProdModel(grids, masks).to(dev)
    print(f"params={sum(p.numel() for p in model.parameters()):,} | train {len(tr.dataset)} val {len(va.dataset)}", flush=True)
    bb_ids = {id(p) for p in model.bb.parameters()}            # discriminative LR: pretrained backbone gentler
    groups = [{"params": list(model.bb.parameters()), "lr": 1.5e-4},
              {"params": [p for p in model.parameters() if id(p) not in bb_ids], "lr": 5e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1.5e-4, 5e-4], epochs=epochs, steps_per_epoch=len(tr), pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda")
    best_iou, best_thr = 0.0, 0.5
    for ep in range(epochs):
        model.train(); tot = 0
        for inp, bev in tr:
            inp, bev = inp.to(dev), bev.to(dev)
            with torch.amp.autocast("cuda"):
                loss = focal(model(inp), bev)
            if not torch.isfinite(loss):                       # NaN guard: skip bad batch, don't poison weights
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            thr, iou = eval_iou(model, va, dev)
            tag = ""
            if iou > best_iou:
                best_iou, best_thr = iou, thr
                torch.save(model.state_dict(), out + ".pt"); tag = " *BEST*"
            print(f"E{ep+1:4d} loss={tot/len(tr):.4f} val_IoU={iou:.4f}@{thr:.2f} (best {best_iou:.4f}){tag}", flush=True)
    np.save(out + ".bg.npy", bg)
    json.dump({"val_iou": best_iou, "thr": best_thr, "backbone": f"resnet18_layer{LAYER}", "epochs": epochs},
              open(out + ".meta.json", "w"), indent=2)
    print(f"PROD_DONE best_val_IoU={best_iou:.4f}@{best_thr:.2f} -> {out}.pt (agent baseline ~0.39)", flush=True)
