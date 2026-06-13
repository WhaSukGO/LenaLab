"""Train 3D Gaussians (gsplat) on a clip of KITTI seq_07, INITIALIZED from DROID's dense reconstruction
(poses + dense depth + images give us SfM for free). Optimize Gaussians to match the real frames, then
render the training views (real-vs-rendered) + a novel interpolated view. Runs in vo-gsplat:1.

usage: python train_gsplat_seq07.py <recon.pth> <out_dir> [clip_start] [clip_len] [iters]
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from gsplat import rasterization

recon, outdir = sys.argv[1], sys.argv[2]
A = int(sys.argv[3]) if len(sys.argv) > 3 else 120
LEN = int(sys.argv[4]) if len(sys.argv) > 4 else 40
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
os.makedirs(outdir, exist_ok=True)
dev = "cuda"


def quat_to_R(q):                                            # q=[qx,qy,qz,qw]
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


d = torch.load(recon, map_location="cpu")
poses, disps = d["poses"].numpy(), d["disps"].numpy()
images, intr = d["images"].numpy(), d["intrinsics"].numpy()
N, H, W = disps.shape
A = min(A, N - LEN); idx = list(range(A, A + LEN))
print(f"clip frames {A}..{A+LEN} of {N}, depth {H}x{W}", flush=True)

# --- camera params (world->cam viewmats + full-res Ks) + GT images ---
viewmats, Ks, gts = [], [], []
for i in idx:
    R = quat_to_R(poses[i, 3:7]); t = poses[i, :3]
    vm = np.eye(4); vm[:3, :3] = R; vm[:3, 3] = t       # world->cam
    viewmats.append(vm)
    fx, fy, cx, cy = intr[i] * 8.0
    Ks.append([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    img = images[i].transpose(1, 2, 0)[:, :, ::-1] / 255.0   # BGR->RGB
    gts.append(img)
viewmats = torch.tensor(np.array(viewmats), dtype=torch.float32, device=dev)
Ks = torch.tensor(np.array(Ks), dtype=torch.float32, device=dev)
gts = torch.tensor(np.array(gts), dtype=torch.float32, device=dev)

# --- init Gaussians from back-projected dense points of the clip ---
pts, cols = [], []
for j, i in enumerate(idx):
    fx, fy, cx, cy = intr[i] * 8.0
    R = quat_to_R(poses[i, 3:7]); t = poses[i, :3]; C = -R.T @ t
    depth = 1.0 / np.clip(disps[i], 1e-4, None)
    vs, us = np.mgrid[0:H:2, 0:W:2]; vs, us = vs.ravel(), us.ravel()
    z = depth[vs, us]; m = (z > 0.5) & (z < 60)
    vs, us, z = vs[m], us[m], z[m]
    cam = np.stack([z * (us - cx) / fx, z * (vs - cy) / fy, z], 1)
    pts.append(cam @ R + C)
    cols.append(gts[j].cpu().numpy()[vs, us])
pts = np.concatenate(pts); cols = np.concatenate(cols)
k = np.random.default_rng(0).choice(len(pts), min(150000, len(pts)), replace=False)
pts, cols = pts[k], cols[k]
print(f"init {len(pts)} gaussians", flush=True)

# DIAGNOSTIC: manually project the init points through frame-0 viewmat -> where do they land?
vm0 = viewmats[0].cpu().numpy(); K0 = Ks[0].cpu().numpy()
Xc = (vm0[:3, :3] @ pts.T + vm0[:3, 3:4]).T            # world->cam
front = Xc[:, 2] > 0.01
uv = (K0 @ Xc[front].T).T; uv = uv[:, :2] / uv[:, 2:3]
onx = (uv[:, 0] >= 0) & (uv[:, 0] < W); ony = (uv[:, 1] >= 0) & (uv[:, 1] < H)
print(f"DIAG frame0: {front.mean()*100:.0f}% in front of cam; of those {((onx&ony).mean())*100:.0f}% on-screen; "
      f"u range [{uv[:,0].min():.0f},{uv[:,0].max():.0f}] v [{uv[:,1].min():.0f},{uv[:,1].max():.0f}] (img {W}x{H})", flush=True)

means = torch.tensor(pts, dtype=torch.float32, device=dev, requires_grad=True)
# init scale ~ mean spacing (rough)
extent = np.linalg.norm(pts.max(0) - pts.min(0))
# small scale (huge scales -> everything blends to gray). default 0.5% of extent; tune via SCALE_FRAC.
s0 = np.log(extent * float(os.environ.get("SCALE_FRAC", "0.0004")) + 1e-5)
log_scales = torch.full((len(pts), 3), float(s0), device=dev, requires_grad=True)
quats = torch.zeros(len(pts), 4, device=dev); quats[:, 0] = 1.0; quats.requires_grad_(True)
logit_op = torch.full((len(pts),), 1.5, device=dev, requires_grad=True)   # sigmoid(1.5)~0.82 (occlude, not bleed)
colc = np.clip(cols, 0.01, 0.99)                            # init colors as LOGIT (render applies sigmoid)
colors = torch.tensor(np.log(colc / (1 - colc)), dtype=torch.float32, device=dev, requires_grad=True)

def _render(vm, K, m, q, ls, lo, co):
    with torch.no_grad():
        out, _, _ = rasterization(m, F.normalize(q, dim=-1), torch.exp(ls), torch.sigmoid(lo),
                                  torch.sigmoid(co), vm, K, W, H, sh_degree=None, near_plane=0.5)
    return (out[0].clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)


# DIAGNOSTIC: render the UNTRAINED init from frame 0 (is the geometry/convention right BEFORE training?)
cv2.imwrite(os.path.join(outdir, "gsplat_init_render.png"),
            np.hstack([(gts[0].cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8),
                       _render(viewmats[0:1], Ks[0:1], means, quats, log_scales, logit_op, colors)]))
print(f"init render done (extent={extent:.2f}, scale0={np.exp(s0):.4f})", flush=True)

opt = torch.optim.Adam([{"params": [means], "lr": extent * 1e-4},
                        {"params": [log_scales], "lr": 5e-3},
                        {"params": [quats], "lr": 1e-3},
                        {"params": [logit_op], "lr": 5e-2},
                        {"params": [colors], "lr": 1e-2}])
for it in range(ITERS):
    c = np.random.randint(len(idx))
    out, _, _ = rasterization(means, F.normalize(quats, dim=-1), torch.exp(log_scales),
                              torch.sigmoid(logit_op), torch.sigmoid(colors),
                              viewmats[c:c + 1], Ks[c:c + 1], W, H, sh_degree=None, near_plane=0.5)
    loss = F.l1_loss(out[0], gts[c])
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 200 == 0 or it == ITERS - 1:
        print(f"  iter {it}: L1 {loss.item():.4f}", flush=True)

# --- render: real vs rendered (3 frames) + a novel mid-baseline view ---
def render(vm, K):
    with torch.no_grad():
        out, _, _ = rasterization(means, F.normalize(quats, dim=-1), torch.exp(log_scales),
                                  torch.sigmoid(logit_op), torch.sigmoid(colors), vm, K, W, H, sh_degree=None, near_plane=0.5)
    return (out[0].clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)


rows = []
for c in [0, len(idx) // 2, len(idx) - 1]:
    real = (gts[c].cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
    rend = render(viewmats[c:c + 1], Ks[c:c + 1])
    rows.append(np.hstack([real, np.full((H, 6, 3), 255, np.uint8), rend]))
sep = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
stacked = rows[0]
for r in rows[1:]:
    stacked = np.vstack([stacked, sep, r])
cv2.imwrite(os.path.join(outdir, "gsplat_seq07_realvsrendered.png"), stacked)
# novel view: midpoint between cam 0 and 1, lifted a bit
vm_novel = viewmats[len(idx) // 2:len(idx) // 2 + 1].clone()
vm_novel[0, 1, 3] += 0.3 * (extent / len(idx))             # shift down (novel viewpoint real data lacks)
cv2.imwrite(os.path.join(outdir, "gsplat_seq07_novel.png"), render(vm_novel, Ks[len(idx) // 2:len(idx) // 2 + 1]))
print("wrote real-vs-rendered + novel view", flush=True)
torch.save({"means": means.detach().cpu(), "quats": quats.detach().cpu(), "log_scales": log_scales.detach().cpu(),
            "logit_op": logit_op.detach().cpu(), "colors": colors.detach().cpu()},
           os.path.join(outdir, "gaussians_seq07.pth"))
print("saved gaussians", flush=True)
