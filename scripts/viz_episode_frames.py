"""Per-episode inline input frames: Ep1 TUM mono (grayscale, as the algorithm saw it), Ep3 RGB-D
(RGB + the depth channel that defines the episode), Ep6 KITTI driving. Small, scene-grounding images."""
import cv2, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
H = Path.home(); R = Path("/home/ws/devel/whasuk/LenaLab")
TUM = H / ".cache/vo_lab/tum"

# Ep1 — TUM fr1_xyz, monocular grayscale (what the mono VO actually ingested)
f = sorted((TUM / "rgbd_dataset_freiburg1_xyz/rgb").glob("*.png"))[150]
g = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
fig, ax = plt.subplots(figsize=(5.2, 4)); ax.imshow(g, cmap="gray")
ax.set_title("Episode 1 input — TUM fr1_xyz, monocular grayscale", fontsize=10)
ax.set_xlabel("indoor desk, hand-held · one camera, no depth, no scale", fontsize=8.5)
ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout()
fig.savefig(R / "artifacts/blog/ep1_data_tum.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# Ep3 — RGB-D: RGB + depth channel (depth = the new ingredient that restores metric scale)
rgb = cv2.cvtColor(cv2.imread(str(sorted((TUM / "rgbd_dataset_freiburg1_xyz/rgb").glob("*.png"))[150])), cv2.COLOR_BGR2RGB)
dep = cv2.imread(str(sorted((TUM / "rgbd_dataset_freiburg1_xyz/depth").glob("*.png"))[150]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
dep[dep == 0] = np.nan
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(rgb); ax[0].set_title("RGB", fontsize=10)
im = ax[1].imshow(dep, cmap="turbo"); ax[1].set_title("Depth (metres) — restores real scale", fontsize=10)
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="m")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.suptitle("Episode 3 input — TUM RGB-D: every pixel now has a distance", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(R / "artifacts/blog/ep3_data_rgbd.png", dpi=120, bbox_inches="tight"); plt.close(fig)

# Ep6 — KITTI driving
k = cv2.cvtColor(cv2.imread(str(H / ".cache/vo_lab/kitti/dataset/sequences/07/image_0/000200.png")), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(8, 2.7)); ax.imshow(k)
ax.set_title("Episode 6 input — KITTI seq 07, outdoor driving (forward camera)", fontsize=10)
ax.set_xlabel("a car driving a town · long forward motion, metres over a ~km drive", fontsize=8.5)
ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout()
fig.savefig(R / "artifacts/blog/ep6_data_kitti.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("wrote ep1_data_tum.png, ep3_data_rgbd.png, ep6_data_kitti.png")
