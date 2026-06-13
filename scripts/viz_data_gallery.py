"""Data gallery: one representative input frame from each of the three worlds the agent worked in —
TUM indoor handheld (Episodes 1-5), KITTI outdoor driving (Episode 6, M1/M2, sim-to-real), and the
contamination-clean synthetic corridor (probe, M3, rung 3). So a reader can SEE what each algorithm
was actually looking at, not just its error number."""
import cv2, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
H = Path.home()
R = Path("/home/ws/devel/whasuk/LenaLab")

frames = [
    (H / ".cache/vo_lab/tum/rgbd_dataset_freiburg1_desk/rgb/1305031459.427646.png",
     "TUM indoor (handheld RGB-D)",
     "Episodes 1-5  ·  a desk, ~2-5 m room  ·  hand-held wobble\nscene scale: centimetres matter"),
    (H / ".cache/vo_lab/kitti/dataset/sequences/07/image_0/000200.png",
     "KITTI outdoor (car, stereo grey)",
     "Episode 6, M1/M2, sim-to-real  ·  a car driving a town, ~hundreds of m\nscene scale: metres over a ~km drive"),
    (R / "_vo_synth_learned_impl_run/cache/data/vo-synthlearn-train/train/seq_ltrA/left_000080.png",
     "Synthetic (procedural, ray-cast)",
     "contamination probe, M3, rung 3  ·  a textured corridor\nexact ground truth + provably-unseen (can't be memorised)"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
for ax, (p, title, sub) in zip(axes, frames):
    img = cv2.imread(str(p))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    if img is not None:
        ax.imshow(img)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(sub, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("The three worlds the agent worked in — what each algorithm actually 'saw'", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = R / "artifacts/blog/data_gallery.png"
fig.savefig(out, dpi=120, bbox_inches="tight"); print("wrote", out)
