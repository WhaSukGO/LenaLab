"""DroidAdapter runtime — runs INSIDE the vo-droid:1 container. Takes our seq input (left_*.png +
intrinsics.txt fx fy cx cy baseline) and runs DROID-SLAM (monocular), then writes our grading contract:
  <out>/poses_<seq>.txt (KITTI 3x4 cam->world, 12/line) + <out>/traj_<seq>.txt (tx ty tz centres).
DROID's traj_est is (N,7) [tx ty tz qx qy qz qw] camera-to-world per kept frame; we map quaternion->R.

usage (in container): python run_droid_slam.py <seq_input_dir> <out_dir> <seq> <weights.pth> [stride]
"""
import sys, os, glob
import numpy as np
import torch
import cv2

sys.path.append("/droid/droid_slam")
from droid import Droid                                      # noqa: E402


STEREO = os.environ.get("DROID_STEREO", "0") == "1"   # stereo -> metric scale (kills monocular scale-drift)
RECON = os.environ.get("DROID_RECON", "0") == "1"     # save dense reconstruction (poses+depth+images) for mapping


def image_stream(imagedir, fx, fy, cx, cy, stride):
    lefts = sorted(glob.glob(os.path.join(imagedir, "left_*.png")))[::stride]
    rights = sorted(glob.glob(os.path.join(imagedir, "right_*.png")))[::stride] if STEREO else None
    for t, f in enumerate(lefts):
        raw = [cv2.imread(f)] + ([cv2.imread(rights[t])] if STEREO else [])   # [left] or [left,right], 3-ch
        h0, w0 = raw[0].shape[:2]
        area = float(os.environ.get("DROID_AREA", str(384 * 512)))
        h1 = int(h0 * np.sqrt(area / (h0 * w0)))
        w1 = int(w0 * np.sqrt(area / (h0 * w0)))
        views = []
        for im in raw:
            im = cv2.resize(im, (w1, h1))[: h1 - h1 % 8, : w1 - w1 % 8]
            views.append(torch.as_tensor(im).permute(2, 0, 1))
        image = torch.stack(views, 0)                        # [N,3,H,W]  (N=1 mono, N=2 stereo)
        intr = torch.as_tensor([fx, fy, cx, cy])
        intr[0::2] *= (w1 / w0); intr[1::2] *= (h1 / h0)
        yield t, image, intr


class Args:                                                  # Droid() expects an argparse-like namespace
    def __init__(self, weights):
        self.weights = weights; self.image_size = [240, 320]
        self.buffer = int(os.environ.get("DROID_BUFFER", "1024"))  # km-scale seqs need a big keyframe buffer (48GB RAM)
        self.disable_vis = True; self.upsample = RECON; self.stereo = STEREO  # upsample=full-res depth for mapping
        self.beta = 0.3; self.filter_thresh = 2.4; self.warmup = 8
        self.keyframe_thresh = 4.0; self.frontend_thresh = 16.0; self.frontend_window = 25
        self.frontend_radius = 2; self.frontend_nms = 1
        self.backend_thresh = 22.0; self.backend_radius = 2; self.backend_nms = 3
        self.asynchronous = False; self.frontend_device = "cuda"; self.backend_device = "cuda"


def quat_to_R(q):                                            # q = [qx qy qz qw]
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    seq_input, out_dir, seq, weights = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    stride = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    fx, fy, cx, cy, _b = np.loadtxt(os.path.join(seq_input, "intrinsics.txt"))
    torch.multiprocessing.set_start_method("spawn", force=True)
    droid = None
    for (t, image, intr) in image_stream(seq_input, fx, fy, cx, cy, stride):
        if droid is None:
            a = Args(weights); a.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(a)
        droid.track(t, image, intrinsics=intr)
    traj = droid.terminate(image_stream(seq_input, fx, fy, cx, cy, stride))  # (N,7) cam->world
    os.makedirs(out_dir, exist_ok=True)
    pf = open(os.path.join(out_dir, f"poses_{seq}.txt"), "w")
    tf = open(os.path.join(out_dir, f"traj_{seq}.txt"), "w")
    for p in traj:
        t3 = p[:3]; R = quat_to_R(p[3:7])
        pf.write(" ".join(f"{v:.8e}" for v in [*R[0], t3[0], *R[1], t3[1], *R[2], t3[2]]) + "\n")
        tf.write(f"{t3[0]:.6f} {t3[1]:.6f} {t3[2]:.6f}\n")
    print(f"DROID wrote {len(traj)} poses for {seq}")
    if RECON:                                            # dense reconstruction: per-keyframe poses+depth+images
        v = droid.video; n = v.counter.value
        torch.save({"tstamps": v.tstamp[:n].cpu(), "images": v.images[:n].cpu(),
                    "disps": v.disps_up[:n].cpu(), "poses": v.poses[:n].cpu(),
                    "intrinsics": v.intrinsics[:n].cpu()}, os.path.join(out_dir, f"recon_{seq}.pth"))
        print(f"DROID saved reconstruction ({n} keyframes) for {seq}")


if __name__ == "__main__":
    main()
