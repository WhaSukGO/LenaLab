"""Animate the SLAM trajectory being built frame-by-frame (DROID estimate vs GT) -> the loop closing.
Runs in vo-droid (matplotlib + cv2). usage: python viz_trajectory_anim.py <traj.txt> <gt.txt> <out.mp4>"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cv2


def sim3(s, d):
    n = min(len(s), len(d)); s, d = s[:n], d[:n]; ms, md = s.mean(0), d.mean(0); a, b = s - ms, d - md
    U, D, Vt = np.linalg.svd((b.T @ a) / n); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    c = D.sum() / (a ** 2).sum() * n; return (c * (R @ a.T)).T + md


tr = np.loadtxt(sys.argv[1]); gt = np.loadtxt(sys.argv[2])
n = min(len(tr), len(gt)); tr, gt = tr[:n], gt[:n]
al = sim3(tr, gt)
lo = np.minimum(gt.min(0), al.min(0)) - 5; hi = np.maximum(gt.max(0), al.max(0)) + 5
W, Ht = 720, 720
vw = cv2.VideoWriter(sys.argv[3], cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, Ht))
fig = plt.figure(figsize=(7.2, 7.2), dpi=100); ax = fig.add_subplot(111)
step = max(1, n // 160)
for k in range(2, n + 1, step):
    ax.clear()
    ax.plot(gt[:k, 0], gt[:k, 2], "k-", lw=2.5, label="ground truth")
    ax.plot(al[:k, 0], al[:k, 2], "r--", lw=1.5, label="DROID-SLAM (stereo)")
    ax.scatter([al[k - 1, 0]], [al[k - 1, 2]], c="red", s=60, zorder=5)
    ax.scatter([gt[0, 0]], [gt[0, 2]], c="lime", s=70, marker="*", zorder=5, label="start")
    err = np.sqrt(((al[:k] - gt[:k]) ** 2).sum(1).mean())
    ax.set_title(f"DROID-SLAM building the seq_07 loop — frame {k}/{n}  (ATE so far {err:.2f}m)")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[2], hi[2]); ax.set_aspect("equal")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    vw.write(cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR))
# hold the final closed loop
for _ in range(40): vw.write(cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR))
vw.release()
plt.imsave(sys.argv[3].replace(".mp4", "_final.png"), buf)
print(f"  wrote {sys.argv[3]} ({n} poses) + final frame")
