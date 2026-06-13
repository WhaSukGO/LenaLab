"""M2 DE-RISK (offline, non-billed): does loop closure have real headroom on KITTI, and does our
official t_err metric respond to it?

Stages FULL-strided seq_07 (1101 frames @ stride 3 -> ~367 frames, a genuine loop: frame ~1065
returns to frame ~14). Runs the REFERENCE stereo VO (times it vs the 15-min grader budget),
measures baseline t_err (no closure), then applies an IDEAL pose-graph loop closure (GT-detected
loop pair = upper bound on place recognition) by distributing the loop drift along the chain, and
re-measures t_err. If closed << baseline, M2 has headroom and is worth a billed run.
"""
import os, sys, time, subprocess, tempfile, json, shutil
from pathlib import Path
import numpy as np
import cv2

ROOT = Path("/home/ws/devel/whasuk/LenaLab")
sys.path.insert(0, str(ROOT))
from vo_lab.plugins.vo_kitti import KITTIOdomProvider

STRIDE, MAXF, SEQ = 3, 400, "07"
work = Path(tempfile.mkdtemp(prefix="m2derisk_"))
hd = work / "heldout"; sq = hd / f"seq_{SEQ}"; (sq / "input").mkdir(parents=True)
art = work / "art"; art.mkdir(); ev = work / "ev"

print(f"[1] staging full-strided seq_{SEQ} (stride {STRIDE}, max {MAXF}) ...", flush=True)
prov = KITTIOdomProvider(dev="00", heldout=(SEQ,), stride=STRIDE, max_frames=MAXF)
n = prov._materialize(SEQ, sq / "input", gt_path=sq / "gt.txt")
print(f"    staged {n} frames", flush=True)

print("[2] running reference stereo VO (timing vs 15-min budget) ...", flush=True)
env = dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(art))
t0 = time.time()
subprocess.run([sys.executable, str(ROOT / "vo_lab/plugins/vo_ref/run_kitti_stereo.py")],
               env=env, check=True)
dt = time.time() - t0
print(f"    VO ran {dt:.1f}s for {n} frames ({dt/n*1000:.0f} ms/frame); full-rate "
      f"~{dt:.0f}s vs 900s budget", flush=True)
shutil.copy(art / "traj.txt", art / f"traj_{SEQ}.txt")
shutil.copy(art / "poses.txt", art / f"poses_{SEQ}.txt")

def eval_terr(tag):
    e = ev / tag; e.mkdir(parents=True, exist_ok=True)
    env2 = dict(os.environ, LAB_DATA=str(hd), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(e))
    r = subprocess.run([sys.executable, str(ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py")],
                       env=env2, capture_output=True, text=True)
    f = e / "heldout.json"
    if not f.exists():
        print(f"    eval_{tag} no json. stderr:", r.stderr[-400:]); return None
    return json.load(open(f))

print("[3] baseline (no loop closure):", flush=True)
base = eval_terr("base")
if base: print(f"    t_err = {base['t_err_pct']:.3f}%  r_err = {base.get('r_err_deg_m')}  mode={base.get('metric_mode')}")

# [4] ideal loop closure: distribute loop drift along the chain
poses_vo = np.loadtxt(art / f"poses_{SEQ}.txt").reshape(-1, 3, 4)
G = np.loadtxt(sq / "gt_poses.txt").reshape(-1, 3, 4)
def to44(P): T = np.eye(4); T[:3, :] = P; return T
gc = G[:, :, 3]
R, gap = 15.0, 60; best = None
for j in range(gap + 1, len(gc)):
    dd = np.linalg.norm(gc[:j - gap] - gc[j], axis=1); i = int(dd.argmin())
    if dd[i] < R and (best is None or dd[i] < best[2]): best = (i, j, float(dd[i]))
if best is None:
    print("[4] no loop pair found in strided GT -> cannot close"); sys.exit(0)
i, j, dist = best
print(f"[4] ideal loop closure: revisit pair ({i},{j}) {dist:.1f}m apart in GT", flush=True)
Tvo = [to44(p) for p in poses_vo]
M = np.linalg.inv(to44(G[i])) @ to44(G[j])     # ideal relative i->j (place-recognition upper bound)
desired_j = Tvo[i] @ M
C = desired_j @ np.linalg.inv(Tvo[j])          # world correction needed at j
rc, _ = cv2.Rodrigues(C[:3, :3]); rc = rc.ravel(); tc = C[:3, 3]
print(f"    loop drift to distribute: |t|={np.linalg.norm(tc):.1f}m  |rot|={np.degrees(np.linalg.norm(rc)):.1f}deg", flush=True)
closed = []
for k in range(len(Tvo)):
    f = 0.0 if k < i else (1.0 if k > j else (k - i) / max(1, j - i))
    Rk, _ = cv2.Rodrigues(rc * f); Ck = np.eye(4); Ck[:3, :3] = Rk; Ck[:3, 3] = tc * f
    closed.append((Ck @ Tvo[k])[:3, :])
np.savetxt(art / f"poses_{SEQ}.txt", np.array(closed).reshape(len(closed), 12), fmt="%.8e")
np.savetxt(art / f"traj_{SEQ}.txt", np.array([T[:, 3] for T in closed]), fmt="%.6f")
print("[5] after ideal loop closure:", flush=True)
cl = eval_terr("closed")
if cl: print(f"    t_err = {cl['t_err_pct']:.3f}%  r_err = {cl.get('r_err_deg_m')}")

print("\n=== M2 HEADROOM VERDICT (full-strided seq_07, real loop) ===")
if base and cl:
    b, c = base['t_err_pct'], cl['t_err_pct']
    print(f"  baseline VO (no closure): {b:.3f}%")
    print(f"  ideal loop closure      : {c:.3f}%   ({(b-c)/b*100:+.0f}% relative)")
    print(f"  ORB-SLAM2 anchor        : 1.15%   |   VO ran {dt:.0f}s vs 900s budget")
    print(f"  -> {'HEADROOM: M2 worth billing' if c < b*0.85 else 'little gain here -> reconsider M2'}")
print(f"\n(workdir {work})")
