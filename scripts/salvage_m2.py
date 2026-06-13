"""Salvage-grade the M2 attempt-2 WIP (interrupted by a machine restart) on the held-out 07/09,
locally, GT-isolated, on the OFFICIAL KITTI metric. Non-billed. Each held-out seq's gt lives as a
SIBLING of input/, so running main.py with LAB_DATA=seq/input is already GT-free."""
import os, shutil, subprocess, sys, tempfile, json, time
from pathlib import Path

ROOT = Path("/home/ws/devel/whasuk/LenaLab")
HO = ROOT / "_vo_kitti_slam_run2/cache/heldout/vo-kitti-heldout-07_09"
MAIN = ROOT / "artifacts/agent_authored_vo_kitti_slam_v2_wip.py"
art = Path(tempfile.mkdtemp(prefix="salvage_m2_"))

for sq in sorted(HO.glob("seq_*")):
    s = sq.name.replace("seq_", "")
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(tmp))
    print(f"running WIP SLAM main.py on held-out seq_{s} ({len(list((sq/'input').glob('left_*.png')))} frames) ...", flush=True)
    t0 = time.time()
    try:
        subprocess.run([sys.executable, str(MAIN)], env=env, check=True, timeout=1200)
        print(f"  seq_{s} done in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  seq_{s} FAILED: {str(e)[:200]}", flush=True)
    for f in ("traj.txt", "poses.txt"):
        if (tmp / f).exists():
            shutil.copy(tmp / f, art / f"{f.split('.')[0]}_{s}.txt")

evout = art / "eval"
env = dict(os.environ, LAB_DATA=str(HO), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(evout))
subprocess.run([sys.executable, str(ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py")], env=env, check=True)
d = json.load(open(evout / "heldout.json"))
print("\n=== M2 ATTEMPT-2 SALVAGE RESULT (official KITTI metric) ===")
print("mode:", d.get("metric_mode"), "| t_err%:", round(d["t_err_pct"], 3),
      "| r_err deg/m:", d.get("r_err_deg_m"), "| ATE m:", round(d["ate_rmse"], 3))
print("per-seq t_err%:", {k: round(v["t_err_pct"], 2) for k, v in d["per_seq"].items()})
print("anchors: basic VO 2.81% (07=2.41,09=3.22) | bar 1.8% | ideal closure 1.32% | ORB-SLAM2 1.15%")
print("attempt 1 was: 6.53% (07=2.61, 09=10.44 corrupted)")
