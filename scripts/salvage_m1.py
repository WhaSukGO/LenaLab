"""One-off: salvage-grade the M1 work-in-progress (agent's BA main.py) on the held-out, locally,
GT-isolated, on the OFFICIAL KITTI metric. Non-billed. The held-out gt lives as a SIBLING of
each seq's input/ dir, so running main.py with LAB_DATA=seq/input is already GT-free."""
import os, shutil, subprocess, sys, tempfile, json
from pathlib import Path

ROOT = Path("/home/ws/devel/whasuk/LenaLab")
HO = ROOT / "_vo_kitti_ba_run/cache/heldout/vo-kitti-heldout-05_07"
MAIN = ROOT / "artifacts/agent_authored_vo_kitti_ba_wip.py"
art = Path(tempfile.mkdtemp())

for sq in sorted(HO.glob("seq_*")):
    s = sq.name.replace("seq_", "")
    tmp = Path(tempfile.mkdtemp())
    env = dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(tmp))
    print(f"running agent BA main.py on held-out seq_{s} ...", flush=True)
    try:
        subprocess.run([sys.executable, str(MAIN)], env=env, check=True, timeout=900)
    except Exception as e:
        print(f"  seq_{s} FAILED: {str(e)[:200]}", flush=True)
    for f in ("traj.txt", "poses.txt"):
        if (tmp / f).exists():
            shutil.copy(tmp / f, art / f"{f.split('.')[0]}_{s}.txt")

evout = art / "eval"
env = dict(os.environ, LAB_DATA=str(HO), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(evout))
subprocess.run([sys.executable, str(ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py")], env=env, check=True)
d = json.load(open(evout / "heldout.json"))
print("\n=== M1 SALVAGE RESULT (official KITTI metric) ===")
print("mode:", d.get("metric_mode"), "| t_err%:", round(d["t_err_pct"], 3),
      "| r_err deg/m:", d.get("r_err_deg_m"), "| ATE m:", round(d["ate_rmse"], 3))
print("per-seq t_err%:", {k: round(v["t_err_pct"], 2) for k, v in d["per_seq"].items()})
print("anchors: reference basic VO 2.04% | M1 bar 1.8% | ORB-SLAM2 1.15%")
