"""Salvage-grade the M2 GPU attempt's main.py (a torch.cuda SLAM, premature-killed at 2h33m) on
the held-out 07/09, in the vo-gpu-torch:1 container (so torch.cuda is available), GT-isolated, on
the OFFICIAL KITTI metric. Non-billed. Each seq's gt is a SIBLING of input/ -> running with
LAB_DATA=seq/input is GT-free."""
import os, shutil, subprocess, sys, tempfile, json, time
from pathlib import Path

ROOT = Path("/home/ws/devel/whasuk/LenaLab")
HO = ROOT / "_vo_kitti_slam_gpu_run/cache/heldout/vo-kitti-heldout-07_09"
MAIN = ROOT / "artifacts/agent_authored_vo_kitti_slam_gpu_v1.py"
codedir = Path(tempfile.mkdtemp(prefix="gpucode_")); shutil.copy(MAIN, codedir / "main.py")
art = Path(tempfile.mkdtemp(prefix="salvage_m2gpu_"))

for sq in sorted(HO.glob("seq_*")):
    s = sq.name.replace("seq_", "")
    out = Path(tempfile.mkdtemp()); inp = sq / "input"
    nf = len(list(inp.glob("left_*.png")))
    print(f"running GPU SLAM main.py in container on held-out seq_{s} ({nf} frames) ...", flush=True)
    t0 = time.time()
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--network", "none",
           "-v", f"{inp}:/data:ro", "-v", f"{out}:/artifacts", "-v", f"{codedir}:/code:ro",
           "-e", "LAB_DATA=/data", "-e", "LAB_ARTIFACTS=/artifacts", "-e", "LAB_CODE=/code",
           "-e", "HOME=/tmp", "vo-gpu-torch:1", "bash", "-c", "cd /code && python3 main.py"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        print(f"  seq_{s} exit={r.returncode} in {time.time()-t0:.0f}s", flush=True)
        if r.returncode != 0:
            print("  stderr tail:", r.stderr[-400:], flush=True)
    except Exception as e:
        print(f"  seq_{s} FAILED: {str(e)[:200]}", flush=True)
    for f in ("traj.txt", "poses.txt"):
        if (out / f).exists():
            shutil.copy(out / f, art / f"{f.split('.')[0]}_{s}.txt")

evout = art / "eval"
env = dict(os.environ, LAB_DATA=str(HO), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(evout))
subprocess.run([sys.executable, str(ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py")], env=env, check=True)
d = json.load(open(evout / "heldout.json"))
print("\n=== M2 GPU-ATTEMPT SALVAGE RESULT (official KITTI metric) ===")
print("mode:", d.get("metric_mode"), "| t_err%:", round(d["t_err_pct"], 3),
      "| r_err deg/m:", d.get("r_err_deg_m"), "| ATE m:", round(d["ate_rmse"], 3))
print("per-seq t_err%:", {k: round(v["t_err_pct"], 2) for k, v in d["per_seq"].items()})
print("anchors: basic VO 2.81% | bar 1.8% | ideal closure 1.32% | ORB-SLAM2 1.15%")
print("prior M2: attempt1 6.53% | attempt2(interrupted) 7.40%")
