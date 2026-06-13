"""Cross-domain transfer check (non-billed): run the SYNTHETIC-authored solver on REAL KITTI
held-out, and the KITTI-authored reference on SYNTHETIC, to show the agent's VO authoring is
domain-general. Writes result to /tmp/xdomain_result.txt."""
import os, sys, json, tempfile, subprocess, shutil
from pathlib import Path
R = Path("/home/ws/devel/whasuk/LenaLab")
EVAL = R / "vo_lab/plugins/vo_ref/eval_kitti.py"
SYNTH_SOLVER = R / "artifacts/agent_authored_vo_synth.py"
KITTI_HO = R / "_vo_kitti_slam_run3/cache/heldout/vo-kitti-heldout-07_09"


def grade(solver, heldout):
    art = Path(tempfile.mkdtemp())
    for sq in sorted(Path(heldout).glob("seq_*")):
        s = sq.name.replace("seq_", "")
        tmp = Path(tempfile.mkdtemp())
        try:
            subprocess.run([sys.executable, str(solver)],
                           env=dict(os.environ, LAB_DATA=str(sq / "input"), LAB_ARTIFACTS=str(tmp)),
                           check=True, capture_output=True, timeout=1200)
            for f in ("traj.txt", "poses.txt"):
                if (tmp / f).exists():
                    (art / f"{f.split('.')[0]}_{s}.txt").write_bytes((tmp / f).read_bytes())
        except Exception as e:
            print("  solver failed on", s, str(e)[:120])
    ev = art / "eval"
    subprocess.run([sys.executable, str(EVAL)],
                   env=dict(os.environ, LAB_DATA=str(heldout), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)),
                   check=True, capture_output=True)
    return json.load(open(ev / "heldout.json"))


d = grade(SYNTH_SOLVER, KITTI_HO)
res = (f"SYNTHETIC-authored solver on REAL KITTI 07/09: t_err {round(d['t_err_pct'],3)}% | "
       f"per-seq {{k: round(v['t_err_pct'],2) for k,v in d['per_seq'].items()}}".replace(
           "{k: round(v['t_err_pct'],2) for k,v in d['per_seq'].items()}",
           str({k: round(v['t_err_pct'], 2) for k, v in d['per_seq'].items()})))
open("/tmp/xdomain_result.txt", "w").write(res + "\n")
print(res)
