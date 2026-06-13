"""Phase-1 validation: the C++ stereo VO + Ceres windowed BA on CLEAN synthetic stereo (synth1/synth2,
no blackouts). Gate: decisively beat the degenerate baseline (99% t_err) and approach the Python
reference VO (~1.7% t_err). Reuses cached held-out data; builds the binary in the vo-cpp-ceres image."""
import sys, os, json, tempfile, subprocess
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401

REPO = Path("/home/ws/devel/whasuk/LenaLab")
EVAL = REPO / "vo_lab/plugins/vo_ref/eval_kitti.py"
CPP = REPO / "cpp_vio"
IMG = "vo-cpp-ceres:1"
HELDOUT = REPO / "_vo_vio_run/cache/heldout/vo-vio-heldout-vio1_vio2"
SEQS = ["vio1", "vio2"]
REF = 3.83  # Python M3 agent VIO on these blackout seqs


def main():
    if not all((HELDOUT / f"seq_{s}" / "gt.txt").exists() for s in SEQS):
        print("missing cached synth stereo held-out:", HELDOUT); return 1
    art = Path(tempfile.mkdtemp(prefix="cppvio2_art_"))

    def drun(args, **kw):
        return subprocess.run(["docker", "run", "--rm", "-v", f"{CPP}:/code", "-v", f"{HELDOUT}:/data",
                               "-v", f"{art}:/art", IMG] + args, **kw)

    b = drun(["bash", "-c", "cd /code && rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
              ">/tmp/cm.log 2>&1 && cmake --build build -j2 >/tmp/bd.log 2>&1 && echo BUILT || "
              "(echo BUILD_FAIL; tail -25 /tmp/bd.log)"], capture_output=True, text=True)
    print("[build]", b.stdout.strip()[-600:])
    if "BUILT" not in b.stdout:
        return 1
    for s in SEQS:
        r = drun(["/code/build/vio", f"/data/seq_{s}/input", "/art", s], capture_output=True, text=True)
        print("[run]", r.stdout.strip() or r.stderr[-300:])
        if r.returncode != 0:
            print("RUN FAIL:", r.stderr[-400:]); return 1

    ev = art / "eval"
    g = subprocess.run([sys.executable, str(EVAL)],
                       env=dict(os.environ, LAB_DATA=str(HELDOUT), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)),
                       capture_output=True, text=True)
    if g.returncode != 0:
        print("GRADER FAIL:", g.stderr[-600:]); return 1
    d = json.load(open(ev / "heldout.json"))
    overall = d.get("t_err_pct", -1)
    per = {k: round(v.get("t_err_pct", 0), 2) for k, v in d.get("per_seq", {}).items()}
    print("=" * 70)
    print(f"PHASE-2 C++ VIO (IMU-fused): t_err = {overall:.2f}%  per-seq {per}  (ref ~{REF}%)")
    ok = overall <= 5.0   # a real VO must be far below the 99% degenerate; near ref is the goal
    print(f"PHASE 2 {'PASS' if ok else 'FAIL'}: "
          f"{'working VIO, bridges blackouts (beats degenerate; '+('~matches' if overall<=3.83 else 'above')+' ref)' if ok else 'not a working VO yet — iterate'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
