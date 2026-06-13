"""Phase-0 validation of the C++ VIO module: stand up vo-cpp-ceres:1 + prove the image, the toolchain
(Ceres/Eigen/OpenCV link), the I/O contract, and the held-out grader end-to-end with an identity binary.
The identity poses are a DEGENERATE baseline the grader must REJECT (high t_err) — that's the pass
condition for Phase 0 (pipeline works; the real estimator lands in Phases 1-2)."""
import sys, os, json, tempfile, subprocess
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401  bootstrap lab
from lab.models import DatasetRef, FrameworkSpec
from lab.image_registry import ImageRegistry
from vo_lab.plugins.vo_synth import SyntheticVIOProvider, DEFAULT_VIO_HELDOUT

REPO = Path("/home/ws/devel/whasuk/LenaLab")
EVAL = REPO / "vo_lab/plugins/vo_ref/eval_kitti.py"
CPP = REPO / "cpp_vio"
IMG = "vo-cpp-ceres:1"


def main():
    # 1. the harness can resolve the new framework -> image
    reg = ImageRegistry(str(REPO / "images/registry.yaml"))
    resolved = reg.resolve(FrameworkSpec(name="cpp-ceres", version="", cuda=""))
    print(f"[1] framework cpp-ceres resolves -> {resolved.image}")

    # 2. held-out VIO seqs (input frames + imu + gt). Reuse the M3 cache if present (the procedural
    #    renderer is slow); else materialize once via the provider.
    cache = REPO / "_vo_vio_run/cache/heldout/vo-vio-heldout-vio1_vio2"
    if all((cache / f"seq_{s[0]}" / "gt.txt").exists() for s in DEFAULT_VIO_HELDOUT):
        heldout = cache
        print(f"[2] reusing cached held-out VIO data -> {heldout}")
    else:
        heldout = Path(tempfile.mkdtemp(prefix="cppvio_ho_"))
        SyntheticVIOProvider().fetch(DatasetRef(name="ho", source="x", held_out=True), heldout)
        print(f"[2] rendered held-out VIO data -> {heldout}")
    seqs = [s[0] for s in DEFAULT_VIO_HELDOUT]
    art = Path(tempfile.mkdtemp(prefix="cppvio_art_"))

    def drun(args, **kw):
        return subprocess.run(["docker", "run", "--rm", "-v", f"{CPP}:/code", "-v", f"{heldout}:/data",
                               "-v", f"{art}:/art", IMG] + args, **kw)

    # 3. build the binary inside the container (CMake; proves Ceres/OpenCV/Eigen link)
    b = drun(["bash", "-c", "cd /code && rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release "
              ">/tmp/cm.log 2>&1 && cmake --build build -j2 >>/tmp/cm.log 2>&1 && echo BUILT || "
              "(echo BUILD_FAIL; tail -20 /tmp/cm.log)"], capture_output=True, text=True)
    print(f"[3] build: {b.stdout.strip()[-400:]}")
    if "BUILT" not in b.stdout:
        print("BUILD FAILED"); return 1

    # 4. run the binary per held-out seq -> poses_<s>.txt + traj_<s>.txt
    for s in seqs:
        r = drun(["/code/build/vio", f"/data/seq_{s}/input", "/art", s], capture_output=True, text=True)
        print(f"[4] {r.stdout.strip()}")
        if r.returncode != 0:
            print("RUN FAIL:", r.stderr[-400:]); return 1

    # 5. grade with the SAME held-out grader the harness uses
    ev = art / "eval"
    g = subprocess.run([sys.executable, str(EVAL)],
                       env=dict(os.environ, LAB_DATA=str(heldout), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)),
                       capture_output=True, text=True)
    if g.returncode != 0:
        print("GRADER FAIL:", g.stderr[-500:]); return 1
    d = json.load(open(ev / "heldout.json"))
    overall = d.get("t_err_pct", -1)
    per = {k: round(v.get("t_err_pct", 0), 1) for k, v in d.get("per_seq", {}).items()}
    print("=" * 70)
    print(f"[5] PHASE-0 GRADE (identity/degenerate baseline): t_err = {overall:.2f}%  per-seq {per}")
    rejected = overall > 10.0   # identity (no motion) must score badly
    print(f"PHASE 0 {'PASS' if rejected else 'SUSPECT'}: image+toolchain+IO+grader work end-to-end; "
          f"degenerate baseline {'REJECTED (high t_err) as expected' if rejected else 'NOT rejected — investigate'}.")
    print("Next: Phase 1 — stereo VO front-end + Ceres windowed reprojection BA.")
    return 0 if rejected else 1


if __name__ == "__main__":
    sys.exit(main())
