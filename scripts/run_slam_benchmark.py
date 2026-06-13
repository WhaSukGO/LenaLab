"""SLAM-verification benchmark — Phase 0: the SystemAdapter pattern + run on REAL environment-labeled
KITTI data (city/road/residential), graded by our held-out Sim3-ATE grader. Phase 0 wraps systems we
already have (classical Python stereo VO; our C++ Ceres VO) to prove the adapter plumbing; DROID-SLAM
slots in later as another adapter behind the same interface. Writes artifacts/slam_benchmark/results.json
+ a per-environment figure.
"""
import sys, os, json, tempfile, subprocess, shutil
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401

REPO = Path("/home/ws/devel/whasuk/LenaLab")
EVAL = REPO / "vo_lab/plugins/vo_ref/eval_learned.py"        # Sim3 ATE (length-flexible)
REF_VO = REPO / "vo_lab/plugins/vo_ref/run_kitti_stereo.py"
CPP = REPO / "cpp_vio"; CPP_IMG = "vo-cpp-ceres:1"
DRIVES = [  # (env, seq_dir)
    ("city", REPO / "data/kitti_raw/seq_city_20110926_0001"),
    ("road", REPO / "data/kitti_raw/seq_road_20110926_0015"),
    ("residential", REPO / "data/kitti_raw/seq_residential_20110926_0019"),
]


class SystemAdapter:
    """Wrap any localization system so the harness can grade it. run() writes art/{traj_<seq>, poses_<seq>}."""
    name = "system"
    def setup(self): pass
    def run(self, seq_input: Path, art: Path, seq: str) -> bool: raise NotImplementedError


class ReferenceStereoVO(SystemAdapter):
    name = "classical-stereo-vo (python)"
    def run(self, seq_input, art, seq):
        tmp = Path(tempfile.mkdtemp())
        r = subprocess.run([sys.executable, str(REF_VO)],
                           env=dict(os.environ, LAB_DATA=str(seq_input), LAB_ARTIFACTS=str(tmp)),
                           capture_output=True, text=True)
        if r.returncode != 0 or not (tmp / "traj.txt").exists(): return False
        shutil.copy(tmp / "traj.txt", art / f"traj_{seq}.txt")
        if (tmp / "poses.txt").exists(): shutil.copy(tmp / "poses.txt", art / f"poses_{seq}.txt")
        return True


class CppCeresVO(SystemAdapter):
    name = "cpp-ceres-vo (our C++)"
    def setup(self):
        subprocess.run(["docker", "run", "--rm", "-v", f"{CPP}:/code", CPP_IMG, "bash", "-c",
                        "cd /code && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 && "
                        "cmake --build build -j2 >/dev/null 2>&1"], check=True)
    def run(self, seq_input, art, seq):
        r = subprocess.run(["docker", "run", "--rm", "-v", f"{CPP}:/code",
                            "-v", f"{seq_input.parent}:/data", "-v", f"{art}:/art", CPP_IMG,
                            "/code/build/vio", f"/data/{seq_input.name}", "/art", seq],
                           capture_output=True, text=True)
        return r.returncode == 0 and (art / f"traj_{seq}.txt").exists()


def grade(art: Path) -> dict:
    """Sim3-ATE per seq via eval_learned: heldout dir = seq_<env>/gt.txt; artifacts = traj_<env>.txt."""
    ho = Path(tempfile.mkdtemp())
    for env, d in DRIVES:
        (ho / f"seq_{env}").mkdir(parents=True)
        shutil.copy(d / "gt.txt", ho / f"seq_{env}" / "gt.txt")
    ev = art / "eval"
    g = subprocess.run([sys.executable, str(EVAL)],
                       env=dict(os.environ, LAB_DATA=str(ho), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)),
                       capture_output=True, text=True)
    if g.returncode != 0: return {}
    d = json.load(open(ev / "heldout.json"))
    return {k.replace("seq_", ""): round(v.get("ate_rmse", -1), 2) for k, v in (d.get("per_seq") or {}).items()}


def main():
    systems = [ReferenceStereoVO(), CppCeresVO()]
    results = {}
    for sysm in systems:
        try: sysm.setup()
        except Exception as e: print(f"[{sysm.name}] setup failed: {e}"); continue
        art = Path(tempfile.mkdtemp())
        for env, d in DRIVES:
            ok = sysm.run(d / "input", art, env)
            print(f"  {sysm.name} on {env}: {'ran' if ok else 'FAILED'}")
        results[sysm.name] = grade(art)
        print(f"[{sysm.name}] ATE by env: {results[sysm.name]}")
    out = REPO / "artifacts/slam_benchmark"; out.mkdir(parents=True, exist_ok=True)
    payload = {"grader": "Sim3 ATE (m)", "drives": {e: str(d.name) for e, d in DRIVES}, "systems": results}
    json.dump(payload, open(out / "results.json", "w"), indent=2)
    print("\n=== SLAM benchmark (real KITTI, Sim3 ATE m) ===")
    envs = [e for e, _ in DRIVES]
    print("system".ljust(28) + "".join(e.ljust(14) for e in envs))
    for name, r in results.items():
        print(name.ljust(28) + "".join(str(r.get(e, "—")).ljust(14) for e in envs))
    print("wrote", out / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
