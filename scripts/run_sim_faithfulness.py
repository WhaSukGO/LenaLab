"""Sim-faithfulness experiment (flagship): is a 3DGS-rendered scene faithful enough to VALIDATE a
learned SLAM? Run DROID-SLAM on (a) the REAL city scene and (b) the SAME scene re-rendered via
GSplatModule (stereo-depth reprojection at the GT trajectory), and compare held-out Sim3 ATE. A small
delta => the rendered sim preserves what the SLAM needs (sim is faithful); a large delta => it doesn't.
City-scale (fits the 15GB-RAM DROID budget). Writes artifacts/slam_benchmark/sim_faithfulness.{json,png}.
"""
import sys, os, json, tempfile, subprocess, shutil, time
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
import vo_lab  # noqa: F401
import numpy as np
import cv2
from vo_lab.plugins.vo_ref.gaussian_kitti import depth_from_stereo, cloud_from_frame, render_view

REPO = Path("/home/ws/devel/whasuk/LenaLab")
# .resolve() -> ABSOLUTE path. docker -v requires absolute paths; a relative scene arg made the 'real' mount
# fail silently (THE real bug behind every 'real run failed', long misattributed to RAM/OOM).
CITY = (Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data/kitti_raw/seq_city_20110926_0001").resolve()
TAG = CITY.name.replace("seq_", "")
EVAL = REPO / "vo_lab/plugins/vo_ref/eval_learned.py"
STRIDE = int(os.environ.get("SF_STRIDE", "2"))   # 47GB RAM now -> stride 2 (denser, full quality)


def render_city(out: Path):
    """Re-render the city scene at its GT trajectory (real-appearance reprojection) -> our seq format."""
    fx, fy, cx, cy, base = np.loadtxt(CITY / "input/intrinsics.txt")
    K = (fx, fy, cx, cy)
    poses = np.loadtxt(CITY / "gt_poses.txt").reshape(-1, 3, 4)
    lefts = sorted((CITY / "input").glob("left_*.png"))
    rights = sorted((CITY / "input").glob("right_*.png"))
    n = len(lefts); H, W = cv2.imread(str(lefts[0]), cv2.IMREAD_GRAYSCALE).shape
    (out / "input").mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "input/intrinsics.txt", [fx, fy, cx, cy, base], fmt="%.6f")

    def Twc(i):
        T = np.eye(4); T[:3, :4] = poses[i]; return T
    cache = {}

    def cloud(i):
        if i not in cache:
            L = cv2.imread(str(lefts[i]), cv2.IMREAD_GRAYSCALE)
            R = cv2.imread(str(rights[i]), cv2.IMREAD_GRAYSCALE)
            cache[i] = cloud_from_frame(L, depth_from_stereo(L, R, fx, base), K, Twc(i), step=2, zmax=45)
        return cache[i]
    for i in range(n):
        srcs = [s for s in (i - 1, i, i + 1) if 0 <= s < n]
        P = np.concatenate([cloud(s)[0] for s in srcs]); C = np.concatenate([cloud(s)[1] for s in srcs])
        cv2.imwrite(str(out / "input" / f"left_{i:06d}.png"), render_view(P, C, K, Twc(i), H, W, splat=2))
        # DROID is monocular (left only); also drop a dummy right so intrinsics loader is happy if needed
        cache.pop(i - 1, None)
    shutil.copy(CITY / "gt.txt", out / "gt.txt"); shutil.copy(CITY / "gt_poses.txt", out / "gt_poses.txt")
    return n


def _wait_gpu_free():
    """Kill any leaked vo-droid containers + wait for the GPU to actually release (spawn containers
    survive docker-run completion and poison the next run — the real bug behind 'real fails')."""
    subprocess.run("docker ps -q --filter ancestor=vo-droid:1 | xargs -r docker kill",
                   shell=True, capture_output=True)
    subprocess.run(["docker", "rm", "-f", "droidsf"], capture_output=True)
    for _ in range(40):
        u = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                           shell=True, capture_output=True, text=True).stdout.strip()
        if u and int(u) < 1500:
            return
        time.sleep(2)


def droid(seq_input: Path, tag: str) -> Path:
    _wait_gpu_free()                                          # clean GPU BEFORE each run
    out = Path(tempfile.mkdtemp())
    # CRITICAL: redirect DROID's output to a FILE, not capture_output=True. DROID uses multiprocessing.spawn
    # and floods stdout; a pipe (capture_output) fills its 64KB buffer and DEADLOCKS the children (this — not
    # RAM — is why 'real' runs silently failed: real frames produce more output than degraded rendered ones).
    logf = open(out / f"droid_{tag}.log", "w")
    subprocess.run(
        ["docker", "run", "--rm", "--name", "droidsf", "--gpus", "all",
         "-e", f"DROID_AREA={os.environ.get('DROID_AREA', 196608)}",   # full res (384*512); 47GB RAM fits it
         "-v", f"{REPO}/scripts:/scripts", "-v", f"{seq_input.parent}:/data",
         "-v", f"{out}:/out", "-v", f"{Path.home()}/.cache/vo_lab/droid:/w", "vo-droid:1",
         "python", "/scripts/run_droid_slam.py", f"/data/{seq_input.name}", "/out", tag, "/w/droid.pth", str(STRIDE)],
        stdout=logf, stderr=subprocess.STDOUT, timeout=1200)
    logf.close()
    _wait_gpu_free()                                          # clean GPU AFTER each run
    return out


def grade(traj_file: Path, tag: str) -> float:
    g = np.loadtxt(CITY / "gt.txt")[::STRIDE]
    n = sum(1 for _ in open(traj_file))
    ho = Path(tempfile.mkdtemp()); (ho / f"seq_{tag}").mkdir()
    np.savetxt(ho / f"seq_{tag}" / "gt.txt", g[:n], fmt="%.6f")
    art = Path(tempfile.mkdtemp()); shutil.copy(traj_file, art / f"traj_{tag}.txt")
    ev = art / "eval"
    subprocess.run([sys.executable, str(EVAL)], env=dict(os.environ, LAB_DATA=str(ho),
                   LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev)), capture_output=True)
    return float(json.load(open(ev / "heldout.json"))["per_seq"][f"seq_{tag}"]["ate_rmse"])


def main():
    if len(sys.argv) > 2 and sys.argv[2] == "--render-only":   # child: render + exit (releases RAM)
        render_city(Path(sys.argv[3])); return 0
    print("[1] rendering in a SEPARATE process (so its RAM is freed before the memory-heavy DROID runs)...", flush=True)
    rdir = Path(tempfile.mkdtemp(prefix="rendered_"))
    subprocess.run([sys.executable, __file__, str(CITY), "--render-only", str(rdir)], check=True)
    print(f"    rendered -> {rdir}")
    print("[2] DROID on REAL city...", flush=True)
    real_out = droid(CITY / "input", "real")
    print("[3] DROID on RENDERED city...", flush=True)
    rend_out = droid(rdir / "input", "rendered")
    ate_real = grade(real_out / "traj_real.txt", "real") if (real_out / "traj_real.txt").exists() else None
    ate_rend = grade(rend_out / "traj_rendered.txt", "rendered") if (rend_out / "traj_rendered.txt").exists() else None
    res = {"scene": TAG, "stride": STRIDE,
           "droid_ate_real_m": ate_real, "droid_ate_rendered_m": ate_rend,
           "delta_m": (abs(ate_real - ate_rend) if ate_real and ate_rend else None)}
    out = REPO / "artifacts/slam_benchmark"; out.mkdir(parents=True, exist_ok=True)
    multi = out / "sim_faithfulness_multi.json"
    allres = json.load(open(multi)) if multi.exists() else []
    allres = [r for r in allres if r.get("scene") != TAG] + [res]   # upsert this scene
    json.dump(allres, open(multi, "w"), indent=2)
    json.dump(res, open(out / "sim_faithfulness.json", "w"), indent=2)
    print("=" * 64)
    print(f"SIM-FAITHFULNESS (DROID-SLAM on city):")
    print(f"  real scene   -> ATE {ate_real} m")
    print(f"  3DGS-rendered-> ATE {ate_rend} m")
    print(f"  delta = {res['delta_m']} m  -> {'sim FAITHFUL (DROID behaves ~same)' if res['delta_m'] is not None and res['delta_m'] < 0.5 else 'sim gap' if res['delta_m'] is not None else 'run failed'}")
    # trajectory overlay figure — Sim(3)-ALIGNED (the retraction's figure plotted RAW monocular trajs,
    # which looked like failure regardless of the number; aligning is what the ATE metric actually scores)
    def sim3_align(src, dst):
        n = min(len(src), len(dst)); src, dst = src[:n], dst[:n]
        mu_s, mu_d = src.mean(0), dst.mean(0); s, d = src - mu_s, dst - mu_d
        U, D, Vt = np.linalg.svd((d.T @ s) / n); R = U @ Vt
        if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
        c = D.sum() / (s ** 2).sum() * n
        return (c * (R @ s.T)).T + mu_d, dst   # FIX: align CENTERED src (was R@src -> constant offset)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        gt = np.loadtxt(CITY / "gt.txt")[::STRIDE]
        tr = np.loadtxt(real_out / "traj_real.txt"); te = np.loadtxt(rend_out / "traj_rendered.txt")
        trA, gtA = sim3_align(tr, gt); teA, _ = sim3_align(te, gt)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(gtA[:, 0], gtA[:, 2], "k-", lw=2, label="GT")
        ax.plot(trA[:, 0], trA[:, 2], "b--", lw=1.5, label=f"DROID on REAL ({ate_real:.2f} m)")
        ax.plot(teA[:, 0], teA[:, 2], "r:", lw=1.8, label=f"DROID on 3DGS-RENDERED ({ate_rend:.2f} m)")
        ax.set_title(f"Sim-faithfulness ({TAG}): DROID on real vs rendered, Sim(3)-aligned (delta {res['delta_m']:.2f} m)")
        ax.axis("equal"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(out / f"sim_faithfulness_{TAG}.png", dpi=120); print("wrote", out / f"sim_faithfulness_{TAG}.png")
    except Exception as e:
        print("figure skipped:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
