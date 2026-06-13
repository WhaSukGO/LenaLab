"""Option A (perturbed-KITTI) CONTAMINATION PROBE — local, non-billed.

Goal: produce a PERTURBED copy of real held-out KITTI stereo (seq_07, seq_09) where the
*specific sequence* is novel enough to defeat an LLM's memorized recall of KITTI, while the
geometry stays a VALID right-handed stereo-VO problem with EXACT transformed ground truth.
Then PROVE validity by running the proven classical reference stereo VO on it (positive
control): it must still reproduce the un-perturbed ~2.81% mean t_err (07=2.41, 09=3.22).

Perturbations, composed (each individually GT-exact):
  (P) PHOTOMETRIC : gamma + brightness/contrast + mild gaussian noise. Geometry untouched.
                    GT: identity. Defeats pixel-hash recall, weak on structure.
  (R) TEMPORAL REVERSE : play frames backwards. Reverse the frame order of left/right images
                    AND reverse the GT pose/centre sequence, then re-reference world to the
                    new first frame (T0^-1 @ Ti). A reversed drive is still a valid VO problem.
  (M) HORIZONTAL MIRROR : the strong recall-defeat — flip the whole path left/right.
                    KITTI cam frame is x-right, y-down, z-forward; world = frame-0 cam frame.
                    Mirror = reflection across the x=0 plane, S = diag(-1,1,1).
                    Images : flip each image about the vertical axis. A horizontal flip reverses
                    the optical x-axis, which reverses the sign of the stereo baseline (right cam
                    is no longer +x of left). To restore a proper left-then-right rig we SWAP
                    left<->right in addition to flipping each. cx -> (W-1)-cx.
                    GT     : reflect each cam->world pose by conjugation T' = S @ T @ S
                    (R' = S R S keeps det=+1, a valid rotation; t' = S t flips path x).

The positive control is the correctness oracle: if a transform has a geometry bug the reference
VO t_err blows up well above ~2.8%.

SETTLED OUTCOME (positive control = 2.81%, band 2.81 +/- 0.5):
  I  identity .................. 2.812%  PASS  (harness reproduces baseline exactly: 07=2.41,09=3.22)
  P  photometric only ......... 2.976%  PASS
  M  MIRROR only .............. 3.231%  PASS  <-- SETTLED strongest VALID perturbation (07=3.53,09=2.93)
  R  temporal reverse only .... 5.243%  FAIL  (07=6.75: the reference VO's forward-motion stereo
                                              gating degrades on a reversed drive -> dropped)
  PM photometric + MIRROR ..... 3.5-4.3% FAIL  (composition pushes seq_09 over the band; the
                                              mirror already supplies the strong recall-defeat,
                                              so photometric is dropped as a fragile add-on)
=> SETTLED = MIRROR alone. Run `python3 scripts/contamination_perturb_kitti.py SETTLED`.

Writes everything under _contamination_A_run/ (gitignored by the `_*_run/` pattern).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/ws/devel/whasuk/LenaLab")
SRC = ROOT / "_vo_kitti_slam_run3/cache/heldout/vo-kitti-heldout-07_09"
OUT = ROOT / "_contamination_A_run"
REF = ROOT / "vo_lab/plugins/vo_ref/run_kitti_stereo.py"
EVAL = ROOT / "vo_lab/plugins/vo_ref/eval_kitti.py"
SEQS = ("07", "09")

# reflection across the camera/world x=0 plane (mirror handedness fix lives in the swap+flip).
# 4x4 homogeneous so it conjugates cam->world 4x4 poses directly.
S = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float64)

RNG = np.random.default_rng(20260607)


# ----------------------------- photometric -----------------------------------

def photometric(img: np.ndarray) -> np.ndarray:
    """gamma + brightness/contrast + mild gaussian noise. Geometry untouched (per-pixel)."""
    # DETERMINISTic only (no random noise): gamma + brightness/contrast. Random noise was found
    # to add ORB-match variance that pushed the composed photometric+mirror just out of the +/-0.5
    # positive-control band on seq_09, so it is dropped — recall-defeat here comes from the mirror.
    x = img.astype(np.float32) / 255.0
    x = np.power(x, 1.07)              # gamma (mild)
    x = 1.05 * x + 0.02               # contrast/brightness (mild)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


# ----------------------------- GT helpers -------------------------------------

def to44(p12: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :] = p12.reshape(3, 4)
    return T


def from44(T: np.ndarray) -> np.ndarray:
    return T[:3, :].reshape(-1)


def reref(poses44: list[np.ndarray]) -> list[np.ndarray]:
    """re-reference the world frame to the new first pose (so frame 0 == identity, KITTI style)."""
    T0inv = np.linalg.inv(poses44[0])
    return [T0inv @ T for T in poses44]


# ----------------------------- build one seq ----------------------------------

def build_seq(s: str, mode: str, dst: Path) -> int:
    src = SRC / f"seq_{s}"
    sin = src / "input"
    din = dst / "input"
    din.mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy, baseline = np.loadtxt(sin / "intrinsics.txt")
    n = len(sorted(sin.glob("left_*.png")))

    do_reverse = "R" in mode
    do_mirror = "M" in mode
    do_photo = "P" in mode

    # frame index mapping (temporal reverse)
    order = list(range(n - 1, -1, -1)) if do_reverse else list(range(n))

    # one image to read width
    probe = cv2.imread(str(sin / "left_000000.png"))
    W = probe.shape[1]
    cx_new = (W - 1) - cx if do_mirror else cx

    for out_i, in_i in enumerate(order):
        l = cv2.imread(str(sin / f"left_{in_i:06d}.png"))
        r = cv2.imread(str(sin / f"right_{in_i:06d}.png"))
        if do_mirror:
            # flip each about the vertical axis, then SWAP so new-left/new-right keep right=+x
            lf = cv2.flip(l, 1)
            rf = cv2.flip(r, 1)
            new_left, new_right = rf, lf
        else:
            new_left, new_right = l, r
        if do_photo:
            new_left = photometric(new_left)
            new_right = photometric(new_right)
        cv2.imwrite(str(din / f"left_{out_i:06d}.png"), new_left)
        cv2.imwrite(str(din / f"right_{out_i:06d}.png"), new_right)

    # intrinsics: baseline & focal unchanged; only cx reflects under mirror
    np.savetxt(din / "intrinsics.txt",
               np.array([fx, fy, cx_new, cy, baseline]).reshape(-1, 1), fmt="%.6f")

    # ---- transform GT correspondingly ----
    gt_poses = np.loadtxt(src / "gt_poses.txt").reshape(-1, 12)
    T = [to44(p) for p in gt_poses]
    T = [T[i] for i in order]                       # temporal reverse
    if do_mirror:
        T = [S @ Ti @ S for Ti in T]                # reflection by conjugation
    T = reref(T)                                    # frame 0 -> identity

    poses_out = np.array([from44(Ti) for Ti in T])
    centres_out = np.array([Ti[:3, 3] for Ti in T])
    np.savetxt(dst / "gt_poses.txt", poses_out, fmt="%.8e")
    np.savetxt(dst / "gt.txt", centres_out, fmt="%.6f")
    return n


# ----------------------------- run + grade ------------------------------------

def run_and_grade(mode: str, label: str) -> dict:
    base = OUT / label
    if base.exists():
        shutil.rmtree(base)
    hd = base / "heldout"
    art = base / "art"
    art.mkdir(parents=True, exist_ok=True)
    ev = base / "eval"

    for s in SEQS:
        n = build_seq(s, mode, hd / f"seq_{s}")
        print(f"  [{label}] built seq_{s}: {n} frames", flush=True)
        tmp = Path(tempfile.mkdtemp(prefix=f"contam_{s}_"))
        env = dict(os.environ, LAB_DATA=str(hd / f"seq_{s}" / "input"), LAB_ARTIFACTS=str(tmp))
        subprocess.run([sys.executable, str(REF)], env=env, check=True,
                       capture_output=True, text=True)
        shutil.copy(tmp / "traj.txt", art / f"traj_{s}.txt")
        shutil.copy(tmp / "poses.txt", art / f"poses_{s}.txt")
        shutil.rmtree(tmp, ignore_errors=True)

    env = dict(os.environ, LAB_DATA=str(hd), LAB_ARTIFACTS=str(art), LAB_EVAL_OUT=str(ev))
    subprocess.run([sys.executable, str(EVAL)], env=env, check=True)
    d = json.load(open(ev / "heldout.json"))
    print(f"  [{label}] mode={d.get('metric_mode')}  t_err={d['t_err_pct']:.3f}%  "
          f"per-seq={ {k: round(v['t_err_pct'],2) for k,v in d['per_seq'].items()} }", flush=True)
    return d


def verdict(d: dict, label: str) -> bool:
    t = d["t_err_pct"]
    ok = 2.31 <= t <= 3.31           # 2.81 +/- 0.5
    print(f"  => {label}: t_err {t:.3f}%  vs target 2.81% (+/-0.5)  "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    print(f"source: {SRC}")
    print(f"output: {OUT}\n")

    all_stages = {
        "I":   ("",    "IDENTITY (un-perturbed control through this harness)"),
        "P":   ("P",   "photometric only"),
        "R":   ("R",   "temporal reverse only"),
        "M":   ("M",   "MIRROR only"),
        "SETTLED": ("M", "MIRROR (SETTLED: strongest perturbation that PASSES the control)"),
        "PM":  ("PM",  "photometric + MIRROR"),
        "PR":  ("PR",  "photometric + temporal reverse"),
        "PRM": ("PRM", "photometric + reverse + MIRROR"),
    }
    sel = sys.argv[1:] if len(sys.argv) > 1 else list(all_stages.keys())
    stages = [(all_stages[k][0], all_stages[k][1], k) for k in sel]
    results = {}
    for mode, desc, key in stages:
        print(f"[stage {key}] {desc}", flush=True)
        d = run_and_grade(mode, f"stage_{key}")
        results[key] = (d, verdict(d, key))
        print(flush=True)

    print("=== POSITIVE-CONTROL SUMMARY (un-perturbed baseline 2.81%: 07=2.41, 09=3.22) ===")
    for mode, desc, key in stages:
        d, ok = results[key]
        ps = {k: round(v["t_err_pct"], 2) for k, v in d["per_seq"].items()}
        print(f"  {key:4s} {desc:50s} t_err={d['t_err_pct']:.3f}%  {ps}  "
              f"{'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
