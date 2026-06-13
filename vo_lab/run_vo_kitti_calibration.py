"""KITTI stereo calibration (LOCAL, non-billed): measures the reference STEREO VO on the
LEADERBOARD-FORM metric (KITTI t_err %, length-normalized translational drift), checks the gate
discriminates (reference passes, degenerate fails), and reports where we stand vs PUBLISHED SOTA.

  bash scripts/fetch_kitti_odometry.sh        # once, ~21.6 GB
  python -m vo_lab.run_vo_kitti_calibration

The GATE bar is set just above the reference (a discrimination check — basic stereo VO ~2%).
The PUBLISHED anchors are the aspiration the agent climbs toward (the 'progressive SOTA' ladder):
basic frame-to-frame stereo VO ~2-3%, ORB-SLAM2 stereo (BA+loop closure) ~1.15%, learned DROID
~0.4%."""
from __future__ import annotations

import sys

from lab.models import ExperimentRecord

from .agents.vo_implementer import (KITTI_SOTA_TERR, kitti_degenerate_author,
                                    kitti_stereo_reference_author, vo_impl_task_kitti)
from .factory import build_vo_implementer_harness
from .plugins.vo_kitti import KITTIOdomProvider


def main(root: str = "./_vo_kitti_run", dev: str = "00",
         heldout: tuple[str, ...] = ("05", "07"), stride: int = 3, max_frames: int = 300,
         margin: float = 1.25) -> int:
    prov = KITTIOdomProvider(dev=dev, heldout=heldout, stride=stride, max_frames=max_frames)

    print(f"Measuring reference STEREO VO (KITTI t_err %) on held-out {heldout} "
          f"(stride={stride}, max_frames={max_frames})...")
    hp = build_vo_implementer_harness(root, task=vo_impl_task_kitti(1e9, dev=dev, heldout=heldout),
                                      provider=prov, author_fn=kitti_stereo_reference_author(),
                                      job_mode="local")
    pos = hp.run_experiment(ExperimentRecord(id="kitti-cal-pos", hypothesis="reference stereo VO"))
    if not (pos.verdict and "t_err_pct" in pos.verdict.measured_metrics):
        print("measurement failed; see log:", pos.log_path); return 1
    mm = pos.verdict.measured_metrics
    terr_ref = float(mm["t_err_pct"]); bar = round(terr_ref * margin, 3)

    hn = build_vo_implementer_harness(root, task=vo_impl_task_kitti(bar, dev=dev, heldout=heldout),
                                      provider=prov, author_fn=kitti_degenerate_author(),
                                      job_mode="local")
    neg = hn.run_experiment(ExperimentRecord(id="kitti-cal-neg", hypothesis="degenerate"))
    terr_neg = float(neg.verdict.measured_metrics.get("t_err_pct", 1e9))
    opened = (terr_ref <= bar) and (terr_neg > bar)

    r_err = mm.get("r_err_deg_m")
    mode = mm.get("metric_mode", "?")
    print("=" * 70)
    rtxt = f", r_err {r_err:.4f} deg/m" if r_err is not None else ""
    print(f"reference stereo VO — held-out KITTI [{mode}] t_err = {terr_ref:.3f}%{rtxt}  "
          f"(ATE {mm.get('ate_rmse'):.3f} m)")
    print(f"  per-seq t_err: { {k: round(v['t_err_pct'],2) for k,v in mm.get('per_seq',{}).items()} }")
    print(f"gate bar (reference x{margin}) = {bar}%   degenerate = {terr_neg:.1f}% -> "
          f"{'REJECTED' if terr_neg>bar else 'PASS?!'}")
    print(f"CALIBRATION GATE: {'OPEN' if opened else 'LOCKED'}")
    print("-" * 70)
    # LITERATURE-REPRODUCTION sanity check: our basic stereo VO must land in the PUBLISHED band
    # for basic frame-to-frame stereo VO on KITTI (t_err ~2-3%, r_err ~0.005-0.015 deg/m). If it
    # didn't, the metric implementation or setup would be suspect.
    LIT_T = (1.0, 4.0); LIT_R = (0.002, 0.030)
    t_ok = LIT_T[0] <= terr_ref <= LIT_T[1]
    r_ok = (r_err is None) or (LIT_R[0] <= r_err <= LIT_R[1])
    reproduced = (mode == "official") and t_ok and r_ok
    print("LITERATURE REPRODUCTION (basic stereo VO published band: "
          f"t_err {LIT_T[0]}-{LIT_T[1]}%, r_err {LIT_R[0]}-{LIT_R[1]} deg/m):")
    print(f"  reference t_err={terr_ref:.2f}% {'IN' if t_ok else 'OUT-OF'} band; "
          f"r_err={r_err if r_err is None else round(r_err,4)} {'IN' if r_ok else 'OUT-OF'} band "
          f"-> {'REPRODUCED (metric validated)' if reproduced else 'NOT reproduced / centre-approx'}")
    print("-" * 70)
    print("WHERE WE STAND vs PUBLISHED (the SOTA ladder to climb):")
    print(f"  reference (this lab, basic stereo VO) : {terr_ref:.2f}% t_err")
    for name, val in KITTI_SOTA_TERR.items():
        gap = terr_ref / val if val else float('inf')
        print(f"  {name:22s}: {val:.2f}%   (we are {gap:.1f}x its t_err)")
    print("  -> rung 2 = add bundle adjustment + loop closure (target ORB-SLAM2 ~1.15%);")
    print("     rung 3 = learned (target DROID ~0.4%, needs more compute).")
    print("=" * 70)
    if opened:
        print(f">>> live KITTI Track B (gate bar):  python -m vo_lab.run_vo_kitti_implement {bar}")
    return 0 if opened else 1


if __name__ == "__main__":
    sys.exit(main())
