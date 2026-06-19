# BEV perception — a second problem class for the lab (technical note, 2026-06-14)

**What we tested:** can the lab's research loop — analyze → research → implement + train →
confirm it generalizes — carry beyond ego-motion (VO/SLAM) to a **fundamentally different
perception task**: multi-camera **Bird's-Eye-View** semantic occupancy. A held-out split keeps
the scoreboard honest underneath.

## The task
Given a nuScenes sample's **6 surround cameras** + their intrinsics + camera→ego extrinsics,
predict a top-down **vehicle-occupancy grid** in the ego frame (100 m × 100 m @ 0.5 m → 200×200).
This is the canonical Lift-Splat vehicle-segmentation task. It exercises geometry the VO domain
never touched: cross-view fusion, per-pixel depth lifting, and a metric ego-frame raster.

## The measurement layer (kept out of the solver's reach, so the score stays honest)
- **Held-out split** — official nuScenes `mini_val` scenes (`scene-0103`, `scene-0916`),
  disjoint from `mini_train`. 323 train / 81 val samples.
- **Ground truth** — `scripts/prep_nuscenes_bev.py` rasterizes vehicle 3D-box footprints into the
  ego BEV grid (global→ego via the sample ego-pose; `cv2.fillConvexPoly` on bottom corners).
  Confirmed two ways: geometrically by eye (`artifacts/bev_gt_check.png`: footprints cluster in
  parking-row structure around ego, not noise) **and** by learnability (a model fits it — a broken
  GT can't be).
- **Metric + scoring** — `scripts/grade_bev.py` owns the val GT, the IoU metric, and the 0.0-logit
  threshold. It imports only the solver's `build_model()` + weights and computes IoU itself, so the
  number on the board is measured independently of the code that produced it.

## Reference baseline (Lift-Splat-Shoot, `scripts/bev_lss.py`)
ResNet-18 → per-pixel (softmax depth distribution × context) → lift to a camera frustum of 3D
points using the **real** scaled intrinsics + cam→ego extrinsics → voxel-pool (scatter-add) into
the ego BEV grid → conv head → occupancy logits. Trained on `mini_train`, BCE (pos_weight 8),
OneCycle, 24 epochs.

**Held-out vehicle-IoU = 0.169 ± 0.002** (n=3 seeds: 0.1709 / 0.1690 / 0.1657). Tight variance →
a stable bar, not a lucky run. On busy held-out samples IoU reaches ~0.22
(`artifacts/bev/bev_pred_heldout.png`, green=TP / red=FN / blue=FP on unseen scenes).

## Honest scope (stated, not hidden)
- **Small-data regime.** nuScenes **mini** = 10 scenes total. 0.17 IoU is well below full-nuScenes
  LSS (~0.32 over 28k samples); the gap is data quantity, not a method flaw. The point here is
  *the loop generalizes*, demonstrated end-to-end on a held-out split — not a SOTA BEV number.
- **Agent-authored result: capable but NOT robust** (n=3, 2026-06-15). A sandboxed Claude agent
  researched and authored a Lift-Splat network from scratch three times → held-out IoU
  **0.1075 / 0.0376 / 0.1107** (**2/3 ≥ bar 0.08; mean 0.085 ± 0.034**). Diagnostic: a
  fixed-architecture reference at 3 seeds is stable (**0.141 ± 0.002**), so the variance is the
  agent's redesign latitude, **not** the task; the failing run self-sabotaged (15 %-holdout on tiny
  data + simpler flip aug). A single run would have over-claimed "VERIFIED 0.1075" — measuring at
  n=3 surfaced the non-robustness. Robust-result paths (more data / fixed-architecture scaffold) are
  future work, not re-rolls. Full build + diagnosis:
  [`bev_track_b_report_2026-06-15.md`](bev_track_b_report_2026-06-15.md); figure
  `artifacts/bev/bev_variance_n3.png`; algorithm `artifacts/agent_authored_bev_v1.py` (run 1).
- Vehicle class only (the standard LSS sub-task); other classes are future work.

## Artifacts
- `scripts/prep_nuscenes_bev.py` — nuScenes→BEV adapter (held-out split + GT)
- `scripts/bev_lss.py` — reference Lift-Splat model + trainer (`build_model()` entry point)
- `scripts/grade_bev.py` — independent IoU scorer (held-out split)
- `scripts/viz_bev_pred.py` — surround→BEV prediction-vs-GT visualizer
- `artifacts/bev_gt_check.png` — GT geometry sanity check
- `artifacts/bev/bev_pred_heldout.png` — predictions on held-out scenes
- `artifacts/bev/bev_ref_seed{0,1,2}.pt` — the three reference checkpoints

**Takeaway:** the analyze → research → implement + train → confirm-it-generalizes loop transfers
cleanly to multi-view BEV perception. A new domain = `{dataset adapter, GT + metric, held-out
split, reference bar}`, all four built and variance-checked here. Sharpening the agent's authoring
to clear the bar robustly is the open follow-up.
