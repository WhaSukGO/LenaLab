# Design: Smart-Space Occupancy — a self-verifying top-down map of a space

*Design spec · 2026-06-20 · LenaLab 7th domain · `/sc:design` (spec only — no implementation yet).*
*Grounds: `research_vertical_panel_2026-06-20.md` (the "map not a camera" + "trust on day one" thesis),
the proven add-a-domain playbook, and the existing `occ` domain files this mirrors.*

## 1. What this domain proves
The portfolio thesis the panel earned: **"a self-installing, self-verifying top-down map of any space,
from cameras you already have."** This domain demonstrates it on real, public, *non-driving* data —
**static multi-camera warehouses** — where an agent authors an occupancy model and the harness verifies it
**on footage the model never saw**. The held-out gate *is* the "trust on day one" claim made literal.

It also lands the edge from the research report: a **fresh, under-modeled domain** (vs the saturated
nuScenes race), reusing our exact lift-splat + GT-generation + verification machinery.

## 2. The target dataset (verified format)
**NVIDIA Physical AI Smart Spaces** (`nvidia/PhysicalAI-SmartSpaces`, Hugging Face) — AI City MTMC editions:

| Edition | Scenes | Cameras | Size (no-depth) |
|---|---|---|---|
| MTMC_Tracking_2024 | 90 | 953 | 217 GB |
| MTMC_Tracking_2025 | 23 | 504 | **70 GB** |
| MTMC_Tracking_2026 | 28 | 353 | **50 GB** |

**Per scene:** `videos/*.mp4` (H.264, 1080p, 30 fps, one per static camera) · `calibration.json` ·
`ground_truth.json` · `map.png` (top-down) · `depth_maps/` (we **skip** depth).
- **`calibration.json`** (per camera): intrinsic K (3×3), extrinsic (3×4), homography (3×3),
  `translationToGlobalCoordinates` → a shared **world frame**. Cameras are **static**.
- **`ground_truth.json`** (per frame, per object): `object_type` ∈ {Person, Forklift, PalletTruck,
  Transporter, NovaCarter, FourierGR1T2, AgilityDigit}, `object_id`, `3d_location [x,y,z]` (world),
  `3d_bounding_box_scale [w,l,h]`, `rotation [pitch,roll,yaw]`.

## 3. Why it's *cleaner* than nuScenes (the geometry win)
| | nuScenes (our occ) | Smart-space (this) |
|---|---|---|
| Cameras | 6, moving with ego, barely overlap | many, **static**, **heavily overlapping** the same volume |
| Pose | per-frame ego pose needed | **one fixed `cam2world` per camera** (precompute once) |
| Grid frame | ego-centric, re-anchored each frame | **world-anchored** over the fixed floor |
| GT boxes | ego-relative, need transform | **already in world coords** → rasterize directly |

Net: our `lift_splat` transfers almost verbatim with `cam2ego → cam2world` (now constant), and GT
voxelization gets *simpler* (no ego transform). Richer overlap should help multi-view fusion.

## 4. Data contract (on-disk, mirrors `occ`)
Per sample = one timestamp in one scene, sampled at low rate (e.g. 1 frame / 1–2 s, not 30 fps):
```
imgs        (N, H, W, 3) uint8     N static-camera views (resized, e.g. 128×352 like occ)
intrins     (N, 3, 3)   float32    pinhole K scaled to resized images
cam2world   (N, 4, 4)   float32    fixed camera→world extrinsic (incl. translationToGlobal)
grid_bounds (4,)        float32    [x0,x1,y0,y1] world extent this grid covers (per-scene)
bev         (XG, YG)    uint8      HARNESS-OWNED GT: agent occupancy on the floor grid
```
*(2D BEV floor occupancy for v1; a 3D `(XG,YG,ZG)` voxel variant is a later toggle — both already exist
in our code.)* Provider mirrors `NuScenesOccProvider`: train npz (full) / `test_input` (GT stripped) /
held-out (`<id>_bev.npy`).

## 5. Architecture & file manifest (new files, each mirroring an `occ` sibling)
```
scripts/prep_smartspace.py                         # adapter: videos+calib+GT → world BEV-occ npz   (≈ prep_nuscenes_occ.py)
vo_lab/plugins/smartspace.py                        # provider: train/test_input/held-out            (≈ occ_nuscenes.py)
scripts/smartspace_lss.py                           # reference fixed-cam lift-splat + variance        (≈ occ_lss.py)
vo_lab/plugins/vo_ref/eval_smartspace.py            # IoU grader (+ caveats in heldout.json)           (≈ eval_occ.py)
vo_lab/plugins/vo_ref/run_smartspace_learned.py     # self-contained from-scratch reference main.py     (≈ run_occ_learned.py)
vo_lab/plugins/vo_ref/smartspace_scaffold.py        # LOCKED scaffold core (geometry+aug+train)        (≈ occ_scaffold.py)
vo_lab/plugins/vo_ref/smartspace_scaffold_model_ref.py  # reference model.py for the scaffold
vo_lab/agents/smartspace_implementer.py             # task specs (free-form + scaffold) + authors      (≈ occ_implementer.py)
vo_lab/run_smartspace_calibration.py                # non-billed gate (reference vs degenerate)
vo_lab/run_smartspace_implement.py  (+ _scaffold)   # live agent run CLIs
tests/test_smartspace_implementer.py                # anti-tamper (degenerate REJECTED; tamper overridden)
```
Reuse unchanged: Track-B `build_vo_implementer_harness`, `LAB_JOB_MODE` local/docker, the cloud pipeline,
the `vo-bev:1` image (cv2 reads MP4 via `VideoCapture`; no new deps — depth skipped).

## 6. The two design decisions that define the experiment

### (a) The held-out split = the whole thesis
- **v1 — per-space self-verification (recommended, achievable, IS the product):** for each scene, train on
  the **first ~70% of its timeline**, hold out the **last ~30%** (unseen time, same space). This is exactly
  "stand up + verify a model *for this space* from a short calibration window" — the deployable claim.
- **v2 — cross-space generalization (stretch headline):** train on N scenes, hold out an **entirely unseen
  scene** (new layout, new cameras). Harder; report honestly (likely lower). This is the bolder "works on a
  brand-new space day one" claim.
- **Plan:** ship v1 as the gated result; attempt v2 as the headline ambition with honest numbers.

### (b) Variable camera count
Scenes have different camera counts/placements. **v1:** select a **fixed N-camera subset** per scene (best
floor coverage) so the model input is fixed (like occ's 6). **Later:** a set/attention model over variable
N (a genuine research extension — and a differentiator).

## 7. Metric & gate
- **Metric:** per-sample mean floor-occupancy **IoU** (≈ `eval_occ`), `caveats` embedded in `heldout.json`
  ("box-derived agent occupancy: people+robots", "per-space self-verified, held-out = unseen time",
  "static-camera world-grid").
- **Calibration gate:** reference fixed-cam lift-splat (from-scratch) sets the bar; a degenerate/trivial
  predictor must be **REJECTED** (bar = ref/1.3, as in occ). Anti-tamper tests carried over.

## 8. Phased plan (add-a-domain playbook, proven 4×)
- **Phase 0 — De-risk (the critical unknown is data access + geometry).** Pull **ONE** scene from HF
  (`huggingface_hub.snapshot_download(..., allow_patterns="MTMC_Tracking_2026/<scene>/*", )`, no-depth
  edition ~ a few GB). Verify: parse `calibration.json` + `ground_truth.json`; extract a few frames
  (`cv2.VideoCapture`); build the world grid from `map.png`/bounds; rasterize GT for one timestamp; project
  **one** camera's frustum into the grid and confirm it lines up with the GT footprint + `map.png`.
  → *Go/no-go before any building.* (~½ day.)
- **Phase 1 — Adapter** (`prep_smartspace.py`): scene(s) → npz (sampled frames, fixed N-cam, world GT).
- **Phase 2 — Reference** (`smartspace_lss.py`): from-scratch fixed-cam lift-splat → reference IoU +
  variance → sets the bar.
- **Phase 3 — Grader + provider + gate** + anti-tamper tests (offline, no API).
- **Phase 4 — Live agent authoring** (free-form + scaffold) on the cloud pipeline (RunPod/local mode).
- **Phase 5 — Docs + viz**: top-down pred-vs-GT figures (the literal "map"), the per-space self-verify
  result, the "what the agent figured out" entry, the cross-space honest attempt.

## 9. Risks & open questions
| Risk | Mitigation |
|---|---|
| **Single-scene download** (full set is 50–217 GB) | `snapshot_download` with `allow_patterns` for one scene's `videos/ calibration.json ground_truth.json`; skip `depth_maps/`. Resolve in Phase 0. |
| Cross-space generalization may be weak | Lead with per-space v1 (achievable); report v2 honestly — negative results kept honest per our norm. |
| Variable N cameras | Fixed N-subset v1; variable-N set model later. |
| 1080p × many cams = VRAM | Aggressive resize (128×352 like occ); fixed N-subset; cloud A100 if needed. |
| World-grid bounds differ per scene | Pass `grid_bounds` per sample; fixed resolution (≈0.25 m), fixed XG×YG by crop/pad. Finalize in Phase 0. |
| Frame/GT time alignment | Sample on GT-annotated frame indices; verify in Phase 0. |

## 10. Decisions — LOCKED (2026-06-20)
1. **Target:** ✅ **2D BEV floor occupancy** (X×Y floor-cell map). 3D voxel deferred.
2. **Split:** ✅ **Per-space self-verification** — train on a scene's first ~70% of timeline, hold out the
   last ~30% (unseen time, same space). This is the deployable claim ("verify a model *for this space*").
   Cross-space generalization deferred to a later stretch.
3. **Edition / scope:** ✅ **MTMC_Tracking_2026** (smallest, ~50 GB no-depth), **one scene end-to-end**
   first (Phase 0→4), then expand.
4. **Cameras:** fixed N-camera subset (best floor coverage) for v1; variable-N deferred.

**Locked v1 = the shortest path to a verified, demo-able "self-verifying top-down map of a real space."**

## 11. Phase 0 result — ✅ GREEN (2026-06-20)
De-risked on `MTMC_Tracking_2026/train/Warehouse_000` (open download, not gated; one scene = 19 cams /
3.25 GB videos, `ground_truth.json` 9000 frames @ 30 fps = 5 min, world-meter 3D boxes). Verified:
1. **Calibration projects GT onto the image** — `cameraMatrix` (world→image) projected 3D-box centers
   **inside the dataset's own 2D boxes for 7/8 (88%)** genuinely-visible objects; visually confirmed
   (`artifacts/smartspace_derisk/phase0_cam0_fullres.png` — person mid-aisle boxed, center dot on them).
2. **BEV floor-occupancy rasterizes cleanly** — oriented XY footprints → world grid; ~100×100 m warehouse,
   514×504 @ 0.2 m, sparse (0.45%) as expected.
3. **Frame + world frame align** — video frame 0 ↔ GT key `"0"`; world meters; richer per-camera overlap
   than nuScenes, exactly as predicted (`cam2ego → cam2world`, GT already in world coords).
Repro: `scripts/derisk_smartspace.py <scene_dir> <out_png>`. **Go for Phase 1 (the adapter).**

## 12. Next step
Phase 1 — `prep_smartspace.py` (the adapter): sampled frames + fixed N-cam subset + world BEV-occupancy GT
→ npz `train`/`val`, per-space split (first 70% time / last 30%). Pure local code, no GPU/cost; mirrors
`prep_nuscenes_occ.py`. Then Phases 2–5 (reference → grader+gate → live agent → docs).
