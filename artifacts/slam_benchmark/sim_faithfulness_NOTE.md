# Sim-faithfulness: the honest, evidence-based verdict (rev. 2026-06-12)

## Short version
The blanket claim "a 3DGS-rendered scene is faithful enough to validate a learned SLAM" **does not hold.**
It is **scene-dependent**: the reprojection-rendered sim is faithful on **short, close-structure** scenes,
and **fails badly on long / far-field** scenes. The original retraction stands; we now have the mechanism.

## How we got here
1. Original flagship (city_0001, straight): delta 0.033 m → claimed PROVEN → **RETRACTED** (straight scene =
   trivial metric, "not reproducible", misleading figure).
2. Audit found the *real* reason the non-straight runs failed: a **relative `docker -v` path bug** (the scene
   dir was passed relative; the `real` mount failed silently), NOT the 16 GB RAM as long assumed. Fixed
   (`CITY.resolve()`). Also fixed a Sim(3)-alignment bug in the *figure* (uncentered umeyama).
3. With those fixed + 48 GB RAM, ran DROID real-vs-rendered across 7 scenes (4 genuinely curvy).

## The data (DROID Sim(3) ATE, m; delta = |rendered − real|)
| scene | curvature | path len | real | rendered | delta | rendered faithful? |
|---|---|---|---|---|---|---|
| residential_0035 | 8.3 m | 60 m | 0.10 | 0.11 | **0.014** | ✅ |
| city_0005 | 13 m | 69 m | 0.49 | 0.59 | **0.103** (n=2 reproduced) | ✅ |
| city_0013 | 4.8 m | 173 m | 0.30 | 0.44 | 0.136 | ✅ |
| residential_0019 | 8.9 m | **406 m** | 0.61 | 2.81 | **2.207** | ❌ |
| road_0015 | 8.9 m | **363 m** | 0.39 | **9.75** | **9.367** (reproduced) | ❌✗ |
| city_0001 (straight) | 2.2 m | 107 m | 0.08 | 0.12 | 0.032 | ✅ (degenerate metric) |
| residential_0079 (straight) | 1.1 m | 26 m | 0.10 | 0.08 | 0.012 | ✅ |

## Verdict
- **The splitting variable is path length / far-field content, not curvature.** Short scenes (≤~170 m,
  close structure) → delta ≤ 0.14 m, rendered ≈ real. Long scenes (>350 m, far-field road) → delta 2–9 m,
  rendered DROID drifts badly.
- **Mechanism:** the renderer is stereo-depth reprojection (point-splatting of real pixels). Far-field
  structure has poor/no stereo depth, so long scenes render with degraded geometry → DROID accumulates
  drift on the rendered frames while tracking the real frames fine. The road_0015 figure shows the rendered
  trajectory diverging early.
- **So: the rendered sim can validate a SLAM on short, close-range scenes, but is NOT a substitute on
  long-range driving.** A blanket "sim validates SLAM" claim is false; a qualified one is true. Optimised
  3DGS (vs crude reprojection) is the obvious thing that might extend faithfulness to long range — open.

## Why this matters (process)
Strengthening *before* un-retracting caught a second would-be overclaim: city_0005 alone (delta 0.10 m)
looked like a clean un-retraction, but adding long curvy scenes revealed the failure. The retraction was
correct; the honest result is the *scene-dependence*, not a yes/no.
