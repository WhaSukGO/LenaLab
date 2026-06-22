# Research: Datasets for Cross-Space Generalization (smart-space occupancy)

*Research report · 2026-06-22 · LenaLab · for option (b): scale up the training-space diversity so a
model can generalize to an unseen space. Report only — no implementation.*

## Executive summary

Our cross-space failure (unseen-warehouse IoU 0.037–0.044 vs ~0.39 per-space) is a **training-space
diversity** problem — we trained on **4** warehouses. The fix is *more distinct spaces*, and the single
best source is **the rest of the dataset we already use**: NVIDIA Physical AI Smart Spaces has **~141
distinct scenes** across three editions and **6 environment types** (warehouse, retail, hospital, …) —
we used **4**. Same format → our adapter largely already works. Beyond that: **MMPTRACK** (5 *real*
indoor environments, a real-data cross-check), and **synthetic generation via NVIDIA Isaac
Sim/Replicator** (unlimited spaces — how NVIDIA built this dataset in the first place). Classic
multi-camera sets (WILDTRACK, MultiviewX, Campus, Shelf) are **single-scene** → useful only as extra
*test* spaces, not for training diversity. *(Confidence: high on scene counts — verified on HuggingFace.)*

## Findings — datasets ranked by fitness for *cross-space training*

| Dataset | Distinct spaces | Multi-cam + calib? | Real/Synth | Fit for cross-space | Notes |
|---|---|---|---|---|---|
| **NVIDIA Physical AI Smart Spaces** (all editions) | **~141** (2024: 90 · 2025: 23 · 2026: 28) | ✅ yes | Synthetic (Omniverse) | ⭐⭐⭐ **best** | **6 env types** (warehouse/retail/hospital…). Same family → our `prep_smartspace_xspace.py` ~works. We used 4. |
| **MMPTRACK** (Microsoft) | **5** environments | ✅ calibrated, overlapping | Real | ⭐⭐ good cross-check | ~9.6 h, dense person annotations; tracking labels → derive floor occupancy. Real-world diversity (modest count). |
| **Isaac Sim / Omniverse Replicator** | **unlimited** (procedural) | ✅ auto calib + labels | Synthetic | ⭐⭐ principled, high-effort | Generate our own diverse spaces; the method NVIDIA used. Big setup cost. |
| **MC-BEVRO** | few (traffic intersections) | ✅ 4 static cams → BEV occ | Real/sim | ⭐ adjacent | Outdoor traffic BEV occupancy (vehicles+pedestrians); different domain but same task shape. |
| **WILDTRACK** | **1** (outdoor square, 7 cams) | ✅ | Real | ✗ training / ✓ test | Single scene → can't teach cross-space; good as an extra unseen *test* space. |
| **MultiviewX** | **1** (synthetic, 6 cams) | ✅ | Synthetic | ✗ training / ✓ test | Single scene; WILDTRACK-like. |
| **Campus / Shelf** | 1 each (3–5 cams) | ✅ | Real | ✗ | Tiny; 3D pose tracking. |

## The recommended path (cheapest, highest-yield first)

1. **Scale within NVIDIA Smart Spaces — the obvious win.** Go from 4 → **~20–40 diverse scenes**,
   deliberately spanning **environment types** (add retail + hospital, not just warehouses) and the
   2024 edition (90 scenes). That's a **5–35× increase in space diversity** from a dataset we already
   parse, on the *exact same task*. This is the highest-probability way to actually move the
   unseen-space number.
2. **Add MMPTRACK as a real-world cross-check** — if the model trained on synthetic NVIDIA scenes also
   transfers to MMPTRACK's *real* indoor spaces, that's a much stronger generalization claim (sim→real).
3. **If still short, generate spaces with Isaac Sim/Replicator** — unlimited procedural indoor
   multi-camera scenes with calibration + occupancy labels. Principled but a real engineering project.
4. **Reserve WILDTRACK / MultiviewX as held-out *test* spaces** — diverse unseen evals, not training.

## Honest caveats (cost & compatibility)

- **Download size is the real cost.** Smart-space scenes are ~1.5–3 GB each → 30 scenes ≈ **60–100 GB**,
  90 ≈ **200 GB+**. Mitigation: subsample frames/cameras, or pull a curated ~20–30-scene diverse set
  rather than everything.
- **2024 edition differs.** Scenes are named `scene_001…` (not `Warehouse_XXX`) and span 6 env types —
  the **annotation schema may differ**, so the adapter likely needs a small tweak/verification per
  edition (a Phase-0 check).
- **Compute scales with scenes.** 30 scenes is ~7× our current 4 → the cross-space train would want a
  cloud GPU (or a long local run), not a quick local job.
- **Synthetic-only risk.** NVIDIA Smart Spaces is *synthetic*; a model that generalizes across synthetic
  warehouses may still not transfer to *real* cameras — which is exactly why **MMPTRACK (real)** as a
  cross-check matters before claiming real-world cross-space ability.

## Recommendation (for your decision)
**Do #1 first**: a curated ~20–30-scene, multi-environment-type pull from NVIDIA Smart Spaces, retrain
the scene-agnostic model (resume from our saved pretrained checkpoint), and re-measure unseen-space IoU.
It's the cheapest credible shot at closing the cross-space gap, reuses everything we built, and the
result is honest either way (it works → strong claim; it doesn't → the gap is fundamental, also a finding).
Add **MMPTRACK** only if #1 shows promise, to test sim→real.

## Sources
- NVIDIA Smart Spaces (90 scenes / 6 env types): [HF dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) · [NVIDIA blog (largest indoor synthetic dataset)](https://blogs.nvidia.com/blog/ai-city-challenge-omniverse-cvpr/) · [dataset blog (Z. Tang)](https://zhengthomastang.github.io/posts/2025/03/blog-post-1/) — scene counts (2024:90 / 2025:23 / 2026:28) verified on the HF file tree.
- [MMPTRACK (5 calibrated indoor environments)](https://ar5iv.labs.arxiv.org/html/2111.15157)
- [WILDTRACK](https://www.semanticscholar.org/paper/WILDTRACK:-A-Multi-camera-HD-Dataset-for-Dense-Chavdarova-Baqu%C3%A9/36bccfb2ad847096bc76777e544f305813cd8f5b) · [MultiviewX (in MVDet/lifting-to-BEV)](https://arxiv.org/pdf/2403.12573)
- [Isaac Sim / Omniverse Replicator synthetic generation](https://developer.nvidia.com/blog/generating-synthetic-datasets-isaac-sim-data-replicator/)
- [MC-BEVRO (static multi-cam BEV occupancy)](https://arxiv.org/html/2502.11287v1)

*Confidence: high on NVIDIA scene counts + single-scene-dataset classification; medium on 2024 schema
compatibility (naming differs) and on whether more synthetic scenes alone close the sim→real gap.*
