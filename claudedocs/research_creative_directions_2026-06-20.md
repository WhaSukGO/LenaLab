# Research: What's our *edge*? Creative directions for the perception lab

*Research report · 2026-06-20 · LenaLab · depth: standard (2-hop, web-sourced)*

## Executive summary

**Don't build the 500th nuScenes BEV model.** BEV/occupancy *on autonomous-driving benchmarks* is a
saturated, compute-bound race owned by big labs — a solved problem where our from-scratch, single-GPU
result will always trail SOTA and add nothing new. Chasing a "full-scale nuScenes number" buys
credibility but **no edge**.

Our actual edge is **not the model — it's the lab**: a verification-first system where an AI agent
*autonomously researches, builds, trains, and honestly verifies* a perception model, and where **we
own/generate the ground truth**. That edge pays off on problems that are *new*, where being first with
a *verified working model* is the contribution — not on saturated leaderboards.

**The highest-leverage move:** re-aim the exact BEV/occupancy machinery we already built at
**"physical-AI smart spaces"** — static multi-camera BEV/occupancy for **warehouses, retail, hospitals,
and robots** — using NVIDIA's **Physical AI Smart Spaces** dataset (public on Hugging Face, ~250 h /
~1,500 cameras, built for 3D occupancy + multi-camera tracking, *far* less saturated than nuScenes).
Same camera→lift-splat→voxel math we mastered, a fresh and commercially live domain, and a place where
our verification rigor and agent-authoring are genuine differentiators. *(Confidence: high.)*

---

## Q1 (your factual check): did we really train models on nuScenes mini?
**Yes.** Real GPU training, real held-out measurement — both the reference Lift-Splat nets (checkpoints
saved) and the agent-authored nets (the agent wrote the training loop, trained, was graded on held-out
`mini_val`). The numbers (BEV ~0.13, occ ~0.08 IoU) are genuine. "mini" just means a 10-scene slice, so
they're real-but-small.

---

## The core finding: where the field is crowded vs open

| | Saturation | Who owns it | Our position |
|---|---|---|---|
| **Ego-vehicle BEV/occ (nuScenes, Waymo)** | extreme — COCO/nuScenes appear in 60k+ papers/yr; dozens of occ methods (TPVFormer, SurroundOcc, FB-OCC…) | big labs, huge compute | can't win; would just trail SOTA |
| **Static / infrastructure-camera BEV** (roadside, intersections) | growing but young (UrbanIng-V2X, TUMTraf-V2X, InScope, H-V2X — all 2024-25) | emerging, fragmented | **open** — and the math transfers |
| **Indoor / smart-space multi-cam BEV+occ** (warehouse, retail, robots) | new (NVIDIA Physical AI Smart Spaces, AI City Challenge '24/'25) | **public dataset, underexplored** | **strong fit** — reuses our code |
| **Aerial / drone semantic BEV** (agri, wildlife, disaster) | data exists, *annotation-starved* | scattered, no dominant benchmark | **open** — verification/GT-gen edge |
| **Indoor occupancy for assistive tech** (navigation for blind) | tiny (SliceOcc, a few 2025-26 papers) | wide open | high social value, niche |

**The pattern:** the entire BEV/occupancy field is ~95% pointed at the moving ego-vehicle. The *same
geometry* (multi-camera → lift to a BEV/voxel grid) applied to **fixed cameras, indoor scenes, aerial
views, or robots** is comparatively wide open — and that's exactly the geometry our lab already
implements and verifies.

---

## Where a small verification-first agent lab actually beats big labs

We win on three axes the giants under-serve:
1. **Novelty over scale.** On a fresh domain, a *verified working model where none existed* is the
   contribution. No leaderboard to lose to.
2. **Verification & honesty as the product.** The field is sloppy about held-out rigor and reproducibility
   (real-world data "falls short," long-tail ignored). Our harness-owned GT + held-out grading + variance
   checks + retraction discipline is a *methodological* differentiator, not a model one.
3. **Owning the benchmark.** Our superpower is *generating* ground truth (we rasterize BEV/occ GT from
   3D boxes). On an underexplored domain we can **create a new verified benchmark** — itself a research
   contribution — then have the agent author + verify a model on it. "Use data nobody trained on" *and*
   "create a new dataset" collapse into one move we're uniquely good at.

---

## Creative directions, ranked (the "out of the box, yet in life" answers)

### ⭐ 1. Physical-AI smart spaces — static multi-cam BEV/occupancy for warehouses & robots
- **The gap:** BEV/occ assumes a moving car; *fixed overhead/wall cameras* watching a fixed space is a
  different, far-less-saturated problem with a live market (warehouse automation, retail analytics,
  embodied/"physical AI", safety).
- **The data (underexplored, public):** NVIDIA **Physical AI Smart Spaces** (`nvidia/PhysicalAI-SmartSpaces`
  on Hugging Face) — ~250 h, ~1,500 cameras, warehouse/retail/hospital, with 3D boxes + calibration +
  service-robot/humanoid classes; AI City Challenge '24/'25. Public, fresh, ~30k downloads but *modeled by
  few* relative to nuScenes.
- **Why it's our edge:** the camera→calib→lift-splat→voxel math we already wrote **transfers almost
  directly** (multi-cam + calibration + a GT grid — the same data contract). We could even **generate our
  own BEV/occupancy GT** from its 3D boxes (exactly our nuScenes recipe), creating a verified smart-space
  benchmark + an agent-authored model on it.
- **Application:** warehouse robot safety, "is this aisle/floor-cell occupied," retail flow, fall
  detection in hospitals — real, fundable, and aligned with the embodied-AI wave.
- **Feasibility:** *high* — biggest reuse of existing code; the cloud pipeline we just proved runs it.

### 2. Aerial / drone semantic-BEV for a high-value, annotation-starved domain
- **The gap:** overhead drone imagery *is* a bird's-eye view, yet semantic-BEV/occupancy framing is rare;
  the bottleneck is **annotations**, not models.
- **The data:** agriculture (CSCD crop UAV), disaster (Open Cities AI 400 km², CRASAR-U-DROIDs building
  damage), wildlife aerial counts — all under-modeled vs AD.
- **Why it's our edge:** annotation-scarcity is exactly where *GT-generation + verification-first* shines;
  and "first verified model for X" is a real contribution. High social value (conservation, disaster).
- **Feasibility:** *medium* — different sensor geometry (single overhead vs surround), more adaptation.

### 3. The methodology *as* the product — "an agent that cracks a new perception problem, verified"
- **The gap/claim:** the model isn't the novelty; the **autonomous agent that takes a brand-new
  multi-camera setup and produces a verified, generalizing model where none existed** is. Smart-spaces (#1)
  is the *demo* of this; the framing is the differentiator for a portfolio/role.
- **Why it's our edge:** nobody's "BEV model" is novel; "a lab where an AI agent does the research +
  proves it honestly, on demand, for a new domain" is.
- **Feasibility:** *high* — it's a reframing of #1, not new infra.

### 4. (Niche, high-meaning) Indoor occupancy for assistive navigation
- Wide-open, socially valuable, small data — good story, smaller market. *Confidence/market: lower.*

---

## Recommendation (for your decision — not yet implemented)
Pursue **#1 + framing #3 together**: take the agent-research lab to **NVIDIA Physical AI Smart Spaces**,
generate a harness-owned BEV/occupancy GT, and have the agent author + verify a static-multi-camera model
— a *new* problem, public-but-underexplored data, reusing our proven math + cloud pipeline, in a live
market (physical/embodied AI). That's a genuine edge — first-with-a-verified-model on fresh ground —
versus "another nuScenes number" that competes where we can't win.

A useful *complement* (not a substitute): one full-scale nuScenes number purely as a **credibility
anchor** ("the pipeline produces real-benchmark results") — but the *creative, edge-bearing* work is #1.

**Next step options (you decide):** `/sc:design` a smart-spaces domain plan, or a quick de-risk
(download a slice of the NVIDIA dataset, confirm our lift-splat math + GT-generation transfer).

## Sources
- NVIDIA Physical AI Smart Spaces / AI City Challenge: [HF dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) · [9th AI City Challenge (ICCV'25)](https://openaccess.thecvf.com/content/ICCV2025W/AICity/papers/Tang_The_9th_AI_City_Challenge_ICCVW_2025_paper.pdf) · [dataset blog](https://zhengthomastang.github.io/posts/2025/03/blog-post-1/)
- Multi-cam BEV tracking (indoor/retail/warehouse): [BEV-SUSHI / MCBLT](https://arxiv.org/abs/2412.00692)
- Roadside / infrastructure BEV: [UrbanIng-V2X](https://arxiv.org/abs/2510.23478) · [calibration-free roadside BEV (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252405/) · [InScope](https://arxiv.org/pdf/2407.21581)
- Indoor occupancy: [SGR-OCC / embodied occ](https://arxiv.org/pdf/2603.14076) · [indoor occ for visually impaired](https://arxiv.org/pdf/2602.16385)
- Aerial: [UAV semantic-seg review](https://www.sciencedirect.com/science/article/abs/pii/S0924271624000844) · [Open Cities AI / CRASAR-U-DROIDs](https://arxiv.org/pdf/2407.17673)
- Dataset-saturation context: [CV datasets 2026](https://www.cvat.ai/resources/blog/popular-computer-vision-datasets) · [real data falls short (SKY ENGINE)](https://www.skyengine.ai/blog/why-real-world-data-will-fall-short-in-your-computer-vision-project-in-2026)

*Confidence: high on the AD-saturation / non-AD-openness split and the NVIDIA dataset's availability +
fit; medium on specific market sizing and aerial adaptation effort. Report only — no code changes made.*
