# Research: What odometry/SLAM algorithm should the lab move to next?
*(from an autonomous-driving team's perspective)*

- Date: 2026-06-04 · Depth: standard (2 search rounds, 5 queries, 4 hops) · Research-only (no implementation)
- Context: LenaLab just reached **rung 1** — basic frame-to-frame **stereo VO at ~2.0 % KITTI t_err**, single RTX 3080 16 GB, camera-only. Question: what algorithm is the right next climb?

---

## Executive summary (the recommendation)

The field has moved decisively from **classical geometric SLAM → learned VO/SLAM → 3D-foundation-model SLAM**, and an AV team would not invest further in *pure classical VO* as a destination. Three grounded recommendations, in priority order:

1. **PRIMARY — move to learned VO, specifically DPVO / DPV-SLAM.** It is the current *efficient* learned SOTA, **fits the RTX 3080 trivially** (60 FPS, ~4–5 GB on a 3090), and **crushes DROID-SLAM on KITTI** (DPV-SLAM ATE **25.8 m** vs DROID **118.7 m**). The lab already has a learned-VO GPU pipeline, so this is a direct upgrade of the naive DeepVO-lite — and a genuine SOTA-class target the single GPU can afford. *Confidence: HIGH.*
2. **PARALLEL (AV-realism) — add the inertial axis (VIO).** No production AV stack uses *pure* VO; every one fuses IMU (and usually LiDAR + GNSS + HD maps). KITTI ships IMU/GPS (OXTS). Visual-**inertial** odometry is the most AV-faithful next step and adds scale observability + robustness. *Confidence: HIGH that AV uses it; MEDIUM that it's the best *next* lab step vs DPVO.*
3. **FRONTIER bet (later) — MASt3R-SLAM / foundation-model SLAM (CVPR 2025).** The 2025 direction: dense monocular SLAM built on the DUSt3R/MASt3R 3D-reconstruction prior, real-time (15 FPS), robust on low-texture and in-the-wild video. Heavier, and it *uses a pretrained foundation model* — which makes it a sharp **rung-4 contamination probe** (does the agent understand it or recall it?). *Confidence: MEDIUM (frontier, fast-moving).*

**De-prioritize:** classical ORB-SLAM-style BA+loop-closure *as an end goal* (the field has surpassed it — though loop closure as a *component* is high-leverage: it takes DPV-SLAM from 53 → 26 m); and LiDAR fusion (the lab has no LiDAR sensor/data).

---

## 1. The landscape (what changed)

**Classical geometric SLAM is being surpassed on robustness.** In 2024–25 multi-paradigm evaluations, **ORB-SLAM3 fails critically under degradation** (e.g. 0 % tracking under dense haze), while learning-based methods stay robust (MASt3R lowest degraded ATE 0.027 m; DUSt3R 96.5 % tracking success) [VSLAM-LAB; UAV multi-paradigm eval].

**Learned VO/SLAM is the working SOTA.** The five systems the field now benchmarks are **ORB-SLAM3, DPVO, DROID-SLAM, DUSt3R, MASt3R** — spanning classical, CNN-patch, recurrent, and ViT/foundation paradigms [VSLAM-LAB].

**Efficiency winner = DPVO/DPV-SLAM.** Deep Patch VO runs **2–5× real-time on a single GPU** (60 FPS / ~4.9 GB on a 3090), uses **1/3 the memory and 3× the speed of DROID**, and **DPV-SLAM beats DROID by large margins on KITTI** (ATE 25.8 vs 118.7 m) — DROID degrades badly at KITTI's outdoor scale [Deep Patch VO, arXiv 2208.04726; Deep Patch Visual SLAM, ECCV 2024].

**2025 frontier = 3D-foundation-model SLAM.** **MASt3R-SLAM** (CVPR 2025): real-time dense monocular SLAM from the MASt3R two-view 3D prior, globally consistent, *no fixed camera model assumption*, 15 FPS [arXiv 2412.12392]. Follow-ons already appearing (FoundationSLAM; multi-agent MASt3R). Also **GigaSLAM** (2025): large-scale monocular SLAM via hierarchical Gaussian splatting.

## 2. What an autonomous-driving team actually runs (production reality)

Production AV localization is **multi-sensor fusion, not pure VO**: **LiDAR-Inertial-Visual Odometry (LIVO)** (e.g. FAST-LIVO2 on edge platforms), tightly-coupled **LiDAR-Inertial** with factor-graph optimization, **GNSS-fused** LIO for global consistency, and **HD maps** for map-based localization [FAST-LIVO2, arXiv 2501.13876; GNSS+LIO, 2503.23199; AV maps survey, 2509.12632]. Benchmarks have also moved beyond KITTI (2012) to **4Seasons** — 300+ km over a year, nine environments, for *challenging-condition* VSLAM/relocalization [4Seasons, arXiv 2301.01147].

**Implication for the lab:** an AV team values, in order: (a) **robustness** to night/weather/dynamics, (b) **inertial fusion** (always present), (c) **real-time/edge** efficiency, (d) **map/loop-closure** global consistency. The lab's pure stereo VO has none of (a),(b),(d) yet.

## 3. Candidates mapped to the lab's rungs (compute fit + expected gain)

| Candidate | Paradigm | Fits RTX 3080? | KITTI strength | Why / why not | Confidence |
|---|---|---|---|---|---|
| **DPVO / DPV-SLAM** | learned patch VO (+classical loop closure) | **Yes, easily** (~5 GB) | **Strong** (25.8 m, beats DROID) | efficient learned SOTA; direct upgrade of the lab's learned track | **HIGH** |
| **VIO (visual-inertial)** | classical/learned + IMU | Yes (CPU/GPU light) | n/a (different axis) | AV-realistic; adds scale + robustness; KITTI has OXTS IMU | HIGH (relevance) |
| **Add BA + loop closure** to current stereo VO | classical component | Yes | big local gain (53→26 m effect) | high-leverage *building block*, but not a destination | HIGH |
| **MASt3R-SLAM / DUSt3R** | 3D foundation model | Heavier (ViT); likely OK at low res | robust, dense, 15 FPS | 2025 frontier; great rung-4 contamination probe | MEDIUM |
| **DROID-SLAM** | recurrent learned | borderline (VRAM-heavy) | **weak on KITTI** (118 m) | superseded by DPVO on efficiency *and* KITTI | LOW (skip) |
| **LiDAR-Inertial / LIVO** | sensor fusion | n/a | production standard | the lab has **no LiDAR data** — out of scope unless added | n/a |

## 4. Recommendation for human decision

**If the goal is "climb the SOTA ladder on the hardware we have":** go **DPVO/DPV-SLAM** (rung 3, learned). It is the single best-justified move — current, efficient, KITTI-dominant, and the lab's GPU pipeline already supports it. Expected: from ~2 % t_err (basic VO) toward the learned-SOTA regime, on one GPU.

**If the goal is "be a credible autonomous-driving lab":** add the **inertial axis (VIO)** in parallel — pure VO is not AV-realistic, and IMU fusion is the cheapest step toward production fidelity (and toward the 4Seasons-style robustness AV actually cares about).

**Highest-credibility experiment regardless of choice:** evaluate on a **post-KITTI, challenging-conditions benchmark (4Seasons)** and/or a **contamination probe** — KITTI is 2012 and almost certainly in the model's training data, so any KITTI "SOTA" is partly recall. Moving to 4Seasons (or MASt3R-SLAM, which leans on a foundation model) is where "the lab does research" separates from "the lab recalls papers."

**Suggested sequence:** DPVO learned-VO climb (rung 3) → add loop closure (the 53→26 m lever) → VIO for AV-realism → 4Seasons / contamination probe (rung 4) → MASt3R-SLAM frontier.

---

## Sources

- [Deep Patch Visual Odometry (arXiv 2208.04726)](https://arxiv.org/abs/2208.04726v1) — DPVO efficiency (60 FPS, ~4.9 GB, single GPU; 1/3 memory, 3× faster than DROID)
- [Deep Patch Visual SLAM, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00272.pdf) / [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72627-9_24) — DPV-SLAM KITTI ATE 25.8 m vs DROID 118.7 m
- [VSLAM-LAB (arXiv 2504.04457)](https://arxiv.org/html/2504.04457v1) — unified framework comparing ORB-SLAM3, DPVO, DROID, DUSt3R, MASt3R
- [Robust Visual SLAM for UAV, multi-paradigm eval (arXiv 2605.03678)](https://arxiv.org/html/2605.03678) — classical (ORB-SLAM3) fails under degradation; learned methods robust
- [MASt3R-SLAM, CVPR 2025 (arXiv 2412.12392)](https://arxiv.org/abs/2412.12392) / [project](https://edexheim.github.io/mast3r-slam/) / [code](https://github.com/rmurai0610/MASt3R-SLAM) — real-time dense monocular foundation-model SLAM, 15 FPS
- [FAST-LIVO2 (arXiv 2501.13876)](https://arxiv.org/pdf/2501.13876) — LiDAR-Inertial-Visual odometry on edge
- [GNSS + LiDAR-Inertial localization (arXiv 2503.23199)](https://arxiv.org/pdf/2503.23199) — production AV fusion
- [Maps for Autonomous Driving survey (arXiv 2509.12632)](https://arxiv.org/html/2509.12632v1) — HD maps + map-based localization
- [4Seasons benchmark (arXiv 2301.01147)](https://arxiv.org/html/2301.01147) — challenging-condition AV VSLAM benchmark beyond KITTI

*Confidence overall: HIGH on the field direction (classical→learned→foundation) and DPVO's fit; MEDIUM on the exact ordering of VIO vs DPVO and on the fast-moving 2025 frontier. Per /sc:research, this is a report for your decision — no changes made.*
