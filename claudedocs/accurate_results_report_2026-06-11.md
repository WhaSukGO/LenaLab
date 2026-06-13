# Accurate-Results Report — post-trustworthiness-audit

*2026-06-11. Outcome of the audit triggered by the retracted sim-faithfulness "flagship." Goal: every
headline number is now either **verified-reproducible**, **variance-bounded**, or **honestly retracted**.*

---

## How each claim was stressed
- **Scene-degeneracy check** — measured trajectory curvature of every test scene (a straight path makes
  Sim(3) ATE trivial). *Result: only the sim-faithfulness city scene was degenerate; all others are curvy.*
- **Variance** — learned-VO rungs re-trained **n=3** with different seeds → mean ± std (GPU training is not
  bit-deterministic even at fixed seed; measured ±~1.5 m).
- **Reproducibility** — classical/C++ results re-run to confirm determinism.

## Verdict table

| # | Claim (original) | Audited result | Verdict |
|---|---|---|---|
| 17 | Fidelity ladder: rendered **27.1** ≈ real **27.2**, procedural **69.6** | n=3: procedural **69.71±0.05**, rendered **27.35±1.49**, real **25.61±1.64** | ✅ **HOLDS** — rendered≈real within noise; false 0.1 m precision removed |
| 16 | Learned VO synth in-domain **0.45 m** (beats ref 3.26 ~7×) | n=3: **0.55 ± 0.05 m** (~6×) | ✅ holds (magnitude corrected; 0.45 was a low single draw) |
| 16 | sim→real collapse 0.45 → **69.6 m** | 0.55 → 69.7 m (~125×) | ✅ holds |
| 18b | classical VO by env: 0.16 / 4.04 / 0.93 | reproduced **identical** every run | ✅ deterministic |
| 18b | C++ VO diverges on road/residential (55 / 100 m) | city/residential reproduce (0.08 / 100.76); **road non-deterministic (52–64 m) but always diverged** | ✅ qualitative holds; exact road magnitude unstable (expected for lost track) |
| 12 | Contamination probe: agent **1.20%** beats reference **1.91%** | agent **1.197%** (recorded, =1.20%) on curvy non-degenerate scenes; **positive control reproduced** (ref VO 1.711% on fresh scenes); agent < reference → **beats it** | ✅ holds (deterministic-classical; ordering robust) |
| 18a | C++ VO Phase-1 **2.18%** | reproduced **exactly 2.18%** (synth1 1.97, synth2 2.4) | ✅ deterministic |
| 9 | KITTI BA (M1) **2.03% robust (4 attempts)** | recorded best **2.026%**, but attempts ranged **2.03–8.7%**; re-run gave **2.914%** | ⚠️ **corrected** — 2.03% was the *best* attempt, not "robust"; M1 is variable; near-miss regardless |
| 15 | M3 VIO **3.83%** (agent fuses IMU; vs VO-alone 18%, ref 4.2%) | agent **3.830%** (recorded, =blog); de-risk reproduced anchors **VO-alone 18.0%, ref VIO 4.2%** | ✅ holds (agent VIO beats VO-alone, ≈ ref) |
| 8 | Learned VO on GPU **19.8 m** (beats ref 31.5) | n=3: **18.50 ± 0.68 m** (low variance) | ✅ holds (recorded 19.77 within range; beats ref robustly) |
| 17 | rendered+aug **26.9 m** | n=3: **26.49 ± 0.93 m** — within noise of rendered & real | ✅ holds (aug doesn't break ceiling) |
| 18b | **sim-faithfulness** delta 0.033 m ("sim validates a learned SLAM") | retraction stands; root cause was a relative-`docker -v` path bug (not RAM). Re-run across 7 scenes: **SCENE-DEPENDENT** — faithful on short scenes (delta ≤0.14m, e.g. city_0005 0.10 reproduced) but FAILS on long (road_0015 delta **9.4m**, residential_0019 **2.2m**) | ❌ blanket claim false; ✅ qualified ("short close-range scenes") |
| 2 | mono VO **0.052 m** "beats reference" | already corrected (shape-only, 40× rescale) by earlier review | ✅ caveated |
| 5 | SLAM **0.185 m** | already corrected (in-sample: dev=test) by earlier review | ✅ caveated |
| 3 / 6 | RGB-D **0.033 m** (SE3) / KITTI **3.53 m** (SE3) | deterministic, reproduced 5×/2× (prior stat note); curvy scenes | ✅ solid |

## Headline conclusions after the audit
1. **The fidelity-ladder result survives and is strengthened.** "Rendering closes the sim-to-real appearance
   gap" is robust: rendered ≈ real ≈ 26 ± 1.5 m, both far below procedural ~70. What was wrong was reporting
   single-run numbers as if precise to 0.1 m.
2. **The sim-faithfulness flagship was overclaimed and retracted — and, re-run properly, is SCENE-DEPENDENT.**
   The DROID *integration* (builds + runs SOTA learned SLAM on real data) is real. The blanket accuracy claim
   is false; the qualified one holds: rendered sim ≈ real on **short, close-range** scenes (delta ≤0.14 m,
   reproduced), but **fails on long/far-field** scenes (road_0015 delta 9.4 m, residential_0019 2.2 m, both
   reproduced) because the stereo-depth reprojection has no far-field geometry.
3. **The classical/agent-authored numbers are deterministic and trustworthy**; the C++ VO's divergence on
   hard drives is real (the exact diverged value is not, which is expected).
4. **Two of my own diagnoses were wrong and are corrected here:** (a) the "real DROID runs fail" was a
   **relative `docker -v` path bug**, not RAM; (b) the misleading sim-faithfulness figure was *my own*
   uncentered Sim(3) alignment. The 16 GB limit was real (a WSL2 `.wslconfig` cap on a 64 GB host, raised to
   48 GB — which also stopped the "reboots", actually WSL memory-pressure restarts), but it was **not** the
   cause of the sim-faithfulness failures.

## Honest caveats that remain
- Learned numbers carry ±~1.5 m training noise; only comparative claims that survive it are kept.
- Sim-faithfulness is established only for **short close-range** scenes; long-range remains a real gap of the
  reprojection renderer. **Optimised 3DGS** (vs crude reprojection) is the open follow-up that might close it.
- "Rendered" = stereo-depth reprojection of real pixels (precursor to optimised 3DGS), not synthetic-photoreal.
- Only 1 genuinely curvy *short* scene (city_0005) is reproduced n=2; more would tighten the short-scene claim.
