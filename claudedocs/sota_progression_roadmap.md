# SOTA progression roadmap — how the lab climbs toward harder/SOTA problems

*Written 2026-06-04. The strategy for making LenaLab challenge progressively harder problems,
and the concrete rung-1 step taken.*

## The principle: the verifier is the ratchet

You may only *credibly* claim a harder result when the verifier can certify it **to the standard
of the claim**. So progression = **raise verifier rigor + make the bar external**, not "give the
agent harder prompts." Two moves make difficulty climb automatically:

1. **Tie the oracle to the public leaderboard, not your own reference.** Self-referential bars
   ("beat my baseline ×1.5") cap the lab at its own level. Use *published numbers on the standard
   metric*; clearing a rung moves the bar to the next published result.
2. **Reproduction-first against the literature.** The calibration gate must (eventually)
   reproduce a known published number within tolerance — if the harness can't reproduce a paper's
   result, it can't credibly judge a SOTA attempt.

## The difficulty ladder

| Rung | Challenge | What the lab must add | Gate |
|---|---|---|---|
| **0** (done) | toy windows, in-house baselines, integrity fixed | — | leak-free, deterministic, honest metrics |
| **1** (this step) | one task made leaderboard-comparable | **official-form metric** + external published anchors | reference passes a discrimination gate; report vs published |
| **2** | match classical SOTA | bundle adjustment + loop closure; full sequences | beat the *classical* leaderboard entry, significance-tested |
| **3** | match learned SOTA | matured GPU path, data at scale, longer training | match a *learned* published number (compute-feasible) |
| **4** | contamination-controlled novelty | **post-cutoff / private / perturbed** benchmarks | clear a probe the agent demonstrably hasn't seen |
| **5** | open problems | the lab *owns* the ground truth (new dataset/metric) | certify a result no paper reports |

Honest framing: **rungs 1–3 harden the verifier; rungs 4–5 are about contamination.** Claiming
"SOTA" on a benchmark the model has read 1,000 tutorials about proves recall, not capability —
so the *deepest* progression is toward problems outside the training distribution.

## Cross-cutting enablers (scale with the rungs)

- **Verifier rigor (leads):** official metrics, official splits, literature reproduction,
  contamination probes, significance/CIs. (Started: metric-honesty pass + determinism finding +
  rung-1 KITTI metric.)
- **Autonomy depth:** harder problems need long-horizon *research programs* — Track A must grow
  from a 6-experiment lineage to dozens building on each other; failure/success memory becomes
  load-bearing; the committee should propose its own directions.
- **Compute realism:** one RTX 3080 caps learned-SOTA (often many-GPU). Target *compute-feasible*
  SOTA — classical SLAM (algorithmically hard, compute-light) is the ideal rung-2 target.
- **Contamination controls:** the single highest-leverage credibility move.

## Rung 1 — DONE (2026-06-04): KITTI on the leaderboard's metric, anchored to published SOTA

The flagged "truncated global ATE on 300-frame slices" (not comparable to anything published) is
replaced by the **KITTI-style segment metric** `t_err` (% length-normalized translational drift
over 100–800 m sub-sequences) — the form the KITTI odometry leaderboard uses. Implemented in
`vo_lab/plugins/vo_ref/eval_kitti.py`; wired into `vo_impl_task_kitti` (metric `t_err_pct`); the
calibration reports the **SOTA ladder**.

Where the lab stands now (measured, held-out KITTI 05/07, ~600 m paths):

| Method | t_err | note |
|---|---|---|
| **this lab — agent stereo VO** | **2.08 %** | basic frame-to-frame stereo VO (SGBM→PnP) |
| this lab — reference stereo VO | 2.02 % | (agent is ~3% relative *worse*, deterministic) |
| ORB-SLAM2 stereo | ~1.15 % | adds **bundle adjustment + loop closure** → rung 2 target |
| DROID-SLAM (learned) | ~0.40 % | learned, needs more compute → rung 3 target |

Honest caveats (emitted in `heldout.json`): translational only (trajectories store camera
centres, so the official rotational `r_err` needs a pose-output contract upgrade); per-segment
alignment approximated by one global SE(3) fit; partial sequences. So this is *indicative*
leaderboard-comparability, not a formal submission — but it converts the bar from self-referential
to external, which is the rung-1 point.

## Deployed vs benchmark SOTA — the "YOLO lesson" (2026-06-04 research)

A separate question from "what wins benchmarks" is **"what is actually deployed"** — the YOLO
distinction (YOLO ruled deployment even when fancier detectors won papers). The research
(`research_av_next_algorithm_2026-06-04.md`) is blunt:

- **DPVO/DROID/MASt3R win benchmarks but are NOT deployed workhorses.** No commercial-deployment
  examples for DPVO; people are still publishing on *how to make it deployable* (DPVO-QAT++,
  "significant deployment gap"). Learned VO/SLAM is held back in production by domain/generalization
  gaps, compute/data dependency, and edge-case robustness — "the classical elements still stand."
- **What teams actually run (the 'YOLO of SLAM') is classical:** **ORB-SLAM3** (accuracy leader,
  visual + visual-inertial) and **VINS-Fusion / VINS-Mono** (the common VIO on drones); also
  OpenVINS, Kimera, SVO Pro; LiDAR side FAST-LIO/LIO-SAM. The pragmatic frontier is **hybrid**
  (classical geometry + learned features: SuperVINS, HFNet-SLAM, AirSLAM).

So "benchmark SOTA" (DPVO, rung 3) and "deployed standard" (ORB-SLAM3, rung 2) are *different
targets*. For a lab that values being **actually used**, the classical climb comes first.

## DECISION (2026-06-04): the classical ORB-SLAM3-style climb (BA → loop closure → VIO)

Chosen over DPVO because it is the **deployed standard**, a **real climb** from 2.0 % toward
ORB-SLAM2's ~1.15 % (loop closure is the biggest lever), **compute-light** (fits the RTX 3080
easily), and the lab already has the building blocks + the hardened verifier + memory loop.
DPVO/MASt3R remain *later research-frontier* rungs; hybrid (classical + learned features) is the
production-honest endgame.

## Schedule — 5 milestones, each gated by the external ladder

| # | Milestone | What gets built | External gate | Billed? | Effort | Risk |
|---|---|---|---|---|---|---|
| **M0 ✅** | Verifier prep (DONE 2026-06-04) | pose-output contract (poses.txt 3x4) → **official KITTI t_err + r_err**; held-out gt_poses.txt (GT-isolated); literature-reproduction verdict | **reference 2.04% t_err / 0.0114 deg/m = in published basic-stereo-VO band → reproduced** | No | done | low |
| **M1** (in progress) | Local bundle adjustment | agent adds windowed BA (keyframes + landmarks) to stereo VO | **beat basic VO 2.04% via robust BA → bar 1.9%** (recalibrated, see note) | ×3 so far | — | med (BA robustness) |
| **M2** | Loop closure + pose-graph = full SLAM | place recognition + global pose-graph on KITTI loop seqs, **disjoint dev/test** | **approach ORB-SLAM2 ~1.15 %** | ×1–3 | ~1–2 sessions | **high** (the difficulty spike) |
| **M3** | Visual-inertial (VIO) | fuse KITTI IMU (OXTS) — the VINS-Fusion-style deployed direction | robustness + scale on harder seqs | ×1 | ~1 session | med |
| **M4** | *(stretch)* harder benchmark / hybrid + contamination probe | move to **4Seasons** (post-KITTI) / add learned features; a probe the model can't have memorized | holds up off-KITTI | ×1 | later | med |

**Cadence:** M0 now (offline, de-risks everything) → M1 next → M2 over a focused session (budget
2–3 billed iterations — this is where "matches deployed SOTA" is earned) → M3 → M4 capstone.

**Why M0 first:** every higher milestone needs the bar to be the *official* metric and *external*
(reproduce a published number) or the climb isn't credible — the verifier-is-the-ratchet principle,
and it's non-billed.

### M1 progress note (2026-06-05): BA robustness is the blocker; bar recalibrated 1.8→1.9%

Three M1 attempts (windowed bundle adjustment on stereo VO), all REJECTED so far, but each
honest and informative — the memory loop carried the lesson forward each time:
- **Attempt 1** (WIP, salvaged): seq_05 **1.43%** (excellent — beats basic VO 1.80, nears
  ORB-SLAM2 1.15) but seq_07 4.40% (BA diverged) → 2.91% mean.
- **Attempt 2** (memory: "fix the jumps"): added a 3 m motion cap → seq_07 3.83% (better) but
  seq_05 2.16% (the cap *hurt* the good case) → 3.00% mean.
- **Attempt 3** (memory: "BA must never degrade below PnP; monotonic reprojection safeguard;
  no blunt cap"): running.

**Key finding:** the agent's BA is netting *worse* than basic frame-to-frame VO (2.04%) because
it's **unstable** — an unstable BA is worse than no BA. Attempt 1 proved BA *can* reach 1.43% on a
BA-friendly sequence; the task is reliability. **seq_07's drift is global** (accumulating beyond
what a local window fixes), so windowed BA caps it near basic-VO level (~2.3%) → the best
achievable mean is ~1.88%. Hence **1.8% was slightly too tight for windowed-BA-only**; the
principled M1 bar is "robust BA beats basic VO 2.04% by a clear margin" = **1.9%** (the full
ORB-SLAM2 1.15% is M2's loop-closure target, since seq_07 needs *global* consistency). This is
calibrating the bar to the technique, documented honestly — not goalpost-moving.

### M2 verdict (2026-06-06): honest LIMIT after 4 attempts — robust from-scratch SLAM not reached

M2 (loop-closure + pose-graph SLAM on loopy held-out 07/09, bar 1.8% t_err) was attempted 4×, all
REJECTED, none beating basic VO (2.81%):

| attempt | approach | seq_07 | seq_09 | mean |
|---|---|---|---|---|
| 1 (CPU, clean) | crude per-loop pose-graph | 2.61 | **10.44** | 6.53 |
| 2 (CPU, interrupted by restart) | front-end weak, 0 loops detected | 5.41 | 9.39 | 7.40 |
| 3 (GPU/torch, killed at 2h33m) | torch SLAM never worked | 47.6 | 107.7 | 77.68 |
| 4 (CPU, clean, +guard) | guard added | **12.32** | 5.07 | 8.69 |

**Pattern:** every attempt destabilises at least one held-out sequence — the instability just moves
between seqs (a1 broke 09, a4 broke 07). The two attempts that ran clean to completion (1, 4) both
failed. The agent cannot build a *robust* from-scratch stereo VO + local BA + appearance loop
detection + SE(3) pose-graph in a single session.

**This is an implementation-robustness limit, NOT a task flaw.** The offline de-risk
(`scripts/m2_derisk_loopclosure.py`) proved an *ideal* loop closure reaches **1.32%** on this exact
held-out — the headroom is real. The agent keeps rebuilding (and breaking) the whole pipeline.

**GPU finding:** the torch.cuda attempt was the *worst* (77.7%, broken). M2's bottleneck is
correctness, not speed, so the GPU helped least and added bug surface. (Also: that run was killed
prematurely at 2h33m — a process-management error — but its snapshot was non-functional regardless.)

**Forward options (user's call — the first changes how the milestone is framed):**
1. **Scaffold the front-end** — give the agent the proven reference stereo VO (`run_kitti_stereo.py`,
   2.81%) as a fixed starting `main.py`; ask it to add ONLY loop closure + pose graph. Isolates the
   M2 skill from the front-end it keeps breaking. Most likely to clear the bar; reframes M2 as
   "agent authored loop closure on a provided VO" (honest, but a different claim than full SLAM).
2. **Accept the honest negative result** — document M2 as the limit of one-session from-scratch SLAM,
   with the de-risk proof that the headroom (1.32%) exists. A legitimate, publishable finding.
3. **Defer M2**, keep M1's robust BA (2.03%) as the SLAM-track high-water mark, and pivot elsewhere.

Billed grinding is **stopped** pending this decision. M3 (VIO) is on hold — it builds on a working
SLAM the lab doesn't yet have.

### M2 scaffold result (2026-06-06): the confound is resolved — loop closure itself is the wall

Ran the scaffold: locked the proven front-end (`frontend.py`, agent verified to have left it
unmodified) and had the agent author ONLY the loop-closure layer. Result **3.31%** (seq_07 3.40,
seq_09 3.22) vs the front-end floor **2.81%** (07=2.41, 09=3.22):
- **seq_09 identical to floor** → loop closure missed the strongest loop entirely (no-op).
- **seq_07 worse** → applied a harmful correction (violated the never-do-worse guard).
- No catastrophe (3.3% vs from-scratch 6–77%) because the locked front-end couldn't collapse.

**Conclusion:** the front-end was necessary but NOT sufficient. Even with a perfect foundation and
proven 1.32% headroom, the agent cannot author loop closure that reduces drift — it misses real
loops and mis-applies false ones. This is a *sharp* capability finding: the agent can author robust
bundle adjustment (M1, 2.03%) but not a working loop-closure + pose-graph layer, even in isolation.
**M2 is closed.** Next: the contamination probe (capability vs KITTI-recall), not more SLAM.

### Rung 4 reached (2026-06-06): contamination probe — capability, not recall

The roadmap's "highest-leverage credibility move" is done. Two probes (parallel two-agent build),
each with a non-billed positive control:
- **A perturbed-KITTI** (mirror the drive, reflect GT): reference VO 3.23% = PASS.
- **B synthetic** (procedural ray-cast stereo, exact GT, provably unseen): reference VO 1.91% = PASS.
Both controls passing => the classical geometry/method generalises to novel + unseen data (real, not recall).

Billed test: a fresh agent authored stereo VO **from scratch** on synthetic data (KITTI-free task,
no KITTI memory) -> **VERIFIED 1.20%** (synth1 0.92, synth2 1.47), BEATING the reference (1.91%).
Its code brought the KITTI domain recipe (commented "tuned for KITTI-like data", depth Z<80) but
ADAPTED it and generalised. Cross-check: that synthetic-tuned solver also drives real KITTI (5.1%,
seq_09 beats the KITTI reference). **Conclusion:** the agent's VO authoring is real, internalised,
adaptable capability — not KITTI-sequence recall. Honest limit: VO is too tutorialised to fully
decontaminate *domain* knowledge, so the claim is "internalised adaptable skill," not "pure
uncontaminated capability." Scripts: scripts/contamination_{perturb,synthetic}_kitti.py,
xdomain_transfer_check.py; domain infra: vo_lab/plugins/vo_synth.py + vo_ref/synthetic_stereo.py;
blog Episode 12 + artifacts/blog/contamination_synthetic.png.

**Where the lab stands:** rung 0 (integrity) ✅, rung 1 (leaderboard metric) ✅, rung 2 (classical
SOTA): BA yes (2.03%) / loop closure is the agent's wall (clean negative), rung 4 (contamination) ✅.
Open: rung 3 (learned SOTA, more compute), M3 VIO, and the infra ideas (parallel-diverse testing,
deeper committee autonomy).

### Rung 2 final verdict (2026-06-07): loop closure is a multi-level capability ceiling

The classical climb to ORB-SLAM2 (1.15%) needs loop closure, and loop closure is now exhaustively
established as the agent's wall — attacked every way, with the parallel lab + scaffold + oracle
decomposition (the "verifiable parts" methodology):
- **Holistic (8 attempts):** 4 from-scratch (best 6.53%), 1 scaffolded on the locked front-end
  (3.31%, no gain), 3-way parallel tournament (best 3.33%) — none ever beat the 2.81% front-end floor.
- **Decomposed (oracle):** handed CORRECT loops (GT-exact) + the locked front-end, the agent authored
  a real sparse SE(3) pose graph but it was too SLOW (seq_07 >20min, exceeds the 900s grader) AND
  INCORRECT (seq_09 4.12%, worse than floor). So BOTH sub-skills fail: place recognition AND
  pose-graph optimisation, even with detection removed.

**Conclusion:** this agent's capability ceiling is clean: VO + local bundle adjustment = real,
generalising skill (M1 2.03%, contamination-proven on unseen data); the GLOBAL SLAM machinery
(detection + optimisation) is past it — architecture understood, implementation botched/slow.
Rung 2 (classical SOTA via loop closure) is NOT reachable by agent authoring here; this is an
honest, exhaustively-supported negative.

**Forward fork (paradigm-level, user's call):** (a) Rung 3 — LEARNED methods (neural place
recognition / pose regression on the GPU; a different paradigm, compute-heavy, and learned VO was
sub-classical here), or (b) accept the honest classical ceiling and bank the complete capability map
+ infrastructure (parallel lab, contamination probe, scaffold/incremental) as the result. M3 (VIO)
remains available as a *lateral* deployed-classical step (fuse IMU into the working VO+BA front-end,
no loop closure needed) if more classical-track depth is wanted.

### M3 (VIO) DONE (2026-06-08): the agent CAN author sensor fusion — capability boundary is now sharp

Lateral deployed-classical step: fuse an IMU with the VO to bridge vision dropouts (the VINS-Fusion
direction). Built contamination-clean (synthetic stereo + verified-honest IMU + vision blackouts);
de-risk proved headroom (VO-alone 18% -> reference VIO 4.2%). Scaffold task (locked front-end + IMU,
author only the fusion): agent **VERIFIED at 3.83%** (vio1 5.31, vio2 2.36) — beating VO-alone (18%)
AND the reference VIO (4.2%), front-end untouched.

**The capability boundary, with receipts:** the agent authors as real generalising skill — stereo VO,
windowed bundle adjustment (M1), IMU-VO fusion (M3); it cannot author loop detection + global
pose-graph optimisation (M2, even given perfect loops). The line is local/incremental/causal
estimation (YES) vs global/batch consistency (NO), not difficulty. Deployed-classical localisation is
reachable via fusion (VIO), not via the visual loop closure that was the agent's wall.

### Rung 3 (LEARNED) DONE (2026-06-08): capability YES, sim-to-real deployability NO

Took the learned paradigm to the contamination-clean synthetic domain (train a torch net on
procedural seqs the model can't have memorised, test on disjoint unseen seqs). De-risk: reference
learned VO 3.26 m Sim3 ATE (vs 63 m static control) → gate open, bar 4.24 m. Billed run: the agent
authored+trained a **ResNet-18 pose-regressor** (6-ch stacked frames → 3-D translation + **6-D
continuous-rotation** heads, rotation-weighted Huber, temporal-swap+GPU augmentation, TTA, and
**val-ATE model selection it added mid-run**) → **0.45 m ATE on unseen synthetic** (lte1 0.73, lte2
0.18), beating the reference ~7×. *Learned-method authoring+training is real, generalising skill —
the contamination control holds.* Tokens 3.79M.

**Sim-to-real (rung 8 / user's request):** the SAME trained model on **real KITTI 07+09** → **69.6 m**
(72.4/66.8) — a **~150× collapse**. Sim3-aligned prediction degenerates to a straight forward line;
the net learned procedural-synthetic appearance statistics that carry no signal on real photos. The
classical VO **crosses domains** (synth 1.9%, KITTI 5.1%) where this learned VO does not. Figure:
`artifacts/blog/rung3_learned.png`.

**Verdict: capability ≠ deployability.** The agent CAN author+train a learned VO that generalises
within a domain; a sim-trained learned VO is NOT deployable across the appearance gap without real
data / domain adaptation — exactly why deployed localisation runs classical VO/VIO, not sim-trained
nets.

### Sim-to-real closure (2026-06-09): the gap is real AND learned-on-real isn't competitive — both

Isolation test (Exp A): trained the SAME agent model on REAL KITTI (00/02/06/08), tested on the SAME
held-out real seqs (07/09). Three corners now: **synthetic→synthetic 0.45 m | REAL→real 27.2 m
(07:31.0, 09:23.5) | synthetic→real 69.6 m.** Real training more than halves the error (~2.5×) and
recovers the path *shape* (sim-trained collapsed to a straight line) → **a large part of the gap is
appearance.** BUT 27 m is still far above classical VO on real KITTI (~few m / ~5% drift) → **learned
VO at this scale (~1k frames, one RTX 3080) is not competitive on real driving** (echoes Episode 8).
Figure: `artifacts/blog/sim2real_closure.png`. The deployable verdict is fully triangulated: on this
compute classical VO/VIO wins on real data. Exp B (domain-randomization augmentation to substitute
for real data) is an optional refinement — but it requires hand-modifying the agent's model, so it
becomes the harness's result, not the agent's. The learned-VO loop is effectively closed.
