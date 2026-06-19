# A Sixth Domain in 3D: Camera→Occupancy, and a Variance Finding that Replicates

*Project report · 2026-06-19 · LenaLab*

---

## 1. Why this exists

The five prior domains were 2D (VO/SLAM trajectories, BEV's flat grid). **3D semantic occupancy** is
the current autonomous-driving perception frontier (Occ3D, TPVFormer, SurroundOcc) — and a genuinely
harder test: predict, for a **voxel grid** around the ego car, which cells are occupied. Two
questions: (a) does the implement→verify→improve loop work in **3D**, and (b) does the BEV finding —
*the agent's free-form variance is its design latitude, and a scaffold collapses it* — **replicate**
on a harder task, or was it BEV-specific?

## 2. The task

Same input as BEV (6 surround cameras + intrinsics + cam→ego extrinsics, nuScenes mini), but the
output gains **Z**: a **200×200×12** voxel grid (ego frame, x,y ∈ [-50,50] m, z ∈ [-2,4] m @ 0.5 m),
scored by **per-voxel IoU**. The Z range was set empirically in the de-risk (occupied voxels land in
z ∈ [-1.75, 3.25] m). Vehicle voxels are **sparse (~0.3 %** of the grid), so occupancy IoU is lower
in absolute terms than 2D BEV — expected, not a defect.

## 3. Harness-owned GT

`scripts/prep_nuscenes_occ.py` fills each vehicle's oriented **3D box extent** into the ego voxel
grid (each voxel centre tested inside the box via its rotation + half-sizes). Held-out = official
nuScenes `mini_val`. 323 train / 81 val. Validated in the de-risk by eye (height-coloured top-down +
Z-slices show coherent vehicle volumes, `artifacts/occ/occ_gt_check.png`) and by learnability.
*Honest caveat:* this is **box-derived vehicle occupancy**, not dense semantic Occ3D (road/17
classes) — self-contained and harness-owned, the deliberate simplification for the small-data regime.

## 4. Reference — Lift-Splat-**to-3D**

`scripts/occ_lss.py` / `vo_lab/plugins/vo_ref/run_occ_learned.py`: ResNet-18 → per-pixel (depth
distribution × context) → lift to a camera frustum of 3-D points → **voxel-pool into X×Y×Z** → a
**3-D-conv head** → per-voxel logits. The only structural change from BEV is pooling on (x,y,z) and a
3D head. De-risk confirmed it fits 16 GB trivially (**0.57 GB** at this grid).

**From-scratch reference, n=3 seeds: 0.096 / 0.103 / 0.099 → 0.099 ± 0.003** — a stable, learned 3D
baseline (the camera→voxel mapping genuinely works), same tight reference behaviour as every prior
domain.

## 5. Agent implements (free-form), n=3 — and the BEV variance replicates

A sandboxed agent authored a camera→3D-occupancy net from scratch, three times (bar 0.0585 from the
harness-graded reference):

| run | held-out voxel IoU | verdict |
|---|---|---|
| 1 | 0.0543 | ❌ REJECTED |
| 2 | 0.1127 | ✅ VERIFIED |
| 3 | 0.0921 | ✅ VERIFIED |
| **n=3** | **0.086 ± 0.034** | **2/3** |

The agent *can* author real 3D occupancy (run 2 at 0.113 beats the reference), but **not reliably** —
one run self-sabotaged. The free-form variance (std 0.024) is **~8× the reference's** (0.003). This is
the **same signature as BEV**, now on a harder 3D task: the finding is not BEV-specific.

### 5a. What "the agent trains itself" actually looks like

The agent doesn't just emit a model — it **authors a training pipeline and improves a network with it**.
Run 2's agent wrote a **two-stage curriculum**: 40 epochs on 90% of the data, validating on the held-out
10% each checkpoint (keeping the best), then a threshold sweep, then 8 fine-tune epochs on 100%. Its own
training trajectory (`artifacts/occ/agent_training_curve.png`):

![Agent training: loss falls as its validation mIoU climbs, then fine-tune](../artifacts/occ/agent_training_curve.png)

This is implement→**train**→self-verify→improve in one session: loss 1.78→0.64, the agent's val mIoU
climbing 0.05→0.33 (★ = each new best it kept). Two honest notes: (1) the agent's *own* validation
(0.335) is far rosier than the harness's **held-out grade (0.092)** — precisely why the independent
verifier exists; self-assessment overstates, the held-out keeps it honest. (2) In the **scaffold** runs
(§6) the harness owns this training loop and the agent authors only the network — so "the agent trains
itself" is fully true in *free-form*, and deliberately scoped in the *scaffold* (that's the point of the
scaffold: take away the freedom that drives variance).

## 6. Improve via scaffold (evidence-justified), n=3

Because the variance replicated, the scaffold fix is justified by evidence, not assumed. The locked
core (`vo_lab/plugins/vo_ref/occ_scaffold.py`, seeded as `occ_core.py`) owns the 3D geometry, the
correct surround flip augmentation, training, and calibration; the agent authors **only** `model.py`
(`build_encoder` + `build_occ_head`). Calibration: scaffold reference 0.0664 → bar 0.051, trivial net
0.008 REJECTED, gate OPEN.

| condition | n | held-out voxel IoU | mean ± std | pass |
|---|---|---|---|---|
| reference | 3 | 0.096 / 0.103 / 0.099 | 0.099 ± 0.003 | — |
| agent free-form | 3 | 0.054 / 0.113 / 0.092 | 0.086 ± **0.024** | 2/3 |
| agent **scaffold** | 3 | 0.076 / 0.084 / 0.076 | **0.079 ± 0.004** | **3/3** |

**The variance collapses again** — at a clean n=3, the scaffold runs cluster at std **0.004** vs
free-form's **0.024** (**~6× tighter**), all three clear the bar, and each left the locked
`occ_core.py` byte-for-byte unmodified (diff-verified). So the BEV finding **replicates on 3D**: the
agent's free-form variance is its design latitude over the fragile geometry/augmentation, and locking
those makes it reliable. (Scaffold run 1 was lost to the §7 hang; the n=3 here is two original runs +
a hang-protected redo after the job-timeout fix.)

**An honest nuance (where occupancy differs from BEV).** On BEV the scaffold also *lifted* the mean to
reference quality. Here it makes results **reliable but not higher** than free-form's best run (0.113):
the locked core's fixed capacity (24 epochs, C=32) is a ceiling the constrained agent can't exceed.
So the scaffold's value on 3D is **reliability**, traded against the peak an unconstrained agent
occasionally reaches. Reported as-is — the cycle is honest about what the fix does and doesn't buy.

## 7. An infrastructure failure, surfaced honestly

One scaffold run **hung mid-training (~3 h, stuck at epoch 12)** holding the GPU; the harness's lease
timeout doesn't kill hung *jobs*, so the dead container starved the rest. It was caught, killed, the
run abandoned, and a **container-age watchdog** added to auto-reap hangs. Recorded because the lab's
norm is to surface what broke, not hide it — and it's a real gap (job-level timeouts) worth fixing.

## 8. Honest scope

- nuScenes **mini** (10 scenes), **vehicle-class only**, box-derived occupancy (not semantic Occ3D),
  from-scratch backbones → absolute voxel IoU is modest and below full-Occ3D literature.
- The claim is **the harness generalizes to 3D and the variance finding replicates** — not a SOTA
  occupancy number.

## 9. Files

```
scripts/prep_nuscenes_occ.py             harness-owned nuScenes→3D-voxel adapter
scripts/occ_lss.py                       Lift-Splat-to-3D reference + variance trainer
scripts/occ_derisk_gt.py                 Phase-0 GT rasterizer + Z probe
scripts/viz_occ_pred.py                  occupancy pred-vs-GT viz (height-colored + TP/FN/FP)
vo_lab/plugins/occ_nuscenes.py           Track-B provider
vo_lab/plugins/vo_ref/eval_occ.py        harness-owned 3D voxel-IoU grader
vo_lab/plugins/vo_ref/run_occ_learned.py from-scratch reference main.py
vo_lab/plugins/vo_ref/occ_scaffold.py    LOCKED 3D scaffold core (+ occ_scaffold_model_ref.py)
vo_lab/agents/occ_implementer.py         free-form + scaffold task specs + authors
vo_lab/run_occ_{calibration,implement}.py        free-form
vo_lab/run_occ_scaffold_{calibration,implement}.py  scaffold
artifacts/occ/                           GT check, pred viz, comparison figure
```

**Verdict:** the implement→train→verify→improve loop works in 3D. A from-scratch reference is stable;
a free-form agent is capable-but-high-variance (replicating BEV); a scaffold collapses the variance
(with an honest reliability-vs-peak nuance). **Six domains now** — monocular VO, RGB-D VO, SLAM, KITTI
stereo, BEV, and 3D occupancy — and a cross-domain finding (agent-freedom is the variance source,
scaffolding scopes it) that now holds in **both** 2D and 3D.
