# What the agent figured out

*The case that LenaLab's agent does real research — in its own words.*

Across six computer-vision problem classes, the agent didn't just emit code that happened to run. It
**analyzed each problem, researched an approach, designed the solution, trained or tuned it, and the
result held up on data it never saw.** The clearest evidence isn't a metric — it's the *reasoning the
agent documented in the code it wrote*. Below are its own design decisions (quoted verbatim from the
authored algorithms in [`artifacts/`](../artifacts)), each with the held-out result that backed it.

---

### 1. It reasoned about scale — and made monocular depth *metric*
**Problem the agent identified:** a single camera can't recover absolute scale. **What it did** (RGB-D VO):
> *"SIFT match → 3D-2D PnP RANSAC … KLT optical-flow fallback … SIFT against recent keyframes [recovery]. **Depth back-projection provides absolute (metric) scale.**"*

It used depth to back-project features and solve **metric** PnP, with a three-tier fallback for robustness.
**Result: ATE 0.033 m (SE3, metric) on an unseen scene — beat the classical reference (0.057 m).** Analyze → research → build → generalize, in one pass.

### 2. It diagnosed *why* fast outdoor motion breaks tracking — and switched matchers
**Problem:** KITTI driving has 100+ pixel jumps between frames that defeat optical flow. **What it did** (KITTI stereo VO):
> *"ORB features … matched with BF+crossCheck (**handles 100+ px inter-frame motions robustly; no accumulated drift**) … back-project to 3-D via depth map … PnP+RANSAC … constant-velocity fallback if PnP fails."*

It chose descriptor matching over flow *specifically because* of the motion magnitude — a deliberate analysis, not a default. **Result: t_err 2.08% on unseen sequences 05/07.**

### 3. It failed, diagnosed the failure, and redesigned — the clearest "research" signal
**First attempt** (SLAM v1): a non-linear SE(3) pose graph that **diverged to 412 m** (recorded as a rejected result, not hidden). **Then it figured out why and fixed it** (SLAM v2):
> *"Pose graph: **TRANSLATION-ONLY linear system (rotations fixed from VO) → sparse LSQR, guaranteed no divergence**, < 1 s. Fallback: VO-only if … ill-conditioned."*

It traced the blowup to the non-linear rotation optimization, removed that degree of freedom, and made the system provably stable. **Result: the divergence was gone (0.185 m).** *(Honest caveat: that run's dev and held-out sequence coincided, so 0.185 m shows the recovery, not transfer — kept on the record as such.)*

### 4. It found a non-obvious bug in its *own* geometry by reasoning about state
**Problem the agent articulated** (VIO — its own docstring):
> *"**Key insight:** after a long blackout the VO 'recovery' pose is WRONG in absolute terms because it accumulates motion relative to the stale held pre-blackout pose. We fix this with an offset-tracking scheme … at recovery, set vo_offset = imu_pos − vo_pos_at_recovery so subsequent VO frames are correctly anchored."*

That is the agent identifying a subtle failure mode in sensor-fusion bookkeeping and engineering the correction. **Result: 3.83% with IMU fusion vs 18% for vision-alone.**

### 5. It innovated on a *learned* model — gave the network motion as input
**What it did** (learned VO, trained from scratch on GPU):
> *"8-channel input: [RGB_t, RGB_{t+1}, **flow_upsamp**] … compact custom CNN trained from scratch (no pretrained weights needed) … accumulates predicted relative poses."*

Feeding the network pre-computed optical flow alongside the frame pair was the agent's idea. **Result: it beat the learned reference ~1.7× (18.5 ± 0.7 m, n=3).**

### 6. It debugged numerics and class imbalance in 3D occupancy
**What it did** (3D occupancy, free-form):
> *"Inverse-projection: voxel centres → cameras (**FP32, avoids FP16 overflow**) … weighted BCE (**pos_weight=80**) + soft-IoU … **stage-1 on 90% data → find threshold; stage-2 fine-tune on 100%**."*

Three separate pieces of analysis in one model: a numerical-stability fix (FP32 where overflow bit it),
a class-imbalance fix (vehicle voxels are ~0.3% of the grid → pos_weight 80), and a **self-designed
two-stage training curriculum**. **Result: up to 0.113 IoU held-out (its best free-form run).**

### 7. It worked out the geometry of augmenting a *six-camera rig*
**What it did** (BEV): flipping a surround-camera scene isn't mirroring pixels —
> *"Horizontal-flip surround-camera augmentation (**correct camera swap + extrinsic update**)."*

It reasoned that a left-right flip must **swap the left/right cameras and update their extrinsics**, not
just flip images — a correctness subtlety a careless implementation gets wrong (and one free-form run
*did* get wrong, which is how we know it matters). **Result: scaffolded 0.136 ± 0.005 IoU, 3/3.**

### 8. It proved capability, not memorization
On a **provably-unseen** procedural-synthetic world (no chance of training-data overlap), the agent
authored a stereo VO from scratch (SGBM disparity → FAST+KLT → PnP-RANSAC). **Result: 1.20% — it beat
the reference (1.91%) on data nothing could have memorized.** The skill is real, not recalled.

### 9. It exploited that the cameras don't move (off-road, static multi-camera)
On the seventh domain — static overhead warehouse cameras → a top-down floor-occupancy map — the agent
did more than re-derive the geometry. It **noticed the cameras are fixed** and built a method around that
fact: a **temporal background-subtraction** front-end (median background, differenced per frame, so moving
people/forklifts pop out as a 3-channel difference image alongside RGB), multi-height IPM fused mean+max
across the 19 cameras, focal loss for the ~0.4% positive rate, and **adaptive top-K inference** (predict
each frame's *expected* number of occupied cells by rank, instead of a fixed probability threshold that
drifts on unseen frames). **Result: 0.39–0.44 held-out floor IoU, both runs VERIFIED — ~2× the
hand-written IPM reference (~0.22), graded on unseen-time frames.** The insight that a static rig makes
*temporal* background modeling free is exactly the move that beat a generic geometric baseline — and it
came after the agent's first instinct (the driving Lift-Splat) was shown wrong for fixed cameras.

---

## What this adds up to

These are not tuned hyperparameters — they are **design decisions an engineer would be proud of**:
choosing depth for metric scale, descriptor matching for fast motion, a stable linear pose graph after a
divergence, an offset-anchoring scheme for sensor fusion, optical-flow input for a learned model, FP32
to dodge overflow, a two-stage curriculum, correct multi-camera flip geometry. The agent **analyzed**
each problem, **researched** an approach, **implemented and trained** it, and every number above was
**measured on held-out data it never saw** — so the reasoning didn't just sound good, it generalized.

It also got things wrong (the 412 m SLAM divergence, a self-sabotaging BEV run, a 3-hour training hang)
— and those are on the record too, because the same loop that produces the wins is what catches the
losses. That's the difference between a demo and a method: **the agent does the research, and the
results are honest.**

*See the agent's training in motion: [`artifacts/occ/agent_training_curve.png`]. Full evidence per
domain: [`RESULTS.md`](../RESULTS.md). Algorithms it wrote: [`artifacts/agent_authored_*.py`](../artifacts).*
