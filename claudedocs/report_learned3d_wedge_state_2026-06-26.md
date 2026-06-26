# Research State Report — Learned-3D-Store + QKV ("the wedge")

*2026-06-26 · visual summary of where the project stands: the idea, the architecture, the result, next steps.*

---

## 1. What we're building (in one picture)

**Goal:** an image generator that can take *one image + a target camera* and produce the *same scene from a
new angle, consistently* — by keeping an internal **3D memory** it can *query*. The open research bet: make
that 3D memory **learned from the generation loss** (not handed in by an off-the-shelf depth model) and
**queried by attention**. That exact combination was *empty* in the literature — it's the contribution.

We test it as a **minimal fork of Captain Safari** (an open video model that already has a QKV-queried 3D
memory): swap its *frozen, pre-baked* memory for a *trainable, in-graph* one and see if it helps.

![architecture](../artifacts/learned3d_wedge/figs/fig1_architecture.png)

**How to read it:** the **grey** path is the released baseline — geometry is *handed in* (StreamVGGT,
pre-baked). The **green** path is our wedge — a **trainable store** that the generation loss shapes
end-to-end (the dashed green arrow is the gradient reaching it). Everything else (the QKV retriever, the DiT)
is reused. The whole experiment is: *does the green store beat the grey one?*

---

## 2. The result so far (the go/no-go)

We got the real model running on real data (VAE/T5-encoded clip) and ran a **held-out** test: train the
store on some noise/timestep samples, then measure loss on **disjoint samples it never saw**, vs the frozen
store.

![results](../artifacts/learned3d_wedge/figs/fig2_results.png)

- **Left:** the learned store's loss drops on **both** train *and* held-out — so it **generalizes** (it
  isn't just memorizing). It crosses below the frozen baselines (dashed lines).
- **Right:** on the 5 held-out samples, **5/5 improved**; mean **0.469 → 0.456 (+2.7%)**.

**Verdict: a real but *weak* GO.** The learned store genuinely beats the frozen one on real, held-out data —
consistent and generalizing — **but the margin is modest (~2.7%)**, and the held-out axis is **noise/timestep
only** (the demo clip has a single camera/query frame, so the headline claim — helping across *viewpoints* —
is **not yet tested**).

**Honesty notes (what this is NOT):** not a large win; not on our target (stylized) domain; not yet a
cross-*viewpoint* result. It validates the *mechanism on real data*, not (yet) the full thesis.

---

## 2b. The decisive result: cross-VIEWPOINT held-out (STRONG GO)

The +2.7% above was the *weak* axis (noise/timestep, 1 query frame). The test that actually matters — **does
the learned store beat frozen across *unseen camera viewpoints*?** — is now done, on real VAE-encoded frames
along the clip's real (moving) camera path: train the store on one set of viewpoints, test on **disjoint
held-out viewpoints**.

![cross-viewpoint results](../artifacts/learned3d_wedge/figs/fig4_crossview.png)

- **Held-out viewpoints: 5/5 improved**, mean **0.664 → 0.555 = +16.4%** (≈6× the noise/timestep gain).
- **Largest gains on the *genuinely-new* viewpoints** (frames 16, 19, *outside* the memory's key window):
  **+20.5%** — so it's real **viewpoint generalization**, not just adapting to seen poses.
- **Held-out gain (+16.4%) > train gain (+14.3%)** and the held-out curve falls monotonically → **generalizes,
  doesn't overfit.**

**Verdict: STRONG GO** — a *learned, queryable* 3D store clearly beats the *frozen, handed-in* one across
unseen camera viewpoints on real data. This is the result the whole direction hinged on.

**Honesty caveats:** it's **one scene/clip**, viewpoints along **one camera trajectory** (a dolly path, not
arbitrary orbits), and **within-scene** (cross-*scene* generalization untested). It's a strong signal on the
right variable, not yet a multi-scene or stylized-domain claim.

## 2c. Generation demo — the visual reality check (important caveat)

We finally ran *actual generation* (full 30-step sampler + VAE-decode), not loss, at two held-out
viewpoints — **real target | frozen-store generation | learned-store generation**:

![view 16](../artifacts/learned3d_wedge/gen_demo/view16_sidebyside.png)
![view 19](../artifacts/learned3d_wedge/gen_demo/view19_sidebyside.png)

**What the pixels actually show (honest):**
- The model generates **coherent, plausible coastal scenes** — but **neither** generation reproduces the
  real target's **camera viewpoint**: the REAL frames are near **top-down cove** shots; both generations
  render an **oblique coastline**. The shared memory/context dominates; the held-out query pose only weakly
  steers the framing.
- **Frozen vs learned is subtle** — the learned one is marginally crisper (cliff structure, water/shore
  edge), frozen slightly hazier, but it is **not** a night-and-day difference. It matches the *modest* loss
  gain, not a dramatic one.

**This tempers §2b's "strong go."** The +16–20% is real **in denoising loss**, but **in pixels it does not
yet translate to a clearly-better *or* viewpoint-faithful prediction.** The honest synthesis: the learned
store is *measurably* better, but **denoising loss was an optimistic proxy** — the actual generation quality
at far held-out viewpoints is the real bottleneck.
- *Caveats on the demo itself:* CFG was off and the sampler simplified, so this isn't the model's best
  possible output; but the frozen-vs-learned comparison is apples-to-apples, so "subtle visual difference"
  holds. Far, outside-window viewpoints are the hardest case (absolute loss stays high ~0.55–0.74).

**Revised verdict: a real *quantitative* signal (learned > frozen, generalizing), but *not yet* a visually
convincing one.** Worth continuing — but the next phase must optimize/measure **actual generation quality**,
not just denoising loss.

## 3. How we got here, and what's next

![roadmap](../artifacts/learned3d_wedge/figs/fig3_roadmap.png)

**Phase 0 (feasibility) — COMPLETE.** Design → confirmed the architecture is swappable (coupling gate) →
proved gradients reach a trainable store (gradient-flow) → ran the real 5B model on real data → found & fixed
a backward-pass bug (by decoupling encoding from training) → got the go/no-go number (+2.7%, weak go).

**~~Next step: cross-viewpoint test~~ — DONE ✅ (STRONG GO, §2b).** The gain extends to unseen camera
viewpoints (+16.4%, largest on genuinely-new views) — the decisive question came back positive.

**Next step (now): multi-*scene* validation.** The strong result is on **one scene**. The cheap next check
is to repeat the held-out-viewpoint test on **2–3 more Captain-Safari clips** (different scenes) → confirm
the gain isn't scene-specific before the big build. (Needs pulling a few more clips — small.)

**Then (big): the toon-multiview pipeline** — our own stylized-but-geometric data (rendered toon-3D from
many known cameras), train the wedge on **our domain** + a fuller architecture (a real trainable
store-builder from frames, not just a trainable copy of the released memory).

**Goal: a stylized "world model"** — feed it a drawing + a camera, get a consistent new view, and accumulate
an explorable world.

---

## 4. Practical state
- **Everything reproducible:** scripts, results, env-lock, and these figures in `artifacts/learned3d_wedge/`.
- **Pod stopped** (resumable; `/workspace` has the 10 GB weights + encoded inputs staged).
- **Cost lessons baked in:** download from HuggingFace not ModelScope (40× faster); the resume wipes the pip
  env (only `/workspace` persists) → restore from the saved lock; decouple encode from train to keep the
  autograd graph clean.

**One-line status:** *learned > frozen is real and generalizes **in denoising loss** (+16% across unseen
viewpoints), but the **generation demo shows the win is subtle and not yet viewpoint-faithful in pixels** —
a measurable signal, not yet a visually convincing one. Continue, but next must measure **actual generation
quality** (not just loss), on the real architecture + multi-scene + stylized domain.*
