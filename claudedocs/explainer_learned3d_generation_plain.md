# Plain-Language Explainer: A Generator With a Real Sense of 3D Space

*Written to be readable without CV background. Companion to
`research_learned3d_generation_litgap_2026-06-23.md`.*

---

## 1. The problem we're trying to solve

**Goal in one sentence:** make an AI that can generate a scene, then show you that *same* scene from a
*different camera angle* — and have everything stay **consistent** (the chair is still the chair, the wall
is still where it was, nothing morphs or drifts).

**Why this is hard.** Today's image/video generators are, deep down, *re-drawing* the picture each time.
They don't actually "know" there's a 3D room — they know what rooms *tend to look like*. So when you say
"now show me the other side," they hallucinate a plausible other side that often **contradicts** what you
saw before. Walk around long enough and the world quietly mutates.

**The analogy.** Think about how *you* picture a room you've walked through. You don't store a flipbook of
flat snapshots — you build a **mental 3D model**. Once you have it, you can imagine any new angle and it
stays consistent, because every imagined view is just "look at my 3D model from here." We want the
generator to have that: a **memory of the 3D space** it can consult, instead of re-guessing every frame.

**The deeper version (the research bet).** The model shouldn't just *use* a 3D model someone hands it — it
should **build its 3D understanding itself, just from watching 2D video**, the way a person does. And when
it draws a new view, it should **"ask" that 3D memory** ("what's over here?") rather than mechanically
replaying it.

---

## 2. What Latent Spatial Memory ("Mirage") already solved

Mirage (Microsoft, June 2026) is the closest existing system, and it got the **big idea right**:

- ✅ **It gives the generator a 3D memory.** As it generates, it keeps a running **3D scratchpad** of the
  scene and reuses it for new camera angles — so the world stops drifting/contradicting itself.
- ✅ **It stores "meaning," not pixels.** Instead of saving raw colored dots, it saves compact *feature*
  codes at 3D locations (cheaper and smarter). Think of sticky notes pinned in 3D space that say "sofa-ish
  texture here," not a photograph.
- ✅ **It's efficient.** ~10× faster and ~55× less memory than keeping a giant colored point cloud.
- ✅ **It measurably reduces drift** — you can drive the camera in a loop and roughly return to where you
  started.

So Mirage proved: *a 3D latent memory makes generation far more camera-consistent.* That's the foundation.

---

## 3. What Mirage did **not** solve (our openings)

| Gap | In plain terms |
|---|---|
| **Geometry is borrowed, not learned** | Mirage uses an off-the-shelf "depth estimator" to decide where things are in 3D. It's using **someone else's eyes** and trusting them. |
| **It can't fix bad depth** | If those borrowed eyes are wrong, the mistake gets **baked into the memory and never corrected** — errors pile up. (Mirage's own paper admits this.) |
| **It "replays" the memory, doesn't "ask" it** | To make a new view, Mirage **re-projects** its 3D notes onto the new camera (a mechanical paste), rather than letting the network **attentively query** the memory ("what belongs here, given everything I know?"). |
| **Static scenes only** | Moving things (people, cars) are *deleted* from the memory because their geometry is unreliable. |
| **Indoor tours only** | Trained on real-estate walkthrough videos — not driving, not the open world. |

**Our three targets** (from that list): (1) let the model **learn its own geometry** from the video
itself; (2) make it **fix its own mistakes** (geometry gets corrected because it has to explain all the
views); (3) **query the memory with attention** instead of mechanical replay.

---

## 4. Our model: what goes in, what comes out

**Input (what you give it):**
- A **starting view** — one image or a short video clip of a scene.
- A **target camera path** — where you want the camera to go next (e.g., "pan right," "orbit around,"
  "drive forward"), given as camera positions/angles.

**Inside (what it builds — the new part):**
- A **3D memory** of the scene, built **by the model itself** from the input video (not from borrowed
  depth). The model figures out the 3D by a simple rule: *whatever 3D it imagines must correctly explain
  every 2D frame it has seen* — if it doesn't, it adjusts. (This self-checking is called
  "analysis-by-synthesis": render your guess back to 2D, compare to reality, fix the difference.)
- When generating the next view, the model **queries that 3D memory with attention** — it asks "for this
  pixel/region from this new angle, what does my 3D memory say should be here?" — and uses the answer to
  draw a consistent frame.

**Output (what you get back):**
- A **video from the new camera angle(s)** that stays **consistent** with what you already saw.
- *(Bonus)* the **3D structure itself** (depth / a 3D map), as a by-product — usable for other tasks.

**One-line contrast with Mirage:** *Mirage = borrowed 3D + mechanical replay. Ours = self-learned 3D
(that can correct itself) + attention-based querying.* The headline claim we'd test: **ours stays
consistent even when depth is hard/wrong, where Mirage degrades.**

---

## 5. How we'll know it works (the test)

We don't grade the 3D directly (you can't easily "label" 3D). Instead:
- **Render the model's idea back to 2D and check it matches the real frames** (color + motion).
- **Score geometric consistency** with an automatic metric (does the same point stay put across angles?).
- **Use driving data as a fair referee:** driving footage comes with ground-truth 3D (LiDAR, GPS). We
  **hide that during training** (so the model must *learn* geometry) and **only use it to grade** at the
  end. Abundant truth becomes our *referee, not a crutch*.

---

## 6. What it's good for (applications)

- **Self-driving simulation / world models.** Generate realistic, *camera-consistent* driving scenes and
  "what happens next" from any viewpoint — for testing and training autonomy stacks cheaply. (This is also
  the user's career direction, and the most natural home for the idea.)
- **Controllable video generation.** "Same scene, move the camera here" that actually holds together — for
  film/previz, content tools, and any app needing camera control without the world melting.
- **Robotics / embodied AI.** A robot that can *imagine* unseen viewpoints of its surroundings consistently
  is better at planning and navigating.
- **3D-from-video, for free.** Because the model builds real 3D internally, you get usable 3D structure out
  of ordinary video — useful for mapping, AR/VR, and reconstruction.

---

### TL;DR
Generators today "re-guess" every frame, so worlds drift when the camera moves. The fix is to give the
generator a **3D memory**. Mirage proved that works — but it **borrows its 3D, can't fix 3D mistakes, and
replays the memory mechanically**. Our bet: a generator that **learns its own 3D from video, corrects it by
having to explain every view, and queries it with attention** — tested with driving data's ground-truth 3D
as a hidden referee. If it holds consistency where Mirage breaks, that's the contribution.
