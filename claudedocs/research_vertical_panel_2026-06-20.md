# Where can we use this? — a vertical-panel "look into their life" study

*Strategy report · 2026-06-20 · LenaLab · method: 6 parallel domain-expert agents (Jobs framing:
"people don't know what they want; look into their life"), then synthesis. No web search; grounded in
lived domain knowledge. Companion to `research_creative_directions_2026-06-20.md`.*

**The capability examined:** turn a handful of ordinary cameras (no LiDAR) watching a space into a live
**top-down "what's where" map** — which areas are occupied and by what — that's **cheap (cameras-only)**
and can be **built + honestly self-verified for a brand-new space fast** (the lab generates its own GT and
grades itself on unseen data). This is exactly the multi-cam → BEV/occupancy machinery we built.

---

## The two breakthrough insights (each surfaced independently by multiple experts)

### 1. "A map, not a camera" — the privacy limitation *is* the feature
*(Eldercare and Home experts both ranked this #1, independently.)*

The rooms that most need watching are exactly the rooms a camera is **forbidden** — the bathroom, the
bedroom, a child's play area, an aging parent's home. Every existing solution either **surrenders** those
rooms (hallway motion sensors only) or **violates dignity** (actual video that gets unplugged). Top-down
occupancy dissolves the tradeoff: it perceives a *fall* as **geometry** (a person-shape gone horizontal,
no transition for 18 min) while being **constitutionally incapable** of producing a recognizable image of
a vulnerable person. It doesn't make surveillance more acceptable — it makes the thing *not surveillance*.
That reframing turns a market that says "absolutely not in here" into one that says **"I didn't know I was
allowed to want this."**

### 2. The gap between what a system *believes* and what's *physically there*
*(Logistics, Retail, and Hospitality experts all converged here.)*

Every space already runs on a model that's quietly, **daily wrong** because nothing can actually *see* the
space from above: the WMS says the lane is empty (it's jammed); the host thinks Table 12 is occupied (it
emptied 6 min ago); the shopper thinks the line is long (it's 4 people chatting). People have normalized
**walking over to look with their own eyes.** A cameras-only top-down map is the first cheap source of
**physical ground truth** that reconciles belief with reality — recovering invisible losses (the customer
who turned away because a clump *looked* like a wait; the table-turn never made; the detention fee).

---

## The hidden through-line: our "edge" is the deployment unlock, not a footnote

In **every single vertical**, the experts independently flagged the same blocker: *every space is unique*
(every home, barn, DC, venue, sanctuary), there is **no off-the-shelf labeled data**, and nobody trusts a
safety/ops decision on a vendor's promise. The thing that makes any of these products real is a system
that **stands itself up on a brand-new space and honestly proves its own accuracy on unseen footage.**

That is *precisely* LenaLab's core competency — harness-owned GT generation + held-out grading +
verification-first honesty. So our methodological edge isn't a portfolio nicety; **it's the exact
capability this entire product class is bottlenecked on.** "Self-verify on a new space" + "cameras-only"
together are what let this reach the millions of one-off spaces no sensor vendor will ever custom-install.

---

## Ranked shortlist (latent-demand × who-pays × our-tech-fit × data-available-now)

| Opportunity | Latent demand | Who pays | Tech fit | Data *now* | Verdict |
|---|---|---|---|---|---|
| **A. Eldercare "rhythm of the home" / fall-as-geometry** | ★★★★★ | adult children, monthly | ★★★★ | ★★ (create/synthesize) | **biggest creative bet** |
| **B. Warehouse/smart-space ops** (near-miss ledger, belief-vs-reality, trust-on-day-1) | ★★★★ | EHS/ops, clear ROI | ★★★★★ | ★★★★★ (NVIDIA public set) | **buildable-now flagship** |
| C. Restaurant/venue "phantom table" + crowd "pressure gauge" | ★★★★ | owners, venue safety | ★★★★ | ★★ | strong, data-harder |
| D. Retail "is the line actually long" walk-away recovery | ★★★★ | shop owners | ★★★★ | ★★ | great story |
| E. Livestock night-calving / drone paddock map | ★★★★ ("sells back sleep") | ranchers | ★★★ (single overhead) | ★★ | high-meaning, niche |

**Convergence worth noting:** opportunity **B is the same target the web-research report independently
recommended** (NVIDIA Physical AI Smart Spaces dataset — public, multi-cam, occupancy-ready). Two
different methods (market scan vs. lived-life panel) pointing at the same buildable thing is a strong
signal: **it's the one we can prove *now*, on real public data, fast, and it's adjacent to AD perception**
(embodied/"physical AI" is hot and job-relevant).

---

## Recommendation (for your decision — research only, nothing implemented)

**Frame the work as a *horizontal* capability, demonstrate it on the buildable wedge, narrate it with the
visionary one:**

- **The capability (the real product):** *"a self-installing, self-verifying top-down map of any space,
  from cameras you already have — a map, not a camera."*
- **Build/prove it on B (smart-space ops)** — public NVIDIA dataset, exact reuse of our lift-splat +
  GT-generation + verification harness + the proven cloud pipeline. This is the credible, fast,
  job-relevant flagship.
- **Headline it with A (eldercare "a map not a camera")** — the Jobs-grade *why it matters in life*: huge
  latent demand, privacy-as-feature, dignity. If you want a true "dataset nobody trained on / create a new
  dataset" play, A is where our GT-generation edge becomes a *new verified benchmark* (a real contribution,
  not a leaderboard also-ran).

**The one-line pitch this exercise earns:** *Don't build a better self-driving-car model. Build the thing
that turns any ordinary cameras into a private, self-verifying map of a space — and prove it where it's
both buildable today (warehouses/robots) and most needed in life (watching the rooms you're not allowed to
film).*

**Decision points (you pick):**
1. **B as the next build** → `/sc:design` a "smart-space occupancy" domain on the NVIDIA dataset (de-risk:
   pull a slice, confirm our lift-splat + GT-gen transfer).
2. **A as a create-a-benchmark research bet** → design a privacy-first top-down "home-rhythm" benchmark
   (synthetic or self-generated GT) — higher novelty, more data work.
3. **Both** → B proves the engine, A is the vision it serves.

*Confidence: high that the two insights are real and cross-cutting (independent multi-expert convergence);
high that B is buildable now; medium on A's data path and market sizing. Report only — no code changes.*
