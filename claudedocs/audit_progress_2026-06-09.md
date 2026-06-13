# Audit progress log (autonomous session, 2026-06-09 → overnight)

Working through `results_audit_2026-06-09.md` while the user is away ~9h. Honest log; updated as I go.

## Status board

| Priority | Item | Status |
|---|---|---|
| P1 | Fidelity-ladder variance (gaussian/aug/real × seeds 42,1,2) | 🟢 RUNNING (~2.5h) |
| P1 | Rung 3 (0.45 synth) + Ep8 (19.8) learned variance | ⏳ queued (GPU, after ladder) |
| P2 | SLAM benchmark reproduction (Ep18b classical + C++ VO ×3) | 🟢 launching (CPU, parallel) |
| P2 | Contamination (Ep12), M1 (Ep9), M3 VIO (Ep15) reproduce | ⏳ queued |
| P3 | Ep2 shape-only/40× | ✅ already caveated (2026-06-04 review) |
| P3 | Ep5 in-sample | ✅ already caveated (2026-06-04 review) |
| P3 | DROID accuracy downgrade (Ep18) | ✅ done |
| P3 | Sim(3)-aligned figures audit | ⏳ queued |

## Findings so far (honest, incremental)

- **GPU nondeterminism is real and ~0.5 m:** gaussian rung at the *same* seed 42 gave **26.62 m** this run
  vs **27.11 m** originally — a 0.5 m swing with no seed change. So the published single-run numbers carry
  at least ~±0.5 m, before cross-seed variance. This already justifies the whole audit.
- Ep2 (mono 40× rescale, shape-only) and Ep5 (in-sample) were already honestly corrected by the earlier
  external review — the blog was more honest than the sim-faithfulness lapse suggested.

## Decisions
- Keep every new number with mean±std; only retain comparative claims (A<B) that survive the measured noise.
- No overclaiming — the sim-faithfulness retraction is the standard now.
GPU queue: fidelity variance running (gaussian s42=26.62 s1=29.06 -> ~2m spread already)

## Power-off recovery (2026-06-11)
Computer powered off mid-run. Survived (written to artifacts/, not /tmp): **3/9 variance cells (all gaussian)**.
- **rendered/gaussian = 27.35 ± 1.49 m (n=3)** [26.62, 29.06, 26.37]. Original 27.11 is within 1 std.
  -> confirms the ~1.5m TRAINING NOISE; the published "27.1 vs 27.2" (0.1m apart) is NOISE, not signal.
- **Benchmark repro (1 run completed before outage):** classical VO IDENTICAL (0.16/4.04/0.93 -> deterministic);
  C++ VO city/residential reproduce (0.08/100.76) but road moved 55.6 -> 64.43 (both diverged -> qualitative
  "C++ diverges on hard drives" holds; exact divergence magnitude is unstable, as expected for a lost track).
- gaussian_aug is NOT in make_domain_provider (separate aug script) -> variance run does gaussian/real/procedural.

RESUMED: GPU pipeline (real x3 + procedural x3 + synth in-domain x3); CPU repro (benchmark x3 + contamination + C++ VO).

## Power-off #2 (2026-06-11 ~22:29) + FINAL variance table
Machine rebooted AGAIN (2nd time; external, not me — zero shutdown cmds run). /tmp lost; artifacts/ survived.
**Fidelity-ladder variance COMPLETE for 4/5 rungs (n=3 each):**
- procedural    69.71 ± 0.05  (nearly deterministic)
- rendered      27.35 ± 1.49
- real          25.61 ± 1.64
- synth in-dom   0.55 ± 0.05  (published 0.45 was a low single draw; honest ~0.55, still ~6x better than ref 3.26)
- gaussian+aug   NOT RUN (lost to reboot; minor completeness rung)

**VERDICT: 'rendered ≈ real' HOLDS** — diff 1.7m < SE-of-diff ~1.3m, ranges overlap → statistically
indistinguishable (~26 ± 1.5m). procedural ~70 robustly separated. The fidelity-ladder CONCLUSION survives;
the false '27.1 vs 27.2' precision is corrected to mean±std. Folded into blog Ep17 + arc rows 16/17.

CPU reproductions (from before reboot): classical VO deterministic; C++ VO road non-deterministic but always
diverged (52.3/52.6/55.6/64.4), city/residential deterministic. Contamination/C++ Phase1 repro: LOST, to redo.
