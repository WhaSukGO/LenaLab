# How LenaLab works

A guide to the architecture: what runs, in what order, and **exactly where the AI is (and
isn't) involved**.

---

## 1. The one idea

LenaLab is split into two halves that never trust each other:

- a **solver** (LLM "expert" agents) that *produces* a solution to a vision problem, and
- a **verifier** (deterministic, no LLM) that *decides whether to believe it* by measuring it
  on a **held-out** split the solver never saw and cannot edit.

A result is "done" **only** when the verifier signs it. "It ran" is never success.

```
            produces a solution                      decides whether to trust it
   ┌────────────────────────────┐            ┌────────────────────────────────────┐
   │  SOLVER  (AI lives here)    │  artifact  │  VERIFIER  (no AI — pure Python)     │
   │  • committee proposes, or   │ ─────────► │  run in a sandbox, then measure on   │
   │  • agent authors code        │  (code /   │  a HELD-OUT split with a grader the  │
   │                              │  trajectory)│  solver can't see  →  VERIFIED / FAIL │
   └────────────────────────────┘            └────────────────────────────────────┘
```

The verifier is the constant; the solver is swappable. This is the generator⟂evaluator
separation from Anthropic's harness-design work — the single most important lever.

---

## 2. Two packages: solver on top of the verifier spine

```
LenaLab/  (this repo)                          blueberry_ver2 / "Touchstone"  (imported)
  vo_lab/                                        lab/
   ├ plugins/                                     ├ loop.py        Harness state machine
   │  ├ vo.py        monocular VO domain          ├ evaluator.py   ScriptEvaluator (held-out)
   │  ├ vo_rgbd.py   RGB-D + generalization        ├ registry.py    crash-resumable SQLite
   │  └ vo_ref/      run.py / eval*.py (grader)    ├ budget.py      tokens + experiments
   ├ agents/                                       ├ gpu_lease.py   single-GPU mutex
   │  ├ vo_committee.py   Track A (meeting)         ├ image_registry CUDA image matrix
   │  └ vo_implementer.py Track B (authoring)       ├ dataset_cache  download-once cache
   ├ memory.py     cross-run failure ledger         ├ job_runner.py  Docker/local job exec
   ├ factory.py     wires a Harness                 ├ notebook.py    per-run lab notebook
   └ run_*.py       entry points                    └ models.py      ExperimentContract, …
```

LenaLab **reimplements none of the verifier** — it imports `lab` and adds only the vision
domain (data, reference code, the grader, expert prompts). The harness depends only on three
Protocols (`Planner`, `Evaluator`, `DatasetProvider`), so a "domain" is just a plugin.

---

## 3. The lifecycle of one experiment

`lab/loop.py::Harness.run_experiment` drives this state machine. Reasoning steps (the AI)
are marked **★**; everything else is deterministic Python.

```
  PROPOSED
     │  ★ planner.propose_contract()        ← AI: committee meeting OR agent authors code
     ▼
  CONTRACTED        the agreed, gradable definition of "done" (metric, datasets, commands)
     │  image_registry.resolve()            deterministic: pick a prebuilt CUDA image
     ▼
  ENV_READY
     │  dataset_cache.ensure()              deterministic: download ONCE, then reuse
     ▼                                       (held-out split is NOT mounted for the solver)
  DATA_READY
     │  gpu_lease + job_runner.run()        run the solution as a JOB (IO, not an AI turn)
     ▼
  ARTIFACTS_READY   the produced trajectory / checkpoint
     │  metric_extractor.extract()          records the solver's SELF-REPORT (kept, distrusted)
     ▼
  EVALUATING
     │  evaluator.evaluate()                ← independent, deterministic: measure on HELD-OUT
     ▼
  VERIFIED ✔  /  REJECTED ✘                 verdict from the held-out measurement, signed
                                            (FAILED if any step errors — the loop survives)
```

Key consequence: **downloads, builds, training, and inference run inside the harness as
jobs — they consume wall-clock, not the model's turn/token budget.** That is the fix for the
original "ran out of turns on downloads" failure. Budget (`budget.py`) is counted in
**tokens + experiments**; IO is free.

---

## 4. Where the AI is involved (and where it is NOT)

```
  AI (LLM via claude-agent-sdk)                 NOT AI (deterministic Python)
  ─────────────────────────────                 ──────────────────────────────
  • Track A: PI drafts a proposal,              • picking the CUDA image
    experts review, PI decides next             • downloading / caching datasets
  • Track B: agent writes & debugs              • running the job (Docker/local)
    the algorithm in a sandbox                  • the GRADER: ATE / RPE / PnP / Sim3·SE3
                                                • the held-out split & the oracle bar
                                                • the calibration gate
                                                • budget, registry, GPU lease
```

The AI lives **only in the solver**. The verifier is LLM-free — grading is closed-form
geometry. This is deliberate: a model grading its own work inflates the result, so the thing
that says "VERIFIED" is code, not a model.

The AI dependency is also behind an **injectable seam**: the committee takes a `run_fn`, the
implementer takes an `author_fn`. Tests inject fakes, so the offline suite and calibrations
run with **no API key and no `claude` binary** at all.

### How the AI is actually called

- It uses the **`claude-agent-sdk`** Python package (which bundles its own `claude` binary)
  with `ANTHROPIC_API_KEY` — **not** the interactive Claude Code app. You launch a plain
  `python -m vo_lab.…` and the SDK spawns the model itself.
- Each call is a **fresh, isolated session** (`setting_sources=[]`, no shared context) — so
  the proposer and any reviewer can't contaminate each other.

---

## 5. The two solver tracks

### Track A — the committee "meeting" (`vo_committee.py`)
A menu-constrained panel. It can only **select a vetted recipe and set its declared
parameters** — it cannot invent commands.

```
  ★ PI drafts proposal ──► ★ each expert reviews (Geometry/SLAM, Data): param overrides + concerns
        └────────────► deterministic synthesis: Menu validates recipe + CLAMPS every param
                       ► ExperimentContract  (no raw model string ever reaches the shell)
   …experiment runs & is verified… ► ★ PI decides the next experiment (lineage + memory)
```

### Track B — the implementer (`vo_implementer.py`)  ← genuine algorithm authoring
A sandboxed agent **writes the algorithm from scratch**.

```
  ★ agent writes main.py, runs it via a container-only `run` tool (NO host shell, NO network,
     writes confined to its code dir, eval.py OFF-LIMITS), iterating until it works
        └────────► harness runs the authored code ► independent grader measures on HELD-OUT
```
The harness owns `eval.py` (the grader) and the oracle (metric + threshold come from the
task), so the agent **cannot grade or game its own work**. Even if it writes a fake grader,
the evaluator re-instantiates the real one before judging (anti-tamper).

### Cross-run failure memory (`memory.py`) — the structured handoff

A fresh authoring session has no memory of the last one, so a relaunched agent used to
re-tread the same dead end (the SLAM agent never learned its pose graph had diverged). The
fix is a **repo-level, cross-run failure ledger**, keyed by domain (`slam` / `vo-rgbd` /
`vo-mono`):

```
  experiment REJECTED/REJECTED ─► record_from_experiment(domain, rec) ─► lab_memory/failures.{jsonl,md}
                                                                              │
  next live run ◄── inject_failure_memory(task, domain) prepends prior failures ──┘
                    to task.description  →  the agent reads "last time the optimizer
                    diverged to 412 m; here's how to avoid it" BEFORE it starts
```

This is pure solver-side plumbing — the verifier is untouched. The three live Track-B entry
points inject memory **before** the run and record the failure **after** (only on
REJECTED/FAILED). It augments ver2's per-run `failed_approaches.md` (which starts blank on
each relaunch) with memory that *persists across runs*.

---

## 6. Data & grading (the integrity mechanics)

- **Visible vs held-out.** The solver sees only input frames. The **ground-truth trajectory
  is held-out** — mounted only for the evaluator. (RGB-D experiment: held-out is *separate
  sequences* the solver never authored against, and GT lives outside the input dir so the
  code can't read it.)
- **The metric.** `eval.py` computes **ATE-RMSE** (trajectory error). Monocular scale is
  unobservable, so it aligns with **Sim(3)** (scale-corrected); the RGB-D grader uses
  **SE(3)** (metric — depth must supply scale) and also reports **RPE** (drift) + scale error.
  The alignment policy is fixed in the grader so the solver can't choose a flattering one.
- **The oracle.** A fixed bar: reproduce a known number, or "beat the classical reference
  baseline" (the harness runs the reference to set the bar).
- **The calibration gate.** Before any autonomy, the verifier must VERIFY a known-good run
  **and** REJECT a deliberately degenerate one — proving it isn't a rubber stamp.

---

## 7. What needs what (to run each part)

| Part | Command | Needs |
|---|---|---|
| Offline self-test / gate | `python -m vo_lab.selftest` | Python + numpy/opencv (+ ver2) |
| Unit tests | `pytest tests/` | same — **no API, no Docker** |
| Real-data calibration | `python -m vo_lab.run_vo_tum_rgbd_calibration` | + downloads a TUM sequence (once) |
| Demo render | `python -m vo_lab.visualize …` | + matplotlib/imageio/ffmpeg |
| Track A (live) | `python -m vo_lab.run_vo_committee` | + `claude-agent-sdk` + `ANTHROPIC_API_KEY` |
| Track B (live) | `python -m vo_lab.run_vo_tum_rgbd_implement <bar>` | + that **+ Docker** (sandbox image) |

Everything is a plain Python entry point. The live tracks depend on the **SDK package + an
API key** (+ Docker for Track B) — not on any interactive session.

---

## 8. Why it's built this way (the failures it prevents)

| Failure mode | The mechanism that prevents it |
|---|---|
| Ran out of turns babysitting downloads/training | IO runs as harness **jobs**; budget = tokens + experiments |
| "Open source runs" treated as success | success = **held-out metric ≤ bar**, measured independently |
| Model praises its own work | grader is **separate + deterministic**; self-reports are distrusted |
| Solver games scale / picks easy data / edits grader | fixed alignment, **held-out + GT isolated**, grader re-instantiated |
| Overfit one scene | RGB-D grader scores on **unseen sequences** (generalization) |
| A turn limit discards working code | **resilient author** grades whatever the agent left |
| A hung container stalls the lab | external **watchdog** kills sandbox jobs (threshold > grader timeout) |
| A relaunched agent re-treads a known dead end | **cross-run failure memory** injected into the next session's prompt |

---

*See `claudedocs/` for the full research report, architecture design, and trial write-ups.*
