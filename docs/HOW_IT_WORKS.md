# How LenaLab works

A guide to the architecture: an AI agent that researches, builds, and trains computer-vision
algorithms — what runs, in what order, and **where the agent's research and engineering live**.

---

## 1. The one idea

An AI agent does the research: it analyzes the problem → researches an approach →
implements + trains → confirms it generalizes. LenaLab gives that work two parts:

- a **solver** (the LLM agent) that *researches and produces* a solution to a vision problem, and
- a **verifier** (deterministic, no LLM) that *measures it* on a **held-out** split, giving the
  agent an honest read on whether the solution generalizes.

The numbers are trustworthy because they're measured on data the agent never trained on; the
agent supplies the research, held-out measurement keeps the scoreboard honest.

```
            researches & produces a solution         measures it on held-out data
   ┌────────────────────────────┐            ┌────────────────────────────────────┐
   │  SOLVER  (AI lives here)    │  artifact  │  VERIFIER  (no AI — pure Python)     │
   │  • committee proposes, or   │ ─────────► │  run in a sandbox, then measure on   │
   │  • agent authors code        │  (code /   │  a HELD-OUT split with a grader the  │
   │                              │  trajectory)│  solver doesn't author  → VERIFIED/FAIL│
   └────────────────────────────┘            └────────────────────────────────────┘
```

The solver is where the research happens; the verifier is the constant that keeps the
measurement honest. This is the generator⟂evaluator separation from Anthropic's harness-design
work — the single most important lever.

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
     │  metric_extractor.extract()          records the solver's SELF-REPORT (kept; verdict uses held-out)
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

## 4. Where the agent does its research (and what the lab handles)

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

The agent's research and engineering live **in the solver**. The verifier is LLM-free —
grading is closed-form geometry. This is deliberate: keeping measurement independent of the
model means the "VERIFIED" verdict reflects the work generalizing, not the model assessing
itself.

The AI dependency is also behind an **injectable seam**: the committee takes a `run_fn`, the
implementer takes an `author_fn`. Tests inject fakes, so the offline suite and calibrations
run with **no API key and no `claude` binary** at all.

### How the AI is actually called

- It uses the **`claude-agent-sdk`** Python package (which bundles its own `claude` binary)
  with `ANTHROPIC_API_KEY` — **not** the interactive Claude Code app. You launch a plain
  `python -m vo_lab.…` and the SDK spawns the model itself.
- Each call is a **fresh, isolated session** (`setting_sources=[]`, no shared context) — so
  the proposer and any reviewer each reason independently.

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
task), so the agent gets an **independent read on its own work**. The evaluator always
re-instantiates the real grader before judging, so the score reflects the held-out
measurement regardless of what's in the sandbox.

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

## 6. Data & grading (the measurement backbone)

- **Visible vs held-out.** The solver sees only input frames. The **ground-truth trajectory
  is held-out** — mounted only for the evaluator. (RGB-D experiment: held-out is *separate
  sequences* the solver never authored against, and GT lives outside the input dir so the
  code can't read it.)
- **The metric.** `eval.py` computes **ATE-RMSE** (trajectory error). Monocular scale is
  unobservable, so it aligns with **Sim(3)** (scale-corrected); the RGB-D grader uses
  **SE(3)** (metric — depth must supply scale) and also reports **RPE** (drift) + scale error.
  The alignment policy is fixed in the grader so every run is measured the same way.
- **The oracle.** A fixed bar: reproduce a known number, or "beat the classical reference
  baseline" (the harness runs the reference to set the bar).
- **The calibration gate.** Before any autonomy, the verifier must VERIFY a known-good run
  **and** REJECT a deliberately degenerate one — confirming the measurement discriminates
  good work from bad before the agent relies on it.

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

## 8. Why it's built this way (what each choice buys the agent's research)

| Failure mode | The mechanism that prevents it |
|---|---|
| Ran out of turns babysitting downloads/training | IO runs as harness **jobs**; budget = tokens + experiments |
| "Open source runs" mistaken for a result | success = **held-out metric ≤ bar**, measured independently |
| Self-assessment inflates the result | grader is **separate + deterministic**; the verdict comes from held-out data |
| Inconsistent scale / data / grading across runs | fixed alignment, **held-out + GT isolated**, grader re-instantiated |
| Overfit one scene | RGB-D grader scores on **unseen sequences** (generalization) |
| A turn limit discards working code | **resilient author** grades whatever the agent left |
| A hung container stalls the lab | external **watchdog** kills sandbox jobs (threshold > grader timeout) |
| A relaunched agent re-treads a known dead end | **cross-run failure memory** injected into the next session's prompt |

---

*See `claudedocs/` for the full research report, architecture design, and trial write-ups.*
