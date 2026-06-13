"""The VO research-team "meeting" (Track A) — ver2's Committee with a VO expert panel.

This is the SOLVER's reasoning seat. The PI drafts a menu-constrained proposal; each
expert reviews (suggesting param overrides within range and raising concerns); a
deterministic synthesis builds the contract through the Menu, which validates the recipe
and clamps every parameter. The experts can ONLY select + parameterize a vetted recipe —
they cannot author commands, datasets, or code — so a bad suggestion can't escape the menu.

What Track A demonstrates is the research-loop *machinery and safety properties*, not a
guaranteed improvement curve: menu-constrained proposal -> independent held-out
verification -> lineage with memory, all gated behind reproduction-first calibration.
Genuine algorithm authoring lives in Track B (the Implementer). Live use bills tokens
(ANTHROPIC_API_KEY); offline tests inject a fake run_fn."""
from __future__ import annotations

from lab.agents.committee import PI, Committee, Expert
from lab.agents.sdk import DEFAULT_MODEL, RunFn, run_agent
from lab.history import ResearchHistory
from lab.notebook import Notebook

from ..plugins.vo import vo_menu

GEOMETRY = Expert("Geometry/SLAM", (
    "You are the geometry / SLAM expert on a visual-odometry research team. Review the "
    "draft from the standpoint of epipolar geometry, essential-matrix conditioning, "
    "outlier rejection (RANSAC threshold), feature count, and monocular scale drift. "
    "Suggest overrides ONLY among the recipe's declared params, within their ranges. "
    "Raise concrete concerns; approve when the configuration is geometrically sound."))

MODELING = Expert("Modeling", (
    "You are the estimation expert. Given the draft, suggest declared-param overrides "
    "(within range) you expect to lower trajectory error. Raise concerns if the draft "
    "looks weak; approve when sound."))

DATA = Expert("Data", (
    "You are the data expert. Check the held-out split for leakage (does any held-out "
    "sequence overlap the visible frames?) and the soundness of the Sim(3) scale "
    "alignment used to score monocular VO. You rarely change hyperparameters; raise "
    "concrete data concerns and approve or reject."))

VO_EXPERTS = [GEOMETRY, DATA]  # minimal panel; add MODELING only if 2-expert proves weak


def vo_committee(*, model: str = DEFAULT_MODEL, run_fn: RunFn = run_agent,
                 notebook: Notebook | None = None, history: ResearchHistory | None = None,
                 experts=None, menu=None) -> Committee:
    return Committee(menu if menu is not None else vo_menu(), model=model, run_fn=run_fn, pi=PI,
                     experts=VO_EXPERTS if experts is None else experts,
                     notebook=notebook, history=history)
