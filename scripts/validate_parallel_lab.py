"""Offline validation of the parallel lab (non-billed): a COMPETITION tournament where the
reference VO should beat a degenerate control, plus a 2-stage COOPERATION chain that locks stage-1's
winner into stage-2. Uses job_mode='local' (no Docker, no API) on a small KITTI slice. Writes the
verdict to /tmp/parallel_val.txt."""
import sys
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")

from vo_lab.parallel_lab import tournament, cooperative_pipeline, ref_factory
from vo_lab.agents.vo_implementer import (vo_impl_task_kitti, kitti_stereo_reference_author,
                                          kitti_degenerate_author)
from vo_lab.plugins.vo_kitti import KITTIOdomProvider

prov = KITTIOdomProvider(dev="00", heldout=("07",), stride=3, max_frames=120)
task = vo_impl_task_kitti(6.0, dev="00", heldout=("07",))
out = []

# --- COMPETITION: reference vs degenerate vs a 2nd reference, in parallel ---
variants = [
    {"label": "reference", "task": task, "author_factory": ref_factory(kitti_stereo_reference_author())},
    {"label": "degenerate", "task": task, "author_factory": ref_factory(kitti_degenerate_author())},
    {"label": "reference2", "task": task, "author_factory": ref_factory(kitti_stereo_reference_author())},
]
res = tournament(variants, provider=prov, root="./_parallel_val/compete",
                 op="<=", job_mode="local", max_workers=3)
w = res["winner"]
ranked_labels = [(r["label"], r["status"], r["metric"]) for r in res["ranked"]]
ok_compete = (w is not None and w["label"].startswith("reference") and w["passed"]
              and any(r["label"] == "degenerate" and not r["passed"] for r in res["ranked"]))
out.append(f"COMPETITION: winner={w['label'] if w else None} ({w['metric'] if w else None}) | "
           f"ranked={ranked_labels} | PASS={ok_compete}")

# --- COOPERATION: 2-stage chain; stage-1 winner locked into stage-2 as 'base_prev.py' ---
coop = cooperative_pipeline(
    stages=[
        {"name": "frontend", "variants": [
            {"label": "ref", "task": task, "author_factory": ref_factory(kitti_stereo_reference_author())}]},
        {"name": "refine", "seed_as": "base_prev.py", "variants": [
            {"label": "ref", "task": task, "author_factory": ref_factory(kitti_stereo_reference_author())}]},
    ],
    provider=prov, root="./_parallel_val/coop", op="<=", job_mode="local", max_workers=2)
final = coop["final"]
# check stage-2 actually received the locked seed file
stage2_code = Path("./_parallel_val/coop/stage1_refine/ref/workspaces")
seeded = list(stage2_code.glob("*/code/base_prev.py")) if stage2_code.exists() else []
ok_coop = (final is not None and final["passed"] and len(seeded) > 0)
out.append(f"COOPERATION: final={final['label'] if final else None} "
           f"({final['metric'] if final else None}) | seed_locked_into_stage2={len(seeded)>0} | PASS={ok_coop}")

verdict = "\n".join(out) + f"\n\nALL PASS: {ok_compete and ok_coop}"
open("/tmp/parallel_val.txt", "w").write(verdict + "\n")
print(verdict)
