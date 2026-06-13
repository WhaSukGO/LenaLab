"""Billed M2 loop-closure TOURNAMENT (first real use of the parallel lab).

Three diverse loop-closure approaches author ONLY the loop-closure layer on the LOCKED, verified
front-end (frontend.py, scaffold seed). The front-end is a never-rebuilt checkpoint; the independent
verifier grades each approach on the loopy held-out (07/09) and the best VERIFIED wins. Toward the
final goal — break the M2 wall toward ORB-SLAM2 ~1.15%.

Re-runnable: each variant runs in its own root (./_vo_m2_tournament/<label>); partial progress
persists on disk (resilient author salvages any main.py it left). Result written to
/tmp/m2_tournament_result.txt.
"""
import sys, dataclasses, json
from pathlib import Path
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")

from vo_lab.parallel_lab import tournament, sdk_factory
from vo_lab.agents.vo_implementer import vo_impl_task_kitti_slam_scaffold, _FRONTEND_KITTI_CODE
from vo_lab.plugins.vo_kitti import KITTIOdomProvider

base = vo_impl_task_kitti_slam_scaffold(1.8, dev="06", heldout=("07", "09"))

APPROACHES = {
    "bow_fullgraph": (
        "APPROACH FOR THIS ATTEMPT — aggressive global SLAM: build a bag-of-visual-words vocabulary "
        "from the frontend ORB descriptors and detect loops by BoW similarity (find as many true "
        "revisits as possible, geometrically verified by PnP RANSAC). Then run a FULL SE(3) "
        "pose-graph optimisation (Gauss-Newton with Lie-algebra increments) over ALL keyframes with "
        "odometry + every verified loop edge."),
    "conservative_drift": (
        "APPROACH FOR THIS ATTEMPT — conservative & safe: accept ONLY very high-confidence loops "
        "(large inlier count, low reprojection residual, temporal consistency). For each accepted "
        "loop, distribute the accumulated drift smoothly along the loop (a simple, robust correction "
        "— NOT a full nonlinear graph). The overriding rule: NEVER make the trajectory worse than the "
        "front-end; if unsure, leave it unchanged."),
    "robust_verified": (
        "APPROACH FOR THIS ATTEMPT — strict verification + robust graph: match each keyframe's "
        "descriptors against all sufficiently-old keyframes, verify candidates with a strict PnP "
        "RANSAC (high inlier threshold) using the frontend 3-D points, then optimise an SE(3) "
        "pose-graph with a ROBUST (Huber) loss. Before committing, CHECK that the optimised graph "
        "reduces trajectory inconsistency; reject the closure if it does not."),
}

variants = [
    {"label": label,
     "task": dataclasses.replace(base, description=base.description + "\n\n" + hint),
     "author_factory": sdk_factory(model="claude-sonnet-4-6", max_turns=100),
     "seed_files": {"frontend.py": _FRONTEND_KITTI_CODE},
     "hypothesis": f"M2 loop closure via {label}"}
    for label, hint in APPROACHES.items()
]

prov = KITTIOdomProvider(dev="06", heldout=("07", "09"), stride=3, max_frames=600)
res = tournament(variants, provider=prov, root="./_vo_m2_tournament", op="<=",
                 job_mode="docker", lease_timeout_s=3600.0, max_workers=2)

ranked = [{"label": r["label"], "status": r["status"], "t_err": r.get("metric"),
           "code": r.get("code_path")} for r in res["ranked"]]
w = res["winner"]
summary = {
    "winner": (w["label"] if w else None),
    "winner_t_err": (w.get("metric") if w else None),
    "winner_passed": (w.get("passed") if w else None),
    "ranked": ranked,
    "anchors": {"floor_frontend": 2.81, "bar": 1.8, "ideal_closure": 1.32, "orbslam2": 1.15},
}
Path("/tmp/m2_tournament_result.txt").write_text(json.dumps(summary, indent=2))
print("\n=== M2 TOURNAMENT RESULT ===")
print(json.dumps(summary, indent=2))
