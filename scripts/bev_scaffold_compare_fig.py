"""Scaffold experiment payoff figure: does locking the geometry+augmentation collapse the agent's
variance? Plots the three conditions side by side against the oracle bar.

usage: python bev_scaffold_compare_fig.py <out.png> <bar> <scaffold_iou1> <scaffold_iou2> ...
  free-form agent n=3 and the fixed-recipe reference are baked in (from the prior experiment).
"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1]
BAR = float(sys.argv[2])
scaffold = [float(x) for x in sys.argv[3:]]                       # the new scaffold agent runs

freeform = [0.1075, 0.0376, 0.1107]                              # prior free-form agent n=3
reference = [0.1383, 0.1423, 0.1425]                            # fixed-recipe reference (seeds)

groups = [("fixed-recipe\nreference", reference, "#2e86de"),
          ("AGENT free-form\n(authors everything)", freeform, "#7f8c8d"),
          ("AGENT scaffold\n(authors only the net)", scaffold, "#8e44ad")]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.axhline(BAR, color="#c0392b", ls="--", lw=1.4, label=f"oracle bar = {BAR}")
for i, (name, vals, col) in enumerate(groups, 1):
    if not vals:
        continue
    xs = np.full(len(vals), i) + np.linspace(-0.07, 0.07, len(vals))
    cols = [("#27ae60" if v >= BAR else "#e67e22") for v in vals] if i >= 2 else col
    ax.scatter(xs, vals, s=95, c=cols, zorder=3, edgecolors="k", linewidths=0.5)
    ax.plot([i - 0.22, i + 0.22], [np.mean(vals)] * 2, color=col, lw=2.5)
    npass = sum(v >= BAR for v in vals)
    ax.annotate(f"mean {np.mean(vals):.3f}\nstd {np.std(vals):.3f}\n{npass}/{len(vals)} pass",
                (i, max(vals) + 0.008), ha="center", fontsize=8.5)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("held-out vehicle IoU (nuScenes mini_val)")
ax.set_ylim(0, max(0.18, max(scaffold + freeform + reference) + 0.03))
ax.set_title("Does locking the architecture collapse the agent's variance?")
ax.legend(loc="lower right", fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"wrote {OUT}: scaffold std {np.std(scaffold):.4f} vs free-form std {np.std(freeform):.4f}"
      if scaffold else f"wrote {OUT} (no scaffold data yet)")
