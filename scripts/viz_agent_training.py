"""Make the AGENT's training VISIBLE: parse a Track-B run's train.log and plot the agent's own
training trajectory — loss + its validation metric over the epochs IT chose, with the best-checkpoint
(★) marks and any stage boundary. Shows train→self-verify→improve, not just the final graded number.

usage: python viz_agent_training.py <train.log> <out.png> [held_out_iou] [title]
"""
import sys, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG, OUT = sys.argv[1], sys.argv[2]
HELD = float(sys.argv[3]) if len(sys.argv) > 3 else None
TITLE = sys.argv[4] if len(sys.argv) > 4 else "Agent self-training trajectory"
txt = open(LOG).read()

ep, loss, vx, vy, vstar = [], [], [], [], []
e = 0
stage2_at = None
for line in txt.splitlines():
    m = re.search(r"Ep\s+(\d+)/(\d+)\s+loss=([0-9.]+)", line)
    ft = re.search(r"\[FT\]\s+ep\s+(\d+)/(\d+)\s+loss=([0-9.]+)", line)
    if m:
        e += 1; ep.append(e); loss.append(float(m.group(3)))
        v = re.search(r"val_mIoU=([0-9.]+)", line)
        if v:
            vx.append(e); vy.append(float(v.group(1))); vstar.append("★" in line)
    elif ft:
        if stage2_at is None:
            stage2_at = e + 0.5
        e += 1; ep.append(e); loss.append(float(ft.group(3)))

fig, ax1 = plt.subplots(figsize=(9, 4.6))
ax1.plot(ep, loss, color="#7f8c8d", lw=1.8, label="training loss (agent-authored loop)")
ax1.set_xlabel("epoch (the agent's own schedule)"); ax1.set_ylabel("training loss", color="#7f8c8d")
ax1.tick_params(axis="y", labelcolor="#7f8c8d")
if stage2_at:
    ax1.axvline(stage2_at, color="#999", ls=":", lw=1)
    ax1.text(stage2_at + 0.3, max(loss) * 0.96, "→ stage-2\n   fine-tune", fontsize=8, color="#555")
ax2 = ax1.twinx()
ax2.plot(vx, vy, color="#27ae60", lw=2, marker="o", ms=5, label="val mIoU (agent verifies on its own split)")
for x, y, s in zip(vx, vy, vstar):
    if s:
        ax2.annotate("★", (x, y), textcoords="offset points", xytext=(-4, 5), color="#27ae60", fontsize=9)
ax2.set_ylabel("validation mIoU", color="#27ae60"); ax2.tick_params(axis="y", labelcolor="#27ae60")
ax2.set_ylim(0, max(vy) * 1.25 if vy else 1)
if HELD is not None:
    ax2.axhline(HELD, color="#c0392b", ls="--", lw=1.2)
    ax2.text(ep[-1], HELD, f" held-out IoU {HELD:.3f}\n (harness grade)", color="#c0392b", fontsize=8, va="bottom", ha="right")
ax1.set_title(TITLE)
l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="center right", fontsize=8.5, framealpha=.95)
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"wrote {OUT}: {len(ep)} epochs, {len(vx)} val checks, best val {max(vy):.3f}" if vy else f"wrote {OUT}")
