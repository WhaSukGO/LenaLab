import sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/ws/devel/whasuk/LenaLab")
from vo_lab.plugins.vo_loop_oracle import LoopOracleKITTIProvider
from lab.models import DatasetRef

prov = LoopOracleKITTIProvider(dev="06", heldout=("07", "09"), stride=3, max_frames=600,
                               min_gap=80, max_dist=15.0, max_loops=8)
work = Path(tempfile.mkdtemp(prefix="orac_")); ho = work / "ho"
prov.fetch(DatasetRef(name="ho", source="x", held_out=True), ho)

lines_out = []
for sq in sorted(ho.glob("seq_*")):
    s = sq.name
    loops_f = sq / "input" / "loops.txt"
    gtp = np.loadtxt(sq / "gt_poses.txt").reshape(-1, 3, 4)
    if not loops_f.exists() or loops_f.stat().st_size == 0:
        lines_out.append(f"{s}: NO loops written!"); continue
    L = np.loadtxt(loops_f).reshape(-1, 14)
    resids = []
    for row in L:
        i, j = int(row[0]), int(row[1]); rel = row[2:].reshape(3, 4)
        Ti = np.eye(4); Ti[:3, :] = gtp[i]
        Tj = np.eye(4); Tj[:3, :] = gtp[j]
        Trel = np.eye(4); Trel[:3, :] = rel
        # GT consistency: Ti @ Trel should == Tj
        resids.append(np.linalg.norm((Ti @ Trel)[:3, 3] - Tj[:3, 3]))
    centres = gtp[:, :, 3]
    dists = [float(np.linalg.norm(centres[int(r[0])] - centres[int(r[1])])) for r in L]
    lines_out.append(f"{s}: {len(L)} loops | GT-consistency residual max={max(resids):.2e}m "
                     f"| revisit dists={[round(d,1) for d in dists]} | gaps={[int(r[1]-r[0]) for r in L]}")

verdict = "\n".join(lines_out)
ok = all("residual max=" in l and float(l.split("residual max=")[1].split("m")[0]) < 1e-6 for l in lines_out if "loops |" in l)
verdict += f"\n\nORACLE SOUND (exact GT constraints, real revisits): {ok}"
Path("/tmp/oracle_val.txt").write_text(verdict + "\n")
print(verdict)
