"""Visualize feature tracking on a KITTI sequence — keypoints + KLT optical-flow trails as the car drives
(what a VO/SLAM front-end 'sees'). Output: an MP4 + a 4-frame annotated strip PNG. Runs with cv2 (vo-droid
or vo-cpu-opencv image, or host if cv2 present).

usage: python viz_feature_tracking.py <left_frames_dir> <out_prefix> [n_frames]
"""
import sys, glob, os
import numpy as np
import cv2


def main():
    framedir, outpref = sys.argv[1], sys.argv[2]
    nmax = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    files = sorted(glob.glob(os.path.join(framedir, "left_*.png")))[:nmax]
    if not files:
        files = sorted(glob.glob(os.path.join(framedir, "*.png")))[:nmax]
    g0 = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    H, W = g0.shape
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, (500, 3)).tolist()
    p0 = cv2.goodFeaturesToTrack(g0, 400, 0.01, 8)
    tracks = {i: [tuple(p0[i, 0])] for i in range(len(p0))}   # id -> list of points (trail)
    nextid = len(p0); ids = list(range(len(p0)))
    prev = g0
    writer = None
    try:
        writer = cv2.VideoWriter(outpref + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 12, (W, H))
    except Exception:
        writer = None
    strip_frames = []
    strip_at = {0, len(files) // 3, 2 * len(files) // 3, len(files) - 1}
    for fi, f in enumerate(files[1:], 1):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        p0 = np.array([tracks[i][-1] for i in ids], np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None,
                                             winSize=(21, 21), maxLevel=3)
        new_ids = []
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for k, i in enumerate(ids):
            if st[k][0] == 1:
                pt = tuple(p1[k, 0]); tracks[i].append(pt); new_ids.append(i)
                trail = tracks[i][-12:]                       # last 12 positions = motion trail
                for a, b in zip(trail[:-1], trail[1:]):
                    cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colors[i % 500], 1)
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, colors[i % 500], -1)
        ids = new_ids
        if len(ids) < 250:                                   # re-seed features (front-end behavior)
            mask = np.full((H, W), 255, np.uint8)
            for i in ids:
                cv2.circle(mask, (int(tracks[i][-1][0]), int(tracks[i][-1][1])), 8, 0, -1)
            extra = cv2.goodFeaturesToTrack(g, 300, 0.01, 8, mask=mask)
            if extra is not None:
                for e in extra:
                    tracks[nextid] = [tuple(e[0])]; ids.append(nextid); nextid += 1
        cv2.putText(vis, f"frame {fi}  |  {len(ids)} tracked features",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if writer is not None:
            writer.write(vis)
        if fi in strip_at:
            strip_frames.append(vis.copy())
        prev = g
    if writer is not None:
        writer.release()
    # 4-frame vertical strip
    if strip_frames:
        strip = np.vstack(strip_frames)
        cv2.imwrite(outpref + "_strip.png", strip)
    print(f"  wrote {outpref}.mp4 + {outpref}_strip.png ({len(files)} frames, ~{nextid} total features)")


if __name__ == "__main__":
    main()
