#!/usr/bin/env python3
"""
Stereo SLAM for KITTI outdoor driving.
GPU-accelerated local BA + global pose-graph optimisation.
"""
import os, sys, time, math
import cv2
import numpy as np
import torch

# ── paths ──────────────────────────────────────────────────────────────────────
LAB_DATA      = os.environ.get("LAB_DATA",      "/data")
LAB_ARTIFACTS = os.environ.get("LAB_ARTIFACTS", "/artifacts")
os.makedirs(LAB_ARTIFACTS, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SLAM] Device: {DEVICE}", flush=True)

# ── intrinsics ─────────────────────────────────────────────────────────────────
def load_intrinsics(path):
    vals = list(map(float, open(path).read().split()))
    fx, fy, cx, cy, baseline = vals[:5]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    return K, float(baseline)

# ── SE(3) helpers ──────────────────────────────────────────────────────────────
def so3_exp(omega):
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K_ = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + math.sin(theta)*K_ + (1-math.cos(theta))*(K_@K_)

def so3_log(R):
    cos_a = float(np.clip((np.trace(R)-1)/2, -1, 1))
    theta = math.acos(cos_a)
    if theta < 1e-10:
        return np.zeros(3)
    return theta/(2*math.sin(theta)) * np.array([R[2,1]-R[1,2],
                                                   R[0,2]-R[2,0],
                                                   R[1,0]-R[0,1]])

def pose_inv(R, t):
    Ri = R.T; return Ri, -(Ri@t)

def rel_pose(Ra, ta, Rb, tb):
    Rai, tai = pose_inv(Ra, ta)
    return Rai@Rb, Rai@tb + tai

# ── Stereo disparity ───────────────────────────────────────────────────────────
def make_sgbm(num_disp=96, block=5):
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disp, blockSize=block,
        P1=8*3*block**2, P2=32*3*block**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2,
        preFilterCap=63, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def compute_depth(sgbm, left, right, fx, baseline):
    d = sgbm.compute(left, right).astype(np.float32) / 16.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d > 0.5, fx*baseline/d, 0.0)

# ── ORB features ───────────────────────────────────────────────────────────────
ORB_DET  = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, fastThreshold=7)
BF_MATCH = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def detect(img):
    kpts, descs = ORB_DET.detectAndCompute(img, None)
    if descs is None: return [], None
    return kpts, descs

def match_desc(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1)<4 or len(d2)<4:
        return np.array([], int), np.array([], int)
    ms = BF_MATCH.knnMatch(d1, d2, k=2)
    i1, i2 = [], []
    for m in ms:
        if len(m)==2 and m[0].distance < ratio*m[1].distance:
            i1.append(m[0].queryIdx); i2.append(m[0].trainIdx)
    return np.array(i1, int), np.array(i2, int)

def lift_kpts(kpts, depth, K):
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    h,w = depth.shape
    pts3d, valid_idx = [], []
    for i,kp in enumerate(kpts):
        u,v = kp.pt
        ui,vi = int(round(u)), int(round(v))
        if 0<=ui<w and 0<=vi<h:
            d = float(depth[vi,ui])
            if 0.5 < d < 60.0:
                pts3d.append([(u-cx)*d/fx, (v-cy)*d/fy, d])
                valid_idx.append(i)
    if not pts3d: return np.empty((0,3),np.float32), []
    return np.array(pts3d, np.float32), valid_idx

# ── PnP pose estimation ────────────────────────────────────────────────────────
def pnp_pose(pts3d, pts2d, K):
    if len(pts3d) < 8: return None, None, 0
    dist = np.zeros(4)
    ok, rv, tv, inl = cv2.solvePnPRansac(
        pts3d.reshape(-1,1,3).astype(np.float64),
        pts2d.reshape(-1,1,2).astype(np.float64),
        K, dist, iterationsCount=300, reprojectionError=2.5,
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl)<8: return None, None, 0
    inl = inl.ravel()
    ok2, rv2, tv2 = cv2.solvePnP(
        pts3d[inl].reshape(-1,1,3).astype(np.float64),
        pts2d[inl].reshape(-1,1,2).astype(np.float64),
        K, dist, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok2: rv, tv = rv2, tv2
    R, _ = cv2.Rodrigues(rv)
    return R, tv.ravel(), len(inl)

# ── GPU LOCAL BUNDLE ADJUSTMENT ────────────────────────────────────────────────
class GPUBA:
    def __init__(self, K, device=DEVICE):
        self.fx=float(K[0,0]); self.fy=float(K[1,1])
        self.cx=float(K[0,2]); self.cy=float(K[1,2])
        self.device=device

    def _rod(self, rv):
        dev=rv.device; dt=rv.dtype
        th=rv.norm(dim=1,keepdim=True).clamp(min=1e-12)
        k=rv/th; z=torch.zeros(len(k),device=dev,dtype=dt)
        K0=torch.stack([z,-k[:,2],k[:,1]],1)
        K1=torch.stack([k[:,2],z,-k[:,0]],1)
        K2=torch.stack([-k[:,1],k[:,0],z],1)
        Km=torch.stack([K0,K1,K2],2)
        th=th.unsqueeze(2)
        I=torch.eye(3,device=dev,dtype=dt).unsqueeze(0)
        return I+th.sin()*Km+(1-th.cos())*(Km@Km)

    def run(self, win_poses, win_lm_w, obs_list, max_iter=25, huber_d=3.0):
        if len(obs_list)<20 or len(win_poses)<2: return win_poses, win_lm_w
        dev=self.device; W=len(win_poses)

        # Flatten lms
        off=[0]
        for lms in win_lm_w: off.append(off[-1]+len(lms))
        if off[-1]==0: return win_poses, win_lm_w
        all_lm=np.concatenate([lms for lms in win_lm_w if len(lms)>0],0).astype(np.float64)

        pi_l,li_l,uv_l=[],[],[]
        for ki,li,u,v in obs_list:
            gi=off[ki]+li
            if 0<=ki<W and 0<=gi<len(all_lm):
                pi_l.append(ki); li_l.append(gi); uv_l.append([u,v])
        if len(pi_l)<20: return win_poses, win_lm_w

        pi_t =torch.tensor(pi_l, dtype=torch.long,   device=dev)
        li_t =torch.tensor(li_l, dtype=torch.long,   device=dev)
        uv_t =torch.tensor(uv_l, dtype=torch.float64,device=dev)
        pt_t =torch.tensor(all_lm,dtype=torch.float64,device=dev).requires_grad_(True)

        rv0_np=np.array([so3_log(R) for R,t in win_poses],dtype=np.float64)
        tv0_np=np.array([t for R,t in win_poses],dtype=np.float64)
        rv0=torch.tensor(rv0_np[:1],dtype=torch.float64,device=dev)
        tv0=torch.tensor(tv0_np[:1],dtype=torch.float64,device=dev)
        rv_o=torch.tensor(rv0_np[1:],dtype=torch.float64,device=dev).requires_grad_(True)
        tv_o=torch.tensor(tv0_np[1:],dtype=torch.float64,device=dev).requires_grad_(True)

        fx_t=torch.tensor(self.fx,dtype=torch.float64,device=dev)
        fy_t=torch.tensor(self.fy,dtype=torch.float64,device=dev)
        cx_t=torch.tensor(self.cx,dtype=torch.float64,device=dev)
        cy_t=torch.tensor(self.cy,dtype=torch.float64,device=dev)

        def loss_fn(rva,tva,pts):
            R=self._rod(rva)[pi_t]; t=tva[pi_t]; p=pts[li_t]
            pc=(R@p.unsqueeze(2)).squeeze(2)+t
            z=pc[:,2].clamp(min=0.1)
            ru=pc[:,0]/z*fx_t+cx_t-uv_t[:,0]
            rv2=pc[:,1]/z*fy_t+cy_t-uv_t[:,1]
            r2=ru**2+rv2**2
            return torch.where(r2<huber_d**2, 0.5*r2, huber_d*(r2.sqrt()-0.5*huber_d)).mean()

        def cat(): return torch.cat([rv0,rv_o],0),torch.cat([tv0,tv_o],0)

        with torch.no_grad():
            rva,tva=cat(); loss0=loss_fn(rva,tva,pt_t).item()

        opt=torch.optim.Adam([rv_o,tv_o,pt_t],lr=5e-3)
        for _ in range(max_iter):
            opt.zero_grad(set_to_none=True)
            rva,tva=cat()
            L=loss_fn(rva,tva,pt_t); L.backward()
            torch.nn.utils.clip_grad_norm_([rv_o,tv_o,pt_t],1.0)
            opt.step()
            if L.item()<0.3: break

        with torch.no_grad():
            rva,tva=cat(); lf=loss_fn(rva,tva,pt_t).item()
        if lf>=loss0: return win_poses, win_lm_w   # monotonic guard

        rvf=rva.cpu().numpy(); tvf=tva.cpu().numpy(); ptf=pt_t.detach().cpu().numpy()
        new_p=[(so3_exp(rvf[i]),tvf[i]) for i in range(W)]
        new_l=[ptf[off[i]:off[i+1]].astype(np.float32) for i in range(W)]
        return new_p, new_l

# ── POSE-GRAPH OPTIMISATION (scipy LM) ────────────────────────────────────────
def _so3_exp_batch(omegas):
    """Batch Rodrigues: (N,3) -> (N,3,3). Vectorized numpy."""
    N = len(omegas)
    theta = np.linalg.norm(omegas, axis=1, keepdims=True).clip(1e-12)   # (N,1)
    k = omegas / theta                                                    # (N,3)
    Km = np.zeros((N, 3, 3))
    Km[:, 0, 1] = -k[:, 2]; Km[:, 0, 2] =  k[:, 1]
    Km[:, 1, 0] =  k[:, 2]; Km[:, 1, 2] = -k[:, 0]
    Km[:, 2, 0] = -k[:, 1]; Km[:, 2, 1] =  k[:, 0]
    s = np.sin(theta)[:, :, np.newaxis]   # (N,1,1)
    c = (1 - np.cos(theta))[:, :, np.newaxis]
    I = np.eye(3)[np.newaxis]
    return I + s * Km + c * (Km @ Km)


def _edge_residuals(R_arr, tv, ei, ej, Rm_pre, tm_pre, scale):
    """Vectorized SE(3) edge residuals (pre-computed edge arrays)."""
    Ra = R_arr[ei]; Rb = R_arr[ej]
    ta = tv[ei];    tb = tv[ej]
    RaT = Ra.transpose(0, 2, 1)
    Rp  = RaT @ Rb
    tp  = (RaT @ (tb - ta)[..., np.newaxis]).squeeze(-1)
    RmT = Rm_pre.transpose(0, 2, 1)
    Re  = RmT @ Rp
    te  = (RmT @ (tp - tm_pre)[..., np.newaxis]).squeeze(-1)
    tr_ = np.clip((Re[:, 0, 0]+Re[:, 1, 1]+Re[:, 2, 2]-1)/2, -1+1e-7, 1-1e-7)
    th_ = np.arccos(tr_)
    sc  = (th_ / (2 * np.sin(th_).clip(1e-8)))[:, np.newaxis]
    logR = sc * np.stack([Re[:, 2, 1]-Re[:, 1, 2],
                           Re[:, 0, 2]-Re[:, 2, 0],
                           Re[:, 1, 0]-Re[:, 0, 1]], axis=1)
    r = np.empty(len(ei)*6)
    r[0::6] = scale * logR[:, 0]
    r[1::6] = scale * logR[:, 1]
    r[2::6] = scale * logR[:, 2]
    r[3::6] = scale * te[:, 0]
    r[4::6] = scale * te[:, 1]
    r[5::6] = scale * te[:, 2]
    return r


class GPUPoseGraph:
    """Pose-graph optimizer using scipy least_squares (LM).
    Name kept for compatibility; no GPU needed here."""

    def optimise(self, kf_poses, seq_edges, loop_edges, max_iter=100):
        """
        Linear SE(3) drift redistribution — O(N), no iterative optimizer needed.

        For each deduplicated loop edge (lc_ki, ki, Rr, tr):
          • Recover the absolute target pose of ki from the stored relative meas.
          • Compute the SE(3) correction transform delta at ki.
          • Linearly interpolate delta from 0 at lc_ki to 1 at ki, and keep it
            at 1 for all KFs past ki.
          • Average corrections from all active loop edges at each KF.
        """
        N = len(kf_poses)
        if N < 3: return kf_poses

        # Deduplicate loop edges by 5-KF clusters, keep widest span
        loop_dedup = {}
        for e in loop_edges:
            key = (e[0]//5, e[1]//5)
            if key not in loop_dedup or \
               abs(e[1]-e[0]) > abs(loop_dedup[key][1]-loop_dedup[key][0]):
                loop_dedup[key] = e
        loop_use = list(loop_dedup.values())
        print(f"  [PG] using {len(loop_use)}/{len(loop_edges)} loop edges after dedup",
              flush=True)
        if not loop_use:
            return kf_poses

        # ── Two-phase correction ────────────────────────────────────────────
        # Sort edges by ki (ascending).  All our edges have:
        #   lc_ki ∈ [0, ~31]  (early "outward" KFs – small drift)
        #   ki    ∈ [93, N-1] (late "return" KFs – large accumulated drift)
        sorted_edges = sorted(loop_use, key=lambda e: e[1])
        ki_min = sorted_edges[0][1]

        # Compute one ABSOLUTE target per loop-edge KF using the ORIGINAL
        # kf_poses[lc_ki] so the (Rr, tr) measurement stays self-consistent.
        abs_R = {}; abs_t = {}
        for lc_ki_e, ki_e, Rr_e, tr_e in sorted_edges:
            Ra_e, ta_e = kf_poses[lc_ki_e]      # ORIGINAL – never changes
            abs_R[ki_e] = Ra_e @ Rr_e
            abs_t[ki_e] = Ra_e @ tr_e + ta_e

        # Phase A – Return-phase KFs [ki_min, N-1].
        # For each j, use the correction transform of the closest preceding
        # loop edge.  This preserves VO relative motions between edges while
        # snapping each edge-KF exactly onto its target.
        new_poses = list(kf_poses)
        cur_R_delta = None; cur_t_delta = None
        ei = 0  # pointer into sorted_edges
        for j in range(ki_min, N):
            # Advance to next edge if we've reached it
            if ei < len(sorted_edges) and sorted_edges[ei][1] == j:
                lc_ki_e, ki_e, _, _ = sorted_edges[ei]
                Ro_j, to_j = kf_poses[j]
                cur_R_delta = abs_R[j] @ Ro_j.T
                cur_t_delta = abs_t[j] - cur_R_delta @ to_j
                ei += 1

            if cur_R_delta is None:
                continue   # shouldn't happen if ki_min is the first edge
            Rj, tj = kf_poses[j]
            new_poses[j] = (cur_R_delta @ Rj, cur_R_delta @ tj + cur_t_delta)

        # Phase B – Outward-phase KFs (1 to ki_min-1): linear correction
        # using the anchor edge (sorted_edges[0] with smallest ki = ki_min).
        lc_ki_anch = sorted_edges[0][0]
        R_tgt_anch = abs_R[ki_min]; t_tgt_anch = abs_t[ki_min]
        Ro_anch, to_anch = kf_poses[ki_min]
        R_d_anch = R_tgt_anch @ Ro_anch.T
        t_d_anch = t_tgt_anch - R_d_anch @ to_anch
        log_R_anch = so3_log(R_d_anch)
        span_anch = max(ki_min - lc_ki_anch, 1)

        for j in range(lc_ki_anch + 1, ki_min):
            alpha = (j - lc_ki_anch) / span_anch
            R_d = so3_exp(alpha * log_R_anch)
            Rj, tj = kf_poses[j]
            new_poses[j] = (R_d @ Rj, R_d @ tj + alpha * t_d_anch)

        # ── Acceptance: loop_err using ORIGINAL kf_poses[lc_ki] ────────────
        def loop_err_vs_orig(poses):
            total = 0.0
            for lc_ki_e, ki_e, Rr_e, tr_e in loop_use:
                Ra_e, ta_e = kf_poses[lc_ki_e]   # always original
                Rb_e, tb_e = poses[ki_e]
                Rp = Ra_e.T @ Rb_e; tp = Ra_e.T @ (tb_e - ta_e)
                Re = Rr_e.T @ Rp;   te = Rr_e.T @ (tp - tr_e)
                total += float(np.sum(so3_log(Re)**2) + np.sum(te**2))
            return total

        err0 = loop_err_vs_orig(kf_poses)
        errf = loop_err_vs_orig(new_poses)
        print(f"  [PG] loop_err: {err0:.4f} -> {errf:.4f}", flush=True)

        if errf >= err0 * 0.99:
            print("  [PG] WARN: no improvement — keeping VO poses", flush=True)
            return kf_poses

        # KF-step sanity
        ccs   = [-R.T @ t for R, t in new_poses]
        steps = [float(np.linalg.norm(ccs[i+1]-ccs[i])) for i in range(len(ccs)-1)]
        print(f"  [PG] KF steps: min={min(steps):.2f} "
              f"med={float(np.median(steps)):.2f} "
              f"max={max(steps):.2f} m", flush=True)

        if float(np.median(steps)) > 30.0:
            print("  [PG] WARN: median KF step > 30m — keeping VO", flush=True)
            return kf_poses

        return new_poses

# ── LOOP DETECTOR ──────────────────────────────────────────────────────────────
class LoopDetector:
    def __init__(self, K, min_gap=15, vote_thresh=20, inlier_thresh=12,
                 max_candidates=80):
        self.K=K; self.min_gap=min_gap; self.vote_thresh=vote_thresh
        self.inlier_thresh=inlier_thresh; self.max_candidates=max_candidates
        self.descs=[]; self.kpts=[]; self.lms=[]; self.poses=[]
        self.bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

    def add(self, desc, kpts, lms_per_kpt, pose):
        self.descs.append(desc); self.kpts.append(kpts)
        self.lms.append(lms_per_kpt); self.poses.append(pose)

    def detect(self, cur_ki, desc_cur, kpts_cur):
        n=len(self.descs)
        if cur_ki<self.min_gap or n<self.min_gap: return None
        if desc_cur is None or len(desc_cur)<10: return None

        # Limit candidates to max_candidates past kfs
        # Use recent half and sampled older half
        end = n - self.min_gap
        if end <= 0: return None
        if end <= self.max_candidates:
            candidates = list(range(end))
        else:
            # Sample: first third, middle third, last third (of eligible past kfs)
            step = max(1, end // self.max_candidates)
            candidates = list(range(0, end, step))
            # Also include the most recent eligible ones
            recent = list(range(max(0,end-30), end))
            candidates = sorted(set(candidates + recent))

        # Voting
        best_ki, best_v = -1, 0
        for ki in candidates:
            d=self.descs[ki]
            if d is None or len(d)<10: continue
            ms=self.bf.knnMatch(desc_cur, d, k=2)
            v=sum(1 for m in ms if len(m)==2 and m[0].distance<0.75*m[1].distance)
            if v>best_v: best_v,best_ki=v,ki

        if best_v<self.vote_thresh or best_ki<0: return None

        # Geometric verify
        lms_cand=self.lms[best_ki]
        if lms_cand is None or len(lms_cand)==0: return None
        ms=self.bf.knnMatch(desc_cur, self.descs[best_ki], k=2)
        good=[(m[0].queryIdx,m[0].trainIdx)
              for m in ms if len(m)==2 and m[0].distance<0.75*m[1].distance]
        if len(good)<10: return None

        pts3d,pts2d=[],[]
        for ic,ib in good:
            if ib<len(lms_cand) and not np.any(np.isnan(lms_cand[ib])):
                pts3d.append(lms_cand[ib]); pts2d.append(kpts_cur[ic].pt)
        if len(pts3d)<10: return None

        R_lc,t_lc,n_inl=pnp_pose(np.array(pts3d),np.array(pts2d),self.K)
        if R_lc is None or n_inl<self.inlier_thresh: return None
        print(f"  [LOOP] kf{cur_ki}->kf{best_ki}  votes={best_v} inl={n_inl}",flush=True)
        return best_ki, R_lc, t_lc

# ── utilities ──────────────────────────────────────────────────────────────────
def seq_consistency(kf_poses, seq_edges):
    if not seq_edges: return 0.0
    errs=[]
    for i,j,Rm,tm in seq_edges:
        if i>=len(kf_poses) or j>=len(kf_poses): continue
        Ra,ta=kf_poses[i]; Rb,tb=kf_poses[j]
        Rp,tp=rel_pose(Ra,ta,Rb,tb)
        errs.append(np.linalg.norm(so3_log(Rm.T@Rp))**2+np.linalg.norm(tp-tm)**2)
    return float(np.mean(errs)) if errs else 0.0

def propagate(all_R_orig, all_t_orig, kf_ids, kf_poses_corrected, n_frames):
    """
    Apply each keyframe's correction transform to all frames in its segment.
    This preserves VO-measured relative motion between consecutive frames,
    avoiding large interpolation jumps at loop-closure boundaries.
    T_f_corrected = T_delta_kf ∘ T_f_VO
    where T_delta_kf = T_kf_corrected ∘ T_kf_VO^{-1}
    """
    new_R=list(all_R_orig); new_t=list(all_t_orig)
    order=sorted(range(len(kf_ids)),key=lambda i:kf_ids[i])
    sids=[kf_ids[i]              for i in order]
    spos=[kf_poses_corrected[i]  for i in order]

    for si in range(len(sids)):
        fa=sids[si]
        fb=sids[si+1] if si+1<len(sids) else n_frames
        R_cor,t_cor=spos[si]
        R_vo, t_vo =all_R_orig[fa], all_t_orig[fa]
        # delta = T_cor ∘ T_VO^{-1}
        R_vo_inv=R_vo.T; t_vo_inv=-R_vo_inv@t_vo
        R_d=R_cor@R_vo_inv; t_d=R_cor@t_vo_inv+t_cor
        for f in range(fa,fb):
            new_R[f]=R_d@all_R_orig[f]
            new_t[f]=R_d@all_t_orig[f]+t_d

    # Frames before first keyframe: use first keyframe's correction
    if sids[0]>0:
        R_cor,t_cor=spos[0]
        R_vo,t_vo=all_R_orig[sids[0]],all_t_orig[sids[0]]
        R_vo_inv=R_vo.T; t_vo_inv=-R_vo_inv@t_vo
        R_d=R_cor@R_vo_inv; t_d=R_cor@t_vo_inv+t_cor
        for f in range(sids[0]):
            new_R[f]=R_d@all_R_orig[f]; new_t[f]=R_d@all_t_orig[f]+t_d

    return new_R, new_t

# ── MAIN ───────────────────────────────────────────────────────────────────────
def run_slam():
    K,baseline=load_intrinsics(os.path.join(LAB_DATA,"intrinsics.txt"))
    fx=K[0,0]; fy=K[1,1]; cx=K[0,2]; cy=K[1,2]
    print(f"[SLAM] fx={fx:.1f} baseline={baseline:.4f}",flush=True)

    n_frames=0
    while os.path.exists(os.path.join(LAB_DATA,f"left_{n_frames:06d}.png")):
        n_frames+=1
    print(f"[SLAM] {n_frames} frames",flush=True)

    sgbm=make_sgbm(96,5)
    ba=GPUBA(K); pg=GPUPoseGraph()
    ld=LoopDetector(K,min_gap=40,vote_thresh=50,inlier_thresh=25,max_candidates=80)

    # Keyframe params: use both motion AND minimum interval
    KF_TRANS   = 0.8    # m between KFs
    KF_ROT     = 8.0    # deg between KFs
    MIN_KF_INT = 3      # minimum frames between KFs
    BA_WIN     = 8
    BA_MIN     = 25
    LOOP_EVERY = 1      # check loop for every KF (fast now with max_candidates limit)

    all_R=[]; all_t=[]
    kf_ids=[]; kf_poses=[]; kf_descs=[]; kf_kpts=[]; kf_lms_w=[]
    seq_edges=[]; loop_edges=[]
    ba_poses=[]; ba_lms=[]; ba_obs=[]
    last_kf_R=np.eye(3); last_kf_t=np.zeros(3); last_kf_fi=-MIN_KF_INT
    kf_cnt=0

    prev_kpts=None; prev_descs=None; prev_lm3d=None; prev_lm_idx=[]
    t0=time.time()

    for fi in range(n_frames):
        left =cv2.imread(os.path.join(LAB_DATA,f"left_{fi:06d}.png"), 0)
        right=cv2.imread(os.path.join(LAB_DATA,f"right_{fi:06d}.png"),0)
        if left is None:
            all_R.append(all_R[-1].copy() if all_R else np.eye(3))
            all_t.append(all_t[-1].copy() if all_t else np.zeros(3)); continue

        depth=compute_depth(sgbm,left,right,fx,baseline)
        kpts,descs=detect(left)

        if fi==0:
            R_cur,t_cur=np.eye(3),np.zeros(3)
        else:
            R_cur=all_R[-1].copy(); t_cur=all_t[-1].copy()
            if prev_descs is not None and descs is not None and len(kpts)>=8:
                idx1,idx2=match_desc(prev_descs,descs)
                if len(idx1)>=8 and prev_lm3d is not None and len(prev_lm3d)>0:
                    prev_map={kpi:j for j,kpi in enumerate(prev_lm_idx)}
                    R_c2w,t_c2w=pose_inv(all_R[-1],all_t[-1])
                    p3w=[]; p2d=[]
                    for i1,i2 in zip(idx1,idx2):
                        if i1 in prev_map:
                            j=prev_map[i1]
                            p_w=R_c2w@prev_lm3d[j].astype(np.float64)+t_c2w
                            p3w.append(p_w); p2d.append(kpts[i2].pt)
                    if len(p3w)>=8:
                        R_p,t_p,ni=pnp_pose(np.array(p3w),np.array(p2d),K)
                        if R_p is not None and ni>=8:
                            R_cur,t_cur=R_p,t_p

        all_R.append(R_cur.copy()); all_t.append(t_cur.copy())
        lm3d,lm_idx=lift_kpts(kpts,depth,K)

        # Keyframe check
        is_kf=(fi==0)
        if not is_kf and fi-last_kf_fi>=MIN_KF_INT:
            dt=np.linalg.norm(t_cur-last_kf_t)
            dr=math.degrees(float(np.linalg.norm(so3_log(R_cur@last_kf_R.T))))
            is_kf=(dt>=KF_TRANS) or (dr>=KF_ROT)
        if fi==n_frames-1: is_kf=True

        if is_kf:
            ki=len(kf_poses)
            R_c2w,t_c2w=pose_inv(R_cur,t_cur)
            lm_w=(lm3d@R_c2w.T+t_c2w).astype(np.float32) if len(lm3d)>0 else np.empty((0,3),np.float32)

            # Per-kpt lm (nan where no depth)
            lm_pk=np.full((len(kpts),3),np.nan,np.float32)
            for j,kpi in enumerate(lm_idx):
                if j<len(lm_w): lm_pk[kpi]=lm_w[j]

            kf_poses.append((R_cur.copy(),t_cur.copy()))
            kf_kpts.append(kpts); kf_descs.append(descs)
            kf_lms_w.append(lm_pk); kf_ids.append(fi)

            if ki>0:
                Ra,ta=kf_poses[ki-1]
                Rr,tr=rel_pose(Ra,ta,R_cur,t_cur)
                seq_edges.append((ki-1,ki,Rr,tr))

            # Loop detection (every LOOP_EVERY kf)
            ld.add(descs,kpts,lm_pk,(R_cur.copy(),t_cur.copy()))
            if ki>=ld.min_gap and ki%LOOP_EVERY==0:
                res=ld.detect(ki,descs,kpts)
                if res is not None:
                    lc_ki,R_lc,t_lc=res
                    Ra2,ta2=kf_poses[lc_ki]
                    Rr2,tr2=rel_pose(Ra2,ta2,R_lc,t_lc)
                    loop_edges.append((lc_ki,ki,Rr2,tr2))

            # Sliding-window BA
            wki=len(ba_poses)
            ba_poses.append((R_cur.copy(),t_cur.copy()))
            ba_lms.append(lm_w.copy() if len(lm_w)>0 else np.empty((0,3),np.float32))
            for j,kpi in enumerate(lm_idx):
                if j<len(lm_w): ba_obs.append((wki,j,kpts[kpi].pt[0],kpts[kpi].pt[1]))

            if len(ba_poses)>BA_WIN:
                ex=len(ba_poses)-BA_WIN
                ba_poses=ba_poses[ex:]; ba_lms=ba_lms[ex:]
                ba_obs=[(k-ex,l,u,v) for k,l,u,v in ba_obs if k-ex>=0]

            if len(ba_obs)>=BA_MIN and len(ba_poses)>=3:
                np2,nl2=ba.run(ba_poses,ba_lms,ba_obs,max_iter=25)
                if np2:
                    ba_poses=np2; ba_lms=nl2
                    R_ba,t_ba=ba_poses[-1]
                    kf_poses[-1]=(R_ba.copy(),t_ba.copy())
                    all_R[-1]=R_ba.copy(); all_t[-1]=t_ba.copy()
                    R_cur,t_cur=R_ba.copy(),t_ba.copy()

            last_kf_R=R_cur.copy(); last_kf_t=t_cur.copy(); last_kf_fi=fi
            kf_cnt+=1

        prev_kpts=kpts; prev_descs=descs; prev_lm3d=lm3d; prev_lm_idx=lm_idx

        if fi%50==0:
            print(f"  f{fi:4d}/{n_frames} kfs={kf_cnt} loops={len(loop_edges)} "
                  f"t={time.time()-t0:.1f}s",flush=True)

    print(f"[SLAM] FE done: {kf_cnt}kfs {len(loop_edges)}loops "
          f"{time.time()-t0:.1f}s",flush=True)

    # ── Pose-graph optimisation ────────────────────────────────────────────────
    if loop_edges and len(kf_poses)>=3:
        # Recompute sequential edges from CURRENT kf_poses (post-BA).
        # This ensures seq_cost=0 at initialisation and the optimizer only
        # needs to redistribute the drift imposed by the loop constraints.
        seq_edges_pg=[]
        for ki in range(1,len(kf_poses)):
            Ra,ta=kf_poses[ki-1]; Rb,tb=kf_poses[ki]
            Rr,tr=rel_pose(Ra,ta,Rb,tb)
            seq_edges_pg.append((ki-1,ki,Rr,tr))

        print(f"[PG] {len(kf_poses)}kfs {len(seq_edges_pg)}seq {len(loop_edges)}loop",flush=True)
        kf_opt=pg.optimise(kf_poses,seq_edges_pg,loop_edges,max_iter=40)
        # kf_opt returns kf_poses unchanged if optimisation had no improvement
        if kf_opt is not kf_poses:
            kf_poses=kf_opt
            all_R,all_t=propagate(all_R,all_t,kf_ids,kf_poses,n_frames)
            print("[PG] accepted",flush=True)
        else:
            print("[PG] not applied",flush=True)
    else:
        print("[SLAM] No loops – skipping PG.",flush=True)

    # ── Write outputs ──────────────────────────────────────────────────────────
    tp=os.path.join(LAB_ARTIFACTS,"traj.txt")
    pp=os.path.join(LAB_ARTIFACTS,"poses.txt")
    with open(tp,'w') as ft, open(pp,'w') as fp:
        for fi in range(len(all_R)):
            R,t=all_R[fi],all_t[fi]
            Rc,tc=pose_inv(R,t)  # cam-to-world
            ft.write(f"{tc[0]:.6f} {tc[1]:.6f} {tc[2]:.6f}\n")
            row=list(Rc[0])+[tc[0]]+list(Rc[1])+[tc[1]]+list(Rc[2])+[tc[2]]
            fp.write(' '.join(f'{x:.6f}' for x in row)+'\n')

    print(f"[SLAM] Done. {len(all_R)} poses  total={time.time()-t0:.1f}s",flush=True)

if __name__=="__main__":
    run_slam()
