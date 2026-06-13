"""
Stereo VO for KITTI outdoor driving.
KEY FIX: motion cap uses actual world displacement (tn - t_cw), NOT
         the formula (tn - relR @ t_cw) which blows up at large rotations
         far from the origin.

Pipeline:
  1. SGBM disparity -> metric depth (Z = fx*bl/d)
  2. LK optical flow + FB check
  3. 3D-2D PnP+RANSAC (metric pose)
  4. Motion cap on ACTUAL world displacement (max 3m)
  5. Sliding-window BA (20 KF, Huber TRF, vectorised)
     - Cross-KF features at Z>10m; multi-hop tracking from 2 past KFs
"""

import numpy as np
import cv2
import os
from scipy.optimize import least_squares
import time

LAB_DATA      = os.environ.get('LAB_DATA',      '/data')
LAB_ARTIFACTS = os.environ.get('LAB_ARTIFACTS', '/artifacts')
os.makedirs(LAB_ARTIFACTS, exist_ok=True)

with open(os.path.join(LAB_DATA,'intrinsics.txt')) as f:
    v=f.read().split()
fx,fy,cx,cy,bl=float(v[0]),float(v[1]),float(v[2]),float(v[3]),float(v[4])
K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)

left_files=sorted(f for f in os.listdir(LAB_DATA) if f.startswith('left_') and f.endswith('.png'))
N_FRAMES=len(left_files)
print(f"N={N_FRAMES}, fx={fx:.2f}, bl={bl:.4f}m")

sgbm=cv2.StereoSGBM_create(
    minDisparity=0,numDisparities=128,blockSize=5,
    P1=8*5*5,P2=32*5*5,disp12MaxDiff=1,uniquenessRatio=10,
    speckleWindowSize=100,speckleRange=32,
    preFilterCap=63,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def load_frame(i):
    L=cv2.imread(os.path.join(LAB_DATA,f'left_{i:06d}.png'), cv2.IMREAD_GRAYSCALE)
    R=cv2.imread(os.path.join(LAB_DATA,f'right_{i:06d}.png'),cv2.IMREAD_GRAYSCALE)
    return L,R

def compute_depth(L,R):
    d=sgbm.compute(L,R).astype(np.float32)/16.
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.where(d>1.,fx*bl/d,0.)

def backproject(pts2d,depth,zmin=1.,zmax=80.):
    h,w=depth.shape; N=len(pts2d)
    out=np.zeros((N,3),np.float64); ok=np.zeros(N,dtype=bool)
    for i,(pu,pv) in enumerate(pts2d):
        iu,iv=int(round(pu)),int(round(pv))
        if 0<=iv<h and 0<=iu<w:
            z=float(depth[iv,iu])
            if zmin<z<zmax: out[i]=[(pu-cx)*z/fx,(pv-cy)*z/fy,z]; ok[i]=True
    return out,ok

FEAT=dict(maxCorners=700,qualityLevel=0.01,minDistance=7,blockSize=7)
LK=dict(winSize=(21,21),maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))

def detect(img):
    p=cv2.goodFeaturesToTrack(img,mask=None,**FEAT)
    return p[:,0,:] if p is not None else np.zeros((0,2),np.float32)

def track_lk(im1,im2,pts):
    if len(pts)==0: return pts.copy(),np.zeros(0,dtype=bool)
    p1=pts.astype(np.float32)
    p2,s1,_=cv2.calcOpticalFlowPyrLK(im1,im2,p1,None,**LK)
    p1b,s2,_=cv2.calcOpticalFlowPyrLK(im2,im1,p2,None,**LK)
    fb=np.linalg.norm(p1-p1b,axis=1)
    return p2,(s1.ravel()==1)&(s2.ravel()==1)&(fb<1.5)

def r2m(r): return cv2.Rodrigues(np.asarray(r,np.float64))[0]
def m2r(R): return cv2.Rodrigues(R)[0].ravel()

def motion_ok(relR, world_disp, max_t=3., max_r=15.):
    """
    world_disp = t_cw_new - t_cw_old  (ACTUAL camera displacement in world frame).
    NOT the formula (t_cw_new - relR @ t_cw_old) which diverges far from origin.
    """
    if np.linalg.norm(world_disp) > max_t: return False
    c = np.clip((np.trace(relR)-1)/2, -1, 1)
    return np.degrees(np.arccos(c)) <= max_r

def proj_w(pts,rv,tv):
    R=r2m(rv); pc=(R.T@(pts-tv).T).T
    ok=pc[:,2]>0.1
    u=np.full(len(pts),np.nan); vv=np.full(len(pts),np.nan)
    if ok.any(): u[ok]=fx*pc[ok,0]/pc[ok,2]+cx; vv[ok]=fy*pc[ok,1]/pc[ok,2]+cy
    return np.stack([u,vv],1),ok

# ====================================================================
# Sliding Window BA
# ====================================================================
class SWBA:
    def __init__(self,win=20,max_lm=100):
        self.win=win; self.max_lm=max_lm
        self.poses=[]; self.frames=[]; self.lms=[]

    def add_kf(self,fidx,R_cw,t_cw,pts3d_cam,pts2d_cam,cross_list=None):
        rv=m2r(R_cw); tv=t_cw.ravel().copy()
        ki=len(self.poses)
        self.poses.append((rv,tv)); self.frames.append(fidx)

        for j in range(len(pts3d_cam)):
            pw=R_cw@pts3d_cam[j]+tv
            self.lms.append({'p':pw,'obs':{ki:(float(pts2d_cam[j,0]),float(pts2d_cam[j,1]))}})

        if cross_list:
            for (cp3w,cp2p,cp2c,ki_prev) in cross_list:
                if ki_prev<0 or ki_prev>=ki: continue
                rv_p,tv_p=self.poses[ki_prev]
                uvp,okp=proj_w(cp3w,rv_p,tv_p)
                uvc,okc=proj_w(cp3w,rv,tv)
                gm=okp&okc
                if not gm.any(): continue
                ep=np.hypot(uvp[:,0]-cp2p[:,0],uvp[:,1]-cp2p[:,1])
                ec=np.hypot(uvc[:,0]-cp2c[:,0],uvc[:,1]-cp2c[:,1])
                for j in np.where(gm&(ep<30)&(ec<30))[0]:
                    self.lms.append({'p':cp3w[j].copy(),'obs':{
                        ki_prev:(float(cp2p[j,0]),float(cp2p[j,1])),
                        ki:     (float(cp2c[j,0]),float(cp2c[j,1]))}})

        if len(self.poses)>self.win: self._trim()

    def _trim(self):
        self.poses.pop(0); self.frames.pop(0)
        nl=[]
        for lm in self.lms:
            ob={(k-1):v for k,v in lm['obs'].items() if k>0}
            if ob: lm['obs']=ob; nl.append(lm)
        self.lms=nl

    def run_ba(self,max_iter=20,huber=2.):
        nk=len(self.poses)
        if nk<2: return False
        active=[lm for lm in self.lms
                if len(lm['obs'])>=2 and all(0<=k<nk for k in lm['obs'])]
        if len(active)<5: return False
        if len(active)>self.max_lm:
            # Deterministic stride: uniform temporal coverage (oldest→newest)
            step=len(active)//self.max_lm
            active=[active[i] for i in range(0,len(active),step)][:self.max_lm]

        ki_a,lj_a,uo_a,vo_a=[],[],[],[]
        for j,lm in enumerate(active):
            for ki,(u,v) in lm['obs'].items():
                if 0<=ki<nk: ki_a.append(ki);lj_a.append(j);uo_a.append(u);vo_a.append(v)
        if len(ki_a)<10: return False
        ki_a=np.array(ki_a,np.int32); lj_a=np.array(lj_a,np.int32)
        uo_a=np.array(uo_a); vo_a=np.array(vo_a); nobs=len(ki_a)

        nf=nk-1; nlm=len(active); rv0,tv0=self.poses[0]; off=nf*6
        x0=np.empty(off+nlm*3)
        for i in range(nf):
            rv,tv=self.poses[i+1]; x0[i*6:i*6+3]=rv; x0[i*6+3:i*6+6]=tv
        for j,lm in enumerate(active): x0[off+j*3:off+j*3+3]=lm['p']
        R0fix=r2m(rv0)

        def resid(x):
            lmpos=x[off:].reshape(nlm,3)
            r=np.full(nobs*2,huber*10.)
            for kk in range(nk):
                m=ki_a==kk
                if not m.any(): continue
                if kk==0: Rk=R0fix; tvk=tv0
                else: ii=kk-1; Rk=r2m(x[ii*6:ii*6+3]); tvk=x[ii*6+3:ii*6+6]
                g=m.nonzero()[0]; pw=lmpos[lj_a[g]]
                pc=(Rk.T@(pw-tvk).T).T; ok=pc[:,2]>0.1
                if ok.any():
                    r[g[ok]*2]  =fx*pc[ok,0]/pc[ok,2]+cx-uo_a[g[ok]]
                    r[g[ok]*2+1]=fy*pc[ok,1]/pc[ok,2]+cy-vo_a[g[ok]]
            return r

        try:
            res=least_squares(resid,x0,method='trf',loss='huber',f_scale=huber,
                              max_nfev=max_iter*len(x0),
                              ftol=1e-4,xtol=1e-4,gtol=1e-4)
        except Exception: return False

        xo=res.x
        for i in range(nf):
            if np.linalg.norm(xo[i*6+3:i*6+6]-self.poses[i+1][1])>3.: return False
        for i in range(nf):
            self.poses[i+1]=(xo[i*6:i*6+3].copy(),xo[i*6+3:i*6+6].copy())
        lmopt=xo[off:].reshape(nlm,3)
        for j,lm in enumerate(active): lm['p']=lmopt[j].copy()
        return True

    def latest_pose(self):
        if not self.poses: return None,None
        rv,tv=self.poses[-1]; return r2m(rv),tv.copy()

    def get_frame_idx(self,fidx):
        if fidx in self.frames: return self.frames.index(fidx)
        return -1


# ====================================================================
# Main VO
# ====================================================================
def run_vo():
    t_start=time.time()
    R_cw=np.eye(3); t_cw=np.zeros(3)
    Rlist=[R_cw.copy()]; tlist=[t_cw.copy()]
    # Store recent VALID (relR, world_disp) pairs for constant-velocity fallback
    recR,rec_disp=[],[]; VWIN=5

    ba=SWBA(win=20,max_lm=60)
    KF_STEP=8

    HISTORY=2
    hist_imgs=[]; hist_p2=[]; hist_p3w=[]; hist_fids=[]

    L0,R0=load_frame(0); d0=compute_depth(L0,R0)
    pp=detect(L0)
    p3d,vm=backproject(pp,d0,zmin=10.,zmax=80.)
    if vm.sum()>=5:
        ba.add_kf(0,R_cw,t_cw,p3d[vm],pp[vm])
        hist_imgs.append(L0.copy()); hist_p2.append(pp[vm].copy())
        hist_p3w.append(p3d[vm].copy()); hist_fids.append(0)  # world=cam at frame 0

    dp=d0; Lp=L0; ftimes=[]
    n_cap=0  # count of capped frames (for diagnostics)

    for i in range(1,N_FRAMES):
        ft=time.time()
        Lc,Rc=load_frame(i)
        if len(pp)<20: pp=detect(Lp)

        pc2,gd=track_lk(Lp,Lc,pp)
        ntk=int(gd.sum()); gp=pp[gd]; gc=pc2[gd]

        p3l,vm=backproject(gp,dp,zmin=1.,zmax=80.)
        n3=int(vm.sum())

        if n3<8:
            if recR:
                R_cw=recR[-1]@R_cw
                t_cw=t_cw+rec_disp[-1]
            Rlist.append(R_cw.copy()); tlist.append(t_cw.copy())
            dp=compute_depth(Lc,Rc); pp=detect(Lc); Lp=Lc
            ftimes.append(time.time()-ft); continue

        p3v=p3l[vm]; p2v=gc[vm]
        p3w=(R_cw@p3v.T+t_cw[:,None]).T  # world frame 3D

        ok_pnp=False
        try:
            ret,rv_wc,tv_wc,inl=cv2.solvePnPRansac(
                p3w.astype(np.float32),p2v.astype(np.float32),K,None,
                iterationsCount=200,reprojectionError=3.,confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP)
            if ret and inl is not None and len(inl)>=8:
                inl=inl.ravel()
                _,rv_wc,tv_wc=cv2.solvePnP(
                    p3w[inl].astype(np.float32),p2v[inl].astype(np.float32),
                    K,None,rv_wc,tv_wc,useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                Rwc=r2m(rv_wc); twc=tv_wc.ravel()
                Rn=Rwc.T; tn=-Rwc.T@twc

                relR=Rn@R_cw.T
                world_disp=tn-t_cw  # ACTUAL world displacement (fixed bug)

                if motion_ok(relR,world_disp):
                    R_cw,t_cw=Rn,tn
                    recR.append(relR.copy()); rec_disp.append(world_disp.copy())
                    if len(recR)>VWIN: recR.pop(0); rec_disp.pop(0)
                    ok_pnp=True
                else:
                    n_cap+=1
                    # Full constant velocity fallback (don't trust bad PnP rotation)
                    if recR:
                        R_cw=recR[-1]@R_cw   # use last VALID rotation
                        t_cw=t_cw+rec_disp[-1]  # constant velocity translation
                        ok_pnp=True
        except Exception: pass

        if not ok_pnp:
            if recR:
                R_cw=recR[-1]@R_cw
                t_cw=t_cw+rec_disp[-1]

        Rlist.append(R_cw.copy()); tlist.append(t_cw.copy())

        dc=compute_depth(Lc,Rc); dp=dc
        if ntk<150:
            npts=detect(Lc)
            pp=np.vstack([gc,npts]) if ntk>30 and len(npts)>0 else npts
        else: pp=gc
        Lp=Lc

        # ── KF + BA ──────────────────────────────────────────────
        if i%KF_STEP==0:
            kfp=detect(Lc)
            p3k,vmk=backproject(kfp,dc,zmin=10.,zmax=80.)
            p2kv=kfp[vmk]; p3kv=p3k[vmk]

            cross_list=[]
            for h_img,h_p2,h_wp,h_fid in zip(hist_imgs,hist_p2,hist_p3w,hist_fids):
                if h_img is None or len(h_p2)<10: continue
                tr,trg=track_lk(h_img,Lc,h_p2)
                if trg.sum()<5: continue
                ki_prev=ba.get_frame_idx(h_fid)
                if ki_prev<0: continue
                cross_list.append((h_wp[trg],h_p2[trg],tr[trg],ki_prev))

            if len(p3kv)>=5:
                ba.add_kf(i,R_cw,t_cw,p3kv,p2kv,cross_list if cross_list else None)

            if len(ba.poses)>=3:
                ok_ba=ba.run_ba(max_iter=20,huber=2.)
                if ok_ba:
                    # --- Retroactive correction: update ALL KF frames in BA window ---
                    corrs={}; Rcorrs_ba={}
                    for bk,bfidx in enumerate(ba.frames):
                        rv_k,tv_k=ba.poses[bk]
                        t_k=tv_k.copy(); R_k=r2m(rv_k)
                        if 0<=bfidx<len(Rlist) and np.linalg.norm(t_k-tlist[bfidx])<2.5:
                            corrs[bfidx]=t_k-tlist[bfidx]
                            Rcorrs_ba[bfidx]=R_k
                    if corrs:
                        # Apply corrections to KF frames
                        for bfidx,corr in corrs.items():
                            tlist[bfidx]+=corr
                            Rlist[bfidx]=Rcorrs_ba[bfidx]
                        # Linearly interpolate corrections to non-KF frames
                        kf_s=sorted(corrs.keys())
                        for bk2 in range(len(kf_s)-1):
                            f0,f1=kf_s[bk2],kf_s[bk2+1]
                            c0,c1=corrs[f0],corrs[f1]
                            for jj in range(f0+1,f1):
                                a=(jj-f0)/(f1-f0)
                                tlist[jj]+=(1-a)*c0+a*c1
                        # Update current tracking state from BA
                        if i in corrs:
                            R_cw,t_cw=Rcorrs_ba[i],tlist[i].copy()

            # Append to hist AFTER BA so future cross-frame obs use corrected pose
            hist_imgs.append(Lc.copy()); hist_p2.append(p2kv.copy())
            hist_p3w.append((R_cw@p3kv.T+t_cw[:,None]).T); hist_fids.append(i)
            if len(hist_imgs)>HISTORY:
                hist_imgs.pop(0); hist_p2.pop(0)
                hist_p3w.pop(0); hist_fids.pop(0)

        ftimes.append(time.time()-ft)
        if i%50==0:
            print(f"F{i:03d}/{N_FRAMES}: t=({t_cw[0]:.1f},{t_cw[1]:.1f},{t_cw[2]:.1f})"
                  f" trk={ntk} 3d={n3} cap={n_cap} avg={np.mean(ftimes[-50:])*1000:.0f}ms")

    print(f"Total {time.time()-t_start:.1f}s, avg {np.mean(ftimes)*1000:.0f}ms/fr, {n_cap} capped")
    return Rlist,tlist

def main():
    Rl,tl=run_vo()
    with open(os.path.join(LAB_ARTIFACTS,'traj.txt'),'w') as f:
        for t in tl: f.write(f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")
    with open(os.path.join(LAB_ARTIFACTS,'poses.txt'),'w') as f:
        for R,t in zip(Rl,tl):
            row=np.hstack([R,t.reshape(3,1)]).ravel()
            f.write(' '.join(f'{x:.6e}' for x in row)+'\n')
    print(f"Wrote {len(tl)} frames. Done.")

if __name__=='__main__':
    main()
