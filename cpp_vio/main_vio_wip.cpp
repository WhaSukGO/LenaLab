// Phase-2 C++ stereo Visual-INERTIAL Odometry: Ceres sliding-window factor graph fusing stereo
// reprojection + single-step IMU factors to BRIDGE vision blackouts.
//   pose param stays WORLD->CAM (Snavely; well-conditioned for BA — Phase-1, unchanged); per keyframe
//   also: world velocity + IMU bias (bg,ba). The IMU cost converts world->cam -> body(=cam)->world
//   internally. gravity KNOWN (y-down +9.81). Blackout = no texture -> no visual factors; the IMU
//   factor + carried velocity propagate the state. If imu.txt is absent -> VO-only (== Phase 1).
// usage: vio <seq_input_dir> <out_dir> <seq_name>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace fs = std::filesystem;
using std::vector;

static const double GRAV[3] = {0.0, 9.81, 0.0};
static const double W_ROT = 100.0, W_VEL = 10.0, W_POS = 60.0, W_BIAS = 500.0;
static const double DT = 0.1;

struct ReprojCost {                              // world->cam pose: X_cam = R(aa) X + t (Snavely)
  ReprojCost(double ox, double oy, double fx, double fy, double cx, double cy)
      : ox(ox), oy(oy), fx(fx), fy(fy), cx(cx), cy(cy) {}
  template <typename T> bool operator()(const T* cam, const T* X, T* r) const {
    T Xc[3]; ceres::AngleAxisRotatePoint(cam, X, Xc);
    Xc[0] += cam[3]; Xc[1] += cam[4]; Xc[2] += cam[5];
    if (Xc[2] < T(1e-3)) return false;
    r[0] = T(fx) * Xc[0] / Xc[2] + T(cx) - T(ox);
    r[1] = T(fy) * Xc[1] / Xc[2] + T(cy) - T(oy);
    return true;
  }
  static ceres::CostFunction* Create(double ox, double oy, double fx, double fy, double cx, double cy) {
    return new ceres::AutoDiffCostFunction<ReprojCost, 2, 6, 3>(new ReprojCost(ox, oy, fx, fy, cx, cy));
  }
  double ox, oy, fx, fy, cx, cy;
};

// single-step IMU factor; cam params are WORLD->CAM, converted to body->world inside (R_wb=R(-aa),
// p_wb = -R_wb t). v_* are world velocities, b_* = (bg,ba).
struct ImuCost {
  ImuCost(const double* w, const double* a) { for (int k = 0; k < 3; ++k) { wm[k] = w[k]; am[k] = a[k]; } }
  template <typename T>
  static void to_body(const T* cam, T* naa, T* p) {           // naa=-aa (so R(naa)=R_wb); p=p_wb
    for (int k = 0; k < 3; ++k) naa[k] = -cam[k];
    T t[3] = {cam[3], cam[4], cam[5]}, Rt[3];
    ceres::AngleAxisRotatePoint(naa, t, Rt); p[0] = -Rt[0]; p[1] = -Rt[1]; p[2] = -Rt[2];
  }
  template <typename T>
  bool operator()(const T* ci, const T* vi, const T* bi, const T* cj, const T* vj, const T* bj, T* r) const {
    const T d(DT);
    T nai[3], pi[3], naj[3], pj[3]; to_body(ci, nai, pi); to_body(cj, naj, pj);
    T ac[3] = {T(am[0]) - bi[3], T(am[1]) - bi[4], T(am[2]) - bi[5]};
    T aw[3]; ceres::AngleAxisRotatePoint(nai, ac, aw);        // R_wb_i * (accel - ba)
    aw[0] += T(GRAV[0]); aw[1] += T(GRAV[1]); aw[2] += T(GRAV[2]);
    T pv[3], pp[3];
    for (int k = 0; k < 3; ++k) { pv[k] = vi[k] + aw[k] * d; pp[k] = pi[k] + vi[k] * d + T(0.5) * aw[k] * d * d; }
    // rotation: R_wb_j vs R_wb_i Exp((gyro-bg) dt), via quaternions
    T wc[3] = {(T(wm[0]) - bi[0]) * d, (T(wm[1]) - bi[1]) * d, (T(wm[2]) - bi[2]) * d};
    T qi[4], qd[4], qpred[4], qj[4];
    ceres::AngleAxisToQuaternion(nai, qi); ceres::AngleAxisToQuaternion(wc, qd);
    ceres::QuaternionProduct(qi, qd, qpred); ceres::AngleAxisToQuaternion(naj, qj);
    T qpi[4] = {qpred[0], -qpred[1], -qpred[2], -qpred[3]}, qe[4];
    ceres::QuaternionProduct(qpi, qj, qe);
    r[0] = T(2 * W_ROT) * qe[1]; r[1] = T(2 * W_ROT) * qe[2]; r[2] = T(2 * W_ROT) * qe[3];
    for (int k = 0; k < 3; ++k) {
      r[3 + k] = T(W_VEL) * (vj[k] - pv[k]);
      r[6 + k] = T(W_POS) * (pj[k] - pp[k]);
      r[9 + k] = T(W_BIAS) * (bj[k] - bi[k]);
      r[12 + k] = T(W_BIAS) * (bj[3 + k] - bi[3 + k]);
    }
    return true;
  }
  static ceres::CostFunction* Create(const double* w, const double* a) {
    return new ceres::AutoDiffCostFunction<ImuCost, 15, 6, 3, 6, 6, 3, 6>(new ImuCost(w, a));
  }
  double wm[3], am[3];
};

struct Landmark { cv::Point3d X; int n_obs = 0; };
struct KF { int frame; double cam[6]; double vel[3]; double bias[6]; std::map<int, cv::Point2d> obs;
            bool blackout = false; };

int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: vio <seq_input_dir> <out_dir> <seq_name>\n"; return 2; }
  const fs::path in = argv[1], out = argv[2]; const std::string seq = argv[3];
  std::ifstream fi((in / "intrinsics.txt").string());
  double fx, fy, cx, cy, baseline; fi >> fx >> fy >> cx >> cy >> baseline;
  cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  int n = 0;
  for (const auto& e : fs::directory_iterator(in))
    if (e.path().filename().string().rfind("left_", 0) == 0) ++n;
  vector<std::array<double, 6>> imu;
  { std::ifstream im((in / "imu.txt").string()); std::array<double, 6> row;
    while (im >> row[0] >> row[1] >> row[2] >> row[3] >> row[4] >> row[5]) imu.push_back(row); }
  const bool VIO = (int)imu.size() >= n - 1 && n > 2;
  std::cerr << "seq " << seq << ": n=" << n << (VIO ? " VIO" : " VO-only") << "\n";

  auto sgbm = cv::StereoSGBM::create(0, 128, 7, 8 * 7 * 7, 32 * 7 * 7, 10, 0, 0, 100, 2);
  auto depth_of = [&](const cv::Mat& L, const cv::Mat& R) {
    cv::Mat disp; sgbm->compute(L, R, disp); disp.convertTo(disp, CV_32F, 1.0 / 16.0);
    cv::Mat z(disp.size(), CV_32F);
    for (int y = 0; y < disp.rows; ++y) for (int x = 0; x < disp.cols; ++x) {
      float dd = disp.at<float>(y, x); z.at<float>(y, x) = dd > 0.5f ? float(fx * baseline / dd) : 0.f;
    } return z;
  };
  auto load = [&](int i, const char* lr) {
    char b[64]; std::snprintf(b, sizeof b, "%s_%06d.png", lr, i);
    return cv::imread((in / b).string(), cv::IMREAD_GRAYSCALE);
  };

  std::map<int, Landmark> lms; int next_lm = 0;
  std::deque<KF> window; const int W = VIO ? 12 : 8;
  vector<std::array<double, 12>> poses_out(n);
  auto cam_to_world = [&](const double* cam, cv::Mat& Rwc, cv::Mat& twc) {     // world->cam -> cam->world
    cv::Mat rv = (cv::Mat_<double>(3, 1) << cam[0], cam[1], cam[2]), Rcw; cv::Rodrigues(rv, Rcw);
    cv::Mat tcw = (cv::Mat_<double>(3, 1) << cam[3], cam[4], cam[5]); Rwc = Rcw.t(); twc = -Rwc * tcw;
  };
  auto write_pose = [&](int frame, const double* cam) {
    cv::Mat Rwc, twc; cam_to_world(cam, Rwc, twc); auto& p = poses_out[frame];
    for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) p[r * 4 + c] = Rwc.at<double>(r, c); p[r * 4 + 3] = twc.at<double>(r); }
  };

  cv::Mat prevL; vector<cv::Point2f> prev_pts; vector<int> prev_ids;

  for (int i = 0; i < n; ++i) {
    cv::Mat L = load(i, "left"), Rr = load(i, "right");
    cv::Mat z = depth_of(L, Rr);
    KF kf; kf.frame = i;
    for (int k = 0; k < 3; ++k) kf.vel[k] = 0;
    for (int k = 0; k < 6; ++k) kf.bias[k] = 0;
    vector<cv::Point2f> corners; cv::goodFeaturesToTrack(L, corners, 600, 0.01, 12);
    const bool texture = !VIO || (int)corners.size() >= 30;   // blackout detection only in VIO mode
    kf.blackout = !texture;

    if (i == 0) {
      for (int k = 0; k < 6; ++k) kf.cam[k] = 0.0;
    } else {
      KF& prev = window.back();
      // IMU-propagate the previous state -> predicted WORLD->CAM pose + velocity (blackout fallback + guess)
      double pred[6]; for (int k = 0; k < 6; ++k) pred[k] = prev.cam[k];
      if (VIO) {
        const double* w = imu[i - 1].data(); const double* a = imu[i - 1].data() + 3;
        cv::Mat Rwc, twc; cam_to_world(prev.cam, Rwc, twc);                 // R_wb=Rwc, p_wb=twc
        cv::Mat ac = (cv::Mat_<double>(3, 1) << a[0] - prev.bias[3], a[1] - prev.bias[4], a[2] - prev.bias[5]);
        cv::Mat aw = Rwc * ac + (cv::Mat_<double>(3, 1) << GRAV[0], GRAV[1], GRAV[2]);
        cv::Mat pj = twc.clone();
        for (int k = 0; k < 3; ++k) { pj.at<double>(k) = twc.at<double>(k) + prev.vel[k] * DT + 0.5 * aw.at<double>(k) * DT * DT;
                                      kf.vel[k] = prev.vel[k] + aw.at<double>(k) * DT; }
        cv::Mat dth = (cv::Mat_<double>(3, 1) << (w[0] - prev.bias[0]) * DT, (w[1] - prev.bias[1]) * DT, (w[2] - prev.bias[2]) * DT), Rd; cv::Rodrigues(dth, Rd);
        cv::Mat Rwb = Rwc * Rd, Rcw = Rwb.t(), aa; cv::Rodrigues(Rcw, aa);
        cv::Mat tcw = -Rcw * pj;
        for (int k = 0; k < 3; ++k) { pred[k] = aa.at<double>(k); pred[3 + k] = tcw.at<double>(k); }
        for (int k = 0; k < 6; ++k) kf.bias[k] = prev.bias[k];
      }
      vector<cv::Point2f> cur; vector<uchar> st; vector<float> er;
      if (!prev_pts.empty()) cv::calcOpticalFlowPyrLK(prevL, L, prev_pts, cur, st, er, cv::Size(21, 21), 3);
      vector<cv::Point3f> objp; vector<cv::Point2f> imgp; vector<int> tids; vector<cv::Point2f> tpts;
      for (size_t j = 0; j < st.size(); ++j) {
        if (!st[j]) continue; int id = prev_ids[j]; auto it = lms.find(id);
        if (it == lms.end() || cur[j].x < 0 || cur[j].y < 0 || cur[j].x >= L.cols || cur[j].y >= L.rows) continue;
        objp.push_back(cv::Point3f(it->second.X.x, it->second.X.y, it->second.X.z));
        imgp.push_back(cur[j]); tids.push_back(id); tpts.push_back(cur[j]);
      }
      cv::Mat rvec = (cv::Mat_<double>(3, 1) << pred[0], pred[1], pred[2]);
      cv::Mat tvec = (cv::Mat_<double>(3, 1) << pred[3], pred[4], pred[5]);
      vector<int> inl; bool ok = false;
      if (texture && (int)objp.size() >= 6)
        ok = cv::solvePnPRansac(objp, imgp, K, cv::noArray(), rvec, tvec, true, 150, 2.0, 0.99, inl);
      if (ok) {
        for (int k = 0; k < 3; ++k) { kf.cam[k] = rvec.at<double>(k); kf.cam[3 + k] = tvec.at<double>(k); }
        std::vector<char> isin(tids.size(), 0); for (int idx : inl) isin[idx] = 1;
        for (size_t j = 0; j < tids.size(); ++j)
          if (isin[j]) { kf.obs[tids[j]] = tpts[j]; lms[tids[j]].n_obs++; }
      } else { for (int k = 0; k < 6; ++k) kf.cam[k] = pred[k]; }     // ride IMU (or hold) pose
      // loosely-coupled velocity: finite-diff of consecutive camera centres (world frame)
      cv::Mat Rwc, twc, Rp, tp; cam_to_world(kf.cam, Rwc, twc); cam_to_world(window.back().cam, Rp, tp);
      for (int k = 0; k < 3; ++k) kf.vel[k] = (twc.at<double>(k) - tp.at<double>(k)) / DT;
    }

    // replenish landmarks wherever there is texture (even if PnP failed -> recover tracking)
    vector<cv::Point2f> next_pts; vector<int> next_ids;
    if (texture) {
      cv::Mat Rwc, twc; cam_to_world(kf.cam, Rwc, twc);
      for (auto& c : corners) {
        float zz = z.at<float>(cvRound(c.y), cvRound(c.x));
        if (zz <= 0.5f || zz > 50.f) continue;
        cv::Mat Xc = (cv::Mat_<double>(3, 1) << (c.x - cx) * zz / fx, (c.y - cy) * zz / fy, zz);
        cv::Mat Xw = Rwc * Xc + twc;
        int id = next_lm++; lms[id] = {cv::Point3d(Xw.at<double>(0), Xw.at<double>(1), Xw.at<double>(2)), 1};
        kf.obs[id] = cv::Point2d(c.x, c.y); next_pts.push_back(c); next_ids.push_back(id);
      }
      for (auto& [id, uv] : kf.obs) { next_pts.push_back(cv::Point2f(uv.x, uv.y)); next_ids.push_back(id); }
    }

    window.push_back(kf);

    // ---- sliding-window factor graph (reprojection + IMU) ----
    {
      ceres::Problem problem;
      std::map<int, std::array<double, 3>> Xparam;
      for (auto& wk : window)
        for (auto& [id, uv] : wk.obs)
          if (lms[id].n_obs >= 2 && !Xparam.count(id)) Xparam[id] = {lms[id].X.x, lms[id].X.y, lms[id].X.z};
      for (auto& wk : window)
        for (auto& [id, uv] : wk.obs)
          if (Xparam.count(id))
            problem.AddResidualBlock(ReprojCost::Create(uv.x, uv.y, fx, fy, cx, cy),
                                     new ceres::HuberLoss(2.0), wk.cam, Xparam[id].data());
      // LOOSELY-COUPLED (default): IMU bridges blackouts via dead-reckoning (pose prediction above),
      // not as a BA factor. The tightly-coupled ImuCost below diverged on vio1 (1e45) and needs more
      // work (single-step factor + velocity gauge); kept off behind TIGHT_IMU for a future iteration.
      const bool TIGHT_IMU = false;
      if (VIO && TIGHT_IMU)
        for (size_t k = 1; k < window.size(); ++k) {
          KF& A = window[k - 1]; KF& B = window[k];
          if (B.frame == A.frame + 1)
            problem.AddResidualBlock(ImuCost::Create(imu[A.frame].data(), imu[A.frame].data() + 3),
                                     nullptr, A.cam, A.vel, A.bias, B.cam, B.vel, B.bias);
        }
      if (problem.NumResidualBlocks() > 0) {
        if (problem.HasParameterBlock(window.front().cam)) problem.SetParameterBlockConstant(window.front().cam);
        ceres::Solver::Options o;
        o.linear_solver_type = (VIO && TIGHT_IMU) ? ceres::SPARSE_NORMAL_CHOLESKY : ceres::DENSE_SCHUR;
        o.max_num_iterations = 10; o.logging_type = ceres::SILENT; o.num_threads = 2;
        ceres::Solver::Summary s; ceres::Solve(o, &problem, &s);
        for (auto& [id, X] : Xparam) lms[id].X = cv::Point3d(X[0], X[1], X[2]);
      }
    }

    if ((int)window.size() > W) { KF old = window.front(); window.pop_front(); write_pose(old.frame, old.cam); }
    prevL = L; prev_pts = next_pts; prev_ids = next_ids;
  }
  for (auto& w : window) write_pose(w.frame, w.cam);

  fs::create_directories(out);
  std::ofstream pf((out / ("poses_" + seq + ".txt")).string()), tf((out / ("traj_" + seq + ".txt")).string());
  for (int i = 0; i < n; ++i) {
    auto& p = poses_out[i];
    for (int k = 0; k < 12; ++k) pf << p[k] << (k == 11 ? '\n' : ' ');
    tf << p[3] << ' ' << p[7] << ' ' << p[11] << '\n';
  }
  std::cout << "seq " << seq << ": " << n << " frames, " << next_lm << " landmarks, " << (VIO ? "VIO" : "VO") << " done\n";
  return 0;
}
