// Phase-1 C++ stereo Visual Odometry + Ceres sliding-window reprojection Bundle Adjustment.
//   front-end (OpenCV): SGBM stereo depth, goodFeaturesToTrack + KLT tracking, stereo-triangulated
//                       landmarks, PnP-RANSAC pose init.
//   backend  (Ceres):   sliding window of the last W keyframes; optimise camera poses (world->cam
//                       angle-axis+t, Snavely convention) + landmark world positions against LEFT
//                       reprojection residuals (Huber); the oldest pose in the window is fixed (gauge).
// Output: KITTI poses_<seq>.txt (3x4 cam->world, 12/line) + traj_<seq>.txt (camera centres).
// Phase 2 will add IMU-preintegration factors to this same graph to bridge vision blackouts.
//
// usage: vio <seq_input_dir> <out_dir> <seq_name>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>          // calcOpticalFlowPyrLK
#include <opencv2/imgproc.hpp>

#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace fs = std::filesystem;
using std::vector;

struct ReprojCost {                              // left-camera reprojection (2-D residual)
  ReprojCost(double ox, double oy, double fx, double fy, double cx, double cy)
      : ox(ox), oy(oy), fx(fx), fy(fy), cx(cx), cy(cy) {}
  template <typename T> bool operator()(const T* cam, const T* X, T* r) const {
    T Xc[3];
    ceres::AngleAxisRotatePoint(cam, X, Xc);     // cam[0..2] = angle-axis (world->cam)
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

struct Landmark { cv::Point3d X; int n_obs = 0; };          // world position
struct KF { int frame; double cam[6]; std::map<int, cv::Point2d> obs; };  // pose + {lm_id: uv}

int main(int argc, char** argv) {
  if (argc < 4) { std::cerr << "usage: vio <seq_input_dir> <out_dir> <seq_name>\n"; return 2; }
  const fs::path in = argv[1], out = argv[2];
  const std::string seq = argv[3];

  std::ifstream fi((in / "intrinsics.txt").string());
  double fx, fy, cx, cy, baseline; fi >> fx >> fy >> cx >> cy >> baseline;
  cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  int n = 0;
  for (const auto& e : fs::directory_iterator(in))
    if (e.path().filename().string().rfind("left_", 0) == 0) ++n;

  auto sgbm = cv::StereoSGBM::create(0, 128, 7, 8 * 7 * 7, 32 * 7 * 7, 10, 0, 0, 100, 2);
  auto depth_of = [&](const cv::Mat& L, const cv::Mat& R) {
    cv::Mat disp; sgbm->compute(L, R, disp); disp.convertTo(disp, CV_32F, 1.0 / 16.0);
    cv::Mat z(disp.size(), CV_32F);
    for (int y = 0; y < disp.rows; ++y) for (int x = 0; x < disp.cols; ++x) {
      float d = disp.at<float>(y, x); z.at<float>(y, x) = d > 0.5f ? float(fx * baseline / d) : 0.f;
    } return z;
  };

  std::map<int, Landmark> lms; int next_lm = 0;
  std::deque<KF> window; const int W = 8;
  vector<std::array<double, 12>> poses_out(n);   // cam->world 3x4 per frame
  auto load = [&](int i, const char* lr) {
    char b[64]; std::snprintf(b, sizeof b, "%s_%06d.png", lr, i);
    return cv::imread((in / b).string(), cv::IMREAD_GRAYSCALE);
  };

  auto write_pose = [&](int frame, const double* cam) {            // world->cam -> store cam->world
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << cam[0], cam[1], cam[2]), Rcw; cv::Rodrigues(rvec, Rcw);
    cv::Mat tcw = (cv::Mat_<double>(3, 1) << cam[3], cam[4], cam[5]);
    cv::Mat Rwc = Rcw.t(), twc = -Rwc * tcw;
    auto& p = poses_out[frame];
    for (int r = 0; r < 3; ++r) { for (int c = 0; c < 3; ++c) p[r * 4 + c] = Rwc.at<double>(r, c); p[r * 4 + 3] = twc.at<double>(r); }
  };

  cv::Mat prevL; vector<cv::Point2f> prev_pts; vector<int> prev_ids;

  for (int i = 0; i < n; ++i) {
    cv::Mat L = load(i, "left"), Rr = load(i, "right");
    cv::Mat z = depth_of(L, Rr);
    KF kf; kf.frame = i;

    if (i == 0) {                                                  // anchor: identity pose
      for (int k = 0; k < 6; ++k) kf.cam[k] = 0.0;
    } else {                                                       // track + PnP
      vector<cv::Point2f> cur; vector<uchar> st; vector<float> er;
      cv::calcOpticalFlowPyrLK(prevL, L, prev_pts, cur, st, er, cv::Size(21, 21), 3);
      vector<cv::Point3f> objp; vector<cv::Point2f> imgp; vector<int> tracked_ids; vector<cv::Point2f> tracked_pts;
      for (size_t j = 0; j < st.size(); ++j) {
        if (!st[j]) continue;
        int id = prev_ids[j]; auto it = lms.find(id); if (it == lms.end()) continue;
        if (cur[j].x < 0 || cur[j].y < 0 || cur[j].x >= L.cols || cur[j].y >= L.rows) continue;
        objp.push_back(cv::Point3f(it->second.X.x, it->second.X.y, it->second.X.z));
        imgp.push_back(cur[j]); tracked_ids.push_back(id); tracked_pts.push_back(cur[j]);
      }
      double* prevcam = window.back().cam;
      cv::Mat rvec = (cv::Mat_<double>(3, 1) << prevcam[0], prevcam[1], prevcam[2]);
      cv::Mat tvec = (cv::Mat_<double>(3, 1) << prevcam[3], prevcam[4], prevcam[5]);
      vector<int> inl;
      if (objp.size() >= 6 &&
          cv::solvePnPRansac(objp, imgp, K, cv::noArray(), rvec, tvec, true, 150, 2.0, 0.99, inl)) {
        for (int k = 0; k < 3; ++k) { kf.cam[k] = rvec.at<double>(k); kf.cam[3 + k] = tvec.at<double>(k); }
        std::vector<char> is_inl(tracked_ids.size(), 0); for (int idx : inl) is_inl[idx] = 1;
        for (size_t j = 0; j < tracked_ids.size(); ++j)
          if (is_inl[j]) { kf.obs[tracked_ids[j]] = tracked_pts[j]; lms[tracked_ids[j]].n_obs++; }
      } else { for (int k = 0; k < 6; ++k) kf.cam[k] = prevcam[k]; }   // fallback: hold pose
    }

    // triangulate NEW features from stereo to replenish the map (in world frame via this pose)
    cv::Mat Rcw; { cv::Mat rv = (cv::Mat_<double>(3, 1) << kf.cam[0], kf.cam[1], kf.cam[2]); cv::Rodrigues(rv, Rcw); }
    cv::Mat tcw = (cv::Mat_<double>(3, 1) << kf.cam[3], kf.cam[4], kf.cam[5]);
    cv::Mat Rwc = Rcw.t();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(L, corners, 600, 0.01, 12);
    vector<cv::Point2f> next_pts; vector<int> next_ids;
    for (auto& c : corners) {
      float zz = z.at<float>(cvRound(c.y), cvRound(c.x));
      if (zz <= 0.5f || zz > 50.f) continue;
      cv::Mat Xc = (cv::Mat_<double>(3, 1) << (c.x - cx) * zz / fx, (c.y - cy) * zz / fy, zz);
      cv::Mat Xw = Rwc * (Xc - tcw);
      int id = next_lm++; lms[id] = {cv::Point3d(Xw.at<double>(0), Xw.at<double>(1), Xw.at<double>(2)), 1};
      kf.obs[id] = cv::Point2d(c.x, c.y);
      next_pts.push_back(c); next_ids.push_back(id);
    }
    // carry tracked inliers forward too (so KLT keeps long tracks, not just fresh corners)
    for (auto& [id, uv] : kf.obs) { next_pts.push_back(cv::Point2f(uv.x, uv.y)); next_ids.push_back(id); }

    window.push_back(kf);

    // ---- sliding-window BA ----
    {
      ceres::Problem problem;
      std::map<int, std::array<double, 3>> Xparam;
      for (auto& w : window)
        for (auto& [id, uv] : w.obs)
          if (lms[id].n_obs >= 2 && !Xparam.count(id))
            Xparam[id] = {lms[id].X.x, lms[id].X.y, lms[id].X.z};
      for (auto& w : window) {
        for (auto& [id, uv] : w.obs) {
          if (!Xparam.count(id)) continue;
          problem.AddResidualBlock(ReprojCost::Create(uv.x, uv.y, fx, fy, cx, cy),
                                   new ceres::HuberLoss(2.0), w.cam, Xparam[id].data());
        }
      }
      if (problem.NumResidualBlocks() > 0) {
        problem.SetParameterBlockConstant(window.front().cam);     // gauge: fix oldest pose
        ceres::Solver::Options o; o.linear_solver_type = ceres::DENSE_SCHUR;
        o.max_num_iterations = 8; o.logging_type = ceres::SILENT; o.num_threads = 2;
        ceres::Solver::Summary s; ceres::Solve(o, &problem, &s);
        for (auto& [id, X] : Xparam) lms[id].X = cv::Point3d(X[0], X[1], X[2]);
      }
    }

    if ((int)window.size() > W) { KF old = window.front(); window.pop_front(); write_pose(old.frame, old.cam); }
    prevL = L; prev_pts = next_pts; prev_ids = next_ids;
  }
  for (auto& w : window) write_pose(w.frame, w.cam);                // flush remaining window

  fs::create_directories(out);
  std::ofstream pf((out / ("poses_" + seq + ".txt")).string()), tf((out / ("traj_" + seq + ".txt")).string());
  for (int i = 0; i < n; ++i) {
    auto& p = poses_out[i];
    for (int k = 0; k < 12; ++k) pf << p[k] << (k == 11 ? '\n' : ' ');
    tf << p[3] << ' ' << p[7] << ' ' << p[11] << '\n';
  }
  std::cout << "seq " << seq << ": " << n << " frames, " << next_lm << " landmarks, VO+BA done\n";
  return 0;
}
