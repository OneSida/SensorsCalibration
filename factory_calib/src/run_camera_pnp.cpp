#include "Eigen/Core"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

#include "dataConvert.hpp"
#include "logging.hpp"

#include "apriltags/Tag36h11.h"
#include "apriltags/TagDetector.h"
#include "calibration/pnp_solver.hpp"


int32_t solveSingle(const std::string &jpg_prefix,
    const std::string &cam_id, const std::string &frame_id,
    std::vector<std::vector<float>> &obj_pts,
    std::vector<double> &intrinsic, std::vector<double> &dist) {

  std::string image_path = jpg_prefix + cam_id + "/frame" + frame_id + ".jpg";
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  cv::Mat gray;
  cv::cvtColor(image, gray, CV_BGR2GRAY);

  AprilTags::TagCodes tag_codes(AprilTags::tagCodes36h11);
  AprilTags::TagDetector *tag_detector = new AprilTags::TagDetector(tag_codes);
  std::vector<AprilTags::TagDetection> detections = tag_detector->extractTags(gray);
  
  if (detections.size() * 4 != obj_pts.size()) {
    std::cout << image_path << " detections.size()=" << detections.size() << std::endl;
    return -1;
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    detections[i].draw(image);
  }
  cv::imwrite(jpg_prefix + cam_id + "/frame" + frame_id + "_det.jpg", image);

  std::vector<std::vector<float>> pts2d;
  for (size_t i = 0; i < detections.size(); ++i) {
    for (size_t j = 0; j < 4; j++) {
      std::vector<float> point = {detections[i].p[j].first, detections[i].p[j].second};
      pts2d.emplace_back(std::move(point));
    }
  }
  std::sort(pts2d.begin(), pts2d.end(),
    [](const std::vector<float> &p1, const std::vector<float> &p2) {
      return p1[0] == p2[0] ? p1[1] > p2[1] : p1[0] < p2[0];
    });

  std::vector<float> rvec(3);
  std::vector<float> tvec(3);
  solveCamPnP(obj_pts, pts2d, intrinsic, dist, rvec, tvec);
  std::cout << "\n[rvec]\n";
  for (size_t i = 0; i < rvec.size(); ++i) {
    std::cout << rvec[i] << " ";
  }
  std::cout << "\n[tvec]\n";
  for (size_t i = 0; i < tvec.size(); ++i) {
    std::cout << tvec[i] << " ";
  }
  return 0;
}

int main(int argc, char **argv) {
  size_t tag_rows = 6;
  size_t tag_cols = 6;
  float tag_size = 0.088;
  float tag_spacing = 0.3;
  std::vector<std::vector<float>> obj_pts;
  for (size_t r = 0; r < tag_rows * 2; ++r) {
    for (size_t c = 0; c < tag_cols * 2; ++c) {
      std::vector<float> point(3);
      point[0] = (int) (c / 2) * (1 + tag_spacing) * tag_size
          + (c % 2) * tag_size;
      point[1] = (int) (r / 2) * (1 + tag_spacing) * tag_size
          + (r % 2) * tag_size;
      point[2] = 0.0;
      obj_pts.emplace_back(std::move(point));
    }
  }

  std::string jpg_prefix = "/root/ros_workspace/kalibr_workspace/data/jpg";
  // camera_model: pinhole
  // intrinsics: [461.629, 460.152, 362.680, 246.049]
  // distortion_model: radtan
  // distortion_coeffs: [-0.27695497, 0.06712482, 0.00087538, 0.00011556]
  std::vector<double> cam0_intrinsic = {0.0, 461.629, 460.152, 362.680, 246.049};
  std::vector<double> cam0_dist = {-0.27695497, 0.06712482, 0.00087538, 0.00011556};
  // camera_model: omni
  // intrinsics: [0.80065662, 833.006, 830.345, 373.850, 253.749]
  // distortion_model: radtan
  // distortion_coeffs: [-0.33518750, 0.13211436, 0.00055967, 0.00057686]
  std::vector<double> cam1_intrinsic = {0.80065662, 833.006, 830.345, 373.850, 253.749};
  std::vector<double> cam1_dist = {-0.33518750, 0.13211436, 0.00055967, 0.00057686};
  
  std::string frame_id = "0000";
  if (0 != solveSingle(jpg_prefix, "0", frame_id, obj_pts, cam0_intrinsic, cam0_dist)) {
    return -1;
  }
  if (0 != solveSingle(jpg_prefix, "1", frame_id, obj_pts, cam1_intrinsic, cam1_dist)) {
    return -1;
  }
  
  return 0;
}
