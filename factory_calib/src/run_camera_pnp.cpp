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


int main(int argc, char **argv) {
  if (argc > 1) {
    std::cerr << argv[0];
    return -1;
  }

  // camera_model: pinhole
  // intrinsics: [461.629, 460.152, 362.680, 246.049]
  // distortion_model: radtan
  // distortion_coeffs: [-0.27695497, 0.06712482, 0.00087538, 0.00011556]
  std::vector<std::vector<double>> intrinsic = {
    {461.629, 0, 362.680},
    {0, 460.152, 246.049}
  };
  std::vector<double> dist = {-0.27695497, 0.06712482, 0.00087538, 0.00011556};

  // load image
  cv::Mat image = cv::imread("demo.jpg", cv::IMREAD_COLOR);
  cv::Mat gray;
  cv::cvtColor(image, gray, CV_BGR2GRAY);

  // detect apriltag
  AprilTags::TagCodes tag_codes(AprilTags::tagCodes36h11);
  AprilTags::TagDetector *tag_detector = new AprilTags::TagDetector(tag_codes);
  std::vector<AprilTags::TagDetection> detections = tag_detector->extractTags(gray);
  if (detections.size() == 36) {
    for (size_t i = 0; i < detections.size(); ++i) {
      detections[i].draw(image);
    }
    cv::imshow("apriltags", image);
    cv::waitKey(0);
    cv::imwrite("apriltags_detection.png", image);
  }

  // solve pose
  size_t tag_rows = 6;
  size_t tag_cols = 6;
  float tag_size = 0.088;
  float tag_spacing = 0.3;
  std::vector<std::vector<float>> obj_pts;
  for (size_t r = 0; r < tag_rows; ++r) {
    for (size_t c = 0; c < tag_cols; ++c) {
      std::vector<float> point(3);
      point[0] = (int) (c / 2) * (1 + tag_spacing) * tag_size
          + (c % 2) * tag_size;
      point[1] = (int) (r / 2) * (1 + tag_spacing) * tag_size
          + (r % 2) * tag_size;
      point[2] = 0.0;
      obj_pts.emplace_back(std::move(point));
    }
  }
  std::vector<std::vector<float>> pts2d;
  for (size_t i = 0; i < detections.size(); ++i) {
    for (size_t j = 0; j < 4; j++) {
      pts2d.emplace_back({detections[i].p[j].first, detections[i].p[j].second});
    }
  }
  std::vector<float> rvec, tvec;
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
