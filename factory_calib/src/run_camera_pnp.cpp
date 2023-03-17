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
    std::vector<double> &intrinsic, std::vector<double> &dist,
    std::vector<float> &rvec, std::vector<float> &tvec) {

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

  std::sort(detections.begin(), detections.end(), AprilTags::TagDetection::sortByIdCompare);
  std::vector<std::vector<float>> pts2d;
  int count = 0;
  for (size_t i = 0; i < detections.size(); ++i) {
    for (size_t j = 0; j < 4; j++) {
      std::vector<float> point = {detections[i].p[j].first, detections[i].p[j].second};
      //std::cout << detections[i].id << ": " << point[0] << " " << point[1] << std::endl;
      //std::cout << obj_pts[count][0] << " " << obj_pts[count][1] << std::endl;
      ++count;
      pts2d.emplace_back(std::move(point));
    }
  }


  solveCamPnP(obj_pts, pts2d, intrinsic, dist, rvec, tvec);
  //std::cout << "cam_id=" << cam_id << " frame_id=" << frame_id << std::endl;
  //std::cout << "[rvec]\n";
  for (size_t i = 0; i < rvec.size(); ++i) {
  //  std::cout << rvec[i] << " ";
  }
  //std::cout << "\n[tvec]\n";
  for (size_t i = 0; i < tvec.size(); ++i) {
  //  std::cout << tvec[i] << " ";
  }
  //std::cout << "\n\n";
  return 0;
}

void rottra2extrin(const std::vector<float> &rot,
    const std::vector<float> &tra,
    Eigen::Matrix<float, 4, 4> &extrin) {
  float theta=sqrt(rot[0]*rot[0]+rot[1]*rot[1]+rot[2]*rot[2]);
  float rx=rot[0]/theta;
  float ry=rot[1]/theta;
  float rz=rot[2]/theta;
  extrin(0,0)=0.0;
  extrin(0,1)=sin(theta)*(-rz);
  extrin(0,2)=sin(theta)*(ry);
  extrin(0,3)=tra[0];
  extrin(1,0)=sin(theta)*(rz);
  extrin(1,1)=0.0;
  extrin(1,2)=sin(theta)*(-rx);
  extrin(1,3)=tra[1];
  extrin(2,0)=sin(theta)*(-ry);
  extrin(2,1)=sin(theta)*(rx);
  extrin(2,2)=0.0;
  extrin(2,3)=tra[2];
  extrin(3,0)=0.0;
  extrin(3,1)=0.0;
  extrin(3,2)=0.0;
  extrin(3,3)=1.0;
  Eigen::Matrix<float,3,1> r;
  r(0,0)=rx;
  r(1,0)=ry;
  r(2,0)=rz;
  Eigen::Matrix<float,3,3> rr=(1.0-cos(theta))*r*r.transpose();
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      extrin(i,j)+=rr(i,j);
  for(int i=0;i<3;i++)
    extrin(i,i)+=cos(theta); 
}

int main(int argc, char **argv) {
  if (2 != argc) {
    std::cout << "error arg num: " << argc << std::endl;
    return -1;
  }

  size_t tag_rows = 6;
  size_t tag_cols = 6;
  float tag_size = 0.088;
  float tag_spacing = 0.3;
  std::vector<std::vector<std::vector<float>>> obj_pts(tag_rows * tag_cols);
  for (size_t r = 0; r < tag_rows * 2; ++r) {
    for (size_t c = 0; c < tag_cols * 2; ++c) {
      std::vector<float> point(3);
      point[0] = (int) (c / 2) * (1 + tag_spacing) * tag_size
          + (c % 2) * tag_size;
      point[1] = (int) (r / 2) * (1 + tag_spacing) * tag_size
          + (r % 2) * tag_size;
      point[2] = 0.0;
      int tag_id = (int) (r / 2) * tag_cols + (int) (c / 2);
      // std::cout << "tag_id=" << tag_id << std::endl;
      // std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;
      obj_pts[tag_id].emplace_back(std::move(point));
    }
  }
  std::vector<std::vector<float>> obj_pts_flat;
  for (auto &&pts : obj_pts) {
    obj_pts_flat.emplace_back(std::move(pts[0]));
    obj_pts_flat.emplace_back(std::move(pts[1]));
    obj_pts_flat.emplace_back(std::move(pts[3]));
    obj_pts_flat.emplace_back(std::move(pts[2]));
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
 
  std::string frame_id = argv[1];
  std::vector<float> rvec0 = {0, 3.14, 0};
  std::vector<float> tvec0 = {0, 0, 1.2};
  std::vector<float> rvec1 = {0, 3.14, 0};
  std::vector<float> tvec1 = {0, 0, 1.2};
  if (0 != solveSingle(jpg_prefix, "0", frame_id, obj_pts_flat, cam0_intrinsic, cam0_dist, rvec0, tvec0)
    || 0 != solveSingle(jpg_prefix, "1", frame_id, obj_pts_flat, cam1_intrinsic, cam1_dist, rvec1, tvec1)) {
    return -1;
  }
  Eigen::Matrix<float, 4, 4> pose0;
  Eigen::Matrix<float, 4, 4> pose1;
  rottra2extrin(rvec0, tvec0, pose0);  
  rottra2extrin(rvec1, tvec1, pose1);
  
  Eigen::Matrix<float, 4, 4> delta = pose0 * pose1.inverse();
  Eigen::Matrix3f rot_mat = delta.topLeftCorner(3, 3);
  Eigen::Vector3f euler_angles = rot_mat.eulerAngles(2, 1, 0);
  
  std::ofstream fout("./log-" + frame_id + ".txt");
  fout << "pose0 = \n" << pose0 << std::endl;
  fout << "pose1 = \n" << pose1 << std::endl;
  fout << "yaw(z) pitch(y) roll(x) = " << euler_angles.transpose() << std::endl;
  fout << "translation = " << delta.topRightCorner(3, 1).transpose() << std::endl; 
  fout.close();

  return 0;
}
