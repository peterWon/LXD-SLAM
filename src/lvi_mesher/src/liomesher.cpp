/* This code is the implementation of our paper "SLAMesh: Real-time LiDAR
 Simultaneous Localization and Meshing".

Author: Jianyuan Ruan, Email: <jianyuan.ruan@connect.polyu.hk>

If you find our research helpful, please cite our paper:
[1] Jianyuan Ruan, Bo Li, Yibo Wang, and Yuxiang Sun, "SLAMesh: Real-time
 LiDAR Simultaneous Localization and Meshing" ICRA 2023.

Other related papers:
[2] Jianyuan Ruan, Bo Li, Yinqiang Wang and Zhou Fang, "GP-SLAM+: real-time
 3D lidar SLAM based on improved regionalized Gaussian process map
 reconstruction," IROS 2020.
[3] Bo Li, Yinqiang Wang, Yu Zhang. Wenjie Zhao, Jianyuan Ruan, and Pin Li,
 "GP-SLAM: laser-based SLAM approach based on regionalized Gaussian process
 map reconstruction". Auton Robot 2020.

For commercial use, please contact Dr. Yuxiang SUN < yx.sun@polyu.edu.hk >.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/***
 * This file run a SLAMesh node.
 * Please use this abbreviation dictionary to help you understand our code:
 * param: parameter
 * g_data: global data
 * trf: transformation, 6dof
 * odom: odometry
 * dir: direction, the coordinate that serve as the prediction in gaussian
    process function, like z in z = f(x, y).
 * num: number
 * Idx: index
 * 3Dir: inside one cell, there can be 3 different gaussian process function
    to model complex local surfaces, they have different prediction directions,
    that is, x, y, or z. Functions without "3Dir" means they only allow one
    gaussian process function inside one cell, like paper [3].
 * glb: global map, means in the world frame
 * now: means the current scan
 * avg: average
 * thr: threshold
 * posi: position
 * ary: array
 *
    */
#include <future>
#include <thread>
#include <execution>
#include "liomesher.h"
#include "glog/logging.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/TransformStamped.h"
#include <pcl/filters/random_sample.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>

#include "local_ba.h"

// using namespace cv;
using namespace std;
namespace{
    bool eigenPoseToOpenCV(const Eigen::Matrix4d& eigen_pose, cv::Mat& rvec, cv::Mat& tvec) {
    // 检查输入矩阵是否为4x4
    if (eigen_pose.rows() != 4 || eigen_pose.cols() != 4) {
        std::cerr << "错误:输入矩阵必须是4x4齐次变换矩阵" << std::endl;
        return false;
    }
    
    try {
        // 提取旋转矩阵 (3x3)
        Eigen::Matrix3d R = eigen_pose.block<3, 3>(0, 0);
        
        // 提取平移向量 (3x1)
        Eigen::Vector3d t = eigen_pose.block<3, 1>(0, 3);
        
        // 将Eigen旋转矩阵转换为OpenCV的Mat
        cv::Mat R_cv;
        cv::eigen2cv(R, R_cv);
        
        // 将旋转矩阵转换为旋转向量
        cv::Rodrigues(R_cv, rvec);
        
        // 将平移向量转换为OpenCV的Mat
        cv::eigen2cv(t, tvec);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV异常: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "标准异常: " << e.what() << std::endl;
        return false;
    }
}

struct RGB {
    int r, g, b;
    RGB(int r = 0, int g = 0, int b = 0) : r(r), g(g), b(b) {}
};

RGB distanceToColor(double distance, double minDist, double maxDist) {
    // 归一化距离到 [0,1] 范围
    double normalized = (distance - minDist) / (maxDist - minDist);
    normalized = std::clamp(normalized, 0.0, 1.0);
    
    // 蓝(0,0,255) → 青(0,255,255) → 绿(0,255,0) → 黄(255,255,0) → 红(255,0,0)
    if (normalized < 0.25) {
        // 蓝 → 青
        double t = normalized / 0.25;
        return RGB(0, static_cast<int>(255 * t), 255);
    }
    else if (normalized < 0.5) {
        // 青 → 绿
        double t = (normalized - 0.25) / 0.25;
        return RGB(0, 255, static_cast<int>(255 * (1 - t)));
    }
    else if (normalized < 0.75) {
        // 绿 → 黄
        double t = (normalized - 0.5) / 0.25;
        return RGB(static_cast<int>(255 * t), 255, 0);
    }
    else {
        // 黄 → 红
        double t = (normalized - 0.75) / 0.25;
        return RGB(255, static_cast<int>(255 * (1 - t)), 0);
    }
}
/**
 * @brief 将OpenCV的rvec和tvec转换回Eigen::Matrix4d
 * @param rvec 旋转向量 (cv::Mat 3x1)
 * @param tvec 平移向量 (cv::Mat 3x1)
 * @return Eigen::Matrix4d 4x4齐次变换矩阵
 */
Eigen::Matrix4d openCVToEigenPose(const cv::Mat& rvec, const cv::Mat& tvec) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    
    // 将旋转向量转换为旋转矩阵
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    
    // 将OpenCV矩阵转换为Eigen矩阵
    Eigen::Matrix3d R;
    cv::cv2eigen(R_cv, R);
    
    // 将平移向量转换为Eigen向量
    Eigen::Vector3d t;
    cv::cv2eigen(tvec, t);
    
    // 构建齐次变换矩阵
    pose.block<3, 3>(0, 0) = R;
    pose.block<3, 1>(0, 3) = t;
    
    return pose;
}

/**
 * @brief 从旋转矩阵和平移向量构建Eigen::Matrix4d
 * @param rotation 旋转矩阵 (Eigen::Matrix3d)
 * @param translation 平移向量 (Eigen::Vector3d)
 * @return Eigen::Matrix4d 4x4齐次变换矩阵
 */
Eigen::Matrix4d createEigenPose(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = rotation;
    pose.block<3, 1>(0, 3) = translation;
    return pose;
}

/**
 * @brief 从四元数和平移向量构建Eigen::Matrix4d
 * @param quaternion 四元数 (Eigen::Quaterniond)
 * @param translation 平移向量 (Eigen::Vector3d)
 * @return Eigen::Matrix4d 4x4齐次变换矩阵
 */
Eigen::Matrix4d createEigenPoseFromQuaternion(const Eigen::Quaterniond& quaternion, const Eigen::Vector3d& translation) {
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
    pose.block<3, 1>(0, 3) = translation;
    return pose;
}

/**
 * @brief 打印变换矩阵（用于调试）
 * @param pose 4x4齐次变换矩阵
 * @param name 矩阵名称
 */
void printPose(const Eigen::Matrix4d& pose, const std::string& name = "Pose") {
    std::cout << name << ":\n" << pose << "\n" << std::endl;
}

/**
 * @brief 打印旋转向量和平移向量（用于调试）
 * @param rvec 旋转向量
 * @param tvec 平移向量
 * @param name 名称
 */
void printOpenCVPose(const cv::Mat& rvec, const cv::Mat& tvec, const std::string& name = "OpenCV Pose") {
    std::cout << name << ":" << std::endl;
    std::cout << "rvec: " << rvec.t() << std::endl;
    std::cout << "tvec: " << tvec.t() << "\n" << std::endl;
}


// 重投影误差检测与可视化函数
cv::Mat visualizeReprojectionError(
    const cv::Mat& current_image,                    // 当前帧图像
    const std::vector<cv::Point2f>& observed_pts,    // 观测到的特征点坐标
    const std::vector<cv::Point2f>& estimated_pts,   // 重投影的估计点坐标
    const std::vector<int>& inlier_mask,             // 内点标记（可选）
    float max_error_threshold = 20.0f,               // 最大误差阈值（像素）
    bool show_error_lines = true,                    // 是否显示误差连线
    bool show_error_values = false)                  // 是否显示误差值
{
    // 创建输出图像的副本
    cv::Mat output_image = current_image.clone();
    
    // 确保输入数据有效
    if (observed_pts.size() != estimated_pts.size() || observed_pts.empty()) {
        std::cerr << "Error: Observed and estimated points size mismatch or empty!" << std::endl;
        return cv::Mat();
    }
    
    // 绘制特征点和误差
    for (size_t i = 0; i < observed_pts.size(); ++i) {
        const cv::Point2f& obs_pt = observed_pts[i];
        const cv::Point2f& est_pt = estimated_pts[i];
        
        // 计算重投影误差（欧氏距离）
        float error = cv::norm(obs_pt - est_pt);
        
        // 确定点的颜色基于误差大小
        cv::Scalar color;
        if (error < max_error_threshold / 3) {
            color = cv::Scalar(0, 255, 0);  // 绿色：小误差
        } else if (error < max_error_threshold * 2 / 3) {
            color = cv::Scalar(0, 255, 255); // 黄色：中等误差
        } else {
            color = cv::Scalar(0, 0, 255);   // 红色：大误差
        }
        
        // 如果有内点标记，且当前点是外点，使用特殊颜色
        if (!inlier_mask.empty() && inlier_mask[i] == 0) {
            color = cv::Scalar(255, 0, 0);  // 蓝色：外点
        }
        
        // 绘制观测点（实心圆）
        cv::circle(output_image, obs_pt, 3, color, -1);
        
        // 绘制估计点（圆圈）
        cv::circle(output_image, est_pt, 4, color, 1);
        
        // 如果需要显示误差连线
        if (show_error_lines) {
            cv::line(output_image, obs_pt, est_pt, color, 1);
        }
        
        // 如果需要显示误差值
        if (show_error_values) {
            std::string error_text = cv::format("%.1f", error);
            cv::putText(output_image, error_text, 
                       cv::Point(est_pt.x + 5, est_pt.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
    }
    
    // 添加图例
    cv::putText(output_image, "Green: Observed point", cv::Point(10, 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(output_image, "Circle: Estimated point", cv::Point(10, 40),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(output_image, "Line: Reprojection error", cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    // 显示统计信息
    float avg_error = 0;
    for (size_t i = 0; i < observed_pts.size(); ++i) {
        avg_error += cv::norm(observed_pts[i] - estimated_pts[i]);
    }
    avg_error /= observed_pts.size();
    
    std::string stats = cv::format("Points: %zu | Avg Error: %.2f px", 
                                  observed_pts.size(), avg_error);
    cv::putText(output_image, stats, cv::Point(10, output_image.rows - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    
    return output_image;
    // 显示图像
    // cv::imshow("Reprojection Error Visualization", output_image);
    // cv::waitKey(1);  // 非阻塞显示
}



}

LioMesher::LioMesher(){
    // _filter_params = vio_params;//TODO(wz): move vio params to Parameter Server
    left_camera_ = CameraFactory::instance()->generateCameraFromYamlFile(param_.left_camera_model_path);
    CHECK(left_camera_!=nullptr)<<param_.left_camera_model_path;
    active_submaps_.reset(new ActiveSubmaps(left_camera_));

    YAML::Node node = YAML::LoadFile(param_.left_camera_model_path);
    // ps_ = new cvins::ParameterServer();
    // ps_->readParameters(node);
    
    // init vins feature tracker
    // feature_tracker_.reset(new lvimesher::FeatureTracker(param_.left_camera_model_path));
    // feature_tracker_->setMaxCount(ps_->MAX_CNT);
    // if(!ps_->mask.empty()){
    //     cv::Mat mask_img = cv::imread(ps_->mask, cv::IMREAD_GRAYSCALE);
    //     feature_tracker_->setFisheyeMask(mask_img);
    // }

    left_feature_tracker_.reset(new FeatureTracker());
    left_feature_tracker_->setCamera(left_camera_);
    if(!param_.mask_filename.empty()){
        left_feature_tracker_->setMask(param_.mask_filename);
        mask_ = cv::imread(param_.mask_filename, cv::IMREAD_GRAYSCALE);
    }

    orb_left_ = cv::ORB::create(500,        // 最大特征点数
        1.2f,        // 尺度金字塔参数
        8,           // 金字塔层数
        31,          // 边缘阈值
        0,           // 第一层金字塔级别
        2,           // WTA_K
        cv::ORB::HARRIS_SCORE,  // 评分类型
        31,          // 补丁大小
        20           // 快速阈值
    );
    
    
    // feature_manager_vi_.reset(new cvins::FeatureManager(ps_));
    // vins_.reset(new cvins::Estimator(ps_, feature_manager_vi_));
    if(param_.enable_right_camera){
        right_camera_ = CameraFactory::instance()->generateCameraFromYamlFile(param_.right_camera_model_path);
        CHECK(right_camera_!=nullptr)<<param_.right_camera_model_path;
        right_feature_tracker_.reset(new FeatureTracker());
        right_feature_tracker_->setCamera(right_camera_);
        // TODO(wz): set mask

        orb_right_ = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20 );
    }
    T_B_lC_ = sensors_.GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
    T_B_rC_ = sensors_.GetSensorToBody(SensorType::CAMERA_RIGHT).matrix();
    T_B_lC_inv_ = T_B_lC_.inverse();
    T_B_rC_inv_ = T_B_rC_.inverse();
}

void LioMesher::addImu(const sad::IMU& imu_data){
    if(!initialized_){
        LOG(WARNING)<<"The estimator should be initilized first.";
    }else{
        ieskf_.Predict(imu_data);
    }
}

void LioMesher::addWheelEncoder(const sad::WheelEncoder& wheel_encoder_data){
    if(!initialized_){
        LOG(WARNING)<<"The estimator should be initilized first.";
    }else{
        // LOG(INFO)<<wheel_encoder_data.linear_velocity_.norm()<<", "<<ieskf_.GetNominalState().v_.norm();
        if(!param_.enable_imu){
            ieskf_.PredictWithWheelAndOmega(wheel_encoder_data.linear_velocity_,
                                        wheel_encoder_data.angular_velocity_, wheel_encoder_data.timestamp_, 0.05, 0.02);
        }else{
            cur_wheel_encoder_ = wheel_encoder_data;
        }
    }
}

void LioMesher::addImage(const std::shared_ptr<ImagePkg>& img){
    if(!initialized_) return;
    if(!img){
        LOG(ERROR)<<"Invalid image!";
        return;
    }

    cur_img_pkg_ = img;
    if(img->left_img.channels() == 1){
        left_img_gray_ = img->left_img.clone();
    }else{
        cv::cvtColor(img->left_img, left_img_gray_, cv::COLOR_BGR2GRAY);  
    }
    
    CHECK(left_camera_->imageWidth() == img->left_img.cols && left_camera_->imageHeight() == img->left_img.rows)
                <<"Image size does not match the camera model, image cols: "<<img->left_img.cols
                <<", image rows: "<<img->left_img.rows
                <<", camera model width: "<<left_camera_->imageWidth()
                <<", camera model height: "<<left_camera_->imageHeight();

    // TicToc t_orb_window;
    // feature_manager_orb_.processFrame(img->left_img, cur_frame_id_);
    // feature_manager_orb_.establishMatches();
    // LOG(INFO)<<"ORB window manager update costs: "<<t_orb_window.toc();
    std::vector<cv::Point2f> pred_uvs_left = {};
    std::vector<cv::Point2f> pred_uvs_right = {};
    if(param_.enable_opticflow_with_prediction){
        Mat4d T_w_lc = se3_pose().matrix() * T_B_lC_;
        pred_uvs_left = PredictUVforLK(left_camera_, left_feature_tracker_, depth_filter_left_, T_w_lc);
        if(param_.enable_right_camera){
            Mat4d T_w_rc = se3_pose().matrix() * T_B_rC_;
            pred_uvs_right = PredictUVforLK(right_camera_, right_feature_tracker_, depth_filter_right_, T_w_rc);
        }
    }

    TicToc tlk;
    auto future_leftcam = std::async(std::launch::async, [&]() {
        left_feature_tracker_->processFrame(img->left_img, pred_uvs_left);
        // cv::Mat tracked_img = img->left_img.clone();
        // left_feature_tracker_->drawTrackingResults(tracked_img, false, false);
        // cv::imwrite("/home/wz/codes/ros_ws/livo/lvi_mesher/log/tracked_img/"+std::to_string(cur_frame_id_)+".jpg",  tracked_img);
    });

    
    
    if(param_.enable_right_camera){
        auto future_rightcam = std::async(std::launch::async, [&]() {
            right_feature_tracker_->processFrame(img->right_img, pred_uvs_right);                    
        });
        future_rightcam.get();    
    }
    future_leftcam.get();
  
    /* const auto& tracked_points = left_feature_tracker_->getTrackedPoints();
    int N = tracked_points.size();
    if(N > 10){
        TicToc torb;
        std::vector<cv::Mat> temp_descriptors;
        left_keypoints_.clear();
        temp_descriptors.clear();
        for(int i = 0; i<N; i++){
            if(tracked_points[i].descriptor.empty()) continue;
            left_keypoints_.push_back(tracked_points[i].keypoint);
            temp_descriptors.push_back(tracked_points[i].descriptor);
        }
        if(!temp_descriptors.empty()){
            cv::vconcat(temp_descriptors, left_descriptors_);
        }

        if(!left_keypoints_last_.empty()){
            cv::Ptr<cv::DescriptorMatcher> matcher;
            if (left_descriptors_.type() == CV_8U) {
                matcher = cv::makePtr<cv::FlannBasedMatcher>(
                    cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
            } else {
                matcher = cv::makePtr<cv::FlannBasedMatcher>();
            }

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher->knnMatch(left_descriptors_, left_descriptors_last_, knn_matches, 2);

            // 应用 Lowe's ratio test 筛选匹配
            const float ratio_thresh = 0.7f;
            std::vector<cv::DMatch> good_matches;
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if(knn_matches[i].empty()) continue; //Why this happen?
                if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }
            // 绘制好的匹配
            cv::drawMatches(left_img_gray_, left_keypoints_, left_img_gray_last_, left_keypoints_last_, good_matches, img_match_debug_,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }
        left_keypoints_last_ = left_keypoints_;
        left_descriptors_last_ = left_descriptors_;
    } */
    
    
    left_img_gray_last_ = left_img_gray_;
    right_img_gray_last_ = right_img_gray_;
}

std::shared_ptr<TrackingResult> LioMesher::addPointCloud(const pcl::PointCloud<PointXYZT>::Ptr& xyzt_base){
    if(!initialized_) return nullptr;
    cur_scan_raw_.reset(new pcl::PointCloud<PointXYZT>());
    cur_scan_deskewed_.reset(new pcl::PointCloud<PointXYZT>());
    cur_scan_deskewed_xyz_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    cur_scan_deskewed_xyz_ds_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    // cur_scan_deskewed_cov_ds_.clear();
    cur_scan_deskewed_xyzrgb_.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    cur_scan_raw_->points = xyzt_base->points;
    if(!param_.enable_imu && !param_.enable_wheel_encoder){
        SE3 T_guess = SE3(pose_predictor_.predictAt(xyzt_base->points.back().t).matrix());
        if(param_.enable_vl_rotation_guess){
            if(param_.enable_camera){
                TicToc t_c_guess;
                cv::Mat R, t;
                if(left_feature_tracker_->getRelativePose(R, t)){
                    Mat3d R_eigen;
                    Vec3d t_eigen;
                    cv::cv2eigen(R, R_eigen);
                    cv::cv2eigen(t, t_eigen);

                    Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d Tb = Eigen::Matrix4d::Identity();
                    Tc.block<3, 3>(0, 0) = R_eigen;
                    Tc.block<3, 1>(0, 3) = t_eigen;
                    Tb = T_B_lC_ * Tc.inverse() * T_B_lC_inv_;

                    Eigen::Matrix4d T_pred = T_l_.matrix() * Tb;
                    Eigen::Matrix3d R_guess = T_pred.block<3,3>(0,0);
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_guess, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    Eigen::Matrix3d R_orthogonal = svd.matrixU() * svd.matrixV().transpose();

                    if (R_orthogonal.determinant() < 0) {
                        R_orthogonal = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1, 1, -1) * svd.matrixV().transpose();
                    }
                    ieskf_.Predict(Sophus::SO3d(R_orthogonal), T_guess.translation(), xyzt_base->points.back().t);
                }else{
                    ieskf_.Predict(T_guess.so3(), T_guess.translation(), xyzt_base->points.back().t);
                }
                time_cost_camera_rotation_guess_ += t_c_guess.toc();
            }else{
                if(last_scan_raw_){
                    TicToc t_l_guess;
                    RangeImageRotEstimator::Config cofig;
                    RangeImageRotEstimator lidar_rot_estimator(cofig);
                    Sophus::SO3d delta_rot = lidar_rot_estimator.estimateRotation(cur_scan_raw_, last_scan_raw_);
                    Eigen::Isometry3d Tl = Eigen::Isometry3d::Identity();
                    Tl.prerotate(delta_rot.unit_quaternion());

                    Eigen::Isometry3d T_B_L = sensors_.GetSensorToBody(SensorType::LIDAR1);
                    Eigen::Isometry3d Tb = Eigen::Isometry3d::Identity();
                    Tb = T_B_L * Tl * T_B_L.inverse();

                    Sophus::SO3d rot_pred = T_l_.so3() * Sophus::SO3d(Tb.rotation());
                    ieskf_.Predict(rot_pred, T_guess.translation(), xyzt_base->points.back().t);
                    time_cost_lidar_rotation_guess_ += t_l_guess.toc();
                }else{
                    ieskf_.Predict(T_guess.so3(), T_guess.translation(), xyzt_base->points.back().t);
                }
            }
        }else{
            ieskf_.Predict(T_guess.so3(), T_guess.translation(), xyzt_base->points.back().t);
        }
    }
    
    std::shared_ptr<TrackingResult> result(new TrackingResult());
    if(!first_scan_processed_){
        cur_frame_id_ = 0;
        cur_submap_id_ = 0;
        Mat4d start_pose= se3_pose().matrix();
        xyzt2xyz(*cur_scan_raw_, *cur_scan_deskewed_xyz_);
        ColorizePointCloud(*cur_scan_deskewed_xyz_, *cur_scan_deskewed_xyzrgb_);
        active_submaps_->AddData(cur_frame_id_, start_pose, cv::Mat(), cur_scan_deskewed_xyzrgb_);
        last_pose_ = se3_pose();
        first_scan_processed_ = true;
    }else{
        cur_frame_id_++;
        if(param_.enable_deskew){
            deskew_to_end();
        }else{
            *cur_scan_deskewed_ = *cur_scan_raw_;
        }
        
        xyzt2xyz(*cur_scan_deskewed_, *cur_scan_deskewed_xyz_);
        pcl::PointCloud<pcl::PointXYZ> filtered_points;
        filtered_points = pclVoxelFilter(cur_scan_deskewed_xyz_, param_.voxel_size_for_resgister);
        cur_scan_deskewed_xyz_ds_->points = filtered_points.points;
        
        /* int n = cur_scan_deskewed_xyz_ds_->points.size();
        cur_scan_deskewed_cov_ds_.resize(n);
        for(int i = 0; i < n; i++){
            const pcl::PointXYZ& xyz = cur_scan_deskewed_xyz_ds_->points[i];
            Vec3d pb(xyz.x, xyz.y, xyz.z);
            cur_scan_deskewed_cov_ds_[i] = pointCovariance(pb, param_.range_err, param_.degree_err);
        } */

        ColorizePointCloud(*cur_scan_deskewed_xyz_, *cur_scan_deskewed_xyzrgb_);
        TicToc t_lio_ekf;

        if(param_.enable_camera){
            CHECK(cur_img_pkg_ != nullptr) << "If enabled camera, add image first!";
            if(param_.lv_joint_eskf){
                ieskf_.UpdateUsingCustomObserve([this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
                    ComputeJointResidualAndJacobians(input_pose, HTVH, HTVr);
                });
            }else{
                TicToc t_lo;
                ieskf_.UpdateUsingCustomObserve([this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
                    computeResidualAndJacobians(input_pose, HTVH, HTVr);
                });
                time_cost_lo_update_ += t_lo.toc();
                TicToc t_vo;
                ieskf_.UpdateUsingCustomObserve([this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
                    computeResidualAndJacobiansReprojection(input_pose, HTVH, HTVr);
                });
                time_cost_vo_update_ += t_vo.toc();
            }
            // TODO(wz): remove image data from submap
            TicToc t_mesh;
            active_submaps_->AddData(cur_frame_id_, se3_pose().matrix(), cv::Mat(), cur_scan_deskewed_xyzrgb_);
            time_cost_meshing_ += t_mesh.toc();

            // 更新3D视觉地图点，成熟点位置及描述子
            // 若有新的成熟点，加入到Submap中，新成熟点可能包括以前是成熟的，但跟踪中断的（不同ID）或滤波中断的（同ID），如需要，可在Cell内做进一步判断
            Mat4d T_w_lc = se3_pose().matrix() * T_B_lC_;
            Mat4d T_w_rc = se3_pose().matrix() * T_B_rC_;
            auto xyz_update_left = std::async(std::launch::async, [&]() {
                TicToc t_raycast;
                AssignXYZandUpdateDepth(left_camera_, left_feature_tracker_, depth_filter_left_, T_w_lc);
                time_cost_raycast_ += t_raycast.toc();
                if(param_.enable_vmap_constraint){ //TODO: only matrure
                    for(uint64_t id: depth_filter_left_.newly_matures_){
                        const auto& feat = depth_filter_left_.getFeatureTrackById(id);
                        if(feat->observations.empty()) continue;   
                        VisualPoint3D vpt_w;
                        vpt_w.id = feat->feature_id; 
                        if(depth_filter_left_.getFeatureWorldPoint(id, vpt_w.xyz)){
                            vpt_w.descriptor = feat->observations.back().descriptor;//存最新帧中观测到的描述子
                            active_submaps_->AddVisualMapPoint(vpt_w);
                        }
                    }
                }
            });
            if(param_.enable_right_camera){
                auto xyz_update_right = std::async(std::launch::async, [&]() {
                    AssignXYZandUpdateDepth(right_camera_, right_feature_tracker_, depth_filter_right_, T_w_rc);
                           
                    /* for(uint64_t id: depth_filter_right_.newly_matures_){
                        const auto& feat = depth_filter_right_.getFeatureTrackById(id);
                        if(feat->observations.empty()) continue;   
                        VisualPoint3D vpt_w;
                        vpt_w.id = feat->feature_id; 
                        if(depth_filter_right_.getFeatureWorldPoint(id, vpt_w.xyz)){
                            vpt_w.descriptor = feat->observations.back().descriptor;
                            active_submaps_->AddVisualMapPoint(vpt_w);
                        }
                    } */
                });
                xyz_update_right.get();    
            }
            xyz_update_left.get(); 
            
            // visualization of reprojection error
            cv::Mat err_img_left;
            std::vector<Vec3d> mature_pts_left;
            VisualizeMaturePointsAndReprojectionError(left_camera_, cur_img_pkg_->left_img.clone(), 
                   T_w_lc, *left_feature_tracker_, depth_filter_left_, err_img_left, mature_pts_left);
            if(param_.enable_right_camera){
                cv::Mat err_img_right;
                std::vector<Vec3d> mature_pts_right;
                VisualizeMaturePointsAndReprojectionError(right_camera_, cur_img_pkg_->right_img.clone(), 
                   T_w_rc, *right_feature_tracker_, depth_filter_right_, err_img_right, mature_pts_right);
                CHECK(err_img_left.size() == err_img_right.size())<<err_img_left.size()<<", "<<err_img_right.size();
                cv::hconcat(err_img_left, err_img_right, result->debug_img);
                result->current_mature_points.insert(result->current_mature_points.end(), mature_pts_left.begin(), mature_pts_left.end());
                result->current_mature_points.insert(result->current_mature_points.end(), mature_pts_right.begin(), mature_pts_right.end());
            }else{
                result->debug_img = err_img_left;
                result->current_mature_points.insert(result->current_mature_points.end(), mature_pts_left.begin(), mature_pts_left.end());
            }
        }else{
            TicToc t_lo;
            ieskf_.UpdateUsingCustomObserve([this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
                computeResidualAndJacobians(input_pose, HTVH, HTVr);
            });
            time_cost_lo_update_ += t_lo.toc();
            TicToc t_mesh;
            active_submaps_->AddData(cur_frame_id_, se3_pose().matrix(), cv::Mat(), cur_scan_deskewed_xyzrgb_);
            time_cost_meshing_ += t_mesh.toc();
        }
        
        last_pose_ = se3_pose();
        T_ll_ = T_l_;
        T_l_ = last_pose_;

        /* cv::Mat R, t;
        if(param_.enable_camera && left_feature_tracker_->getRelativePose(R, t)){
            Mat3d R_eigen;
            Vec3d t_eigen;
            cv::cv2eigen(R, R_eigen);
            cv::cv2eigen(t, t_eigen);

            Eigen::Matrix4d Tc = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d Tb = Eigen::Matrix4d::Identity();
            Tc.block<3, 3>(0, 0) = R_eigen;
            Tc.block<3, 1>(0, 3) = t_eigen;
            Tb = T_B_lC_ * Tc.inverse() * T_B_lC_inv_;

            auto relative_pose_lio = T_ll_.inverse() * T_l_;
            Mat3d rrr = relative_pose_lio.so3().matrix().transpose() * Tb.block<3,3>(0,0);
            Eigen::Quaterniond q_rr(rrr);
            Eigen::Quaterniond q(Tb.block<3,3>(0,0));
            Vec3d r = q.toRotationMatrix().eulerAngles(2,1,0) * 180./M_PI;
            Vec3d rr = q_rr.toRotationMatrix().eulerAngles(2,1,0) * 180./M_PI;
            // LOG(INFO) << r(0)<<" "<<r(1)<<" "<<r(2)<<"--->"<<rr(0)<<" "<<rr(1)<<" "<<rr(2);

            Eigen::Matrix4d T_pred = T_l_.matrix() * Tb;
            SE3 T_pred_const = SE3(pose_predictor_.predictAt(xyzt_base->points.back().t).matrix());
            Vec3d t_const = T_pred.block<3, 1>(0, 3);
            t_const.normalize();
            Vec3d t_guess = T_pred_const.translation().norm() * t_const;

            Eigen::Matrix3d R_guess = T_pred.block<3,3>(0,0);
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_guess, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d R_orthogonal = svd.matrixU() * svd.matrixV().transpose();

            // 检查行列式，防止出现镜像矩阵（行列式为 -1）
            if (R_orthogonal.determinant() < 0) {
                R_orthogonal = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1, 1, -1) * svd.matrixV().transpose();
            }
        } */

        /* if(last_scan_raw_){
            RangeImageRotEstimator::Config cofig;
            RangeImageRotEstimator lidar_rot_estimator(cofig);
            Sophus::SO3d delta_rot = lidar_rot_estimator.estimateRotation(cur_scan_raw_, last_scan_raw_);
            auto relative_pose_lio = T_ll_.inverse() * T_l_;
            Mat3d rrr = (relative_pose_lio.so3().inverse() * delta_rot).matrix();
            Eigen::Quaterniond q_rr(rrr);
            Eigen::Quaterniond q(relative_pose_lio.so3().matrix());
            Vec3d r = q.toRotationMatrix().eulerAngles(2,1,0) * 180./M_PI;
            Vec3d rr = q_rr.toRotationMatrix().eulerAngles(2,1,0) * 180./M_PI;
            LOG(INFO) << r(0)<<" "<<r(1)<<" "<<r(2)<<"--->"<<rr(0)<<" "<<rr(1)<<" "<<rr(2);
        } */

        Mat4d pose_opt = last_pose_.matrix();
    }

    if(param_.enable_camera){
        result->timestamp = cur_img_pkg_->timestamp_sec_left;
        result->timestamp_nsec = cur_img_pkg_->timestamp_sec_left*1e9;
        
        // 填充3D特征点，用于后端回环检测及相对位姿解算; 简单化，不用成熟点，直接用当前帧追踪到的点的3D位置和描述子，可能有outlier，配准的时候再剔除
        TrackedOrb3dPoints(left_feature_tracker_, depth_filter_left_, pose(), T_B_lC_inv_,  
                           result->points_3d_cam, result->keypoints, result->descriptors);
        // Mat4d T_w_lc = se3_pose().matrix() * T_B_lC_;
        // VisualPoints3D(cur_img_pkg_->left_img, T_w_lc, result->keypoints, result->descriptors, result->points_3d_cam);
    }else{
        result->timestamp = xyzt_base->points.back().t;
        result->timestamp_nsec = xyzt_base->points.back().t*1e9;
    }
     
    result->frame_id = cur_frame_id_;
    result->local_pose = pose();
    result->active_submaps = active_submaps_->submaps();
    result->deskewed_points = *cur_scan_deskewed_xyzrgb_;
    pose_predictor_.addPose(result->timestamp, Eigen::Isometry3d(pose()));
    last_scan_raw_ = cur_scan_raw_;

    /* if(cur_frame_id_ > 0){
        LOG(INFO) << time_cost_lidar_rotation_guess_ /  (cur_frame_id_ + 1) << ", "
                  << time_cost_camera_rotation_guess_/ (cur_frame_id_ + 1)  << ", "
                  << time_cost_meshing_ / (cur_frame_id_ + 1)  << ", "
                  << time_cost_raycast_ / (cur_frame_id_ + 1)  << ", "
                  << time_cost_lo_update_ / (cur_frame_id_ + 1)  << ", "
                  << time_cost_vo_update_ / (cur_frame_id_ + 1);
    } */
    
    return result;
}


void LioMesher::xyzt2xyz(const pcl::PointCloud<PointXYZT>& xyzt, pcl::PointCloud<pcl::PointXYZ>& xyz){
    size_t n = xyzt.points.size();
    xyz.points.resize(n);
    std::vector<size_t> idx(n);
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        xyz.points[i].x = xyzt.points[i].x;
        xyz.points[i].y = xyzt.points[i].y;
        xyz.points[i].z = xyzt.points[i].z;
    });
}

Mat4d LioMesher::LocalizeCameraByPnP(){
    return Mat4d::Identity();
}

void LioMesher::VisualizeMaturePointsAndReprojectionError(
            const camodocal::CameraPtr& camera, const cv::Mat& img_src, const Mat4d& T_wc,
            const FeatureTracker& feature_tracker, SlidingWindowDepthFilter& depth_filter,
            cv::Mat& err_img, std::vector<Vec3d>& mature_pts){
    err_img = img_src.clone();
    mature_pts.clear();
    const auto& tracked_features = feature_tracker.getTrackedPoints();
    for(const auto& feat: tracked_features){
        if(feat.trackCount < 2) continue;//new feature
        auto f = depth_filter.getFeatureTrackById(feat.id);
        if(!f) continue;
        Vec3d map_point;
        bool updated = depth_filter.getFeatureWorldPoint(feat.id, map_point);
        if(!updated) continue;
        Vec3d pt_c = T_wc.inverse().block<3,3>(0,0) * map_point + T_wc.inverse().block<3,1>(0,3);
        Vec2d uv;
        if(InCameraView(camera, pt_c, uv)){
            Vec2d uv_obs(feat.position.x, feat.position.y);
            if((uv_obs-uv).norm() > param_.visual_residual_threshold) continue;
            if(f->is_pure_visual){
                cv::circle(err_img, cv::Point2f(uv[0], uv[1]), 2, cv::Scalar(255, 0, 0), 5);
            }else{
                cv::circle(err_img, cv::Point2f(uv[0], uv[1]), 2, cv::Scalar(0, 255, 0), 5);
            }
            cv::circle(err_img, feat.position, 2, cv::Scalar(0, 0, 255), 2);
            cv::line(err_img, cv::Point2f(uv[0], uv[1]), feat.position, cv::Scalar(255, 0, 0), 1);

            mature_pts.push_back(map_point);
        }
    }
    /* auto matching_submap = get_matching_submap();
    Mat4d T_ws = matching_submap->pose_odom_;
    Mat4d T_cw = T_wc.inverse();
    // 遍历submap，筛选中心点在视角内的cell (TODO(wz):可更精确)。
    for(auto i_cell : matching_submap->cells_glb){
        const auto& region = i_cell.second.region;
        double cx = (region.x_min + region.x_max) / 2.;
        double cy = (region.y_min + region.y_max) / 2.;
        double cz = (region.z_min + region.z_max) / 2.;
        Vec4d pc_s(cx, cy, cz, 1.0);
        Vec4d pc_c = T_cw * T_ws * pc_s;

        Vec2d uv;
        if(InCameraView(camera, pc_c.segment<3>(0), uv) ){
            if(!i_cell.second.visual_points_.empty()){
                for(const auto& k_vmp: i_cell.second.visual_points_){
                    if(k_vmp.second.descriptor.empty()) continue;
                    Vec3d vp_w = T_ws.block<3,3>(0,0) * k_vmp.second.xyz + T_ws.block<3,1>(0,3);
                    mature_pts.push_back(vp_w);
                }
            }
        }
    } */
}

void LioMesher::AssignXYZandUpdateDepth(
        camodocal::CameraPtr camera, 
        std::shared_ptr<FeatureTracker>& tracker, 
        SlidingWindowDepthFilter& depth_filter, Mat4d T_wc){
    
    std::vector<double> depth_cur_frame; 
    std::vector<double> variance; 
    std::vector<Vec3d> xyz_world;
    std::vector<Vec3d> xyz_submap;
    std::vector<const Cell*> hit_cells;   
    // TicToc t_ray_tracing;
    AssignXYZbyRayTracing(camera, tracker, T_wc, xyz_world, xyz_submap, hit_cells, depth_cur_frame, variance);
    // LOG(INFO)<<"Ray tracing cost: "<<t_ray_tracing.toc();//~0.6
    // TicToc t_inverse_projection;
    // AssignXYZbyInverseProjection(camera, tracker, T_wc, xyz_world, xyz_submap, hit_cells, depth_cur_frame);
    // LOG(INFO)<<"Inverse projection cost: "<<t_inverse_projection.toc();//~1.9
   
    Mat4d T_cw = T_wc.inverse();
    for(int i = 0; i < tracker->getTrackedPoints().size(); i++){
        const auto& feat = tracker->getTrackedPoints()[i];
        if(feat.id < 0) continue;

        Vec3d x3d_norm;
        camera->liftProjective(Vec2d(feat.position.x, feat.position.y), x3d_norm);
        x3d_norm /= x3d_norm[2];
        Observation obs(cur_frame_id_, x3d_norm.head<2>(), cv::Point2f(feat.position.x, feat.position.y));
        obs.keypoint = feat.keypoint;
        obs.descriptor = feat.descriptor;

        if(depth_cur_frame[i] > 0.5){
            obs.has_raycast = true;
            obs.raycast_var = variance[i];
            obs.raycast_depth = depth_cur_frame[i];
        }
        depth_filter.addObservation(feat.id, obs);
    }
    Eigen::Quaterniond q_cw(T_cw.block<3,3>(0,0));
    q_cw.normalize();
    depth_filter.addFramePose(cur_frame_id_, CameraPose(q_cw, T_cw.block<3,1>(0,3)));
    depth_filter.updateFilter();
    depth_filter.updateMaturity();
}

std::vector<cv::Point2f> LioMesher::PredictUVforLK(camodocal::CameraPtr camera, const std::shared_ptr<FeatureTracker>& tracker, 
                        SlidingWindowDepthFilter& depth_filter, Mat4d T_wc_pred){
    Mat4d T_cw_pred = T_wc_pred.inverse();
    const auto& last_tracks = depth_filter.getFeatureTracks();
    const auto& tracked_pts = tracker->getTrackedPoints();
    std::vector<cv::Point2f> pred_uvs = {};
    if(!tracked_pts.empty()){
        pred_uvs.resize(tracked_pts.size());
        for(int i = 0; i < tracked_pts.size(); ++i){
            TrackedPoint tp = tracked_pts[i];
            pred_uvs[i] = tp.position;
            auto it = last_tracks.find(tp.id);
            if(it != last_tracks.end()){
                Vec2d uv;
                Vec3d map_point;
                if(!depth_filter.getFeatureWorldPoint(tp.id, map_point)) continue;
                Vec3d pt_c_pred = T_cw_pred.block<3,3>(0,0) * map_point + T_cw_pred.block<3,1>(0,3);
                if(InCameraView(camera, pt_c_pred, uv)){
                    pred_uvs[i] = cv::Point2f(uv[0], uv[1]);
                }
            }
        }
    }
    return pred_uvs;
}

void LioMesher::ProjectPointCloudToImage(){
    cv::Mat img = cur_img_pkg_->left_img.clone();
    for(const auto& pt: cur_scan_deskewed_->points){
        Vec4d p_imu(pt.x, pt.y, pt.z, 1);
        Vec3d p_lc = (T_B_lC_inv_ * p_imu).head<3>();
        Vec2d uv;
        if(InLeftCamera(p_lc, uv)){
            // cv::circle(img, cv::Point2f(uv[0], uv[1]), 1, cv::Scalar(pt.r, pt.g, pt.b), -1);
            img.at<cv::Vec3b>(cv::Point2f(uv[0], uv[1])) = cv::Vec3b(0, 255, 0);
        }
    }
    cv::imwrite("/home/wz/codes/ros_ws/livo/lvi_mesher/log/projected_points/"+std::to_string(int64_t(cur_img_pkg_->timestamp_sec_left*1e9))+".jpg", img);
}


void LioMesher::AssignXYZbyRayTracing(
        const camodocal::CameraPtr& camera, 
        const std::shared_ptr<FeatureTracker>& tracker, 
        const Mat4d& T_wc,
        std::vector<Vec3d>& xyz_world,
        std::vector<Vec3d>& xyz_submap,
        std::vector<const Cell*>& hit_cells,
        std::vector<double>& depth_cur_frame,
        std::vector<double>& variance){
    auto matching_submap = active_submaps_->matching_submap();
    const Mat4d& T_ws = matching_submap->pose_odom_;
    Mat4d T_sw = T_ws.inverse();
    
    Mat4d T_cw = T_wc.inverse(); 
    Vec3d P_wc = T_wc.block<3,1>(0,3);
    Mat3d R_wc = T_wc.block<3,3>(0,0);
    Mat4d T_sc = T_sw * T_wc;
    Mat4d T_cs = T_sc.inverse();
    Vec3d P_sc = T_sc.block<3, 1>(0,3);
    Mat3d R_sc = T_sc.block<3, 3>(0,0);

    double l = Parameter::GetInstance().grid;
    int n_row =  Parameter::GetInstance().num_test;

    const std::vector<TrackedPoint> tracked_pts = tracker->getTrackedPoints();
    depth_cur_frame.resize(tracked_pts.size(), -1.0);
    variance.resize(tracked_pts.size(), -1.0);

    xyz_world.resize(tracked_pts.size(), Vec3d::Zero());
    xyz_submap.resize(tracked_pts.size(), Vec3d::Zero());
    hit_cells.resize(tracked_pts.size());
    std::vector<size_t> idx(tracked_pts.size());
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        const auto& feature = tracked_pts[i];
        int id = feature.id;
        const cv::Point2f& uv_mm =  feature.position;
        Vec3d dir_cam;
        camera->liftProjective(Vec2d(uv_mm.x, uv_mm.y), dir_cam);      
        dir_cam.normalize();

        // 1. 找到与当前ray相交的cell
        int max_step = (param_.range_max) / l;
        std::vector<const Cell*> cells_hit = matching_submap->getIntersectedCells(
                Ray(P_sc, R_sc * dir_cam), max_step);
        
        // 2. 如果有相交的cell，寻找相交的面片，计算交点
        for(const Cell* cell: cells_hit){
            hit_cells[i] = cell;
            
            double min_variance = 1e9;
            // cell在子地图坐标系下，visual_points_存储在子地图坐标系下
            std::optional<Vec4d> resx = cell->rayCellIntersectionX(Ray(P_sc, R_sc*dir_cam));
            std::optional<Vec4d> resy = cell->rayCellIntersectionY(Ray(P_sc, R_sc*dir_cam));
            std::optional<Vec4d> resz = cell->rayCellIntersectionZ(Ray(P_sc, R_sc*dir_cam));
            if(resx && (*resx)(3) < param_.variance_depth_association){
                min_variance = (*resx)(3);
                xyz_submap[i] = (*resx).head(3);
                xyz_world[i] = T_ws.block<3,3>(0,0) * (*resx).head(3) + T_ws.block<3, 1>(0,3);

                Vec3d pt_cam = T_cs.block<3,3>(0,0) * (*resx).head(3) + T_cs.block<3, 1>(0,3);
                depth_cur_frame[i] = pt_cam[2];  
                variance[i] = min_variance;          
            }

            if(resy && (*resy)(3) < param_.variance_depth_association){
                if(depth_cur_frame[i] < 0 || (depth_cur_frame[i] > 0 && (*resy)(3) < min_variance)){
                    min_variance = (*resy)(3);
                    xyz_submap[i] = (*resy).head(3);
                    xyz_world[i] = T_ws.block<3,3>(0,0) * (*resy).head(3) + T_ws.block<3, 1>(0,3);

                    Vec3d pt_cam = T_cs.block<3,3>(0,0) * (*resy).head(3) + T_cs.block<3, 1>(0,3);
                    depth_cur_frame[i] = pt_cam[2];
                    variance[i] = min_variance;
                }
            }
            
            if(resz && (*resz)(3) < param_.variance_depth_association){
                if(depth_cur_frame[i] < 0 || (depth_cur_frame[i] > 0 && (*resz)(3) < min_variance)){
                    min_variance = (*resz)(3);
                    xyz_submap[i] = (*resz).head(3);
                    xyz_world[i] = T_ws.block<3,3>(0,0) * (*resz).head(3) + T_ws.block<3, 1>(0,3);

                    Vec3d pt_cam = T_cs.block<3,3>(0,0) * (*resz).head(3) + T_cs.block<3, 1>(0,3);
                    depth_cur_frame[i] = pt_cam[2];
                    variance[i] = min_variance;
                }
            }
        }
    });
}


void LioMesher::AssignXYZbyInverseProjection(
        const camodocal::CameraPtr& camera, 
        const std::shared_ptr<FeatureTracker>& tracker, 
        const Mat4d& T_wc,
        std::vector<Vec3d>& xyz_world,
        std::vector<Vec3d>& xyz_submap,
        std::vector<const Cell*>& hit_cells,
        std::vector<double>& depth_cur_frame) {

    // --- 1. 坐标系与参数准备 ---
    auto matching_submap = active_submaps_->matching_submap();
    const Mat4d& T_ws = matching_submap->pose_odom_; // Submap -> World
    Mat4d T_sw = T_ws.inverse();
    
    // 计算 Submap -> Camera 的变换 T_cs (用于把Cell转到相机系)
    // T_sc = T_sw * T_wc;
    // T_cs = T_wc^-1 * T_ws^-1 ? 不，应该是 T_cs = T_sc.inverse()
    Mat4d T_sc = T_sw * T_wc;     // Camera inside Submap
    Mat4d T_cs = T_sc.inverse();  // Submap inside Camera
    
    Mat3d R_cs = T_cs.block<3,3>(0,0);
    Vec3d P_cs = T_cs.block<3,1>(0,3);

    // Ray计算需要的 T_sc (相机光心在Submap系下的位置)
    Vec3d P_sc = T_sc.block<3, 1>(0,3);
    Mat3d R_sc = T_sc.block<3, 3>(0,0);

    double grid_size = Parameter::GetInstance().grid;
    // Cell的半对角线长度，用于计算包围球/盒
    double cell_radius_3d = grid_size * std::sqrt(3.0) / 2.0;

    const std::vector<TrackedPoint> tracked_pts = tracker->getTrackedPoints();
    size_t n_pts = tracked_pts.size();

    // 初始化输出
    depth_cur_frame.assign(n_pts, -1.0);
    xyz_world.assign(n_pts, Vec3d::Zero());
    xyz_submap.assign(n_pts, Vec3d::Zero());
    hit_cells.assign(n_pts, nullptr);

    // 为每个特征点预分配一个候选Cell列表
    std::vector<std::vector<CellCandidate>> feature_candidates(n_pts);

    // --- 2. Broad Phase: 遍历所有Cell，筛选并分配给特征点 ---
    
    // 获取Submap中所有存储了Mesh的Cells (假设有此接口)
    // 如果是hash map，这里需要遍历 values
    // std::vector<Cell*> all_cells;
    // for(auto tc: matching_submap->cells_glb){
    //     all_cells.push_back(&tc.second);
    // }
    const auto& all_cells = matching_submap->cells_glb;

    // 获取相机焦距用于估算投影大小 (假设fx fy近似)
    double fx = 0;
    // 这里假设 CameraPtr 有类似 accessors，如果没有需根据实际库调整
    // 粗略估算投影半径: r_px = (fx * r_world) / Z
    // 如果没有直接接口，可以通过投影单位向量估算，这里简化处理:
    cv::Point2f principal_pt;
    // 简单 hack: 投影 (0,0,1) 和 (1,0,1) 获取焦距像素值，或者直接读取内参
    Vec2d tmp_p, tmp_p2;
    camera->spaceToPlane(Vec3d(0,0,1), tmp_p);
    camera->spaceToPlane(Vec3d(1,0,1), tmp_p2);
    fx = std::abs(tmp_p2.x() - tmp_p.x());

    for (const auto& [key, cell] : all_cells) { // 假设是 map<Key, Cell*>
        const auto& region = cell.region;
        double cx = (region.x_min + region.x_max) / 2.;
        double cy = (region.y_min + region.y_max) / 2.;
        double cz = (region.z_min + region.z_max) / 2.;
        
        // 2.1 将 Cell 中心转换到相机坐标系
        Vec3d center_s(cx, cy, cz);
        Vec3d center_c = R_cs * center_s + P_cs;
        double depth = center_c.z();

        // 2.2 深度裁剪 (在相机后面或太远的不要)
        if (depth < 0.1 || depth > param_.range_max) continue;

        // 2.3 投影 Cell 中心到图像平面
        Vec2d uv_c;
        camera->spaceToPlane(center_c, uv_c);

        // 2.4 计算 Cell 在图像上的投影包围盒 (Conservative AABB)
        // 投影半径 r_px = (fx * cell_radius) / depth
        double radius_px = (fx * cell_radius_3d) / depth;
        // 适当放大一点作为裕量 (buffer)
        radius_px *= 1.2; 

        double u_min = uv_c.x() - radius_px;
        double u_max = uv_c.x() + radius_px;
        double v_min = uv_c.y() - radius_px;
        double v_max = uv_c.y() + radius_px;

        // 2.5 检查哪些特征点落在这个 2D Box 内
        // 如果特征点很多，可以用Grid加速，但通常特征点 < 500，双重循环是可以接受的
        for (int i = 0; i < n_pts; ++i) {
            const auto& pt = tracked_pts[i].position;
            if (pt.x >= u_min && pt.x <= u_max &&
                pt.y >= v_min && pt.y <= v_max) {
                
                // 添加到候选列表
                feature_candidates[i].push_back({&cell, depth});
            }
        }
    }

    // --- 3. Narrow Phase: 对每个特征点，按距离遍历候选Cell进行精确求交 ---
    
    // 并行化处理每个特征点
    std::vector<size_t> idx(n_pts);
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        auto& candidates = feature_candidates[i];
        if (!candidates.empty()) {
            // 3.1 按深度排序 (从近到远)
            std::sort(candidates.begin(), candidates.end());

            // 准备射线 (Submap Frame)
            const cv::Point2f& uv_mm = tracked_pts[i].position;
            Vec3d dir_cam;
            camera->liftProjective(Vec2d(uv_mm.x, uv_mm.y), dir_cam);      
            dir_cam.normalize();
            Ray ray_s(P_sc, R_sc * dir_cam); // Ray in Submap frame

            double min_dist_sq = 1e9;
            bool found_valid_hit = false;

            // 3.2 遍历候选 Cell
            for (const auto& cand : candidates) {
                const Cell* cell = cand.cell;
                
                // 如果我们已经找到了一个交点，并且当前的Cell最近处都比那个交点远，就可以提前退出了
                // 注意：cand.depth 是 Cell 中心深度，这里做判断要小心，加上半径容差
                if (found_valid_hit && cand.depth - cell_radius_3d > std::sqrt(min_dist_sq)) break;

                // 原有的求交逻辑
                std::optional<Vec4d> resx = cell->rayCellIntersectionX(ray_s);
                std::optional<Vec4d> resy = cell->rayCellIntersectionY(ray_s);
                std::optional<Vec4d> resz = cell->rayCellIntersectionZ(ray_s);

                double local_min_var = 1e9;
                Vec3d best_pt_s = Vec3d::Zero();
                bool hit_this_cell = false;

                // 检查 X 面
                if(resx && (*resx)(3) < param_.variance_depth_association){
                    local_min_var = (*resx)(3);
                    best_pt_s = (*resx).head(3);
                    hit_this_cell = true;
                }
                // 检查 Y 面
                if(resy && (*resy)(3) < param_.variance_depth_association){
                    if(!hit_this_cell || (*resy)(3) < local_min_var){
                        local_min_var = (*resy)(3);
                        best_pt_s = (*resy).head(3);
                        hit_this_cell = true;
                    }
                }
                // 检查 Z 面
                if(resz && (*resz)(3) < param_.variance_depth_association){
                    if(!hit_this_cell || (*resz)(3) < local_min_var){
                        local_min_var = (*resz)(3);
                        best_pt_s = (*resz).head(3);
                        hit_this_cell = true;
                    }
                }

                if (hit_this_cell) {
                    // 计算当前命中点在相机系下的深度
                    Vec3d pt_c = R_cs * best_pt_s + P_cs;
                    double depth_c = pt_c.z();

                    // 更新最近点逻辑
                    // 只有当这是一个更近的有效点时才更新
                    if (depth_cur_frame[i] < 0 || depth_c < depth_cur_frame[i]) {
                        depth_cur_frame[i] = depth_c;
                        xyz_submap[i] = best_pt_s;
                        xyz_world[i] = T_ws.block<3,3>(0,0) * best_pt_s + T_ws.block<3, 1>(0,3);
                        hit_cells[i] = cell;
                        found_valid_hit = true;
                    }
                }
            }
        }
    });
}

/* void LioMesher::AssociateDepthToFeatures(std::shared_ptr<ingvio::MapServer> mapserver, double timestamp){
    auto matching_submap = active_submaps_->matching_submap();
    const Mat4d& T_ws = matching_submap->pose_odom_;
    Mat4d T_sw = T_ws.inverse();
    Mat4d T_wc = se3_pose().matrix() * T_B_lC_; 
    Mat4d T_cw = T_wc.inverse(); 
    Vec3d P_wc = T_wc.block<3,1>(0,3);
    Mat3d R_wc = T_wc.block<3,3>(0,0);
    Mat4d T_sc = T_sw * T_wc;
    Mat4d T_cs = T_sc.inverse();
    Vec3d P_sc = T_sc.block<3, 1>(0,3);
    Mat3d R_sc = T_sc.block<3, 3>(0,0);

    std::shared_ptr<ingvio::SE3> anchor_pose(new ingvio::SE3());
    anchor_pose->setValueLinearByMat(R_wc);
    anchor_pose->setValueTrans(P_wc);

    double l = Parameter::GetInstance().grid;
    int n_row =  Parameter::GetInstance().num_test;
                
    for(auto& meas : *mapserver){
        std::shared_ptr<ingvio::FeatureInfo> feature = meas.second;
        std::vector<Cell*> cells_hit;
        if(feature->hasMonoObsAt(timestamp) && !feature->_isAssociateDepth){
            ingvio::MonoMeas mm = *(feature->_mono_obs[timestamp]);
            Vec3d dir_cam(mm._u0, mm._v0, 1.0);
            dir_cam.normalize();
            
            // 1. 找到与当前ray相交的cell
            int max_step = param_.range_max / l;
            std::vector<const Cell*> cells_hit = matching_submap->getIntersectedCells(
                    Ray(P_sc, R_sc * dir_cam), max_step);
            
            if(cells_hit.empty()) continue;
            for(const Cell* cell: cells_hit){
                // cell在子地图坐标系下
                std::optional<Vec4d> resx = cell->rayCellIntersectionX(Ray(P_sc, R_sc*dir_cam));
                std::optional<Vec4d> resy = cell->rayCellIntersectionY(Ray(P_sc, R_sc*dir_cam));
                std::optional<Vec4d> resz = cell->rayCellIntersectionZ(Ray(P_sc, R_sc*dir_cam));
                if(resx){
                    feature->_isAssociateDepth = true;
                    feature->landmark()->resetAnchoredPose(anchor_pose);
                    Vec3d resx_s = T_ws.block<3,3>(0,0) * (*resx).head(3) + T_ws.block<3, 1>(0,3);
                    feature->landmark()->setValuePosXyz(resx_s);
                    // LOG(INFO)<<"Intersects at X: "<<resx->x()<<", "<<resx->y()<<", "<<resx->z();
                    break;
                }else if(resy){
                    feature->_isAssociateDepth = true;
                    feature->landmark()->resetAnchoredPose(anchor_pose);
                    Vec3d resy_s = T_ws.block<3,3>(0,0) * (*resy).head(3) + T_ws.block<3, 1>(0,3);
                    feature->landmark()->setValuePosXyz(resy_s);
                    // LOG(INFO)<<"Intersects at Y: "<<resy->x()<<", "<<resy->y()<<", "<<resy->z();
                    break;
                }else if(resz){
                    feature->_isAssociateDepth = true;
                    feature->landmark()->resetAnchoredPose(anchor_pose);
                    Vec3d resz_s = T_ws.block<3,3>(0,0) * (*resz).head(3) + T_ws.block<3, 1>(0,3);
                    feature->landmark()->setValuePosXyz(resz_s);
                    // LOG(INFO)<<"Intersects at Z: "<<resz->x()<<", "<<resz->y()<<", "<<resz->z();
                    break;
                }
            }
        } 
    }
} */

void LioMesher::ColorizePointCloud(pcl::PointCloud<pcl::PointXYZ>& input, pcl::PointCloud<pcl::PointXYZRGB>& output){
    size_t n = input.size();
    output.points.resize(n);
    for(int i = 0; i < n; i++){
        pcl::PointXYZRGB xyzrgb;
        xyzrgb.x = input[i].x;
        xyzrgb.y = input[i].y;
        xyzrgb.z = input[i].z;
        // xyzrgb.r = 0;
        // xyzrgb.g = 0;
        // xyzrgb.b = 0;

        if(param_.enable_camera) {
            Vec3d p_lc = transformToX(input.points[i], T_B_lC_inv_);
            Vec3d p_rc = transformToX(input.points[i], T_B_rC_inv_);
            Vec2d uv;
            if(cur_img_pkg_ && InLeftCamera(p_lc, uv)){
                bool in_mask = false;
                if(!mask_.empty()){
                    const int u = floor(uv[0]);
                    const int v = floor(uv[1]);
                    int mm = mask_.at<uchar>(v, u);
                    if(mm == 0){
                        in_mask = true;
                    }
                }
                if(!in_mask){
                    Vec3f bgr = getInterpolatedPixel(cur_img_pkg_->left_img, uv);
                    xyzrgb.b = bgr(0);
                    xyzrgb.g = bgr(1);
                    xyzrgb.r = bgr(2);
                    // output.points.push_back(xyzrgb);
                }
            }else if(param_.enable_right_camera){
                if(cur_img_pkg_ && InRightCamera(p_rc, uv)){
                    Vec3f bgr = getInterpolatedPixel(cur_img_pkg_->right_img, uv);
                    xyzrgb.b = bgr(0);
                    xyzrgb.g = bgr(1);
                    xyzrgb.r = bgr(2);
                }
            }
        }else{
            xyzrgb.r = 128;
            xyzrgb.g = 128;
            xyzrgb.b = 128;
        }
        output.points[i] = xyzrgb;
    }   
}

void LioMesher::deskew_to_end() {
    auto imu_state = ieskf_.GetNominalState();  // 最后时刻的状态
    SE3 cur_pose = SE3(imu_state.R_, imu_state.p_);
    SE3 relative_pose = cur_pose.inverse() * last_pose_;
    
    size_t n = cur_scan_raw_->points.size();
    cur_scan_deskewed_->points.resize(n);
    std::vector<size_t> idx(n);
    iota(idx.begin(), idx.end(),  0);
    
    double t_start = cur_scan_raw_->points.front().t;
    double t_end = cur_scan_raw_->points.back().t;
    double t_span = t_end - t_start;
    // LOG(INFO)<<n<<", "<<t_start<<", "<<t_end<<", "<<t_span;
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        auto pi = cur_scan_raw_->points[i];//cur_scan_raw_ should be in imu frame
        Vec3d p={pi.x, pi.y, pi.z};
        
        double relative_t = (cur_scan_raw_->points[i].t - t_start) / t_span;
        double s = 1. - relative_t;//relative to end
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity().slerp(s, relative_pose.unit_quaternion());
        Vec3d t = s * relative_pose.translation();
        
        Vec3d pi_ds = SO3(q) * p + t;
        PointXYZT pt;
        pt.x = pi_ds(0);
        pt.y = pi_ds(1);
        pt.z = pi_ds(2);
        pt.t = 0.;
        cur_scan_deskewed_->points[i] = pt;
    });
}

void LioMesher::deskew_to_mid() {
    auto imu_state = ieskf_.GetNominalState();  // 最后时刻的状态
    SE3 cur_pose = SE3(imu_state.R_, imu_state.p_);
    SE3 relative_pose = last_pose_.inverse() * cur_pose;
    
    size_t n = cur_scan_raw_->points.size();
    cur_scan_deskewed_->points.resize(n);
    std::vector<size_t> idx(n);
    iota(idx.begin(), idx.end(),  0);

    double t_start = cur_scan_raw_->points.front().t;
    double t_end = cur_scan_raw_->points.back().t;
    double t_span = t_end - t_start;

    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        auto pi = cur_scan_raw_->points[i];//cur_scan_raw_ should be in imu frame
        Vec3d p={pi.x, pi.y, pi.z};
        
        double relative_t = (cur_scan_raw_->points[i].t - t_start) / t_span;
        double s = relative_t - 0.5;//relative to mid
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity().slerp(s, relative_pose.unit_quaternion());
        Vec3d t = s * relative_pose.translation();
        
        Vec3d pi_ds = SO3(q) * p + t;
        PointXYZT pt;
        pt.x = pi_ds(0);
        pt.y = pi_ds(1);
        pt.z = pi_ds(2);
        pt.t = 0.;
        cur_scan_deskewed_->points[i] = pt;
    });
}

bool LioMesher::initialize(const double ts, const Mat3d& rotation, const Vec3d& bg, const Vec3d& ba){
    Eigen::Vector3d g_vec(0., 0., -std::abs(param_.gravity));//TODO
    sad::IESKFD::Options option;
    option.gyro_std_ = param_.gyroscope_std;
    option.acce_std_ = param_.accelerometer_std;
    option.bias_gyro_std_ = param_.gyroscope_random_walk;
    option.bias_acce_std_ = param_.accelerometer_random_walk;
    if(param_.enable_imu){
        option.update_bias_gyro_ = true;
        option.update_bias_acce_ = true;
    }else{
        option.update_bias_gyro_ = false;
        option.update_bias_acce_ = false;
    }
    
    Vec3d zero={0,0,0};
    ieskf_.SetInitialConditions(option, bg, ba, g_vec);
    sad::NavStated initial_state(ts, SO3(rotation), zero, zero, bg, ba);
    ieskf_.SetX(initial_state);
    auto imu_state = ieskf_.GetNominalState();
    last_pose_ = SE3(imu_state.R_, imu_state.p_);

    T_l_ = last_pose_;
    T_ll_ = last_pose_;

    _hasInitState = true;
    initialized_ = true;
    last_imu_timestamp_ = ts;

    return true;
}

Mat4d LioMesher::pose(){
    auto pose = se3_pose().matrix();
    return pose;
}


/* void LioMesher::ComputeJointResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr){
    // LiDAR residuals and Jacobians
    {
        auto matching_submap = active_submaps_->matching_submap();
        Mat3d rot_var = ieskf_.GetCov().block<3, 3>(0, 0);
        Mat3d t_var = ieskf_.GetCov().block<3, 3>(3, 3);
        
        std::vector<Point2MeshMatch> matches;
        matching_submap->findMatchPointToVertices(*cur_scan_deskewed_xyz_ds_, pose.matrix(), matches);  
    
        size_t n = matches.size();
        std::vector<int> index(n);
        iota(index.begin(), index.end(),  0);

        std::vector<bool> effect_pts(n, false);
        std::vector<Eigen::Matrix<double, 1, 18>> H(n);
        std::vector<double> res(n);
        std::vector<double> infos(n);
        double lidar_measure_cov = std::pow(param_.lidar_measurement_std, 2);

        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            Point2MeshMatch p2m = matches[idx];
            // pcl::PointXYZ pt_raw = cur_scan_deskewed_xyz_ds_->points[idx];
            Mat3d cov_body = cur_scan_deskewed_cov_ds_[idx];
            if(p2m.valid && p2m.variance < param_.variance_register){
                Vec3d raw_pt = p2m.point_in_body;
                Vec3d norm_vec = p2m.plane.topRows(3);
                double r = p2m.point.dot(p2m.plane.topRows(3)) + p2m.plane(3);
                
                if(abs(r) < param_.residual_threshold){
                    Eigen::Matrix<double, 1, 18> Hi;
                    Hi.setZero();
                    Mat3d point_crossmat = SO3::hat(raw_pt);
                    Mat3d jr_inv = SO3::jr_inv(pose.so3());

                    Hi.block<1, 3>(0, 0) = norm_vec.transpose(); 
                    // Hi.block<1, 3>(0, 6) = norm_vec.transpose() * (-pose.so3().matrix()*point_crossmat)*jr_inv;  // 要不要乘以右雅可比？jr_inv
                    Hi.block<1, 3>(0, 6) = point_crossmat * pose.so3().matrix().transpose() * norm_vec; 

                    H[idx] = Hi;
                    res[idx] = -r;
                    infos[idx] = 1. / lidar_measure_cov;//

                    // Mat3d cov = pose.so3().matrix() * cov_body * pose.so3().matrix().transpose() 
                    //             + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;;
                    // infos[idx] = 1. / (1e-3 + norm_vec.transpose() * cov * norm_vec);//p2m.variance
                    effect_pts[idx] = true;
                }            
            }     
        });
        HTVH.setZero();
        HTVr.setZero();
        
        size_t num=0;
        for (int idx = 0; idx < effect_pts.size(); ++idx) {
            if (!effect_pts[idx]) {
                continue;
            }
            num++;
            HTVH += H[idx].transpose() * infos[idx] * H[idx];
            HTVr += H[idx].transpose() * infos[idx] * res[idx];
        }
    }


    // Visual residuals and Jacobians
    {
        Mat4d T_wi = pose.matrix();
        Eigen::VectorXd res_block_tracker, res_block_left, res_block_right;
        Eigen::MatrixXd H_block_tracker, H_block_left, H_block_right;
        Eigen::MatrixXd Hf_block_tracker, Hf_block_left, Hf_block_right;

        Eigen::MatrixXd H_block_tmp;
        Eigen::MatrixXd Hf_block_tmp;
        Eigen::VectorXd  res_block_tmp;
        
        VisualResidualAndJacobiansHelperTracker(left_camera_, left_feature_tracker_, depth_filter_left_, T_wi, T_B_lC_inv_, 
                        res_block_left, H_block_left, Hf_block_left);
        if(param_.enable_right_camera){
            VisualResidualAndJacobiansHelperTracker(right_camera_, right_feature_tracker_, depth_filter_right_, T_wi, T_B_rC_inv_, 
                            res_block_right, H_block_right, Hf_block_right);
            H_block_tracker = Eigen::MatrixXd(H_block_left.rows() + H_block_right.rows(), H_block_left.cols());
            H_block_tracker << H_block_left, H_block_right;
            Hf_block_tracker = Eigen::MatrixXd(Hf_block_left.rows() + Hf_block_right.rows(), Hf_block_left.cols());
            Hf_block_tracker << Hf_block_left, Hf_block_right;
            res_block_tracker = Eigen::VectorXd(res_block_left.rows() + res_block_right.rows());
            res_block_tracker << res_block_left, res_block_right;
        }else{
            H_block_tracker = H_block_left;
            Hf_block_tracker = Hf_block_left;
            res_block_tracker = res_block_left;
        }
        
        if(param_.enable_vmap_constraint){
            Eigen::VectorXd res_block_submap, res_block_submap_left, res_block_submap_right;
            Eigen::MatrixXd H_block_submap, H_block_submap_left, H_block_submap_right;//dr/dx
            Eigen::MatrixXd Hf_block_submap, Hf_block_submap_left, Hf_block_submap_right;//dr/df
            VisualResidualAndJacobiansHelperSubmap(left_camera_, left_feature_tracker_, T_wi, T_B_lC_inv_, 
                            res_block_submap_left, H_block_submap_left, Hf_block_submap_left);
            
            if(param_.enable_right_camera){
                VisualResidualAndJacobiansHelperSubmap(right_camera_, right_feature_tracker_, T_wi, T_B_rC_inv_, 
                                res_block_submap_right, H_block_submap_right, Hf_block_submap_right);
                H_block_submap = Eigen::MatrixXd(H_block_submap_left.rows() + H_block_submap_right.rows(), H_block_submap_left.cols());
                H_block_submap << H_block_submap_left, H_block_submap_right;
                Hf_block_submap = Eigen::MatrixXd(Hf_block_submap_left.rows() + Hf_block_submap_right.rows(), Hf_block_submap_left.cols());
                Hf_block_submap << Hf_block_submap_left, Hf_block_submap_right;
                res_block_submap = Eigen::VectorXd(res_block_submap_left.rows() + res_block_submap_right.rows());
                res_block_submap << res_block_submap_left, res_block_submap_right;
            }else{
                H_block_submap = H_block_submap_left;
                Hf_block_submap = Hf_block_submap_left;
                res_block_submap = res_block_submap_left;
            }
            // LOG(INFO)<<H_block_submap_left.rows()<<", "<<H_block_submap_right.rows();
            
            H_block_tmp = Eigen::MatrixXd(H_block_tracker.rows() + H_block_submap.rows(), H_block_tracker.cols());
            H_block_tmp << H_block_tracker, H_block_submap;
            Hf_block_tmp = Eigen::MatrixXd(Hf_block_tracker.rows() + Hf_block_submap.rows(), Hf_block_tracker.cols());
            Hf_block_tmp << Hf_block_tracker, Hf_block_submap;
            res_block_tmp = Eigen::VectorXd(res_block_tracker.rows() + res_block_submap.rows());
            res_block_tmp << res_block_tracker, res_block_submap;
        }else{
            H_block_tmp = H_block_tracker;
            Hf_block_tmp = Hf_block_tracker;
            res_block_tmp = res_block_tracker;
        }
        

        int row_cnt = H_block_tmp.rows();
        if (row_cnt <= 3){
            // LOG(WARNING) << "In calc one feat selected jacobian, num of rows <= 3 !";
            return;
        }
           
        // SVD分解，把对地图点的约束转换到位姿上
        Eigen::MatrixXd H_block = H_block_tmp;
        Eigen::VectorXd res_block = res_block_tmp;
        if(param_.marginalize_features){
            Eigen::JacobiSVD<Eigen::MatrixXd> svd_helper(Hf_block_tmp, Eigen::ComputeFullU | Eigen::ComputeThinV);
            Eigen::MatrixXd V = svd_helper.matrixU().rightCols(row_cnt-3);
            
            H_block = V.transpose()*H_block_tmp;
            res_block = V.transpose()*res_block_tmp;
        }
        
        int num = 0;
        double normal_uv_measure_cov = std::pow(param_.normal_uv_measurement_std, 2);
        double info = 1. / normal_uv_measure_cov;
        for (int i = 0; i < H_block.rows(); ++i) {
            num++;
            Eigen::Matrix<double, 1, 18> Hi;
            Hi.setZero();
            Hi.block<1, 3>(0, 0) = H_block.row(i).block<1, 3>(0, 0);
            Hi.block<1, 3>(0, 6) = H_block.row(i).block<1, 3>(0, 3);
            HTVH += Hi.transpose() * info * Hi;
            HTVr += Hi.transpose() * info * res_block[i];
        } 
    }
} */

void LioMesher::computeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr){
    auto matching_submap = active_submaps_->matching_submap();
    Mat3d rot_var = ieskf_.GetCov().block<3, 3>(0, 0);
    Mat3d t_var = ieskf_.GetCov().block<3, 3>(3, 3);

    //  在当前位姿、当前地图、当前观测点云下，点云到mesh的损失
    std::vector<Point2MeshMatch> matches;
    matching_submap->findMatchPointToVertices(*cur_scan_deskewed_xyz_ds_, pose.matrix(), matches);
     
    size_t n = matches.size();
    std::vector<int> index(n);
    iota(index.begin(), index.end(),  0);

    std::vector<bool> effect_pts(n, false);
    std::vector<Eigen::Matrix<double, 1, 18>> H(n);
    std::vector<double> res(n);
    std::vector<double> infos(n);
    double lidar_measure_cov = std::pow(param_.lidar_measurement_std, 2);

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
        Point2MeshMatch p2m = matches[idx];
        // pcl::PointXYZ pt_raw = cur_scan_deskewed_xyz_ds_->points[idx];
        // Mat3d cov_body = cur_scan_deskewed_cov_ds_[idx];
        if(p2m.valid && p2m.variance < param_.variance_register){
            Vec3d raw_pt = p2m.point_in_body;
            Vec3d norm_vec = p2m.plane.topRows(3);
            double r = p2m.point.dot(p2m.plane.topRows(3)) + p2m.plane(3);
            
            if(abs(r) < param_.residual_threshold){
                Eigen::Matrix<double, 1, 18> Hi;
                Hi.setZero();
                Mat3d point_crossmat = SO3::hat(raw_pt);
                Mat3d jr_inv = SO3::jr_inv(pose.so3());

                Hi.block<1, 3>(0, 0) = norm_vec.transpose(); 
                // Hi.block<1, 3>(0, 6) = norm_vec.transpose() * (-pose.so3().matrix()*point_crossmat)*jr_inv;  // 要不要乘以右雅可比？jr_inv
                // Hi.block<1, 3>(0, 6) = -norm_vec.transpose() * pose.so3().matrix() * point_crossmat;
                Hi.block<1, 3>(0, 6) = point_crossmat * pose.so3().matrix().transpose() * norm_vec; 

                H[idx] = Hi;
                res[idx] = -r;

                infos[idx] = 1. / lidar_measure_cov;//
                // double total_cov = lidar_measure_cov + p2m.variance * 0.1;
                // infos[idx] = 1.0 / total_cov;

                // Mat3d cov = pose.so3().matrix() * cov_body * pose.so3().matrix().transpose() 
                //             + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;;
                // infos[idx] = 1. / (1e-3 + norm_vec.transpose() * cov * norm_vec);//p2m.variance + 
                effect_pts[idx] = true;
            }            
        }     
    });
    
    
    HTVH.setZero();
    HTVr.setZero();
    
    size_t num=0;
    for (int idx = 0; idx < effect_pts.size(); ++idx) {
        if (!effect_pts[idx]) {
            continue;
        }
        num++;
        HTVH += H[idx].transpose() * infos[idx] * H[idx];
        HTVr += H[idx].transpose() * infos[idx] * res[idx];
    }

    if(param_.enable_wheel_encoder && param_.enable_nhc){
        Vec3d v_W = ieskf_.GetNominalState().v_; 
        Mat3d R_WB = pose.so3().matrix();
        Vec3d v_B_est = R_WB.transpose() * v_W;
        Vec3d r_w = cur_wheel_encoder_.linear_velocity_ - v_B_est; 
        Eigen::Matrix<double, 3, 18> H_w = Eigen::Matrix<double, 3, 18>::Zero();
        
        H_w.block<3, 3>(0, 3) = -R_WB.transpose();                
        H_w.block<3, 3>(0, 6) = -SO3::hat(v_B_est);

        Mat3d wheel_info = Mat3d::Zero();
        wheel_info(0, 0) = 1.0 / std::pow(param_.wheel_vel_std, 2); // 前向
        wheel_info(1, 1) = 1.0 / std::pow(param_.wheel_nhc_std, 2); // 横向 (NHC)
        wheel_info(2, 2) = 1.0 / std::pow(param_.wheel_nhc_std, 2); // 垂向 (NHC)

        HTVH += H_w.transpose() * wheel_info * H_w;
        HTVr += H_w.transpose() * wheel_info * r_w;
    }
}

/* void LioMesher::VisualResidualAndJacobiansHelperSubmap(const camodocal::CameraPtr camera, 
                                          const std::shared_ptr<FeatureTracker>& feature_tracker,
                                          const Mat4d& T_wi, const Mat4d& T_ci,
                                          Eigen::VectorXd& res, Eigen::MatrixXd& H_block, Eigen::MatrixXd& Hf_block){
    // 只需要call一次，可缓存结果提高迭代配准的效率
    std::vector<Vec3d> vmap_points_w;
    std::vector<cv::Point2f> uvs;
    FindCorrespondenceInVisualMap(camera, feature_tracker, T_wi, T_ci, vmap_points_w, uvs);
    
    // LOG(INFO)<<vmap_points_w.size()<<", "<<uvs.size();

    if(vmap_points_w.empty() || uvs.empty()) return;

    int num_of_obs = vmap_points_w.size();
    const int num_of_cols = 6;
    const int num_of_rows = 2 * num_of_obs;
    Eigen::VectorXd res_block_tmp = Eigen::VectorXd::Zero(num_of_rows);
    Eigen::MatrixXd H_block_tmp = Eigen::MatrixXd::Zero(num_of_rows, 6);//dr/dx
    Eigen::MatrixXd Hf_block_tmp = Eigen::MatrixXd::Zero(num_of_rows, 3);//dr/df
    
    Mat4d T_cw = T_ci * (T_wi.inverse()); 
    Mat4d T_wc = T_cw.inverse();
    Mat3d R_ci = T_ci.block<3, 3>(0, 0);
    Vec3d p_ci = T_ci.block<3, 1>(0, 3);
    Mat3d R_wi = T_wi.block<3, 3>(0, 0);
    Vec3d p_wi = T_wi.block<3, 1>(0, 3);

    int row_cnt = 0;
    for(int i = 0; i < num_of_obs; ++i){
        // 得到相机坐标系下的地图点位置
        const Eigen::Vector3d pf_w = vmap_points_w[i];
        const cv::Point2f uv = uvs[i];

        const Eigen::Matrix3d R_cm2w = T_wc.block<3, 3>(0, 0);
        const Eigen::Vector3d p_cm = T_wc.block<3, 1>(0, 3);
        const Eigen::Vector3d pf_cm = R_cm2w.transpose()*(pf_w-p_cm);
        
        Vec2d est_uv;
        camera->spaceToPlane(pf_cm, est_uv);
        // LOG(INFO)<<Vec2d(est_uv[0]-uv.x, est_uv[1]-uv.y).norm();
        if(Vec2d(est_uv[0]-uv.x, est_uv[1]-uv.y).norm() > param_.visual_residual_threshold) continue;

        Vec3d uv_n;
        camera->liftProjective(Vec2d(uv.x, uv.y), uv_n);
        uv_n[0] /= uv_n[2];
        uv_n[1] /= uv_n[2];
        
        // 归一化坐标对相机坐标系下的地图点的雅可比，Perspective Projection Function
        Eigen::Matrix<double, 2, 3> H_proj = Eigen::Matrix<double, 2, 3>::Zero();
        H_proj(0, 0) = 1.0/pf_cm.z();
        H_proj(0, 2) = -pf_cm.x()/std::pow(pf_cm.z(), 2);
        H_proj(1, 1) = 1.0/pf_cm.z();
        H_proj(1, 2) = -pf_cm.y()/std::pow(pf_cm.z(), 2);
        
        // 地图点对相机位姿的雅可比，Euclidean Transformation
        Vec3d p_fi = pf_w - p_wi;
        Mat3d H_t2pfw = R_ci * R_wi.transpose();//3x3
        Mat3d H_t2Rwi = -R_ci * R_wi.transpose() * sad::math::SKEW_SYM_MATRIX(p_fi(0),p_fi(1), p_fi(2));//3x3
        Mat3d H_t2pwi = -R_ci * R_wi.transpose();//3x3
        
        Eigen::Matrix<double, 3, 6> H_t2x = Eigen::Matrix<double, 3, 6>::Zero();
        H_t2x.block<3, 3>(0, 0) = H_t2pwi;
        H_t2x.block<3, 3>(0, 3) = H_t2Rwi;

        if (H_proj.hasNaN() || H_t2x.hasNaN()) 
            continue;
        
        // 归一化坐标（观测）对相机位姿（状态）的雅可比
        H_block_tmp.block(row_cnt, 0, 2, 6) = H_proj*H_t2x;
        
        // 归一化坐标（观测）对地图点位置（全局xyz）的雅可比
        Hf_block_tmp.block(row_cnt, 0, 2, 3) = H_proj*H_t2pfw;
        
        Vec2d residual;
        residual << uv_n[0] - pf_cm.x()/pf_cm.z(), uv_n[1] - pf_cm.y() / pf_cm.z();

        // 归一化坐标（观测）与估计值的残差
        res_block_tmp.block(row_cnt, 0, 2, 1) = residual;

        row_cnt += 2;
    }
    
    if (row_cnt < num_of_rows){
        res_block_tmp.conservativeResize(row_cnt, 1);
        H_block_tmp.conservativeResize(row_cnt, 6);
        Hf_block_tmp.conservativeResize(row_cnt, 3);
    }
    res = res_block_tmp;
    H_block = H_block_tmp;
    Hf_block = Hf_block_tmp;
    // LOG(INFO)<<"Added " << row_cnt / 2 << " constraints of current view to submap.";
    // LOG(INFO)<<uvs.size() <<", "<< row_cnt / 2;
} */

/* void LioMesher::VisualResidualAndJacobiansHelperTracker(const camodocal::CameraPtr camera,
                                        const std::shared_ptr<FeatureTracker>& feature_tracker,
                                        SlidingWindowDepthFilter& depth_filter, 
                                        const Mat4d& T_wi, const Mat4d& T_ci,
                                        Eigen::VectorXd& res, Eigen::MatrixXd& H_block, Eigen::MatrixXd& Hf_block){
    const auto& cur_tracked_features = feature_tracker->getTrackedPoints();

    int num_of_obs = cur_tracked_features.size();
    const int num_of_cols = 6;
    const int num_of_rows = 2 * num_of_obs;
    Eigen::VectorXd res_block_tmp = Eigen::VectorXd::Zero(num_of_rows);
    Eigen::MatrixXd H_block_tmp = Eigen::MatrixXd::Zero(num_of_rows, 6);//dr/dx
    Eigen::MatrixXd Hf_block_tmp = Eigen::MatrixXd::Zero(num_of_rows, 3);//dr/df
    
    Mat4d T_cw = T_ci * T_wi.inverse(); 
    Mat4d T_wc = T_cw.inverse();
    Mat3d R_ci = T_ci.block<3, 3>(0, 0);
    Vec3d p_ci = T_ci.block<3, 1>(0, 3);
    Mat3d R_wi = T_wi.block<3, 3>(0, 0);
    Vec3d p_wi = T_wi.block<3, 1>(0, 3);

    int row_cnt = 0;
    double normal_uv_measure_cov = std::pow(param_.normal_uv_measurement_std, 2);
    for(int i = 0; i < num_of_obs; ++i){
        const TrackedPoint& pt =  cur_tracked_features[i];
        if(pt.trackCount < 2 || pt.id < 0) continue;

        const SlidingWindowDepthFilter::FeatureTrack* mpt = depth_filter.getFeatureTrackById(pt.id);
        if(mpt==nullptr) continue;
        if(param_.use_mature_points_in_eskf){
            if(!mpt->is_mature || mpt->latest().is_outlier) continue;
        }
        
        // 得到相机坐标系下的地图点位置
        Eigen::Vector3d pf_w = mpt->latest().updated_map_point;
        if(!mpt->is_mature){
            pf_w = mpt->latest().map_point;
        }
        const Eigen::Matrix3d R_cm2w = T_wc.block<3, 3>(0, 0);
        const Eigen::Vector3d p_cm = T_wc.block<3, 1>(0, 3);
        const Eigen::Vector3d pf_cm = R_cm2w.transpose()*(pf_w-p_cm);

        if(pf_cm[2] < 1.0 || pf_cm[2] > 50) continue;

        
        Vec2d est_uv;
        cv::Point2f uv = pt.position;
        camera->spaceToPlane(pf_cm, est_uv);
        if(Vec2d(est_uv[0]-uv.x, est_uv[1]-uv.y).norm() > param_.visual_residual_threshold) continue;

        Vec3d uv_n;
        camera->liftProjective(Vec2d(uv.x, uv.y), uv_n);
        uv_n[0] /= uv_n[2];
        uv_n[1] /= uv_n[2];
        
        // 归一化坐标对相机坐标系下的地图点的雅可比，Perspective Projection Function
        Eigen::Matrix<double, 2, 3> H_proj = Eigen::Matrix<double, 2, 3>::Zero();
        H_proj(0, 0) = 1.0/pf_cm.z();
        H_proj(0, 2) = -pf_cm.x()/std::pow(pf_cm.z(), 2);
        H_proj(1, 1) = 1.0/pf_cm.z();
        H_proj(1, 2) = -pf_cm.y()/std::pow(pf_cm.z(), 2);
        
        // 地图点对相机位姿的雅可比，Euclidean Transformation
        Vec3d p_fi = pf_w - p_wi;
        Mat3d H_t2pfw = R_ci * R_wi.transpose();//3x3
        Mat3d H_t2Rwi = -R_ci * R_wi.transpose() * sad::math::SKEW_SYM_MATRIX(p_fi(0),p_fi(1), p_fi(2));//3x3
        Mat3d H_t2pwi = -R_ci * R_wi.transpose();//3x3
        
        Eigen::Matrix<double, 3, 6> H_t2x = Eigen::Matrix<double, 3, 6>::Zero();
        H_t2x.block<3, 3>(0, 0) = H_t2pwi;
        H_t2x.block<3, 3>(0, 3) = H_t2Rwi;

        if (H_proj.hasNaN() || H_t2x.hasNaN()) 
            continue;
        
        // 归一化坐标（观测）对相机位姿（状态）的雅可比
        H_block_tmp.block(row_cnt, 0, 2, 6) = H_proj*H_t2x;
        
        // 归一化坐标（观测）对地图点位置（全局xyz）的雅可比
        Hf_block_tmp.block(row_cnt, 0, 2, 3) = H_proj*H_t2pfw;
        
        Vec2d residual;
        residual << uv_n[0] - pf_cm.x()/pf_cm.z(), uv_n[1] - pf_cm.y() / pf_cm.z();
      
        // 归一化坐标（观测）与估计值的残差
        res_block_tmp.block(row_cnt, 0, 2, 1) = residual;

        row_cnt += 2;
    }

    if (row_cnt < num_of_rows){
        res_block_tmp.conservativeResize(row_cnt, 1);
        H_block_tmp.conservativeResize(row_cnt, 6);
        Hf_block_tmp.conservativeResize(row_cnt, 3);
    }
    res = res_block_tmp;
    H_block = H_block_tmp;
    Hf_block = Hf_block_tmp;
    // LOG(INFO)<<"Visual constriants: "<<row_cnt/2;400~800
} */

void LioMesher::VisualResidualAndJacobiansHelperTracker(
    const camodocal::CameraPtr camera,
    const std::shared_ptr<FeatureTracker>& feature_tracker,
    SlidingWindowDepthFilter& depth_filter, 
    const Mat4d& T_wi, const Mat4d& T_ci,
    Mat18d& HTVH, Vec18d& HTVr)
{
    const auto& cur_tracked_features = feature_tracker->getTrackedPoints();
    
    Mat4d T_cw = T_ci * T_wi.inverse(); 
    Mat4d T_wc = T_cw.inverse();
    Mat3d R_ci = T_ci.block<3, 3>(0, 0);
    Vec3d p_ci = T_ci.block<3, 1>(0, 3);
    Mat3d R_wi = T_wi.block<3, 3>(0, 0);
    Vec3d p_wi = T_wi.block<3, 1>(0, 3);

    double normal_uv_measure_cov = std::pow(param_.normal_uv_measurement_std, 2);

    for(int i = 0; i < cur_tracked_features.size(); ++i){
        const TrackedPoint& pt = cur_tracked_features[i];
        if(pt.trackCount < 2 || pt.id < 0) continue;

        const auto mpt = depth_filter.getFeatureTrackById(pt.id);
        if(!mpt) continue;
        // if(param_.use_mature_points_in_eskf){
        //     if(!mpt->is_mature) continue;
        // }
        if(!mpt->is_initialized) continue;
        
        // 获取地图点位置
        Eigen::Vector3d pf_w;
        if(!depth_filter.getFeatureWorldPoint(pt.id, pf_w)){
            continue;
        }
        
        const Eigen::Matrix3d R_cm2w = T_wc.block<3, 3>(0, 0);
        const Eigen::Vector3d p_cm = T_wc.block<3, 1>(0, 3);
        const Eigen::Vector3d pf_cm = R_cm2w.transpose()*(pf_w - p_cm);

        if(pf_cm[2] < 0.5) continue;//|| pf_cm[2] > 80

        Vec2d est_uv;
        cv::Point2f uv = pt.position;
        camera->spaceToPlane(pf_cm, est_uv);
        if(Vec2d(est_uv[0]-uv.x, est_uv[1]-uv.y).norm() > param_.visual_residual_threshold) continue;

        Vec3d uv_n;
        camera->liftProjective(Vec2d(uv.x, uv.y), uv_n);
        uv_n[0] /= uv_n[2];
        uv_n[1] /= uv_n[2];
        
        // 归一化坐标对相机坐标系下地图点的雅可比 (Perspective Projection)
        Eigen::Matrix<double, 2, 3> H_proj = Eigen::Matrix<double, 2, 3>::Zero();
        H_proj(0, 0) = 1.0/pf_cm.z();
        H_proj(0, 2) = -pf_cm.x()/std::pow(pf_cm.z(), 2);
        H_proj(1, 1) = 1.0/pf_cm.z();
        H_proj(1, 2) = -pf_cm.y()/std::pow(pf_cm.z(), 2);
        
        // 地图点对相机位姿的雅可比 (Euclidean Transformation)
        Vec3d p_fi = pf_w - p_wi;
        Mat3d H_t2pfw = R_ci * R_wi.transpose();
        Mat3d H_t2Rwi = -R_ci * R_wi.transpose() * sad::math::SKEW_SYM_MATRIX(p_fi(0), p_fi(1), p_fi(2));
        Mat3d H_t2pwi = -R_ci * R_wi.transpose();
        
        Eigen::Matrix<double, 3, 6> H_t2x = Eigen::Matrix<double, 3, 6>::Zero();
        H_t2x.block<3, 3>(0, 0) = H_t2pwi;
        H_t2x.block<3, 3>(0, 3) = H_t2Rwi;

        if (H_proj.hasNaN() || H_t2x.hasNaN()) continue;
        
        Eigen::Matrix2d Info_uv;
        Info_uv = Eigen::Matrix2d::Identity() / normal_uv_measure_cov;
        // TODO

        /* if (!mpt->is_pure_visual) {
            double d_anchor = mpt->anchor_depth / pf_cm.z(); // 深度不确定性对像素空间权重的影响因子
            double inv_depth_var = mpt->inv_depth_var;

            // 对逆深度的雅可比向量 (2x1)
            Eigen::Vector3d v_w = -d_anchor * (pf_w - p_wi); // 使用 p_wi 近似 anchor
            Eigen::Matrix<double, 2, 3> H_f = H_proj * H_t2pfw;
            Eigen::Vector2d J_rho = H_f * v_w;

            // 计算该像素点因深度不确定性引发的协方差
            Eigen::Matrix2d Sigma_uv = J_rho * inv_depth_var * J_rho.transpose();
            Sigma_uv(0, 0) += normal_uv_measure_cov; // 基础像素噪声
            Sigma_uv(1, 1) += normal_uv_measure_cov;
            
            Info_uv = Sigma_uv.inverse(); // 该点最终的权重矩阵
            // Info_uv = Eigen::Matrix2d::Identity() / normal_uv_measure_cov;
        } else {
            // 纯视觉点退化为基础像素噪声权重
            Info_uv = Eigen::Matrix2d::Identity() / (normal_uv_measure_cov * 2);
        } */
        // ----------------------------------------------------------------------

        // 组装系统状态量雅可比 (18维状态)
        Eigen::Matrix<double, 2, 6> H_x = H_proj * H_t2x;
        Eigen::Matrix<double, 2, 18> H_state = Eigen::Matrix<double, 2, 18>::Zero();
        H_state.block<2, 3>(0, 0) = H_x.block<2, 3>(0, 0); // Position
        H_state.block<2, 3>(0, 6) = H_x.block<2, 3>(0, 3); // Rotation

        // 计算残差
        Vec2d residual;
        residual << uv_n[0] - pf_cm.x()/pf_cm.z(), uv_n[1] - pf_cm.y()/pf_cm.z();

        // 直接累加到全局的 HTVH 和 HTVr
        HTVH += H_state.transpose() * Info_uv * H_state;
        HTVr += H_state.transpose() * Info_uv * residual;
    }
}

void LioMesher::VisualResidualAndJacobiansHelperSubmap(
    const camodocal::CameraPtr camera, 
    const std::shared_ptr<FeatureTracker>& feature_tracker,
    const Mat4d& T_wi, const Mat4d& T_ci,
    Mat18d& HTVH, Vec18d& HTVr) // 修改点：直接传入 HTVH 和 HTVr 引用
{
    std::vector<Vec3d> vmap_points_w;
    std::vector<cv::Point2f> uvs;
    FindCorrespondenceInVisualMap(camera, feature_tracker, T_wi, T_ci, vmap_points_w, uvs);
    
    if(vmap_points_w.empty() || uvs.empty()) return;

    Mat4d T_cw = T_ci * (T_wi.inverse()); 
    Mat4d T_wc = T_cw.inverse();
    Mat3d R_ci = T_ci.block<3, 3>(0, 0);
    Vec3d p_ci = T_ci.block<3, 1>(0, 3);
    Mat3d R_wi = T_wi.block<3, 3>(0, 0);
    Vec3d p_wi = T_wi.block<3, 1>(0, 3);

    // VMap 匹配点没有直接可用的 inv_depth_var，采用静态置信度
    double normal_uv_measure_cov = std::pow(param_.normal_uv_measurement_std, 2);
    Eigen::Matrix2d Info_uv = Eigen::Matrix2d::Identity() / normal_uv_measure_cov;

    size_t n = vmap_points_w.size();
    std::vector<int> index(n);
    iota(index.begin(), index.end(),  0);
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx){
        const Eigen::Vector3d pf_w = vmap_points_w[idx];
        const cv::Point2f uv = uvs[idx];

        const Eigen::Matrix3d R_cm2w = T_wc.block<3, 3>(0, 0);
        const Eigen::Vector3d p_cm = T_wc.block<3, 1>(0, 3);
        const Eigen::Vector3d pf_cm = R_cm2w.transpose()*(pf_w-p_cm);
        
        Vec2d est_uv;
        camera->spaceToPlane(pf_cm, est_uv);
        if(Vec2d(est_uv[0]-uv.x, est_uv[1]-uv.y).norm() < param_.visual_residual_threshold){
            Vec3d uv_n;
            camera->liftProjective(Vec2d(uv.x, uv.y), uv_n);
            uv_n[0] /= uv_n[2];
            uv_n[1] /= uv_n[2];
            
            Eigen::Matrix<double, 2, 3> H_proj = Eigen::Matrix<double, 2, 3>::Zero();
            H_proj(0, 0) = 1.0/pf_cm.z();
            H_proj(0, 2) = -pf_cm.x()/std::pow(pf_cm.z(), 2);
            H_proj(1, 1) = 1.0/pf_cm.z();
            H_proj(1, 2) = -pf_cm.y()/std::pow(pf_cm.z(), 2);
            
            Vec3d p_fi = pf_w - p_wi;
            Mat3d H_t2pfw = R_ci * R_wi.transpose();
            Mat3d H_t2Rwi = -R_ci * R_wi.transpose() * sad::math::SKEW_SYM_MATRIX(p_fi(0),p_fi(1), p_fi(2));
            Mat3d H_t2pwi = -R_ci * R_wi.transpose();
            
            Eigen::Matrix<double, 3, 6> H_t2x = Eigen::Matrix<double, 3, 6>::Zero();
            H_t2x.block<3, 3>(0, 0) = H_t2pwi;
            H_t2x.block<3, 3>(0, 3) = H_t2Rwi;

            if (H_proj.hasNaN() || H_t2x.hasNaN()){
                
            }else{
                Eigen::Matrix<double, 2, 6> H_x = H_proj * H_t2x;
                Vec2d residual;
                residual << uv_n[0] - pf_cm.x()/pf_cm.z(), uv_n[1] - pf_cm.y() / pf_cm.z();

                Eigen::Matrix<double, 2, 18> H_state = Eigen::Matrix<double, 2, 18>::Zero();
                H_state.block<2, 3>(0, 0) = H_x.block<2, 3>(0, 0); 
                H_state.block<2, 3>(0, 6) = H_x.block<2, 3>(0, 3); 

                HTVH += H_state.transpose() * Info_uv * H_state;
                HTVr += H_state.transpose() * Info_uv * residual;
            }
        }        
    });
}

void LioMesher::ComputeJointResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr){
    // 确保每次调用前清零
    HTVH.setZero();
    HTVr.setZero();

    // =======================================================
    // 1. LiDAR residuals and Jacobians
    // =======================================================
    {
        auto matching_submap = active_submaps_->matching_submap();
        std::vector<Point2MeshMatch> matches;
        
        matching_submap->findMatchPointToVertices(*cur_scan_deskewed_xyz_ds_, pose.matrix(), matches);  
    
        size_t n = matches.size();
        std::vector<int> index(n);
        iota(index.begin(), index.end(),  0);

        std::vector<bool> effect_pts(n, false);
        std::vector<Eigen::Matrix<double, 1, 18>> H(n);
        std::vector<double> res(n);
        std::vector<double> info_dynamic(n, 0.0);
        
        double lidar_measure_cov = std::pow(param_.lidar_measurement_std, 2);
        double info_lidar = 1.0 / lidar_measure_cov;

        // 并行计算局部雅可比，避免竞争条件
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            Point2MeshMatch p2m = matches[idx];
            if(p2m.valid && p2m.variance < param_.variance_register){
                Vec3d raw_pt = p2m.point_in_body;
                Vec3d norm_vec = p2m.plane.topRows(3);
                double r = p2m.point.dot(p2m.plane.topRows(3)) + p2m.plane(3);
                
                if(abs(r) < param_.residual_threshold){
                    Eigen::Matrix<double, 1, 18> Hi;
                    Hi.setZero();
                    Mat3d point_crossmat = SO3::hat(raw_pt);

                    Hi.block<1, 3>(0, 0) = norm_vec.transpose(); 
                    Hi.block<1, 3>(0, 6) = point_crossmat * pose.so3().matrix().transpose() * norm_vec; 

                    H[idx] = Hi;
                    res[idx] = -r;
                    effect_pts[idx] = true;

                    // double total_cov = lidar_measure_cov + p2m.variance * 0.1;
                    // info_dynamic[idx] = 1.0 / total_cov;
                    info_dynamic[idx] = info_lidar;
                }            
            }     
        });
        
        // 串行累加至全局矩阵
        for (int idx = 0; idx < effect_pts.size(); ++idx) {
            if (!effect_pts[idx]) continue;
            HTVH += H[idx].transpose() * info_dynamic[idx] * H[idx];
            HTVr += H[idx].transpose() * info_dynamic[idx] * res[idx];
        }
    }

    // =======================================================
    // 2. Visual residuals and Jacobians
    // =======================================================
    {
        Mat4d T_wi = pose.matrix();
        
        VisualResidualAndJacobiansHelperTracker(left_camera_, left_feature_tracker_, depth_filter_left_, T_wi, T_B_lC_inv_, HTVH, HTVr);
        if(param_.enable_right_camera){
            VisualResidualAndJacobiansHelperTracker(right_camera_, right_feature_tracker_, depth_filter_right_, T_wi, T_B_rC_inv_, HTVH, HTVr);
        }

        if(param_.enable_vmap_constraint){
            VisualResidualAndJacobiansHelperSubmap(left_camera_, left_feature_tracker_, T_wi, T_B_lC_inv_, HTVH, HTVr);
            if(param_.enable_right_camera){
                VisualResidualAndJacobiansHelperSubmap(right_camera_, right_feature_tracker_, T_wi, T_B_rC_inv_, HTVH, HTVr);
            }
        }
    }

    // NHC约束
    if(param_.enable_wheel_encoder && param_.enable_nhc){
        // 请将下面两行替换为你系统里实际获取速度和轮速的变量
        Vec3d v_w = ieskf_.GetNominalState().v_;    // 当前系统预测的世界系速度
        double v_odom = cur_wheel_encoder_.linear_velocity_.x(); // 轮速计提供的前向测量速度

        Mat3d R_wi = pose.so3().matrix();
        
        // 1. 将世界系速度投影到机体系 (IMU系)
        Vec3d v_ib = R_wi.transpose() * v_w; 

        // 2. 构建残差: 测量值 - 预测值
        // 假设机体系为: X向前, Y向左, Z向上 (若你的坐标系不同, 请调整索引)
        Vec3d res_nhc;
        res_nhc << v_odom - v_ib.x(), // 前向速度残差
                0.0 - v_ib.y(),    // 侧滑速度应为 0 (NHC)
                0.0 - v_ib.z();    // 垂向速度应为 0 (NHC)

        // 3. 构建 3 x 18 的雅可比矩阵
        Eigen::Matrix<double, 3, 18> H_nhc = Eigen::Matrix<double, 3, 18>::Zero();

        // 速度状态的雅可比 (假设 18维状态向量中，Velocity 位于 index 3-5)
        H_nhc.block<3, 3>(0, 3) = -R_wi.transpose();

        // 旋转状态的雅可比 (假设 18维状态向量中，Rotation 位于 index 6-8)
        H_nhc.block<3, 3>(0, 6) = -SO3::hat(v_ib); 
        // 数学解释: 对旋转扰动的导数等于负的预测机体速度的反对称矩阵

        // 4. 构建信息矩阵 (权重)
        // 这里的方差可以根据路面情况调整，通常设置在 0.05m/s ~ 0.1m/s 左右
        Eigen::Matrix3d Info_nhc = Eigen::Matrix3d::Zero();
        Info_nhc(0, 0) = 1.0 / std::pow(0.05, 2); // X方向：轮速计测量置信度较高
        Info_nhc(1, 1) = 1.0 / std::pow(0.05, 2); // Y方向：允许轻微侧滑，给一定容忍度
        Info_nhc(2, 2) = 1.0 / std::pow(0.08, 2); // Z方向：考虑到路面颠簸，垂向方差可稍微放大

        // wheel_info(0, 0) = 1.0 / std::pow(param_.wheel_vel_std, 2); // 前向
        // wheel_info(1, 1) = 1.0 / std::pow(param_.wheel_nhc_std, 2); // 横向 (NHC)
        // wheel_info(2, 2) = 1.0 / std::pow(param_.wheel_nhc_std, 2); // 垂向 (NHC)

        // 5. 累加到全局系统矩阵中
        HTVH += H_nhc.transpose() * Info_nhc * H_nhc;
        HTVr += H_nhc.transpose() * Info_nhc * res_nhc;
    }
}

/* void LioMesher::ComputeJointResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr){
    // 确保传入的全局矩阵被清零
    HTVH.setZero();
    HTVr.setZero();

    // 分配独立的 LiDAR 和 Vision 系统矩阵
    Mat18d HTVH_L = Mat18d::Zero(); Vec18d HTVr_L = Vec18d::Zero();
    Mat18d HTVH_V = Mat18d::Zero(); Vec18d HTVr_V = Vec18d::Zero();
    Mat18d HTVH_N = Mat18d::Zero(); Vec18d HTVr_N = Vec18d::Zero();

    // =======================================================
    // 1. LiDAR residuals and Jacobians (动态方差加权)
    // =======================================================
    {
        auto matching_submap = active_submaps_->matching_submap();
        std::vector<Point2MeshMatch> matches;
        matching_submap->findMatchPointToVertices(*cur_scan_deskewed_xyz_ds_, pose.matrix(), matches);  

        size_t n = matches.size();
        std::vector<int> index(n);
        iota(index.begin(), index.end(), 0);

        std::vector<bool> effect_pts(n, false);
        std::vector<Eigen::Matrix<double, 1, 18>> H(n);
        std::vector<double> res(n);
        std::vector<double> info_dynamic(n, 0.0);
        
        double lidar_measure_cov = std::pow(param_.lidar_measurement_std, 2);

        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            Point2MeshMatch p2m = matches[idx];
            if(p2m.valid && p2m.variance < param_.variance_register){
                Vec3d raw_pt = p2m.point_in_body;
                Vec3d norm_vec = p2m.plane.topRows(3);
                double r = p2m.point.dot(p2m.plane.topRows(3)) + p2m.plane(3);
                
                if(abs(r) < param_.residual_threshold){
                    Eigen::Matrix<double, 1, 18> Hi;
                    Hi.setZero();
                    Mat3d point_crossmat = SO3::hat(raw_pt);

                    Hi.block<1, 3>(0, 0) = norm_vec.transpose();            // Position Jacobian
                    Hi.block<1, 3>(0, 6) = point_crossmat * pose.so3().matrix().transpose() * norm_vec; // Rotation Jacobian

                    H[idx] = Hi;
                    res[idx] = -r;
                    effect_pts[idx] = true;
                    
                    // 利用面片方差计算动态权重
                    // double total_cov = lidar_measure_cov + p2m.variance;
                    // info_dynamic[idx] = 1.0 / total_cov;
                    info_dynamic[idx] = 1.0 / lidar_measure_cov;
                }            
            }     
        });
        
        for (int idx = 0; idx < effect_pts.size(); ++idx) {
            if (!effect_pts[idx]) continue;
            HTVH_L += H[idx].transpose() * info_dynamic[idx] * H[idx];
            HTVr_L += H[idx].transpose() * info_dynamic[idx] * res[idx];
        }
    }

    // =======================================================
    // 2. Visual residuals and Jacobians (调用重构后的Helper)
    // =======================================================
    {
        Mat4d T_wi = pose.matrix();
        VisualResidualAndJacobiansHelperTracker(left_camera_, left_feature_tracker_, depth_filter_left_, T_wi, T_B_lC_inv_, HTVH_V, HTVr_V);
        if(param_.enable_right_camera){
            VisualResidualAndJacobiansHelperTracker(right_camera_, right_feature_tracker_, depth_filter_right_, T_wi, T_B_rC_inv_, HTVH_V, HTVr_V);
        }
        
        if(param_.enable_vmap_constraint){
            VisualResidualAndJacobiansHelperSubmap(left_camera_, left_feature_tracker_, T_wi, T_B_lC_inv_, HTVH_V, HTVr_V);
            if(param_.enable_right_camera){
                VisualResidualAndJacobiansHelperSubmap(right_camera_, right_feature_tracker_, T_wi, T_B_rC_inv_, HTVH_V, HTVr_V);
            }
        }
    }

    // =======================================================
    // 3. 自适应 6-DoF 零空间投影融合 (Core Contribution)
    // =======================================================
    {
        // a. 解耦提取平移和旋转的 Hessian 子块
        Eigen::Matrix3d H_L_pos = HTVH_L.block<3, 3>(0, 0); // Position Block (0-2)
        Eigen::Matrix3d H_L_rot = HTVH_L.block<3, 3>(6, 6); // Rotation Block (6-8)

        // b. 分别进行特征值分解
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_pos(H_L_pos);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_rot(H_L_rot);

        // 提取平移和旋转的最大特征值作为基准 (Baseline)
        double max_eig_pos = eigen_pos.eigenvalues().maxCoeff();
        double max_eig_rot = eigen_rot.eigenvalues().maxCoeff();

        // 设定相对阈值 (Condition Number Threshold)
        double deg_ratio_pos_threshold = 80.0; 
        double deg_ratio_rot_threshold = 100.0; 

        Eigen::Matrix3d P_pos = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d P_rot = Eigen::Matrix3d::Zero();
        int deg_pos_dims = 0, deg_rot_dims = 0;

        // c. 构建平移退化投影矩阵
        for (int i = 0; i < 3; ++i) {
            double current_eig = eigen_pos.eigenvalues()(i);
            // 防御性除零检查，并判断比值
            if (current_eig < 1e-6 || (max_eig_pos / current_eig) > deg_ratio_pos_threshold) {
                // 该方向严重退化，允许视觉约束介入
                P_pos += eigen_pos.eigenvectors().col(i) * eigen_pos.eigenvectors().col(i).transpose();
                deg_pos_dims++;
            }
        }

        // d. 构建旋转退化投影矩阵
        for (int i = 0; i < 3; ++i) {
            double current_eig = eigen_rot.eigenvalues()(i);
            if (current_eig < 1e-6 || (max_eig_rot / current_eig) > deg_ratio_rot_threshold) {
                // 该方向严重退化，允许视觉约束介入
                P_rot += eigen_rot.eigenvectors().col(i) * eigen_rot.eigenvectors().col(i).transpose();
                deg_rot_dims++;
            }
        }

        // 由于平移和旋转雅可比尺度不同，退化阈值需独立设置
        // LOG(INFO)<<"Position eigen values: "<<eigen_pos.eigenvalues()(0)<<","<<eigen_pos.eigenvalues()(1)<<","<<eigen_pos.eigenvalues()(2);
        // LOG(INFO)<<"Rotation eigen values: "<<eigen_rot.eigenvalues()(0)<<","<<eigen_rot.eigenvalues()(1)<<","<<eigen_rot.eigenvalues()(2);

        // e. 组装 18x18 的全局状态投影矩阵
        Eigen::Matrix<double, 18, 18> P_18 = Eigen::Matrix<double, 18, 18>::Identity();
        
        // 只有当存在退化时，才利用视觉去补充；如果没有退化，对应的 P 为 0，视觉在对应维度的贡献被屏蔽
        P_18.block<3, 3>(0, 0) = P_pos; // Position (0-2)
        P_18.block<3, 3>(6, 6) = P_rot; // Rotation (6-8)
        
        // 注意：速度(3-5)及各种Bias保持Identity，因为我们需要视觉协助修正速度和Bias

        // f. 对视觉的 Hessian 和残差应用严格正交子空间投影
        Mat18d HTVH_V_proj = P_18.transpose() * HTVH_V * P_18;
        Vec18d HTVr_V_proj = P_18.transpose() * HTVr_V;

        // 将投影后的视觉约束合并入全局
        HTVH_V = HTVH_V_proj;
        HTVr_V = HTVr_V_proj;
    }

    
    HTVH = HTVH_L + HTVH_V;
    HTVr = HTVr_L + HTVr_V;
} */

void LioMesher::computeResidualAndJacobiansReprojection(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr){
    // 纯视觉重投影情况
    HTVH.setZero();
    HTVr.setZero();

    Mat4d T_wi = pose.matrix();

    // 累加 Tracker 约束 (包含动态协方差处理)
    VisualResidualAndJacobiansHelperTracker(left_camera_, left_feature_tracker_, depth_filter_left_, T_wi, T_B_lC_inv_, HTVH, HTVr);
    
    if(param_.enable_right_camera){
        VisualResidualAndJacobiansHelperTracker(right_camera_, right_feature_tracker_, depth_filter_right_, T_wi, T_B_rC_inv_, HTVH, HTVr);
    }
    
    // 累加 Submap (VMap) 约束
    if(param_.enable_vmap_constraint){
        VisualResidualAndJacobiansHelperSubmap(left_camera_, left_feature_tracker_, T_wi, T_B_lC_inv_, HTVH, HTVr);
        
        if(param_.enable_right_camera){
            VisualResidualAndJacobiansHelperSubmap(right_camera_, right_feature_tracker_, T_wi, T_B_rC_inv_, HTVH, HTVr);
        }
    }
}

/* void LioMesher::computeResidualAndJacobiansReprojection(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr){
    Mat4d T_wi = pose.matrix();
    Eigen::VectorXd res_block_tracker, res_block_left, res_block_right;
    Eigen::MatrixXd H_block_tracker, H_block_left, H_block_right;
    Eigen::MatrixXd Hf_block_tracker, Hf_block_left, Hf_block_right;
    Eigen::MatrixXd H_block_tmp;
    Eigen::MatrixXd Hf_block_tmp;
    Eigen::VectorXd  res_block_tmp;

    VisualResidualAndJacobiansHelperTracker(left_camera_, left_feature_tracker_, depth_filter_left_, T_wi, T_B_lC_inv_, 
                    res_block_left, H_block_left, Hf_block_left);
    if(param_.enable_right_camera){
        VisualResidualAndJacobiansHelperTracker(right_camera_, right_feature_tracker_, depth_filter_right_, T_wi, T_B_rC_inv_, 
                        res_block_right, H_block_right, Hf_block_right);
        H_block_tracker = Eigen::MatrixXd(H_block_left.rows() + H_block_right.rows(), H_block_left.cols());
        H_block_tracker << H_block_left, H_block_right;
        Hf_block_tracker = Eigen::MatrixXd(Hf_block_left.rows() + Hf_block_right.rows(), Hf_block_left.cols());
        Hf_block_tracker << Hf_block_left, Hf_block_right;
        res_block_tracker = Eigen::VectorXd(res_block_left.rows() + res_block_right.rows());
        res_block_tracker << res_block_left, res_block_right;
    }else{
        H_block_tracker = H_block_left;
        Hf_block_tracker = Hf_block_left;
        res_block_tracker = res_block_left;
    }
    
    if(param_.enable_vmap_constraint){
        Eigen::VectorXd res_block_submap, res_block_submap_left, res_block_submap_right;
        Eigen::MatrixXd H_block_submap, H_block_submap_left, H_block_submap_right;//dr/dx
        Eigen::MatrixXd Hf_block_submap, Hf_block_submap_left, Hf_block_submap_right;//dr/df
        VisualResidualAndJacobiansHelperSubmap(left_camera_, left_feature_tracker_, T_wi, T_B_lC_inv_, 
                        res_block_submap_left, H_block_submap_left, Hf_block_submap_left);
        if(param_.enable_right_camera){
            VisualResidualAndJacobiansHelperSubmap(right_camera_, right_feature_tracker_, T_wi, T_B_rC_inv_, 
                            res_block_submap_right, H_block_submap_right, Hf_block_submap_right);
            H_block_submap = Eigen::MatrixXd(H_block_submap_left.rows() + H_block_submap_right.rows(), H_block_submap_left.cols());
            H_block_submap << H_block_submap_left, H_block_submap_right;
            Hf_block_submap = Eigen::MatrixXd(Hf_block_submap_left.rows() + Hf_block_submap_right.rows(), Hf_block_submap_left.cols());
            Hf_block_submap << Hf_block_submap_left, Hf_block_submap_right;
            res_block_submap = Eigen::VectorXd(res_block_submap_left.rows() + res_block_submap_right.rows());
            res_block_submap << res_block_submap_left, res_block_submap_right;
        }else{
            H_block_submap = H_block_submap_left;
            Hf_block_submap = Hf_block_submap_left;
            res_block_submap = res_block_submap_left;
        }
        
        
        H_block_tmp = Eigen::MatrixXd(H_block_tracker.rows() + H_block_submap.rows(), H_block_tracker.cols());
        H_block_tmp << H_block_tracker, H_block_submap;
        Hf_block_tmp = Eigen::MatrixXd(Hf_block_tracker.rows() + Hf_block_submap.rows(), Hf_block_tracker.cols());
        Hf_block_tmp << Hf_block_tracker, Hf_block_submap;
        res_block_tmp = Eigen::VectorXd(res_block_tracker.rows() + res_block_submap.rows());
        res_block_tmp << res_block_tracker, res_block_submap;
    }else{
        H_block_tmp = H_block_tracker;
        Hf_block_tmp = Hf_block_tracker;
        res_block_tmp = res_block_tracker;
    }
    

    int row_cnt = H_block_tmp.rows();
    if (row_cnt <= 3){
        // LOG(WARNING) << "In calc one feat selected jacobian, num of rows <= 3 !";
        return;
    }
        
    // SVD分解，把对地图点的约束转换到位姿上
    Eigen::MatrixXd H_block = H_block_tmp;
    Eigen::VectorXd res_block = res_block_tmp;
    if(param_.marginalize_features){
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_helper(Hf_block_tmp, Eigen::ComputeFullU | Eigen::ComputeThinV);
        Eigen::MatrixXd V = svd_helper.matrixU().rightCols(row_cnt-3);
        
        H_block = V.transpose()*H_block_tmp;
        res_block = V.transpose()*res_block_tmp;
    }
    

    HTVH.setZero();
    HTVr.setZero();
    int num = 0;
    double normal_uv_measure_cov = std::pow(param_.normal_uv_measurement_std, 2);
    double info = 1. / normal_uv_measure_cov;
    for (int i = 0; i < H_block.rows(); ++i) {
        num++;
        Eigen::Matrix<double, 1, 18> Hi;
        Hi.setZero();
        Hi.block<1, 3>(0, 0) = H_block.row(i).block<1, 3>(0, 0);
        Hi.block<1, 3>(0, 6) = H_block.row(i).block<1, 3>(0, 3);
        HTVH += Hi.transpose() * info * Hi;
        HTVr += Hi.transpose() * info * res_block[i];
    }
} */


void LioMesher::computeResidualAndJacobiansXYZRGB(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr){
    auto matching_submap = active_submaps_->matching_submap();
    
    Mat4d cur_pose_guess = pose.matrix();
    std::vector<Vec3d> target_color_gradient;
    pcl::PointCloud<pcl::PointXYZRGBNormal> target_xyzrgbn;
    std::vector<bool> corr_flags = matching_submap->FindCorrespondences(
            *cur_scan_deskewed_xyzrgb_, cur_pose_guess, target_color_gradient, target_xyzrgbn);

    size_t n = corr_flags.size();
    std::vector<int> index(n);
    iota(index.begin(), index.end(),  0);

    std::vector<bool> effect_pts(n, false);
    std::vector<Eigen::Matrix<double, 1, 18>> H_xyz(n);
    std::vector<Eigen::Matrix<double, 1, 18>> H_rgb(n);
    std::vector<double> residual_xyz(n);
    std::vector<double> residual_rgb(n);
    std::vector<double> infos(n);
    double lidar_measure_cov = std::pow(param_.lidar_measurement_std, 2);
    double lambda = param_.lambda_color_icp;

    Vec3d rgb_to_intensity_weight = Eigen::Vector3d::Ones () / 3.0;
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
        if(corr_flags[idx]){
            const pcl::PointXYZRGB& src = cur_scan_deskewed_xyzrgb_->points[idx];
            const pcl::PointXYZRGBNormal& tgt = target_xyzrgbn.points[idx];
            
            const Eigen::Vector3d q_raw = src.getVector3fMap().cast<double>();
            const Eigen::Vector3d q = cur_pose_guess.block<3,3>(0,0) * q_raw + cur_pose_guess.block<3,1>(0,3);
            const Eigen::Vector3d cq = src.getRGBVector3i().cast<double>();
            double iq = cq.dot(rgb_to_intensity_weight) / 255.0;

            const Eigen::Vector3d p = tgt.getVector3fMap().cast<double>();
            const Eigen::Vector3d np = tgt.getNormalVector3fMap().cast<double>();
            const Eigen::Vector3d cp = tgt.getRGBVector3i().cast<double>();
            double ip = cp.dot(rgb_to_intensity_weight) / 255.0;
            const Eigen::Vector3d& dp = target_color_gradient[idx]; //J_cp, Eq. 29

            const Eigen::Vector3d& q_proj = q - (q-p).dot(np) * np;
            const double& iq_proj = ip + dp.dot(q_proj-p);
            const Eigen::Matrix3d& M = Eigen::Matrix3d::Identity() - np * np.transpose();//J_fs, Eq. 29
            const Eigen::Vector3d& dpM = dp.transpose() * M; //
            
            double r_xyz = (q - p).dot(np);
            double r_rgb = iq_proj - iq;
            
            Mat3d point_crossmat = SO3::hat(q_raw);
            Vec3d J_xyz_r = -np.transpose() * pose.so3().matrix() * point_crossmat;
            // Vec3d J_xyz_r = q.cross(np);//这个不对
            Vec3d J_xyz_p = np;
            // Vec3d J_rgb_r = q.cross(dpM);
            Vec3d J_rgb_r = -dpM.transpose() * pose.so3().matrix() * point_crossmat;
            Vec3d J_rgb_p = dpM;

            Eigen::Matrix<double, 1, 18> Hi_xyz;
            Eigen::Matrix<double, 1, 18> Hi_rgb;
            Hi_xyz.setZero();
            Hi_rgb.setZero();

            Hi_xyz.block<1, 3>(0, 0) = J_xyz_p.transpose(); 
            Hi_xyz.block<1, 3>(0, 6) = J_xyz_r.transpose();
            H_xyz[idx] = Hi_xyz;
            residual_xyz[idx] = -r_xyz;
            
            Hi_rgb.block<1, 3>(0, 0) = J_rgb_p.transpose(); 
            Hi_rgb.block<1, 3>(0, 6) = J_rgb_r.transpose();
            H_rgb[idx] = Hi_rgb;
            residual_rgb[idx] = -r_rgb;

            infos[idx] = 1. / lidar_measure_cov;//
            effect_pts[idx] = true;
        }
    });
    
    
    HTVH.setZero();
    HTVr.setZero();
    
    size_t num=0;
    for (int idx = 0; idx < n; ++idx) {
        if (!corr_flags[idx] || abs(residual_xyz[idx]) > param_.residual_threshold) {
            continue;
        }
        num++;
        auto JtJ_xyz = H_xyz[idx].transpose() * infos[idx] * H_xyz[idx];
        auto JtJ_rgb = H_rgb[idx].transpose() * infos[idx] * H_rgb[idx];
        auto Jtr_xyz = H_xyz[idx].transpose() * infos[idx] * residual_xyz[idx];
        auto Jtr_rgb = H_rgb[idx].transpose() * infos[idx] * residual_rgb[idx];
        HTVH += (sqrt(lambda) * JtJ_xyz + sqrt(1-lambda)*JtJ_rgb);
        HTVr += (sqrt(lambda) * Jtr_xyz + sqrt(1-lambda)*Jtr_rgb);
    }
}

void LioMesher::FindCorrespondenceInVisualMap(const camodocal::CameraPtr camera,
                                    const std::shared_ptr<FeatureTracker>& feature_tracker,
                                    const Mat4d& T_wi, const Mat4d& T_ci,
                                    std::vector<Vec3d>& vmap_points_w,
                                    std::vector<cv::Point2f>& uvs){    
    if(camera == left_camera_ && left_correspondence_fid_ == cur_frame_id_){
        vmap_points_w = left_vmap_points_w_;
        uvs = left_uvs_;
        return;
    }
    if(camera == right_camera_ && right_correspondence_fid_ == cur_frame_id_){
        vmap_points_w = right_vmap_points_w_;
        uvs = right_uvs_;
        return;
    }
    
    // if(left_img_gray_.empty()) return;
    // std::vector<cv::Point2f> cur_pts;
    // cv::goodFeaturesToTrack(left_img_gray_, cur_pts, 500, 0.01, 20);
    // left_lk2orb_.computeORBFromTrackedPoints(left_img_gray_, cur_pts, keypoints, descriptors, false);

    std::vector<cv::KeyPoint> keypoints={};
    std::vector<BRIEF::bitset> descriptors={};
    const std::vector<TrackedPoint>& features = feature_tracker->getTrackedPoints();
    for(const TrackedPoint& f: features){
        if(f.descriptor.empty()) continue;//边缘点可能计算失效
        keypoints.push_back(f.keypoint);
        descriptors.push_back(f.descriptor);
    }

    auto matching_submap = get_matching_submap();

    Mat4d T_cw = T_ci * T_wi.inverse(); 
    Mat4d T_wc = T_cw.inverse();
    Mat3d R_ci = T_ci.block<3, 3>(0, 0);
    Vec3d p_ci = T_ci.block<3, 1>(0, 3);
    Mat3d R_wi = T_wi.block<3, 3>(0, 0);
    Vec3d p_wi = T_wi.block<3, 1>(0, 3);
    
    Mat4d T_ws = matching_submap->pose_odom_;
    // 遍历submap，筛选中心点在视角内的cell (TODO(wz):可更精确)。
    for(auto i_cell : matching_submap->cells_glb){
        const auto& region = i_cell.second.region;
        double cx = (region.x_min + region.x_max) / 2.;
        double cy = (region.y_min + region.y_max) / 2.;
        double cz = (region.z_min + region.z_max) / 2.;
        Vec4d pc_s(cx, cy, cz, 1.0);
        Vec4d pc_c = T_cw * T_ws * pc_s;
        Vec2d uv;
        if(InCameraView(camera, pc_c.segment<3>(0), uv) ){
            if(!i_cell.second.visual_points_.empty()){
                for(const auto& k_vmp: i_cell.second.visual_points_){
                    if(k_vmp.second.descriptor.empty()) continue;
                    Vec3d vp_w = T_ws.block<3,3>(0,0) * k_vmp.second.xyz + T_ws.block<3,1>(0,3);
                    int best_distance = 128;
                    int best_index = -1;
                    for(int k = 0; k < keypoints.size(); k++){
                        // LOG(INFO)<<k_vmp.second.descriptor.size()<<", "<<descriptors[k].size();
                        int d = HammingDis(k_vmp.second.descriptor, descriptors[k]);
                        if(d < best_distance){
                            best_distance = d;
                            best_index = k;
                        }
                    }
                    if(best_index != -1 && best_distance < 80){
                        uvs.push_back(keypoints[best_index].pt);
                        vmap_points_w.push_back(vp_w);
                    }
                }
            }
        }
    }

    if(camera == left_camera_){
        left_correspondence_fid_ = cur_frame_id_;
        left_vmap_points_w_ = vmap_points_w;
        left_uvs_ = uvs;
    }
    if(camera == right_camera_){
        right_correspondence_fid_ = cur_frame_id_;
        right_vmap_points_w_ = vmap_points_w;
        right_uvs_ = uvs;
    }
}


void LioMesher::Orb3dPointsInCurrentView(const camodocal::CameraPtr camera,
                                const Mat4d& T_wi, const Mat4d& T_ci,
                                std::vector<Vec3d>& vmap_points_cur,
                                std::vector<BRIEF::bitset>& descriptors){
    auto matching_submap = get_matching_submap();

    Mat4d T_iw = T_wi.inverse(); 
    Mat4d T_cw = T_ci * T_wi.inverse(); 
    Mat4d T_wc = T_cw.inverse();
    Mat3d R_ci = T_ci.block<3, 3>(0, 0);
    Vec3d p_ci = T_ci.block<3, 1>(0, 3);
    Mat3d R_wi = T_wi.block<3, 3>(0, 0);
    Vec3d p_wi = T_wi.block<3, 1>(0, 3);
    
    Mat4d T_ws = matching_submap->pose_odom_;
    // 遍历submap，筛选中心点在视角内的cell (TODO(wz):可更精确)。
    std::vector<Cell> cells;
    for(auto i_cell : matching_submap->cells_glb){
        const auto& region = i_cell.second.region;
        double cx = (region.x_min + region.x_max) / 2.;
        double cy = (region.y_min + region.y_max) / 2.;
        double cz = (region.z_min + region.z_max) / 2.;
        Vec4d pc_s(cx, cy, cz, 1.0);
        Vec4d pc_c = T_cw * T_ws * pc_s;

        Vec2d uv;
        if(InCameraView(camera, pc_c.segment<3>(0), uv) ){
            if(!i_cell.second.visual_points_.empty()){
                cells.push_back(i_cell.second);
                for(const auto& k_vmp: i_cell.second.visual_points_){
                    if(k_vmp.second.descriptor.empty()) continue;
                    Vec3d vp_w = T_ws.block<3,3>(0,0) * k_vmp.second.xyz+T_ws.block<3,1>(0,3);
                    Vec3d vp_i = T_iw.block<3,3>(0,0) * vp_w + T_iw.block<3,1>(0,3);
                    vmap_points_cur.push_back(vp_i);
                    descriptors.push_back(k_vmp.second.descriptor);
                }
            }
        }
    }
}

void LioMesher::TrackedOrb3dPoints(const std::shared_ptr<FeatureTracker>& tracker,
                        SlidingWindowDepthFilter& depth_filter,
                        const Mat4d& T_wb, const Mat4d& T_cb,
                        std::vector<Vec3d>& vmap_points_cur,
                        std::vector<cv::KeyPoint>& keypoints,
                        std::vector<BRIEF::bitset>& descriptors){
    Mat4d T_cw = T_cb * (T_wb.inverse()); 
    for(const auto& feature: tracker->getTrackedPoints()){
        const auto& ft = depth_filter.getFeatureTrackById(feature.id);
        if(ft && !feature.descriptor.empty()){
            Vec3d x3d;
            if(depth_filter.getFeatureWorldPoint(feature.id, x3d)){
                Vec3d x3d_c = T_cw.block<3,3>(0,0) * x3d + T_cw.block<3,1>(0,3);
                vmap_points_cur.push_back(x3d_c);
                descriptors.push_back(feature.descriptor);
                keypoints.push_back(feature.keypoint);    
            }
        }
    }
}

// 重新提取特征点和描述子并关联深度，用于后端回环和匹配，需要耗费额外的计算量
// 旨在减小模态差异提高召回，但实际测试与直接用滑窗内的特征点和深度区别不大
void LioMesher::VisualPoints3D(const cv::Mat& image, const Mat4d& T_wc,
                        std::vector<cv::KeyPoint>& keypoints,
                        std::vector<BRIEF::bitset>& descriptors,
                        std::vector<Vec3d>& map_points_c){
    LKToORBConverter brief_extractor;    
    std::vector<cv::Point2f> cur_pts;
    cv::Mat img = image.clone();
    if(img.channels()==3){
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }

    cv::goodFeaturesToTrack(img, cur_pts, 500, 0.01, 10);
    brief_extractor.computeORBFromTrackedPoints(img, cur_pts, keypoints, descriptors, false);
    
    std::vector<Vec3d> xyz_cam;
    std::vector<double> variance;
    AssignXYZbyRayTracing(T_wc, left_camera_, get_matching_submap(), keypoints, map_points_c, variance);
}

void LioMesher::AssignXYZbyRayTracing(
    const Mat4d& T_wc,
    const camodocal::CameraPtr& camera, 
    const std::shared_ptr<Submap>& submap,
    const std::vector<cv::KeyPoint>& keypoints,
    std::vector<Vec3d>& xyz_cam,
    std::vector<double>& variance){
    const Mat4d& T_ws = submap->pose_odom_;
    Mat4d T_sw = T_ws.inverse();
    
    Mat4d T_cw = T_wc.inverse(); 
    Vec3d P_wc = T_wc.block<3,1>(0,3);
    Mat3d R_wc = T_wc.block<3,3>(0,0);
    Mat4d T_sc = T_sw * T_wc;
    Mat4d T_cs = T_sc.inverse();
    Vec3d P_sc = T_sc.block<3, 1>(0,3);
    Mat3d R_sc = T_sc.block<3, 3>(0,0);

    double l = Parameter::GetInstance().grid;
    int n_row =  Parameter::GetInstance().num_test;

    variance.resize(keypoints.size(), -1.0);
    xyz_cam.resize(keypoints.size(), Vec3d::Zero());

    std::vector<size_t> idx(keypoints.size());
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        const cv::KeyPoint& uv = keypoints[i];
        Vec3d dir_cam;
        camera->liftProjective(Vec2d(uv.pt.x, uv.pt.y), dir_cam);      
        dir_cam.normalize();

        // 1. 找到与当前ray相交的cell
        int max_step = (Parameter::GetInstance().range_max) / l;
        std::vector<const Cell*> cells_hit = submap->getIntersectedCells(
                Ray(P_sc, R_sc * dir_cam), max_step);
        
        // 2. 如果有相交的cell，寻找相交的面片，计算交点
        for(const Cell* cell: cells_hit){                
            double min_variance = 1e9;
            // cell在子地图坐标系下
            std::optional<Vec4d> resx = cell->rayCellIntersectionX(Ray(P_sc, R_sc*dir_cam));
            std::optional<Vec4d> resy = cell->rayCellIntersectionY(Ray(P_sc, R_sc*dir_cam));
            std::optional<Vec4d> resz = cell->rayCellIntersectionZ(Ray(P_sc, R_sc*dir_cam));
            if(resx && (*resx)(3) < Parameter::GetInstance().variance_depth_association){
                min_variance = (*resx)(3);

                xyz_cam[i] = T_cs.block<3,3>(0,0) * (*resx).head(3) + T_cs.block<3, 1>(0,3);
                variance[i] = min_variance;        
            }

            if(resy && (*resy)(3) < Parameter::GetInstance().variance_depth_association){
                if((*resy)(3) < min_variance){
                    min_variance = (*resy)(3);
                    xyz_cam[i] = T_cs.block<3,3>(0,0) * (*resy).head(3) + T_cs.block<3, 1>(0,3);
                    variance[i] = min_variance;
                }
            }
            
            if(resz && (*resz)(3) < Parameter::GetInstance().variance_depth_association){
                if((*resz)(3) < min_variance){
                    min_variance = (*resz)(3);
                    xyz_cam[i] = T_cs.block<3,3>(0,0) * (*resz).head(3) + T_cs.block<3, 1>(0,3);
                    variance[i] = min_variance;
                }
            }
        }
    });
}