#ifndef __GLIOMESH_POSE_GRAPH_H__
#define __GLIOMESH_POSE_GRAPH_H__
#include <iostream>

#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <math.h>
#include <thread>
#include <execution>

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "scancontext/Scancontext.h"

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include "submap.h"
#include "liomesher.h"

// #include "DBoW2/DBoW2.h"
#include "DBoW/DBoW2.h"
#include "DBoW/TemplatedDatabase.h"
#include "DBoW/TemplatedVocabulary.h"

#include "orb3d_registration.h"

#include "SBoW3D.h"

#include "extended_scan_context.h"
#include "scan_context_database.h"
// #include "loop_db.h"

#include <GeographicLib/include/Geocentric.hpp>
#include <GeographicLib/include/LocalCartesian.hpp>
// #include <GeographicLib/include/Geoid.hpp>


using namespace gtsam;
typedef pcl::PointXYZ PointType;

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

struct TimestampedPose{
    int64_t ts_nsec;
    Mat4d pose;
};

struct TimestampedPointCloud{
    int64_t ts_nsec;
    pcl::PointCloud<PointType> point_cloud;
};

struct TimestampedImage{
    int64_t ts_nsec;
    cv::Mat image;
};

struct FrameVisualMapPoints{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<Vec3d> pts_3d_cam;//前端估计得到的3D地图点，在相机坐标系下
    std::vector<BRIEF::bitset> descriptors;//与map_points一一对应

    std::vector<cv::KeyPoint> keypoints_full;//新提取的用于回环及相对位姿解算的关键点
    std::vector<BRIEF::bitset> descriptors_full;//为了召回更多的匹配对，建立足够的2D-3D pair
};



class PoseGraph{
public:
    PoseGraph();
    ~PoseGraph();
    
    // to modify
    void AddTrackingResult(const std::shared_ptr<TrackingResult>& result);
    std::vector<Pose6D> getUpdatedKeyframePoses();

    std::vector<Pose6D> getGPSPath();

    void LoadVocabulary(std::string voc_path){
        voc_.reset(new BriefVocabulary(voc_path));
        db_.reset(new BriefDatabase(*voc_, false, 0));
        orb_ = cv::ORB::create(
            1000,        // 最大特征点数
            1.2f,        // 金字塔层级缩放因子
            8,           // 金字塔层数
            31,          // 边缘阈值
            0,           // 第一层金字塔层级
            2,           // WTA_K
            cv::ORB::HARRIS_SCORE,  // 评分类型
            31,          // 补丁大小
            20           // 快速阈值
        );
    }
    
    // for offline mode usage
    bool finish(); 

    Mat4d estimatePoseRansac(const std::vector<Vec3d>& object_points,
                        const std::vector<Vec3d>& image_points_norm,
                        CameraPtr camera,
                        std::vector<bool>& inliers,
                        double reprojection_error = 3.0,
                        double confidence = 0.99,
                        int max_iterations = 100) {
        if (object_points.size() < 6) {
            LOG(WARNING) << "Not enough points for RANSAC: " << object_points.size();
            inliers = std::vector<bool>(object_points.size(), false);
            return Mat4d::Identity();
        }
        
        // 转换为OpenCV格式
        std::vector<cv::Point3f> cv_object_points;
        std::vector<cv::Point2f> cv_image_points;
        
        for (size_t i = 0; i < object_points.size(); i++) {
            cv_object_points.push_back(cv::Point3f(
                object_points[i].x(),
                object_points[i].y(),
                object_points[i].z()
            ));
            
            // Vec2d pixel;
            // camera->spaceToPlane(image_points_norm[i], pixel);
            cv_image_points.push_back(cv::Point2f(image_points_norm[i](0), image_points_norm[i](1)));
        }
        
        // 相机内参
        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        
        // 输出变量
        cv::Mat rvec, tvec;
        cv::Mat inliers_cv;
        
        // 使用RANSAC进行鲁棒估计
        bool success = cv::solvePnPRansac(cv_object_points, cv_image_points,
                                        camera_matrix, dist_coeffs,
                                        rvec, tvec, false,
                                        max_iterations, reprojection_error,
                                        confidence, inliers_cv, cv::SOLVEPNP_EPNP);
        
        if (!success) {
            LOG(WARNING) << "RANSAC pose estimation failed";
            inliers = std::vector<bool>(object_points.size(), false);
            return Mat4d::Identity();
        }
        
        // 设置内点标记
        inliers = std::vector<bool>(object_points.size(), false);
        for (int i = 0; i < inliers_cv.rows; i++) {
            int idx = inliers_cv.at<int>(i);
            if (idx >= 0 && idx < static_cast<int>(object_points.size())) {
                inliers[idx] = true;
            }
        }
        
        LOG(INFO) << "RANSAC inliers: " << cv::countNonZero(inliers_cv) 
                << "/" << object_points.size();
        
        // 转换为旋转矩阵
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        
        // 构建变换矩阵
        Mat4d T_w_c = Mat4d::Identity();
        T_w_c.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3d>(R.ptr<double>());
        T_w_c.block<3, 1>(0, 3) = Eigen::Map<Eigen::Vector3d>(tvec.ptr<double>());
        
        return T_w_c;
    }

    Mat4d estimatePoseEPnP(const std::vector<Vec3d>& object_points,
                       const std::vector<Vec3d>& image_points_norm,
                       CameraPtr camera) {
        if (object_points.size() < 6) {
            LOG(WARNING) << "Not enough points for EPnP: " << object_points.size();
            return Mat4d::Identity();
        }
        
        // 转换为OpenCV格式
        std::vector<cv::Point3f> cv_object_points;
        std::vector<cv::Point2f> cv_image_points;
        
        for (size_t i = 0; i < object_points.size(); i++) {
            // 3D点（世界坐标系）
            cv_object_points.push_back(cv::Point3f(
                object_points[i].x(),
                object_points[i].y(),
                object_points[i].z()
            ));
            
            cv_image_points.push_back(cv::Point2f(image_points_norm[i].x(), image_points_norm[i].y()));
        }
        
        // 相机内参矩阵
        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        
        // 畸变系数
        cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        
        // 旋转向量和平移向量
        cv::Mat rvec, tvec;
        
        // 使用EPnP算法
        bool success = cv::solvePnP(cv_object_points, cv_image_points,
                                    camera_matrix, dist_coeffs,
                                    rvec, tvec, false, cv::SOLVEPNP_EPNP);
        
        if (!success) {
            LOG(WARNING) << "EPnP pose estimation failed";
            return Mat4d::Identity();
        }
        
        // 转换为旋转矩阵
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        
        // 构建4x4变换矩阵
        Mat4d T_w_c = Mat4d::Identity();
        T_w_c.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3d>(R.ptr<double>());
        T_w_c.block<3, 1>(0, 3) = Eigen::Map<Eigen::Vector3d>(tvec.ptr<double>());
        
        return T_w_c;
    }

    // 重投影误差计算
    double computeReprojectionError(const std::vector<Vec3d>& object_points,
                                    const std::vector<Vec3d>& image_points_norm,
                                    const Mat4d& T_w_c,
                                    CameraPtr camera,
                                    const std::vector<bool>& inliers = {}) {
        double total_error = 0.0;
        int count = 0;
        
        for (size_t i = 0; i < object_points.size(); i++) {
            if (!inliers.empty() && !inliers[i]) {
                continue;
            }
            
            // 将3D点转换到相机坐标系
            Vec3d point_camera = T_w_c.block<3, 3>(0, 0) * object_points[i] + 
                                T_w_c.block<3, 1>(0, 3);
            
            // 投影到图像平面
            Vec2d projected_pixel;
            camera->spaceToPlane(point_camera, projected_pixel);
            
            // 将归一化坐标转换为像素坐标进行比较
            Vec2d observed_pixel;
            camera->spaceToPlane(image_points_norm[i], observed_pixel);
            
            // 计算误差
            double error = (projected_pixel - observed_pixel).norm();
            total_error += error;
            count++;
        }
        
        return count > 0 ? total_error / count : std::numeric_limits<double>::max();
    }

    //////////////////////////////////////////////////////////////////////////////
    // --- 数据结构 ---
    struct MatchPair {
        Vec3d p3d;   // 3D点坐标 (相机坐标系)
        Vec2d p2n;   // 归一化图像平面2D点 (x/z, y/z)
    };

    // ==============================================================================
    // 1. 数学工具函数
    // ==============================================================================

    // 计算向量的反对称矩阵
    inline Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S <<  0.0, -v(2),  v(1),
            v(2),   0.0, -v(0),
            -v(1),  v(0),   0.0;
        return S;
    }

    // Huber 鲁棒核函数权重计算
    inline double HuberWeight(double e2, double delta) {
        double e = std::sqrt(e2);
        if (e <= delta) return 1.0;
        else return delta / e;
    }

    // 将 OpenCV 的 rvec 和 tvec 转换为 Eigen 的 4x4 变换矩阵
    inline Mat4d ConvertToMat4d(const cv::Mat& rvec, const cv::Mat& tvec) {
        Eigen::Vector3d r_eigen, t_eigen;
        cv::cv2eigen(rvec, r_eigen);
        cv::cv2eigen(tvec, t_eigen);
        
        Mat4d T = Mat4d::Identity();
        T.block<3, 3>(0, 0) = Eigen::AngleAxisd(r_eigen.norm(), r_eigen.normalized()).toRotationMatrix();
        T.block<3, 1>(0, 3) = t_eigen;
        return T;
    }

    // ==============================================================================
    // 2. 核心约束与雅可比计算
    // ==============================================================================

    // 计算正向投影 (集合 A: 历史3D -> 当前2D) 的误差与雅可比
    bool computeForwardConstraint(
        const Vec3d& P_H, const Vec2d& p_C_norm,
        const Eigen::Matrix3d& R_CH, const Eigen::Vector3d& t_CH,
        Eigen::Vector2d& error, Eigen::Matrix<double, 2, 6>& Jacobian) 
    {
        Vec3d P_C_prime = R_CH * P_H + t_CH;
        double X = P_C_prime(0), Y = P_C_prime(1), Z = P_C_prime(2);

        if (Z <= 1e-5) return false; 

        double Z_inv = 1.0 / Z;
        double Z_inv2 = Z_inv * Z_inv;

        error = p_C_norm - Vec2d(X * Z_inv, Y * Z_inv);

        Eigen::Matrix<double, 2, 3> J_proj;
        J_proj << -Z_inv,    0.0, X * Z_inv2,
                    0.0, -Z_inv, Y * Z_inv2;

        Eigen::Matrix<double, 3, 6> J_pose;
        J_pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_pose.block<3, 3>(0, 3) = -skewSymmetric(P_C_prime);

        Jacobian = J_proj * J_pose;
        return true;
    }

    // 计算反向投影 (集合 B: 当前3D -> 历史2D) 的误差与雅可比
    bool computeBackwardConstraint(
        const Vec3d& P_C, const Vec2d& p_H_norm,
        const Eigen::Matrix3d& R_CH, const Eigen::Vector3d& t_CH,
        Eigen::Vector2d& error, Eigen::Matrix<double, 2, 6>& Jacobian) 
    {
        Eigen::Matrix3d R_HC = R_CH.transpose();
        Vec3d P_H_prime = R_HC * (P_C - t_CH);
        
        double X = P_H_prime(0), Y = P_H_prime(1), Z = P_H_prime(2);

        if (Z <= 1e-5) return false;

        double Z_inv = 1.0 / Z;
        double Z_inv2 = Z_inv * Z_inv;

        error = p_H_norm - Vec2d(X * Z_inv, Y * Z_inv);

        Eigen::Matrix<double, 2, 3> J_proj;
        J_proj << -Z_inv,    0.0, X * Z_inv2,
                    0.0, -Z_inv, Y * Z_inv2;

        Eigen::Matrix<double, 3, 6> J_pose;
        J_pose.block<3, 3>(0, 0) = -R_HC;
        J_pose.block<3, 3>(0, 3) = R_HC * skewSymmetric(P_C);

        Jacobian = J_proj * J_pose;
        return true;
    }

    // ==============================================================================
    // 3. 联合非线性优化 (Refine)
    // ==============================================================================

    Mat4d RefineWithTwoSets(
        const Mat4d& T_CH_init,
        const std::vector<MatchPair>& forward_matches,
        const std::vector<MatchPair>& backward_matches,
        double huber_delta) 
    {
        Eigen::Matrix3d R_CH = T_CH_init.block<3, 3>(0, 0);
        Eigen::Vector3d t_CH = T_CH_init.block<3, 1>(0, 3);
        
        int max_iterations = 10;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();
            int valid_edges = 0;

            // 集合 A (正向)
            for (const auto& match : forward_matches) {
                Eigen::Vector2d e; Eigen::Matrix<double, 2, 6> J;
                if (computeForwardConstraint(match.p3d, match.p2n, R_CH, t_CH, e, J)) {
                    double weight = HuberWeight(e.squaredNorm(), huber_delta);
                    H += weight * J.transpose() * J;
                    b -= weight * J.transpose() * e;
                    valid_edges++;
                }
            }

            // 集合 B (反向)
            for (const auto& match : backward_matches) {
                Eigen::Vector2d e; Eigen::Matrix<double, 2, 6> J;
                if (computeBackwardConstraint(match.p3d, match.p2n, R_CH, t_CH, e, J)) {
                    double weight = HuberWeight(e.squaredNorm(), huber_delta);
                    H += weight * J.transpose() * J;
                    b -= weight * J.transpose() * e;
                    valid_edges++;
                }
            }

            if (valid_edges < 6) break; // 防止约束不足导致发散

            Eigen::Matrix<double, 6, 1> delta_xi = H.ldlt().solve(b);
            if (std::isnan(delta_xi[0]) || delta_xi.norm() < 1e-6) break;

            // 更新位姿 (左扰动)
            Eigen::Vector3d d_rho = delta_xi.head<3>();
            Eigen::Vector3d d_phi = delta_xi.tail<3>();
            Eigen::Matrix3d d_R = Eigen::AngleAxisd(d_phi.norm(), d_phi.normalized()).toRotationMatrix();
            
            t_CH = d_R * t_CH + d_rho;
            R_CH = d_R * R_CH;
        }

        Mat4d T_CH_opt = Mat4d::Identity();
        T_CH_opt.block<3, 3>(0, 0) = R_CH;
        T_CH_opt.block<3, 1>(0, 3) = t_CH;
        return T_CH_opt;
    }

    // ==============================================================================
    // 4. 顶层接口：双向 PnP RANSAC
    // ==============================================================================

    /**
     * @brief 求解当前帧在历史帧下的相对位姿 (T_HC)
     * @param forward_matches  集合A: {历史3D, 当前归一化2D}
     * @param backward_matches 集合B: {当前3D, 历史归一化2D}
     * @param T_HC_opt         输出：优化后的相对位姿 (从当前帧变换到历史帧)
     * @param focal_length     用于计算归一化平面阈值的近似焦距
     * @param pixel_threshold  RANSAC 的像素级重投影误差阈值
     * @return 是否求解成功
     */
    bool EstimateRelativePoseBidirectional(
        const std::vector<MatchPair>& forward_matches, 
        const std::vector<MatchPair>& backward_matches,
        Mat4d& T_HC_opt,
        double focal_length = 460.0, 
        double pixel_threshold = 3.0) 
    {
        // 如果正向特征点太少，P3P无法进行
        if (forward_matches.size() < 4) return false;

        double thresh_sq = std::pow(0.01, 2); //pixel_threshold / focal_length
        int best_total_inliers = 0;
        Mat4d best_T_CH = Mat4d::Identity();
        
        std::vector<int> indices(forward_matches.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());

        int max_iters = 150;
        for (int i = 0; i < max_iters; ++i) {
            // 1. 从正向集合中随机抽取 4 个点
            std::shuffle(indices.begin(), indices.end(), g);
            std::vector<cv::Point3f> pts3d(4);
            std::vector<cv::Point2f> pts2d(4);
            for (int k = 0; k < 4; ++k) {
                const auto& m = forward_matches[indices[k]];
                pts3d[k] = cv::Point3f(m.p3d.x(), m.p3d.y(), m.p3d.z());
                pts2d[k] = cv::Point2f(m.p2n.x(), m.p2n.y());
            }

            // 2. OpenCV P3P 求解基础位姿候选 T_CH
            cv::Mat rvec, tvec;
            if (!cv::solvePnP(pts3d, pts2d, cv::Mat::eye(3,3,CV_32F), cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_P3P))
                continue;

            Mat4d T_curr = ConvertToMat4d(rvec, tvec);
            Mat4d T_inv = T_curr.inverse(); // T_HC

            // 3. 双向验证 Inliers
            int inliers_A = 0;
            for (const auto& m : forward_matches) {
                Vec3d p_c = T_curr.block<3,3>(0,0) * m.p3d + T_curr.block<3,1>(0,3);
                if (p_c.z() > 0 && (m.p2n - Vec2d(p_c.x()/p_c.z(), p_c.y()/p_c.z())).squaredNorm() < thresh_sq)
                    inliers_A++;
            }

            int inliers_B = 0;
            for (const auto& m : backward_matches) {
                Vec3d p_h = T_inv.block<3,3>(0,0) * m.p3d + T_inv.block<3,1>(0,3);
                if (p_h.z() > 0 && (m.p2n - Vec2d(p_h.x()/p_h.z(), p_h.y()/p_h.z())).squaredNorm() < thresh_sq)
                    inliers_B++;
            }

            // 4. 记录最佳模型
            if (inliers_A + inliers_B > best_total_inliers) {
                best_total_inliers = inliers_A + inliers_B;
                best_T_CH = T_curr;
            }
        }

        // 判断是否找到足够多的内点 (根据实际工程调整此阈值)
        if (best_total_inliers < 6) return false;

        // 5. 提取真正的 Inliers 喂给 Refine (去除大噪声)
        std::vector<MatchPair> inlier_forwards, inlier_backwards;
        Mat4d best_T_inv = best_T_CH.inverse();
        
        for (const auto& m : forward_matches) {
            Vec3d p_c = best_T_CH.block<3,3>(0,0) * m.p3d + best_T_CH.block<3,1>(0,3);
            if (p_c.z() > 0 && (m.p2n - Vec2d(p_c.x()/p_c.z(), p_c.y()/p_c.z())).squaredNorm() < thresh_sq * 1.5) // 放宽一点作为优化的初筛
                inlier_forwards.push_back(m);
        }
        for (const auto& m : backward_matches) {
            Vec3d p_h = best_T_inv.block<3,3>(0,0) * m.p3d + best_T_inv.block<3,1>(0,3);
            if (p_h.z() > 0 && (m.p2n - Vec2d(p_h.x()/p_h.z(), p_h.y()/p_h.z())).squaredNorm() < thresh_sq * 1.5)
                inlier_backwards.push_back(m);
        }

        // 6. 最终的非线性优化
        // Huber 阈值通常设为重投影误差阈值对应的平方误差
        Mat4d T_CH_opt = RefineWithTwoSets(best_T_CH, inlier_forwards, inlier_backwards, thresh_sq);

        // 返回当前帧到历史帧的位姿 T_HC
        T_HC_opt = T_CH_opt.inverse();
        return true;
    }

    void AssignXYZbyRayTracing(
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
private:
    void performSCLoopClosure();
    std::optional<gtsam::Pose3> doICPVirtualRelative(int _loop_kf_idx, int _curr_kf_idx);
    std::optional<gtsam::Pose3> doORB3dVirtualRelative(int _loop_kf_idx, int _curr_kf_idx);
    std::optional<gtsam::Pose3> doICPVirtualRelativeSubmap(int _loop_kf_idx, int _curr_kf_idx);
    void loopFindNearKeyframesCloud(pcl::PointCloud<PointType>::Ptr& nearKeyframes, 
                                    const int& key, const int& submap_size, const int& root_idx);
    pcl::PointCloud<PointType>::Ptr convertMeshToCloud(std::shared_ptr<Submap> submap);
    
    void initNoises();
    void runISAM2opt();
    void updatePoses();
    Pose6D getOdom(TimestampedPose _odom);
    Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2);
    pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf);

    inline gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p){
        return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
    }
    inline gtsam::Pose3 Mat4dtoGTSAMPose3(const Mat4d& pose){
        auto tx = pose.col(3)(0);
        auto ty = pose.col(3)(1);
        auto tz = pose.col(3)(2);
        auto ypr = pose.block<3,3>(0,0).eulerAngles(2, 1, 0);
        Pose6D p = Pose6D{tx, ty, tz, ypr(2), ypr(1), ypr(0)}; 
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z));
    }

    inline Pose6D Mat4dtoPose6D(const Mat4d& pose){
        auto tx = pose.col(3)(0);
        auto ty = pose.col(3)(1);
        auto tz = pose.col(3)(2);
        auto ypr = pose.block<3,3>(0,0).eulerAngles(2, 1, 0);
        return Pose6D{tx, ty, tz, ypr(2), ypr(1), ypr(0)}; 
    }
    void process_pg();
    void process_lcd();
    void process_sc_icp();
    void process_esc_icp();
    void process_icp_submap();
    void process_orb3d_matcher();
    void process_isam();

    void save_node_poses_viral_format();
    void save_submap_poses_viral_format();

    void save_node_poses_kitti_format();
    void save_submap_poses_kitti_format();

    void save_gnss_positions();
    
    void save_submap_meshes();


private:
    SCManager scManagerSubmap;                  // 专门用于 Submap 的 ScanContext 管理器
    std::thread submap_loop_thread_;            // Submap 回环处理线程
    std::queue<size_t> finishedSubmapIdxBuf;    // 已完成待处理的 Submap 索引队列
    std::mutex mBufSubmap;                      // 缓冲区锁
    std::vector<pcl::PointCloud<PointType>::Ptr> keyframeCloudsSubmap;

    // ExtendedScanContext esc_submap_;
    // LoopClosureDatabase loop_db_;

    ScanContextDatabase esc_db_;

    SCManager scManager;    
    bool gtSAMgraphMade = false;
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odomNoise;
    noiseModel::Base::shared_ptr robustLoopNoise;
    noiseModel::Base::shared_ptr robustGPSNoise;

    std::queue<TimestampedPose> odometryBuf;
    std::queue<TimestampedPointCloud> fullResBuf;
    std::queue<TimestampedImage> imageBuf;
    std::queue<std::shared_ptr<TrackingResult>> tracked_data_;
    std::queue<std::pair<int, int> > scLoopICPBuf;
    std::queue<std::pair<int, int> > escLoopICPBuf;
    std::queue<std::pair<int, int> > scLoopOrb3DBuf;
    std::queue<std::pair<int, int> > scLoopSubmapBuf;
    
    std::map<size_t, std::shared_ptr<Submap>> submaps_;
    std::vector<size_t> added_submap_ids_;

    std::mutex mBuf;
    std::mutex mKF;
    std::mutex mtxICP;
    std::mutex mtxPosegraph;
    std::mutex mtxRecentPose;
    
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterScancontext;
    pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
    pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO;
    
    std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds; 
    std::vector<Pose6D> keyframePoses;
    std::vector<Pose6D> keyframePosesUpdated;
    std::vector<double> keyframeTimes;
    std::vector<FrameVisualMapPoints>  keyframeVMapPoints;

    std::vector<Pose6D> allFramePoses;
    std::vector<Pose6D> allFramePosesUpdated;
    std::vector<int64_t> allFrameTimes;
    std::map<int, int> kfid_to_fid_;
    std::map<int, int> fid_to_kfid_;
    std::map<int, DBoW2::EntryId> kfid_to_dbid_;
    std::map<DBoW2::EntryId, int> dbid_to_kfid_;

    std::map<int, int> kfid_to_submapfid_;
    std::map<int, int> submapfid_to_kfid_;

    int recentIdxUpdated = 0;
    
    double recentOptimizedX = 0.0;
    double recentOptimizedY = 0.0; 
    
    int64_t timeLaserOdometry = 0;
    int64_t timeLaser = 0;

    size_t current_frame_id_;

    double keyframeMeterGap;
    double keyframeDegGap, keyframeRadGap;
    double translationAccumulated = 1000000.0; // large value means must add the first given frame.
    double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.
    bool isNowKeyFrame = false; 

    Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
    Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 
    
    pcl::PointCloud<PointType>::Ptr laserCloudMapPGO;
    pcl::VoxelGrid<PointType> downSizeFilterMapPGO;

    std::thread posegraph_slam_thread_;
    std::thread lc_detection_thread_;
    std::thread sc_icp_thread_;
    std::thread esc_icp_thread_;
    std::thread orb3d_matcher_thread_;
    std::thread isam_update_thread_;

    // for offline node
    bool job_finished_ = false;
    // std::thread viz_map_thread_;
    // std::thread viz_path_thread_;

    // ORBDatabase db;
	std::unique_ptr<BriefVocabulary> voc_ = nullptr;
	std::unique_ptr<BriefDatabase> db_ = nullptr;

    // std::unique_ptr<DBoW2::Vocabulary> voc_ = nullptr;
	// std::unique_ptr<DBoW2::Database> db_ = nullptr;
    cv::Ptr<cv::ORB> orb_;

    GeographicLib::LocalCartesian geo_converter_;
    std::vector<std::pair<int64_t, Vec3d>> gnss_positions_;

    bool gnss_initial_aligned_ = false;
    std::vector<sad::NaiveGNSS> gnss_for_initial_alignment_;
    std::vector<int> node_ids_for_initial_alignment_;
    Vec3d last_gnss_position_;
    bool gnss_inited_ = false;
    bool add_new_loop_ = false;

    // for debug only
    camodocal::CameraPtr left_camera_;
    BoW3D::SubmapBoW3D bow3d_;
};

#endif