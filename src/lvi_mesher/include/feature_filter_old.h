#ifndef __FEATURE_FILTER_H__
#define __FEATURE_FILTER_H__
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <algorithm>
#include <numeric>
#include <ceres/ceres.h>
#include "submap.h"
#include "LKtoORBConverter.h"

// 重投影误差计算类
struct ReprojectionError {
    ReprojectionError(const cv::Point2f& observed_uv)
        : observed_uv_(observed_uv) {}

    template <typename T>
    bool operator()(const T* const camera_rotation,
                    const T* const camera_translation,
                    const T* const point_3d,
                    T* residuals) const {
        // 将归一化坐标转换为3D点（使用全局XYZ坐标）
        Eigen::Matrix<T, 3, 1> p(point_3d[0], point_3d[1], point_3d[2]);

        // 应用相机位姿变换
        Eigen::Quaternion<T> q(camera_rotation[0], camera_rotation[1], 
                                camera_rotation[2], camera_rotation[3]);
        Eigen::Matrix<T, 3, 1> t(camera_translation[0], 
                                camera_translation[1], 
                                camera_translation[2]);

        Eigen::Matrix<T, 3, 1> point_camera = q * p + t;

        // 投影到归一化平面
        T predicted_x = point_camera[0] / point_camera[2];
        T predicted_y = point_camera[1] / point_camera[2];

        // 计算残差
        // residuals[0] = T(sqrt_info_) * (predicted_x - T(observed_uv_.x));
        // residuals[1] = T(sqrt_info_) * (predicted_y - T(observed_uv_.y));
        residuals[0] = predicted_x - T(observed_uv_.x);
        residuals[1] = predicted_y - T(observed_uv_.y);
        
        // if(T(residuals[0]) * T(residuals[0]) + T(residuals[1]) * T(residuals[1]) > T(0.09))
        //     return false; // 大于0.09的误差，认为是异常值
        return true;
    }

    static ceres::CostFunction* Create(const cv::Point2f& observed_uv) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
            new ReprojectionError(observed_uv)));
    }

private:
    double sqrt_info_ = 1293. / 1.5;
    cv::Point2f observed_uv_;
};

class PointToPlaneError {
public:
    PointToPlaneError(const Eigen::Vector4d& plane)
        : plane_params_(plane) {}

    // 残差维度为 1：距离值
    static ceres::CostFunction* Create(const Eigen::Vector4d& plane) {
        return (new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 3>(
            new PointToPlaneError(plane)));
    }

    template <typename T>
    bool operator()(const T* const point,  // 3D点 [x, y, z]
                    T* residual) const {
        const T& x = point[0];
        const T& y = point[1];
        const T& z = point[2];

        // 平面参数 [a, b, c, d]
        const T& a = T(plane_params_[0]);
        const T& b = T(plane_params_[1]);
        const T& c = T(plane_params_[2]);
        const T& d = T(plane_params_[3]);

        // 有符号距离公式：(a*x + b*y + c*z + d) / sqrt(a^2 + b^2 + c^2)
        T numerator = a * x + b * y + c * z + d;
        T denom = ceres::sqrt(a * a + b * b + c * c);

        // 避免除零，但一般平面法向量应该已经归一化
        T distance = numerator / denom;

        // 残差即为该距离，我们希望它接近 0
        residual[0] = distance;

        return true;
    }

private:
    Eigen::Vector4d plane_params_;  // 平面参数 [a, b, c, d]
};


class TemporalFeatureFilter {
public:
    struct CameraFrame {
        int frame_id;
        Eigen::Quaterniond q; // world to camera
        Eigen::Vector3d t; // world to camera
    };

    struct FeatureObservation {
        uint64_t frame_id;
        uint64_t feature_id;
        cv::Point2f pixel_pos; // uv in image
        
        // computed by LKtoORBConverter, once mature, they will be computed
        cv::KeyPoint keypoint;
        // cv::Mat descriptor; 
        BRIEF::bitset descriptor; 

        Eigen::Vector3d x3d_norm; // normalized coordinates in camera frame
        Eigen::Vector3d map_point; // 3D point in world coordinates
        Eigen::Vector3d updated_map_point; // after depth update
        Eigen::Vector3d optimized_map_point; // after optimization
        bool is_outlier = false;
    };

    struct FeatureTrack {
        uint64_t id;  // 特征点唯一ID
        std::deque<FeatureObservation> observations;  // 时间窗口内的观测
        float depth_consistency = 0.f;
        float motion_consistency = 0.f;
        bool is_mature = false;
        int continuous_inliers = 0;
        
        // 获取最新观测
        const FeatureObservation& latest() const { return observations.back(); }

        FeatureObservation& variable_latest() { return observations.back(); }
        
        // 获取时间窗口长度
        size_t windowSize() const { return observations.size(); }
    };

    

    ////////////////////////////////////////////////////////////////////////////
    TemporalFeatureFilter(size_t window_size = 5,
                         float depth_thresh = 0.2,
                         float motion_thresh = 3.0,
                         int min_continuous_inliers = 3)
        : window_size_(window_size),
          depth_thresh_(depth_thresh),
          motion_thresh_(motion_thresh),
          min_continuous_inliers_(min_continuous_inliers) {}
    
    // 处理新一帧的特征点观测
    void processFrame(uint64_t frame_id, 
                     const std::vector<std::pair<uint64_t, FeatureObservation>>& frame_features,
                     const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        current_frame_id_ = frame_id;
        CameraFrame new_frame;
        new_frame.frame_id = frame_id;
        new_frame.q = Eigen::Quaterniond(R).normalized();
        new_frame.t = t;
        frames_.push_back(new_frame);

        // 1. 更新或初始化特征轨迹
        updateFeatureTracks(frame_features);

        // 2. 滑动窗口管理
        manageSlidingWindow();
                
        // 3. 时间窗口内的一致性检查
        checkTemporalConsistency();
        
        // 4. 更新特征点成熟度
        updateMaturity();

        /* for (const auto& [id, track] : feature_tracks_) {
            if (track.is_mature && !track.latest().is_outlier) {                
                // 检查角测量与深度滤波的差异
                Vec3d tri_xyz;
                if(triangulateFeature(track, tri_xyz)){
                    LOG(INFO)<<(tri_xyz-track.latest().updated_map_point).norm();
                }
            }
        } */
    }
    
    // 获取成熟的特征点
    std::vector<FeatureObservation> getMatureFeatures() const {
        std::vector<FeatureObservation> mature_features;
        for (const auto& [id, track] : feature_tracks_) {
            if (track.is_mature && !track.latest().is_outlier) {
                mature_features.push_back(track.latest());
            }
        }
        return mature_features;
    }

    std::vector<FeatureObservation> getAllFeatures() const {
        std::vector<FeatureObservation> all_features;
        for (const auto& [id, track] : feature_tracks_) {
            all_features.push_back(track.latest());
        }
        return all_features;
    }
    
    // 获取所有活跃特征点轨迹
    const std::map<uint64_t, FeatureTrack>& getFeatureTracks() const {
        return feature_tracks_;
    }

    // 更新成熟特征点的深度值
    void updateMatureDepths() {
        for (auto& [id, track] : feature_tracks_) {
            if (track.is_mature && !track.latest().is_outlier) {
                updateFeatureDepth(track);
            }
        }
    }

    // 获取深度更新后的成熟特征点
    std::vector<FeatureObservation> getUpdatedFeatures() const {
        std::vector<FeatureObservation> updated_features;
        for (const auto& [id, track] : feature_tracks_) {
            if (track.is_mature && !track.latest().is_outlier) {
                updated_features.push_back(track.latest());
            }
        }
        return updated_features;
    }
    
    void setMotionThreshold(float thresh) {
        motion_thresh_ = thresh;
    }

    FeatureTrack* getFeatureTrackById(uint64_t id) {
        auto it = feature_tracks_.find(id);
        if (it != feature_tracks_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    void Optimize(std::shared_ptr<Submap> submap) {
        ceres::Problem problem;
        options_.linear_solver_type = ceres::DENSE_SCHUR;
        options_.minimizer_progress_to_stdout = false;
        options_.max_num_iterations = 10;

        std::vector<FeatureObservation> mature_features = getMatureFeatures();

        std::vector<Point2MeshMatch> matches;
        std::vector<int> track_ids;
        pcl::PointCloud<pcl::PointXYZ> map_points;
        for(const auto& [id, track] : feature_tracks_){
            if(!track.is_mature || track.latest().is_outlier) continue;
            track_ids.push_back(id);
            map_points.push_back(pcl::PointXYZ(track.latest().updated_map_point[0],
                                          track.latest().updated_map_point[1],
                                          track.latest().updated_map_point[2]));
        }
        submap->findMatchVisualPointToVertices(map_points, matches);

        // 准备相机位姿参数块
        std::unordered_map<int, double*> camera_rotations;
        std::unordered_map<int, double*> camera_translations;

        for (auto& frame : frames_) {
            double* rotation = new double[4];
            rotation[0] = frame.q.w();
            rotation[1] = frame.q.x();
            rotation[2] = frame.q.y();
            rotation[3] = frame.q.z();
            camera_rotations[frame.frame_id] = rotation;

            double* translation = new double[3];
            translation[0] = frame.t.x();
            translation[1] = frame.t.y();
            translation[2] = frame.t.z();
            camera_translations[frame.frame_id] = translation;

            problem.AddParameterBlock(rotation, 4);
            problem.AddParameterBlock(translation, 3);

            // 添加四元数局部参数化
            problem.SetParameterization(rotation, 
                new ceres::EigenQuaternionParameterization());
        }

        // 固定第一帧的位姿（作为参考系）
        if (!frames_.empty()) {
            int first_frame_id = frames_.front().frame_id;
            problem.SetParameterBlockConstant(camera_rotations[first_frame_id]);
            problem.SetParameterBlockConstant(camera_translations[first_frame_id]);
        }
        
        // 固定所有帧的位姿（如果需要）
        // for (auto& frame : frames_){
        //     if(frame.frame_id == current_frame_id_) continue;
        //     problem.SetParameterBlockConstant(camera_rotations[frame.frame_id]);
        //     problem.SetParameterBlockConstant(camera_translations[frame.frame_id]);
        // }

        
        // 准备特征点参数块
        std::unordered_map<int, double*> feature_positions;
        for (const FeatureObservation& mf : mature_features) {
            feature_positions[mf.feature_id] = new double[3];
            feature_positions[mf.feature_id][0] = mf.updated_map_point.x();
            feature_positions[mf.feature_id][1] = mf.updated_map_point.y();
            feature_positions[mf.feature_id][2] = mf.updated_map_point.z();
            problem.AddParameterBlock(feature_positions[mf.feature_id], 3);
        }


        // 添加残差块
        for (const auto& [feature_id, feature_track] : feature_tracks_) {
            if (!feature_track.is_mature) continue;
            for (const auto& obs : feature_track.observations) {
                if (camera_rotations.find(obs.frame_id) == camera_rotations.end()) {
                    continue;
                }

                ceres::CostFunction* cost_function = ReprojectionError::Create(
                        cv::Point2f(obs.x3d_norm[0], obs.x3d_norm[1]));
                
                problem.AddResidualBlock(cost_function,
                                        NULL, // 使用默认的Huber损失函数//new ceres::CauchyLoss(0.5)
                                        camera_rotations[obs.frame_id],
                                        camera_translations[obs.frame_id],
                                        feature_positions[feature_id]);
            }
        }

        int n = track_ids.size();
        CHECK(matches.size() == n);
        for(int i = 0; i < n; i++){
            int track_id = track_ids[i];
            const auto& match = matches[i];
            if(!match.valid || match.variance > Parameter::GetInstance().variance_register) continue;
            ceres::CostFunction* cost_function = PointToPlaneError::Create(match.plane);
            problem.AddResidualBlock(cost_function, nullptr, feature_positions[track_id]);
        }

        // double total_cost = 0.0;
        // std::vector<double> residuals;
        // ceres::Problem::EvaluateOptions eval_options;
        // eval_options.apply_loss_function = true;
        
        // problem.Evaluate(eval_options, &total_cost, &residuals, nullptr, nullptr);

        // 计算残差统计
        // int residual_count = residuals.size();
        // double sum_squared_residuals = 0.0;
        // double max_residual = 0.0;
        
        // for (size_t i = 0; i < residuals.size(); ++i) {
        //     sum_squared_residuals += residuals[i] * residuals[i];
        //     max_residual = std::max(max_residual, std::abs(residuals[i]));
            
        //     // if (i % 2 == 0) { // 每个观测有2个残差(x和y)
        //     //     int obs_index = i / 2;
        //     //     std::cout << "Residual " << obs_index << ": (" 
        //     //             << residuals[i] << ", " << residuals[i+1] << ")\n";
        //     // }
        // }
        
        // double rmse = std::sqrt(sum_squared_residuals / residual_count);
        
        // std::cout << "Residual Validation Report:\n";
        // std::cout << "  Total cost: " << total_cost << "\n";
        // std::cout << "  Number of residuals: " << residual_count << "\n";
        // std::cout << "  RMSE: " << rmse << "\n";
        // std::cout << "  Max residual: " << max_residual << "\n";

        // 执行优化
        ceres::Solver::Summary summary;
        ceres::Solve(options_, &problem, &summary);
        // std::cout << summary.BriefReport() << std::endl;

        // 更新优化后的位姿
        for (auto& frame : frames_) {
            double* rot = camera_rotations[frame.frame_id];
            double* trans = camera_translations[frame.frame_id];
            
            frame.q = Eigen::Quaterniond(rot[0], rot[1], rot[2], rot[3]);
            frame.t = Eigen::Vector3d(trans[0], trans[1], trans[2]);
        }

        // 更新优化后的特征点位置
        for (auto& [feature_id, pos] : feature_positions) {
            feature_tracks_[feature_id].variable_latest().optimized_map_point = Eigen::Vector3d(pos[0], pos[1], pos[2]);
            feature_tracks_[feature_id].variable_latest().is_outlier = false;
        }

        // 释放内存
        for (auto& [_, rot] : camera_rotations) delete[] rot;
        for (auto& [_, trans] : camera_translations) delete[] trans;
        for (auto& [_, pos] : feature_positions) delete[] pos;
    }

    std::vector<uint64_t> newly_matures_;
private:
    std::map<uint64_t, FeatureTrack> feature_tracks_;
    

    std::deque<CameraFrame> frames_; // 滑动窗口内的相机帧
    ceres::Solver::Options options_;

    uint64_t current_frame_id_ = 0;
    // 参数
    size_t window_size_;
    float depth_thresh_;
    float motion_thresh_;
    int min_continuous_inliers_;

    
    // 更新或初始化特征轨迹
    void updateFeatureTracks(const std::vector<std::pair<uint64_t, FeatureObservation>>& frame_features) {
        for (const auto& [id, obs] : frame_features) {
            auto it = feature_tracks_.find(id);
            if (it == feature_tracks_.end()) {
                // 新特征点
                FeatureTrack new_track;
                new_track.id = id;
                new_track.observations.push_back(obs);
                feature_tracks_[id] = new_track;
            } else {
                // 已有特征点，添加新观测
                it->second.observations.push_back(obs);
            }
        }
    }
    

    // 滑动窗口管理
    void manageSlidingWindow() {
        auto it = feature_tracks_.begin();
        while (it != feature_tracks_.end()) {
            auto& track = it->second;
            
            // 移除超出时间窗口的旧观测
            while (track.windowSize() > window_size_) {
                track.observations.pop_front();
            }
            
            // 如果观测为空或者最新观测距离当前帧太旧，移除该轨迹
            if (track.observations.empty() || 
                (current_frame_id_ - track.latest().frame_id) > window_size_) {
                it = feature_tracks_.erase(it);  // 移除空或过期的轨迹
            } else {
                ++it;
            }
        }
        if(frames_.size() > window_size_) {
            frames_.pop_front();
        }
    }
    
    // 时间窗口内的一致性检查
    void checkTemporalConsistency() {
        for (auto& [id, track] : feature_tracks_) {
            if (track.windowSize() < 2) continue;  // 至少需要2个观测
            
            // 1. 深度一致性检查
            checkDepthConsistency(track);
            
            // 2. 运动一致性检查
            // checkMotionConsistency(track);
            
            // 3. 更新连续内点计数
            if (!track.latest().is_outlier) {
                track.continuous_inliers++;
            } else {
                track.continuous_inliers = 0;
            }
        }
    }
    
    // 深度一致性检查
    void checkDepthConsistency(FeatureTrack& track) {
        // 计算时间窗口内的深度变化统计
        std::vector<float> depths;
        for (const auto& obs : track.observations) {
            depths.push_back(obs.map_point.norm());
        }
        
        float mean_depth = std::accumulate(depths.begin(), depths.end(), 0.0f) / depths.size();
        float max_diff = 0.0f;
        for (float d : depths) {
            max_diff = std::max(max_diff, std::abs(d - mean_depth));
        }
        
        // 检查最新观测
        float current_depth = track.latest().map_point.norm();
        float depth_diff = std::abs(current_depth - mean_depth);
        
        if (depth_diff > depth_thresh_ * mean_depth) {
            track.variable_latest().is_outlier = true;
            track.depth_consistency = 0.f;
        } else {
            track.depth_consistency = 1.0f - (depth_diff / (depth_thresh_ * mean_depth));
        }
    }
    
    // 运动一致性检查
    void checkMotionConsistency(FeatureTrack& track) {
        if (track.windowSize() < 3) return;  // 至少需要3个观测
        
        // 计算运动加速度（二阶差分）
        std::vector<Eigen::Vector2f> motions;
        for (size_t i = 1; i < track.observations.size(); ++i) {
            const auto& prev = track.observations[i-1];
            const auto& curr = track.observations[i];
            motions.push_back(Eigen::Vector2f(
                curr.pixel_pos.x - prev.pixel_pos.x,
                curr.pixel_pos.y - prev.pixel_pos.y
            ));
        }
        
        // 计算加速度
        float max_acceleration = 0.0f;
        for (size_t i = 1; i < motions.size(); ++i) {
            Eigen::Vector2f accel = motions[i] - motions[i-1];
            max_acceleration = std::max(max_acceleration, accel.norm());
        }
        
        if (max_acceleration > motion_thresh_) {
            track.variable_latest().is_outlier = true;
            track.motion_consistency = 0.f;
        } else {
            track.motion_consistency = 1.0f - (max_acceleration / motion_thresh_);
        }
    }
    
    // 更新特征点成熟度
    void updateMaturity() {
        newly_matures_.clear();
        for (auto& [id, track] : feature_tracks_) {
            // 成熟条件：
            // 1. 时间窗口满
            // 2. 深度和运动一致性达标
            // 3. 连续内点数达标
            bool mature_before = track.is_mature;
            track.is_mature = (track.windowSize() >= 3) &&
                             (track.depth_consistency > 0.7f) &&
                             /* (track.motion_consistency > 0.7f) && */
                             (track.continuous_inliers >= min_continuous_inliers_);
            bool mature_after = track.is_mature;
            if(!mature_before && mature_after){
                newly_matures_.push_back(id);
            }
        }
    }

    void updateFeatureDepth(FeatureTrack& track) {
        if (track.observations.size() < 3) return; // 至少需要3个观测
        
        // 方法1：加权平均（考虑深度一致性和观测时间）
        Eigen::Vector3d updated_map_point = Eigen::Vector3d::Zero();
        float total_weight = 0.0f;
        
        for (const auto& obs : track.observations) {
            if (obs.is_outlier) continue;
            
            // 权重计算：时间衰减 + 深度一致性
            float time_weight = 1.0f / (1.0f + (current_frame_id_ - obs.frame_id));
            float depth_weight = 1.0f - std::abs(obs.map_point.norm() - track.latest().map_point.norm()) / track.latest().map_point.norm();
            
            float weight = time_weight * depth_weight;
            updated_map_point += obs.map_point * weight;
            total_weight += weight;
        }
        
        if (total_weight > 0.0f) {
            updated_map_point /= total_weight;
            track.variable_latest().updated_map_point = updated_map_point;
        }
        
        // 方法2：中值滤波（更鲁棒，适合有噪声的环境）
        
        // std::vector<float> depths;
        // for (const auto& obs : track.observations) {
        //     if (!obs.is_outlier) {
        //         depths.push_back(obs.map_point.norm());
        //     }
        // }
        
        // if (!depths.empty()) {
        //     std::sort(depths.begin(), depths.end());
        //     float median_depth = depths[depths.size()/2];
        //     track.variable_latest().updated_map_point = track.latest().map_point;
        //     track.variable_latest().updated_map_point.normalize();
        //     track.variable_latest().updated_map_point *= median_depth;
        // }
       
    }


    bool triangulateFeature(const FeatureTrack& track, Eigen::Vector3d& triangulated_point) {
        if (track.observations.size() < 2) {
            // 至少需要两个不同视角的观测
            return false;
        }

        // 构建DLT方程组 Ax = 0
        Eigen::MatrixXd A(2 * track.observations.size(), 4);
        A.setZero();

        for (size_t i = 0; i < track.observations.size(); ++i) {
            const FeatureObservation& obs = track.observations[i];
            uint64_t frame_id = obs.frame_id;

            // 找到对应的相机帧
            const CameraFrame* cam_frame = nullptr;
            for (const auto& frame : frames_) {
                if (frame.frame_id == frame_id) {
                    cam_frame = &frame;
                    break;
                }
            }
            if (!cam_frame) {
                // 理论上不应该发生，因为特征观测的帧应该在滑动窗口内
                continue;
            }

            // 相机位姿：世界 -> 相机 (q, t)
            // 变换矩阵 T_cw = [R_cw  t_cw; 0 1]
            Eigen::Matrix3d R_cw = cam_frame->q.toRotationMatrix();
            Eigen::Vector3d t_cw = cam_frame->t;

            // 观测的归一化坐标 (x, y, 1)
            double x = obs.x3d_norm[0];
            double y = obs.x3d_norm[1];

            // 根据DLT，每个观测贡献两个方程
            // [x*R_cw.row(2) - R_cw.row(0)] * [X; Y; Z; 1] = 0
            // [y*R_cw.row(2) - R_cw.row(1)] * [X; Y; Z; 1] = 0
            A.row(2*i) = x * R_cw.row(2) - R_cw.row(0);
            A.row(2*i + 1) = y * R_cw.row(2) - R_cw.row(1);
        }

        // 求解 Ax = 0 的最小二乘解，即A的最小奇异值对应的右奇异向量
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d homogeneous_point = svd.matrixV().col(3);

        // 转换为非齐次坐标 (X, Y, Z)
        if (homogeneous_point(3) == 0) {
            // 齐次分量为0，无效
            return false;
        }
        triangulated_point = homogeneous_point.head<3>() / homogeneous_point(3);

        // 验证三角化的点是否在所有相机的前方
        for (size_t i = 0; i < track.observations.size(); ++i) {
            const FeatureObservation& obs = track.observations[i];
            uint64_t frame_id = obs.frame_id;

            const CameraFrame* cam_frame = nullptr;
            for (const auto& frame : frames_) {
                if (frame.frame_id == frame_id) {
                    cam_frame = &frame;
                    break;
                }
            }
            if (!cam_frame) continue;

            Eigen::Matrix3d R_cw = cam_frame->q.toRotationMatrix();
            Eigen::Vector3d t_cw = cam_frame->t;

            // 将3D点从世界坐标系转换到相机坐标系
            Eigen::Vector3d point_cam = R_cw * triangulated_point + t_cw;

            // 检查z坐标是否为正（在相机前方）
            if (point_cam.z() < 0.01) { // 留出一点余量，防止数值问题
                return false;
            }
        }

        return true;
    }


    

};
#endif // __FEATURE_FILTER_H__