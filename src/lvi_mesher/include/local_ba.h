#ifndef LOCAL_BA_H
#define LOCAL_BA_H
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <ceres/ceres.h>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

class FeatureCentricBundleAdjustment {
public:
    // 帧观测数据结构
    struct FrameObservation {
        int frame_id;
        cv::Point2f uv; // 归一化坐标
    };

    // 特征点数据结构
    struct FeaturePoint {
        int id; // 特征点ID
        Eigen::Vector3d position; // 全局3D坐标
        std::vector<FrameObservation> observations; // 被哪些帧观测到
    };

    // 相机帧数据结构
    struct CameraFrame {
        int frame_id;
        Eigen::Quaterniond q; // 旋转四元数
        Eigen::Vector3d t; // 平移向量
    };

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
            residuals[0] = predicted_x - T(observed_uv_.x);
            residuals[1] = predicted_y - T(observed_uv_.y);
            
            if(T(residuals[0]) * T(residuals[0]) + T(residuals[1]) * T(residuals[1]) > T(0.09))
                return false; // 大于0.09的误差，认为是异常值
            return true;
        }

        static ceres::CostFunction* Create(const cv::Point2f& observed_uv) {
            return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
                new ReprojectionError(observed_uv)));
        }

    private:
        cv::Point2f observed_uv_;
    };

    // 构造函数
    FeatureCentricBundleAdjustment(int window_size = 10) : window_size_(window_size) {
        options_.linear_solver_type = ceres::DENSE_SCHUR;
        options_.minimizer_progress_to_stdout = false;
        options_.max_num_iterations = 10;
    }

    // 添加新帧到优化窗口
    void AddFrame(const CameraFrame& frame) {
        // 检查是否已存在相同frame_id的帧
        for (const auto& f : frames_) {
            if (f.frame_id == frame.frame_id) {
                std::cerr << "Warning: Frame with ID " << frame.frame_id 
                        << " already exists. Skipping." << std::endl;
                return;
            }
        }

        if (frames_.size() >= window_size_) {
            // 移除最旧的帧
            int old_frame_id = frames_.front().frame_id;
            frames_.pop_front();
            
            // 收集所有需要检查的特征点ID
            std::vector<int> feature_ids_to_check;
            for (auto& [id, feature] : features_) {
                // 从该特征点的观测中移除旧帧的观测
                auto& obs = feature.observations;
                obs.erase(std::remove_if(obs.begin(), obs.end(), 
                    [old_frame_id](const FrameObservation& o) { 
                        return o.frame_id == old_frame_id; 
                    }), obs.end());
                
                // 记录需要检查的特征点ID
                feature_ids_to_check.push_back(id);
            }

            // 检查并移除不再被任何帧观测的特征点
            for (int id : feature_ids_to_check) {
                if (features_[id].observations.empty()) {
                    features_.erase(id);
                }
            }
        }
        
        // 添加新帧
        frames_.push_back(frame);
        
        // 确保窗口大小不超过设定值
        while (frames_.size() > window_size_) {
            int excess_frame_id = frames_.front().frame_id;
            frames_.pop_front();
            
            // 同样需要处理特征点观测
            std::vector<int> ids_to_check;
            for (auto& [id, feature] : features_) {
                auto& obs = feature.observations;
                obs.erase(std::remove_if(obs.begin(), obs.end(), 
                    [excess_frame_id](const FrameObservation& o) { 
                        return o.frame_id == excess_frame_id; 
                    }), obs.end());
                
                ids_to_check.push_back(id);
            }
            
            for (int id : ids_to_check) {
                if (features_[id].observations.empty()) {
                    features_.erase(id);
                }
            }
        }
    }

    // 添加特征点观测
    void AddFeatureObservation(int feature_id, int frame_id, 
                             const cv::Point2f& uv, 
                             const Eigen::Vector3d& initial_position) {
        FrameObservation obs;
        obs.frame_id = frame_id;
        obs.uv = uv;

        // 查找或创建特征点
        if (features_.find(feature_id) == features_.end()) {
            FeaturePoint feature;
            feature.id = feature_id;
            feature.position = initial_position;
            feature.observations.push_back(obs);
            features_[feature_id] = feature;
        } else {
            features_[feature_id].observations.push_back(obs);
            features_[feature_id].position = initial_position; // 更新初始位置
        }
    }

    // 执行BA优化
    void Optimize() {
        ceres::Problem problem;

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
        //     problem.SetParameterBlockConstant(camera_rotations[frame.frame_id]);
        //     problem.SetParameterBlockConstant(camera_translations[frame.frame_id]);
        // }

        // 准备特征点参数块
        std::unordered_map<int, double*> feature_positions;
        for (auto& [feature_id, feature] : features_) {
            feature_positions[feature_id] = new double[3];
            feature_positions[feature_id][0] = feature.position.x();
            feature_positions[feature_id][1] = feature.position.y();
            feature_positions[feature_id][2] = feature.position.z();
            problem.AddParameterBlock(feature_positions[feature_id], 3);
        }


        // 添加残差块
        for (auto& [feature_id, feature] : features_) {
            for (const auto& obs : feature.observations) {
                // 检查该帧是否在当前窗口中
                if (camera_rotations.find(obs.frame_id) == camera_rotations.end()) {
                    continue;
                }

                ceres::CostFunction* cost_function = ReprojectionError::Create(obs.uv);
                
                problem.AddResidualBlock(cost_function,
                                        new ceres::CauchyLoss(0.5), // 使用默认的Huber损失函数
                                        camera_rotations[obs.frame_id],
                                        camera_translations[obs.frame_id],
                                        feature_positions[feature_id]);
            }
        }

        /* double total_cost = 0.0;
        std::vector<double> residuals;
        ceres::Problem::EvaluateOptions eval_options;
        eval_options.apply_loss_function = true;
        
        problem.Evaluate(eval_options, &total_cost, &residuals, nullptr, nullptr);

        // 计算残差统计
        int residual_count = residuals.size();
        double sum_squared_residuals = 0.0;
        double max_residual = 0.0;
        
        for (size_t i = 0; i < residuals.size(); ++i) {
            sum_squared_residuals += residuals[i] * residuals[i];
            max_residual = std::max(max_residual, std::abs(residuals[i]));
            
            if (i % 2 == 0) { // 每个观测有2个残差(x和y)
                int obs_index = i / 2;
                std::cout << "Residual " << obs_index << ": (" 
                        << residuals[i] << ", " << residuals[i+1] << ")\n";
            }
        }
        
        double rmse = std::sqrt(sum_squared_residuals / residual_count);
        
        std::cout << "Residual Validation Report:\n";
        std::cout << "  Total cost: " << total_cost << "\n";
        std::cout << "  Number of residuals: " << residual_count << "\n";
        std::cout << "  RMSE: " << rmse << "\n";
        std::cout << "  Max residual: " << max_residual << "\n"; */

        // 执行优化
        ceres::Solver::Summary summary;
        ceres::Solve(options_, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;

        // 更新优化后的位姿
        for (auto& frame : frames_) {
            double* rot = camera_rotations[frame.frame_id];
            double* trans = camera_translations[frame.frame_id];
            
            frame.q = Eigen::Quaterniond(rot[0], rot[1], rot[2], rot[3]);
            frame.t = Eigen::Vector3d(trans[0], trans[1], trans[2]);
        }

        // 更新优化后的特征点位置
        for (auto& [feature_id, pos] : feature_positions) {
            features_[feature_id].position = Eigen::Vector3d(pos[0], pos[1], pos[2]);
        }

        // 释放内存
        for (auto& [_, rot] : camera_rotations) delete[] rot;
        for (auto& [_, trans] : camera_translations) delete[] trans;
        for (auto& [_, pos] : feature_positions) delete[] pos;
    }

    // 获取优化后的位姿
    const std::deque<CameraFrame>& GetOptimizedFrames() const {
        return frames_;
    }

    // 获取优化后的特征点
    const std::unordered_map<int, FeaturePoint>& GetOptimizedFeatures() const {
        return features_;
    }

    int current_window_size() const {
        return frames_.size();
    }

    int window_size() const {
        return window_size_;
    }

    int current_feature_size() const {
        return features_.size();
    }

private:
    int window_size_;
    std::deque<CameraFrame> frames_;
    std::unordered_map<int, FeaturePoint> features_;
    ceres::Solver::Options options_;
};
#endif // LOCAL_BA_H