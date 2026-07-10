#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <deque>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

using namespace Eigen;

// 相机位姿：世界系到相机系 (T_cw = [R_cw | t_cw])
struct CameraPose {
    Quaterniond q_cw;
    Vector3d t_cw;

    CameraPose(const Quaterniond& q, const Vector3d& t) : q_cw(q), t_cw(t) {}

    Isometry3d T_cw() const {
        Isometry3d T = Isometry3d::Identity();
        T.rotate(q_cw);
        T.pretranslate(t_cw);
        return T;
    }
};

// 单次观测
struct Observation {
    int frame_id;
    Vector2d pt_norm;      // 相机归一化平面坐标 (x/z, y/z)

    cv::Point2f pixel_pos;    // 像素坐标, 及描述子
    cv::KeyPoint keypoint;
    BRIEF::bitset descriptor; 
    
    // Ray-cast 观测
    bool has_raycast = false;
    double raycast_depth = 0.;  // 当前相机坐标系下的深度
    double raycast_var = -1;    // 深度方差

    Observation(int f_id, const Vector2d& pt, const cv::Point2f& pixel, bool has_rc = false, double rc_d = 0.0, double rc_v = 0.0)
        : frame_id(f_id), pt_norm(pt), pixel_pos(pixel), has_raycast(has_rc), raycast_depth(rc_d), raycast_var(rc_v) {}
};

// 特征点 Track（维护滤波状态）
struct FeatureTrack {
    int feature_id;
    int anchor_frame_id;   // 首次观测到该特征的帧
    Vector2d anchor_pt;    // 在锚定帧的归一化坐标

    // 滤波器状态：锚定帧下的 逆深度 均值与方差
    double inv_depth_mu;
    double inv_depth_var;
    bool is_initialized;
    bool is_pure_visual;
    bool is_mature = false;   

    std::vector<Observation> observations;

    FeatureTrack(int id) 
        : feature_id(id), 
          is_initialized(false), 
          inv_depth_mu(0.0), 
          inv_depth_var(0.0),
          is_pure_visual(true) {}

    bool getPointInWorld(const CameraPose& anchor_pose, Vector3d& p_w) const {
        // 如果未初始化，或者逆深度接近0 (甚至小于0，代表在相机后方)，则视为无效点
        if (!is_initialized || inv_depth_mu <= 1e-5) {
            return false;
        }

        // 1. 恢复锚定帧相机坐标系下的 3D 点
        double depth = 1.0 / inv_depth_mu;
        Vector3d P_a(anchor_pt.x() * depth, anchor_pt.y() * depth, depth);

        // 2. 投影到世界坐标系
        // P_c = T_cw * P_w  =>  P_w = T_cw^-1 * P_c
        Isometry3d T_cw = anchor_pose.T_cw();
        p_w = T_cw.inverse() * P_a;

        return true;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////
class SlidingWindowDepthFilter {
private:
    int max_window_size_;
    std::deque<int> window_frame_ids_; // 维护滑动窗口内的帧ID顺序
    std::unordered_map<int, CameraPose> sliding_window_poses_;
    std::unordered_map<int, std::shared_ptr<FeatureTrack>> feature_tracks_;

    double pixel_noise_ = 1.0 / 800.0;

    

public:
    SlidingWindowDepthFilter(int max_size = 10) : max_window_size_(max_size) {}
    
    // TODO
    std::vector<uint64_t> newly_matures_;

    const std::unordered_map<int, std::shared_ptr<FeatureTrack>>& getFeatureTracks() const {
        return feature_tracks_;
    }
    
    std::shared_ptr<FeatureTrack> getFeatureTrackById(int feature_id) {
        auto it = feature_tracks_.find(feature_id);
        if (it != feature_tracks_.end()) {
            return it->second;
        }
        return nullptr;
    }

    /**
     * @brief 从滤波器获取特征点的世界系 3D 坐标
     * * @param feature_id 特征点ID
     * @param p_w [out] 输出的世界系 3D 坐标
     * @return true 获取成功
     * @return false 失败(特征不存在、未初始化，或锚定帧位姿已丢失)
     */
    bool getFeatureWorldPoint(int feature_id, Vector3d& p_w) const {
        auto track_it = feature_tracks_.find(feature_id);
        if (track_it == feature_tracks_.end()) {
            return false;
        }

        const auto& track = track_it->second;
        auto pose_it = sliding_window_poses_.find(track->anchor_frame_id);
        
        // 确保锚定帧的位姿在滑动窗口中仍然有效
        if (pose_it == sliding_window_poses_.end()) {
            return false; 
        }

        return track->getPointInWorld(pose_it->second, p_w);
    }

    // 传入新的一帧位姿，并检查是否需要滑动窗口
    void addFramePose(int frame_id, const CameraPose& pose) {
        sliding_window_poses_.insert({frame_id, pose});
        window_frame_ids_.push_back(frame_id);

        // 如果超过窗口大小，执行滑动机制
        if (window_frame_ids_.size() > max_window_size_) {
            slideWindow();
        }
    }

    // 传入新的一帧对某个特征点的观测
    void addObservation(int feature_id, const Observation& obs) {
        if (feature_tracks_.find(feature_id) == feature_tracks_.end()) {
            // 新特征点，创建Track并设当前帧为锚定帧
            auto track = std::make_shared<FeatureTrack>(feature_id);
            track->anchor_frame_id = obs.frame_id;
            track->anchor_pt = obs.pt_norm;
            feature_tracks_[feature_id] = track;
        }
        feature_tracks_[feature_id]->observations.push_back(obs);
    }

    // 执行滑动窗口内所有特征的深度更新
    void updateFilter() {
        for (auto& pair : feature_tracks_) {
            auto& track = pair.second;
            if (track->observations.size() < 2) continue;

            const auto& anchor_pose = sliding_window_poses_.at(track->anchor_frame_id);
            Isometry3d T_aw = anchor_pose.T_cw();

            // 遍历所有观测进行融合
            for (const auto& obs : track->observations) {
                if (obs.frame_id == track->anchor_frame_id) continue; // 跳过锚定帧自身的观测
                
                const auto& curr_pose = sliding_window_poses_.at(obs.frame_id);
                Isometry3d T_cw = curr_pose.T_cw();
                Isometry3d T_ca = T_cw * T_aw.inverse();

                // 1. 如果有 Ray-cast 深度，优先利用 Ray-cast 更新
                if (obs.has_raycast && obs.raycast_depth > 0.1) {
                    fuseRaycastDepth(track, obs, T_ca);
                } 
                // 2. 如果没有 Ray-cast，尝试纯视觉三角化更新
                else {
                    fuseTriangulationDepth(track, obs, T_ca);
                }
            }
        }
    }

    void updateMaturity() {
        newly_matures_.clear();
        for (auto& [id, track] : feature_tracks_) {
            bool mature_before = track->is_mature;
            
            bool var_converged = (track->inv_depth_var < 0.05); 
            
            track->is_mature = track->observations.size() >= 3 && var_converged;
                              
            if(!mature_before && track->is_mature){
                newly_matures_.push_back(id);
            }
        }
    }

    // 获取滤波后的深度 (返回锚定帧下的深度, 如果未初始化则返回 -1)
    double getEstimatedDepth(int feature_id) const {
        auto it = feature_tracks_.find(feature_id);
        if (it != feature_tracks_.end() && it->second->is_initialized) {
            return 1.0 / it->second->inv_depth_mu;
        }
        return -1.0;
    }
private:
    // 融合 Ray-cast 观测
    void fuseRaycastDepth(std::shared_ptr<FeatureTrack>& track, const Observation& obs, const Isometry3d& T_ca) {
        // 在当前帧的 3D 点
        Vector3d P_c = Vector3d(obs.pt_norm.x(), obs.pt_norm.y(), 1.0) * obs.raycast_depth;
        
        // 转换回锚定帧
        Vector3d P_a = T_ca.inverse() * P_c;
        double depth_a = P_a.z();

        if (depth_a <= 0.1) return;

        // 方差传递（简化处理：假设旋转对沿射线方向的方差影响不大）
        double var_d_a = obs.raycast_var;

        // 转为逆深度及其方差
        double inv_d_obs = 1.0 / depth_a;
        double inv_var_obs = var_d_a / (depth_a * depth_a * depth_a * depth_a);

        apply1DKalmanFilter(track, inv_d_obs, inv_var_obs);

        track->is_pure_visual = false;
    }

    // 融合三角化观测
    void fuseTriangulationDepth(std::shared_ptr<FeatureTrack>& track, const Observation& obs, const Isometry3d& T_ca) {
        Vector3d f_a(track->anchor_pt.x(), track->anchor_pt.y(), 1.0);
        Vector3d f_c(obs.pt_norm.x(), obs.pt_norm.y(), 1.0);
        f_a.normalize();
        f_c.normalize();

        // 检查视差 (Parallax) 是否足够
        Vector3d f_a_in_c = T_ca.rotation() * f_a;
        double cos_parallax = f_a_in_c.dot(f_c);
        if (cos_parallax > 0.9998) return; // 视差太小，接近1度，不适合三角化

        // DLT 三角化求锚定帧下的 3D 点
        Vector3d P_a = triangulatePoint(T_ca, f_a, f_c);
        double depth_a = P_a.z();

        if (depth_a <= 0.1) return;

        // 估算三角化不确定度 (基于极线几何和视差角近似)
        double sin_parallax2 = 1.0 - cos_parallax * cos_parallax;
        double baseline2 = T_ca.translation().squaredNorm();
        // sigma_d^2 = (sigma_px^2 * d^4) / (baseline^2 * sin(alpha)^2)
        double var_d_tri = (pixel_noise_ * pixel_noise_ * std::pow(depth_a, 4)) / (baseline2 * sin_parallax2);

        double inv_d_obs = 1.0 / depth_a;
        double inv_var_obs = var_d_tri / (depth_a * depth_a * depth_a * depth_a);

        apply1DKalmanFilter(track, inv_d_obs, inv_var_obs);
    }

    // 1D 卡尔曼滤波更新 (信息滤波形式)
    void apply1DKalmanFilter(std::shared_ptr<FeatureTrack>& track, double mu_obs, double var_obs) {
        if (!track->is_initialized) {
            track->inv_depth_mu = mu_obs;
            track->inv_depth_var = var_obs;
            track->is_initialized = true;
            return;
        }

        // 融合：将先验分布与观测分布相乘
        double prior_mu = track->inv_depth_mu;
        double prior_var = track->inv_depth_var;

        double K = prior_var / (prior_var + var_obs);
        track->inv_depth_mu = prior_mu + K * (mu_obs - prior_mu);
        track->inv_depth_var = (1.0 - K) * prior_var;
    }

    // DLT 三角化计算 3D 点 (返回锚定帧下的坐标)
    // T_ca: 锚定帧到当前帧的变换
    Vector3d triangulatePoint(const Isometry3d& T_ca, const Vector3d& f_a, const Vector3d& f_c) const {
        Matrix4d A;
        // 锚定帧投影矩阵 P_a = [I | 0]
        A.row(0) = f_a[0] * Matrix<double, 1, 4>::Unit(2) - Matrix<double, 1, 4>::Unit(0);
        A.row(1) = f_a[1] * Matrix<double, 1, 4>::Unit(2) - Matrix<double, 1, 4>::Unit(1);

        // 当前帧投影矩阵 P_c = T_ca
        Matrix<double, 3, 4> P_c = T_ca.matrix().block<3, 4>(0, 0);
        A.row(2) = f_c[0] * P_c.row(2) - P_c.row(0);
        A.row(3) = f_c[1] * P_c.row(2) - P_c.row(1);

        JacobiSVD<Matrix4d> svd(A, ComputeFullV);
        Vector4d pt_homo = svd.matrixV().col(3);
        
        return pt_homo.head<3>() / pt_homo(3);
    }

    // 核心滑动逻辑
    void slideWindow() {
        int oldest_frame_id = window_frame_ids_.front();
        window_frame_ids_.pop_front();

        auto old_pose_it = sliding_window_poses_.find(oldest_frame_id);
        if (old_pose_it == sliding_window_poses_.end()) return;
        
        Isometry3d T_ow = old_pose_it->second.T_cw(); // old_anchor to world

        // 遍历所有特征点，处理与最旧帧相关的逻辑
        for (auto it = feature_tracks_.begin(); it != feature_tracks_.end(); ) {
            auto& track = it->second;

            // 如果该特征的锚定帧就是要被移除的最旧帧
            if (track->anchor_frame_id == oldest_frame_id) {
                // 寻找窗口内该特征点的下一个观测帧作为新的锚定帧
                int new_anchor_id = -1;
                for (const auto& obs : track->observations) {
                    if (obs.frame_id != oldest_frame_id && 
                        sliding_window_poses_.count(obs.frame_id)) {
                        new_anchor_id = obs.frame_id;
                        break;
                    }
                }

                if (new_anchor_id != -1 && track->is_initialized) {
                    // 执行锚定帧转移 (Anchor Shift)
                    Isometry3d T_nw = sliding_window_poses_.at(new_anchor_id).T_cw();
                    Isometry3d T_no = T_nw * T_ow.inverse(); // T_new_old
                    
                    shiftAnchor(track, T_no, new_anchor_id);
                } else if (new_anchor_id == -1) {
                    // 如果该特征点在窗口内没有后续观测了，说明跟踪丢失或生命周期结束
                    // 在此可以将其转入 Backend Map 或直接丢弃，这里作丢弃处理
                    it = feature_tracks_.erase(it);
                    continue;
                } else {
                    // 未初始化但有后续观测，直接更换锚定帧即可，无需方差转移
                    track->anchor_frame_id = new_anchor_id;
                    for (const auto& obs : track->observations) {
                        if (obs.frame_id == new_anchor_id) {
                            track->anchor_pt = obs.pt_norm;
                            break;
                        }
                    }
                }
            }

            // 清理该 track 中属于最旧帧的 observation (释放内存)
            track->observations.erase(
                std::remove_if(track->observations.begin(), track->observations.end(),
                    [oldest_frame_id](const Observation& obs) {
                        return obs.frame_id == oldest_frame_id;
                    }),
                track->observations.end()
            );

            ++it;
        }

        // 移除最旧帧的位姿
        sliding_window_poses_.erase(oldest_frame_id);
    }

    // 锚定帧状态转移函数
    void shiftAnchor(std::shared_ptr<FeatureTrack>& track, const Isometry3d& T_no, int new_anchor_id) {
        double mu_A = track->inv_depth_mu;
        double var_A = track->inv_depth_var;

        Vector3d f_A(track->anchor_pt.x(), track->anchor_pt.y(), 1.0);
        
        // 提取旋转和部分平移分量
        Vector3d R_row3 = T_no.rotation().row(2);
        double t_z = T_no.translation().z();

        // 核心系数 C
        double C = R_row3.dot(f_A);

        // 1. 计算新的逆深度均值 mu_B
        double mu_B = mu_A / (C + mu_A * t_z);

        // 2. 计算雅可比 J 并传播方差 var_B
        double J = C * std::pow(mu_B / mu_A, 2);
        double var_B = J * J * var_A;

        // 3. 更新新锚定帧的归一化平面坐标
        // P_B = R_{BA} P_A + t_{BA}
        Vector3d P_B = T_no.rotation() * (f_A / mu_A) + T_no.translation();
        Vector2d new_anchor_pt = P_B.head<2>() / P_B.z();

        // 更新 Track 状态
        track->inv_depth_mu = mu_B;
        track->inv_depth_var = var_B;
        track->anchor_frame_id = new_anchor_id;
        track->anchor_pt = new_anchor_pt;
    }
};