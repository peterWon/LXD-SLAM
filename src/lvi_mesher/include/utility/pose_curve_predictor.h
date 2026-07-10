#include <iostream>
#include <deque>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/sophus/se3.hpp>

// 定义一个结构体保存时间戳和位姿
struct FrameData {
    double timestamp;
    Sophus::SE3d pose;
};

class PoseCurvePredictor {
public:
    /**
     * @param window_size 滑动窗口大小（推荐 4）
     * @param poly_degree 多项式阶数（推荐 2，即匀加速运动模型）
     */
    PoseCurvePredictor(int window_size = 3, int poly_degree = 1)
        : window_size_(window_size), poly_degree_(poly_degree) {
        if (poly_degree_ >= window_size_) {
            std::cerr << "Warning: Polynomial degree must be less than window size. "
                      << "Adjusting to " << window_size_ - 1 << std::endl;
            poly_degree_ = window_size_ - 1;
        }
    }

    // 传入当前帧的准确时间戳和估计位姿
    void addPose(double timestamp, const Eigen::Isometry3d& pose) {
        FrameData data;
        data.timestamp = timestamp;
        data.pose = Sophus::SE3d(pose.matrix()); // Isometry3d 转 Sophus::SE3d
        
        history_.push_back(data);
        if (history_.size() > window_size_) {
            history_.pop_front();
        }
    }

    // 根据下一帧点云的时间戳，预测其初始位姿
    Eigen::Isometry3d predictAt(double target_timestamp) {
        int n = history_.size();
        
        // 1. 数据不足 2 帧，无法预测，返回上一帧
        if (n < 2) {
            return history_.empty() ? Eigen::Isometry3d::Identity() : Eigen::Isometry3d(history_.back().pose.matrix());
        }
        
        // 2. 数据不足以进行高阶拟合，退化为基于李代数的一阶匀速推算
        if (n < poly_degree_ + 1) {
            const auto& prev = history_[n - 2];
            const auto& curr = history_[n - 1];
            double dt_history = curr.timestamp - prev.timestamp;
            double dt_pred = target_timestamp - curr.timestamp;
            
            if (dt_history < 1e-6) return Eigen::Isometry3d(curr.pose.matrix());

            // 计算李代数上的“速度” xi
            Sophus::SE3d delta_pose = prev.pose.inverse() * curr.pose;
            Eigen::Matrix<double, 6, 1> xi = delta_pose.log();
            
            // 按时间比例缩放 xi 并映射回李群
            Eigen::Matrix<double, 6, 1> xi_pred = xi * (dt_pred / dt_history);
            Sophus::SE3d pose_pred = curr.pose * Sophus::SE3d::exp(xi_pred);
            
            return Eigen::Isometry3d(pose_pred.matrix());
        }

        // 3. 多项式曲线拟合 (基于 Sophus se3 空间)
        Eigen::MatrixXd A(n, poly_degree_ + 1);
        Eigen::MatrixXd Y(n, 6);

        // 以滑动窗口的第一帧作为时间起点 t=0 和 位姿基准 T_ref
        double t0 = history_.front().timestamp;
        Sophus::SE3d T_ref = history_.front().pose;
        Sophus::SE3d T_ref_inv = T_ref.inverse();

        for (int i = 0; i < n; ++i) {
            // 时间归一化：相对时间 dt
            double dt = history_[i].timestamp - t0;
            
            // 构造范德蒙德矩阵 A
            for (int j = 0; j <= poly_degree_; ++j) {
                A(i, j) = std::pow(dt, j);
            }

            // 计算相对位姿，并映射到 se(3) 李代数空间 (6维向量)
            Sophus::SE3d T_rel = T_ref_inv * history_[i].pose;
            Eigen::Matrix<double, 6, 1> xi = T_rel.log(); 
            
            // 将 6维李代数向量存入 Y
            Y.row(i) = xi.transpose();
        }

        // QR 分解求解最小二乘系数矩阵 C: C = (A^T A)^-1 A^T Y
        Eigen::MatrixXd C = A.colPivHouseholderQr().solve(Y);

        // 计算预测目标时间的 dt_target
        double dt_target = target_timestamp - t0;
        
        // 在 se(3) 空间计算预测的 6维向量
        Eigen::Matrix<double, 6, 1> xi_pred = Eigen::Matrix<double, 6, 1>::Zero();
        for (int j = 0; j <= poly_degree_; ++j) {
            xi_pred += C.row(j).transpose() * std::pow(dt_target, j);
        }

        // 通过指数映射 exp() 还原到 SE(3) 群，并乘回基准位姿
        Sophus::SE3d T_pred = T_ref * Sophus::SE3d::exp(xi_pred);

        // 转换回 Eigen::Isometry3d 输出
        return Eigen::Isometry3d(T_pred.matrix());
    }

private:
    int window_size_;
    int poly_degree_;
    std::deque<FrameData> history_;
};