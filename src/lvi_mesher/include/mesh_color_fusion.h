#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using VecXd = Eigen::VectorXd;
using MatXd = Eigen::MatrixXd;

// -------------------------------
// 1. 数据结构定义
// -------------------------------

struct MeshVertex {
    Vec3d position;
    Vec3d normal; // 必须归一化
};

struct CameraIntrinsics {
    double fx, fy, cx, cy;
};

struct CameraExtrinsics {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

struct Camera {
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
};

// -------------------------------
// 2. 颜色融合主类（支持 RGB）
// -------------------------------

class MeshColorFusionGP {
public:
    MeshColorFusionGP() {}

    void AddVertex(const MeshVertex& vertex) {
        vertices_.push_back(vertex);
        vertex_data_.emplace_back();
        vertex_data_.back().vertex = vertex;

        // 每个顶点有 3 个 GP：R, G, B
        gps_.emplace_back(&MeshColorFusionGP::RBFKernel); // R
        gps_.emplace_back(&MeshColorFusionGP::RBFKernel); // G
        gps_.emplace_back(&MeshColorFusionGP::RBFKernel); // B
    }

    void AddFrame(const Camera& camera, const cv::Mat& image) {
        cameras_.push_back(camera);
        images_.push_back(image);
    }

    void UpdateObservations() {
        for (size_t vid = 0; vid < vertices_.size(); ++vid) {
            auto& vdata = vertex_data_[vid];
            const auto& vertex = vdata.vertex;

            // 每个顶点有 3 个观测列表：R, G, B
            vdata.observations_r.clear();
            vdata.observations_g.clear();
            vdata.observations_b.clear();

            for (size_t fid = 0; fid < cameras_.size(); ++fid) {
                const auto& cam = cameras_[fid];
                const auto& img = images_[fid];

                Vec2d pixel;
                if (!ProjectVertexToPixel(vertex, cam, pixel)) continue;

                int x = static_cast<int>(std::round(pixel.x()));
                int y = static_cast<int>(std::round(pixel.y()));
                if (x < 0 || y < 0 || x >= img.cols || y >= img.rows) continue;

                // OpenCV 格式是 BGR
                cv::Vec3b bgr = img.at<cv::Vec3b>(y, x);
                double b = static_cast<double>(bgr[0]);
                double g = static_cast<double>(bgr[1]);
                double r = static_cast<double>(bgr[2]);

                double distance, cos_angle;
                ComputeGeometryInfo(vertex, cam, distance, cos_angle);

                double px_norm = pixel.x() / img.cols;
                double py_norm = pixel.y() / img.rows;

                VecXd feat(4);
                feat << distance, cos_angle, px_norm, py_norm;

                // 分别保存 R / G / B 通道的观测
                vdata.observations_r.emplace_back(feat, b);
                vdata.observations_g.emplace_back(feat, g);
                vdata.observations_b.emplace_back(feat, r);
            }
        }
    }

    void FuseColors() {
        fused_colors_.resize(vertices_.size(), Vec3f(0, 0, 0));

        for (size_t vid = 0; vid < vertices_.size(); ++vid) {
            auto& vdata = vertex_data_[vid];

            Vec3f fused(0, 0, 0);

            // Fusion for R channel (index 0)
            if (!vdata.observations_r.empty()) {
                fused[2] = PredictChannel(vdata.observations_r, 0 /* R */);
            }
            // Fusion for G channel (index 1)
            if (!vdata.observations_g.empty()) {
                fused[1] = PredictChannel(vdata.observations_g, 1 /* G */);
            }
            // Fusion for B channel (index 2)
            if (!vdata.observations_b.empty()) {
                fused[0] = PredictChannel(vdata.observations_b, 2 /* B */);
            }

            fused_colors_[vid] = fused;
        }
    }

    Vec3f GetFusedColor(int vertex_id) const {
        if (vertex_id < 0 || vertex_id >= static_cast<int>(fused_colors_.size()))
            return Vec3f(0, 0, 0);
        return fused_colors_[vertex_id];
    }

    int GetVertexCount() const { return vertices_.size(); }

private:
    struct VertexObservationData {
        MeshVertex vertex;
        std::vector<std::pair<VecXd, double>> observations_r; // B 通道（OpenCV是BGR）
        std::vector<std::pair<VecXd, double>> observations_g;
        std::vector<std::pair<VecXd, double>> observations_b;
    };

    std::vector<MeshVertex> vertices_;
    std::vector<Camera> cameras_;
    std::vector<cv::Mat> images_;
    std::vector<VertexObservationData> vertex_data_;
    std::vector<Vec3f> fused_colors_;
    std::vector<class SimpleGaussianProcess> gps_; // 每个顶点 3 个 GP：R,G,B

    bool ProjectVertexToPixel(const MeshVertex& vertex, const Camera& camera, Vec2d& pixel) const {
        Eigen::Vector3d p_cam = camera.extrinsics.R * vertex.position + camera.extrinsics.t;
        if (p_cam.z() <= 1e-6) return false;

        double x = p_cam.x() / p_cam.z();
        double y = p_cam.y() / p_cam.z();

        double u = camera.intrinsics.fx * x + camera.intrinsics.cx;
        double v = camera.intrinsics.fy * y + camera.intrinsics.cy;

        pixel = Vec2d(u, v);
        return true;
    }

    void ComputeGeometryInfo(
        const MeshVertex& vertex,
        const Camera& camera,
        double& distance,
        double& cos_angle
    ) const {
        Eigen::Vector3d camera_origin = GetCameraOrigin(camera);
        Eigen::Vector3d view_dir = (vertex.position - camera_origin).normalized();
        distance = (vertex.position - camera_origin).norm();
        cos_angle = view_dir.dot(vertex.normal);
    }

    Eigen::Vector3d GetCameraOrigin(const Camera& camera) const {
        return -(camera.extrinsics.R.transpose() * camera.extrinsics.t);
    }

    double PredictChannel(const std::vector<std::pair<VecXd, double>>& obs, int gp_index) const {
        const auto& gps = this->gps_;
        const auto& vertex_obs = obs;

        size_t n = vertex_obs.size();
        if (n == 0) return 0.0;

        MatXd X(n, 4);
        VecXd y(n);
        for (size_t i = 0; i < n; ++i) {
            X.row(i) = vertex_obs[i].first;
            y(i) = vertex_obs[i].second;
        }

        gps[gp_index].Train(X, y);

        // 用第一个观测的特征做预测（或可改进为用均值特征）
        double fused_value = 0.0;
        for (const auto& o : vertex_obs) {
            double pred = gps[gp_index].Predict(o.first);
            fused_value += pred;
        }
        fused_value /= n;

        return fused_value;
    }

    // -------------------------------
    // 3. 高斯过程实现
    // -------------------------------

    class SimpleGaussianProcess {
    public:
        using KernelFunc = std::function<double(const VecXd&, const VecXd&)>;

        SimpleGaussianProcess(KernelFunc kernel, double noise_var = 1e-4)
            : kernel_(kernel), noise_variance_(noise_var) {}

        void Train(const MatXd& X, const VecXd& y) {
            X_train_ = X;
            y_train_ = y;
            n_ = X.rows();

            MatXd K(n_, n_);
            for (int i = 0; i < n_; ++i) {
                for (int j = 0; j < n_; ++j) {
                    K(i, j) = kernel_(X.row(i), X.row(j));
                }
            }
            K += noise_variance_ * MatXd::Identity(n_, n_);

            L_ = K.llt().matrixL();
        }

        double Predict(const VecXd& x_star) const {
            int n = X_train_.rows();

            VecXd k_star(n);
            for (int i = 0; i < n; ++i) {
                k_star(i) = kernel_(x_star, X_train_.row(i));
            }

            VecXd alpha = L_.solve(y_train_);
            alpha = L_.transpose().solve(alpha);

            double mean = k_star.dot(alpha);
            return mean;
        }

    private:
        KernelFunc kernel_;
        double noise_variance_;
        MatXd X_train_;
        VecXd y_train_;
        MatXd L_;
        int n_;
    };

    static double RBFKernel(const VecXd& x1, const VecXd& x2, double length_scale = 1.0, double sigma_f = 1.0) {
        double sqdist = (x1 - x2).squaredNorm();
        return sigma_f * sigma_f * std::exp(-0.5 * sqdist / (length_scale * length_scale));
    }
};