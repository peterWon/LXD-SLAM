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

class TemporalFeatureFilter {
public:
    struct CameraFrame {
        int frame_id;
        Eigen::Quaterniond q; // R_cw: world to camera
        Eigen::Vector3d t;    // t_cw: world to camera
    };

    struct FeatureObservation {
        uint64_t frame_id;
        uint64_t feature_id;
        cv::Point2f pixel_pos; // uv in image
        
        cv::KeyPoint keypoint;
        BRIEF::bitset descriptor; 
        Eigen::Vector3d x3d_norm;           // Normalized coords in camera frame (X/Z, Y/Z, 1)
        
        // --- 3D Map Points ---
        Eigen::Vector3d map_point;          // World frame: Ray-casted from Mesh
        double mesh_variance = 0.1;         // Variance of the intersected mesh face
        
        Eigen::Vector3d updated_map_point;  // World frame: After 1D-KF inverse depth update
        bool is_outlier = false;
    };

    struct FeatureTrack {
        uint64_t id; 
        std::deque<FeatureObservation> observations; 
        
        // --- 1D Kalman Filter States for Inverse Depth ---
        double inv_depth = -1.0;     // State mean: ρ = 1/d
        double inv_depth_var = 1.0;  // State variance: σ_ρ^2
        
        float depth_consistency = 0.f;
        float motion_consistency = 0.f;
        bool is_mature = false;
        int continuous_inliers = 0;
        
        // Return physical depth in meters (clamped to avoid div-by-zero)
        double currentDepth() const { 
            if (inv_depth <= 1e-6) return 10000.0; 
            return 1.0 / inv_depth; 
        }

        const FeatureObservation& latest() const { return observations.back(); }
        FeatureObservation& variable_latest() { return observations.back(); }
        size_t windowSize() const { return observations.size(); }
    };

    TemporalFeatureFilter(size_t window_size = 5,
                         float depth_thresh = 0.2,
                         float motion_thresh = 3.0,
                         int min_continuous_inliers = 3)
        : window_size_(window_size),
          depth_thresh_(depth_thresh),
          motion_thresh_(motion_thresh),
          min_continuous_inliers_(min_continuous_inliers) {}
    
    // Process new frame observations
    void processFrame(uint64_t frame_id, 
                     const std::vector<std::pair<uint64_t, FeatureObservation>>& frame_features,
                     const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
        current_frame_id_ = frame_id;
        CameraFrame new_frame;
        new_frame.frame_id = frame_id;
        new_frame.q = Eigen::Quaterniond(R).normalized();
        new_frame.t = t;
        frames_.push_back(new_frame);

        updateFeatureTracks(frame_features);
        manageSlidingWindow();
        updateDepthsAndCheckConsistency();
        updateMaturity();
    }
    
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
    
    const std::map<uint64_t, FeatureTrack>& getFeatureTracks() const {
        return feature_tracks_;
    }

    FeatureTrack* getFeatureTrackById(uint64_t id) {
        auto it = feature_tracks_.find(id);
        if (it != feature_tracks_.end()) return &it->second;
        return nullptr;
    }

    void setMotionThreshold(float thresh) { motion_thresh_ = thresh; }


    std::vector<uint64_t> newly_matures_;

private:
    std::map<uint64_t, FeatureTrack> feature_tracks_;
    std::deque<CameraFrame> frames_; 
    ceres::Solver::Options options_;

    uint64_t current_frame_id_ = 0;
    size_t window_size_;
    float depth_thresh_;
    float motion_thresh_;
    int min_continuous_inliers_;
    
    void updateFeatureTracks(const std::vector<std::pair<uint64_t, FeatureObservation>>& frame_features) {
        for (const auto& [id, obs] : frame_features) {
            auto it = feature_tracks_.find(id);
            if (it == feature_tracks_.end()) {
                FeatureTrack new_track;
                new_track.id = id;
                new_track.observations.push_back(obs);
                feature_tracks_[id] = new_track;
            } else {
                it->second.observations.push_back(obs);
            }
        }
    }
    
    void manageSlidingWindow() {
        auto it = feature_tracks_.begin();
        while (it != feature_tracks_.end()) {
            auto& track = it->second;
            while (track.windowSize() > window_size_) {
                track.observations.pop_front();
            }
            if (track.observations.empty() || 
                (current_frame_id_ - track.latest().frame_id) > window_size_) {
                it = feature_tracks_.erase(it); 
            } else {
                ++it;
            }
        }
        if(frames_.size() > window_size_) {
            frames_.pop_front();
        }
    }
    
    // Core: 1D-KF Inverse Depth Update + Coordinate Frame Transforms + Outlier Rejection
    void updateDepthsAndCheckConsistency() {
        for (auto& [id, track] : feature_tracks_) {
            if (track.windowSize() < 2) continue; 
            
            auto& latest_obs = track.variable_latest();
            
            const CameraFrame* cam_frame = nullptr;
            for (const auto& frame : frames_) {
                if (frame.frame_id == latest_obs.frame_id) {
                    cam_frame = &frame; break;
                }
            }
            if (!cam_frame) continue;

            // 1. Transform World Map Point to Camera Frame: P_cam = R_cw * P_world + t_cw
            Eigen::Vector3d p_cam = cam_frame->q * latest_obs.map_point + cam_frame->t;
            double d_mesh = p_cam.z(); 
            
            if (d_mesh < 0.1) {
                latest_obs.is_outlier = true;
                track.continuous_inliers = 0;
                continue;
            }

            // 2. Propagate Mesh Variance to Inverse Depth Variance
            double rho_mesh = 1.0 / d_mesh;
            double rho_mesh_var = latest_obs.mesh_variance * std::pow(rho_mesh, 4) + 1e-6;

            // 3. Triangulation Cross-Check
            Eigen::Vector3d p_tri;
            if (triangulateFeature(track, p_tri)) {
                Eigen::Vector3d p_tri_cam = cam_frame->q * p_tri + cam_frame->t;
                double rho_tri = 1.0 / p_tri_cam.z();
                
                if (std::abs(rho_mesh - rho_tri) > depth_thresh_ * rho_mesh) {
                    rho_mesh_var *= 5.0; // Penalize mismatch
                }
            } else {
                rho_mesh_var *= 2.0; // Penalize lack of geometric verification
            }

            // 4. 1D Kalman Filter Update
            if (track.inv_depth < 0) {
                track.inv_depth = rho_mesh;
                track.inv_depth_var = rho_mesh_var;
            } else {
                // 3-Sigma Outlier Rejection
                double innovation = std::abs(rho_mesh - track.inv_depth);
                if (innovation > 3.0 * std::sqrt(track.inv_depth_var + rho_mesh_var)) {
                    latest_obs.is_outlier = true;
                    track.continuous_inliers = 0;
                    continue; 
                }

                // Standard KF Update
                double K = track.inv_depth_var / (track.inv_depth_var + rho_mesh_var);
                track.inv_depth = track.inv_depth + K * (rho_mesh - track.inv_depth);
                track.inv_depth_var = (1.0 - K) * track.inv_depth_var;
            }

            // 5. Reconstruct the World Point using the Filtered Depth
            latest_obs.is_outlier = false;
            track.continuous_inliers++;
            
            double final_depth = track.currentDepth();
            Eigen::Vector3d p_cam_updated = latest_obs.x3d_norm * final_depth;
            
            // P_world = R_cw^{-1} * (P_cam - t_cw)
            latest_obs.updated_map_point = cam_frame->q.inverse() * (p_cam_updated - cam_frame->t);
            
            track.depth_consistency = std::exp(-track.inv_depth_var * 10.0);
        }
    }
    
    void updateMaturity() {
        newly_matures_.clear();
        for (auto& [id, track] : feature_tracks_) {
            bool mature_before = track.is_mature;
            
            bool var_converged = (track.inv_depth_var < 0.05); 
            
            track.is_mature = (track.windowSize() >= 3) &&
                              var_converged &&
                              (track.continuous_inliers >= min_continuous_inliers_);
                              
            if(!mature_before && track.is_mature){
                newly_matures_.push_back(id);
            }
        }
    }

    // Standard DLT Triangulation (Operates in World Frame)
    bool triangulateFeature(const FeatureTrack& track, Eigen::Vector3d& triangulated_point) {
        if (track.observations.size() < 2) return false;

        Eigen::MatrixXd A(2 * track.observations.size(), 4);
        A.setZero();

        for (size_t i = 0; i < track.observations.size(); ++i) {
            const FeatureObservation& obs = track.observations[i];
            const CameraFrame* cam_frame = nullptr;
            for (const auto& frame : frames_) {
                if (frame.frame_id == obs.frame_id) {
                    cam_frame = &frame;
                    break;
                }
            }
            if (!cam_frame) continue;

            Eigen::Matrix3d R_cw = cam_frame->q.toRotationMatrix();
            Eigen::Vector3d t_cw = cam_frame->t;

            double x = obs.x3d_norm[0];
            double y = obs.x3d_norm[1];

            A.row(2*i) = x * R_cw.row(2) - R_cw.row(0);
            A.row(2*i + 1) = y * R_cw.row(2) - R_cw.row(1);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4d homogeneous_point = svd.matrixV().col(3);

        if (std::abs(homogeneous_point(3)) < 1e-6) return false;
        triangulated_point = homogeneous_point.head<3>() / homogeneous_point(3);

        for (size_t i = 0; i < track.observations.size(); ++i) {
            const FeatureObservation& obs = track.observations[i];
            const CameraFrame* cam_frame = nullptr;
            for (const auto& frame : frames_) {
                if (frame.frame_id == obs.frame_id) {
                    cam_frame = &frame;
                    break;
                }
            }
            if (!cam_frame) continue;

            Eigen::Vector3d point_cam = cam_frame->q.toRotationMatrix() * triangulated_point + cam_frame->t;
            if (point_cam.z() < 0.01) return false;
        }
        return true;
    }
};
#endif // __FEATURE_FILTER_H__