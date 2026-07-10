#pragma once

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <cmath>
#include <algorithm>

// 假設您的工程中已有 Mat4d 定義，若無則使用 Eigen::Matrix4d
using Mat4d = Eigen::Matrix4d;

class ExtendedScanContext {
public:
    enum VisibilityState {
        UNKNOWN = 0, 
        FREE = 1,    
        OCCUPIED = 2 
    };

    ExtendedScanContext(int num_rings = 20, int num_sectors = 60, double max_radius = 80.0);

    // 生成接口
    void generateFromPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // 相似度計算
    double computeMaskedSimilarity(const ExtendedScanContext& other, int& best_shift, double iou_threshold = 0.3) const;

public:
    // --- 基礎參數 ---
    int num_rings_;
    int num_sectors_;
    double max_radius_;
    double ring_res_; 
    double sector_res_; 

    // --- 綁定數據與描述子屬性 (修正 Bug 的關鍵) ---
    int frame_id_ = -1;                // 關鍵幀 ID
    Mat4d pose_ = Mat4d::Identity();   // 在 Odom 系下的全局位姿
    Eigen::VectorXd ring_key_;         // 用於快速檢索的 Ring-Key
    int observed_bins_count_ = 0;      // 非 UNKNOWN 的 Bin 總數，加速 IoU 計算

    // --- 三大通道矩陣 ---
    Eigen::MatrixXd height_matrix_;      
    Eigen::MatrixXd density_matrix_;     // 密度通道 (保留自您的原始設計)
    Eigen::MatrixXi visibility_matrix_;  // 可見性通道
};