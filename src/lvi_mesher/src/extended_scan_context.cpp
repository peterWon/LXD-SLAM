#include "extended_scan_context.h"
#include <limits>

ExtendedScanContext::ExtendedScanContext(int num_rings, int num_sectors, double max_radius)
    : num_rings_(num_rings), num_sectors_(num_sectors), max_radius_(max_radius) {
    
    ring_res_ = max_radius_ / num_rings_;
    sector_res_ = 2.0 * M_PI / num_sectors_;

    // 初始化矩陣
    height_matrix_ = Eigen::MatrixXd::Zero(num_rings_, num_sectors_);
    visibility_matrix_ = Eigen::MatrixXi::Constant(num_rings_, num_sectors_, UNKNOWN);
}

void ExtendedScanContext::generateFromPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // 記錄每個扇區最遠的觀測環數，用於後續標記 FREE 區域
    std::vector<int> max_ring_per_sector(num_sectors_, -1);
    
    // Step 1: 遍歷點雲，標記 OCCUPIED 與最大高度
    for (const auto& pt : cloud->points) {
        double radius = std::hypot(pt.x, pt.y);
        if (radius >= max_radius_ || radius < 1e-3) continue; // 忽略過遠或過近的點

        double angle = std::atan2(pt.y, pt.x); // [-pi, pi]
        
        // 映射到行列索引
        int r_idx = static_cast<int>(radius / ring_res_);
        int s_idx = static_cast<int>((angle + M_PI) / sector_res_);
        
        // 防呆保護
        r_idx = std::min(std::max(r_idx, 0), num_rings_ - 1);
        s_idx = std::min(std::max(s_idx, 0), num_sectors_ - 1);

        // 更新高度矩陣 (取最高點)
        if (pt.z > height_matrix_(r_idx, s_idx)) {
            height_matrix_(r_idx, s_idx) = pt.z;
        }

        // 標記為 OCCUPIED
        visibility_matrix_(r_idx, s_idx) = OCCUPIED;

        // 更新該扇區的最大觀測距離
        if (r_idx > max_ring_per_sector[s_idx]) {
            max_ring_per_sector[s_idx] = r_idx;
        }
    }

    // Step 2: 2.5D 極座標射線追蹤，標記 FREE 空間
    observed_bins_count_ = 0;
    for (int s = 0; s < num_sectors_; ++s) {
        int r_max = max_ring_per_sector[s];
        
        // 如果該扇區完全沒有點 (例如固態雷達盲區)，r_max 為 -1，整個扇區保持 UNKNOWN
        for (int r = 0; r <= r_max; ++r) {
            if (visibility_matrix_(r, s) != OCCUPIED) {
                visibility_matrix_(r, s) = FREE; // 射線途經但無點，視為 FREE
            }
            observed_bins_count_++; // 統計非 UNKNOWN 的 Bin 總數
        }
    }
}

double ExtendedScanContext::computeMaskedSimilarity(const ExtendedScanContext& other, int& best_shift, double iou_threshold) const {
    double min_distance = std::numeric_limits<double>::max();
    best_shift = -1;

    // 定義狀態不一致的額外懲罰值 (相當於高度差 1.0 米的懲罰)
    const double state_mismatch_penalty = 1.0; 

    // 遍歷所有可能的列平移 (旋轉)
    for (int shift = 0; shift < num_sectors_; ++shift) {
        int valid_overlap_count = 0;
        double current_shift_dist = 0.0;

        for (int r = 0; r < num_rings_; ++r) {
            for (int s = 0; s < num_sectors_; ++s) {
                int shifted_s = (s + shift) % num_sectors_; // 模擬旋轉

                int state_A = visibility_matrix_(r, shifted_s);
                int state_B = other.visibility_matrix_(r, s);

                // 僅在雙方都有觀測的交集區域計算差異
                if (state_A != UNKNOWN && state_B != UNKNOWN) {
                    valid_overlap_count++;
                    
                    // L1 高度誤差
                    double h_diff = std::abs(height_matrix_(r, shifted_s) - other.height_matrix_(r, s));
                    current_shift_dist += h_diff;

                    // 幾何狀態懲罰 (例如一個是 FREE，另一個是 OCCUPIED)
                    if (state_A != state_B) {
                        current_shift_dist += state_mismatch_penalty;
                    }
                }
            }
        }

        // 避免沒有交集導致的除以零
        if (valid_overlap_count == 0) continue;

        // 計算 IoU (Intersection over Union)
        double union_count = this->observed_bins_count_ + other.observed_bins_count_ - valid_overlap_count;
        double iou = static_cast<double>(valid_overlap_count) / union_count;

        // 如果重疊率太低，直接丟棄該平移角度的結果
        if (iou < iou_threshold) continue;

        // 正規化距離
        double normalized_dist = current_shift_dist / valid_overlap_count;

        if (normalized_dist < min_distance) {
            min_distance = normalized_dist;
            best_shift = shift;
        }
    }

    // 如果所有的 shift 都未通過 IoU 測試，返回 -1.0 表示無效匹配
    if (best_shift == -1) {
        return -1.0;
    }

    return min_distance;
}