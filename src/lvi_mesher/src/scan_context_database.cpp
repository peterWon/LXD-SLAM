#include "scan_context_database.h"
#include <algorithm>
#include <limits>
#include <iostream>

ScanContextDatabase::ScanContextDatabase() {}

Eigen::VectorXd ScanContextDatabase::extractMaskedRingKey(const ExtendedScanContext& sc) const {
    Eigen::VectorXd ring_key = Eigen::VectorXd::Zero(sc.num_rings_);
    
    // 針對每個環 (Ring) 提取特徵
    for (int r = 0; r < sc.num_rings_; ++r) {
        double sum_height = 0.0;
        int occupied_count = 0;

        for (int s = 0; s < sc.num_sectors_; ++s) {
            // 只統計明確被佔用的網格
            if (sc.visibility_matrix_(r, s) == ExtendedScanContext::OCCUPIED) {
                sum_height += sc.height_matrix_(r, s);
                occupied_count++;
            }
        }

        // 使用該環內有效障礙物的平均高度作為特徵 (對遮擋具備一定的魯棒性)
        if (occupied_count > 0) {
            ring_key(r) = sum_height / occupied_count;
        }
    }
    return ring_key;
}

void ScanContextDatabase::addKeyframe(ExtendedScanContext& sc, int frame_id) {
    sc.frame_id_ = frame_id;
    
    // 預先計算並儲存 Ring-Key
    sc.ring_key_ = extractMaskedRingKey(sc);
    ring_keys_[frame_id] = sc.ring_key_;
    
    sc_database_.push_back(sc);
}

std::pair<bool, int> ScanContextDatabase::detectLoopClosure(
    const ExtendedScanContext& query_sc, 
    int current_frame_id, 
    double& min_dist, 
    int& best_shift,
    int exclude_recent_frames, 
    int top_k_candidates, 
    double dist_threshold, 
    double iou_threshold) 
{
    min_dist = std::numeric_limits<double>::max();
    int best_frame_id = -1;
    best_shift = -1;

    // 數據庫太小或剛起步時，不進行回環檢測
    if (sc_database_.size() < exclude_recent_frames + 1) {
        return {false, -1};
    }

    // Step 1: 提取當前幀的 Ring-Key
    Eigen::VectorXd query_ring_key = extractMaskedRingKey(query_sc);

    // Step 2: Phase 1 - 基於 Ring-Key 的快速粗篩選 (Coarse Search)
    // 儲存 <Ring-Key L1距離, 歷史幀ID>
    std::vector<std::pair<double, int>> candidate_distances;
    
    for (const auto& history_sc : sc_database_) {
        // 忽略最近的幀，避免相鄰幀的假陽性回環
        if (std::abs(current_frame_id - history_sc.frame_id_) < exclude_recent_frames) {
            continue;
        }

        // 計算 Ring-Key 之間的 L1 距離 (向量差異)
        double key_dist = (query_ring_key - history_sc.ring_key_).cwiseAbs().sum();
        candidate_distances.push_back({key_dist, history_sc.frame_id_});
    }

    // 將距離由小到大排序
    std::sort(candidate_distances.begin(), candidate_distances.end());

    // 選取前 Top-K 個候選幀
    int search_limit = std::min(top_k_candidates, static_cast<int>(candidate_distances.size()));
    
    // Step 3: Phase 2 - 基於掩碼的幾何精細匹配 (Fine Matching)
    for (int i = 0; i < search_limit; ++i) {
        int candidate_id = candidate_distances[i].second;
        
        // 找到對應的歷史 SC 物件
        auto it = std::find_if(sc_database_.begin(), sc_database_.end(), 
            [candidate_id](const ExtendedScanContext& sc) { return sc.frame_id_ == candidate_id; });
            
        if (it != sc_database_.end()) {
            int current_shift = -1;
            // 調用我們先前實現的掩碼計算函數
            double dist = query_sc.computeMaskedSimilarity(*it, current_shift, iou_threshold);

            // 如果匹配有效 (未被 IoU 拒絕)，且距離更小
            if (dist >= 0.0 && dist < min_dist) {
                min_dist = dist;
                best_shift = current_shift;
                best_frame_id = candidate_id;
            }
        }
    }

    // Step 4: 閾值判斷
    if (min_dist < dist_threshold && best_frame_id != -1) {
        return {true, best_frame_id};
    }

    return {false, -1};
}