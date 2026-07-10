#pragma once

#include "extended_scan_context.h"
#include <vector>
#include <map>
#include <utility>
#include <Eigen/Dense>

class ScanContextDatabase {
public:
    ScanContextDatabase();

    // 將新的 ScanContext 加入數據庫
    void addKeyframe(ExtendedScanContext& sc, int frame_id);

    // 回環檢測介面
    // 返回值: <是否檢測到回環, 匹配到的歷史幀ID>
    // 輸出參數: min_dist (最佳距離), best_shift (最佳旋轉平移量)
    std::pair<bool, int> detectLoopClosure(
        const ExtendedScanContext& query_sc, 
        int current_frame_id, 
        double& min_dist, 
        int& best_shift,
        int exclude_recent_frames = 50, // 排除最近的連續幀
        int top_k_candidates = 10,      // Ring-Key 粗篩選的候選數量
        double dist_threshold = 0.15,   // 最終匹配的距離閾值
        double iou_threshold = 0.3      // IoU 閾值
    );

private:
    // 為帶有可見性掩碼的 SC 計算 Ring-Key
    Eigen::VectorXd extractMaskedRingKey(const ExtendedScanContext& sc) const;

private:
    std::vector<ExtendedScanContext> sc_database_;
    std::map<int, Eigen::VectorXd> ring_keys_; // 快速索引: frame_id -> RingKey
};