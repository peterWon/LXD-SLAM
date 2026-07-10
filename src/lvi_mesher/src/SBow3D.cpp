#include "SBoW3D.h"

namespace BoW3D
{
    SubmapBoW3D::SubmapBoW3D(float thr, int thf, int num_features)
        : m_thr(thr), m_thf(thf), m_num_add_retrieve_features(num_features) {}

    void SubmapBoW3D::addSubmap(Submap* pSubmap, const std::vector<LinK3DDescriptor>& descriptors) {
        int submap_id = pSubmap->get_id(); // 假设 Submap 有获取 ID 的接口
        m_history_database[submap_id] = descriptors;

        for (size_t i = 0; i < descriptors.size(); ++i) {
            for (size_t j = 0; j < 180; ++j) {
                float dist = descriptors[i].descriptor[j];
                if (dist > 0.1) { // 只索引有效距离
                    (*this)[std::make_pair(dist, (int)j)].insert(std::make_pair(submap_id, (int)i));
                }
            }
        }
    }

    bool SubmapBoW3D::retrieve(Submap* pCurrentSubmap, 
                               const std::vector<LinK3DDescriptor>& current_descriptors,
                               SubmapLoopInfo& loop_info) {
        
        std::map<int, int> submap_counts; // 统计候选子地图的共视单词数
        int current_id = pCurrentSubmap->get_id();

        // 1. 词袋检索候选
        for (const auto& desc_obj : current_descriptors) {
            for (int j = 0; j < 180; ++j) {
                float val = desc_obj.descriptor[j];
                if (val <= 0.1) continue;

                auto it = find(std::make_pair(val, j));
                if (it != end()) {
                    for (const auto& occurrence : it->second) {
                        int hist_id = occurrence.first;
                        if (current_id - hist_id < 15) continue; // 跳过时间太近的
                        submap_counts[hist_id]++;
                    }
                }
            }
        }

        // 2. 寻找最佳候选并进行几何验证
        int best_id = -1;
        int max_count = 0;
        for (auto const& [id, count] : submap_counts) {
            if (count > max_count && count >= m_thf) {
                max_count = count;
                best_id = id;
            }
        }

        if (best_id == -1) return false;

        // 3. 几何一致性验证 (SVD)
        const auto& history_descriptors = m_history_database[best_id];
        std::vector<std::pair<int, int>> matches;
        
        // 简单匹配逻辑：由于 LinK3D 的描述子在 Submap 局部坐标系下已生成
        for (size_t i = 0; i < current_descriptors.size(); ++i) {
            for (size_t j = 0; j < history_descriptors.size(); ++j) {
                // 调用 LinK3D 原始的相似度计算（简化版）
                int common = 0;
                for(int k=0; k<180; ++k) {
                    if(current_descriptors[i].descriptor[k] > 0 && history_descriptors[j].descriptor[k] > 0)
                        if(std::abs(current_descriptors[i].descriptor[k] - history_descriptors[j].descriptor[k]) < 0.2)
                            common++;
                }
                if (common > 12) {
                    matches.push_back({(int)i, (int)j});
                    break;
                }
            }
        }

        if (matches.size() >= 5) { // 至少5对匹配点认为初步回环成功
            std::vector<Eigen::Vector3d> pts_curr, pts_hist;
            for (auto& m : matches) {
                pts_curr.push_back(current_descriptors[m.first].position);
                pts_hist.push_back(history_descriptors[m.second].position);
            }
            computeRelativePose(pts_hist, pts_curr, loop_info.R_h_c, loop_info.t_h_c);
            loop_info.matched_submap_id = best_id;
            loop_info.vMatchedIndex = matches;
            return true;
        }

        return false;
    }

    void SubmapBoW3D::computeRelativePose(const std::vector<Eigen::Vector3d>& pts_hist,
                                         const std::vector<Eigen::Vector3d>& pts_curr,
                                         Eigen::Matrix3d& R, Eigen::Vector3d& t) {
        // 使用 Eigen 提供的 Umeyama 算法求解最小二乘位姿
        Eigen::MatrixXd src(3, pts_curr.size());
        Eigen::MatrixXd dst(3, pts_hist.size());
        for (size_t i = 0; i < pts_curr.size(); ++i) {
            src.col(i) = pts_curr[i];
            dst.col(i) = pts_hist[i];
        }
        Eigen::Matrix4d T = Eigen::umeyama(src, dst, false); // 不考虑缩放
        R = T.block<3, 3>(0, 0);
        t = T.block<3, 1>(0, 3);
    }
}