#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "submap.h" // 引入您的子地图类

namespace BoW3D
{
    // 复用原始代码中的哈希函数
    template <typename T>
    inline void hash_combine(std::size_t &seed, const T &val) {
        seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            std::size_t seed = 0;
            hash_combine(seed, p.first);
            hash_combine(seed, p.second);
            return seed;
        }
    };

    // 存储回环候选信息
    struct SubmapLoopInfo {
        int matched_submap_id;
        Eigen::Matrix3d R_h_c; // 相对旋转
        Eigen::Vector3d t_h_c; // 相对平移
        std::vector<std::pair<int, int>> vMatchedIndex; // 特征匹配对索引
    };

    // 逆索引数据库：<描述子维度值, 维度ID> -> <子地图ID, 描述子在子地图中的索引>
    class SubmapBoW3D : public std::unordered_map<std::pair<float, int>, 
                                                 std::unordered_set<std::pair<int, int>, pair_hash>, 
                                                 pair_hash>
    {
    public:
        // thr: 距离比值阈值, thf: 最小共视词数阈值, num_features: 检索时考虑的特征数
        SubmapBoW3D(float thr = 3.5, int thf = 5, int num_features = 10);

        // 将完成建图的子地图加入数据库并建立索引
        void addSubmap(Submap* pSubmap, const std::vector<LinK3DDescriptor>& descriptors);

        // 检索回环候选
        bool retrieve(Submap* pCurrentSubmap, 
                      const std::vector<LinK3DDescriptor>& current_descriptors,
                      SubmapLoopInfo& loop_info);

    private:
        // 计算两个 LinK3D 描述子之间的匹配（考虑旋转不变性）
        void matchDescriptors(const std::vector<float>& desc1, 
                              const std::vector<float>& desc2, 
                              std::vector<std::pair<int, int>>& vMatched);

        // SVD 求解位姿
        void computeRelativePose(const std::vector<Eigen::Vector3d>& pts_hist,
                                const std::vector<Eigen::Vector3d>& pts_curr,
                                Eigen::Matrix3d& R, Eigen::Vector3d& t);

        float m_thr;
        int m_thf;
        int m_num_add_retrieve_features;

        // 存储历史子地图的特征点位置信息，用于 SVD 验证
        std::unordered_map<int, std::vector<LinK3DDescriptor>> m_history_database;
    };
}