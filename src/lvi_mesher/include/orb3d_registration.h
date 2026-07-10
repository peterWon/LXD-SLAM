#ifndef __LVI_MESHER_ORB3D_REGISTRATION__
#define __LVI_MESHER_ORB3D_REGISTRATION__
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// 3D点结构体，包含位置和ORB描述子
struct Point3D {
    Eigen::Vector3d position;
    cv::Mat descriptor;
    
    Point3D(const Eigen::Vector3d& pos, const cv::Mat& desc) 
        : position(pos), descriptor(desc) {}
};

// 匹配对结构体
struct Match {
    int idx1, idx2;
    double distance;
    
    Match(int i1, int i2, double dist) : idx1(i1), idx2(i2), distance(dist) {}
    
    bool operator<(const Match& other) const {
        return distance < other.distance;
    }
};

class FeatureMatcher3D {
private:
    double ratio_thresh_;
    int min_matches_;
    
public:
    FeatureMatcher3D(double ratio_thresh = 0.7, int min_matches = 10)
        : ratio_thresh_(ratio_thresh), min_matches_(min_matches) {}
    
    // 基于ORB特征进行匹配
    std::vector<Match> matchFeatures(const std::vector<Point3D>& points1, 
                                    const std::vector<Point3D>& points2) {
        std::vector<Match> matches;
        
        if (points1.empty() || points2.empty()) {
            LOG(ERROR) << "Empty point clouds!";
            return matches;
        }
        
        std::vector<std::vector<cv::DMatch>> knn_matches;
        
        // 构建描述子矩阵
        cv::Mat descriptors1, descriptors2;
        for (const auto& point : points1) {
            descriptors1.push_back(point.descriptor);
        }
        for (const auto& point : points2) {
            descriptors2.push_back(point.descriptor);
        }
        
        // KNN匹配，k=2用于比率测试
        cv::Ptr<cv::DescriptorMatcher> matcher;
        if (descriptors1.type() == CV_8U) {
            matcher = cv::makePtr<cv::FlannBasedMatcher>(
                cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        } else {
            matcher = cv::makePtr<cv::FlannBasedMatcher>();
        }
        // 使用OpenCV的BFMatcher进行特征匹配
        // cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        
        // 应用Lowe's比率测试筛选好的匹配
        for (const auto& knn_match : knn_matches) {
            if (knn_match.size() == 2) {
                if (knn_match[0].distance < ratio_thresh_ * knn_match[1].distance) {
                    matches.emplace_back(knn_match[0].queryIdx, 
                                       knn_match[0].trainIdx, 
                                       knn_match[0].distance);
                }
            }
        }
        
        // 按匹配距离排序
        std::sort(matches.begin(), matches.end());
        return matches;
    }

    std::vector<cv::DMatch> matchFeaturesCV(const std::vector<Point3D>& points1, 
                                    const std::vector<Point3D>& points2) {
        std::vector<cv::DMatch> good_matches;
        if (points1.empty() || points2.empty()) {
            LOG(ERROR) << "Empty point clouds!";
            return good_matches;
        }
        std::vector<std::vector<cv::DMatch>> knn_matches;
        
        // 构建描述子矩阵
        cv::Mat descriptors1, descriptors2;
        for (const auto& point : points1) {
            descriptors1.push_back(point.descriptor);
        }
        for (const auto& point : points2) {
            descriptors2.push_back(point.descriptor);
        }
        
        // KNN匹配，k=2用于比率测试
        cv::Ptr<cv::DescriptorMatcher> matcher;
        if (descriptors1.type() == CV_8U) {
            matcher = cv::makePtr<cv::FlannBasedMatcher>(
                cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        } else {
            matcher = cv::makePtr<cv::FlannBasedMatcher>();
        }
        // 使用OpenCV的BFMatcher进行特征匹配
        // cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
        
        // 应用Lowe's比率测试筛选好的匹配
        for (const auto& knn_match : knn_matches) {
            if (knn_match.size() == 2) {
                if (knn_match[0].distance < ratio_thresh_ * knn_match[1].distance) {
                    good_matches.push_back(knn_match[0]);
                }
            }
        }        
        return good_matches;
    }
};

class RANSACRegistration {
private:
    int max_iterations_;
    double inlier_threshold_;
    double confidence_;
    
public:
    RANSACRegistration(int max_iter = 10, double inlier_thresh = 0.05, double conf = 0.99)
        : max_iterations_(max_iter), inlier_threshold_(inlier_thresh), confidence_(conf) {}
    
    // 计算两点集之间的变换矩阵（最小二乘解）
    bool computeTransformation(const std::vector<Eigen::Vector3d>& src_points,
                              const std::vector<Eigen::Vector3d>& dst_points,
                              Eigen::Matrix3d& rotation,
                              Eigen::Vector3d& translation) {
        if (src_points.size() != dst_points.size() || src_points.size() < 3) {
            return false;
        }
        
        // 计算质心
        Eigen::Vector3d centroid_src = Eigen::Vector3d::Zero();
        Eigen::Vector3d centroid_dst = Eigen::Vector3d::Zero();
        
        for (size_t i = 0; i < src_points.size(); ++i) {
            centroid_src += src_points[i];
            centroid_dst += dst_points[i];
        }
        
        centroid_src /= src_points.size();
        centroid_dst /= dst_points.size();
        
        // 计算去质心坐标
        std::vector<Eigen::Vector3d> centered_src, centered_dst;
        for (size_t i = 0; i < src_points.size(); ++i) {
            centered_src.push_back(src_points[i] - centroid_src);
            centered_dst.push_back(dst_points[i] - centroid_dst);
        }
        
        // 计算W矩阵
        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < src_points.size(); ++i) {
            W += centered_src[i] * centered_dst[i].transpose();
        }
        
        // SVD分解
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        
        // 计算旋转矩阵
        rotation = V * U.transpose();
        
        // 确保右手坐标系（行列式为正）
        if (rotation.determinant() < 0) {
            V.col(2) *= -1;
            rotation = V * U.transpose();
        }
        
        // 计算平移向量
        translation = centroid_dst - rotation * centroid_src;
        
        return true;
    }
    
    // RANSAC配准
    bool ransacRegistration(const std::vector<Point3D>& points1,
                           const std::vector<Point3D>& points2,
                           const std::vector<Match>& matches,
                           Eigen::Matrix3d& best_rotation,
                           Eigen::Vector3d& best_translation,
                           std::vector<bool>& inlier_mask) {
        
        if (matches.size() < 5) {
            LOG(INFO) << "Not enough matches for RANSAC!";
            return false;
        }
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, matches.size() - 1);
        
        int best_inlier_count = 0;
        std::vector<bool> best_inlier_mask(matches.size(), false);
        
        // 准备匹配点对
        std::vector<Eigen::Vector3d> src_points, dst_points;
        for (const auto& match : matches) {
            src_points.push_back(points1[match.idx1].position);
            dst_points.push_back(points2[match.idx2].position);
        }
        
        int iterations = 0;
        while (iterations < max_iterations_) {
            // 随机选择5个点对
            std::vector<int> sample_indices;
            while (sample_indices.size() < 5) {
                int idx = dis(gen);
                if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
                    sample_indices.push_back(idx);
                }
            }
            
            // 构建样本点集
            std::vector<Eigen::Vector3d> sample_src, sample_dst;
            for (int idx : sample_indices) {
                sample_src.push_back(src_points[idx]);
                sample_dst.push_back(dst_points[idx]);
            }
            
            // 计算变换矩阵
            Eigen::Matrix3d rotation;
            Eigen::Vector3d translation;
            if (!computeTransformation(sample_src, sample_dst, rotation, translation)) {
                continue;
            }
            
            // 计算内点
            std::vector<bool> current_inlier_mask(matches.size(), false);
            int current_inlier_count = 0;
            
            for (size_t i = 0; i < matches.size(); ++i) {
                Eigen::Vector3d transformed = rotation * src_points[i] + translation;
                double error = (transformed - dst_points[i]).norm();
                
                if (error < inlier_threshold_) {
                    current_inlier_mask[i] = true;
                    current_inlier_count++;
                }
            }
            
            // 更新最佳模型
            if (current_inlier_count > best_inlier_count) {
                best_inlier_count = current_inlier_count;
                best_inlier_mask = current_inlier_mask;
                best_rotation = rotation;
                best_translation = translation;
                
                // 动态更新迭代次数
                double inlier_ratio = static_cast<double>(best_inlier_count) / matches.size();
                max_iterations_ = std::min(max_iterations_, 
                                          static_cast<int>(std::log(1 - confidence_) / 
                                                          std::log(1 - std::pow(inlier_ratio, 3))));
            }
            
            iterations++;
        }
        
        LOG(INFO) << "Best model has " << best_inlier_count << " inliers out of " << matches.size() << " matches.";

        inlier_mask = best_inlier_mask;
        
        return best_inlier_count > 0;
    }
    
    // 使用所有内点进行精化
    bool refineTransformation(const std::vector<Point3D>& points1,
                             const std::vector<Point3D>& points2,
                             const std::vector<Match>& matches,
                             const std::vector<bool>& inlier_mask,
                             Eigen::Matrix3d& refined_rotation,
                             Eigen::Vector3d& refined_translation) {
        
        std::vector<Eigen::Vector3d> inlier_src, inlier_dst;
        
        for (size_t i = 0; i < matches.size(); ++i) {
            if (inlier_mask[i]) {
                inlier_src.push_back(points1[matches[i].idx1].position);
                inlier_dst.push_back(points2[matches[i].idx2].position);
            }
        }
        
        if (inlier_src.size() < 3) {
            return false;
        }
        
        return computeTransformation(inlier_src, inlier_dst, refined_rotation, refined_translation);
    }
};

#endif