#ifndef __LXD_SLAM_ROT_ESTIMATOR__
#define __LXD_SLAM_ROT_ESTIMATOR__
#pragma once

#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/sophus/so3.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <utility/pointcloud_types.h>


class RangeImageRotEstimator {
public:
    struct Config {
        int width = 1024;                // 水平分辨率
        int height = 64;                 // 垂直线数
        double fov_up_deg = 15.0;        // 垂直视场角上限 (度)
        double fov_down_deg = -25.0;     // 垂直视场角下限 (度)
        double max_range = 100.0;        // 最大有效测距
        double min_range = 1.0;          // 最小有效测距
    };

    explicit RangeImageRotEstimator(const Config& config) 
        : cfg_(config) {
        fov_up_rad_ = cfg_.fov_up_deg * M_PI / 180.0;
        fov_down_rad_ = cfg_.fov_down_deg * M_PI / 180.0;
        fov_v_rad_ = fov_up_rad_ - fov_down_rad_;
        
        cv::createHanningWindow(hanning_window_, cv::Size(cfg_.width, cfg_.height), CV_64F);
    }

    /**
     * @brief 将 PCL 点云投影为 2D 距离图像 (CV_64FC1)
     * @param cloud 输入的 PointXYZT 格式的 PCL 点云指针
     * @return cv::Mat 距离图像
     */
    cv::Mat projectToRangeImage(const pcl::PointCloud<PointXYZT>::Ptr& cloud) const {
        cv::Mat range_img(cfg_.height, cfg_.width, CV_64FC1, cv::Scalar(0.0));

        // 判空保护
        if (!cloud || cloud->points.empty()) {
            return range_img;
        }

        for (const auto& pt : cloud->points) {
            // 过滤无效点 (PCL 中常见的 NaN 处理)
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                continue;
            }

            // PCL 点云坐标通常是 float，为保持精度转为 double 计算
            double x = pt.x;
            double y = pt.y;
            double z = pt.z;

            // 计算距离 (L2 Norm)
            double range = std::sqrt(x * x + y * y + z * z);
            if (range < cfg_.min_range || range > cfg_.max_range) {
                continue;
            }

            double yaw = std::atan2(y, x);
            double pitch = std::asin(z / range);

            int u = std::round((0.5 - yaw / (2.0 * M_PI)) * cfg_.width);
            int v = std::round((fov_up_rad_ - pitch) / fov_v_rad_ * cfg_.height);

            if (u < 0) u += cfg_.width;
            if (u >= cfg_.width) u -= cfg_.width;
            if (v < 0 || v >= cfg_.height) continue;

            double current_range = range_img.at<double>(v, u);
            if (current_range == 0.0 || range < current_range) {
                range_img.at<double>(v, u) = range;
            }
        }
        return range_img;
    }

    /**
     * @brief 利用相位相关快速估计两帧点云之间的相对旋转
     * @param src_cloud 上一帧点云指针
     * @param tgt_cloud 当前帧点云指针
     * @return Sophus::SO3d 相对旋转
     */
    Sophus::SO3d estimateRotation(const pcl::PointCloud<PointXYZT>::Ptr& src_cloud, 
                                  const pcl::PointCloud<PointXYZT>::Ptr& tgt_cloud) {
        cv::Mat img_src = projectToRangeImage(src_cloud);
        cv::Mat img_tgt = projectToRangeImage(tgt_cloud);

        return estimateRotationFromImages(img_src, img_tgt);
    }

    // 核心解算函数保持不变
    Sophus::SO3d estimateRotationFromImages(const cv::Mat& img_src, const cv::Mat& img_tgt) {
        cv::Point2d shift = cv::phaseCorrelate(img_src, img_tgt, hanning_window_);

        double delta_yaw = -(shift.x / cfg_.width) * 2.0 * M_PI; 
        double delta_pitch = (shift.y / cfg_.height) * fov_v_rad_;
        double delta_roll = 0.0;

        Eigen::AngleAxisd yawAngle(delta_yaw, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd pitchAngle(delta_pitch, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rollAngle(delta_roll, Eigen::Vector3d::UnitX());

        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
        return Sophus::SO3d(q);
    }

private:
    Config cfg_;
    double fov_up_rad_;
    double fov_down_rad_;
    double fov_v_rad_;
    cv::Mat hanning_window_;
};
#endif