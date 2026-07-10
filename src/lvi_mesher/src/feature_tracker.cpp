#include "feature_tracker.h"
#include <iostream>
#include <random>
#include <algorithm>

FeatureTracker::FeatureTracker(int gridRows, int gridCols, int maxPointsPerGrid,
                             double qualityLevel, double minDistance, int detectInterval)
    : gridRows_(gridRows),
      gridCols_(gridCols),
      maxPointsPerGrid_(maxPointsPerGrid),
      qualityLevel_(qualityLevel),
      minDistance_(minDistance),
      detectInterval_(detectInterval),
      frameCounter_(0),
      nextPointId_(0),
      winSize_(31, 31),
      termCrit_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03),
      subPixWinSize_(10, 10) {
    
    // 初始化网格计数
    gridPointCounts_.resize(gridRows_, std::vector<int>(gridCols_, 0));
}

void FeatureTracker::setMask(const std::string& maskPath) {
    mask_ = cv::imread(maskPath, cv::IMREAD_GRAYSCALE);
    if (mask_.empty()) {
        std::cerr << "Warning: Failed to load mask from " << maskPath << std::endl;
    } else {
        // 确保 mask 是严格的二值图：大于128的为255(有效)，否则为0(剔除)
        cv::threshold(mask_, mask_, 128, 255, cv::THRESH_BINARY);
    }
}

void FeatureTracker::setMaskMat(const cv::Mat& mask) {
    if (!mask.empty()) {
        if (mask.channels() > 1) {
            cv::cvtColor(mask, mask_, cv::COLOR_BGR2GRAY);
        } else {
            mask_ = mask.clone();
        }
        cv::threshold(mask_, mask_, 128, 255, cv::THRESH_BINARY);
    }
}

void FeatureTracker::processFrame(const cv::Mat& frame, const std::vector<cv::Point2f>& pred_curr_pts) {
    if (frame.empty()) {
        return;
    }
    
    if (frameSize_.empty()) {
        frameSize_ = frame.size();
        gridCellWidth_ = static_cast<double>(frameSize_.width) / gridCols_;
        gridCellHeight_ = static_cast<double>(frameSize_.height) / gridRows_;
    }

    if (!mask_.empty() && mask_.size() != frame.size()) {
        cv::resize(mask_, mask_, frame.size(), 0, 0, cv::INTER_NEAREST);
    }
    
    cv::Mat frameGray;
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);
    
    if (EQUALIZE_){
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(frameGray, frameGray);
    }
    
    if (prevGray_.empty()) {
        // 第一帧，只检测特征点
        frameGray.copyTo(prevGray_);
        detectNewFeatures(frameGray);
        return;
    }
    
    // 更新现有特征点
    updateExistingFeatures(prevGray_, frameGray, pred_curr_pts);
    
    // 更新网格计数
    updateGridCounts();
    
    // 定期检测新特征点
    frameCounter_++;
    // if (frameCounter_ % detectInterval_ == 0) {
        
    //     frameCounter_ = 0;
    // }

    detectNewFeatures(frameGray);
    
    // 更新前一帧图像
    frameGray.copyTo(prevGray_);
}

void FeatureTracker::detectNewFeatures(const cv::Mat& frameGray) {
    // 首先检查哪些网格需要新点
    std::vector<std::pair<int, int>> gridsToFill;
    for (int y = 0; y < gridRows_; ++y) {
        for (int x = 0; x < gridCols_; ++x) {
            if (gridNeedsPoints(x, y)) {
                gridsToFill.emplace_back(x, y);
            }
        }
    }
    
    if (gridsToFill.empty()) {
        return;
    }
    
    // 为每个需要点的网格检测特征点
    for (const auto& grid : gridsToFill) {
        int gridX = grid.first;
        int gridY = grid.second;
        
        // 计算网格区域
        int x1 = static_cast<int>(gridX * gridCellWidth_);
        int y1 = static_cast<int>(gridY * gridCellHeight_);
        int x2 = static_cast<int>((gridX + 1) * gridCellWidth_);
        int y2 = static_cast<int>((gridY + 1) * gridCellHeight_);
        
        cv::Rect gridRect(x1, y1, x2 - x1, y2 - y1);
        
        // 确保区域在图像范围内
        gridRect &= cv::Rect(0, 0, frameGray.cols, frameGray.rows);
        if (gridRect.empty()) {
            continue;
        }
        
        // 提取网格区域
        cv::Mat gridROI = frameGray(gridRect);
        cv::Mat maskROI;
        if (!mask_.empty()) {
            maskROI = mask_(gridRect);
        }
        
        // 检测网格内的特征点，传入 maskROI
        std::vector<cv::Point2f> newPoints;
        cv::goodFeaturesToTrack(gridROI, newPoints, maxPointsPerGrid_ - gridPointCounts_[gridY][gridX],
                               qualityLevel_, minDistance_, maskROI);
        
        // // 检测网格内的特征点
        // std::vector<cv::Point2f> newPoints;
        // cv::goodFeaturesToTrack(gridROI, newPoints, maxPointsPerGrid_ - gridPointCounts_[gridY][gridX],
        //                        qualityLevel_, minDistance_);
        
        // 转换坐标到全局
        for (auto& point : newPoints) {
            point.x += x1;
            point.y += y1;
        }
        
        // 亚像素精确化
        if (!newPoints.empty()) {
            cv::cornerSubPix(frameGray, newPoints, subPixWinSize_, cv::Size(-1, -1), termCrit_);
        }
        
        // 添加新追踪点
        for (const auto& point : newPoints) {
            cv::Scalar color = generateRandomColor();
            trackedPoints_.emplace_back(point, nextPointId_++, color, gridX, gridY);
            prevPoints_.push_back(point);
            
            // 初始化轨迹
            pointTrajectories_[trackedPoints_.back().id].push_back(point);
            
            // 更新网格计数
            gridPointCounts_[gridY][gridX]++;
            
            // 如果达到最大点数，跳出循环
            if (gridPointCounts_[gridY][gridX] >= maxPointsPerGrid_) {
                break;
            }
        }
    }
}

void FeatureTracker::updateExistingFeatures(const cv::Mat& prevGray, const cv::Mat& currGray, const std::vector<cv::Point2f>& pred_curr_pts) {
    if (prevPoints_.empty()) {
        return;
    }
    
    std::vector<uchar> status;
    std::vector<float> err;
    currPoints_.clear();
    
    // 计算光流
    if(pred_curr_pts.empty()){
        cv::calcOpticalFlowPyrLK(prevGray, currGray, prevPoints_, currPoints_, 
                            status, err, winSize_, 3, termCrit_, 0, 0.001);
    }else{
        assert(pred_curr_pts.size() == prevPoints_.size());
        currPoints_ = pred_curr_pts;
        cv::calcOpticalFlowPyrLK(prevGray, currGray, prevPoints_, currPoints_, 
                            status, err, winSize_, 3, termCrit_, cv::OPTFLOW_USE_INITIAL_FLOW, 0.001);
    }
    
    
    // // 更新追踪点和统计信息
    // std::vector<TrackedPoint> updatedPoints;
    // std::vector<cv::Point2f> updatedPrevPoints;
    
    // for (size_t i = 0; i < prevPoints_.size(); ++i) {
    //     if (status[i] && err[i] < 25.0f) {  // 成功追踪且误差较小，err为像素平方
    //         // 找到对应的TrackedPoint
    //         for (auto& trackedPoint : trackedPoints_) {
    //             if (cv::norm(trackedPoint.position - prevPoints_[i]) < 1.0) {
    //                 trackedPoint.position = currPoints_[i];
    //                 trackedPoint.trackCount++;
                    
    //                 // 更新网格坐标
    //                 getGridCoordinates(currPoints_[i], trackedPoint.gridX, trackedPoint.gridY);
                    
    //                 updatedPoints.push_back(trackedPoint);
    //                 updatedPrevPoints.push_back(currPoints_[i]);
                    
    //                 // 记录轨迹
    //                 pointTrajectories_[trackedPoint.id].push_back(currPoints_[i]);
    //                 break;
    //             }
    //         }
    //     }
    // }
    
    // // 更新数据
    // trackedPoints_ = std::move(updatedPoints);
    // prevPoints_ = std::move(updatedPrevPoints);


    // ======================================
    // 新增：本质矩阵外点过滤（核心逻辑）
    // ======================================
    std::vector<cv::Point2f> validPrevPts, validCurrPts;
    std::vector<int> validPointIds;  // 存储有效特征点的ID
    std::vector<size_t> validOrigIndices;  // 存储原始索引（用于后续匹配）

    // 步骤1：先筛选光流追踪成功的点（基础过滤）
    for (size_t i = 0; i < prevPoints_.size(); ++i) {
        if (status[i] && err[i] < 25.0f) {  // 光流成功且误差较小
            // 找到对应的TrackedPoint，记录ID和坐标
            // for (size_t j = 0; j < trackedPoints_.size(); ++j) {
            //     if (cv::norm(trackedPoints_[j].position - prevPoints_[i]) < 1.0) {
            //         validPrevPts.push_back(prevPoints_[i]);
            //         validCurrPts.push_back(currPoints_[i]);
            //         validPointIds.push_back(trackedPoints_[j].id);
            //         validOrigIndices.push_back(j);
            //         break;
            //     }
            // }
            bool in_valid_mask = true;
            if (!mask_.empty()) {
                int px = cvRound(currPoints_[i].x);
                int py = cvRound(currPoints_[i].y);
                // 检查边界
                if (px >= 0 && px < mask_.cols && py >= 0 && py < mask_.rows) {
                    if (mask_.at<uchar>(py, px) == 0) { // 黑色区域剔除
                        in_valid_mask = false;
                    }
                } else {
                    in_valid_mask = false; // 越界也剔除
                }
            }

            if (in_valid_mask) {
                // 找到对应的TrackedPoint，记录ID和坐标
                for (size_t j = 0; j < trackedPoints_.size(); ++j) {
                    if (cv::norm(trackedPoints_[j].position - prevPoints_[i]) < 1.0) {
                        validPrevPts.push_back(prevPoints_[i]);
                        validCurrPts.push_back(currPoints_[i]);
                        validPointIds.push_back(trackedPoints_[j].id);
                        validOrigIndices.push_back(j);
                        break;
                    }
                }
            }
        }
    }

    // 步骤2：像素坐标→归一化坐标（用于本质矩阵计算）
    std::vector<cv::Point2f> prevNormPts = pixelToNormalized(validPrevPts);
    std::vector<cv::Point2f> currNormPts = pixelToNormalized(validCurrPts);

    // 步骤3：RANSAC过滤外点
    std::vector<bool> isInlier;
    if (!prevNormPts.empty() && !currNormPts.empty()) {
        filterOutliersByEssentialMatrix(prevNormPts, currNormPts, validPointIds, isInlier);
    }

    // ======================================
    // 原有逻辑修改：只保留RANSAC内点
    // ======================================
    std::vector<TrackedPoint> updatedPoints;
    std::vector<cv::Point2f> updatedPrevPoints;
    std::vector<cv::Point2f> curUvInliers;
    pointIdToIndex_.clear();  // 重置ID-索引映射

    // === 新增：清理用于位姿计算的容器 ===
    latest_prev_norm_pts_.clear();
    latest_curr_norm_pts_.clear();

    // 遍历RANSAC筛选后的内点
    for (size_t i = 0; i < validPointIds.size(); ++i) {
        if (isInlier[i]) {  // 仅保留内点
            size_t origIdx = validOrigIndices[i];
            TrackedPoint& trackedPoint = trackedPoints_[origIdx];
            
            // 更新特征点位置和追踪计数
            trackedPoint.position = validCurrPts[i];
            trackedPoint.trackCount++;
            
            // 更新网格坐标
            getGridCoordinates(validCurrPts[i], trackedPoint.gridX, trackedPoint.gridY);
            
            // 记录轨迹
            pointTrajectories_[trackedPoint.id].push_back(validCurrPts[i]);

            // 保存更新后的数据
            updatedPoints.push_back(trackedPoint);
            curUvInliers.push_back(trackedPoint.position);
            updatedPrevPoints.push_back(validCurrPts[i]);
            pointIdToIndex_[trackedPoint.id] = updatedPoints.size() - 1;

            // === 新增：保存用于本质矩阵分解的归一化内点 ===
            latest_prev_norm_pts_.push_back(prevNormPts[i]);
            latest_curr_norm_pts_.push_back(currNormPts[i]);
        }
    }
    // 步骤3：RANSAC过滤外点
    // std::vector<bool> isInlier;
    // if (!prevNormPts.empty() && !currNormPts.empty()) {
    //     filterOutliersByEssentialMatrix(prevNormPts, currNormPts, validPointIds, isInlier);
    // }

    // ======================================
    // 原有逻辑修改：只保留RANSAC内点
    // ======================================
    /* std::vector<TrackedPoint> updatedPoints;
    std::vector<cv::Point2f> updatedPrevPoints;
    std::vector<cv::Point2f> curUvInliers;
    pointIdToIndex_.clear();  // 重置ID-索引映射

    // 遍历RANSAC筛选后的内点
    for (size_t i = 0; i < validPointIds.size(); ++i) {
        if (isInlier[i]) {  // 仅保留内点
            size_t origIdx = validOrigIndices[i];
            TrackedPoint& trackedPoint = trackedPoints_[origIdx];
            
            // 更新特征点位置和追踪计数
            trackedPoint.position = validCurrPts[i];
            trackedPoint.trackCount++;
            
            // 更新网格坐标
            getGridCoordinates(validCurrPts[i], trackedPoint.gridX, trackedPoint.gridY);
            
            // 记录轨迹
            pointTrajectories_[trackedPoint.id].push_back(validCurrPts[i]);

            // 保存更新后的数据
            updatedPoints.push_back(trackedPoint);
            curUvInliers.push_back(trackedPoint.position);
            updatedPrevPoints.push_back(validCurrPts[i]);
            pointIdToIndex_[trackedPoint.id] = updatedPoints.size() - 1;
        }
    } */

    // 更新数据（只包含内点）
    trackedPoints_ = std::move(updatedPoints);
    prevPoints_ = std::move(updatedPrevPoints);
    prevNormPoints_ = std::move(currNormPts);  // 保存当前帧归一化坐标，供下一帧使用

    // 计算ORB描述子
    std::vector<cv::KeyPoint> keypoints;
    std::vector<BRIEF::bitset> descriptors;
    // 返回的descriptor有可能为空
    orb_converter_.computeORBFromTrackedPoints(currGray, curUvInliers, keypoints, descriptors, false);
    for(int i = 0; i<keypoints.size(); i++){
        trackedPoints_[i].keypoint = keypoints[i];
        trackedPoints_[i].descriptor = descriptors[i];
    }
}

const std::vector<TrackedPoint>& FeatureTracker::getTrackedPoints() const {
    return trackedPoints_;
}

void FeatureTracker::getGridStats(std::vector<std::vector<int>>& gridCounts) const {
    gridCounts = gridPointCounts_;
}

void FeatureTracker::getTrackingStats(int& totalPoints, int& avgTrackCount, int& coveredGrids) const {
    totalPoints = trackedPoints_.size();
    coveredGrids = 0;
    
    // 计算被覆盖的网格数量
    for (const auto& row : gridPointCounts_) {
        for (int count : row) {
            if (count > 0) {
                coveredGrids++;
            }
        }
    }
    
    if (trackedPoints_.empty()) {
        avgTrackCount = 0;
        return;
    }
    
    int sum = 0;
    for (const auto& point : trackedPoints_) {
        sum += point.trackCount;
    }
    avgTrackCount = sum / totalPoints;
}

void FeatureTracker::drawTrackingResults(cv::Mat& outputImage, bool drawTrails, bool drawGrid) const {
    if (outputImage.empty()) {
        return;
    }
    
    // 绘制网格
    if (drawGrid) {
        for (int i = 1; i < gridRows_; ++i) {
            int y = static_cast<int>(i * gridCellHeight_);
            cv::line(outputImage, cv::Point(0, y), cv::Point(outputImage.cols, y), 
                    cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
        }
        for (int i = 1; i < gridCols_; ++i) {
            int x = static_cast<int>(i * gridCellWidth_);
            cv::line(outputImage, cv::Point(x, 0), cv::Point(x, outputImage.rows), 
                    cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
        }
    }
    
    // 绘制当前特征点
    for (const auto& point : trackedPoints_) {
        cv::circle(outputImage, point.position, 5, point.color, -1);
        cv::circle(outputImage, point.position, 6, cv::Scalar(255, 255, 255), 1);
        
        // 显示ID和追踪次数
        std::string text = std::to_string(point.id);
        cv::putText(outputImage, text, 
                   cv::Point(point.position.x + 8, point.position.y - 8),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }
    
    // 绘制轨迹
    if (drawTrails) {
        for (const auto& point : trackedPoints_) {
            const auto& trajectory = pointTrajectories_.at(point.id);
            if (trajectory.size() > 1) {
                for (size_t i = 1; i < trajectory.size(); ++i) {
                    cv::line(outputImage, trajectory[i-1], trajectory[i], 
                            point.color, 2, cv::LINE_AA);
                }
            }
        }
    }
    
    // 绘制网格统计信息
    if (drawGrid) {
        for (int y = 0; y < gridRows_; ++y) {
            for (int x = 0; x < gridCols_; ++x) {
                int cellX = static_cast<int>(x * gridCellWidth_ + 5);
                int cellY = static_cast<int>(y * gridCellHeight_ + 20);
                std::string countText = std::to_string(gridPointCounts_[y][x]);
                cv::putText(outputImage, countText, cv::Point(cellX, cellY),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
            }
        }
    }
}

void FeatureTracker::reset() {
    trackedPoints_.clear();
    prevPoints_.clear();
    currPoints_.clear();
    prevGray_ = cv::Mat();
    frameSize_ = cv::Size();
    pointTrajectories_.clear();
    frameCounter_ = 0;
    nextPointId_ = 0;
    
    // 重置网格计数
    for (auto& row : gridPointCounts_) {
        std::fill(row.begin(), row.end(), 0);
    }
}

void FeatureTracker::setGridParameters(int gridRows, int gridCols, int maxPointsPerGrid) {
    gridRows_ = gridRows;
    gridCols_ = gridCols;
    maxPointsPerGrid_ = maxPointsPerGrid;
    
    // 重新初始化网格计数
    gridPointCounts_.resize(gridRows_, std::vector<int>(gridCols_, 0));
    for (auto& row : gridPointCounts_) {
        std::fill(row.begin(), row.end(), 0);
    }
}

void FeatureTracker::getGridCoordinates(const cv::Point2f& point, int& gridX, int& gridY) const {
    if (frameSize_.empty()) {
        gridX = gridY = -1;
        return;
    }
    
    gridX = static_cast<int>(point.x / gridCellWidth_);
    gridY = static_cast<int>(point.y / gridCellHeight_);
    
    // 确保在网格范围内
    gridX = std::clamp(gridX, 0, gridCols_ - 1);
    gridY = std::clamp(gridY, 0, gridRows_ - 1);
}

bool FeatureTracker::gridNeedsPoints(int gridX, int gridY) const {
    return gridPointCounts_[gridY][gridX] < maxPointsPerGrid_;
}

cv::Scalar FeatureTracker::generateRandomColor() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> dis(50, 255);
    
    return cv::Scalar(dis(gen), dis(gen), dis(gen));
}

void FeatureTracker::updateGridCounts() {
    // 重置网格计数
    for (auto& row : gridPointCounts_) {
        std::fill(row.begin(), row.end(), 0);
    }
    
    // 重新计算每个网格的点数
    for (const auto& point : trackedPoints_) {
        int gridX, gridY;
        getGridCoordinates(point.position, gridX, gridY);
        if (gridX >= 0 && gridY >= 0) {
            gridPointCounts_[gridY][gridX]++;
        }
    }
}

bool FeatureTracker::getRelativePose(cv::Mat& R, cv::Mat& t) const {
    // 确保有足够的点进行本质矩阵计算 (至少8个点)
    if (latest_prev_norm_pts_.size() < 8 || latest_curr_norm_pts_.size() < 8) {
        LOG(WARNING) << "Not enough points to compute relative pose.";
        return false;
    }

    // 重新计算本质矩阵 (此时输入的全是上一轮 filterOutliersByEssentialMatrix 确认的内点)
    // 使用单位矩阵作为相机内参，因为输入已经是归一化坐标了
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(latest_prev_norm_pts_, latest_curr_norm_pts_, 
                                     cv::Mat::eye(3, 3, CV_64F), 
                                     cv::RANSAC, 0.999, 1.0 / 460.0, mask); // threshold 可以给一个小值，因为已经是内点了

    if (E.empty() || E.rows != 3 || E.cols != 3) {
        LOG(WARNING) << "Essential matrix computation failed.";
        return false;
    }

    // 从本质矩阵恢复位姿 R 和 t
    // 注意：recoverPose 会输出满足大多数点为正深度的 R 和 t
    int inliers_count = cv::recoverPose(E, latest_prev_norm_pts_, latest_curr_norm_pts_, 
                                        cv::Mat::eye(3, 3, CV_64F), R, t, mask);

    // 如果满足正深度的点太少，可能结果不可靠
    if (inliers_count < 5) {
        // LOG(WARNING) << "RecoverPose failed, too few valid points: " << inliers_count;
        return false;
    }

    return true;
}
