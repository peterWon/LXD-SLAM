#ifndef FEATURETRACKER_H
#define FEATURETRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "camera_models/Camera.h"
#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"

#include "LKtoORBConverter.h"

struct TrackedPoint {
    uint64_t id;
    cv::Point2f position;
    int trackCount;
    cv::Scalar color;
    int gridX;  // 网格X坐标
    int gridY;  // 网格Y坐标

    // computed from LKtoORBConverter
    cv::KeyPoint keypoint;
    BRIEF::bitset descriptor;
    
    TrackedPoint(cv::Point2f pos, int pointId, cv::Scalar pointColor, int gridX, int gridY)
        : position(pos), id(pointId), trackCount(1), color(pointColor), 
          gridX(gridX), gridY(gridY) {}
};

class FeatureTracker {
public:
    FeatureTracker(int gridRows = 8, 
                  int gridCols = 8,
                  int maxPointsPerGrid = 5,
                  double qualityLevel = 0.01, 
                  double minDistance = 20.0,
                  int detectInterval = 15);
    
    ~FeatureTracker() = default;

    void setCamera(camodocal::CameraPtr cam){
        camera_ = cam;
    }

    // 新增：设置Mask文件路径 (黑色区域为剔除区域，白色区域为有效区域)
    void setMask(const std::string& maskPath);    
    void setMaskMat(const cv::Mat& mask);
    
    // 处理新帧
    void processFrame(const cv::Mat& frame, const std::vector<cv::Point2f>& pred_curr_pts={});
    
    // 获取当前追踪的点
    const std::vector<TrackedPoint>& getTrackedPoints() const;

    std::vector<cv::Point2f> getTrackedPointsCV() {
        return currPoints_;
    };
    
    // 获取网格统计信息
    void getGridStats(std::vector<std::vector<int>>& gridCounts) const;
    
    // 获取追踪统计信息
    void getTrackingStats(int& totalPoints, int& avgTrackCount, int& coveredGrids) const;
    
    // 绘制追踪结果和网格
    void drawTrackingResults(cv::Mat& outputImage, bool drawTrails = true, bool drawGrid = true) const;
    
    // 重置追踪器
    void reset();
    
    // 设置网格参数
    void setGridParameters(int gridRows, int gridCols, int maxPointsPerGrid);

    /**
     * 计算前后帧的相对位姿 (本质矩阵分解)
     * @param R 输出相对旋转矩阵 (3x3)，表示前一帧到当前帧的旋转
     * @param t 输出相对平移向量 (3x1)
     * @return 如果计算成功返回 true，否则返回 false (例如匹配点太少)
     */
    bool getRelativePose(cv::Mat& R, cv::Mat& t) const;
    
private:
    // 检测新特征点（网格化）
    void detectNewFeatures(const cv::Mat& frameGray);
    
    // 更新现有特征点的追踪
    void updateExistingFeatures(const cv::Mat& prevGray, const cv::Mat& currGray, const std::vector<cv::Point2f>& pred_curr_pts);
    
    // 移除丢失的点
    void removeLostPoints();
    
    // 获取点所在的网格坐标
    void getGridCoordinates(const cv::Point2f& point, int& gridX, int& gridY) const;
    
    // 检查网格是否需要新点
    bool gridNeedsPoints(int gridX, int gridY) const;
    
    // 生成随机颜色
    cv::Scalar generateRandomColor() const;
    
    // 更新网格计数
    void updateGridCounts();
    
private:
    // 网格参数
    int gridRows_;
    int gridCols_;
    int maxPointsPerGrid_;
    double gridCellWidth_;
    double gridCellHeight_;
    
    // 网格计数
    std::vector<std::vector<int>> gridPointCounts_;
    
    // 参数
    double qualityLevel_;
    double minDistance_;
    int detectInterval_;
    int frameCounter_;
    
    // 图像数据
    cv::Mat prevGray_;
    cv::Size frameSize_;
    cv::Mat mask_;
    
    // 追踪数据
    std::vector<TrackedPoint> trackedPoints_;
    std::vector<cv::Point2f> prevPoints_;
    std::vector<cv::Point2f> currPoints_;
    
    // ID管理
    int nextPointId_;
    
    // 光流参数
    cv::Size winSize_;
    cv::TermCriteria termCrit_;
    cv::Size subPixWinSize_;
    
    // 轨迹绘制
    std::map<int, std::vector<cv::Point2f>> pointTrajectories_;
    bool EQUALIZE_ = true;
    
    // for feature descriptors computation
    LKToORBConverter orb_converter_;

    // for essential matrix rejection
    camodocal::CameraPtr camera_ = nullptr;
    double ransacReprojThreshold_ = 1.0;
    double ransacConfidence_ = 0.99;
    std::vector<cv::Point2f> prevNormPoints_;
    std::map<int, size_t> pointIdToIndex_;

    // 专门用于保存最近一次成功追踪的前后帧内点(归一化坐标)，供外部获取相对位姿使用
    std::vector<cv::Point2f> latest_prev_norm_pts_;
    std::vector<cv::Point2f> latest_curr_norm_pts_;


    // 新增：基于liftProjective的像素→归一化坐标转换（输出2D归一化点，z=1）
    std::vector<cv::Point2f> pixelToNormalized(const std::vector<cv::Point2f>& pixelPoints) const {
        std::vector<cv::Point2f> normPoints;
        if (pixelPoints.empty()) return normPoints;

        for (const auto& uv : pixelPoints) {
            cv::Point3f xyz_norm;
            // 调用已有函数：像素坐标uv → 归一化相机坐标xyz_norm（x,y,z=1）
            Eigen::Vector2d pixel(uv.x, uv.y);
            Eigen::Vector3d xyz;
            camera_->liftProjective(pixel, xyz);
            
            // 提取x,y作为2D归一化点（本质矩阵计算仅需方向）
            normPoints.emplace_back(xyz[0] / xyz[2], xyz[1] / xyz[2]);
        }
        return normPoints;
    }

    void filterOutliersByEssentialMatrix(const std::vector<cv::Point2f>& prevNormPts,
                                        const std::vector<cv::Point2f>& currNormPts,
                                        const std::vector<int>& pointIds,
                                        std::vector<bool>& isInlier) {
        isInlier.assign(pointIds.size(), false);
        if (prevNormPts.size() < 8 || currNormPts.size() < 8) return;

        cv::Mat E, mask;
        E = cv::findEssentialMat(prevNormPts, currNormPts, 
                                cv::Mat::eye(3, 3, CV_64F),  // 归一化坐标下内参为单位矩阵
                                cv::FM_RANSAC,
                                ransacConfidence_,
                                ransacReprojThreshold_,
                                mask);

        if (E.empty() || mask.empty()) return;
        for (size_t i = 0; i < mask.rows; ++i) {
            if (mask.at<uchar>(i) == 1) {
                isInlier[i] = true;
            }
        }
    }
};

#endif // FEATURETRACKER_H