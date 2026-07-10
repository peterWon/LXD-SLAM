#ifndef __LK_TO_ORB_CONVERTER_H__
#define __LK_TO_ORB_CONVERTER_H__
#include <opencv2/opencv.hpp>
#include <vector>

#include "DVision/DVision.h"
#include "DVision/BRIEF.h"
#include "utility/parameter.h"
#include "glog/logging.h"

using namespace DVision;
using namespace std;
class BriefExtractor{
public:
//   void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const{
//     m_brief.compute(im, keys, descriptors);
//   };
  BriefExtractor(const std::string &pattern_file){
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;
    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;
    m_brief.importPairs(x1, y1, x2, y2);
  }

  DVision::BRIEF m_brief;
};

class LKToORBConverter {
    public:
    LKToORBConverter(int nfeatures = 1000, 
                    float scaleFactor = 1.2f, 
                    int nlevels = 8, 
                    int edgeThreshold = 31,
                    int firstLevel = 0, 
                    int WTA_K = 2, 
                    int scoreType = cv::ORB::HARRIS_SCORE,
                    int patchSize = 31, 
                    int fastThreshold = 20) {        
        orb_extractor_ = cv::ORB::create(1000,        // 最大特征点数
            1.2f,        // 尺度金字塔参数
            8,           // 金字塔层数
            31,          // 边缘阈值
            0,           // 第一层金字塔级别
            2,           // WTA_K
            cv::ORB::FAST_SCORE,  // 评分类型
            31,          // 补丁大小
            20           // 快速阈值
        );
        
        int descriptorSize = orb_extractor_->descriptorSize(); 
        // std::cout<<"描述子字节数： "<<descriptorSize<<std::endl;

        brief_extractor_.reset(new BriefExtractor(Parameter::GetInstance().brief_pattern_file_path));
    }
    
    // 完整的转换流程
    bool computeORBFromTrackedPoints(
        const cv::Mat& current_frame,
        const std::vector<cv::Point2f>& tracked_points,
        std::vector<cv::KeyPoint>& output_keypoints,
        // cv::Mat& output_descriptors,
        std::vector<BRIEF::bitset>& brief_descriptors,
        bool compute_orientation = false) {
        
        if (current_frame.empty() || tracked_points.empty()) {
            return false;
        }
        
        // 步骤1: 将跟踪点转换为KeyPoint
        output_keypoints = convertPointsToKeyPoints(tracked_points);
        
        // 步骤2: 如果需要，计算关键点方向
        if (compute_orientation) {
            computeKeypointOrientations(current_frame, output_keypoints);
        }
        // 步骤3: 计算ORB描述子[citation:2]
        // orb_extractor_->compute(current_frame, output_keypoints, output_descriptors);
        brief_extractor_->m_brief.compute(current_frame, output_keypoints, brief_descriptors);
       
        return !brief_descriptors.empty();
    }
private:
    cv::Ptr<cv::ORB> orb_extractor_;
    std::shared_ptr<BriefExtractor> brief_extractor_;    
    
    std::vector<cv::KeyPoint> convertPointsToKeyPoints(
        const std::vector<cv::Point2f>& tracked_points,
        float size = 31.0f, 
        float angle = -1.0f, 
        float response = 0.0f, 
        int octave = 0) {
        
        std::vector<cv::KeyPoint> keypoints;
        keypoints.reserve(tracked_points.size());
        
        for (const auto& point : tracked_points) {
            // 创建KeyPoint对象
            cv::KeyPoint kp;
            kp.pt = point;          // 位置
            kp.size = size;         // 特征点邻域直径[citation:2]
            kp.angle = angle;       // 方向角度，-1表示未知
            kp.response = response; // 响应强度
            kp.octave = octave;     // 金字塔层级[citation:2]
            
            keypoints.push_back(kp);
        }
        
        return keypoints;
    }

    
    // 计算关键点方向
    void computeKeypointOrientations(const cv::Mat& image,
                                   std::vector<cv::KeyPoint>& keypoints) {
        
        if (image.empty() || keypoints.empty()) return;
        
        cv::Mat gray_image;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        } else {
            gray_image = image;
        }
        
        // 使用图像矩计算方向[citation:2]
        for (auto& kp : keypoints) {
            if (kp.angle < 0) {  // 只计算未设置方向的关键点
                const int patch_size = static_cast<int>(kp.size);
                const int half_size = patch_size / 2;
                const cv::Point2f& pt = kp.pt;
                
                // 检查边界
                if (pt.x - half_size < 0 || pt.x + half_size >= gray_image.cols ||
                    pt.y - half_size < 0 || pt.y + half_size >= gray_image.rows) {
                    kp.angle = 0.0f;  // 边界点设为默认方向
                    continue;
                }
                
                // 提取特征点邻域
                cv::Rect patch_rect(pt.x - half_size, pt.y - half_size, patch_size, patch_size);
                cv::Mat patch = gray_image(patch_rect);
                
                // 计算图像矩
                cv::Moments m = cv::moments(patch, true);
                if (m.m00 == 0) {
                    kp.angle = 0.0f;
                    continue;
                }
                
                // 计算方向角度
                float centroid_x = static_cast<float>(m.m10 / m.m00);
                float centroid_y = static_cast<float>(m.m01 / m.m00);
                kp.angle = static_cast<float>(std::atan2(centroid_y - half_size, 
                                                        centroid_x - half_size) * 180.0 / CV_PI);
                if (kp.angle < 0) kp.angle += 360.0f;
            }
        }
    }
};
#endif