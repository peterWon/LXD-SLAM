#ifndef __LXD_SLAM_BASE_NODE_H__
#define __LXD_SLAM_BASE_NODE_H__

#include "glog/logging.h"
#include "ros/ros.h"
#include "utility/parameter.h"
#include "utility/imu.h"
#include "liomesher.h"
#include "posegraph.h"
#include <condition_variable>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/JointState.h"
#include "sensor_msgs/NavSatFix.h"
#include "opencv4/opencv2/opencv.hpp"
#include "camera_models/Camera.h"
#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"
#include <cv_bridge/cv_bridge.h>
#include <livox_ros_driver/CustomMsg.h>
// #include "mono_tracker.h"
#include <tf2_ros/transform_broadcaster.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>

using namespace camodocal;

class BaseNode{
    struct SyncResult {
        std::shared_ptr<ImagePkg> imgpkg;          // 图像数据
        pcl::PointCloud<PointXYZT>::Ptr scan_msg;  // LiDAR点云
        sensor_msgs::NavSatFixPtr gnss;            // GNSS定位数据，暂不支持RTK
        std::vector<sensor_msgs::ImuPtr> imus;     // IMU数据
        std::vector<sad::WheelEncoder> wheels;     // 轮速计数据
    };
public:
    BaseNode();
    ~BaseNode();

    void imuCallback(const sensor_msgs::ImuPtr & imu_msg);
    void wheelEncoderCallback(const sensor_msgs::JointStatePtr & wheel_msg);//fusionportableV2
    void pointCloudCallback(const sensor_msgs::PointCloud2Ptr & pcl_msg);
    void pointCloudCallback2(const sensor_msgs::PointCloud2Ptr & pcl_msg);
    void livoxCallback(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void livoxCallback2(const livox_ros_driver::CustomMsg::ConstPtr &msg);
    void imgLeftCallback(const sensor_msgs::ImagePtr & img_msg);
    void imgRightCallback(const sensor_msgs::ImagePtr & img_msg);
    void gnssCallback(const sensor_msgs::NavSatFixPtr & gnss_msg);
protected:
    bool initialize(const std::vector<sensor_msgs::ImuPtr>& imus);
    bool initialize_without_imu(double ts);
    bool syncAllData(SyncResult& sync_result);

    sad::NaiveGNSS NavSatFix2NaiveGNSS(const sensor_msgs::NavSatFixPtr input){
        sad::NaiveGNSS res;
        res.timestamp_ = input->header.stamp.toSec();
        res.latitude = input->latitude;
        res.longitude = input->longitude;
        res.altitude = input->altitude;
        for(int r = 0; r < 3; r++){
            for(int c = 0; c < 3; c++){
                res.position_covariance(r, c) = input->position_covariance[r*3+c];
            }
        }
        return res;
    }
    
    // 辅助函数：清理LiDAR缓存（提取重复逻辑）
    void cleanLidarBuffer(std::vector<pcl::PointCloud<PointXYZT>::Ptr>& lidar_buf, double time_threshold) {
        for(auto it = lidar_buf.begin(); it != lidar_buf.end();) {
            if((*it)->points.empty()) {
                LOG(WARNING) << "LiDAR scan is empty, should not happen!";
                it = lidar_buf.erase(it);
            } else {
                if((*it)->points.front().t > time_threshold) break;
                if((*it)->points.back().t > time_threshold) break;
                if((*it)->points.back().t < time_threshold) {
                    it = lidar_buf.erase(it);
                } else {
                    it++;
                }
            }
        }
    }

    // 辅助函数：同步多LiDAR数据（提取重复逻辑）
    void syncMultiLidarData(pcl::PointCloud<PointXYZT>::Ptr& scan_msg, 
                                    double start_ts, double end_ts) {
        // 处理主LiDAR
        scan_msg = pcl::PointCloud<PointXYZT>::Ptr(new pcl::PointCloud<PointXYZT>);
        for(int i = 0; i < scan1_buf_.size(); ++i) {
            for(int j = 0; j < scan1_buf_[i]->points.size(); ++j) {
                PointXYZT xyzt = scan1_buf_[i]->points[j];
                if(xyzt.t < end_ts && xyzt.t > start_ts) {
                    scan_msg->points.emplace_back(xyzt);
                }
                if(xyzt.t > end_ts) {
                    break;
                }
            }
        }
        
        // 处理第二个LiDAR（如果启用）
        if(param_.enable_lidar2) {
            for(int i = 0; i < scan2_buf_.size(); ++i) {
                for(int j = 0; j < scan2_buf_[i]->points.size(); ++j) {
                    PointXYZT xyzt = scan2_buf_[i]->points[j];
                    if(xyzt.t < end_ts && xyzt.t > start_ts) {
                        scan_msg->points.emplace_back(xyzt);
                    }
                    if(xyzt.t > end_ts) {
                        break;
                    }
                }
            }
        }
    }

     inline Vec3d transformToX(const pcl::PointXYZ& pt_imu, const Mat4d& T_I2X){
        Vec3d p_i = {pt_imu.x, pt_imu.y, pt_imu.z};
        Vec3d p_x = T_I2X.block<3,3>(0,0) * p_i + T_I2X.block<3,1>(0,3);
        return p_x;
    }

    

    cv::Mat ImgFromRos(const sensor_msgs::ImagePtr& img_msg){
        cv::Mat img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
        return img;
    }
    
    sensor_msgs::NavSatFixPtr getClosestAndPop(double t) {
        // std::lock_guard<std::mutex> lock(gnss_buf_mutex_);

        if (gnss_msg_buf_.empty()) {
            return nullptr;
        }

        while (gnss_msg_buf_.size() >= 2) {
            double t_front = gnss_msg_buf_.front()->header.stamp.toSec();
            if (t_front < t - param_.gnss_time_tolerance) {
                gnss_msg_buf_.pop();
            } else {
                break;
            }
        }
        return gnss_msg_buf_.front();
    }

    void process_loop();

    std::unique_ptr<LioMesher> liomesher_;
    std::unique_ptr<PoseGraph> posegraph_;

    std::thread liomehser_thread_;

    std::vector<std::shared_ptr<Submap>> submaps_;
    std::vector<Pose6D> keyframe_poses_opt_;
    
    Mat4d T_body_lidar1_;
    Mat4d T_body_lidar2_;
    Mat4d T_body_wheel_;
    Mat4d T_body_gnss_;
    int init_cached_frames_ = 0;

    std::mutex mtx_buf_;
    std::mutex mtx_lio_;
    std::condition_variable convar_;
    int raw_left_img_idx_ = 0;
    int raw_right_img_idx_ = 0;

    double last_imu_t_ = -1.;
    double last_wheel_t_ = -1.;
    double last_lidar1_t_ = -1.;
    double last_img_left_t_ = -1.;
    double last_img_right_t_ = -1.;
    double last_img_t_ = -1.;
    double last_scan_t_ = -1.;
    
    std::queue<sensor_msgs::ImuPtr> imu_msg_buf_;
    std::queue<sad::WheelEncoder> wheel_msg_buf_;
    std::queue<sensor_msgs::ImagePtr> img_left_buf_;
    std::queue<sensor_msgs::ImagePtr> img_right_buf_;
    std::queue<sensor_msgs::NavSatFixPtr> gnss_msg_buf_;

    std::vector<pcl::PointCloud<PointXYZT>::Ptr> scan1_buf_;
    std::vector<pcl::PointCloud<PointXYZT>::Ptr> scan2_buf_;

    std::vector<sensor_msgs::ImuConstPtr> imus_for_initialization_;

    Parameter& param_=Parameter::GetInstance();
    double range_min_sq_ = 1.;
    double range_max_sq_ = 10000.;
    bool initialized_ = false;

    bool finish_add_data_ = false;

    // for fusionportableV2 wheel encoder
    bool first_call_ = true;
    double distance_x_prev_ = 0.;
    double distance_y_prev_ = 0.;

    const double wheel_track_ = 0.76;
    const double wheel_radius_ = 0.22;
    const double wheel_radius_r_ = 0.22;
    const double wheel_radius_l_ = 0.22;
    // 溢出处理阈值 (参考原文件 1024*1024)
    const double OVERFLOW_LIMIT = 1048576.0;
    const double OVERFLOW_CORRECTION = 4294967.296;

    std::shared_ptr<TrackingResult> result_ = nullptr;
    std::mutex mtx_result_;
    std::mutex mtx_flag_;

    double t_cost_total_ = 0.;

    // bool gnss_inited_ = false;
    // GeographicLib::LocalCartesian geo_converter_;
    // std::vector<Vec3d> gnss_positions_;
};
#endif
