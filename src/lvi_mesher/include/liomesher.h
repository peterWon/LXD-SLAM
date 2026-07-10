//This file is for the class SLAMesh, it is the main class of the project, and it is the node of ROS.
#ifndef __LIOMESHER_H__
#define __LIOMESHER_H__


#include "cell.h"
#include "submap.h"

#include <pcl/registration/icp.h>
#include "utility/ieskf.hpp"
#include "utility/pointcloud_types.h"
#include "utility/parameter.h"

#include "opencv4/opencv2/opencv.hpp"
#include <opencv4/opencv2/features2d.hpp>

#include "camera_models/Camera.h"
#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"

#include "feature_tracker.h"

#include "vins/estimator.h"
#include "vins/estimator_interface.h"

#include "local_ba.h"
#include "sliding_window_filter.h"

#include "LKtoORBConverter.h"
#include "utility/sensor_manager.h"

#include "utility/pose_curve_predictor.h"

#include "rot_estimator.h"

using namespace camodocal;

struct TrackingResult {
    // for backend usage
    double timestamp;
    int64_t timestamp_nsec;
    size_t frame_id;
    Mat4d local_pose;
    Mat4d local_pose_camera;
    pcl::PointCloud<pcl::PointXYZRGB> deskewed_points;
    std::vector<std::shared_ptr<Submap>> active_submaps;//TODO(wz): protected by const
    
    cv::Mat image_raw;
    std::vector<BRIEF::bitset> descriptors;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<Vec3d> points_3d_cam;//只用了左相机视角内的点
    
    std::shared_ptr<sad::NaiveGNSS> gnss = nullptr;

    // debug and visualization
    std::vector<Vec3d> current_mature_points;
    cv::Mat debug_img;
};

struct ImagePkg{
    double timestamp_sec_left;
    double timestamp_sec_right;
    cv::Mat left_img; 
    cv::Mat right_img;
};


class LioMesher{
    // node of this project
public:
    LioMesher();
    
    void addImu(const sad::IMU& imu_data);

    void addWheelEncoder(const sad::WheelEncoder& wheel_encoder_data);

    void addImage(const std::shared_ptr<ImagePkg>& img);

    std::shared_ptr<TrackingResult> addPointCloud(const pcl::PointCloud<PointXYZT>::Ptr& xyzt_base);
    
    // point cloud are organized into the left image's timestamp reference
    std::shared_ptr<TrackingResult> addImageAndPointCloud(
            const std::shared_ptr<ImagePkg>& img, 
            const pcl::PointCloud<PointXYZT>::Ptr& xyzt_imu);

    bool initialize(const double ts, const Mat3d& rotation, const Vec3d& bg, const Vec3d& ba);
   
    // TODO, should be const
    std::shared_ptr<Submap> get_matching_submap() {
        return active_submaps_->matching_submap();
    }

    Mat4d pose();
private:
    void xyzt2xyz(const pcl::PointCloud<PointXYZT>& xyzt, pcl::PointCloud<pcl::PointXYZ>& xyz);
    inline Vec3d transformToX(const pcl::PointXYZ& pt_imu, const Mat4d& T_I2X){
        Vec3d p_i = {pt_imu.x, pt_imu.y, pt_imu.z};
        Vec3d p_x = T_I2X.block<3,3>(0,0) * p_i + T_I2X.block<3,1>(0,3);
        return p_x;
    }

    inline bool InCameraView(camodocal::CameraPtr camera, const Vec3d& p_cam, Vec2d& uv){
        if(p_cam[2] < 0.) return false;
        camera->spaceToPlane(p_cam, uv);
        const int border = 2;
        return uv(0) > 0 && uv(1) > 0 && uv(0)<camera->imageWidth()-border && uv(1)<camera->imageHeight()-border;
    }

    inline bool InLeftCamera(const Vec3d& p_lc, Vec2d& uv){
        if(p_lc[2] < 0.) return false;
        left_camera_->spaceToPlane(p_lc, uv);
        int border = 2;
        return uv(0) > 0 && uv(1) > 0 && uv(0)<left_camera_->imageWidth()-border && uv(1)<left_camera_->imageHeight()-border;
    }

    inline bool InRightCamera(const Vec3d& p_rc, Vec2d& uv){
        if(p_rc[2] < 0.) return false;
        right_camera_->spaceToPlane(p_rc, uv);
        int border = 2;
        return uv(0) > 0 && uv(1) > 0 && uv(0)<right_camera_->imageWidth()-border && uv(1)<right_camera_->imageHeight()-border;
    }
    
    inline int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b){
        BRIEF::bitset xor_of_bitset = a ^ b;
        int dis = xor_of_bitset.count();
        return dis;
    }

    void ProjectPointCloudToImage(); 

    // from fastlivo2
    Vec3f getInterpolatedPixel(const cv::Mat& img, const Vec2d& pc){
        int width = img.cols;
        const float u_ref = pc[0];
        const float v_ref = pc[1];
        const int u_ref_i = floorf(pc[0]);
        const int v_ref_i = floorf(pc[1]);
        const float subpix_u_ref = (u_ref - u_ref_i);
        const float subpix_v_ref = (v_ref - v_ref_i);
        const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
        const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
        const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
        const float w_ref_br = subpix_u_ref * subpix_v_ref;
        uint8_t *img_ptr = (uint8_t *)img.data + ((v_ref_i)*width + (u_ref_i)) * 3;
        float B = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] + w_ref_br * img_ptr[width * 3 + 0 + 3];
        float G = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] + w_ref_br * img_ptr[width * 3 + 1 + 3];
        float R = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] + w_ref_br * img_ptr[width * 3 + 2 + 3];
        Vec3f pixel(B, G, R);
        return pixel;
    }


    Vec3f getPixel(const cv::Mat& img, const Vec2d& uv){
        int width = img.cols;
        int height = img.rows;
        const int u = floor(uv[0]);
        const int v = floor(uv[1]);
        Vec3b bgr = img.at<Vec3b>(v, u);
        
        Vec3f pixel(bgr[0], bgr[1], bgr[2]);
        return pixel;
    }

    Mat3d pointCovariance(const Vec3d& pb, const float range_err, const float degree_err){
        Mat3d cov;
        float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
        float range_var = range_err * range_err;

        const float deg2rad = M_PI / 180.; 
        Eigen::Matrix2d direction_var;
        direction_var << std::pow(std::sin(deg2rad*degree_err), 2), 0, 0, std::pow(std::sin(deg2rad*degree_err), 2);
        Vec3d direction(pb);
        direction.normalize();
        Eigen::Matrix3d direction_hat;
        direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
        Vec3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
        base_vector1.normalize();
        Vec3d base_vector2 = base_vector1.cross(direction);
        base_vector2.normalize();
        Eigen::Matrix<double, 3, 2> N;
        N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
        Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
        cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
        return cov;
    }
private:
    void computeResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr);
    void computeResidualAndJacobiansXYZRGB(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr);
    void computeResidualAndJacobiansReprojection(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr);
    /* void VisualResidualAndJacobiansHelperTracker(const camodocal::CameraPtr camera, 
                                          const std::shared_ptr<FeatureTracker>& feature_tracker,
                                          SlidingWindowDepthFilter& filter, //TODO
                                          const Mat4d& T_wi, const Mat4d& T_ci,
                                          Eigen::VectorXd& res, Eigen::MatrixXd& H_block, Eigen::MatrixXd& Hf_block);
    void VisualResidualAndJacobiansHelperSubmap(const camodocal::CameraPtr camera, 
                                          const std::shared_ptr<FeatureTracker>& feature_tracker,
                                          const Mat4d& T_wi, const Mat4d& T_ci,
                                          Eigen::VectorXd& res, Eigen::MatrixXd& H_block, Eigen::MatrixXd& Hf_block); */
    void VisualResidualAndJacobiansHelperTracker(const camodocal::CameraPtr camera,
                                const std::shared_ptr<FeatureTracker>& feature_tracker,
                                SlidingWindowDepthFilter& depth_filter, 
                                const Mat4d& T_wi, const Mat4d& T_ci,
                                Mat18d& HTVH, Vec18d& HTVr);

    void VisualResidualAndJacobiansHelperSubmap(const camodocal::CameraPtr camera, 
                                    const std::shared_ptr<FeatureTracker>& feature_tracker,
                                    const Mat4d& T_wi, const Mat4d& T_ci,
                                    Mat18d& HTVH, Vec18d& HTVr);
    void ComputeJointResidualAndJacobians(const SE3& pose, Mat18d& HTVH, Vec18d& HTVr);
    // void AssociateDepthToFeatures(std::shared_ptr<ingvio::MapServer> mapserver, double timestamp);

    void AssignXYZbyRayTracing(const camodocal::CameraPtr& camera, 
                               const std::shared_ptr<FeatureTracker>& tracker, 
                               const Mat4d& T_wc,
                                std::vector<Vec3d>& xyz_world,
                                std::vector<Vec3d>& xyz_submap,
                                std::vector<const Cell*>& hit_cells,
                                std::vector<double>& depth_cur_frame,
                                std::vector<double>& variance);
    void AssignXYZbyRayTracing(const Mat4d& T_wc,
                                const camodocal::CameraPtr& camera, 
                                const std::shared_ptr<Submap>& submap,
                                const std::vector<cv::KeyPoint>& keypoints,
                                std::vector<Vec3d>& xyz_cam,
                                std::vector<double>& variance);
    void AssignXYZbyInverseProjection(const camodocal::CameraPtr& camera, 
                               const std::shared_ptr<FeatureTracker>& tracker, 
                               const Mat4d& T_wc,
                                std::vector<Vec3d>& xyz_world,
                                std::vector<Vec3d>& xyz_submap,
                                std::vector<const Cell*>& hit_cells,
                                std::vector<double>& depth_cur_frame);
    void AssignXYZandUpdateDepth(camodocal::CameraPtr camera, std::shared_ptr<FeatureTracker>& tracker, 
                                SlidingWindowDepthFilter& depth_filter, Mat4d T_wc);
    std::vector<cv::Point2f> PredictUVforLK(camodocal::CameraPtr camera, const std::shared_ptr<FeatureTracker>& tracker, 
                        SlidingWindowDepthFilter& depth_filter, Mat4d T_wc_pred);
    void VisualizeMaturePointsAndReprojectionError(const camodocal::CameraPtr& camera, const cv::Mat& img_src, const Mat4d& T_wc,
            const FeatureTracker& feature_tracker, SlidingWindowDepthFilter& depth_filter, cv::Mat& err_img, std::vector<Vec3d>& mature_pts);
    Mat4d LocalizeCameraByPnP();

    void FindCorrespondenceInVisualMap(const camodocal::CameraPtr camera,
                                    const std::shared_ptr<FeatureTracker>& feature_tracker,
                                    const Mat4d& T_wi, const Mat4d& T_ci,
                                    std::vector<Vec3d>& vmap_points_w,
                                    std::vector<cv::Point2f>& uvs);

    void Orb3dPointsInCurrentView(const camodocal::CameraPtr camera,
                                  const Mat4d& T_wi, const Mat4d& T_ci,
                                  std::vector<Vec3d>& vmap_points_cur,
                                  std::vector<BRIEF::bitset>& descriptors);

    void TrackedOrb3dPoints(const std::shared_ptr<FeatureTracker>& tracker,
                            SlidingWindowDepthFilter& depth_filter,
                            const Mat4d& T_wi, const Mat4d& T_ci,
                            std::vector<Vec3d>& vmap_points_cur,
                            std::vector<cv::KeyPoint>& keypoints,
                            std::vector<BRIEF::bitset>& descriptors);
    
    void VisualPoints3D(const cv::Mat& image, const Mat4d& T_wc,
                        std::vector<cv::KeyPoint>& keypoints,
                        std::vector<BRIEF::bitset>& descriptors,
                        std::vector<Vec3d>& map_points_c);

    SE3 se3_pose(){
        auto imu_state = ieskf_.GetNominalState();
        return SE3(imu_state.R_, imu_state.p_);
    }

    void deskew_to_end();
    void deskew_to_mid();
    void ColorizePointCloud(pcl::PointCloud<pcl::PointXYZ>& input, pcl::PointCloud<pcl::PointXYZRGB>& output);
    
    bool initialized_ = false;
    bool first_scan_processed_ = false;
    sad::IESKFD ieskf_;
    SE3 last_pose_;

    // for pose predict in lidar-only mode
    SE3 T_l_; //pose of t-1
    SE3 T_ll_;//pose of t-2

    pcl::PointCloud<PointXYZT>::Ptr cur_scan_raw_ = nullptr; //transformed to base frame
    pcl::PointCloud<PointXYZT>::Ptr cur_scan_deskewed_ = nullptr; //transformed to base frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_scan_deskewed_xyz_ = nullptr; //transformed to base frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_scan_deskewed_xyz_ds_ = nullptr; //transformed to base frame
    std::vector<Mat3d> cur_scan_deskewed_cov_ds_ = {}; 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_scan_deskewed_xyzrgb_ = nullptr; //transformed to base frame
    
    
    
    size_t cur_frame_id_ = 0;
    size_t cur_submap_id_ = 0;
    
    std::mutex mbuf_;
    double last_imu_timestamp_ = -1.0;

    std::shared_ptr<ActiveSubmaps> active_submaps_ = nullptr;
    Parameter& param_=Parameter::GetInstance();
    SensorExtrinsicsManager& sensors_ = SensorExtrinsicsManager::GetInstance();
public:
    
    ///////////////////////////////////////
    bool _hasImageCome = false;
    bool _hasInitState = false;

    int _max_lm_feats = 0;
    int _max_sw_clones = 35;//TODO(wz): should be a parameter

    Mat4d last_cam_pose_;
    cv::Mat last_img_;

    cv::Mat left_img_gray_;
    cv::Mat right_img_gray_;
    cv::Mat left_img_gray_last_;
    cv::Mat right_img_gray_last_;
    std::vector<cv::KeyPoint> left_keypoints_, left_keypoints_last_;
    cv::Mat left_descriptors_, left_descriptors_last_;
    std::vector<cv::KeyPoint> right_keypoints_, right_keypoints_last_;
    cv::Mat right_descriptors_, right_descriptors_last_;

    // cvins::ParameterServer* ps_ = nullptr;
    // std::shared_ptr<cvins::FeatureManager> feature_manager_vi_ = nullptr;
    // std::shared_ptr<cvins::Estimator> vins_ = nullptr;

    const int left_camera_id_ = 0;

    // FeatureCentricBundleAdjustment local_ba_;
    Mat4d last_keyframe_Twc_;


    std::shared_ptr<ImagePkg> cur_img_pkg_ = nullptr; //synchronized images
    camodocal::CameraPtr left_camera_;
    camodocal::CameraPtr right_camera_;

    sad::WheelEncoder cur_wheel_encoder_; //synchronized wheel encoder
    
    Mat4d T_B_lC_;
    Mat4d T_B_rC_;
    Mat4d T_B_lC_inv_;
    Mat4d T_B_rC_inv_;
    SlidingWindowDepthFilter depth_filter_left_;
    SlidingWindowDepthFilter depth_filter_right_;

    cv::Ptr<cv::ORB> orb_left_;
    cv::Ptr<cv::ORB> orb_right_;
    std::shared_ptr<FeatureTracker> left_feature_tracker_ = nullptr;
    std::shared_ptr<FeatureTracker> right_feature_tracker_ = nullptr;

    LKToORBConverter left_lk2orb_;
    LKToORBConverter right_lk2orb_;
    // FeatureManager feature_manager_orb_;

    // cache visual constraints to submap visual points
    int left_correspondence_fid_ = -1; 
    int right_correspondence_fid_ = -1;
    std::vector<Vec3d> left_vmap_points_w_;
    std::vector<Vec3d> right_vmap_points_w_;
    std::vector<cv::Point2f> left_uvs_;
    std::vector<cv::Point2f> right_uvs_;

    cv::Mat img_match_debug_;
    cv::Mat mask_;

    PoseCurvePredictor pose_predictor_;
    
    pcl::PointCloud<PointXYZT>::Ptr last_scan_raw_ = nullptr;

    double time_cost_lidar_rotation_guess_ = 0.;
    double time_cost_camera_rotation_guess_ = 0.;
    double time_cost_meshing_ = 0.;
    double time_cost_raycast_ = 0.;
    double time_cost_lo_update_ = 0.;
    double time_cost_vo_update_ = 0.;
};

#endif 