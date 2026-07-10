#ifndef __LIOMESHER_NODE_H__
#define __LIOMESHER_NODE_H__

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

#include "base_node.h"

using namespace camodocal;

class SlamNode: public BaseNode{
public:
    SlamNode(ros::NodeHandle& nh);
    ~SlamNode();

    void publish_loop();

    // frontend  
    void publishOdometry();
    void publishRegisteredPoints(const std::shared_ptr<TrackingResult>& result);
    void publishAssociatedDepthPoints(const std::shared_ptr<TrackingResult>& result);
    void publishColorizedPoints(const std::shared_ptr<TrackingResult>& result);
    void publishSubmapPoints(const std::shared_ptr<TrackingResult>& result);
    void publishPathVIO(const std::shared_ptr<TrackingResult>& result);
    void publishTrackedImage(const std::shared_ptr<TrackingResult>& result);
    void publishPathLIO(const std::shared_ptr<TrackingResult>& result);
    void publishMesh();
    void publishVertices();

    void publishPgoPath();
    void publishGNSSPath();
    
    ros::Publisher odom_pub;
    ros::Publisher path_odom_pub;
    ros::Publisher full_points_pub;
    ros::Publisher registered_points_pub;
    ros::Publisher assoicate_depth_points_pub;
    ros::Publisher mesh_pub;
    ros::Publisher mesh_vtx_pub;
    ros::Publisher mesh_pub_local;
    ros::Publisher _path_w_pub;
    image_transport::Publisher img_pub;

    ros::Subscriber pointcloud_sub1;
    ros::Subscriber pointcloud_sub2;
    ros::Subscriber imu_sub;
    ros::Subscriber wheel_encoder_sub;
    ros::Subscriber img_left_sub;
    ros::Subscriber img_right_sub;
    ros::Subscriber gnss_sub;

    tf2_ros::TransformBroadcaster tf_broadcaster;

    // backend
    ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO, pubGNSS;;
    ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
    
    std::thread publisher_thread_;

    // debug
    nav_msgs::Path path_lio_;
    nav_msgs::Path path_vio_;
};
#endif
