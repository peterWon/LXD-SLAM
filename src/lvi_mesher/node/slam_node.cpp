#include <execution>
#include "slam_node.h"
#include <sensor_msgs/point_cloud_conversion.h>

#include <exception>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <geometry_msgs/TransformStamped.h>

using namespace std;
 
SlamNode::SlamNode(ros::NodeHandle& nh){
    odom_pub = nh.advertise<nav_msgs::Odometry>(param_.frontend_odometry_name, 1);
    registered_points_pub = nh.advertise<sensor_msgs::PointCloud2>(param_.frontend_rgt_points_name, 1);
    assoicate_depth_points_pub = nh.advertise<sensor_msgs::PointCloud2>(param_.associate_depth_points_name, 1);
    full_points_pub = nh.advertise<sensor_msgs::PointCloud2>(param_.frontend_full_points_name, 1);
    path_odom_pub = nh.advertise<nav_msgs::Path>(param_.frontend_path_odom_name, 1);
    mesh_pub = nh.advertise<mesh_msgs::MeshGeometryStamped>(param_.frontend_mesh_full_name, 1);
    mesh_vtx_pub = nh.advertise<mesh_msgs::MeshVertexColorsStamped>(param_.frontend_vtxcolor_full_name, 1);
    mesh_pub_local = nh.advertise<mesh_msgs::MeshGeometryStamped>(param_.frontend_mesh_now_name, 1);
    
    if(param_.enable_imu){
        imu_sub = nh.subscribe(param_.imu_topic, 500, &BaseNode::imuCallback, dynamic_cast<BaseNode*>(this));
    }

    if(param_.enable_wheel_encoder){
        wheel_encoder_sub = nh.subscribe(param_.wheel_encoder_topic, 500, &BaseNode::wheelEncoderCallback, dynamic_cast<BaseNode*>(this));
    }

    if(param_.lidar1_type==std::string("livox")){
        LOG(INFO)<<"Subscribe to livox1: "<<param_.lidar1_topic;
        pointcloud_sub1 = nh.subscribe(param_.lidar1_topic, 100, &BaseNode::livoxCallback, dynamic_cast<BaseNode*>(this));
    }else{
        LOG(INFO)<<"Subscribe to lidar1: "<<param_.lidar1_topic;
        pointcloud_sub1 = nh.subscribe(param_.lidar1_topic, 100, &BaseNode::pointCloudCallback, dynamic_cast<BaseNode*>(this));
    }

    if(param_.enable_lidar2){
        if(param_.lidar2_type==std::string("livox")){
            LOG(INFO)<<"Subscribe to livox2: "<<param_.lidar2_topic;
            pointcloud_sub2 = nh.subscribe(param_.lidar2_topic, 100, &BaseNode::livoxCallback2, dynamic_cast<BaseNode*>(this));
        }else{
            LOG(INFO)<<"Subscribe to lidar2: "<<param_.lidar2_topic;
            pointcloud_sub2 = nh.subscribe(param_.lidar2_topic, 100, &BaseNode::pointCloudCallback2, dynamic_cast<BaseNode*>(this));
        }
    }
    
    if(param_.enable_camera){
        img_left_sub = nh.subscribe(param_.image_left_topic, 100, &BaseNode::imgLeftCallback, dynamic_cast<BaseNode*>(this));
    }

    if(param_.enable_right_camera){
        img_right_sub = nh.subscribe(param_.image_right_topic, 100, &BaseNode::imgRightCallback, dynamic_cast<BaseNode*>(this));
    }

    if(param_.enable_gnss){
        gnss_sub = nh.subscribe(param_.gnss_topic, 100, &BaseNode::gnssCallback, dynamic_cast<BaseNode*>(this));
    }

    image_transport::ImageTransport it(nh);
    img_pub = it.advertise(param_.frontend_track_image, 1);
    
    pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>(Parameter::GetInstance().backend_odom_aft_pgo_name, 100);
    // pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>(Parameter::GetInstance().backend_odom_repub_name, 100);
    pubPathAftPGO = nh.advertise<nav_msgs::Path>(Parameter::GetInstance().backend_path_aft_pgo_name, 100);
    pubGNSS = nh.advertise<nav_msgs::Path>(Parameter::GetInstance().backend_gnss_path_name, 100);
    pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>(Parameter::GetInstance().backend_map_aft_pgo_name, 100);
    pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>(
        Parameter::GetInstance().backend_loop_scan_local_name, 100);
    pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>(
        Parameter::GetInstance().backend_loop_submap_local_name, 100);
    
    // put the backend message publisher in another thread to avoid blocking
    publisher_thread_ = std::thread(&SlamNode::publish_loop, this);
    _path_w_pub = nh.advertise<nav_msgs::Path>("path_w", 1);
}

SlamNode::~SlamNode(){}


void SlamNode::publishOdometry(){
    nav_msgs::Odometry odom_msg;
    odom_msg.header.frame_id = "map";
    odom_msg.child_frame_id = "base_link";
    
    odom_msg.header.stamp = ros::Time::now();
    Mat4d transf_now = liomesher_->pose();
    odom_msg.pose = transf2PoseWithCovariance(transf_now);
    odom_pub.publish(odom_msg);

    //pub tf
    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped tsf;
    tsf.header.stamp = odom_msg.header.stamp;
    tsf.header.frame_id = "map";
    tsf.child_frame_id = "base_link";
    tsf.transform.translation.x = odom_msg.pose.pose.position.x;
    tsf.transform.translation.y = odom_msg.pose.pose.position.y;
    tsf.transform.translation.z = odom_msg.pose.pose.position.z;
    tsf.transform.rotation.x = odom_msg.pose.pose.orientation.x;
    tsf.transform.rotation.y = odom_msg.pose.pose.orientation.y;
    tsf.transform.rotation.z = odom_msg.pose.pose.orientation.z;
    tsf.transform.rotation.w = odom_msg.pose.pose.orientation.w;
    
    br.sendTransform(tsf);
}


void SlamNode::publishRegisteredPoints(const std::shared_ptr<TrackingResult>& result){
    const Mat4d& pose = result->local_pose;
    const pcl::PointCloud<pcl::PointXYZRGB>& dps = result->deskewed_points;

    pcl::PointCloud<pcl::PointXYZRGB> pcl_points;
    size_t n = dps.points.size();
    pcl_points.points.resize(n);
    // LOG(INFO)<<"Registered point with size: "<<n;
    for (unsigned int i = 0; i < n; ++i) {
        Vec4d hpt={dps.points[i].x, dps.points[i].y, dps.points[i].z, 1};
        Vec4d hpt_w = pose * hpt;
        pcl_points.points[i].x = hpt_w(0);
        pcl_points.points[i].y = hpt_w(1);
        pcl_points.points[i].z = hpt_w(2);
        pcl_points.points[i].r = dps.points[i].r;
        pcl_points.points[i].g = dps.points[i].g;
        pcl_points.points[i].b = dps.points[i].b;
    }
    sensor_msgs::PointCloud2 rgt_points_msg;
    pcl::toROSMsg(pcl_points, rgt_points_msg);
    rgt_points_msg.header.stamp = ros::Time::now();
    rgt_points_msg.header.frame_id = "map";
    registered_points_pub.publish(rgt_points_msg);
}

void SlamNode::publishAssociatedDepthPoints(const std::shared_ptr<TrackingResult>& result){
    const Mat4d& pose = result->local_pose;

    pcl::PointCloud<pcl::PointXYZRGB> pcl_points;
    size_t n = result->current_mature_points.size();
    pcl_points.points.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
        Vec3d pt = result->current_mature_points[i];
        pcl_points.points[i].x = pt(0);
        pcl_points.points[i].y = pt(1);
        pcl_points.points[i].z = pt(2);
        pcl_points.points[i].r = 255;
        pcl_points.points[i].g = 0;
        pcl_points.points[i].b = 0;
    }
    sensor_msgs::PointCloud2 rgt_points_msg;
    pcl::toROSMsg(pcl_points, rgt_points_msg);
    rgt_points_msg.header.stamp = ros::Time::now();
    rgt_points_msg.header.frame_id = "map";
    assoicate_depth_points_pub.publish(rgt_points_msg);
}

void SlamNode::publishColorizedPoints(const std::shared_ptr<TrackingResult>& result){
    const Mat4d& pose = result->local_pose;
    const pcl::PointCloud<pcl::PointXYZRGB>& dps = result->deskewed_points;
    pcl::PointCloud<pcl::PointXYZ> xyz;
    xyzrgb2xyz(dps, xyz);
    pcl::PointCloud<pcl::PointXYZRGB> pcl_points;
    auto matching_submap = liomesher_->get_matching_submap();
    matching_submap->RetrieveColors(xyz, pose, pcl_points);
    for (unsigned int i = 0; i < xyz.points.size(); ++i) {
        Vec4d hpt={xyz.points[i].x, xyz.points[i].y, xyz.points[i].z, 1};
        Vec4d hpt_w = pose * hpt;
        pcl_points.points[i].x = hpt_w(0);
        pcl_points.points[i].y = hpt_w(1);
        pcl_points.points[i].z = hpt_w(2);
    }
    sensor_msgs::PointCloud2 rgt_points_msg;
    pcl::toROSMsg(pcl_points, rgt_points_msg);
    rgt_points_msg.header.stamp = ros::Time::now();
    rgt_points_msg.header.frame_id = "map";
    registered_points_pub.publish(rgt_points_msg);

    // if(result->frame_id % 3 == 0){
    //     pcl::io::savePLYFile("/home/wz/codes/ros_ws/lios/slamesh_ws/log/ouster_color/"+std::to_string(result->frame_id)+".ply", pcl_points);
    // }
}

void SlamNode::publishSubmapPoints(const std::shared_ptr<TrackingResult>& result){
    auto matching_submap = liomesher_->get_matching_submap();
    pcl::PointCloud<pcl::PointXYZRGB> pcl_points;
    matching_submap->GetAllPoints(pcl_points);
    sensor_msgs::PointCloud2 rgt_points_msg;
    pcl::toROSMsg(pcl_points, rgt_points_msg);
    rgt_points_msg.header.stamp = ros::Time::now();
    rgt_points_msg.header.frame_id = "map";
    registered_points_pub.publish(rgt_points_msg);
}

void SlamNode::publishTrackedImage(const std::shared_ptr<TrackingResult>& result){
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(
        std_msgs::Header(), "bgr8", result->debug_img).toImageMsg();
    img_pub.publish(img_msg);
}

void SlamNode::publishPathVIO(const std::shared_ptr<TrackingResult>& result){
   /*  Mat4d tsf = result->local_pose_camera;
    geometry_msgs::PoseStamped_<std::allocator<void>> pose;
    pose.pose.position.x = tsf(0, 3);
    pose.pose.position.y = tsf(1, 3);
    pose.pose.position.z = tsf(2, 3);

    tf::Matrix3x3 tmp_m(tsf(0, 0), tsf(0, 1), tsf(0, 2),
                        tsf(1, 0), tsf(1, 1), tsf(1, 2),
                        tsf(2, 0), tsf(2, 1), tsf(2, 2));
    double roll, yaw, pitch;
    tmp_m.getEulerYPR(yaw, pitch, roll);
    tf::Quaternion tmp_q;
    tmp_q.setRPY(roll, pitch, yaw);
    pose.pose.orientation.x = tmp_q.x();
    pose.pose.orientation.y = tmp_q.y();
    pose.pose.orientation.z = tmp_q.z();
    pose.pose.orientation.w = tmp_q.w();

    path_vio_.poses.push_back(pose);
    path_vio_.header.stamp = ros::Time::now();
    path_vio_.header.frame_id = "map";
    path_odom_pub.publish(path_vio_); */
    // ... 在你的节点内部 ...
    

    // 在回调函数或主循环中
    geometry_msgs::TransformStamped transform_stamped;

    // 填充时间戳和坐标系关系
    transform_stamped.header.stamp = ros::Time::now();
    transform_stamped.header.frame_id = "map"; // 父坐标系，通常是世界坐标系
    transform_stamped.child_frame_id = "camera_link"; // 子坐标系，即相机坐标系

    // 填充平移部分 (x, y, z)
    Mat4d tsf = result->local_pose_camera;
    transform_stamped.transform.translation.x = tsf(0, 3);
    transform_stamped.transform.translation.y = tsf(1, 3);
    transform_stamped.transform.translation.z = tsf(2, 3);

    // 填充旋转部分 (四元数)
    // 如果你的位姿是 geometry_msgs::Pose，可以直接使用其 orientation
    Eigen::Quaterniond q = Eigen::Quaterniond(tsf.block<3,3>(0,0)).normalized();
    transform_stamped.transform.rotation.x = q.x();
    transform_stamped.transform.rotation.y = q.y();
    transform_stamped.transform.rotation.z = q.z();
    transform_stamped.transform.rotation.w = q.w();

    // 发布变换
    tf_broadcaster.sendTransform(transform_stamped);
}

void SlamNode::publishPathLIO(const std::shared_ptr<TrackingResult>& result){
    Mat4d tsf = liomesher_->pose();
    geometry_msgs::PoseStamped_<std::allocator<void>> pose;
    pose.pose.position.x = tsf(0, 3);
    pose.pose.position.y = tsf(1, 3);
    pose.pose.position.z = tsf(2, 3);

    tf::Matrix3x3 tmp_m(tsf(0, 0), tsf(0, 1), tsf(0, 2),
                        tsf(1, 0), tsf(1, 1), tsf(1, 2),
                        tsf(2, 0), tsf(2, 1), tsf(2, 2));
    double roll, yaw, pitch;
    tmp_m.getEulerYPR(yaw, pitch, roll);
    tf::Quaternion tmp_q;
    tmp_q.setRPY(roll, pitch, yaw);
    pose.pose.orientation.x = tmp_q.x();
    pose.pose.orientation.y = tmp_q.y();
    pose.pose.orientation.z = tmp_q.z();
    pose.pose.orientation.w = tmp_q.w();
    
    path_lio_.poses.push_back(pose);
    path_lio_.header.stamp = ros::Time::now();
    path_lio_.header.frame_id = "map";
    _path_w_pub.publish(path_lio_);
}

void SlamNode::publishMesh(){
    auto matching_submap = liomesher_->get_matching_submap();
    if(mesh_pub_local.getNumSubscribers() > 0){//
        matching_submap->filterMeshLocal();
        mesh_pub_local.publish(matching_submap->mesh_msg_now_);
    }

    if(mesh_pub.getNumSubscribers() > 0){//
        matching_submap->filterMeshGlb();
        mesh_pub.publish(matching_submap->mesh_msg_full_);
    }
}

void SlamNode::publishVertices(){
    auto matching_submap = liomesher_->get_matching_submap();
    if(!matching_submap) return;
    matching_submap->filterVerticesByVariance(true, 1000);

    const Mat4d& pose = matching_submap->pose_odom_;
    PointMatrix& vertices = matching_submap->vertices_filted;

    pcl::PointCloud<pcl::PointXYZRGB> pcl_points;
    size_t n = vertices.num_point;
    pcl_points.points.reserve(n);

    for (unsigned int i = 0; i < n; ++i) {
        pcl::PointXYZRGB xyz_rgb;
        Vec3d xyz = vertices.point.col(i);
        Vec3d rgb = vertices.rgb.col(i);
        Vec4d hpt={xyz(0), xyz(1), xyz(2), 1};
        Vec4d hpt_w = pose * hpt;
        xyz_rgb.x = hpt_w(0);
        xyz_rgb.y = hpt_w(1);
        xyz_rgb.z = hpt_w(2);
        xyz_rgb.r = rgb(0);
        xyz_rgb.g = rgb(1);
        xyz_rgb.b = rgb(2);
        pcl_points.points.push_back(xyz_rgb);
    }
    sensor_msgs::PointCloud2 rgt_points_msg;
    pcl::toROSMsg(pcl_points, rgt_points_msg);
    rgt_points_msg.header.stamp = ros::Time::now();
    rgt_points_msg.header.frame_id = "map";
    registered_points_pub.publish(rgt_points_msg);
}

void SlamNode::publishPgoPath(){
    keyframe_poses_opt_ = posegraph_->getUpdatedKeyframePoses();
    // keyframe_poses_opt_ = posegraph_->getGPSPath();
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "map";
    for (int node_idx=0; node_idx < keyframe_poses_opt_.size(); node_idx++){
        const Pose6D& pose_est = keyframe_poses_opt_.at(node_idx); // upodated poses
        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "map";
        odomAftPGOthis.child_frame_id = "base_link_aft_pgo";
        // odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(
            pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "map";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    pubPathAftPGO.publish(pathAftPGO); // poses 
}

void SlamNode::publishGNSSPath(){
    auto gnss_path = posegraph_->getGPSPath();
    // std::vector<Pose6D> gnss_path;
    // for(int i = 0; i < gnss_positions_.size(); i++){
    //     Pose6D gnssi;
    //     gnssi.x = gnss_positions_[i](0);
    //     gnssi.y = gnss_positions_[i](1);
    //     gnssi.z = gnss_positions_[i](2);
    //     gnss_path.push_back(gnssi);
    // }    
    nav_msgs::Odometry node;
    nav_msgs::Path path;
    path.header.frame_id = "map";
    for (int node_idx=0; node_idx < gnss_path.size(); node_idx++){
        const Pose6D& pose_est = gnss_path.at(node_idx); // upodated poses
        nav_msgs::Odometry node;
        node.header.frame_id = "map";
        node.child_frame_id = "gnss";
        node.pose.pose.position.x = pose_est.x;
        node.pose.pose.position.y = pose_est.y;
        node.pose.pose.position.z = pose_est.z;
        node.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(
            pose_est.roll, pose_est.pitch, pose_est.yaw);

        geometry_msgs::PoseStamped stamped_pose;
        stamped_pose.header = node.header;
        stamped_pose.pose = node.pose.pose;

        path.header.stamp = node.header.stamp;
        path.header.frame_id = "map";
        path.poses.push_back(stamped_pose);
    }
    pubGNSS.publish(path); // poses 
}


void SlamNode::publish_loop(){
    float hz = 10.0; 
    ros::Rate rate(hz);
    while(ros::ok()) {
        rate.sleep();
        // publish frontend states
        mtx_result_.lock();
        // publishColorizedPoints(result);
        // publishSubmapPoints(result);
        if(result_){
            publishRegisteredPoints(result_);
            publishAssociatedDepthPoints(result_);
            publishTrackedImage(result_);
            publishPathLIO(result_);
        }
        mtx_result_.unlock();
        publishPgoPath();
        publishGNSSPath();
        // publishVertices();
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    LOG(INFO)<<"====START lvimesher_node====";
    ros::init(argc, argv, "lvimesher");

    ros::NodeHandle nh;
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    ros::Rate r(10);

    Parameter& param_=Parameter::GetInstance();
    param_.initParameter(nh);

    SlamNode lvimesher(nh);

    r.sleep();
    ros::spin();

    return 0;
}