#include <execution>
#include "base_node.h"
#include <sensor_msgs/point_cloud_conversion.h>

#include <exception>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

#include <geometry_msgs/TransformStamped.h>

using namespace std;

BaseNode::BaseNode(){
    liomesher_.reset(new LioMesher());
    posegraph_.reset(new PoseGraph());
    posegraph_->LoadVocabulary(param_.voc_file_path);

    T_body_lidar1_ = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::LIDAR1).matrix();
    T_body_lidar2_ = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::LIDAR2).matrix();
    T_body_wheel_ = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::WHEEL).matrix();
    T_body_gnss_ = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::GNSS).matrix();

    range_min_sq_ = param_.range_min * param_.range_min;
    range_max_sq_ = param_.range_max * param_.range_max;

    liomehser_thread_ = std::thread(&BaseNode::process_loop, this);
    // LOG(INFO)<<"Base node initialized.";
}

BaseNode::~BaseNode(){
    if(liomesher_) liomesher_.release();
    if(posegraph_) posegraph_.release();
}


bool BaseNode::initialize(const std::vector<sensor_msgs::ImuPtr>& imus){
    std::copy(imus.begin(), imus.end(), std::back_inserter(imus_for_initialization_));
    size_t n = imus_for_initialization_.size();
    if(init_cached_frames_++ < 5){
        LOG(WARNING)<<"Wait for more IMU readings to initialize.";
        return false;
    }
    double ts = -1.;
    Eigen::Vector3d mean_lin_acc;
    Eigen::Vector3d mean_ang_vel;
    mean_lin_acc.setZero();
    mean_ang_vel.setZero();
    for(const auto& imu: imus_for_initialization_){
        ts = imu->header.stamp.toSec();
        Eigen::Vector3d la(imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z);
        Eigen::Vector3d av(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z);
        mean_lin_acc += la;
        mean_ang_vel += av;
    }
    mean_lin_acc /= n;
    mean_ang_vel /= n;
    
    Eigen::Vector3d g_vec(0., 0., -std::abs(param_.gravity));//TODO
    Eigen::Vector3d zero(0., 0., 0.);
    Eigen::Vector3d ba(0., 0., 0.);

    Eigen::Matrix3d R = Eigen::Matrix3d(Eigen::Quaternion<double>::FromTwoVectors(mean_lin_acc, -g_vec));
    ba = R.transpose() * g_vec + mean_lin_acc;

    LOG(INFO)<<"Initial Rotation: "<<R;
    LOG(INFO)<<"Initial Ba: "<<ba;

    liomesher_->initialize(ts, R, mean_ang_vel, ba);
    initialized_ = true;
    return true;
}

bool BaseNode::initialize_without_imu(double ts){
    Eigen::Vector3d zero(0., 0., 0.);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    liomesher_->initialize(ts, R, zero, zero);
    initialized_ = true;
    LOG(INFO)<<"Initialize without IMU data.";
    return true;
}


void BaseNode::imuCallback(const sensor_msgs::ImuPtr & imu_msg){
    if(!param_.enable_imu) return;
    if(imu_msg->header.stamp.toSec() < last_imu_t_){
        LOG(WARNING)<<"Imu readings in disorder!";
        return;
    }
    // TODO(wz)
    if(param_.rescale_imu_data){
        imu_msg->linear_acceleration.x *= abs(param_.gravity);
        imu_msg->linear_acceleration.y *= abs(param_.gravity);
        imu_msg->linear_acceleration.z *= abs(param_.gravity);
    }
    mtx_buf_.lock();
    imu_msg_buf_.push(imu_msg);
    last_imu_t_ = imu_msg->header.stamp.toSec();
    mtx_buf_.unlock();
    convar_.notify_all();
}

// Adapted from fusionportableV2: https://github.com/fusionportable/fusionportable_dataset_tools/tree/main/calibration/encoder_calc/encoder_calc
void BaseNode::wheelEncoderCallback(const sensor_msgs::JointStatePtr& msg){
    if(!param_.enable_wheel_encoder) return;
    if (msg->position.size() < 2) return;
    if (first_call_) {
        distance_x_prev_ = msg->position[0];
        distance_y_prev_ = msg->position[1];
        last_wheel_t_ = msg->header.stamp.toSec();
        first_call_ = false;
        return;
    }
    ros::Time current_time = msg->header.stamp;
    
    double dt = current_time.toSec() - last_wheel_t_;
    if (dt <= 0) return;

    mtx_buf_.lock();

    double dl = msg->position[0] - distance_x_prev_;
    double dr = msg->position[1] - distance_y_prev_;

    if (dl > OVERFLOW_LIMIT)  dl -= OVERFLOW_CORRECTION;
    else if (dl < -OVERFLOW_LIMIT) dl += OVERFLOW_CORRECTION;

    if (dr > OVERFLOW_LIMIT)  dr -= OVERFLOW_CORRECTION;
    else if (dr < -OVERFLOW_LIMIT) dr += OVERFLOW_CORRECTION;

    // 4. 将编码器增量转换为车轮位移 (单位: m)
    // 计算公式: (增量 / 1000) * 2 * PI * 半径
    // 注意：这里的 /1000 取决于你编码器的分辨率定义，参考原代码
    double ds_l = (dl / 1000.0) * M_PI * 2.0 * wheel_radius_l_;
    double ds_r = (dr / 1000.0) * M_PI * 2.0 * wheel_radius_r_;

    // 5. 计算线速度 (m/s) 和角速度 (rad/s)
    // 线速度 v = (左轮位移 + 右轮位移) / (2 * dt)
    double linear_vel = (ds_l + ds_r) / (2.0 * dt);
    
    // 角速度 w = (右轮位移 - 左轮位移) / (轮距 * dt)
    double angular_vel = (ds_r - ds_l) / (wheel_track_ * dt);

    // 7. 更新状态
    distance_x_prev_ = msg->position[0];
    distance_y_prev_ = msg->position[1];
    
    sad::WheelEncoder wheel_msg_new;
    wheel_msg_new.timestamp_ = current_time.toSec();
    Vec3d linear_vel_vec(linear_vel, 0., 0.);
    Vec3d angular_vel_vec(0., 0., angular_vel);
    wheel_msg_new.angular_velocity_ = T_body_wheel_.block<3, 3>(0, 0) * angular_vel_vec;
    wheel_msg_new.linear_velocity_ = T_body_wheel_.block<3, 3>(0, 0) * linear_vel_vec +  wheel_msg_new.angular_velocity_.cross(T_body_wheel_.block<3, 1>(0, 3));
    wheel_msg_buf_.push(wheel_msg_new);

    last_wheel_t_ = current_time.toSec();
    mtx_buf_.unlock();
    convar_.notify_all();
}

void BaseNode::gnssCallback(const sensor_msgs::NavSatFixPtr & gnss_msg){
    if(!param_.enable_gnss) return;
    mtx_buf_.lock();
    gnss_msg_buf_.push(gnss_msg);
    // LOG(INFO)<<gnss_msg->header.stamp.toSec() - last_img_left_t_;
    // Vec3d enu;
    // if(!gnss_inited_){
    //     geo_converter_.Reset(gnss_msg->latitude, gnss_msg->longitude, gnss_msg->altitude);
    //     gnss_inited_ = true;
    // }
    // geo_converter_.Forward(gnss_msg->latitude, gnss_msg->longitude, gnss_msg->altitude, enu[0], enu[1], enu[2]);
    // gnss_positions_.push_back(enu);
    mtx_buf_.unlock();
    convar_.notify_all();
}

void BaseNode::imgLeftCallback(const sensor_msgs::ImagePtr & img_msg){
    if(!param_.enable_camera) return;
    double raw_ts = img_msg->header.stamp.toSec();
    double new_ts = raw_ts + param_.left_img_time_offset;
    img_msg->header.stamp = ros::Time(new_ts);
    if(img_msg->header.stamp.toSec() < last_img_left_t_){
        LOG(WARNING)<<"Left image readings in disorder!";
        return;
    }
    if(raw_left_img_idx_++ % (param_.skip_image_frames + 1) != 0) return;
    if (last_lidar1_t_ < 0) return;
    if (abs(new_ts - last_img_left_t_) < 0.001) return;
    mtx_buf_.lock();
    img_left_buf_.push(img_msg);
    last_img_left_t_ = img_msg->header.stamp.toSec();
    mtx_buf_.unlock();
    convar_.notify_all();
}

void BaseNode::imgRightCallback(const sensor_msgs::ImagePtr & img_msg){
    if(!param_.enable_right_camera) return;
    double raw_ts = img_msg->header.stamp.toSec();
    double new_ts = raw_ts + param_.right_img_time_offset;
    img_msg->header.stamp = ros::Time(new_ts);
    if(img_msg->header.stamp.toSec() < last_img_right_t_){
        LOG(WARNING)<<"Right image readings in disorder!";
        return;
    }
    if(raw_right_img_idx_++ % (param_.skip_image_frames + 1) != 0) return;
    if (last_lidar1_t_ < 0) return;
    if (abs(new_ts - last_img_right_t_) < 0.001) return;
    mtx_buf_.lock();
    img_right_buf_.push(img_msg);
    last_img_right_t_ = img_msg->header.stamp.toSec();
    mtx_buf_.unlock();
    convar_.notify_all();
}


void BaseNode::pointCloudCallback(const sensor_msgs::PointCloud2Ptr & msg){
    double raw_ts = msg->header.stamp.toSec();
    double new_ts = raw_ts + param_.lidar1_time_offset;
    msg->header.stamp = ros::Time(new_ts);
    double timestamp = msg->header.stamp.toSec();
    if(last_lidar1_t_ > timestamp) return;
    mtx_buf_.lock();
    pcl::PointCloud<PointXYZT>::Ptr scan(new pcl::PointCloud<PointXYZT>());
    scan->points.reserve(msg->height*msg->width);
    if(param_.lidar1_type == "ouster"){
        pcl::PointCloud<OusterPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // important, some points are in disorder
        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](
                const OusterPointXYZIRT &a, const OusterPointXYZIRT &b) {
            return a.t < b.t;
        });
        int64_t t_end = pc_raw.points.back().t;
        int64_t t_start = pc_raw.points.front().t;
        int64_t ts = msg->header.stamp.toNSec();
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar1_.block(0, 0, 3, 3) * pi_L + T_body_lidar1_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
            
            xyzt.t = (ts - t_end + point.t) * 1e-9;//TODO(wz): verify
            // xyzt.t = (ts + point.t) * 1e-9;
            scan->points.emplace_back(xyzt);
        }
        timestamp = scan->points.back().t;
    }else if(param_.lidar1_type == "velodyne"){
        pcl::PointCloud<VeloPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // LOG(INFO)<<pc_raw.points.size();

        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](const VeloPointXYZIRT &a, const VeloPointXYZIRT &b) {
            return a.time < b.time;
        });
        double ts = msg->header.stamp.toSec();
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar1_.block(0, 0, 3, 3) * pi_L + T_body_lidar1_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;

            xyzt.t = ts + point.time;//TODO(wz): verify
            scan->points.emplace_back(xyzt);
        }
        timestamp = scan->points.back().t;
    }else if(param_.lidar1_type == "robosense"){
        pcl::PointCloud<RsPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // important, some points are in disorder
        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](const RsPointXYZIRT &a, const RsPointXYZIRT &b) {
            return a.timestamp < b.timestamp;
        });
        // Robosense ROS消息的stamp记录的是最后点的采集时间
        double st = msg->header.stamp.toSec();
        double ts = pc_raw.points.front().timestamp;
        double te = pc_raw.points.back().timestamp;
        // LOG(INFO)<<st<<" "<<ts<<" "<<te;
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar1_.block(0, 0, 3, 3) * pi_L + T_body_lidar1_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
            
            xyzt.t = point.timestamp;
            // LOG(INFO)<<"point timestamp: "<<xyzt.t-ts;
            scan->points.emplace_back(xyzt);
        }
        timestamp = te;
    }else if(param_.lidar1_type == "hesai"){
        pcl::PointCloud<HesaiPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // important, some points are in disorder
        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](const HesaiPointXYZIRT &a, const HesaiPointXYZIRT &b) {
            return a.timestamp < b.timestamp;
        });
        double st = msg->header.stamp.toSec();
        double ts = pc_raw.points.front().timestamp;
        double te = pc_raw.points.back().timestamp;
        
        // hesai每个点的timestamp(second)是世界时间不是相对第一个点的时间
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar1_.block(0, 0, 3, 3) * pi_L + T_body_lidar1_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
            
            xyzt.t = point.timestamp;
            scan->points.emplace_back(xyzt);
        }
        timestamp = te;
    }else if(param_.lidar1_type == "XYZI"){
        pcl::PointCloud<pcl::PointXYZI> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        double st = msg->header.stamp.toSec()-0.1;
        timestamp = st + 0.1;
        int n = pc_raw.points.size();
        for (int i = 0; i < n; i++) {
            const auto& point = pc_raw.points[i];
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar1_.block(0, 0, 3, 3) * pi_L + T_body_lidar1_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;

            xyzt.t = st + double(i) * 0.1 / double(n);
            scan->points.emplace_back(xyzt);
        }
    }
    scan1_buf_.push_back(scan);
    last_lidar1_t_ = timestamp;
    mtx_buf_.unlock();
    convar_.notify_all();
}

void BaseNode::pointCloudCallback2(const sensor_msgs::PointCloud2Ptr & msg){
    if(!param_.enable_lidar2) return;
    double raw_ts = msg->header.stamp.toSec();
    double new_ts = raw_ts + param_.lidar2_time_offset;
    msg->header.stamp = ros::Time(new_ts);
    
    double timestamp = msg->header.stamp.toSec();
    mtx_buf_.lock();
    pcl::PointCloud<PointXYZT>::Ptr scan(new pcl::PointCloud<PointXYZT>());
    scan->points.reserve(msg->height*msg->width);
    if(param_.lidar2_type == "ouster"){
        pcl::PointCloud<OusterPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // important, some points are in disorder5
        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](
                const OusterPointXYZIRT &a, const OusterPointXYZIRT &b) {
            return a.t < b.t;
        });
        int64_t t_end = pc_raw.points.back().t;
        int64_t t_start = pc_raw.points.front().t;
        int64_t ts = msg->header.stamp.toNSec();
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar2_.block(0, 0, 3, 3) * pi_L + T_body_lidar2_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
            
            xyzt.t = (ts - t_end + point.t) * 1e-9;//TODO(wz): verify
            scan->points.emplace_back(xyzt);
        }
    }else if(param_.lidar2_type == "velodyne"){
        pcl::PointCloud<VeloPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // LOG(INFO)<<pc_raw.points.size();

        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](const VeloPointXYZIRT &a, const VeloPointXYZIRT &b) {
            return a.time < b.time;
        });
        double st = msg->header.stamp.toSec();
        double te = pc_raw.points.back().time;
        double ts = pc_raw.points.front().time;
        double et = st + te - ts;
        // LOG(INFO)<<ts<<" "<<te<<", "<<te-ts;
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar2_.block(0, 0, 3, 3) * pi_L + T_body_lidar2_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;

            xyzt.t = et + point.time;//TODO(wz): verify
            scan->points.emplace_back(xyzt);
        }
        timestamp = et;
    }else if(param_.lidar2_type == "robosense"){
        pcl::PointCloud<RsPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // important, some points are in disorder
        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](const RsPointXYZIRT &a, const RsPointXYZIRT &b) {
            return a.timestamp < b.timestamp;
        });
        // Robosense ROS消息的stamp记录的是最后点的采集时间
        double st = msg->header.stamp.toSec();
        double ts = pc_raw.points.front().timestamp;
        double te = pc_raw.points.back().timestamp;
        // LOG(INFO)<<st<<" "<<ts<<" "<<te;
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar2_.block(0, 0, 3, 3) * pi_L + T_body_lidar2_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
            
            xyzt.t = point.timestamp;
            // LOG(INFO)<<"point timestamp: "<<xyzt.t-ts;
            scan->points.emplace_back(xyzt);
        }
        timestamp = te;
    }else if(param_.lidar2_type == "hesai"){
        pcl::PointCloud<HesaiPointXYZIRT> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        // important, some points are in disorder
        std::sort(pc_raw.points.begin(), pc_raw.points.end(), [](const HesaiPointXYZIRT &a, const HesaiPointXYZIRT &b) {
            return a.timestamp < b.timestamp;
        });
        double st = msg->header.stamp.toSec();
        double ts = pc_raw.points.front().timestamp;
        double te = pc_raw.points.back().timestamp;
        
        // hesai每个点的timestamp(second)是世界时间不是相对第一个点的时间
        for (const auto& point : pc_raw.points) {
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar2_.block(0, 0, 3, 3) * pi_L + T_body_lidar2_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
            
            xyzt.t = point.timestamp;
            scan->points.emplace_back(xyzt);
        }
        timestamp = te;
    }else if(param_.lidar2_type == "XYZI"){
        pcl::PointCloud<pcl::PointXYZI> pc_raw;
        pcl::fromROSMsg(*msg, pc_raw);
        double st = msg->header.stamp.toSec()-0.1;
        timestamp = st + 0.1;
        int n = pc_raw.points.size();
        for (int i = 0; i < n; i++) {
            const auto& point = pc_raw.points[i];
            Vec3d pi_L = {point.x, point.y, point.z};
            Vec3d pi_B = T_body_lidar2_.block(0, 0, 3, 3) * pi_L + T_body_lidar2_.block(0, 3, 3, 1);
            PointXYZT xyzt;
            xyzt.x = pi_B(0);
            xyzt.y = pi_B(1);
            xyzt.z = pi_B(2);
            double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
            if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;

            xyzt.t = st + double(i) * 0.1 / double(n);
            scan->points.emplace_back(xyzt);
        }
    }
    scan2_buf_.push_back(scan);
    mtx_buf_.unlock();
    convar_.notify_all();
}

void BaseNode::livoxCallback(const livox_ros_driver::CustomMsg::ConstPtr &msg){
    double timestamp = msg->header.stamp.toSec();
    if(last_lidar1_t_ > timestamp) return;
    mtx_buf_.lock();
    pcl::PointCloud<PointXYZT>::Ptr scan(new pcl::PointCloud<PointXYZT>());
    std::vector<PointXYZT> tmp_pts;
    scan->points.reserve(msg->point_num);
    double st = msg->header.stamp.toSec();
    for (int point_id = 0; point_id < msg->point_num; ++point_id) {
        auto point = msg->points[point_id]; 
        if (pcl_isnan(point.x) || pcl_isnan(point.y) || pcl_isnan(point.z)) {
            continue;
        }
        
        Vec3d pi_L = {point.x, point.y, point.z};
        Vec3d pi_B = T_body_lidar1_.block(0, 0, 3, 3) * pi_L + T_body_lidar1_.block(0, 3, 3, 1);
        PointXYZT xyzt;
        xyzt.x = pi_B(0);
        xyzt.y = pi_B(1);
        xyzt.z = pi_B(2);
        double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
        if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
        double offset_time = double(point.offset_time) * 1e-9;
        if( offset_time > 0.11 || offset_time < 0) continue;
        xyzt.t = st + offset_time;
        scan->points.push_back(xyzt);
    }
    std::sort(scan->points.begin(), scan->points.end(), [](const PointXYZT &a, const PointXYZT &b) {
        return a.t < b.t;
    });
    scan1_buf_.push_back(scan);
    last_lidar1_t_ = st;
    mtx_buf_.unlock();
    convar_.notify_all();
}

void BaseNode::livoxCallback2(const livox_ros_driver::CustomMsg::ConstPtr &msg){
    if(!param_.enable_lidar2) return;
    double timestamp = msg->header.stamp.toSec();
    if(last_lidar1_t_ > timestamp) return;
    mtx_buf_.lock();
    pcl::PointCloud<PointXYZT>::Ptr scan(new pcl::PointCloud<PointXYZT>());
    std::vector<PointXYZT> tmp_pts;
    scan->points.reserve(msg->point_num);
    double st = msg->header.stamp.toSec();
    for (int point_id = 0; point_id < msg->point_num; ++point_id) {
        auto point = msg->points[point_id]; 
        if (pcl_isnan(point.x) || pcl_isnan(point.y) || pcl_isnan(point.z)) {
            continue;
        }
        
        Vec3d pi_L = {point.x, point.y, point.z};
        Vec3d pi_B = T_body_lidar2_.block(0, 0, 3, 3) * pi_L + T_body_lidar2_.block(0, 3, 3, 1);
        PointXYZT xyzt;
        xyzt.x = pi_B(0);
        xyzt.y = pi_B(1);
        xyzt.z = pi_B(2);
        double rsq = xyzt.x*xyzt.x + xyzt.y*xyzt.y+xyzt.z*xyzt.z;
        if(rsq < range_min_sq_ || rsq > range_max_sq_) continue;
        double offset_time = double(point.offset_time) * 1e-9;
        if( offset_time > 0.11 || offset_time < 0) continue;
        xyzt.t = st + offset_time;
        scan->points.push_back(xyzt);
    }
    std::sort(scan->points.begin(), scan->points.end(), [](const PointXYZT &a, const PointXYZT &b) {
        return a.t < b.t;
    });
    scan2_buf_.push_back(scan);
    // last_lidar1_t_ = st;
    mtx_buf_.unlock();
    convar_.notify_all();
}

bool BaseNode::syncAllData(SyncResult& sync_result) {
    TicToc tic;
    
    // 初始化同步结果
    sync_result.imgpkg = std::make_shared<ImagePkg>();
    sync_result.scan_msg = nullptr;
    sync_result.imus.clear();
    sync_result.wheels.clear();
    
    // 基础检查：主LiDAR必须有数据
    if (scan1_buf_.empty()) {
        // LOG(WARNING) << "Main LiDAR buffer is empty!";
        return false;
    }
    
    // ========== 场景1：启用相机（以图像时间戳为基准） ==========
    if (param_.enable_camera) {
        if (img_left_buf_.empty()) {
            return false;
        }
        if (param_.enable_right_camera && img_right_buf_.empty()) {
            return false;
        }
        
        double imgl_ts = img_left_buf_.front()->header.stamp.toSec();
        double imgr_ts = -1.0;
        
        if (param_.enable_right_camera) {
            imgr_ts = img_right_buf_.front()->header.stamp.toSec();
            if (abs(imgl_ts - imgr_ts) > 0.1) {
                LOG(WARNING) << "Images time difference too large: " << abs(imgl_ts - imgr_ts);
            }
        }
        
        bool data_ready = true;
        if (param_.enable_imu) {
            if (imu_msg_buf_.empty()) {
                LOG(WARNING) << "IMU buffer is empty!";
                return false;
            }
            double latest_imu_ts = imu_msg_buf_.back()->header.stamp.toSec();
            data_ready &= (latest_imu_ts >= imgl_ts);
        }
        
        double latest_point_ts = scan1_buf_.back()->points.back().t;
        data_ready &= (latest_point_ts >= imgl_ts);
        
        if (!data_ready) {
            return false;
        }
        
        // 清理过期图像
        double front_imu_ts = param_.enable_imu ? imu_msg_buf_.front()->header.stamp.toSec() : 0.0;
        while (!img_left_buf_.empty() && front_imu_ts > imgl_ts) {
            LOG(WARNING) << "Throw old left image (only at beginning)";
            img_left_buf_.pop();
            if (!img_left_buf_.empty()) imgl_ts = img_left_buf_.front()->header.stamp.toSec();
        }
        if (img_left_buf_.empty()) return false;
        
        if (param_.enable_right_camera) {
            while (!img_right_buf_.empty() && front_imu_ts > imgr_ts) {
                LOG(WARNING) << "Throw old right image (only at beginning)";
                img_right_buf_.pop();
                if (!img_right_buf_.empty()) imgr_ts = img_right_buf_.front()->header.stamp.toSec();
            }
            if (img_right_buf_.empty()) return false;
        }
        
        // 清理过期LiDAR数据
        cleanLidarBuffer(scan1_buf_, last_img_t_);
        if (scan1_buf_.empty()) return false;
        
        if (param_.enable_lidar2) {
            cleanLidarBuffer(scan2_buf_, last_img_t_);
        }
        
        // 提取图像数据
        sync_result.imgpkg->left_img = ImgFromRos(img_left_buf_.front());
        img_left_buf_.pop();
        if (param_.enable_right_camera) {
            sync_result.imgpkg->right_img = ImgFromRos(img_right_buf_.front());
            img_right_buf_.pop();
        }
        
        // 同步IMU数据（如果启用）
        if (param_.enable_imu) {
            while (!imu_msg_buf_.empty()) {
                if (imu_msg_buf_.front()->header.stamp.toSec() > imgl_ts) break;
                sync_result.imus.emplace_back(imu_msg_buf_.front());
                imu_msg_buf_.pop();
            }
            if (sync_result.imus.empty()) {
                LOG(WARNING) << "No IMU data between two images!";
                return false;
            }
        }

        // 同步轮速计数据（如果启用）
        if (param_.enable_wheel_encoder) {
            if (wheel_msg_buf_.empty()) {
                LOG(WARNING) << "Wheel encoder buffer is empty!";
                return false;
            }
            
            double latest_wheel_ts = wheel_msg_buf_.back().timestamp_;
            while (!wheel_msg_buf_.empty()) {
                if (wheel_msg_buf_.front().timestamp_ > imgl_ts) break;
                sync_result.wheels.emplace_back(wheel_msg_buf_.front());
                wheel_msg_buf_.pop();
            }
            
            if (sync_result.wheels.empty()) {
                LOG(WARNING) << "No wheel data between two scans!";
                return false;
            }
        }

        // 同步GNSS数据（如果启用）
        if (param_.enable_gnss) {
            sensor_msgs::NavSatFixPtr gnss = getClosestAndPop(imgl_ts);
            if(gnss){
                sync_result.gnss = gnss;
            }
        }
        
        // 同步LiDAR数据
        syncMultiLidarData(sync_result.scan_msg, last_img_t_, imgl_ts);
        if (sync_result.scan_msg->points.empty()) {
            LOG(WARNING) << "No LiDAR points between two images!";
            return false;
        }
        
        // 更新时间戳
        sync_result.imgpkg->timestamp_sec_left = imgl_ts;
        sync_result.imgpkg->timestamp_sec_right = imgr_ts;
        last_img_t_ = imgl_ts;
    // ========== 场景2：禁用相机（以LiDAR时间戳为基准） ==========
    } else {
        sync_result.scan_msg = scan1_buf_.front();
        auto it = scan1_buf_.begin();
        it = scan1_buf_.erase(it);

        double start_point_ts = sync_result.scan_msg->points.front().t;
        double end_point_ts = sync_result.scan_msg->points.back().t;
        
        // 同步IMU数据（如果启用）
        if (param_.enable_imu) {
            if (imu_msg_buf_.empty()) {
                LOG(WARNING) << "IMU buffer is empty!";
                return false;
            }
            
            while (!imu_msg_buf_.empty()) {
                if (imu_msg_buf_.front()->header.stamp.toSec() > end_point_ts) break;
                if (imu_msg_buf_.front()->header.stamp.toSec() > start_point_ts) {
                    sync_result.imus.emplace_back(imu_msg_buf_.front());
                }
                imu_msg_buf_.pop();
            }
            
            if (sync_result.imus.empty()) {
                LOG(WARNING) << "No IMU data between two scans!";
                return false;
            }
        }
        
        // 同步轮速计数据（如果启用）
        if (param_.enable_wheel_encoder) {
            if (wheel_msg_buf_.empty()) {
                LOG(WARNING) << "Wheel encoder buffer is empty!";
                return false;
            }
            
            double latest_wheel_ts = wheel_msg_buf_.back().timestamp_;
            while (!wheel_msg_buf_.empty()) {
                if (wheel_msg_buf_.front().timestamp_ > end_point_ts) break;
                if (wheel_msg_buf_.front().timestamp_ > start_point_ts) {
                    sync_result.wheels.emplace_back(wheel_msg_buf_.front());
                }
                wheel_msg_buf_.pop();
            }
            
            if (sync_result.wheels.empty()) {
                LOG(WARNING) << "No wheel data between two scans!";
                return false;
            }
        }

        // 同步GNSS数据（如果启用）
        if (param_.enable_gnss) {
            sensor_msgs::NavSatFixPtr gnss = getClosestAndPop(end_point_ts);
            if(gnss){
                sync_result.gnss = gnss;
            }
        }
        
        // 处理第二个LiDAR缓存
        if (param_.enable_lidar2 ) {
            if(scan2_buf_.empty()){
                LOG(ERROR) << "Enable lidar2 while lidar2 data is empty!";
                return false;
            }
            for(auto it = scan2_buf_.begin(); it != scan2_buf_.end();){
                if((*it)->points.empty()){
                    LOG(WARNING)<<"scan2 is empty, should not happen!";
                    it = scan2_buf_.erase(it);
                }else{
                    if((*it)->points.front().t > sync_result.scan_msg->points.front().t) break;
                    if((*it)->points.back().t > sync_result.scan_msg->points.front().t) break;
                    if((*it)->points.back().t < sync_result.scan_msg->points.front().t){
                        it = scan2_buf_.erase(it);
                    }else{
                        it++;
                    }
                }
            }
            // scan2 can be empty
            for(int i = 0; i < scan2_buf_.size(); ++i){
                for(int j = 0; j < scan2_buf_[i]->points.size(); ++j){
                    PointXYZT xyzt = scan2_buf_[i]->points[j];
                    if(xyzt.t < end_point_ts && xyzt.t > start_point_ts){
                        sync_result.scan_msg->points.emplace_back(xyzt);
                    }
                    if(xyzt.t > end_point_ts){
                        break;
                    }
                }
            }
        }
    }
    
    return true;
}

void BaseNode::process_loop(){
    float hz = 20.0; 
    ros::Rate rate(hz);
    while(ros::ok()){
        TicToc tic;
        SyncResult synced_data;
        mtx_buf_.lock();
        bool get_sync_data = syncAllData(synced_data);
        mtx_buf_.unlock();
        convar_.notify_all();

        if(!get_sync_data){
            if(finish_add_data_){
                break;
            }else{
                rate.sleep();
                continue;
            }
        } 
        // LOG(INFO);

        if(!synced_data.scan_msg || synced_data.scan_msg->points.empty() ){
            rate.sleep();
            continue;
        }

        if(param_.enable_camera){
            if(synced_data.imgpkg->left_img.empty()){
                rate.sleep();
                continue;
            }
            if(param_.enable_right_camera && synced_data.imgpkg->right_img.empty()){
                rate.sleep();
                continue;
            }
        }

        if(param_.enable_imu){
            if(synced_data.imus.empty()){
                rate.sleep();
                continue;
            } 
        }

        if(param_.enable_wheel_encoder){
            if(synced_data.wheels.empty()){
                rate.sleep();
                continue;
            } 
        }

        mtx_lio_.lock();
        if(param_.enable_camera && param_.resize_image_scale > 0){
            int newcol = synced_data.imgpkg->left_img.cols * param_.resize_image_scale;
            int newrow = synced_data.imgpkg->left_img.rows * param_.resize_image_scale;
            cv::resize(synced_data.imgpkg->left_img, synced_data.imgpkg->left_img, cv::Size(newcol, newrow), 0, 0, cv::INTER_LINEAR);
            if(param_.enable_right_camera){
                cv::resize(synced_data.imgpkg->right_img, synced_data.imgpkg->right_img, cv::Size(newcol, newrow), 0, 0, cv::INTER_LINEAR);
            }
        }
        
        if(!initialized_ ){
            if(param_.enable_imu){
                if(!synced_data.imus.empty()) initialize(synced_data.imus);
            }else{
                initialize_without_imu(synced_data.scan_msg->points.back().t);
            }
        }

        if(!initialized_){
            mtx_lio_.unlock();
            continue;
        }

        // LOG(INFO)<<imus.back()->header.stamp.toSec()-scan_msg->points.back().t<<", "<<imus.back()->header.stamp.toSec()-img_left->header.stamp.toSec();
        // propagate
        if(param_.enable_imu){
            for(const auto& imu_msg: synced_data.imus){
                Eigen::Vector3d la(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                Eigen::Vector3d av(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                sad::IMU imu(imu_msg->header.stamp.toSec(), av, la);
                liomesher_->addImu(imu);
            }
        }

        if(param_.enable_wheel_encoder){
            for(const auto& we: synced_data.wheels){
                liomesher_->addWheelEncoder(we);
            }
        }

        if(param_.enable_camera){
            liomesher_->addImage(synced_data.imgpkg);
        }

        mtx_result_.lock();
        result_ = liomesher_->addPointCloud(synced_data.scan_msg);

        t_cost_total_ += tic.toc();
        // if(result_->frame_id > 0){
        //     LOG(INFO)<<"Front-end costs: "<<t_cost_total_ / result_->frame_id <<" ms.";
        // }
        
        
        // push tracking results to the backend
        CHECK(result_ != nullptr);
        if(param_.enable_gnss && synced_data.gnss){
            result_->gnss.reset(new sad::NaiveGNSS());
            *result_->gnss = NavSatFix2NaiveGNSS(synced_data.gnss);
        }
        if(param_.enable_camera){
            result_->image_raw = synced_data.imgpkg->left_img.clone();
        }
        posegraph_->AddTrackingResult(result_);
        mtx_result_.unlock();

        mtx_lio_.unlock();

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}