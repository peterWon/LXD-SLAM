#include <execution>
#include "offline_node.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// 新增：Velodyne 解包相关的头文件
#include <velodyne_msgs/VelodyneScan.h>
#include <velodyne_pointcloud/rawdata.h>
#include <velodyne_pointcloud/pointcloudXYZIRT.h>
#include <pcl_conversions/pcl_conversions.h>

namespace ros_cv{
// 函数：将 CompressedImagePtr 转为 ImagePtr
sensor_msgs::ImagePtr compressedImageToImage(const sensor_msgs::CompressedImagePtr& compressed_msg){
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
            compressed_msg,
            sensor_msgs::image_encodings::BGR8  
        );

        cv_bridge::CvImage ros_image;
        ros_image.header = compressed_msg->header;  
        ros_image.encoding = sensor_msgs::image_encodings::BGR8;  
        ros_image.image = cv_ptr->image;  

        return ros_image.toImageMsg();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception while converting compressed image to image: %s", e.what());
        return nullptr;
    }
}
}

OfflineNode::OfflineNode(ros::NodeHandle& nh){

}

OfflineNode::~OfflineNode(){
    if(liomesher_) liomesher_.release();
    if(posegraph_) posegraph_.release();
}

bool OfflineNode::frontend_finished(){
    if(liomehser_thread_.joinable()){
        liomehser_thread_.join();
        mtx_flag_.lock();
        job_finished_ = true;
        mtx_flag_.unlock();
        return true;
    }else{
        return false;
    }
}

bool OfflineNode::backend_finished(){
    if(job_finished_){
        bool finish = posegraph_->finish();
        return finish;
    }
    return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    LOG(INFO)<<"====START lvimesher_offline_node====";
    ros::init(argc, argv, "lvimesher");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~"); // 用于向 Velodyne 解析器传递模型和标定参数
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    ros::Rate r(5);

    Parameter& param_=Parameter::GetInstance();
    param_.initParameter(nh);

    OfflineNode lvimesher_offline(nh);

    // 初始化 Velodyne 解包器
    // 如果系统中有多个不同型号的雷达，可以实例化多个 raw_data_parser
    // 填入你之前在 launch 文件中的参数，这是计算单点时间戳和标定的关键
    private_nh.setParam("model", "32E"); // NCLT 数据集是 HDL-32E
    private_nh.setParam("calibration", "/opt/ros/noetic/share/velodyne_pointcloud/params/32db.yaml");
    private_nh.setParam("min_range", 0.4);
    private_nh.setParam("max_range", 130.0);
    private_nh.setParam("view_direction", 3.1415926);
    private_nh.setParam("view_width", 5.4769);
    velodyne_rawdata::RawData raw_data_parser1;
    if (!raw_data_parser1.setup(private_nh)) {
        LOG(ERROR) << "Failed to setup velodyne raw data parser!";
    } else {
        LOG(INFO) << "Velodyne RawData parser initialized.";
    }
    raw_data_parser1.setParameters(0.4, 130.0, 0.0, 2.0 * M_PI);

    velodyne_rawdata::RawData raw_data_parser2;
    if (param_.enable_lidar2) {
        // 如果 lidar2 的参数不同（例如通过 ~lidar2/model 传入），可以按需调整参数读取域
        if (!raw_data_parser2.setup(private_nh)) {
            LOG(ERROR) << "Failed to setup velodyne raw data parser 2!";
        }
    }
    raw_data_parser2.setParameters(0.4, 130.0, 0.0, 2.0 * M_PI);

    mkdir(param_.log_dir.c_str(), S_IRWXU);
    if(param_.enable_log_mesh){
        mkdir(param_.log_odom_mesh_dir.c_str(), S_IRWXU);
        mkdir(param_.log_slam_mesh_dir.c_str(), S_IRWXU);
    }
    if(param_.enable_log_pcd){
        mkdir(param_.log_pcd_dir.c_str(), S_IRWXU);
    }
    
    rosbag::Bag bag;
    std::string bag_filename = param_.bag_filename;
    LOG(INFO)<<bag_filename;
    bag.open(bag_filename);
    if(!bag.isOpen()){
        LOG(ERROR)<<"Failed to open "<<bag_filename;
    }  

    std::vector<std::string> topics;
    topics.push_back(param_.lidar1_topic);
    if(param_.enable_lidar2){
        topics.push_back(param_.lidar2_topic);
    }
    if(param_.enable_imu){
        topics.push_back(param_.imu_topic);
    }
    if(param_.enable_camera){
        topics.push_back(param_.image_left_topic);
        std::string compressed_topic = param_.image_left_topic + "/compressed";
        topics.push_back(compressed_topic);
    }
    if(param_.enable_right_camera){
        topics.push_back(param_.image_right_topic);
        std::string compressed_topic = param_.image_right_topic + "/compressed";
        topics.push_back(compressed_topic);
    }
    if(param_.enable_wheel_encoder){
        topics.push_back(param_.wheel_encoder_topic);
    }
    if(param_.enable_gnss){
        topics.push_back(param_.gnss_topic);
    }
    
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    LOG(INFO)<<"Enabled topics: ";
    for(auto& t: topics){
        LOG(INFO) << t;
    }
    
    for(rosbag::MessageInstance m: view){
        if(param_.enable_imu){
            sensor_msgs::ImuPtr imu = m.instantiate<sensor_msgs::Imu>();
            if (imu != nullptr){
                lvimesher_offline.imuCallback(imu);
                continue;
            }
        }
        
        // -------------------------------------------------------------------
        // 适配 ROS Noetic 的原生 Velodyne 解包逻辑
        // -------------------------------------------------------------------
        velodyne_msgs::VelodyneScan::ConstPtr velodyne_scan = m.instantiate<velodyne_msgs::VelodyneScan>();
        if(velodyne_scan != nullptr) {
            std::string topic_name = m.getTopic();

            const double v_max_range = 130.0; 
            const double v_min_range = 0.4;   

            if(topic_name == param_.lidar1_topic){
                // 【核心修复】：必须传入雷达真实的 numLasers 和 scansPerPacket 预分配内存
                velodyne_pointcloud::PointcloudXYZIRT container(
                    v_max_range, 
                    v_min_range, 
                    velodyne_scan->header.frame_id, 
                    velodyne_scan->header.frame_id, 
                    raw_data_parser1.scansPerPacket()    // 动态获取：对应型号的单包扫描数
                );
                
                container.setup(velodyne_scan);

                for (size_t i = 0; i < velodyne_scan->packets.size(); ++i) {
                    raw_data_parser1.unpack(velodyne_scan->packets[i], container, velodyne_scan->header.stamp);
                }
                sensor_msgs::PointCloud2Ptr pc2_msg(new sensor_msgs::PointCloud2(container.finishCloud()));
                pc2_msg->header.stamp = velodyne_scan->header.stamp;
                lvimesher_offline.pointCloudCallback(pc2_msg);
            }
            else if(param_.enable_lidar2 && topic_name == param_.lidar2_topic){
                // 对于二号雷达，同样需要使用 parser2 获取对应参数
                velodyne_pointcloud::PointcloudXYZIRT container(
                    v_max_range, 
                    v_min_range, 
                    velodyne_scan->header.frame_id, 
                    velodyne_scan->header.frame_id, 
                    raw_data_parser2.scansPerPacket()
                );
                
                container.setup(velodyne_scan);

                for (size_t i = 0; i < velodyne_scan->packets.size(); ++i) {
                    raw_data_parser2.unpack(velodyne_scan->packets[i], container, velodyne_scan->header.stamp);
                }
                
                sensor_msgs::PointCloud2Ptr pc2_msg(new sensor_msgs::PointCloud2(container.finishCloud()));
                pc2_msg->header.stamp = velodyne_scan->header.stamp;
                lvimesher_offline.pointCloudCallback2(pc2_msg);
            }
            continue;
        }

        // 处理已解析的 PointCloud2 (如果 bag 里已经是解析好的)
        sensor_msgs::PointCloud2Ptr pc = m.instantiate<sensor_msgs::PointCloud2>();
        if(pc != nullptr){
            std::string topic_name = m.getTopic();
            if(topic_name == param_.lidar1_topic){
                lvimesher_offline.pointCloudCallback(pc);
            }else if(param_.enable_lidar2 && topic_name == param_.lidar2_topic){
                lvimesher_offline.pointCloudCallback2(pc);
            }
            continue;
        }
        
        livox_ros_driver::CustomMsg::ConstPtr livox_pc = m.instantiate<livox_ros_driver::CustomMsg>();
        if(livox_pc != nullptr){
            std::string topic_name = m.getTopic();
            if(topic_name==param_.lidar1_topic){
                lvimesher_offline.livoxCallback(livox_pc);
            }else if(param_.enable_lidar2 && topic_name==param_.lidar2_topic){
                lvimesher_offline.livoxCallback2(livox_pc);
            }
            continue;
        }
        
        if(param_.enable_camera){
            std::string topic_name = m.getTopic();
            sensor_msgs::ImagePtr img = m.instantiate<sensor_msgs::Image>();
            if(img != nullptr){
                if(topic_name == param_.image_left_topic){
                    lvimesher_offline.imgLeftCallback(img);
                }else if(param_.enable_right_camera && topic_name == param_.image_right_topic){
                    lvimesher_offline.imgRightCallback(img);
                }
                continue;
            }

            sensor_msgs::CompressedImagePtr comp_img = m.instantiate<sensor_msgs::CompressedImage>();
            if(comp_img != nullptr){
                if(topic_name.find(param_.image_left_topic) != std::string::npos){
                    lvimesher_offline.imgLeftCallback(ros_cv::compressedImageToImage(comp_img));
                }else if(param_.enable_right_camera && topic_name.find(param_.image_right_topic) != std::string::npos){
                    lvimesher_offline.imgRightCallback(ros_cv::compressedImageToImage(comp_img));
                }else{
                    LOG(WARNING)<<"Used compressed image, Check image topic name in offline mode.";
                }
                continue;
            }
        }

        if(param_.enable_wheel_encoder){
            sensor_msgs::JointStatePtr wheel_msg = m.instantiate<sensor_msgs::JointState>();
            if(wheel_msg != nullptr){
                lvimesher_offline.wheelEncoderCallback(wheel_msg);
                continue;
            }
        }
        if(param_.enable_gnss){
            sensor_msgs::NavSatFixPtr gnss_msg = m.instantiate<sensor_msgs::NavSatFix>();
            if(gnss_msg != nullptr){
                lvimesher_offline.gnssCallback(gnss_msg);
                continue;
            }
        }
    }
    bag.close();
    LOG(INFO)<<"Rosbag closed.";

    lvimesher_offline.set_finish_flag();
    while(ros::ok()){
        bool ff = lvimesher_offline.frontend_finished();
        if(ff){
            LOG(INFO)<<"Frontend finished.";
        }

        bool bf = lvimesher_offline.backend_finished();
        if (bf){
            LOG(INFO)<<"Backend finished.";
            ros::shutdown();
            break;
        }        
        r.sleep();
    }
    LOG(INFO)<<"DONE! Results saved in "<<param_.log_dir;
    ros::shutdown();
    return 0;
}