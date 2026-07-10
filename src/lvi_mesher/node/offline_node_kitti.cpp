#include <execution>
#include "offline_node_kitti.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

OfflineNodeKITTI::OfflineNodeKITTI(){}

OfflineNodeKITTI::~OfflineNodeKITTI(){
    if(liomesher_) liomesher_.release();
    if(posegraph_) posegraph_.release();
}


bool OfflineNodeKITTI::frontend_finished(){
    if(liomehser_thread_.joinable()){
        liomehser_thread_.join();
        job_finished_ = true;
        return true;
    }else{
        return false;
    }
}

bool OfflineNodeKITTI::backend_finished(){
    if(job_finished_){
        bool finish = posegraph_->finish();
        return finish;
    }
    return false;
}


//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    LOG(INFO)<<"====START lvimesher_offline_kitti_node====";
    ros::init(argc, argv, "lvimesher");

    ros::NodeHandle nh;
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    Parameter& param_=Parameter::GetInstance();
    param_.initParameter(nh);

    OfflineNodeKITTI lvimesher_offline;

    mkdir(param_.log_dir.c_str(), S_IRWXU);
    if(param_.enable_log_mesh){
        mkdir(param_.log_odom_mesh_dir.c_str(), S_IRWXU);
        mkdir(param_.log_slam_mesh_dir.c_str(), S_IRWXU);
    }
    if(param_.enable_log_pcd){
        mkdir(param_.log_pcd_dir.c_str(), S_IRWXU);
    }
   
    std::string dataset_folder, sequence_number, output_bag_file;
    dataset_folder = param_.dataset_folder;
    sequence_number = param_.seq;
    LOG(INFO)<<"Reading KITTI odometry at: "<<dataset_folder;
    LOG(INFO)<<"Processing sequence: "<<sequence_number;
    
    std::string timestamp_path = "data_odometry_color/dataset/sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);

    std::string ground_truth_path = "data_odometry_poses/dataset/poses/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);

    
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Quaterniond q_transform(R_transform);

    std::string line;
    std::size_t line_num = 0;

    while (std::getline(timestamp_file, line) && ros::ok())
    {
        float timestamp = stof(line);
        
        std::getline(ground_truth_file, line);
        std::stringstream pose_stream(line);
        std::string s;
        Eigen::Matrix<double, 3, 4> gt_pose;
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s);
            }
        }

        Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>());
        Eigen::Quaterniond q = q_transform * q_w_i;
        q.normalize();
        Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();

        // read lidar point cloud
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << "data_odometry_velodyne/dataset/sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());

        std::vector<Eigen::Vector3d> lidar_points;
        std::vector<float> lidar_intensities;
        pcl::PointCloud<pcl::PointXYZI> laser_cloud;
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            lidar_intensities.push_back(lidar_data[i+3]);

            pcl::PointXYZI point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            point.intensity = lidar_data[i + 3];
            laser_cloud.push_back(point);
        }

        sensor_msgs::PointCloud2Ptr laser_cloud_msg;
        laser_cloud_msg.reset(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(laser_cloud, *laser_cloud_msg);
        laser_cloud_msg->header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg->header.frame_id = "/camera_init";
        lvimesher_offline.pointCloudCallback(laser_cloud_msg);

        if(param_.enable_camera){
            std::stringstream left_image_path;
            left_image_path << dataset_folder <<"data_odometry_color/dataset/sequences/" + sequence_number + "/image_2/" << std::setfill('0') << std::setw(6) << line_num << ".png";
            cv::Mat left_image = cv::imread(left_image_path.str(), cv::IMREAD_COLOR);
            sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg->header, "bgr8", left_image).toImageMsg();
            lvimesher_offline.imgLeftCallback(image_left_msg);
            if(param_.enable_right_camera){
                std::stringstream right_image_path;
                right_image_path << dataset_folder << "data_odometry_color/dataset/sequences/" + sequence_number + "/image_3/" << std::setfill('0') << std::setw(6) << line_num << ".png";
                cv::Mat right_image = cv::imread(right_image_path.str(), cv::IMREAD_COLOR);
                sensor_msgs::ImagePtr image_right_msg = cv_bridge::CvImage(laser_cloud_msg->header, "bgr8", right_image).toImageMsg();
                lvimesher_offline.imgLeftCallback(image_right_msg);
            }
        }

        line_num ++;
        // r.sleep();
    }

    lvimesher_offline.set_finish_flag();
    // wait for processing
    ros::Rate r(5);
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