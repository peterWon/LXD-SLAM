

#include "glog/logging.h"
#include <sys/stat.h>
#include <sys/types.h>

#include "utility/parameter.h"
#include "utility/sensor_manager.h"

void Parameter::initParameter(ros::NodeHandle& nh){
    //<!--  read param  -->
    nh.param("lvimesher/grt_available", grt_available, false);
    nh.param("lvimesher/odom_available", odom_available, false);
    nh.param("lvimesher/read_offline_pcd", read_offline_pcd, false);
    nh.param("lvimesher/imu_feedback", imu_feedback, false);//? TO DO
    nh.param("lvimesher/dataset_folder", dataset_folder, std::string(""));
    nh.param("lvimesher/dataset", dataset, 6);
    nh.param("lvimesher/seq", seq, std::string(""));
    nh.param("lvimesher/max_steps", max_steps, 1);
    nh.param("lvimesher/save_mesh_map", save_mesh_map, false);

    //<!--  register param  -->-
    nh.param("lvimesher/range_max",  range_max,  100.0);
    nh.param("lvimesher/range_min",  range_min,  1.0);
    nh.param("lvimesher/range_unit", range_unit, 1.0);
    nh.param("lvimesher/register_times", register_times, 5);
    nh.param("lvimesher/cross_overlap", cross_overlap, false);
    nh.param("lvimesher/cross_cell_overlap_length", cross_cell_overlap_length, 0);
    nh.param("lvimesher/num_margin_old_cell", num_margin_old_cell, -1);
    nh.param("lvimesher/point2mesh", point2mesh, false);
    nh.param("lvimesher/residual_combination", residual_combination, true);
    //<!--  visualize parameter  -->
    nh.param("lvimesher/meshing_tsdf", meshing_tsdf, false);
    nh.param("lvimesher/full_cover", full_cover, true);
    nh.param("lvimesher/rescale_imu_data", rescale_imu_data, false);
    nh.param("lvimesher/lv_joint_eskf", lv_joint_eskf, true);
    nh.param("lvimesher/enable_vmap_constraint", enable_vmap_constraint, true);
    nh.param("lvimesher/visualisation_type", visualisation_type, 1);

    //<!--  gp param  -->
    nh.param("lvimesher/num_thread", num_thread, 1);
    nh.param("lvimesher/grid", grid, 1.0);
    nh.param("lvimesher/min_points_num_to_gp", min_points_num_to_gp, 8);
    nh.param("lvimesher/num_test",  num_test, 10);
    nh.param("lvimesher/voxel_size", voxel_size, 0.05);
    // voxel_size = grid*1.0/num_test;
    nh.param("lvimesher/voxel_size_for_resgister", voxel_size_for_resgister, 0.1);

    nh.param("lvimesher/variance_register", variance_register, 0.1);
    nh.param("lvimesher/variance_map_update", variance_map_update, 0.1);
    nh.param("lvimesher/variance_map_show", variance_map_show, 0.1);
    nh.param("lvimesher/variance_depth_association", variance_depth_association, 0.5);
    nh.param("lvimesher/variance_min", variance_min, 5.0);
    nh.param("lvimesher/variance_sensor", variance_sensor, 0.1);

    nh.param("lvimesher/test_param", test_param, 0.0);

    nh.param("lvimesher/enable_imu", enable_imu, false);
    nh.param("lvimesher/enable_wheel_encoder", enable_wheel_encoder, false);
    nh.param("lvimesher/enable_nhc", enable_nhc, true);
    nh.param("lvimesher/enable_camera", enable_camera, false);
    nh.param("lvimesher/enable_gnss", enable_gnss, false);
    nh.param("lvimesher/enable_right_camera", enable_right_camera, false);
    nh.param("lvimesher/enable_lidar2", enable_lidar2, false);

    nh.param("lvimesher/enable_deskew", enable_deskew, true);
    nh.param("lvimesher/enable_vl_rotation_guess", enable_vl_rotation_guess, true);
    nh.param("lvimesher/enable_opticflow_with_prediction", enable_opticflow_with_prediction, false);
    nh.param("lvimesher/marginalize_features", marginalize_features, false);
    nh.param("lvimesher/use_mature_points_in_eskf", use_mature_points_in_eskf, false);

    nh.param("lvimesher/enable_sliding_window_optimization", enable_sliding_window_optimization, false);
    nh.param("lvimesher/enable_mesh_texture", enable_mesh_texture, false);

    nh.param("lvimesher/imu_topic", imu_topic, std::string("/imu"));
    nh.param("lvimesher/wheel_encoder_topic", wheel_encoder_topic, std::string("/wheel_encoder"));
    nh.param("lvimesher/lidar1_type", lidar1_type, std::string("ouster"));
    nh.param("lvimesher/lidar2_type", lidar2_type, std::string("ouster"));
    nh.param("lvimesher/lidar1_topic", lidar1_topic, std::string("/points_1"));
    nh.param("lvimesher/lidar2_topic", lidar2_topic, std::string(""));
    nh.param("lvimesher/image_left_topic", image_left_topic, std::string("/image_left_topic"));
    nh.param("lvimesher/image_right_topic", image_right_topic, std::string(""));
    nh.param("lvimesher/gnss_topic", gnss_topic, std::string(""));
    
    nh.param("lvimesher/lidar1_time_offset", lidar1_time_offset, 0.0);
    nh.param("lvimesher/lidar2_time_offset", lidar2_time_offset, 0.0);
    nh.param("lvimesher/left_img_time_offset", left_img_time_offset, 0.0);
    nh.param("lvimesher/gnss_time_tolerance", gnss_time_tolerance, 0.05);
    nh.param("lvimesher/right_img_time_offset", right_img_time_offset, 0.0);
    nh.param("lvimesher/resize_image_scale", resize_image_scale, -1.0);
    nh.param("lvimesher/uv_motion_thresh", uv_motion_thresh, 10.0);
    nh.param("lvimesher/left_camera_model_path", left_camera_model_path, std::string(""));
    nh.param("lvimesher/right_camera_model_path", right_camera_model_path, std::string(""));
    nh.param("lvimesher/voc_file_path", voc_file_path, std::string(""));
    nh.param("lvimesher/brief_pattern_file_path", brief_pattern_file_path, std::string(""));
    nh.param("lvimesher/skip_image_frames", skip_image_frames, 3);
    nh.param("lvimesher/accelerometer_std", accelerometer_std, 0.1);
    nh.param("lvimesher/accelerometer_random_walk", accelerometer_random_walk, 0.001);
    nh.param("lvimesher/gyroscope_std", gyroscope_std, 0.1);
    nh.param("lvimesher/gyroscope_random_walk", gyroscope_random_walk, 0.001);
    nh.param("lvimesher/lidar_measurement_std", lidar_measurement_std, 0.01);
    nh.param("lvimesher/normal_uv_measurement_std", normal_uv_measurement_std, 0.03);
    nh.param("lvimesher/wheel_vel_std", wheel_vel_std, 0.3);
    nh.param("lvimesher/wheel_nhc_std", wheel_nhc_std, 0.1);
    nh.param("lvimesher/rot_prior_std", rot_prior_std, 0.01);
    nh.param("lvimesher/gravity", gravity, 9.8);
    nh.param("lvimesher/residual_threshold", residual_threshold, 0.5);
    nh.param("lvimesher/visual_residual_threshold", visual_residual_threshold, 10.0);
    nh.param("lvimesher/lambda_color_icp", lambda_color_icp, 0.968);
    nh.param("lvimesher/frames_for_submap", frames_for_submap, 100);

    nh.param("lvimesher/enable_scan_context", enable_scan_context, false);
    nh.param("lvimesher/enable_loop_orb3d", enable_loop_orb3d, false);
    nh.param("lvimesher/enable_extended_scan_context", enable_extended_scan_context, false);
    nh.param("lvimesher/extended_scan_context_threshold", extended_scan_context_threshold, 0.3);
    nh.param("lvimesher/loop_orb3d_threshold", loop_orb3d_threshold, 0.02);
    nh.param("lvimesher/enable_log_mesh", enable_log_mesh, false);
    nh.param("lvimesher/enable_log_pcd", enable_log_pcd, false);
    nh.param("lvimesher/enable_color_pcd", enable_color_pcd, false);
    nh.param("lvimesher/mask_filename", mask_filename, std::string(""));
    nh.param("lvimesher/bag_filename", bag_filename, std::string(""));
    nh.param("lvimesher/log_dir", log_dir, std::string("log"));
    nh.param("lvimesher/log_odom_mesh_dir", log_odom_mesh_dir, std::string("odom_mesh"));
    nh.param("lvimesher/log_slam_mesh_dir", log_slam_mesh_dir, std::string("slam_mesh"));
    nh.param("lvimesher/log_pcd_dir", log_pcd_dir, std::string("pcd"));

    // extrinsics
    Eigen::Matrix4d T_Body_IMU = Eigen::Matrix4d::Identity(); //IMU in Body frame
    Eigen::Matrix4d T_Body_Wheel = Eigen::Matrix4d::Identity(); 
    Eigen::Matrix4d T_Body_CamLeft = Eigen::Matrix4d::Identity(); 
    Eigen::Matrix4d T_Body_CamRight = Eigen::Matrix4d::Identity(); 
    Eigen::Matrix4d T_Body_Lidar1 = Eigen::Matrix4d::Identity(); 
    Eigen::Matrix4d T_Body_Lidar2 = Eigen::Matrix4d::Identity(); 
    Eigen::Matrix4d T_Body_GNSS = Eigen::Matrix4d::Identity();

    std::vector<double> extrinsics;
    CHECK(!lidar1_topic.empty());
    nh.param<std::vector<double>>("lvimesher/T_Body_Lidar1", extrinsics, std::vector<double>());
    CHECK(extrinsics.size() == 16);
    T_Body_Lidar1 = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extrinsics.data(), 4, 4);
    
    if(enable_imu){
        CHECK(!imu_topic.empty());
        nh.param<std::vector<double>>("lvimesher/T_Body_IMU", extrinsics, std::vector<double>());
        CHECK(extrinsics.size() == 16);
        T_Body_IMU = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extrinsics.data(), 4, 4);
    }
    if(enable_wheel_encoder){
        CHECK(!wheel_encoder_topic.empty());
        nh.param<std::vector<double>>("lvimesher/T_Body_Wheel", extrinsics, std::vector<double>());
        CHECK(extrinsics.size() == 16);
        T_Body_Wheel = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extrinsics.data(), 4, 4);
    }
    
    if(enable_camera){
        CHECK(!image_left_topic.empty());
        nh.param<std::vector<double>>("lvimesher/T_Body_CamLeft", extrinsics, std::vector<double>());
        CHECK(extrinsics.size() == 16);
        T_Body_CamLeft = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extrinsics.data(), 4, 4);
    }
    
    if(enable_right_camera){
        CHECK(!image_right_topic.empty());
        nh.param<std::vector<double>>("lvimesher/T_Body_CamRight", extrinsics, std::vector<double>());
        CHECK(extrinsics.size() == 16);
        T_Body_CamRight = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extrinsics.data(), 4, 4);
    }

    if(enable_lidar2){
        CHECK(!lidar2_topic.empty());
        nh.param<std::vector<double>>("lvimesher/T_Body_Lidar2", extrinsics, std::vector<double>());
        CHECK(extrinsics.size() == 16);
        T_Body_Lidar2 = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extrinsics.data(), 4, 4);
    }

    if(enable_gnss){
        CHECK(!gnss_topic.empty());
        nh.param<std::vector<double>>("lvimesher/T_Body_GNSS", extrinsics, std::vector<double>());
        CHECK(extrinsics.size() == 16);
        T_Body_GNSS = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extrinsics.data(), 4, 4);
    }

    // add sensors
    // 添加所有传感器，通过配置参数来实现body系的自动选择
    SensorExtrinsicsManager::GetInstance().SetSensorToParentExtrinsic(SensorType::IMU, Eigen::Isometry3d(T_Body_IMU));
    SensorExtrinsicsManager::GetInstance().SetSensorToParentExtrinsic(SensorType::WHEEL, Eigen::Isometry3d(T_Body_Wheel));
    SensorExtrinsicsManager::GetInstance().SetSensorToParentExtrinsic(SensorType::CAMERA_LEFT, Eigen::Isometry3d(T_Body_CamLeft));
    SensorExtrinsicsManager::GetInstance().SetSensorToParentExtrinsic(SensorType::CAMERA_RIGHT, Eigen::Isometry3d(T_Body_CamRight));
    SensorExtrinsicsManager::GetInstance().SetSensorToParentExtrinsic(SensorType::LIDAR1, Eigen::Isometry3d(T_Body_Lidar1));
    SensorExtrinsicsManager::GetInstance().SetSensorToParentExtrinsic(SensorType::LIDAR2, Eigen::Isometry3d(T_Body_Lidar2));
    SensorExtrinsicsManager::GetInstance().SetSensorToParentExtrinsic(SensorType::GNSS, Eigen::Isometry3d(T_Body_GNSS));
    // LOG(INFO)<<"Body frame: "<<SensorExtrinsicsManager::GetInstance().GetBodyFrame();

    eigen_1 = 48;
    eigen_2 = 0.95;
    eigen_3 = 0.2;
    converge_thr = 0.00001;
}