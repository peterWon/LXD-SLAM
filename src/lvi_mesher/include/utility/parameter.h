#ifndef SLAMESH_PARAMETER_HPP
#define SLAMESH_PARAMETER_HPP

#include <string>
#include <vector>
#include "eigen3/Eigen/Core"
#include "ros/ros.h"
#include "utility/pointMatrix.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>

class Parameter{
    //algorithm parameter
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    static Parameter& GetInstance(){
        static Parameter param;
        return param;
    }
    
    void initParameter(ros::NodeHandle& nh);

    bool enable_imu = false;
    bool enable_wheel_encoder = false;
    bool enable_camera = false;
    bool enable_gnss = false;

    bool enable_nhc = true;

    
    bool enable_deskew = true;
    bool enable_vl_rotation_guess = true;

    bool enable_scan_context = false;
    bool enable_loop_orb3d = false;
    bool enable_extended_scan_context = false;

    double extended_scan_context_threshold = 0.3;
    double loop_orb3d_threshold = 0.02;

    bool enable_log_mesh = false;
    bool enable_log_pcd = false;

    bool enable_sliding_window_optimization = false;
    bool enable_color_pcd = false;
    bool enable_mesh_texture = false;

    bool enable_right_camera = false;
    bool enable_lidar2 = false;

    bool enable_opticflow_with_prediction = false;

    bool marginalize_features = false;
    bool use_mature_points_in_eskf = false;

    double resize_image_scale = -1.;

    std::string mask_filename = "";
    std::string log_pcd_dir = "./pcd";

    std::string imu_topic="/imu";
    std::string wheel_encoder_topic="/wheel_encoder";

    std::string lidar1_topic="/points_1";
    std::string lidar2_topic="";
    
    std::string image_left_topic="/image_left";
    std::string image_right_topic="";
    std::string gnss_topic="/gnss";
    // std::string feature_left_topic="/feature_left";
    // std::string feature_right_topic="/feature_right";
    
    double left_img_time_offset = 0.0;
    double right_img_time_offset = 0.0;
    double lidar1_time_offset = 0.0;
    double lidar2_time_offset = 0.0;
    double gnss_time_tolerance = 0.05;

    double uv_motion_thresh = 10.0;

    std::string left_camera_model_path="config/camera_models/left_camera.yaml";
    std::string right_camera_model_path="config/camera_models/right_camera.yaml";
    std::string voc_file_path = "";
    std::string brief_pattern_file_path = "";
    
    std::string bag_filename = "rosbag.bag";
    std::string log_dir = "log/";
    std::string log_odom_mesh_dir = "log/odom_mesh/";
    std::string log_slam_mesh_dir = "log/odom_slam/";

    // noises
    double accelerometer_std = 0.1;
    double accelerometer_random_walk = 0.001;
    double gyroscope_std = 0.1;
    double gyroscope_random_walk = 0.001;
    double lidar_measurement_std = 0.01;
    double normal_uv_measurement_std = 0.001;

    double range_err = 0.03;
    double degree_err = 0.03;

    double wheel_vel_std = 0.3;
    double wheel_nhc_std = 0.1;

    double rot_prior_std = 0.01;

    double gravity = 9.8;

    double lambda_color_icp = 0.968;

    double residual_threshold = 0.5;//meter, lidar
    double visual_residual_threshold = 10.0;//pixel

    

    int skip_image_frames = 3;
    

    int    max_steps{10000}, register_times, num_test, min_points_num_to_gp, num_thread, cross_cell_overlap_length, dataset;
    double range_max, range_min, range_unit;
    double variance_register, variance_map_update, variance_map_show, variance_min, variance_sensor, variance_depth_association;
    double grid, voxel_size, converge_thr;

    double voxel_size_for_resgister = 0.1;

    double correction_x{0}, correction_y{0}, correction_z{0},
    correction_roll_degree{0}, correction_pitch_degree{0}, correction_yaw_degree{0};
    double test_param;
    double eigen_1, eigen_2, eigen_3;//PCA

    std::string dataset_folder, seq;
    bool three_dir;//features fixed
    bool odom_available, read_offline_pcd, cross_overlap, grt_available, imu_feedback,
            meshing_tsdf, full_cover, save_raw_point_clouds, point2mesh{true},
            residual_combination{true}, save_mesh_map;
    int visualisation_type;
    int num_margin_old_cell;

    int frames_for_submap;
    double bias_acc_x, bias_acc_y;
    
    std::string lidar1_type;
    std::string lidar2_type;

    bool rescale_imu_data = false;

    bool lv_joint_eskf = true;
    bool enable_vmap_constraint = true;
    
    // PoseGraph Parameters
    // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 
    double sc_dist_threshold = 0.2;
    double sc_maximum_radius = 80;

    double keyframe_meter_gap = 2.0;
    double keyframe_degree_gap = 10.0;
    
    bool publish_mesh = true;

    // for posegraph and visualization
    std::string frontend_rgt_points_name = "/frontend/registered_points";
    std::string associate_depth_points_name = "/frontend/associate_depth_points";
    std::string frontend_full_points_name = "/frontend/full_points_deskewed";
    std::string frontend_mesh_now_name = "/frontend/mesh_now";
    std::string frontend_mesh_full_name = "/frontend/mesh_full";
    std::string frontend_vtxcolor_full_name = "/frontend/mesh_vtxcolor_full";

    std::string frontend_map_vertices_glb_name = "/frontend/map_vertices_glb";
    std::string frontend_map_vertices_now_name = "/frontend/map_vertices_now";
    std::string frontend_overlap_point_glb_name = "/frontend/overlap_point_glb";
    std::string frontend_overlap_point_now_name = "/frontend/overlap_point_now";
    
    std::string frontend_odometry_name = "/frontend/base_in_map";
    std::string frontend_path_odom_name = "/frontend/path";
    std::string frontend_track_image = "/frontend/track_image";

    std::string backend_odom_aft_pgo_name = "/backend/odom_aft_pgo";
    std::string backend_odom_repub_name = "/backend/odom_repub";
    std::string backend_path_aft_pgo_name = "/backend/path_aft_pgo";
    std::string backend_gnss_path_name = "/backend/gnss_path";
    std::string backend_map_aft_pgo_name = "/backend/map_aft_pgo";
    std::string backend_loop_scan_local_name = "/backend/loop_scan_local";
    std::string backend_loop_submap_local_name = "/backend/loop_submap_local";
private:
    Parameter(){};
    ~Parameter(){};
    Parameter(const Parameter &pmt)=delete;
    const Parameter &operator=(const Parameter &pmt)=delete;
};


#endif

