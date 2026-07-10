
#include "posegraph.h"
#include "glog/logging.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include "utility/parameter.h"
#include <gtsam/inference/Symbol.h>
#include "utility/sensor_manager.h"

// #include "BoW3D/BoW3D.h"

using symbol_shorthand::N;
using symbol_shorthand::S;

PoseGraph::PoseGraph(){
    laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
    laserCloudMapAfterPGO.reset(new pcl::PointCloud<PointType>());
    laserCloudMapPGO.reset(new pcl::PointCloud<PointType>());

    keyframeMeterGap = Parameter::GetInstance().keyframe_meter_gap;
    keyframeDegGap = Parameter::GetInstance().keyframe_degree_gap;
    keyframeRadGap = keyframeDegGap * 180. / 3.1415926;
    
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(Parameter::GetInstance().sc_dist_threshold);
    scManager.setMaximumRadius(Parameter::GetInstance().sc_maximum_radius);

    scManagerSubmap.setSCdistThres(Parameter::GetInstance().sc_dist_threshold);
    scManagerSubmap.setMaximumRadius(Parameter::GetInstance().sc_maximum_radius);

    float filter_size = 0.4; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

    double mapVizFilterSize = 0.4;
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

    if(Parameter::GetInstance().enable_scan_context){
        sc_icp_thread_ = std::thread(&PoseGraph::process_sc_icp, this);
    }
    if(Parameter::GetInstance().enable_extended_scan_context){
        esc_icp_thread_ = std::thread(&PoseGraph::process_esc_icp, this);
    }
    if(Parameter::GetInstance().enable_loop_orb3d){
        orb3d_matcher_thread_ = std::thread(&PoseGraph::process_orb3d_matcher, this);
    }
    
    posegraph_slam_thread_ = std::thread(&PoseGraph::process_pg, this);
    isam_update_thread_ = std::thread(&PoseGraph::process_isam, this);

    if(Parameter::GetInstance().enable_camera){
        left_camera_ = CameraFactory::instance()->generateCameraFromYamlFile(
            Parameter::GetInstance().left_camera_model_path);
    }
}

PoseGraph::~PoseGraph(){
    if(posegraph_slam_thread_.joinable()) posegraph_slam_thread_.join();
    if(Parameter::GetInstance().enable_scan_context && sc_icp_thread_.joinable()) sc_icp_thread_.join();
    if(Parameter::GetInstance().enable_extended_scan_context && esc_icp_thread_.joinable()) esc_icp_thread_.join();
    if(isam_update_thread_.joinable()) isam_update_thread_.join();
    if(Parameter::GetInstance().enable_loop_orb3d && orb3d_matcher_thread_.joinable()) orb3d_matcher_thread_.join();
    if(isam) delete isam;    
}



std::vector<Pose6D> PoseGraph::getUpdatedKeyframePoses(){
    std::vector<Pose6D> res;
    mBuf.lock();
    // res = keyframePosesUpdated;
    res = allFramePosesUpdated;
    mBuf.unlock();
    return res;
}

std::vector<Pose6D> PoseGraph::getGPSPath(){
    std::vector<Pose6D> res;
    mBuf.lock();
    for(int i = 0; i < gnss_positions_.size(); i++){
        Pose6D gnssi;
        Vec3d xyz = gnss_positions_[i].second;
        gnssi.x = xyz(0);
        gnssi.y = xyz(1);
        gnssi.z = xyz(2);
        res.push_back(gnssi);
    }    
    mBuf.unlock();
    return res;
}


void PoseGraph::AddTrackingResult(const std::shared_ptr<TrackingResult>& result){
    mBuf.lock();
    tracked_data_.push(result);
    for(const auto& submap: result->active_submaps){
        auto itr = submaps_.find(submap->get_id());
        if(itr == submaps_.end()){
            submaps_.insert({submap->get_id(), submap});
        }
    }    
    mBuf.unlock();
}

bool PoseGraph::finish(){
    mBuf.lock();
    bool finished = (!allFramePoses.empty()) && tracked_data_.empty() && scLoopICPBuf.empty() && scLoopOrb3DBuf.empty();
    mBuf.unlock();
    if(finished){
        mtxPosegraph.lock();
        // TODO(wz): add constraints of the last submap that may not finished.
        runISAM2opt();
        runISAM2opt();
        mtxPosegraph.unlock();
        
        job_finished_ = true;
        if(posegraph_slam_thread_.joinable()){
            posegraph_slam_thread_.join();
            LOG(INFO)<<"Exit posegraph_slam_thread";
        } 
        if(Parameter::GetInstance().enable_scan_context && sc_icp_thread_.joinable()){
            sc_icp_thread_.join();
            LOG(INFO)<<"Exit sc_icp_thread";
        }
        if(Parameter::GetInstance().enable_extended_scan_context && esc_icp_thread_.joinable()){
            esc_icp_thread_.join();
            LOG(INFO)<<"Exit esc_icp_thread";
        }
        if(Parameter::GetInstance().enable_loop_orb3d && orb3d_matcher_thread_.joinable()){
            isam_update_thread_.join();
            LOG(INFO)<<"Exit visual loop thread";
        }
        if(isam_update_thread_.joinable()){
            isam_update_thread_.join();
            LOG(INFO)<<"Exit isam_update_thread";
        }

        // save results.
        save_node_poses_viral_format();
        save_submap_poses_viral_format();

        save_node_poses_kitti_format();
        save_submap_poses_kitti_format();

        if(Parameter::GetInstance().enable_gnss){
            save_gnss_positions();
        }
        
        if(Parameter::GetInstance().enable_log_mesh){
            save_submap_meshes();
        }
        
        return true;
    }else{
        return false;
    }
}

void PoseGraph::save_node_poses_viral_format(){
    std::string logdir = Parameter::GetInstance().log_dir;
    std::string filename_odom = logdir + "/" + "node_odom.viral";
    std::string filename_slam = logdir + "/" + "node_slam.viral";

    ofstream ofs_odom(filename_odom);
    if(!ofs_odom){
        LOG(ERROR)<<"Failed to log to "<<filename_odom;
        return;
    }

    ofstream ofs_slam(filename_slam);
    if(!ofs_slam){
        LOG(ERROR)<<"Failed to log to "<<filename_slam;
        return;
    }
    uint32_t seq = 0;
    ofs_odom << "\%time,field.header.seq,field.header.stamp,field.pose.position.x,field.pose.position.y,field.pose.position.z,field.pose.orientation.x,field.pose.orientation.y,field.pose.orientation.z,field.pose.orientation.w\n";
    ofs_slam << "\%time,field.header.seq,field.header.stamp,field.pose.position.x,field.pose.position.y,field.pose.position.z,field.pose.orientation.x,field.pose.orientation.y,field.pose.orientation.z,field.pose.orientation.w\n";
    for(int i = 0; i < allFramePoses.size(); ++i){
        const auto& pose = allFramePoses[i];
        Eigen::AngleAxisd yaw(pose.yaw, Vector3d::UnitZ()); // Rotation about the Z-axis (yaw)
        Eigen::AngleAxisd pitch(pose.pitch, Vector3d::UnitY()); // Rotation about the Y-axis (pitch)
        Eigen::AngleAxisd roll(pose.roll, Vector3d::UnitX()); // Rotation about the X-axis (roll)
        Quaterniond q = yaw * pitch * roll; // Order is important!

        ofs_odom << std::to_string(allFrameTimes[i]) <<"," << seq++ 
            << "," << std::to_string(allFrameTimes[i]) << ","
            << pose.x << ","<< pose.y << ","<< pose.z << ","
            << q.x() << ","<< q.y() << ","<< q.z() << ","<< q.w() << "\n";
        
        const auto& pose_slam = allFramePosesUpdated[i];
        Eigen::AngleAxisd yaw_slam(pose_slam.yaw, Vector3d::UnitZ());
        Eigen::AngleAxisd pitch_slam(pose_slam.pitch, Vector3d::UnitY());
        Eigen::AngleAxisd roll_slam(pose_slam.roll, Vector3d::UnitX());
        Quaterniond q_slam = yaw_slam * pitch_slam * roll_slam;
        ofs_slam << std::to_string(allFrameTimes[i]) <<"," << seq 
            << "," << std::to_string(allFrameTimes[i]) << ","
            << pose_slam.x << ","<< pose_slam.y << ","<< pose_slam.z << ","
            << q_slam.x() << ","<< q_slam.y() << ","<< q_slam.z() << ","<< q_slam.w() << "\n";
    }
    ofs_odom.close();
    ofs_slam.close();
}

void PoseGraph::save_node_poses_kitti_format(){
    std::string logdir = Parameter::GetInstance().log_dir;
    std::string filename_odom = logdir + "/" + "node_odom.kitti";
    std::string filename_slam = logdir + "/" + "node_slam.kitti";

    ofstream ofs_odom(filename_odom);
    if(!ofs_odom){
        LOG(ERROR)<<"Failed to log to "<<filename_odom;
        return;
    }

    ofstream ofs_slam(filename_slam);
    if(!ofs_slam){
        LOG(ERROR)<<"Failed to log to "<<filename_slam;
        return;
    }
    uint32_t seq = 0;
    for(int i = 0; i < allFramePoses.size(); ++i){
        const auto& pose = allFramePoses[i];
        Eigen::AngleAxisd yaw(pose.yaw, Vector3d::UnitZ()); // Rotation about the Z-axis (yaw)
        Eigen::AngleAxisd pitch(pose.pitch, Vector3d::UnitY()); // Rotation about the Y-axis (pitch)
        Eigen::AngleAxisd roll(pose.roll, Vector3d::UnitX()); // Rotation about the X-axis (roll)
        Quaterniond q = yaw * pitch * roll; // Order is important!
        q.normalize();
        Eigen::Matrix3d R = q.toRotationMatrix();

        ofs_odom << R(0,0) << " "<< R(0,1) << " "<< R(0,2) << " " <<pose.x << " "
                 << R(1,0) << " "<< R(1,1) << " "<< R(1,2) << " " <<pose.y << " "
                 << R(2,0) << " "<< R(2,1) << " "<< R(2,2) << " " <<pose.z << "\n";
        
        const auto& pose_slam = allFramePosesUpdated[i];
        Eigen::AngleAxisd yaw_slam(pose_slam.yaw, Vector3d::UnitZ());
        Eigen::AngleAxisd pitch_slam(pose_slam.pitch, Vector3d::UnitY());
        Eigen::AngleAxisd roll_slam(pose_slam.roll, Vector3d::UnitX());
        Quaterniond q_slam = yaw_slam * pitch_slam * roll_slam;
        Eigen::Matrix3d R_slam = q_slam.toRotationMatrix();
        ofs_slam << R_slam(0,0) << " "<< R_slam(0,1) << " "<< R_slam(0,2) << " " <<pose_slam.x << " "
                 << R_slam(1,0) << " "<< R_slam(1,1) << " "<< R_slam(1,2) << " " <<pose_slam.y << " "
                 << R_slam(2,0) << " "<< R_slam(2,1) << " "<< R_slam(2,2) << " " <<pose_slam.z << "\n";
    }
    ofs_odom.close();
    ofs_slam.close();
}

void PoseGraph::save_gnss_positions(){
    if(gnss_positions_.empty()) return;
    std::string logdir = Parameter::GetInstance().log_dir;
    std::string filename_gnss = logdir + "/" + "node_gnss.txt";

    ofstream ofs(filename_gnss);
    if(!ofs){
        LOG(ERROR)<<"Failed to log to "<<filename_gnss;
        return;
    }

    for(int i = 0; i < gnss_positions_.size(); ++i){
        const auto& gnss = gnss_positions_[i];
        ofs << gnss.first << " "<< gnss.second(0) << " "<< gnss.second(1) << " " <<gnss.second(2) << "\n";
    }
    ofs.close();
}

void PoseGraph::save_submap_poses_viral_format(){
    std::string logdir = Parameter::GetInstance().log_dir;
    std::string filename_odom = logdir + "/" + "submap_odom.viral";
    std::string filename_slam = logdir + "/" + "submap_slam.viral";

    ofstream ofs_odom(filename_odom);
    if(!ofs_odom){
        LOG(ERROR)<<"Failed to log to "<<filename_odom;
        return;
    }

    ofstream ofs_slam(filename_slam);
    if(!ofs_slam){
        LOG(ERROR)<<"Failed to log to "<<filename_slam;
        return;
    }
    uint32_t seq = 0;
    ofs_odom << "\%time,field.header.seq,field.header.stamp,field.pose.position.x,field.pose.position.y,field.pose.position.z,field.pose.orientation.x,field.pose.orientation.y,field.pose.orientation.z,field.pose.orientation.w\n";
    ofs_slam << "\%time,field.header.seq,field.header.stamp,field.pose.position.x,field.pose.position.y,field.pose.position.z,field.pose.orientation.x,field.pose.orientation.y,field.pose.orientation.z,field.pose.orientation.w\n";
    for(int i = 0; i < submaps_.size(); ++i){
        size_t sid = i;
        Mat4d pose = submaps_[sid]->pose_odom_;
        Vec4d p = pose.col(3);
        Eigen::Quaterniond q = Eigen::Quaterniond(pose.block<3,3>(0,0));

        ofs_odom << std::to_string(i) <<"," << seq++ 
            << "," << std::to_string(i) << ","
            << p(0) << ","<< p(1) << ","<< p(2) << ","
            << q.x() << ","<< q.y() << ","<< q.z() << ","<< q.w() << "\n";

        pose = submaps_[sid]->pose_slam_;
        p = pose.col(3);
        q = Eigen::Quaterniond(pose.block<3,3>(0,0));

        ofs_slam << std::to_string(i) <<"," << seq 
            << "," << std::to_string(i) << ","
            << p(0) << ","<< p(1) << ","<< p(2) << ","
            << q.x() << ","<< q.y() << ","<< q.z() << ","<< q.w() << "\n";        
    }
    ofs_odom.close();
    ofs_slam.close();
}

void PoseGraph::save_submap_poses_kitti_format(){
    std::string logdir = Parameter::GetInstance().log_dir;
    std::string filename_odom = logdir + "/" + "submap_odom.kitti";
    std::string filename_slam = logdir + "/" + "submap_slam.kitti";

    ofstream ofs_odom(filename_odom);
    if(!ofs_odom){
        LOG(ERROR)<<"Failed to log to "<<filename_odom;
        return;
    }

    ofstream ofs_slam(filename_slam);
    if(!ofs_slam){
        LOG(ERROR)<<"Failed to log to "<<filename_slam;
        return;
    }
    uint32_t seq = 0;
    
    for(int i = 0; i < submaps_.size(); ++i){
        size_t sid = i;
        Mat4d pose = submaps_[sid]->pose_odom_;
        Vec4d p = pose.col(3);
        Eigen::Quaterniond q = Eigen::Quaterniond(pose.block<3,3>(0,0));
        Eigen::Matrix3d R = q.toRotationMatrix();

        ofs_odom << pose(0,0) << " "<< pose(0,1) << " "<< pose(0,2) << " " <<pose(0, 3) << " "
                 << pose(1,0) << " "<< pose(1,1) << " "<< pose(1,2) << " " <<pose(1, 3) << " "
                 << pose(2,0) << " "<< pose(2,1) << " "<< pose(2,2) << " " <<pose(2, 3) << "\n";

        pose = submaps_[sid]->pose_slam_;
        p = pose.col(3);
        q = Eigen::Quaterniond(pose.block<3,3>(0,0));

        ofs_slam << pose(0,0) << " "<< pose(0,1) << " "<< pose(0,2) << " " <<pose(0, 3) << " "
                 << pose(1,0) << " "<< pose(1,1) << " "<< pose(1,2) << " " <<pose(1, 3) << " "
                 << pose(2,0) << " "<< pose(2,1) << " "<< pose(2,2) << " " <<pose(2, 3) << "\n";
    }
    ofs_odom.close();
    ofs_slam.close();
}

void PoseGraph::save_submap_meshes(){
    std::string logdir = Parameter::GetInstance().log_dir;
    std::string logdir_odom = Parameter::GetInstance().log_odom_mesh_dir;
    std::string logdir_slam = Parameter::GetInstance().log_slam_mesh_dir;

    for(const auto& submap: submaps_){
        //the last submap may have not been added to the pose graph
        if(submaps_.size() > 1 && !submap.second->finished_) continue;
        std::string mesh_filename_odom = logdir_odom +"/" + std::to_string(submap.first) + "_odom.ply";
        std::string mesh_filename_slam = logdir_slam + "/" + std::to_string(submap.first) + "_slam.ply";

        submap.second->filterMeshGlb(false);
        submap.second->outputMeshAsPly(mesh_filename_odom, submap.second->mesh_msg_full_);
        

        submap.second->filterMeshGlb(true);
        submap.second->outputMeshAsPly(mesh_filename_slam, submap.second->mesh_msg_full_);

        if(Parameter::GetInstance().enable_mesh_texture){
            submap.second->textureAndSaveObj(false);
        }
    }
}


void PoseGraph::initNoises(){
    gtsam::Vector priorNoiseVector6(6);
    // priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoiseVector6 << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    // odomNoiseVector6 << 1e-8, 1e-8, 1e-8, 1e-6, 1e-6, 1e-6;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

void PoseGraph::runISAM2opt(){
    // called when a variable added 
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if(add_new_loop_){
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        add_new_loop_ = false;
    }
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

void PoseGraph::updatePoses(){
    mKF.lock(); 
    for (int node_idx=0; node_idx < allFramePoses.size(); node_idx++){
        Pose6D& p =allFramePosesUpdated[node_idx];
        auto ita = isamCurrentEstimate.find(N(node_idx));
        if(ita != isamCurrentEstimate.end()){
            p.x = isamCurrentEstimate.at<gtsam::Pose3>(N(node_idx)).translation().x();
            p.y = isamCurrentEstimate.at<gtsam::Pose3>(N(node_idx)).translation().y();
            p.z = isamCurrentEstimate.at<gtsam::Pose3>(N(node_idx)).translation().z();
            p.roll = isamCurrentEstimate.at<gtsam::Pose3>(N(node_idx)).rotation().roll();
            p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(N(node_idx)).rotation().pitch();
            p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(N(node_idx)).rotation().yaw();
        }
    }

    for(int kfid = 0; kfid < keyframePosesUpdated.size(); kfid++){
        int fid = kfid_to_fid_[kfid];
        Pose6D& p =keyframePosesUpdated[kfid];
        p = allFramePosesUpdated[fid];
    }

    for(int i=0; i<added_submap_ids_.size(); i++){
        size_t sid = added_submap_ids_[i];
        auto t = isamCurrentEstimate.at<gtsam::Pose3>(S(sid)).translation();
        auto R = isamCurrentEstimate.at<gtsam::Pose3>(S(sid)).rotation();
        submaps_[sid]->pose_slam_.block<3,3>(0,0)=R.matrix();
        submaps_[sid]->pose_slam_.block<3,1>(0,3)=t;
    }
    mKF.unlock();

    // mtxRecentPose.lock();
    // const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    // recentOptimizedX = lastOptimizedPose.translation().x();
    // recentOptimizedY = lastOptimizedPose.translation().y();

    // recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

    // mtxRecentPose.unlock();
}

void PoseGraph::performSCLoopClosure(){
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts 0 and ends n-1
        LOG(INFO) << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx;

        mBuf.lock();
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        mBuf.unlock();
    }
} 


std::optional<gtsam::Pose3> PoseGraph::doICPVirtualRelative(int _loop_kf_idx, int _curr_kf_idx){
    int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx); // use same root of loop kf idx 
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx); 

    // loop verification 
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "map";
    // pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "map";
    // pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
 
    float loopFitnessScoreThreshold = 0.3; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        // LOG(INFO) << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop.";
        return std::nullopt;
    } else {
        LOG(INFO) << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop.";
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

    return poseFrom.between(poseTo);
}

std::optional<gtsam::Pose3> PoseGraph::doICPVirtualRelativeSubmap(int _loop_kf_idx, int _curr_kf_idx){
    int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    cureKeyframeCloud = keyframeCloudsSubmap[_curr_kf_idx];
    targetKeyframeCloud = keyframeCloudsSubmap[_loop_kf_idx];

    // loop verification 
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "map";
    // pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "map";
    // pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
 
    float loopFitnessScoreThreshold = 0.3; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        // LOG(INFO) << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop.";
        return std::nullopt;
    } else {
        LOG(INFO) << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop.";
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

    return poseFrom.between(poseTo);
}

std::optional<gtsam::Pose3> PoseGraph::doORB3dVirtualRelative(int _loop_kf_idx, int _curr_kf_idx){
    mKF.lock();
    FrameVisualMapPoints loop_kf_vmap_points = keyframeVMapPoints[_loop_kf_idx];
    FrameVisualMapPoints cur_kf_vmap_points = keyframeVMapPoints[_curr_kf_idx];
    mKF.unlock();
    
    Mat4d T_B_C = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
    Mat4d T_C_B = T_B_C.inverse();

    std::vector<Vec3d> src;
    std::vector<Vec3d> dst_norm;
    std::vector<cv::Point2f> src_points;  // 用于绘制的源图像点
    std::vector<cv::Point2f> dst_points;  // 用于绘制的目标图像点
    std::vector<int> match_distances;     // 匹配距离
    for(int i = 0; i < loop_kf_vmap_points.pts_3d_cam.size(); i++){
        Vec3d xyz_c = loop_kf_vmap_points.pts_3d_cam[i];
        if(xyz_c.norm()<1e-3) continue;//未关联深度的点
        // Vec3d xyz_c = T_C_B.block<3,3>(0,0)*xyz + T_C_B.block<3,1>(0,3);
        Vec2d src_uv;
        left_camera_->spaceToPlane(xyz_c, src_uv);

        // 遍历当前帧，查找最相似的点
        BRIEF::bitset desc = loop_kf_vmap_points.descriptors[i];
        int best_distance = 128;
        int best_index = -1;
        for(int k = 0; k < cur_kf_vmap_points.keypoints.size(); k++){
            BRIEF::bitset xor_of_bitset = desc ^ cur_kf_vmap_points.descriptors[k];
            int d = xor_of_bitset.count();
            if(d < best_distance){
                best_distance = d;
                best_index = k;
            }           
        }
        if(best_index != -1 && best_distance < 80){
            src.push_back(xyz_c);
            cv::KeyPoint kp = cur_kf_vmap_points.keypoints[best_index];
            Vec2d dst_uv(kp.pt.x, kp.pt.y);
            Vec3d pn;
            left_camera_->liftProjective(dst_uv, pn);
            pn(0) /= pn(2);
            pn(1) /= pn(2);
            dst_norm.push_back(pn);

            // 保存匹配点对
            src_points.push_back(cv::Point2f(src_uv(0), src_uv(1)));
            dst_points.push_back(cv::Point2f(dst_uv(0), dst_uv(1)));
            match_distances.push_back(best_distance);
        }
    }
    
    // 绘制匹配关系
    /* if(!src_points.empty() && !dst_points.empty()){
        cv::Mat img1 = keyframeOrbImg[_loop_kf_idx].clone();
        cv::Mat img2 = keyframeOrbImg[_curr_kf_idx].clone();
        // 方法1: 绘制左右并排的图像，并用连线表示匹配关系
        int height = std::max(img1.rows, img2.rows);
        int width = img1.cols + img2.cols;
        
        // 创建并排的图像
        cv::Mat side_by_side(height, width, img1.type(), cv::Scalar(0, 0, 0));
        
        // 将两张图像放置到并排图像中
        cv::Mat roi1 = side_by_side(cv::Rect(0, 0, img1.cols, img1.rows));
        img1.copyTo(roi1);
        
        cv::Mat roi2 = side_by_side(cv::Rect(img1.cols, 0, img2.cols, img2.rows));
        img2.copyTo(roi2);
        
        // 绘制关键点
        for(size_t i = 0; i < src_points.size(); i++){
            cv::Point2f pt1 = src_points[i];
            cv::Point2f pt2 = dst_points[i];
            pt2.x += img1.cols;  // 调整第二张图像中点的x坐标
            
            // 根据匹配距离设置颜色（距离越小，匹配越好，颜色越绿）
            int distance = match_distances[i];
            int green = cv::saturate_cast<int>(255 * (1.0 - distance / 80.0));
            cv::Scalar color(0, green, 255-green);  // BGR格式
            
            // 绘制连线
            cv::line(side_by_side, pt1, pt2, color, 1, cv::LINE_AA);
            
            // 绘制关键点
            cv::circle(side_by_side, pt1, 3, cv::Scalar(0, 255, 0), -1);  // 源图像关键点（绿色）
            cv::circle(side_by_side, pt2, 3, cv::Scalar(255, 0, 0), -1);  // 目标图像关键点（蓝色）
            
            // 可选：在连线旁边显示匹配距离
            if(i % 5 == 0){  // 每5个匹配显示一次距离，避免过于密集
                cv::Point2f mid_pt = (pt1 + pt2) * 0.5;
                std::string dist_text = std::to_string(distance);
                cv::putText(side_by_side, dist_text, mid_pt, 
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
        
        // 添加标题
        std::string title = "Feature Matches: " + std::to_string(src_points.size()) + " matches";
        cv::putText(side_by_side, title, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        LOG(INFO)<<side_by_side.cols<<", "<<side_by_side.rows;
        cv::imwrite("/home/wz/codes/ros_ws/livo/lvi_mesher/log/orb_loop/"+std::to_string(_curr_kf_idx)+"-"+std::to_string(_loop_kf_idx)+".jpg", side_by_side);
    } */
    
    if(src.size() <= 6) return std::nullopt;

    std::vector<bool> inliers = {};
    Mat4d T_loop2cur_cam = estimatePoseRansac(src, dst_norm, left_camera_, inliers, 0.005, 0.99, 100);// pix / fx

    // 统计内点
    int num_inliers = 0;
    std::vector<Vec3d> inlier_points_3d;
    std::vector<Vec3d> inlier_points_2d;
    for (size_t i = 0; i < src.size(); i++) {
        if (inliers[i]) {
            num_inliers++;
            inlier_points_3d.push_back(src[i]);
            inlier_points_2d.push_back(dst_norm[i]);
        }
    }
    
    // 如果内点足够，使用所有内点重新估计位姿
    if (num_inliers >= 4) {
        // 方法2：使用所有内点进行更精确的EPnP估计
        // T_loop2cur_cam = estimatePoseEPnP(inlier_points_3d, inlier_points_2d, left_camera_);
        // 计算重投影误差
        // double reprojection_error = computeReprojectionError(
        //     inlier_points_3d, inlier_points_2d, T_loop2cur_cam, left_camera_);
        
        // LOG(INFO) << "Final pose estimation with " << num_inliers 
        //         << " inliers, reprojection error: " << reprojection_error;
        // if(reprojection_error < 200){
        //     Mat4d T_c2l = T_B_C * T_loop2cur_cam.inverse() * T_C_B;//转换回body系的相对运动
        //     Mat3d rot = T_c2l.block<3,3>(0,0);
        //     Vec3d pos = T_c2l.block<3,1>(0,3);
        //     gtsam::Pose3 poseTo = Pose3(Rot3(rot),  Point3(pos));
        //     gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
        //     LOG(INFO) << "Add RANSAC constraint!";
        //     return poseFrom.between(poseTo);
        // }else{
        //     return std::nullopt;
        // }
        Mat4d T_c2l = T_B_C * T_loop2cur_cam.inverse() * T_C_B;//转换回body系的相对运动
        Mat3d rot = T_c2l.block<3,3>(0,0);
        Vec3d pos = T_c2l.block<3,1>(0,3);
        if(pos.norm() < 5.0){// if the estimated loop closure is too far, it is likely wrong. (user parameter but fixed value is safe.)
            gtsam::Pose3 poseTo = Pose3(Rot3(rot),  Point3(pos));
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
            LOG(INFO) << "Add RANSAC constraint!";
            return poseFrom.between(poseTo);
        } 
    } else {
        LOG(WARNING) << "Not enough inliers for reliable pose estimation";
        
    }
    return std::nullopt;
}

void PoseGraph::loopFindNearKeyframesCloud(pcl::PointCloud<PointType>::Ptr& nearKeyframes, 
        const int& key, const int& submap_size, const int& root_idx){
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i; // see https://github.com/gisbi-kim/SC-A-LOAM/issues/7 ack. @QiMingZhenFan found the error and modified as below. 
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
            continue;

        mKF.lock(); 
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

pcl::PointCloud<PointType>::Ptr PoseGraph::local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf){
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        // cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

Pose6D PoseGraph::diffTransformation(const Pose6D& _p1, const Pose6D& _p2){
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
}


void PoseGraph::process_pg(){
    while(1){
        if(job_finished_) break;
        // Add node-to-node constraints and launch loop detection task
		// while(!odometryBuf.empty() && !fullResBuf.empty()){
		while(!tracked_data_.empty()){
			mBuf.lock();       
            std::shared_ptr<TrackingResult> cur_data = tracked_data_.front();
            
            // Time equal check
            timeLaserOdometry = cur_data->timestamp_nsec;
            timeLaser = cur_data->timestamp_nsec;

            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());

            if(Parameter::GetInstance().enable_log_pcd){
                std::string log_pcd_dir = Parameter::GetInstance().log_pcd_dir;
                std::string filename = log_pcd_dir + "/" + std::to_string(cur_data->timestamp_nsec) + ".pcd";
                // pcl::PointCloud<pcl::PointXYZRGB> colored_pcd_w;
                // pcl::transformPointCloud(cur_data->deskewed_points, colored_pcd_w, cur_data->local_pose);
                // pcl::io::savePCDFileBinary(filename, colored_pcd_w); 
                // 保存原始Body系点云，全局点云由脚本与位姿对应生成
                pcl::io::savePCDFileBinary(filename, cur_data->deskewed_points); 
            }

            xyzrgb2xyz(cur_data->deskewed_points, *thisKeyFrame);
            // *thisKeyFrame = cur_data->deskewed_points;

            Pose6D pose_curr = Mat4dtoPose6D(cur_data->local_pose);
            current_frame_id_ = cur_data->frame_id;

            // 在当前帧IMU坐标系下，只用了左相机视角内的点
            FrameVisualMapPoints cur_vmap_points;
            cur_vmap_points.keypoints = cur_data->keypoints;
            cur_vmap_points.descriptors = cur_data->descriptors;
            cur_vmap_points.pts_3d_cam = cur_data->points_3d_cam; 
            cv::Mat img = cur_data->image_raw.clone();

            tracked_data_.pop();

            mBuf.unlock(); 

            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  

            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap 
                || cur_data->gnss != nullptr) {
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            } else {
                isNowKeyFrame = false;
            }

            allFramePoses.push_back(pose_curr);
            allFramePosesUpdated.push_back(pose_curr);
            allFrameTimes.push_back(timeLaserOdometry);
            
            const int prev_node_idx = allFramePoses.size() - 2; 
            const int curr_node_idx = allFramePoses.size() - 1;

            if(!gtSAMgraphMade) {/* prior node */                
                const int init_node_idx = 0; 
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(allFramePoses.at(init_node_idx));

                mtxPosegraph.lock();
                {
                    // prior factor 
                    initialEstimate.insert(N(init_node_idx), poseOrigin);
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(N(init_node_idx), poseOrigin, priorNoise));
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();

                gtSAMgraphMade = true; 
                LOG(INFO) << "posegraph prior node " << init_node_idx << " added.";
            } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1 
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(allFramePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(allFramePoses.at(curr_node_idx));

                mtxPosegraph.lock();
                {
                    // odom factor
                    gtsam::Pose3 relPose = poseFrom.between(poseTo);
                    initialEstimate.insert(N(curr_node_idx), poseTo); 
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(prev_node_idx), N(curr_node_idx), relPose, odomNoise));
                }
                mtxPosegraph.unlock();

                if(curr_node_idx % 100 == 0)
                    LOG(INFO) << "posegraph odom node " << curr_node_idx << " added.";
            }

            if(cur_data->gnss ){//&& initialEstimate.find(N(curr_node_idx)) != initialEstimate.end() 
                Vec3d enu;
                if(!gnss_inited_){
                    geo_converter_.Reset(cur_data->gnss->latitude, cur_data->gnss->longitude, cur_data->gnss->altitude);
                    gnss_inited_ = true;
                }
                geo_converter_.Forward(cur_data->gnss->latitude, cur_data->gnss->longitude, cur_data->gnss->altitude, enu[0], enu[1], enu[2]);
                gnss_positions_.push_back({timeLaserOdometry, enu});
                
                // add gnss factor
                double nx = cur_data->gnss->position_covariance(0,0);
                double ny = cur_data->gnss->position_covariance(1,1);
                double nz = cur_data->gnss->position_covariance(2,2);
                if((enu - last_gnss_position_).norm() > 0.5 && abs(nx) < 8.0 && abs(ny) < 8.0){
                    if(!gnss_initial_aligned_){
                        gnss_for_initial_alignment_.push_back(*cur_data->gnss);
                        node_ids_for_initial_alignment_.push_back(curr_node_idx);
                        if(gnss_for_initial_alignment_.size()>20){
                            for(int i = 0; i < gnss_for_initial_alignment_.size(); i++){
                                sad::NaiveGNSS gnss = gnss_for_initial_alignment_[i];
                                int id = node_ids_for_initial_alignment_[i];
                                double nxi = gnss.position_covariance(0,0);
                                double nyi = gnss.position_covariance(1,1);
                                double nzi = gnss.position_covariance(2,2);
                                gtsam::Vector noisei(3);
                                noisei << nxi, nyi, nzi;
                                Vec3d enui;
                                geo_converter_.Forward(gnss.latitude, gnss.longitude, gnss.altitude, enui[0], enui[1], enui[2]);
                                noiseModel::Diagonal::shared_ptr gps_noisei = noiseModel::Diagonal::Variances(noisei);
                                gtsam::GPSFactor gps_factori(N(id), gtsam::Point3(enui(0), enui(1), enui(2)), gps_noisei);
                                mtxPosegraph.lock();
                                gtSAMgraph.add(gps_factori);
                                add_new_loop_ = true;
                                mtxPosegraph.unlock();
                            }
                            gnss_initial_aligned_ = true;
                        }
                    }else{
                        gtsam::Vector noise(3);
                        noise << nx, ny, nz;
                        noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(noise);
                        gtsam::GPSFactor gps_factor(N(curr_node_idx), gtsam::Point3(enu(0), enu(1), enu(2)), gps_noise);
                        mtxPosegraph.lock();
                        gtSAMgraph.add(gps_factor);
                        add_new_loop_ = true;
                        mtxPosegraph.unlock();
                    }
                }
                last_gnss_position_ = enu;
            }

            // TODO(wz):
            // Add node-to-submap constraints
            // 放到内部循环，防止遗漏Submap约束，可优化，减少不必要的计算
            mBuf.lock();
            if(!submaps_.empty()){
                for(auto it = submaps_.begin(); it != submaps_.end(); it++){
                    if(it->second->finished_ && !it->second->added_to_graph_){
                        Mat4d submap_pose = it->second->pose_odom_;                 
                        std::vector<size_t>& frame_ids = it->second->frame_ids_;
                        std::vector<Mat4d>& node_poses = it->second->node_poses_;
                        CHECK(frame_ids.size() == node_poses.size());
                        gtsam::Pose3 poseTo = Mat4dtoGTSAMPose3(submap_pose);
                        mtxPosegraph.lock();
                        initialEstimate.insert(S(it->first), poseTo); 
                        for(size_t i = 0; i < frame_ids.size(); i++){
                            if(initialEstimate.find(N(frame_ids[i])) == initialEstimate.end()) continue;//TODO(wz): more rigorous way
                            gtsam::Pose3 poseFrom = Mat4dtoGTSAMPose3(node_poses[i]);
                            gtsam::Pose3 relPose = poseFrom.between(poseTo);
                            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(frame_ids[i]), S(it->first), relPose, odomNoise));
                            added_submap_ids_.push_back(it->first);
                        }                    
                        mtxPosegraph.unlock();
                        it->second->added_to_graph_ = true;
                    }
                }
            }
            mBuf.unlock();
            

            if( ! isNowKeyFrame ) 
                continue; 

            //
            // Save data and Add consecutive node 
            //
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            downSizeFilterScancontext.filter(*thisKeyFrameDS);

            mKF.lock(); 
            
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr); // init
            keyframeTimes.push_back(timeLaserOdometry);
            keyframeVMapPoints.push_back(cur_vmap_points);

            kfid_to_fid_[keyframePoses.size()-1] = allFramePoses.size()-1;
            fid_to_kfid_[allFramePoses.size()-1] = keyframePoses.size()-1;

            if(Parameter::GetInstance().enable_scan_context){
                keyframeLaserClouds.push_back(thisKeyFrameDS);
                scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);
                performSCLoopClosure();
            }

            if(Parameter::GetInstance().enable_extended_scan_context){
                if(!Parameter::GetInstance().enable_scan_context){
                    keyframeLaserClouds.push_back(thisKeyFrameDS);
                }
                
                ExtendedScanContext esc;
                esc.generateFromPointCloud(thisKeyFrameDS);

                int kfid = keyframePoses.size()-1;
                double min_dist;
                int best_offset;
                std::pair<bool, int> res = esc_db_.detectLoopClosure(
                    esc, kfid, min_dist, best_offset, Parameter::GetInstance().extended_scan_context_threshold);
                if(res.first){
                    // LOG(INFO)<<"Detect extended scan context loop: "<<min_dist;
                    const int prev_node_idx = res.second;
                    const int curr_node_idx = kfid; 
                    LOG(INFO) << "ESC Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << " with distance: " << min_dist;

                    mBuf.lock();
                    escLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
                    mBuf.unlock();
                }
                esc_db_.addKeyframe(esc, kfid);
            }
            
            // 用Orb3D检测回环
            if(Parameter::GetInstance().enable_loop_orb3d){
                LKToORBConverter brief_extractor;
                std::vector<BRIEF::bitset> descriptors;
                std::vector<cv::KeyPoint> keypoints;
                vector<cv::Point2f> cur_pts;
                if(img.channels()==3){
                    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                }
                cv::goodFeaturesToTrack(img, cur_pts, 1000, 0.01, 10);
                
                brief_extractor.computeORBFromTrackedPoints(img, cur_pts, keypoints, descriptors, false);
                keyframeVMapPoints.back().keypoints_full = keypoints;
                keyframeVMapPoints.back().descriptors_full = descriptors;
                
                if (!descriptors.empty()) {
                    int kfid = keyframePoses.size()-1;
                    if(keyframePoses.size() > 50){
                        DBoW2::QueryResults ret;
                        db_->query(descriptors, ret, 4, kfid - 50);
                        // 历史帧->当前帧
                        int submap_size = Parameter::GetInstance().frames_for_submap;
                        
                        if(!ret.empty() && ret[0].Score>Parameter::GetInstance().loop_orb3d_threshold 
                                && kfid_to_fid_[kfid] - kfid_to_fid_[int(dbid_to_kfid_[ret[0].Id])] > 2*submap_size){
                            mBuf.lock();
                            scLoopOrb3DBuf.push(std::pair<int, int>(dbid_to_kfid_[ret[0].Id], kfid));
                            mBuf.unlock();
                            // cv::Mat catimg;
                            // cv::hconcat(img, keyframeOrbImg[dbid_to_kfid_[ret[0].Id]], catimg);
                            // cv::imwrite("/home/wz/codes/ros_ws/livo/lvi_mesher/log/orb_loop/"+std::to_string(keyframePoses.size()-1)+"-"+std::to_string(dbid_to_kfid_[ret[0].Id])+".jpg", catimg);
                            LOG(INFO)<<"Detected loop: "<<ret[0].Score;
                        }
                    }
                    DBoW2::EntryId dbid = db_->add(descriptors);
                    kfid_to_dbid_[kfid] = dbid; 
                    dbid_to_kfid_[dbid] = kfid; 
                }
            }
            
            mKF.unlock(); 
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraph::process_lcd(){
    float loopClosureFrequency = 1.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        if(job_finished_) break;
        performSCLoopClosure();
    }
} 

void PoseGraph::process_sc_icp(){
    while(1){
        if(job_finished_) break;
		while ( !scLoopICPBuf.empty()){
            if( scLoopICPBuf.size() > 30 ) {
                // ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock(); 

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            const int prev_fid = kfid_to_fid_[prev_node_idx];
            const int curr_fid = kfid_to_fid_[curr_node_idx];
            std::ofstream ofs(Parameter::GetInstance().log_dir + "/lidar_sc_loop_node_pair.txt", std::ios_base::app);
            if(ofs.is_open()){
                ofs << prev_fid <<", "<< curr_fid <<"\n";
                ofs.close();
            }
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                int prev_id = kfid_to_fid_[prev_node_idx];
                int curr_id = kfid_to_fid_[curr_node_idx];
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(prev_id), N(curr_id), relative_pose, robustLoopNoise));
                // writeEdge({prev_node_idx, curr_node_idx}, relative_pose, edges_str); // giseop
                // runISAM2opt();
                mtxPosegraph.unlock();
                Eigen::Matrix4d T_loop = relative_pose.matrix();
                std::ofstream ofs_cst(Parameter::GetInstance().log_dir + "/lidar_sc_loop_constraint.txt", std::ios_base::app);
                if(ofs_cst.is_open()){
                    ofs_cst << prev_fid <<", "<< curr_fid << ", "
                            << T_loop(0,0)<<", "<<T_loop(0, 1)<< ", "<<T_loop(0, 2)<<", "<<T_loop(0, 3) <<", "
                            << T_loop(1,0)<<", "<<T_loop(1, 1)<< ", "<<T_loop(1, 2)<<", "<<T_loop(1, 3) <<", "
                            << T_loop(2,0)<<", "<<T_loop(2, 1)<< ", "<<T_loop(2, 2)<<", "<<T_loop(2, 3) <<"\n";
                    ofs_cst.close();
                }
            } 
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

// TODO: Remove
void PoseGraph::process_esc_icp(){
    while(1){
        if(job_finished_) break;
		while ( !escLoopICPBuf.empty()){
            if( escLoopICPBuf.size() > 30 ) {
                // ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = escLoopICPBuf.front();
            escLoopICPBuf.pop();
            mBuf.unlock(); 

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            const int prev_fid = kfid_to_fid_[prev_node_idx];
            const int curr_fid = kfid_to_fid_[curr_node_idx];
            std::ofstream ofs(Parameter::GetInstance().log_dir + "/lidar_esc_loop_node_pair.txt", std::ios_base::app);
            if(ofs.is_open()){
                ofs << prev_fid <<", "<< curr_fid <<"\n";
                ofs.close();
            }
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                int prev_id = kfid_to_fid_[prev_node_idx];
                int curr_id = kfid_to_fid_[curr_node_idx];
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(prev_id), N(curr_id), relative_pose, robustLoopNoise));
                // writeEdge({prev_node_idx, curr_node_idx}, relative_pose, edges_str); // giseop
                // runISAM2opt();
                mtxPosegraph.unlock();
                Eigen::Matrix4d T_loop = relative_pose.matrix();
                std::ofstream ofs_cst(Parameter::GetInstance().log_dir + "/lidar_esc_loop_constraint.txt", std::ios_base::app);
                if(ofs_cst.is_open()){
                    ofs_cst << prev_fid <<", "<< curr_fid << ", "
                            << T_loop(0,0)<<", "<<T_loop(0, 1)<< ", "<<T_loop(0, 2)<<", "<<T_loop(0, 3) <<", "
                            << T_loop(1,0)<<", "<<T_loop(1, 1)<< ", "<<T_loop(1, 2)<<", "<<T_loop(1, 3) <<", "
                            << T_loop(2,0)<<", "<<T_loop(2, 1)<< ", "<<T_loop(2, 2)<<", "<<T_loop(2, 3) <<"\n";
                    ofs_cst.close();
                }
            } 
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraph::process_icp_submap(){
    while(1){
        if(job_finished_) break;
		while ( !scLoopSubmapBuf.empty()){
            if( scLoopSubmapBuf.size() > 30 ) {
                // ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = scLoopSubmapBuf.front();
            scLoopSubmapBuf.pop();
            mBuf.unlock(); 

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            auto relative_pose_optional = doICPVirtualRelativeSubmap(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                int prev_id = kfid_to_submapfid_[prev_node_idx];
                int curr_id = kfid_to_submapfid_[curr_node_idx];
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(prev_id), N(curr_id), relative_pose, robustLoopNoise));
                mtxPosegraph.unlock();
            } 
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraph::process_orb3d_matcher(){
    while(1){
        if(job_finished_) break;
		while ( !scLoopOrb3DBuf.empty()){
            if( scLoopOrb3DBuf.size() > 30 ) {
                // ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }
            // LOG(INFO)<<scLoopOrb3DBuf.size();

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = scLoopOrb3DBuf.front();
            scLoopOrb3DBuf.pop();
            mBuf.unlock(); 

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;

            const int prev_fid = kfid_to_fid_[prev_node_idx];
            const int curr_fid = kfid_to_fid_[curr_node_idx];
            std::ofstream ofs(Parameter::GetInstance().log_dir + "/visual_loop_node_pair.txt", std::ios_base::app);
            if(ofs.is_open()){
                ofs << prev_fid <<", "<< curr_fid <<"\n";
                ofs.close();
            }

            std::vector<MatchPair> forward_matches;  // Loop 3D -> Cur 2D
            std::vector<MatchPair> backward_matches; // Cur 3D -> Loop 2D

            FrameVisualMapPoints loop_kf_vmap_points = keyframeVMapPoints[prev_node_idx];
            FrameVisualMapPoints cur_kf_vmap_points = keyframeVMapPoints[curr_node_idx];
            Mat4d T_B_C = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
            Mat4d T_C_B = T_B_C.inverse();

            // A. 正向匹配 (你原有的逻辑)
            for(int i = 0; i < loop_kf_vmap_points.pts_3d_cam.size(); i++){
                Vec3d p_loop = loop_kf_vmap_points.pts_3d_cam[i];
                if(p_loop.norm() < 1e-3) continue;

                int best_dist = 80; int best_idx = -1;
                for(int k = 0; k < cur_kf_vmap_points.descriptors_full.size(); k++){
                    int d = (loop_kf_vmap_points.descriptors[i] ^ cur_kf_vmap_points.descriptors_full[k]).count();
                    if(d < best_dist) { best_dist = d; best_idx = k; }
                }
                if(best_idx != -1) {
                    Vec3d pn;
                    left_camera_->liftProjective(Vec2d(cur_kf_vmap_points.keypoints_full[best_idx].pt.x, 
                                                cur_kf_vmap_points.keypoints_full[best_idx].pt.y), pn);
                    forward_matches.push_back({p_loop, Vec2d(pn(0)/pn(2), pn(1)/pn(2))});
                }
            }

            // B. 反向匹配 (当前帧 3D -> 历史帧 2D)
            for(int i = 0; i < cur_kf_vmap_points.pts_3d_cam.size(); i++){
                Vec3d p_cur = cur_kf_vmap_points.pts_3d_cam[i];
                if(p_cur.norm() < 1e-3) continue;

                int best_dist = 80; int best_idx = -1;
                for(int k = 0; k < loop_kf_vmap_points.descriptors_full.size(); k++){
                    int d = (cur_kf_vmap_points.descriptors[i] ^ loop_kf_vmap_points.descriptors_full[k]).count();
                    if(d < best_dist) { best_dist = d; best_idx = k; }
                }
                if(best_idx != -1) {
                    Vec3d pn;
                    left_camera_->liftProjective(Vec2d(loop_kf_vmap_points.keypoints_full[best_idx].pt.x, 
                                                loop_kf_vmap_points.keypoints_full[best_idx].pt.y), pn);
                    backward_matches.push_back({p_cur, Vec2d(pn(0)/pn(2), pn(1)/pn(2))});
                }
            }

            Mat4d T_c2l;
            bool success = EstimateRelativePoseBidirectional(forward_matches, backward_matches, T_c2l);
            if(success){
                Mat4d T_loop = T_B_C * T_c2l * T_B_C.inverse();
                Mat3d rot = T_loop.block<3,3>(0,0);
                Vec3d pos = T_loop.block<3,1>(0,3);
                gtsam::Pose3 poseTo = Pose3(Rot3(rot),  Point3(pos));
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
                gtsam::Pose3 relative_pose = poseFrom.between(poseTo);

                mtxPosegraph.lock();
                int prev_id = kfid_to_fid_[prev_node_idx];
                int curr_id = kfid_to_fid_[curr_node_idx];
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(prev_id), N(curr_id), relative_pose, robustLoopNoise));
                mtxPosegraph.unlock();
                std::ofstream ofs_cst(Parameter::GetInstance().log_dir + "/visual_loop_constraint.txt", std::ios_base::app);
                if(ofs_cst.is_open()){
                    ofs_cst << prev_fid <<", "<< curr_fid << ", "
                            << T_loop(0,0)<<", "<<T_loop(0, 1)<< ", "<<T_loop(0, 2)<<", "<<T_loop(0, 3) <<", "
                            << T_loop(1,0)<<", "<<T_loop(1, 1)<< ", "<<T_loop(1, 2)<<", "<<T_loop(1, 3) <<", "
                            << T_loop(2,0)<<", "<<T_loop(2, 1)<< ", "<<T_loop(2, 2)<<", "<<T_loop(2, 3) <<"\n";
                    ofs_cst.close();
                }
                LOG(INFO) << "Add RANSAC constraint!";
            }

            /*auto relative_pose_optional = doORB3dVirtualRelative(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                int prev_id = kfid_to_fid_[prev_node_idx];
                int curr_id = kfid_to_fid_[curr_node_idx];
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(prev_id), N(curr_id), relative_pose, robustLoopNoise));
                mtxPosegraph.unlock();
            } else{
                auto relative_pose_optional_inv = doORB3dVirtualRelative(curr_node_idx, prev_node_idx);
                if(relative_pose_optional_inv){
                    gtsam::Pose3 relative_pose = relative_pose_optional_inv.value().inverse();
                    mtxPosegraph.lock();
                    int prev_id = kfid_to_fid_[prev_node_idx];
                    int curr_id = kfid_to_fid_[curr_node_idx];
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(N(prev_id), N(curr_id), relative_pose, robustLoopNoise));
                    mtxPosegraph.unlock();
                }
            }*/
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraph::process_isam(){
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if(job_finished_) break;
        if( gtSAMgraphMade ) {
            mtxPosegraph.lock();
            runISAM2opt();
            // LOG(INFO) << "running isam2 optimization ...";
            mtxPosegraph.unlock();
            // saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
            // saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
            // saveGTSAMgraphG2oFormat(isamCurrentEstimate);
        }
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

Pose6D PoseGraph::getOdom(TimestampedPose _odom){
    auto tx = _odom.pose.col(3)(0);
    auto ty = _odom.pose.col(3)(1);
    auto tz = _odom.pose.col(3)(2);
    
    auto ypr = _odom.pose.block<3,3>(0,0).eulerAngles(2, 1, 0);
    return Pose6D{tx, ty, tz, ypr(2), ypr(1), ypr(0)}; 
}

pcl::PointCloud<PointType>::Ptr PoseGraph::convertMeshToCloud(std::shared_ptr<Submap> submap) {
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    
    
    submap->filterMeshGlb();
    *cloud = submap->getAllVertices();

    pcl::PointCloud<PointType>::Ptr cloud_kp(new pcl::PointCloud<PointType>());
    std::vector<Vec3d> keypoints;
    submap->aggregateCellFeatures(keypoints);
    cloud_kp->points.resize(keypoints.size());
    for(int i = 0; i < keypoints.size(); ++i){
        const Vec3d& kp  = keypoints[i];
        PointType p;
        p.x = kp(0);
        p.y = kp(1);
        p.z = kp(2);
        cloud_kp->points[i] = p;
    }
    std::string logdir = Parameter::GetInstance().log_dir;
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud_kp->width = cloud_kp->points.size();
    cloud_kp->height = 1;

    pcl::transformPointCloud(*cloud, *cloud, submap->pose_odom_.inverse().cast<float>());
    pcl::io::savePCDFileASCII(logdir + "/" + std::to_string(submap->get_id())+".pcd", *cloud);
    pcl::io::savePCDFileASCII(logdir + "/" + std::to_string(submap->get_id())+"_kp.pcd", *cloud_kp);
    LOG(INFO)<<cloud->size()<<", "<<cloud_kp->size();
    LOG(INFO)<<"Saved to "<<logdir;

    
    // 降采样以提高后续 SC 和 ICP 效率
    pcl::PointCloud<PointType>::Ptr cloud_ds(new pcl::PointCloud<PointType>());
    downSizeFilterScancontext.setInputCloud(cloud);
    downSizeFilterScancontext.filter(*cloud_ds);

    return cloud_ds;
}