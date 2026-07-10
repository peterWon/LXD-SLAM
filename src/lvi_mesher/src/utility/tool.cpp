#include "utility/tool.h"

//format transform
Transf state2trans3(State state){
    Transf result_transf;
    Eigen::Matrix3d  rotation = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 3, 1 > translation;
    translation << state(0, 0), state(1, 0), state(2, 0);
    double roll = state(3,0), pitch = state(4, 0), yaw = state(5, 0);
    rotation << cos(yaw) * cos(pitch) - sin(roll) * sin(yaw) * sin(pitch),
            -1 * cos(roll) * sin(yaw), cos(yaw) * sin(pitch) + cos(pitch) * sin(roll) * sin(yaw),
            cos(pitch) * sin(yaw) + cos(yaw) * sin(roll) * sin(pitch),
            cos(roll) * cos(yaw), sin(yaw) * sin(pitch) - cos(yaw) * cos(pitch) * sin(roll),
            -1 * cos(roll) * sin(pitch), sin(roll), cos(roll) * cos(pitch);
    result_transf << rotation, translation, 0,0,0,1;
    return result_transf;
}
Transf state2quat2trans3(State state){
    Transf result_transf;
    tf::Quaternion q_tmp;
    Eigen::Matrix3d rotation;
    Eigen::Matrix<double, 3, 1 > translation;

    double roll, pitch, yaw;
    roll = state(3, 0) ;
    pitch = state(4, 0);
    yaw = state(5, 0);
    q_tmp.setRPY(roll, pitch, yaw);
    tf::Matrix3x3 matrix(q_tmp);

    rotation << matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2];
    translation << state(0, 0), state(1, 0), state(2, 0);

    result_transf << rotation, translation, 0,0,0,1;
    return result_transf;
}
Transf state2quat2trans3_down(State state){
    Transf result_transf;
    tf::Quaternion q_tmp;
    Eigen::Matrix3d rotation;
    Eigen::Matrix<double, 3, 1 > translation;

    double roll, pitch, yaw;
    pitch = state(3, 0);
    roll = state(4, 0);
    yaw = state(5, 0);
    q_tmp.setRPY(roll, pitch, yaw);
    tf::Matrix3x3 matrix(q_tmp);

    rotation << matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2];
    translation << state(0, 0), state(1, 0), state(2, 0);

    result_transf << rotation, translation, 0,0,0,1;
    return result_transf;
}
State trans32state(Transf & transf){
    State result_state;
    result_state(0, 0) = transf(0, 3);
    result_state(1, 0) = transf(1, 3);
    result_state(2, 0) = transf(2, 3);
    double roll, pitch, yaw;
    roll = atan2(transf(2, 1), sqrt(pow(transf(0, 1), 2) + pow(transf(1, 1), 2)));
    pitch = atan2(-1 * transf(2, 0), transf(2, 2));
    yaw = atan2(-1 * transf(0, 1), transf(1, 1));
    result_state(3, 0) = roll;
    result_state(4, 0) = pitch;
    result_state(5, 0) = yaw;
    return result_state;
}
sensor_msgs::PointCloud matrix2DtoPclMsg(Eigen::MatrixXd matrix, int num_points){
    if((matrix.cols() < num_points) || (matrix.rows() < 2)){
        ROS_INFO("Error! matrix has not correct point");
    }
    sensor_msgs::PointCloud cloud;
    cloud.header.stamp = ros::Time::now();
    cloud.header.frame_id = "sensor_frame";//"sensor_frame";

    cloud.points.resize(num_points);
    for (unsigned int i = 0; i < num_points; ++i) {
        cloud.points[i].x = matrix(0, i);
        cloud.points[i].y = matrix(1, i);
        cloud.points[i].z = 0;
    }
    return cloud;
}
sensor_msgs::PointCloud matrix3DtoPclMsg(Eigen::MatrixXd matrix, int num_points, int skip_step){
    if((matrix.cols() < num_points) || (matrix.rows() < 3)){
        ROS_INFO("Error! matrix has not correct point");
    }
    if(skip_step == 0 || skip_step < 0) skip_step = 1;
    sensor_msgs::PointCloud cloud;
    cloud.header.stamp = ros::Time::now();
    cloud.header.frame_id = "map";//"sensor_frame";

    num_points = num_points / skip_step;
    cloud.points.resize(num_points);
    for (unsigned int i = 0; i < num_points; ++i) {
        cloud.points[i].x = matrix(0, i * skip_step);
        cloud.points[i].y = matrix(1, i * skip_step);
        cloud.points[i].z = matrix(2, i * skip_step);
    }
    return cloud;
}

sensor_msgs::PointCloud matrix3DtoPclMsg(Eigen::MatrixXd matrix, int num_points){
    if((matrix.cols() < num_points) || (matrix.rows() < 3)){
        ROS_INFO("Error! matrix has not correct point");
    }
    sensor_msgs::PointCloud cloud;
    cloud.header.stamp = ros::Time::now();
    cloud.header.frame_id = "map";//"sensor_frame";

    cloud.points.resize(num_points);
    for (unsigned int i = 0; i < num_points; ++i) {
        cloud.points[i].x = matrix(0, i);
        cloud.points[i].y = matrix(1, i);
        cloud.points[i].z = matrix(2, i);
    }
    return cloud;
}
Eigen::Matrix<double, 1, 6> txt2matrixSeattleOdom(int line, std::ifstream & file_odom){
    Eigen::Matrix<double, 1, 6> odom;
    odom.fill(0);
    char buffer[30];
    double irng;
    int i = 0;
    while((i < 5) && (!file_odom.eof())){
        file_odom.getline(buffer, 30, ','); //
        sscanf(buffer, "%lf, ", & irng );//%d mean int, %lf mean double, %f mean float
        odom(0, i) = irng;
        i++;
    }
    file_odom.getline(buffer, 30, '\n'); //
    sscanf(buffer, "%lf, ", & irng );
    odom(0, 5) = irng;
    return odom;
}
pcl::PointCloud<pcl::PointXYZ> matrix3D2pcl(const PointMatrix & points){

    pcl::PointCloud<pcl::PointXYZ> pcl_result;
    pcl_result.header.frame_id = "map";//"sensor_frame";

    unsigned int num_points = points.num_point;
    pcl_result.width = points.num_point;
    pcl_result.height = 1;
    pcl_result.points.resize(num_points);
    pcl_result.getMatrixXfMap(3, 4, 0) = points.point.leftCols(points.num_point).cast<float>();
    return pcl_result;
}
geometry_msgs::PoseWithCovariance transf2PoseWithCovariance(Transf transf) {
    geometry_msgs::PoseWithCovariance pose;
    //pose
    pose.pose.position.x = transf(0, 3);
    pose.pose.position.y = transf(1, 3);
    pose.pose.position.z = transf(2, 3);
    //orientation
    tf::Matrix3x3 tmp_m(transf(0, 0), transf(0, 1), transf(0, 2),
                        transf(1, 0), transf(1, 1), transf(1, 2),
                        transf(2, 0), transf(2, 1), transf(2, 2));
    double roll, yaw, pitch;
    tmp_m.getEulerYPR(yaw, pitch, roll);
    tf::Quaternion tmp_q;
    tmp_q.setRPY(roll, pitch, yaw);
    pose.pose.orientation.x = tmp_q.x();
    pose.pose.orientation.y = tmp_q.y();
    pose.pose.orientation.z = tmp_q.z();
    pose.pose.orientation.w = tmp_q.w();

    return pose;
}
Transf PoseWithCovariance2transf(geometry_msgs::PoseWithCovariance pose) {
    Transf transf;
    tf::Quaternion quat;
    tf::quaternionMsgToTF(pose.pose.orientation, quat);
    //tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    tf::Matrix3x3 matrix(quat);
    transf << matrix[0][0], matrix[0][1], matrix[0][2], pose.pose.position.x,
              matrix[1][0], matrix[1][1], matrix[1][2], pose.pose.position.y,
              matrix[2][0], matrix[2][1], matrix[2][2], pose.pose.position.z,
              0, 0, 0, 1;

    return transf;
}
bool readKitti(const std::string & file_dataset, const std::string & seq, int line_num, int dataset,
               pcl::PointCloud<pcl::PointXYZ> & laser_cloud){//partially refer to A-LOAM
    std::stringstream bin_point_cloud_path;
    if(dataset == 1){//kitti
        bin_point_cloud_path << file_dataset << seq << "/velodyne/" << std::setfill('0') << std::setw(6) << line_num << ".bin";
    }
    else if(dataset == 2){//mai_city
        bin_point_cloud_path << file_dataset << seq << "/velodyne/" << std::setfill('0') << std::setw(5) << line_num << ".bin";
    }
    std::cout<<std::setprecision(7)<<setiosflags(std::ios::fixed);
    std::ifstream bin_point_cloud_file(bin_point_cloud_path.str(), std::ifstream::in | std::ifstream::binary);
    if(!bin_point_cloud_file.good()){
        return false;
    }
    bin_point_cloud_file.seekg(0, std::ios::end);
    const size_t num_elements = bin_point_cloud_file.tellg() / sizeof(float);
    bin_point_cloud_file.seekg(0, std::ios::beg);
    std::vector<float> lidar_data(num_elements);
    bin_point_cloud_file.read(reinterpret_cast<char*>(&lidar_data[0]), num_elements * sizeof(float));
    std::cout << "totally " << int(lidar_data.size() / 4.0) << " points in this lidar frame \n";

    std::vector<Eigen::Vector3d> lidar_points;
    std::vector<float> lidar_intensities;
    for (std::size_t i = 0; i < lidar_data.size(); i += 4)
    {
        lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
        lidar_intensities.push_back(lidar_data[i+3]);

        pcl::PointXYZ point;
        point.x = lidar_data[i];
        point.y = lidar_data[i + 1];
        point.z = lidar_data[i + 2];
        //if(point.z > -2.5){//there are some underground outliers in kitti dataset, remove them
            laser_cloud.push_back(point);
        //}
    }
    return true;
}

//transformation
Point trans3Dpoint(int x, int y, int z, const Transf& transf){
    Eigen::Matrix<double, 4, 1> point_expend;
    point_expend << x, y, z, 1;
    point_expend = transf * point_expend;
    Point point_turned = point_expend.topRows(3);
    return point_turned;
}
Eigen::MatrixXd trans3Dpoints(const Eigen::MatrixXd & bef_rotate, Eigen::Isometry3d & T){
    Eigen::MatrixXd aft_rotate;
    aft_rotate = Eigen::MatrixXd::Zero(4, 1).replicate(1, bef_rotate.cols());
    aft_rotate.topRightCorner(bef_rotate.rows(), bef_rotate.cols()) = bef_rotate;
    aft_rotate = T.matrix() * aft_rotate;
    return aft_rotate.topRightCorner(bef_rotate.rows(), bef_rotate.cols());
}
Transf createTrans(double x, double y, double z, double roll, double pitch, double yaw){
    //theta unit: degree, just for convenience
    State state;
    //state << x, y, z, pitch / 180.0 * M_PI, roll / 180.0 * M_PI, yaw / 180.0 * M_PI;
    state << x, y, z, roll / 180.0 * M_PI, pitch / 180.0 * M_PI, yaw / 180.0 * M_PI;
    Transf result_transf = state2trans3(state);
    return result_transf;
}

//filter
sensor_msgs::PointCloud2 pclMsg2VoxelFilter(const sensor_msgs::PointCloud2::ConstPtr & pcl_msg, double voxel_size) {
    //msg to pcl2
    auto *pcl2Ptr = new pcl::PCLPointCloud2;
    pcl::PCLPointCloud2ConstPtr cloudPtr(pcl2Ptr);
    pcl_conversions::toPCL(*pcl_msg, *pcl2Ptr);
    //pcl2 filter
    pcl::PCLPointCloud2 msg_filtered;
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(cloudPtr);
    sor.setLeafSize(voxel_size, voxel_size, voxel_size);
    sor.filter(msg_filtered);
    //pcl2 to msg
    sensor_msgs::PointCloud2 point_output;
    pcl_conversions::fromPCL(msg_filtered, point_output);
    return point_output;
    //view it
/*    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudT(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*pcl_msg, *cloudT);*/
    //pcl::visualization::CloudViewer viewer("Cloud Viewer");
    //viewer.showCloud(cloudT);
    //you can publish point_output
}

pcl::PointCloud<pcl::PointXYZ> pclVoxelFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr & pcl, double voxel_size) {

    auto voxel_size_f = float(voxel_size);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(pcl);//ptr
    sor.setLeafSize(voxel_size_f, voxel_size_f, voxel_size_f);
    sor.filter(*output);//object
    return *output;
}

void xyzrgb2xyz(const pcl::PointCloud<pcl::PointXYZRGB>& input, pcl::PointCloud<pcl::PointXYZ>&output){
    size_t n = input.size();
    output.points.resize(n);
    for(size_t i = 0; i< n; ++i){
        output.points[i].x = input.points[i].x;
        output.points[i].y = input.points[i].y;
        output.points[i].z = input.points[i].z;
    }
}

pcl::PointCloud<pcl::PointXYZRGB> pclVoxelFilter(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pcl, double voxel_size) {

    auto voxel_size_f = float(voxel_size);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(pcl);//ptr
    sor.setLeafSize(voxel_size_f, voxel_size_f, voxel_size_f);
    sor.filter(*output);//object
    return *output;
}

//visualize
void scanPrint2D(ros::Publisher & cloud_pub, PointMatrix & scan){//ok

    sensor_msgs::PointCloud cloud_msg = matrix2DtoPclMsg(scan.point, scan.num_point);
    cloud_pub.publish(cloud_msg);
}

void scanPrint3D(ros::Publisher& cloud_pub, PointMatrix & scan, int skip_step){//ok
    sensor_msgs::PointCloud cloud_msg = matrix3DtoPclMsg(scan.point, scan.num_point, skip_step);
    cloud_pub.publish(cloud_msg);
}

void posePrint(ros::Publisher& cloud_pub, Eigen::Matrix<double, 3, Eigen::Dynamic> & pose, int pose_num){//ok
    sensor_msgs::PointCloud cloud_msg = matrix3DtoPclMsg(pose, pose_num, 1);
    cloud_pub.publish(cloud_msg);
}

bool visualizeArrow(ros::Publisher& lines_pub, PointMatrix start_points, PointMatrix end_points){
    visualization_msgs::Marker marker;
    // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "/map";
    marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "basic_shapes";
    marker.id = 0;
    // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
    marker.type = visualization_msgs::Marker::LINE_LIST;
    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    for(int i_point = 0; i_point < start_points.num_point; i_point ++){
        geometry_msgs::Point p_start, p_end;
        p_start.x = start_points.point(0, i_point);
        p_start.y = start_points.point(1, i_point);
        p_start.z = start_points.point(2, i_point);
        p_end.x = end_points.point(0, i_point);
        p_end.y = end_points.point(1, i_point);
        p_end.z = end_points.point(2, i_point);
        marker.points.push_back(p_start);
        marker.points.push_back(p_end);
    }
    double scale = 0.1;
    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();
    lines_pub.publish(marker);
}

//others
double getPosi(double x_min, double y_min, double z_min, double grid){
    double posi = x_min/double(grid)*100000 +
                  z_min/double(grid)*1 +
                  y_min/double(grid)*0.00001;// +  _direction*0.1;
    return posi;
}
double getPosiWithTime(double x_min, double y_min, double z_min, double grid, int visited_times){
    double posi = visited_times * 10000000000.0 +
                  x_min/double(grid)*100000 +
                  z_min/double(grid)*1 +
                  y_min/double(grid)*0.00001;// +  _direction*0.1;
    return posi;
}
float computeMSE(const PointMatrix & points_source, const PointMatrix & points_target){
    //target = map_glb
    pcl::PointCloud<pcl::PointXYZ> turned_pcl_source = matrix3D2pcl(points_source);
    pcl::PointCloud<pcl::PointXYZ> turned_pcl_target = matrix3D2pcl(points_target);
    pcl::PCLPointCloud2::Ptr cloud_target (new pcl::PCLPointCloud2 ());
    pcl::PCLPointCloud2::Ptr cloud_source (new pcl::PCLPointCloud2 ());
    pcl::toPCLPointCloud2(turned_pcl_source, *cloud_source);
    pcl::toPCLPointCloud2(turned_pcl_target, *cloud_target);


    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_source (new pcl::PointCloud<pcl::PointXYZ> ());
    fromPCLPointCloud2 (*cloud_source, *xyz_source);
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_target (new pcl::PointCloud<pcl::PointXYZ> ());
    fromPCLPointCloud2 (*cloud_target, *xyz_target);

    pcl::PCLPointCloud2 output;
    const std::string &correspondence_type = "nn";

    pcl::PointCloud<pcl::PointXYZI>::Ptr output_xyzi (new pcl::PointCloud<pcl::PointXYZI> ());
    output_xyzi->points.resize (xyz_source->points.size ());
    output_xyzi->height = xyz_source->height;
    output_xyzi->width = xyz_source->width;

    float rmse = 0.0f;

    if (correspondence_type == "index"){
        if (xyz_source->points.size () != xyz_target->points.size ()){
            pcl::console::print_error ("Source and target clouds do not have the same number of points.\n");
            return -1;
        }

        for (std::size_t point_i = 0; point_i < xyz_source->points.size (); ++point_i){
            if (!std::isfinite (xyz_source->points[point_i].x) || !std::isfinite (xyz_source->points[point_i].y) || !std::isfinite (xyz_source->points[point_i].z))
                continue;
            if (!std::isfinite (xyz_target->points[point_i].x) || !std::isfinite (xyz_target->points[point_i].y) || !std::isfinite (xyz_target->points[point_i].z))
                continue;


            float dist = squaredEuclideanDistance (xyz_source->points[point_i], xyz_target->points[point_i]);
            rmse += dist;

            output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
            output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
            output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
            output_xyzi->points[point_i].intensity = dist;
        }
        rmse = std::sqrt (rmse / static_cast<float> (xyz_source->points.size ()));
    }
    else if (correspondence_type == "nn"){
//    print_highlight (stderr, "Computing using the nearest neighbor correspondence heuristic.\n");

        pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
        tree->setInputCloud (xyz_target);

        for (std::size_t point_i = 0; point_i < xyz_source->points.size (); ++ point_i){
            if (!std::isfinite (xyz_source->points[point_i].x) || !std::isfinite (xyz_source->points[point_i].y) || !std::isfinite (xyz_source->points[point_i].z))
                continue;

            std::vector<int> nn_indices (1);
            std::vector<float> nn_distances (1);
            if (!tree->nearestKSearch (xyz_source->points[point_i], 1, nn_indices, nn_distances))
                continue;
            std::size_t point_nn_i = nn_indices.front();

            float dist = squaredEuclideanDistance (xyz_source->points[point_i], xyz_target->points[point_nn_i]);
            rmse += dist;

            output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
            output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
            output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
            output_xyzi->points[point_i].intensity = dist;
        }
        rmse = std::sqrt (rmse / static_cast<float> (xyz_source->points.size ()));

    }
    else if (correspondence_type == "nnplane")
    {
//    print_highlight (stderr, "Computing using the nearest neighbor plane projection correspondence heuristic.\n");

        pcl::PointCloud<pcl::Normal>::Ptr normals_target (new pcl::PointCloud<pcl::Normal> ());
        //fromPCLPointCloud2 (*cloud_target, *normals_target);

        pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
        tree->setInputCloud (xyz_target);

        for (std::size_t point_i = 0; point_i < xyz_source->points.size (); ++ point_i)
        {
            if (!std::isfinite (xyz_source->points[point_i].x) || !std::isfinite (xyz_source->points[point_i].y) || !std::isfinite (xyz_source->points[point_i].z))
                continue;

            std::vector<int> nn_indices (1);
            std::vector<float> nn_distances (1);
            if (!tree->nearestKSearch (xyz_source->points[point_i], 1, nn_indices, nn_distances))
                continue;
            std::size_t point_nn_i = nn_indices.front();

            Eigen::Vector3f normal_target = normals_target->points[point_nn_i].getNormalVector3fMap (),
                    point_source = xyz_source->points[point_i].getVector3fMap (),
                    point_target = xyz_target->points[point_nn_i].getVector3fMap ();

            float dist = normal_target.dot (point_source - point_target);
            rmse += dist * dist;

            output_xyzi->points[point_i].x = xyz_source->points[point_i].x;
            output_xyzi->points[point_i].y = xyz_source->points[point_i].y;
            output_xyzi->points[point_i].z = xyz_source->points[point_i].z;
            output_xyzi->points[point_i].intensity = dist * dist;
        }
        rmse = std::sqrt (rmse / static_cast<float> (xyz_source->points.size ()));
    }
    else
    {
//    print_error ("Unrecognized correspondence type. Check legal arguments by using the -h option\n");
        return rmse;
    }

    //toPCLPointCloud2 (*output_xyzi, output);

    //pcl::console::print_highlight ("RMSE Error: %f\n", rmse);
    return rmse;
}
bool extendEigen1dVector(Eigen::Matrix<double, 1, Eigen::Dynamic> & ori_vector, int extend_length){

    ori_vector.conservativeResize(Eigen::NoChange_t(1), ori_vector.cols() + extend_length);
    ori_vector.rightCols(extend_length).fill(0);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

// Möller-Trumbore射线-三角形求交算法
std::optional<Vec3d> rayTriangleIntersectionB(
    const Ray& ray, 
    const Vec3d& v0, 
    const Vec3d& v1, 
    const Vec3d& v2) {
    
    const float EPSILON = 1e-6f;
    
    Vec3d edge1 = v1 - v0;
    Vec3d edge2 = v2 - v0;
    Vec3d h = ray.direction.cross(edge2);
    float det = edge1.dot(h);
    
    // 射线与三角形平行?
    if (std::abs(det) < EPSILON) {
        return std::nullopt;
    }
    
    float invDet = 1.0f / det;
    Vec3d s = ray.origin - v0;
    float u = invDet * s.dot(h);
    
    // u不在[0,1]范围内
    if (u < 0.0f || u > 1.0f) {
        return std::nullopt;
    }
    
    Vec3d q = s.cross(edge1);
    float v = invDet * ray.direction.dot(q);
    
    // v不在[0,1]范围内或u+v>1
    if (v < 0.0f || u + v > 1.0f) {
        return std::nullopt;
    }
    
    // 计算t值
    float t = invDet * edge2.dot(q);
    
    // 交点在射线正方向上
    if (t > EPSILON) {
        return ray.at(t);
    }
    
    return std::nullopt;
};

// 射线与网格求交
std::optional<Vec3d> rayGridIntersection(const Ray& ray, const HeightGrid& grid) {
    // 1. 计算射线与网格包围盒的交点
    float t_min = std::numeric_limits<float>::lowest();
    float t_max = std::numeric_limits<float>::max();
    
    // 检查与x平面的交点
    if (std::abs(ray.direction.x()) > 1e-6) {
        float tx1 = (grid.x_min - ray.origin.x()) / ray.direction.x();
        float tx2 = (grid.x_max - ray.origin.x()) / ray.direction.x();
        t_min = std::max(t_min, std::min(tx1, tx2));
        t_max = std::min(t_max, std::max(tx1, tx2));
    }
    
    // 检查与y平面的交点
    if (std::abs(ray.direction.y()) > 1e-6) {
        float ty1 = (grid.y_min - ray.origin.y()) / ray.direction.y();
        float ty2 = (grid.y_max - ray.origin.y()) / ray.direction.y();
        t_min = std::max(t_min, std::min(ty1, ty2));
        t_max = std::min(t_max, std::max(ty1, ty2));
    }
    
    // 射线与网格包围盒不相交
    if (t_min > t_max || t_max < 0) {
        return std::nullopt;
    }
    
    // 确保t_min至少为0
    t_min = std::max(t_min, 0.0f);
    
    // 2. 计算射线进入网格的点
    Vec3d entry_point = ray.at(t_min);
    
    // 3. 计算起始网格索引
    int i0 = static_cast<int>((entry_point.x() - grid.x_min) / grid.dx());
    int j0 = static_cast<int>((entry_point.y() - grid.y_min) / grid.dy());
    
    // 确保索引在范围内
    i0 = std::clamp(i0, 0, grid.nx - 2);
    j0 = std::clamp(j0, 0, grid.ny - 2);
    
    // 4. 计算步进方向和增量
    int di = (ray.direction.x() > 0) ? 1 : -1;
    int dj = (ray.direction.y() > 0) ? 1 : -1;
    
    // 计算到下一个网格边的距离
    float tx_next, ty_next;
    if (di > 0) {
        tx_next = grid.x_min + (i0 + 1) * grid.dx();
    } else {
        tx_next = grid.x_min + i0 * grid.dx();
    }
    
    if (dj > 0) {
        ty_next = grid.y_min + (j0 + 1) * grid.dy();
    } else {
        ty_next = grid.y_min + j0 * grid.dy();
    }
    
    // 计算t增量
    float t_delta_x = (di != 0) ? grid.dx() / std::abs(ray.direction.x()) : std::numeric_limits<float>::max();
    float t_delta_y = (dj != 0) ? grid.dy() / std::abs(ray.direction.y()) : std::numeric_limits<float>::max();
    
    // 计算到下一个边界的t值
    float t_x = (tx_next - ray.origin.x()) / ray.direction.x();
    float t_y = (ty_next - ray.origin.y()) / ray.direction.y();
    
    // 5. 遍历网格单元
    int i = i0, j = j0;
    while (i >= 0 && i < grid.nx - 1 && j >= 0 && j < grid.ny - 1) {
        // 获取当前网格单元的四个顶点
        Vec3d p00 = grid.get_point(i, j);
        Vec3d p10 = grid.get_point(i + 1, j);
        Vec3d p01 = grid.get_point(i, j + 1);
        Vec3d p11 = grid.get_point(i + 1, j + 1);
        
        // 检查与两个三角形的交点
        // 三角形1: (00, 10, 11)
        auto intersect1 = rayTriangleIntersectionB(ray, p00, p10, p11);
        if (intersect1) {
            return intersect1;
        }
        
        // 三角形2: (00, 11, 01)
        auto intersect2 = rayTriangleIntersectionB(ray, p00, p11, p01);
        if (intersect2) {
            return intersect2;
        }
        
        // 移动到下一个网格单元
        if (t_x < t_y) {
            t_x += t_delta_x;
            i += di;
        } else {
            t_y += t_delta_y;
            j += dj;
        }
    }
    
    return std::nullopt;
};





