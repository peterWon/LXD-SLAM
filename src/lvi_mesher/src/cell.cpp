#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp> // needed for L2 distance; skip PCL_NO_PRECOMPILE
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>  // needed for L1 distance
#include "cell.h"
#include "utility/tool.h"


void VoxelFilter2D(int num_input, int & num_output, double grid,
                   Matrix1xd & f, Matrix1xd & train_x, Matrix1xd & train_y, int num_test){
    //principled and fast 2D voxel filter to downsample the training points in GP process. Faster than kdtree.
    double interval = grid / num_test;

    double bot_x = floor(train_x(0,0) / grid) * grid;
    double bot_y = floor(train_y(0,0) / grid) * grid;

    Eigen::MatrixXd point_result(4, num_test * num_test);
    point_result.topRows(3).fill(0);
    point_result.row(3).fill(-1);// -1 means not filled yet

    //downsample mode chose
    bool just_first_point = false;//true false
    bool even_closest_one_point = true;
    for(int i = 0; i < num_input; i++){
        int index = 1 * int((train_x(0, i) - bot_x) / interval) +
                  num_test * int((train_y(0,i)-bot_y) / interval);
        if(just_first_point){
            if(point_result(3, index) < 1){
                point_result(3, index) = 1;
                point_result(0, index) = train_x(0, i);
                point_result(1, index) = train_y(0, i);
                point_result(2, index) = f(0, i);
            }
        }
        else if(even_closest_one_point){
            int index_x = int((train_x(0, i) - bot_x) / interval);//0 offset
            int index_y = int((train_y(0, i) - bot_y) / interval);
            int index_linear = 1 * index_x + num_test * index_y;

            double  iy = (index_linear / num_test) * interval + bot_y + 0.5 * interval,
                    ix = (index_linear % num_test) * interval + bot_x + 0.5 * interval;
            double dis_now = pow(train_x(0,i) - ix, 2) + pow(train_y(0,i) - iy, 2);
            if(point_result(3, index_linear) < 0){
                point_result(3, index_linear) = dis_now;
                point_result(0, index_linear) = train_x(0, i);
                point_result(1, index_linear) = train_y(0, i);
                point_result(2, index_linear) = f(0, i);
            }
            else if(dis_now < point_result(3, index_linear)){
                point_result(3, index_linear) = dis_now;
                point_result(0, index_linear) = train_x(0, i);
                point_result(1, index_linear) = train_y(0, i);
                point_result(2, index_linear) = f(0, i);
            }
        }
    }

    int col = 0;
    for(size_t i = 0; i < num_test*num_test; i++){
        if(point_result(3,i) > 0){
            train_x(0, col) = point_result(0, i);
            train_y(0, col) = point_result(1, i);
            f      (0, col) = point_result(2, i);
            col ++;
        }
    }
    num_output = col;
}

Vec3d Cell::getVertice(Direction dir, int r,  int c) const{
    int n_row = Parameter::GetInstance().num_test;
    int col_idx = r * n_row + c;
    return ary_cell_vertices[dir].point.col(col_idx);
}

Vec4d Cell::getVerticeAndVariance(Direction dir, int r,  int c) const{
    int n_row = Parameter::GetInstance().num_test;
    int col_idx = r * n_row + c;
    Vec4d res;
    res.head(3) << ary_cell_vertices[dir].point.col(col_idx);
    res.tail(1) << ary_cell_vertices[dir].variance.col(col_idx);
    return res;
}
Vec3d Cell::getVertice(Direction dir, int k) const{
    return ary_cell_vertices[dir].point.col(k);
}

Vec4d Cell::getVerticeAndVariance(Direction dir, int k) const{
    Vec4d res;
    res.head(3) << ary_cell_vertices[dir].point.col(k);
    res.tail(1) << ary_cell_vertices[dir].variance.col(k);
    return res;
}

void Cell::gaussianProcess(enum Direction gp_direction){
    //standard gaussianProcess
    bool full_cover = Parameter::GetInstance().full_cover; //whether the first and final test points lie on the border or not.
    //full_cover: test point location start from 0.5*step, like 0.5, 1.5, ..., 9.5
    //if full_cover == false: test point location start from 0*step, like 0, 1, ..., 10
    int num_test_side = Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    double interval = grid / num_test_side;
    double variance_sensor = Parameter::GetInstance().variance_sensor;
    double kernel_length = 1.2;// gp kernel
    //create test point
    Eigen::MatrixXd I_test =  Eigen::MatrixXd::Identity(num_test_square, num_test_square);
    Eigen::Matrix<double, 1, Eigen::Dynamic>  test_x, test_y;
    Eigen::MatrixXd points_testm(num_test_square, 2);

    Direction prediction_dir = Direction((gp_direction + 0) % 3);
    Direction location_dir_x = Direction((gp_direction + 1) % 3);
    Direction location_dir_y = Direction((gp_direction + 2) % 3);

    //set test locations
    if(gp_direction == X){
        evenSetLinSpaced(test_x, num_test_side, region.y_min, region.y_max, full_cover);
        evenSetLinSpaced(test_y, num_test_side, region.z_min, region.z_max, full_cover);
    }
    else if(gp_direction == Y){
        evenSetLinSpaced(test_x, num_test_side, region.z_min, region.z_max, full_cover);
        evenSetLinSpaced(test_y, num_test_side, region.x_min, region.x_max, full_cover);
    }
    else if(gp_direction == Z){
        evenSetLinSpaced(test_x, num_test_side, region.x_min, region.x_max, full_cover);
        evenSetLinSpaced(test_y, num_test_side, region.y_min, region.y_max, full_cover);
    }

    for(int row=0; row < num_test_square; row++){
        points_testm(row, 0) = test_x(0, row / num_test_side);
        points_testm(row, 1) = test_y(0, row % num_test_side);
    }

    //creat train point
    int num_train = -1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> f_raw, train_x_raw, train_y_raw;
    Eigen::MatrixXd f(1, num_test_square), train_x(1, num_test_square), train_y(1, num_test_square);
    double train_x_min, train_x_max, train_y_min, train_y_max;
    
    //voxelfilter to downsample train point
    int num_train_raw = cell_raw_points.num_point;
    f_raw       = cell_raw_points.point.row(prediction_dir).leftCols(num_train_raw);
    train_x_raw = cell_raw_points.point.row(location_dir_x).leftCols(num_train_raw);//train_x---points_testm.col(0)---gp_direction + 1
    train_y_raw = cell_raw_points.point.row(location_dir_y).leftCols(num_train_raw);
    train_x_min = train_x_raw.minCoeff();
    train_x_max = train_x_raw.maxCoeff();
    train_y_min = train_y_raw.minCoeff();
    train_y_max = train_y_raw.maxCoeff();

    VoxelFilter2D(num_train_raw, num_train, grid, f_raw, train_x_raw, train_y_raw, Parameter::GetInstance().num_test);
    if(num_train > num_test_square ){
        LOG(ERROR)<<"Too many trian pointin num in gp! num_train_after_voxel_filter";
    }
    if(num_train == 0 ){
        LOG(ERROR)<<"0 trian pointin num in gp! num_train_after_voxel_filter";
    }
    f.leftCols      (num_train) = f_raw.leftCols      (num_train);
    train_x.leftCols(num_train) = train_x_raw.leftCols(num_train);
    train_y.leftCols(num_train) = train_y_raw.leftCols(num_train);

    Eigen::MatrixXd I_train = Eigen::MatrixXd::Identity(num_train, num_train);
    
    //gp
    double mean = f.leftCols(num_train).mean();
    f.leftCols(num_train).array() -= mean;

    Eigen::MatrixXd f_starm(num_test_square, 1),
             variance_starm(num_test_square, 1);
    Eigen::MatrixXd       k(num_test_square, num_test_square),
                         ky(num_test_square, num_test_square),
                     ky_inv(num_test_square, num_test_square),
                    k_starm(num_test_square, num_test_square),
                        kky(num_test_square, num_test_square);
    
    // 训练集
    for(int row = 0; row < num_train; row++){
        for(int col = 0; col < num_train; col++){
            // kernel函数，与训练点的距离相关
            k(row, col) = exp_quick(-kernel_length * sqrt(pow((train_x(0, col) - train_x(0, row)), 2) +
                                                          pow((train_y(0, col) - train_y(0, row)), 2)));

        }
    }
    // 计算测试集每个点与训练集里面每个点的kernel值
    for(int row = 0; row < num_test_square; row++){
        for(int col = 0; col < num_train; col++){
            k_starm(row, col) = exp_quick(-kernel_length * sqrt(pow((train_x(0, col) - points_testm(row, 0)), 2) +
                                                                pow((train_y(0, col) - points_testm(row, 1)), 2)));

        }
    }

    // 添加传感器测量噪声
    ky.block(0, 0, num_train, num_train) = k.block(0, 0, num_train, num_train);
    for(int row = 0; row < num_train; row++){
        ky(row, row) = k(row, row) + variance_sensor * variance_sensor;
    }

    //choose the method to calculate the inverse of ky
    //kky = k_starm.adjoint()*(ky.inverse());//PartialPivLU
    //kky = k_starm.adjoint()*(ky.ldlt().solve(I_train));//LDLT
    // GP-SLAM公式1
    kky.block(0, 0, num_test_square, num_train) = k_starm.block(0, 0, num_test_square, num_train) *
                                                  (ky.block(0, 0, num_train, num_train).llt().solve(I_train));//LLT fastest
    // 预测值
    f_starm = (kky.block(0, 0, num_test_square, num_train) * ((f.leftCols(num_train)).transpose())).array() + mean;

    variance_starm = (I_test - kky.block(0, 0, num_test_square, num_train) *
                               k_starm.block(0, 0, num_test_square, num_train).transpose()) .diagonal();
    
    // 更新测试点坐标及预测值到成员变量
    ary_cell_vertices[gp_direction] = Eigen::MatrixXd::Zero(6, num_test_square); //BUG here, xyzrgb
    ary_cell_vertices[gp_direction].point.row(prediction_dir).leftCols(num_test_square) = f_starm.transpose();
    ary_cell_vertices[gp_direction].point.row(location_dir_x).leftCols(num_test_square) = points_testm.col(0).transpose();
    ary_cell_vertices[gp_direction].point.row(location_dir_y).leftCols(num_test_square) = points_testm.col(1).transpose();
    ary_cell_vertices[gp_direction].num_point = num_test_square;
    ary_cell_vertices[gp_direction].variance.leftCols(num_test_square) = variance_starm.transpose();
    ary_cell_vertices[gp_direction].variance_sum = ary_cell_vertices[gp_direction].variance.leftCols(num_test_square).sum();
    //limit the prediction inside the cell
    bool bound_point = true;//false true
    if(bound_point){
        for(int i_point = 0; i_point < num_test_square; i_point++){
            //prediction should not exceed the region
            if(ary_cell_vertices[gp_direction].point(prediction_dir, i_point) > region.getDirMax(gp_direction)){
                ary_cell_vertices[gp_direction].point(prediction_dir, i_point) = region.getDirMax(gp_direction);
            }
            if(ary_cell_vertices[gp_direction].point(prediction_dir, i_point) < region.getDirMin(gp_direction)){
                ary_cell_vertices[gp_direction].point(prediction_dir, i_point) = region.getDirMin(gp_direction);
            }
            //if the locations of test points exceed min/max locations of training points, the variance are punished.
            if(ary_cell_vertices[gp_direction].point(location_dir_x, i_point) > train_x_max ||
               ary_cell_vertices[gp_direction].point(location_dir_x, i_point) < train_x_min ||
               ary_cell_vertices[gp_direction].point(location_dir_y, i_point) > train_y_max ||
               ary_cell_vertices[gp_direction].point(location_dir_y, i_point) < train_y_min){
               ary_cell_vertices[gp_direction].variance(0, i_point) *= 2;
            }
        }
    }
    empty = false;
}

void Cell::gaussianProcessRGB(enum Direction gp_direction, enum Color gp_color){
    //standard gaussianProcess
    bool full_cover = Parameter::GetInstance().full_cover; //whether the first and final test points lie on the border or not.
    //full_cover: test point location start from 0.5*step, like 0.5, 1.5, ..., 9.5
    //if full_cover == false: test point location start from 0*step, like 0, 1, ..., 10
    int num_test_side = Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    double interval = grid / num_test_side;
    double variance_sensor = Parameter::GetInstance().variance_sensor;
    double kernel_length = 1.2;// gp kernel
    //create test point
    Eigen::MatrixXd I_test =  Eigen::MatrixXd::Identity(num_test_square, num_test_square);
    Eigen::Matrix<double, 1, Eigen::Dynamic>  test_x, test_y;
    Eigen::MatrixXd points_testm(num_test_square, 2);

    Direction location_dir_x = Direction((gp_direction + 1) % 3);
    Direction location_dir_y = Direction((gp_direction + 2) % 3);
    
    Color prediction_channel = gp_color;

    //set test locations
    if(gp_direction == X){
        evenSetLinSpaced(test_x, num_test_side, region.y_min, region.y_max, full_cover);
        evenSetLinSpaced(test_y, num_test_side, region.z_min, region.z_max, full_cover);
    }
    else if(gp_direction == Y){
        evenSetLinSpaced(test_x, num_test_side, region.z_min, region.z_max, full_cover);
        evenSetLinSpaced(test_y, num_test_side, region.x_min, region.x_max, full_cover);
    }
    else if(gp_direction == Z){
        evenSetLinSpaced(test_x, num_test_side, region.x_min, region.x_max, full_cover);
        evenSetLinSpaced(test_y, num_test_side, region.y_min, region.y_max, full_cover);
    }

    for(int row=0; row < num_test_square; row++){
        points_testm(row, 0) = test_x(0, row / num_test_side);
        points_testm(row, 1) = test_y(0, row % num_test_side);
    }

    //creat train point
    int num_train = -1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> f_raw, train_x_raw, train_y_raw;
    Eigen::MatrixXd f(1, num_test_square), train_x(1, num_test_square), train_y(1, num_test_square);
    double train_x_min, train_x_max, train_y_min, train_y_max;
    //voxelfilter to downsample train point
    int num_train_raw = cell_raw_points.num_point;
    if(num_train_raw == 0){
        ROS_ERROR("No trian pointin num in gp! num_train_raw");
        while(ros::ok());
    }

    f_raw       = cell_raw_points.rgb.row(prediction_channel).leftCols(num_train_raw);
    train_x_raw = cell_raw_points.point.row(location_dir_x).leftCols(num_train_raw);//train_x---points_testm.col(0)---gp_direction + 1
    train_y_raw = cell_raw_points.point.row(location_dir_y).leftCols(num_train_raw);
    train_x_min = train_x_raw.minCoeff();
    train_x_max = train_x_raw.maxCoeff();
    train_y_min = train_y_raw.minCoeff();
    train_y_max = train_y_raw.maxCoeff();

    VoxelFilter2D(num_train_raw, num_train, grid, f_raw, train_x_raw, train_y_raw, Parameter::GetInstance().num_test);
    if(num_train > num_test_square ){
        ROS_ERROR("Too many trian pointin num in gp! num_train_after_voxel_filter");
        while(ros::ok());
    }
    if(num_train == 0 ){
        ROS_ERROR("0 trian pointin num in gp! num_train_after_voxel_filter");
        while(ros::ok());
    }
    f.leftCols      (num_train) = f_raw.leftCols      (num_train);
    train_x.leftCols(num_train) = train_x_raw.leftCols(num_train);
    train_y.leftCols(num_train) = train_y_raw.leftCols(num_train);

    Eigen::MatrixXd I_train = Eigen::MatrixXd::Identity(num_train, num_train);

    //gp
    double mean = f.leftCols(num_train).mean();
    f.leftCols(num_train).array() -= mean;

    Eigen::MatrixXd f_starm(num_test_square, 1),
             variance_starm(num_test_square, 1);
    Eigen::MatrixXd       k(num_test_square, num_test_square),
                         ky(num_test_square, num_test_square),
                     ky_inv(num_test_square, num_test_square),
                    k_starm(num_test_square, num_test_square),
                        kky(num_test_square, num_test_square);
    // 训练集
    for(int row = 0; row < num_train; row++){
        for(int col = 0; col < num_train; col++){
            // kernel函数，与训练点的距离相关
            k(row, col) = exp_quick(-kernel_length * sqrt(pow((train_x(0, col) - train_x(0, row)), 2) +
                                                          pow((train_y(0, col) - train_y(0, row)), 2)));

        }
    }
    // 计算测试集每个点与训练集里面每个点的kernel值
    for(int row = 0; row < num_test_square; row++){
        for(int col = 0; col < num_train; col++){
            k_starm(row, col) = exp_quick(-kernel_length * sqrt(pow((train_x(0, col) - points_testm(row, 0)), 2) +
                                                                pow((train_y(0, col) - points_testm(row, 1)), 2)));
        }
    }
    // 添加传感器测量噪声
    ky.block(0, 0, num_train, num_train) = k.block(0, 0, num_train, num_train);
    for(int row = 0; row < num_train; row++){
        ky(row, row) = k(row, row) + variance_sensor * variance_sensor;
    }
    //choose the method to calculate the inverse of ky
    //kky = k_starm.adjoint()*(ky.inverse());//PartialPivLU
    //kky = k_starm.adjoint()*(ky.ldlt().solve(I_train));//LDLT
    // GP-SLAM公式1
    kky.block(0, 0, num_test_square, num_train) = k_starm.block(0, 0, num_test_square, num_train) *
                                                  (ky.block(0, 0, num_train, num_train).llt().solve(I_train));//LLT fastest
    // 预测值
    f_starm = (kky.block(0, 0, num_test_square, num_train) * ((f.leftCols(num_train)).transpose())).array() + mean;

    variance_starm = (I_test - kky.block(0, 0, num_test_square, num_train) *
                               k_starm.block(0, 0, num_test_square, num_train).transpose()) .diagonal();
    
    // 更新色值
    ary_cell_vertices[gp_direction].rgb.row(prediction_channel).leftCols(num_test_square) = f_starm.transpose();

    // 计算该方向下测试点的颜色梯度

    // 更新测试点坐标及预测值到成员变量，由location gaussian process完成
    // ary_cell_vertices[gp_direction] = Eigen::MatrixXd::Zero(3, num_test_square);
    // ary_cell_vertices[gp_direction].point.row(location_dir_x).leftCols(num_test_square) = points_testm.col(0).transpose();
    // ary_cell_vertices[gp_direction].point.row(location_dir_y).leftCols(num_test_square) = points_testm.col(1).transpose();
    // ary_cell_vertices[gp_direction].num_point = num_test_square;
    // ary_cell_vertices[gp_direction].variance.leftCols(num_test_square) = variance_starm.transpose();
    // ary_cell_vertices[gp_direction].variance_sum = ary_cell_vertices[gp_direction].variance.leftCols(num_test_square).sum();
    
    //limit the prediction inside the range
    bool bound_point = true;
    if(bound_point){
        for(int i_point = 0; i_point < num_test_square; i_point++){
            //prediction should not exceed the region
            if(ary_cell_vertices[gp_direction].rgb(prediction_channel, i_point) > 255){
                ary_cell_vertices[gp_direction].rgb(prediction_channel, i_point) = 255;
            }
            if(ary_cell_vertices[gp_direction].rgb(prediction_channel, i_point) < 0){
                ary_cell_vertices[gp_direction].rgb(prediction_channel, i_point) = 0;
            }
            // if the locations of test points exceed min/max locations of training points, the variance are punished.
            // updated by location gaussian process
        }
    }
    empty = false;
}

void Cell::ColorGradient(enum Direction gp_direction){
    const auto& rgb_raw       = cell_raw_points.rgb;
    const auto& xyz_raw       = cell_raw_points.point;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target(new pcl::PointCloud<pcl::PointXYZRGB>);
    target->resize(xyz_raw.cols());
    for(size_t i = 0; i < xyz_raw.cols(); i++){
        pcl::PointXYZRGB point;
        point.x = xyz_raw.col(i)(0);
        point.y = xyz_raw.col(i)(1);
        point.z = xyz_raw.col(i)(2);
        point.r = rgb_raw.col(i)(0);
        point.g = rgb_raw.col(i)(1);
        point.b = rgb_raw.col(i)(2);
        target->points[i] = point;
    }

    double search_radius = 0.5;
    
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(target);

    const PointMatrix& pts_test = ary_cell_vertices[gp_direction];
    Vec3d rgb_to_intensity_weight_ = Vec3d::Ones() / 3.;
    int nn=0;
    for(int k = 0; k < pts_test.num_point; k++){
        const Vec3d p = pts_test.point.col(k);
        const Vec3d cp = pts_test.rgb.col(k);
        double ip = cp.dot(rgb_to_intensity_weight_) / 255.0;
        
        // kdtree默认返回为sorted
        std::vector<int> indices(1);
        std::vector<float> sq_dist(1);
        pcl::PointXYZRGB p_search(p.x(), p.y(), p.z());
        kdtree->radiusSearch(p_search, search_radius, indices, sq_dist);
        size_t nn_size = indices.size();
        if(nn_size < 5){
            continue;
        }

        // 估计法向量
        std::vector<Vec3d> near_points;
        for(int i = 0; i < 5; ++i){
            near_points.push_back(target->points[indices[i]].getVector3fMap().cast<double>());
        }
        Vec4d plane_para;
        estiPlane(near_points, 0.1f, plane_para);
        Vec3d np = plane_para.head<3>();

        // 最小二乘估计颜色梯度
        Eigen::Matrix3Xd A(3, nn_size);
        Eigen::VectorXd b(nn_size);
        for (int j = 0; j < nn_size; ++j) {
            const Vec3d pp = target->points[indices[j]].getVector3fMap().cast<double>();
            const Vec3d cpp = target->points[indices[j]].getRGBVector3i().cast<double>();
            double ipp = cpp.dot(rgb_to_intensity_weight_) / 255.0;

            // pp在tangent plane上的对应点
            const Vec3d& pp_proj = pp - (pp-p).dot(np)*np;
            // tangent plane上p指向pp的向量
            A.block<3, 1>(0, j) = pp_proj - p;
            // 近邻点与当前点的颜色差值
            b(j) = ipp - ip;
        }
        // add orthogonal constraint dp.dot(np) = 0 with weight (nn_size-1)
        A.block<3, 1>(0, nn_size-1) = (nn_size-1) * np;
        b (nn_size-1) = 0; 

        // solve for dp
        Vec3d X = (A * A.transpose()).ldlt().solve(A * b);
        ary_cell_vertices[gp_direction].color_gradient.col(k) = X;
        nn+=1;
    }
    // LOG(INFO)<<"Estimated color gradient for "<<nn<<" points in direction "<<gp_direction<<", total points: "<<xyz_raw.cols();
}


void Cell::reconstructSurfaces(bool glb_cell_not_surface){
    //use gaussian process to reconstruct the local surfaces inside a cell, one cell can have 3 surfaces in 3 directions
    //TicToc t_mc_gp_init;
    double num_test = Parameter::GetInstance().num_test;
    double eig_3ratio2_not_surface = Parameter::GetInstance().eigen_1,
           eig_2minus1_distinctive_angle = Parameter::GetInstance().eigen_2,
           eig_3minus2_obscure_angle = Parameter::GetInstance().eigen_3;
    for(auto & i_vertices : ary_cell_vertices){
        i_vertices.clear();
    }
    //decide direction of GP, based on PCA (Principal Component Analysis)
    bool direction_list[3];
    memset(direction_list, false, sizeof(direction_list));
    memset(updated_times, 0, sizeof(updated_times));
    Eigen::Matrix<double, 1, 3> angle_a, angle_b;
    // std::map<double, int> sorted_angle_a, sorted_angle_b;
    std::vector<std::pair<double, int>> sorted_angle_a;
    std::vector<std::pair<double, int>> sorted_angle_b;
    
    // xyz主成分分析
    cell_raw_points.eigenDecomposition();
    angle_a = (cell_raw_points.eig_sorted.begin() ->second.transpose() * Eigen::MatrixXd::Identity(3, 3)).array().acos();
    angle_b = (cell_raw_points.eig_sorted.rbegin()->second.transpose() * Eigen::MatrixXd::Identity(3, 3)).array().acos();

    // 钝角转锐角 + 填充vector（绑定角度和索引）
    for(int i = 0; i < 3; i++){
        // 钝角转锐角（原有逻辑不变）
        angle_a(0, i) = (angle_a(0, i) > M_PI / 2) ? M_PI - angle_a(0, i) : angle_a(0, i);
        angle_b(0, i) = (angle_b(0, i) > M_PI / 2) ? M_PI - angle_b(0, i) : angle_b(0, i);
        
        // 替换map的emplace：直接push_back到vector，确保3个元素都保留（即使角度相等）
        sorted_angle_a.push_back(std::make_pair(angle_a(0, i), i));
        sorted_angle_b.push_back(std::make_pair(angle_b(0, i), i));
    }

    // 对vector按角度值（pair.first）从小到大排序（核心步骤）
    std::sort(sorted_angle_a.begin(), sorted_angle_a.end(), 
            [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first < p2.first; // 按角度升序排列
            });
    std::sort(sorted_angle_b.begin(), sorted_angle_b.end(), 
            [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first < p2.first;
            });
    // 最小主成分特征值与次小主成分特征值之比判断是否是表面
    if(cell_raw_points.eig_sorted.rbegin()->first / (++cell_raw_points.eig_sorted.begin())->first > eig_3ratio2_not_surface){//30
        //3/2 > threshold, not surface
        not_surface = true;
        // Log::GetInstance().not_a_surface_cell_num(0, Log::GetInstance().step) ++;
        //unless map_glb say its cell on this region is a surface, no reconstruction will be conducted
        if(!glb_cell_not_surface){
            // sorted_angle_b[0]是最小角度对应的索引，后续逻辑不变（仅替换map为vector访问）
            direction_list[(sorted_angle_b[0].second + 1) % 3] = true;//do not predict the max beta angle direction
            direction_list[(sorted_angle_b[0].second + 2) % 3] = true;
        }
    }
    else{
        //can form surface(s)
        not_surface = false;
        // 次小角度（index1） - 最小角度（index0）：直接通过vector索引访问，无迭代器问题
        if(sorted_angle_a[1].first - sorted_angle_a[0].first > eig_2minus1_distinctive_angle){//0.2
            //2-1 > threshold, distinctive angle, only one gp function is enough
            direction_list[sorted_angle_a[0].second] = true; // 最小角度对应的索引
        }
        else {
            //<0.2：最大角度（index2） - 次小角度（index1）
            if(sorted_angle_a[2].first - sorted_angle_a[1].first > eig_3minus2_obscure_angle){//0.2
                //3-2 obscure angle(OA)：两个主成分（最小角度+次小角度）
                
                direction_list[sorted_angle_a[0].second] = true; // index0：最小角度索引
                direction_list[sorted_angle_a[1].second] = true; // index1：次小角度索引（绝对合法，0/1/2）
            }
            else{
                //2-1<0.2 && 3-2<0.2, use 3 gp function
                for(auto & i : direction_list) i = true;//all is true
            }
        }
    }
    //carry out reconstruction
    // 对每个需要表面重建的方向执行高斯过程拟合
    for(int i = 0; i < 3; i++){
        if(direction_list[i]){
            gaussianProcess(Direction(i));
            // for(int j = 0; j < 3; j++){
            //     gaussianProcessRGB(Direction(i), Color(j));
            // }
            // 计算该方向下测试点的颜色(灰度值)梯度
            // ColorGradient(Direction(i));
            updated_times[i] = 1;
        }
        else{
            updated_times[i] = 0;
        }
    }
}

void Cell::updateVerticeColor(Direction dir, int idx, Vec3d color, double distance, Vec3d view_dir){
    int ns = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int v_idx = dir * ns + idx;
    VertexObservation obs;
    obs.vertexIndex = v_idx;
    obs.color = color;
    obs.distance = distance;
    obs.viewDirection = view_dir;
    vertex_color_predictor_.addObservations({obs});
    vertex_color_predictor_.trainVertex(v_idx);

    Eigen::Vector3d variance;//not used now
    ary_cell_vertices[dir].rgb.col(idx) = vertex_color_predictor_.predictColor(
        v_idx, Vec3d(0,0,1), 2.0, variance);
}
void Cell::updateVertices(Cell & cell_new, enum Direction update_direction){
    //update the vertices of a local surface (with certain prediction direction) inside a cell
    ROS_DEBUG("updateVertices");
    //init
    //double dyna_thr_grid_ratio = 0.5;
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    double variance_map_update = Parameter::GetInstance().variance_map_update;
    Eigen::Matrix<double, 1, Eigen::Dynamic> update, observe, variance_update ,variance_observe;
    update = observe = variance_update = variance_observe = Eigen::MatrixXd::Zero(1, num_test_square);

    PointMatrix & points_old = ary_cell_vertices[update_direction];
    PointMatrix & points_new = cell_new.ary_cell_vertices[update_direction];

    update = points_old.point.leftCols(num_test_square).row(update_direction);
    observe = points_new.point.leftCols(num_test_square).row(update_direction);
    variance_update = points_old.variance.leftCols(num_test_square);
    variance_observe = points_new.variance.leftCols(num_test_square);

    //iteratively least square update
    for(int i = 0; i < num_test_square; i++){
        if((variance_update(0, i) <= variance_map_update) && (variance_observe(0, i) <= variance_map_update)){
            update(0, i)   = (update(0, i) * variance_observe(0, i) + observe(0, i) * variance_update(0, i))
                             / (variance_update(0, i) + variance_observe(0, i));
            variance_update(0, i) = variance_update(0, i) * variance_observe(0, i) / (variance_update(0, i) + variance_observe(0, i));
        }
        else if((variance_update(0, i) > variance_map_update) || (variance_observe(0, i) > variance_map_update)){
            update(0, i)   = (variance_update(0, i) <= variance_observe(0, i)) ? (update  (0, i)) : (observe (0, i));
            variance_update(0, i) = (variance_update(0, i) <= variance_observe(0, i)) ? (variance_update(0, i)) : (variance_observe(0, i));
        }
    }
    updated_times[update_direction] ++;

    points_old.point.leftCols(num_test_square).row(update_direction) = update;
    points_old.variance.leftCols(num_test_square) = variance_update;

    points_old.variance_sum = points_old.variance.leftCols(num_test_square).sum();
}

void Cell::updateViewedLocation(const Transf & transf_viewed) {
    //iteratively update the view direction from sensor to certain surface inside a cell
    ROS_DEBUG("updateViewedLocation");
    Point viewed_pose = trans3Dpoint(0, 0, 0, transf_viewed);
    Point viewed_arrow = viewed_pose - center;
    Point current_viewed_dir = viewed_arrow / viewed_arrow.norm();

    viewed_location = (viewed_location * viewed_dir_count + current_viewed_dir) /
                      (viewed_location * viewed_dir_count + current_viewed_dir).norm();
    average_viewed_distance = (average_viewed_distance * viewed_dir_count + viewed_arrow.norm()) / (1 + viewed_dir_count);
    viewed_dir_count ++;
}

// 射线与cell网格求交
std::optional<Vec4d> Cell::rayCellIntersectionX(const Ray& ray) const {
    Direction dir = Direction::X;
    if(ary_cell_vertices[dir].num_point==0) return std::nullopt;;

    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'


    
    // 1. 计算射线与网格包围盒的交点
    float t_min = std::numeric_limits<float>::lowest();
    float t_max = std::numeric_limits<float>::max();
    
    // 检查与y平面的交点
    if (std::abs(ray.direction.y()) > 1e-6) {
        float tx1 = (region.y_min - ray.origin.y()) / ray.direction.y();
        float tx2 = (region.y_max - ray.origin.y()) / ray.direction.y();
        t_min = std::max(t_min, std::min(tx1, tx2));
        t_max = std::min(t_max, std::max(tx1, tx2));
    }
    
    // 检查与z平面的交点
    if (std::abs(ray.direction.z()) > 1e-6) {
        float ty1 = (region.z_min - ray.origin.z()) / ray.direction.z();
        float ty2 = (region.z_max - ray.origin.z()) / ray.direction.z();
        t_min = std::max(t_min, std::min(ty1, ty2));
        t_max = std::min(t_max, std::max(ty1, ty2));
    }
    
    // 射线与Cell包围盒不相交
    if (t_min > t_max || t_max < 0) {
        return std::nullopt;
    }
    
    // 确保t_min至少为0
    t_min = std::max(t_min, 0.0f);
    
    // 2. 计算射线进入cell的点
    Vec3d entry_point = ray.at(t_min);
    
    // 3. 计算起始网格索引
    int i0 = static_cast<int>((entry_point.y() - region.y_min) / resolution);
    int j0 = static_cast<int>((entry_point.z() - region.z_min) / resolution);
    
    // 确保索引在范围内
    i0 = std::clamp(i0, 0, n_row - 2);
    j0 = std::clamp(j0, 0, n_row - 2);
    
    // 4. 计算步进方向和增量
    int di = (ray.direction.y() > 0) ? 1 : -1;
    int dj = (ray.direction.z() > 0) ? 1 : -1;
    
    // 计算到下一个网格边的距离
    float tx_next, ty_next;
    if (di > 0) {
        tx_next = region.y_min + (i0 + 1) * resolution;
    } else {
        tx_next = region.y_min + i0 * resolution;
    }
    
    if (dj > 0) {
        ty_next = region.z_min + (j0 + 1) * resolution;
    } else {
        ty_next = region.z_min + j0 * resolution;
    }
    
    // 计算t增量
    float t_delta_x = (di != 0) ? resolution / std::abs(ray.direction.y()) : std::numeric_limits<float>::max();
    float t_delta_y = (dj != 0) ? resolution / std::abs(ray.direction.z()) : std::numeric_limits<float>::max();
    
    // 计算到下一个边界的t值
    float t_x = (tx_next - ray.origin.y()) / ray.direction.y();
    float t_y = (ty_next - ray.origin.z()) / ray.direction.z();
    
    // 5. 遍历网格单元
    int i = i0, j = j0;
    while (i >= 0 && i < n_row - 1 && j >= 0 && j < n_row - 1) {
        // 获取当前网格单元的四个顶点
        Vec4d p00 = getVerticeAndVariance(dir, i, j);
        Vec4d p10 = getVerticeAndVariance(dir, i + 1, j);
        Vec4d p01 = getVerticeAndVariance(dir, i, j + 1);
        Vec4d p11 = getVerticeAndVariance(dir, i + 1, j + 1);
        
        // 检查与两个三角形的交点
        // 三角形1: (00, 10, 11)
        auto intersect1 = rayTriangleIntersectionB(ray, p00.head(3), p10.head(3), p11.head(3));
        if (intersect1) {
            Vec4d pt_var;
            double max_var = std::max(p00(3), p10(3));
            max_var = std::max(max_var, p11(3));
            pt_var.head(3) << (*intersect1);
            pt_var.tail(1) << max_var;
            return pt_var;
        }
        
        // 三角形2: (00, 11, 01)
        auto intersect2 = rayTriangleIntersectionB(ray, p00.head(3), p11.head(3), p01.head(3));
        if (intersect2) {
            Vec4d pt_var;
            double max_var = std::max(p00(3), p11(3));
            max_var = std::max(max_var, p01(3));
            pt_var.head(3) << (*intersect2);
            pt_var.tail(1) << max_var;
            return pt_var;
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

// 射线与cell网格求交
std::optional<Vec4d> Cell::rayCellIntersectionY(const Ray& ray) const {
    Direction dir = Direction::Y;
    if(ary_cell_vertices[dir].num_point==0) return std::nullopt;;

    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'
   
    // 1. 计算射线与网格包围盒的交点
    float t_min = std::numeric_limits<float>::lowest();
    float t_max = std::numeric_limits<float>::max();
    
    // 检查与z平面的交点
    if (std::abs(ray.direction.z()) > 1e-6) {
        float tz1 = (region.z_min - ray.origin.z()) / ray.direction.z();
        float tz2 = (region.z_max - ray.origin.z()) / ray.direction.z();
        t_min = std::max(t_min, std::min(tz1, tz2));
        t_max = std::min(t_max, std::max(tz1, tz2));
    }
    
    // 检查与x平面的交点
    if (std::abs(ray.direction.x()) > 1e-6) {
        float tx1 = (region.x_min - ray.origin.x()) / ray.direction.x();
        float tx2 = (region.x_max - ray.origin.x()) / ray.direction.x();
        t_min = std::max(t_min, std::min(tx1, tx2));
        t_max = std::min(t_max, std::max(tx1, tx2));
    }
    
    // 射线与Cell包围盒不相交
    if (t_min > t_max || t_max < 0) {
        return std::nullopt;
    }
    
    // 确保t_min至少为0
    t_min = std::max(t_min, 0.0f);
    
    // 2. 计算射线进入cell的点
    Vec3d entry_point = ray.at(t_min);
    
    // 3. 计算起始网格索引
    int i0 = static_cast<int>((entry_point.z() - region.z_min) / resolution);
    int j0 = static_cast<int>((entry_point.x() - region.x_min) / resolution);
    
    // 确保索引在范围内
    i0 = std::clamp(i0, 0, n_row - 2);
    j0 = std::clamp(j0, 0, n_row - 2);
    
    // 4. 计算步进方向和增量
    int di = (ray.direction.z() > 0) ? 1 : -1;
    int dj = (ray.direction.x() > 0) ? 1 : -1;
    
    // 计算到下一个网格边的距离
    float tz_next, tx_next;
    if (di > 0) {
        tz_next = region.z_min + (i0 + 1) * resolution;
    } else {
        tz_next = region.z_min + i0 * resolution;
    }
    
    if (dj > 0) {
        tx_next = region.x_min + (j0 + 1) * resolution;
    } else {
        tx_next = region.x_min + j0 * resolution;
    }
    
    // 计算t增量
    float t_delta_z = (di != 0) ? resolution / std::abs(ray.direction.z()) : std::numeric_limits<float>::max();
    float t_delta_x = (dj != 0) ? resolution / std::abs(ray.direction.x()) : std::numeric_limits<float>::max();
    
    // 计算到下一个边界的t值
    float t_z = (tz_next - ray.origin.z()) / ray.direction.z();
    float t_x = (tx_next - ray.origin.x()) / ray.direction.x();
    
    // 5. 遍历网格单元
    int i = i0, j = j0;
    while (i >= 0 && i < n_row - 1 && j >= 0 && j < n_row - 1) {
        // 获取当前网格单元的四个顶点
        Vec4d p00 = getVerticeAndVariance(dir, i, j);
        Vec4d p10 = getVerticeAndVariance(dir, i + 1, j);
        Vec4d p01 = getVerticeAndVariance(dir, i, j + 1);
        Vec4d p11 = getVerticeAndVariance(dir, i + 1, j + 1);
        
        // 检查与两个三角形的交点
        // 三角形1: (00, 10, 11)
        auto intersect1 = rayTriangleIntersectionB(ray, p00.head(3), p10.head(3), p11.head(3));
        if (intersect1) {
            Vec4d pt_var;
            double max_var = std::max(p00(3), p10(3));
            max_var = std::max(max_var, p11(3));
            pt_var.head(3) << (*intersect1);
            pt_var.tail(1) << max_var;
            return pt_var;
        }
        
        // 三角形2: (00, 11, 01)
        auto intersect2 = rayTriangleIntersectionB(ray, p00.head(3), p11.head(3), p01.head(3));
        if (intersect2) {
            Vec4d pt_var;
            double max_var = std::max(p00(3), p11(3));
            max_var = std::max(max_var, p01(3));
            pt_var.head(3) << (*intersect2);
            pt_var.tail(1) << max_var;
            return pt_var;
        }
        // // 获取当前网格单元的四个顶点
        // Vec3d p00 = getVertice(dir, i, j);
        // Vec3d p10 = getVertice(dir, i + 1, j);
        // Vec3d p01 = getVertice(dir, i, j + 1);
        // Vec3d p11 = getVertice(dir, i + 1, j + 1);
        
        // // 检查与两个三角形的交点
        // // 三角形1: (00, 10, 11)
        // auto intersect1 = rayTriangleIntersectionB(ray, p00, p10, p11);
        // if (intersect1) {
        //     return intersect1;
        // }
        
        // // 三角形2: (00, 11, 01)
        // auto intersect2 = rayTriangleIntersectionB(ray, p00, p11, p01);
        // if (intersect2) {
        //     return intersect2;
        // }
        
        // 移动到下一个网格单元
        if (t_z < t_x) {
            t_z += t_delta_z;
            i += di;
        } else {
            t_x += t_delta_x;
            j += dj;
        }
    }
    return std::nullopt;
};


// 射线与cell网格求交
std::optional<Vec4d> Cell::rayCellIntersectionZ(const Ray& ray) const {
    Direction dir = Direction::Z;
    if(ary_cell_vertices[dir].num_point==0) return std::nullopt;;
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'
    
    // 1. 计算射线与网格包围盒的交点
    float t_min = std::numeric_limits<float>::lowest();
    float t_max = std::numeric_limits<float>::max();
    
    // 检查与x平面的交点
    if (std::abs(ray.direction.x()) > 1e-6) {
        float tx1 = (region.x_min - ray.origin.x()) / ray.direction.x();
        float tx2 = (region.x_max - ray.origin.x()) / ray.direction.x();
        t_min = std::max(t_min, std::min(tx1, tx2));
        t_max = std::min(t_max, std::max(tx1, tx2));
    }
    
    // 检查与y平面的交点
    if (std::abs(ray.direction.y()) > 1e-6) {
        float ty1 = (region.y_min - ray.origin.y()) / ray.direction.y();
        float ty2 = (region.y_max - ray.origin.y()) / ray.direction.y();
        t_min = std::max(t_min, std::min(ty1, ty2));
        t_max = std::min(t_max, std::max(ty1, ty2));
    }
    
    // 射线与Cell包围盒不相交
    if (t_min > t_max || t_max < 0) {
        return std::nullopt;
    }
    
    // 确保t_min至少为0
    t_min = std::max(t_min, 0.0f);
    
    // 2. 计算射线进入cell的点
    Vec3d entry_point = ray.at(t_min);
    
    // 3. 计算起始网格索引
    int i0 = static_cast<int>((entry_point.x() - region.x_min) / resolution);
    int j0 = static_cast<int>((entry_point.y() - region.y_min) / resolution);
    
    // 确保索引在范围内
    i0 = std::clamp(i0, 0, n_row - 2);
    j0 = std::clamp(j0, 0, n_row - 2);
    
    // 4. 计算步进方向和增量
    int di = (ray.direction.x() > 0) ? 1 : -1;
    int dj = (ray.direction.y() > 0) ? 1 : -1;
    
    // 计算到下一个网格边的距离
    float tx_next, ty_next;
    if (di > 0) {
        tx_next = region.x_min + (i0 + 1) * resolution;
    } else {
        tx_next = region.x_min + i0 * resolution;
    }
    
    if (dj > 0) {
        ty_next = region.y_min + (j0 + 1) * resolution;
    } else {
        ty_next = region.y_min + j0 * resolution;
    }
    
    // 计算t增量
    float t_delta_x = (di != 0) ? resolution / std::abs(ray.direction.x()) : std::numeric_limits<float>::max();
    float t_delta_y = (dj != 0) ? resolution / std::abs(ray.direction.y()) : std::numeric_limits<float>::max();
    
    // 计算到下一个边界的t值
    float t_x = (tx_next - ray.origin.x()) / ray.direction.x();
    float t_y = (ty_next - ray.origin.y()) / ray.direction.y();
    
    // 5. 遍历网格单元
    int i = i0, j = j0;
    while (i >= 0 && i < n_row - 1 && j >= 0 && j < n_row - 1) {
        // // 获取当前网格单元的四个顶点
        // Vec3d p00 = getVertice(dir, i, j);
        // Vec3d p10 = getVertice(dir, i + 1, j);
        // Vec3d p01 = getVertice(dir, i, j + 1);
        // Vec3d p11 = getVertice(dir, i + 1, j + 1);
        
        // // 检查与两个三角形的交点
        // // 三角形1: (00, 10, 11)
        // auto intersect1 = rayTriangleIntersectionB(ray, p00, p10, p11);
        // if (intersect1) {
        //     return intersect1;
        // }
        
        // // 三角形2: (00, 11, 01)
        // auto intersect2 = rayTriangleIntersectionB(ray, p00, p11, p01);
        // if (intersect2) {
        //     return intersect2;
        // }
        // 获取当前网格单元的四个顶点
        Vec4d p00 = getVerticeAndVariance(dir, i, j);
        Vec4d p10 = getVerticeAndVariance(dir, i + 1, j);
        Vec4d p01 = getVerticeAndVariance(dir, i, j + 1);
        Vec4d p11 = getVerticeAndVariance(dir, i + 1, j + 1);
        
        // 检查与两个三角形的交点
        // 三角形1: (00, 10, 11)
        auto intersect1 = rayTriangleIntersectionB(ray, p00.head(3), p10.head(3), p11.head(3));
        if (intersect1) {
            Vec4d pt_var;
            double max_var = std::max(p00(3), p10(3));
            max_var = std::max(max_var, p11(3));
            pt_var.head(3) << (*intersect1);
            pt_var.tail(1) << max_var;
            return pt_var;
        }
        
        // 三角形2: (00, 11, 01)
        auto intersect2 = rayTriangleIntersectionB(ray, p00.head(3), p11.head(3), p01.head(3));
        if (intersect2) {
            Vec4d pt_var;
            double max_var = std::max(p00(3), p11(3));
            max_var = std::max(max_var, p01(3));
            pt_var.head(3) << (*intersect2);
            pt_var.tail(1) << max_var;
            return pt_var;
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

//////////////////////////////////////////////////////////
// cell.cpp 实现

void Cell::computeVertexCurvature(Direction dir, int r, int c, double& curvature, double& linearity) {
    int n_side = Parameter::GetInstance().num_test;
    
    // 1. 收集邻域点 (使用 3x3 网格邻域)
    std::vector<Vec3d> neighbors;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            int nr = r + dr;
            int nc = c + dc;
            if (nr >= 0 && nr < n_side && nc >= 0 && nc < n_side) {
                neighbors.push_back(getVertice(dir, nr, nc));
            }
        }
    }

    if (neighbors.size() < 4) {
        curvature = 0; linearity = 0; return;
    }

    // 2. 构建协方差矩阵
    Vec3d mean = Vec3d::Zero();
    for (const auto& p : neighbors) mean += p;
    mean /= static_cast<double>(neighbors.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : neighbors) {
        Vec3d diff = p - mean;
        cov += diff * diff.transpose();
    }

    // 3. 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Vec3d eigenvalues = solver.eigenvalues(); // 升序排列: L3 < L2 < L1
    
    double l1 = eigenvalues(2);
    double l2 = eigenvalues(1);
    double l3 = eigenvalues(0);

    // 4. 计算几何指标
    // 曲率 (Surface Variation): l3 / (l1 + l2 + l3)
    double sum = l1 + l2 + l3;
    curvature = (sum > 1e-6) ? l3 / sum : 0;
    
    // 线性度 (Linearity): (l1 - l2) / l1 -> 用于判断是否为边缘
    linearity = (l1 > 1e-6) ? (l1 - l2) / l1 : 0;
}

void Cell::extractLinK3DFeatures(std::vector<FeaturePoint>& edge_points, 
                                 std::vector<FeaturePoint>& planar_points) {
    if (empty) return;

    int n_side = Parameter::GetInstance().num_test;
    // 阈值设定（可根据实验调整）
    const double curvature_threshold = 0.05; 
    const double linearity_threshold = 0.7;

    for (int d = 0; d < 3; ++d) {
        Direction dir = static_cast<Direction>(d);
        if (ary_cell_vertices[dir].num_point == 0) continue;

        for (int r = 0; r < n_side; ++r) {
            for (int c = 0; c < n_side; ++c) {
                double curvature, linearity;
                computeVertexCurvature(dir, r, c, curvature, linearity);

                FeaturePoint fp;
                fp.pos = getVertice(dir, r, c);
                fp.curvature = curvature;
                fp.index = r * n_side + c;
                fp.dir = dir;

                // 适配 LinK3D: 
                // Linearity 高且曲率大的通常是边缘点 (Edge Points)
                if (linearity > linearity_threshold && curvature > curvature_threshold) {
                    edge_points.push_back(fp);
                } 
                // 曲率极小的是平面点 (Planar Points)
                else if (curvature < 0.01) {
                    planar_points.push_back(fp);
                }
            }
        }
    }
}
