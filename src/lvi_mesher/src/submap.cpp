#include <execution>
#include "submap.h"
#include <execution>
// #include <igl/boundary_loop.h>
// #include <igl/lscm.h>
#include <pcl/common/transforms.h>
// #include "xatlas.h"
// #include <igl/opengl/glfw/Viewer.h>
// #include <igl/map_vertices_to_circle.h>
#include <cmath>

// 判断点 (px, py) 是否在三角形 (x0,y0), (x1,y1), (x2,y2) 内部
bool pointInTriangle(double px, double py,
                     double x0, double y0,
                     double x1, double y1,
                     double x2, double y2) {
    auto sign = [](double x1, double y1, double x2, double y2, double x3, double y3) -> double {
        return (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3);
    };

    double d1 = sign(px, py, x0, y0, x1, y1);
    double d2 = sign(px, py, x1, y1, x2, y2);
    double d3 = sign(px, py, x2, y2, x0, y0);

    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

// 三角形重心坐标插值（可选，如果你想更准确地插值 3D 点）
// 但为了简化，这里直接使用三角形的重心 3D 点
Eigen::Vector3d computeBarycentricInterpolatedPoint(
    double u, double v,
    double u0, double v0,
    double u1, double v1,
    double u2, double v2,
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2) {

    // 重心坐标
    double denom = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
    if (std::abs(denom) < 1e-8) {
        return (p0 + p1 + p2) / 3.0;  // fallback to centroid
    }

    double lambda1 = ((u1 - u) * (v2 - v) - (u2 - u) * (v1 - v)) / denom;
    double lambda2 = ((u2 - u) * (v0 - v) - (u0 - u) * (v2 - v)) / denom;
    double lambda3 = 1.0 - lambda1 - lambda2;

    return lambda1 * p0 + lambda2 * p1 + lambda3 * p2;
}

/* void SaveXatlasMeshToObj(const std::string& filename,
                     const xatlas::Mesh& mesh,
                     const std::vector<Eigen::Vector3d>& mesh_vertices,
                    const int texture_width,
                    const int texture_height) {
    const auto& xatlas_verts = mesh.vertexArray;
    const auto& xatlas_faces = mesh.indexArray;

    std::ofstream objFile(filename);
    if (!objFile.is_open()) {
        std::cerr << "无法打开 OBJ 文件进行写入: " << filename << std::endl;
        return;
    }

    // OBJ 格式是不要求顶点索引全局唯一的，但通常我们按顺序写入

    // 1. 写入顶点位置 (v x y z)
    for (size_t i = 0; i < mesh.vertexCount; ++i) {
        const xatlas::Vertex& vtx = xatlas_verts[i];
        int vert_idx = vtx.xref;  // 原始顶点索引
        const Eigen::Vector3d& pos = mesh_vertices[vert_idx];
        objFile << "v " << pos.x() << " " << pos.y() << " " << pos.z() << "\n";
    }

    // 2. 写入纹理坐标 (vt u v)     ⚠️ u 和 v 必须是 [0,1] 范围
    for (size_t i = 0; i < mesh.vertexCount; ++i) {
        const xatlas::Vertex& vtx = xatlas_verts[i];
        double u = double(vtx.uv[0]) / double(texture_width);     // ✅ 这是 [0,1] 的！
        double v = double(vtx.uv[1]) / double(texture_height);    // ✅ 这是 [0,1] 的！
        objFile << "vt " << u << " " << v << "\n";
    }

    // 3. 写入面片 (f v1/vt1 v2/vt2 v3/vt3)
    for (size_t f = 0; f < mesh.indexCount / 3; ++f) {
        uint32_t i0 = xatlas_faces[f * 3 + 0];
        uint32_t i1 = xatlas_faces[f * 3 + 1];
        uint32_t i2 = xatlas_faces[f * 3 + 2];

        // OBJ 面片索引是从 1 开始的！
        objFile << "f "
                << (i0 + 1) << "/" << (i0 + 1) << " "
                << (i1 + 1) << "/" << (i1 + 1) << " "
                << (i2 + 1) << "/" << (i2 + 1) << "\n";
    }

    objFile.close();
    std::cout << "OBJ 文件已写入: " << filename << std::endl;
} */

/* void visualizeTexturedMeshWithLibIGL(
    const std::vector<float>& xatlas_vertices,     // 来自 xatlas，float[n_vertices * 3]
    const std::vector<uint32_t>& xatlas_indices,   // 来自 xatlas，uint32_t[n_triangles * 3]
    const std::vector<Vec2d>& UV,                  // 每个顶点的 UV，范围 [0,1]，大小 n_vertices
    const cv::Mat& baked_texture_,                 // 烘焙的纹理图，CV_8UC3，BGR
    const std::vector<Eigen::Vector3i>& mesh_faces // 原始面片，假设未改变
) {
    using namespace Eigen;

    // -------------------------------
    // 1. 构造 libigl 格式的顶点和面片
    // -------------------------------

    // xatlas_vertices: float[n_vertices * 3]
    int n_vertices = xatlas_vertices.size() / 3;
    MatrixXd V_xatlas(n_vertices, 3);
    for (int i = 0; i < n_vertices; ++i) {
        V_xatlas(i, 0) = xatlas_vertices[i * 3 + 0];
        V_xatlas(i, 1) = xatlas_vertices[i * 3 + 1];
        V_xatlas(i, 2) = xatlas_vertices[i * 3 + 2];
    }

    // 面片：假设是三角形，且未改变，直接使用 mesh_faces
    MatrixXi F;
    {
        int n_faces = mesh_faces.size();
        F.resize(n_faces, 3);
        for (int i = 0; i < n_faces; ++i) {
            F(i, 0) = mesh_faces[i](0);
            F(i, 1) = mesh_faces[i](1);
            F(i, 2) = mesh_faces[i](2);
        }
    }

    // -------------------------------
    // 2. 构造 UV 矩阵：n_vertices x 2
    // -------------------------------
    MatrixXd UV_xatlas(n_vertices, 2);
    for (int i = 0; i < n_vertices; ++i) {
        UV_xatlas(i, 0) = UV[i](0); // u
        UV_xatlas(i, 1) = UV[i](1); // v
    }

    // -------------------------------
    // 3. 转换 baked_texture_ (CV_8UC3) 为 libigl 格式：R, G, B
    // -------------------------------
    int tex_h = baked_texture_.rows;
    int tex_w = baked_texture_.cols;

    Matrix<unsigned char, Dynamic, Dynamic> R(tex_h, tex_w);
    Matrix<unsigned char, Dynamic, Dynamic> G(tex_h, tex_w);
    Matrix<unsigned char, Dynamic, Dynamic> B(tex_h, tex_w);

    for (int y = 0; y < tex_h; ++y) {
        for (int x = 0; x < tex_w; ++x) {
            cv::Vec3b bgr = baked_texture_.at<cv::Vec3b>(y, x);
            B(y, x) = bgr[0];
            G(y, x) = bgr[1];
            R(y, x) = bgr[2];
        }
    }

    // -------------------------------
    // 4. 使用 libigl 渲染
    // -------------------------------
    igl::opengl::glfw::Viewer viewer;

    // 设置 Mesh
    viewer.data().set_mesh(V_xatlas, F);

    // 设置 UV
    viewer.data().set_uv(UV_xatlas);

    // 设置纹理
    viewer.data().set_texture(R, G, B);

    std::cout << "✅ 带纹理的 Mesh 已加载，使用 libigl Viewer 查看！" << std::endl;
    std::cout << "🖱️  鼠标拖拽旋转视角，滚轮缩放。" << std::endl;

    // 启动 Viewer
    viewer.launch();
} */

Eigen::MatrixXd toEigenMatrix(const std::vector<Eigen::Vector3d>& vec) {
    Eigen::MatrixXd V(vec.size(), 3);
    for (int i = 0; i < vec.size(); ++i) {
        V.row(i) = vec[i];
    }
    return V;
}

// 将 std::vector<Eigen::Vector3i> 转 Eigen::MatrixXi
Eigen::MatrixXi toEigenMatrix(const std::vector<Eigen::Vector3i>& vec) {
    Eigen::MatrixXi F(vec.size(), 3);
    for (int i = 0; i < vec.size(); ++i) {
        F.row(i) = vec[i];
    }
    return F;
}


Submap::KeyType Submap::IndexToKey(const Eigen::Array3i& index) {
  KeyType k_0(static_cast<uint32_t>(index[0]));
  KeyType k_1(static_cast<uint32_t>(index[1]));
  KeyType k_2(static_cast<uint32_t>(index[2]));
  return (k_0 << 2 * 32) | (k_1 << 1 * 32) | k_2;
}

// 从世界坐标到图像像素坐标的投影
// 输入：
//   - point_world: 3D 点在世界坐标系 (Eigen::Vector3d)
//   - K: 相机内参 3x3
//   - R: 相机旋转矩阵 3x3 (世界到相机)
//   - t: 相机平移向量 3x1 (世界原点在相机坐标系的位置)
// 输出：
//   - pixel: 图像中的 2D 像素坐标 (x,y)，如果不可见返回 false
bool Submap::project_point_to_pixel(
    const Eigen::Vector3d& point_world,
    const Eigen::Matrix4d& T_cw,
    Eigen::Vector2i& pixel,
    Eigen::Vector3d& point_camera,
    bool& is_visible
) {
    
    // Transform to camera coordinates: X_c = R * (X_w - t)
    point_camera = T_cw.block<3, 3>(0, 0) * point_world + T_cw.block<3, 1>(0, 3);

    is_visible = (point_camera.z() > 0);  // 必须在相机前方
    if (!is_visible) return false;

    // 投影到图像平面: x = K * [Xc_x, Xc_y, Xc_z]
    Vec2d pxpy;
    camera_left_->spaceToPlane(point_camera, pxpy);
    pixel = Eigen::Vector2i(static_cast<int>(pxpy.x() + 0.5), static_cast<int>(pxpy.y() + 0.5));
    return true;
}

// 假设：
//   - V: Mesh 顶点 (Eigen::MatrixXd, size n x 3)
//   - UV: 每个顶点的 UV 坐标 (Eigen::MatrixXd, size n x 2), 范围 [0,1] 或实际纹理分辨率
//   - frames: 每帧图像的数据
//      struct FrameData {
//          cv::Mat image;           // 图像 (H,W,3), BGR
//          Eigen::Matrix3d K;       // 相机内参
//          Eigen::Matrix3d R;       // 世界到相机旋转
//          Eigen::Vector3d t;       // 世界到相机平移
//      };
//   - texture_map: 输出的纹理图 (比如 1024x1024)

void Submap::generate_texture_from_multiview(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& UV,
    cv::Mat& texture_map
) {
    int tex_width = texture_map.cols;
    int tex_height = texture_map.rows;

    // 遍历每个顶点
    for (int vid = 0; vid < V.rows(); ++vid) {
        Eigen::Vector3d point_world = V.row(vid);
        Eigen::Vector2d uv = UV.row(vid);

        // 假设 UV 是 [0,1]，映射到纹理图尺寸
        int u = static_cast<int>(uv.x() * (tex_width - 1));
        int v = static_cast<int>(uv.y() * (tex_height - 1));

        // 限制范围
        u = std::max(0, std::min(u, tex_width - 1));
        v = std::max(0, std::min(v, tex_height - 1));

        // 默认颜色（比如黑色/无效）
        cv::Vec3b best_color(0, 0, 0);
        double best_depth = 1e10;
        bool has_valid = false;

        // 遍历所有帧
        Mat4d T_I_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
        for (int i = 0; i < images_for_texture_.size(); ++i) {
            const auto& img = images_for_texture_[i];
            Eigen::Vector2i pixel;
            Eigen::Vector3d point_cam;
            bool visible = false;

            const Mat4d& T_wi = node_poses_[i];
            Mat4d T_cw = (T_wi * T_I_lC).inverse();
            if (project_point_to_pixel(point_world, T_cw, pixel, point_cam, visible)) {
                int px = pixel.x();
                int py = pixel.y();

                // 检查是否在图像范围内
                if (px >= 0 && px < img.cols && py >= 0 && py < img.rows) {
                    cv::Vec3b color = img.at<cv::Vec3b>(py, px);

                    // 可选：选择深度最小的（最前面的）
                    if (visible && point_cam.z() < best_depth) {
                        best_depth = point_cam.z();
                        best_color = color;
                        has_valid = true;
                    }
                }
            }
        }

        if (has_valid) {
            texture_map.at<cv::Vec3b>(v, u) = best_color;  // OpenCV 是 (row, col) = (y, x)
        }
    }
}


bool Submap::preprocessScan(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_raw_ptr, 
                   PointMatrix& points_result, 
                   pcl::PointCloud<pcl::PointXYZRGB> & pcl_got,
                   double voxel_filter_size){
    TicToc t_get_pcl;
    double range_max = Parameter::GetInstance().range_max, range_min = Parameter::GetInstance().range_min;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_filtered_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);//downsampled
    PointMatrix points_raw, points_filted;
    

    //voxel_filter_size filter
    TicToc t_down_sample;
    if(voxel_filter_size > 0){
        *pcl_filtered_ptr = pclVoxelFilter(pcl_raw_ptr, voxel_filter_size);
    }
    else{
        pcl_filtered_ptr = pcl_raw_ptr;
    }
    
    points_raw = PointMatrix(*pcl_filtered_ptr);
    // points_raw = pcl_filtered_ptr->getMatrixXfMap().cast<double>();
    // LOG(INFO)<<ptsrgb.cols()<<", "<<ptsrgb.rows(); //n,8

    //range filter
    int filter_point_step = 1;//simple down-sample
    double range_max_square = range_max * range_max,
           range_min_sqaure = range_min * range_min;
    
    // range小于100才过滤
    if(range_max <= 100){
        points_result.clear_quick();
        for(int i = 0; i < points_raw.num_point; i += filter_point_step){
            double range_sqaure = points_raw.point(0, i) * points_raw.point(0, i) +
                                  points_raw.point(1, i) * points_raw.point(1, i) +
                                  points_raw.point(2, i) * points_raw.point(2, i);
            if(range_sqaure < range_max_square && range_sqaure > range_min_sqaure ){
                points_result.addPoint(points_raw.point.col(i), points_raw.rgb.col(i));
            }
        }
    }else{
        points_result.clear_quick();
        points_result = points_raw;
    }
    if(points_result.num_point <= 0){
        return false;
    }
    pcl_got = *pcl_filtered_ptr;
    return true;
}



void Submap::initializeMap(size_t frame_id, const Mat4d& pose, const cv::Mat& image, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_raw_ptr){
    // name = string("submap_") + std::to_string(submap_id);
    step = frame_id;

    frame_ids_.push_back(frame_id);
    node_poses_.push_back(pose);
    
    // Submap在odom下的朝向与世界系一致，原点为初始帧的位置。经优化后，除第一张submap外，位姿可能变化
    pose_odom_.setIdentity();
    pose_odom_.block<3,1>(0,3) = pose.block<3,1>(0,3);
    pose_slam_ = pose;

    // 降采样
    preprocessScan(pcl_raw_ptr, points_turned, pcl_raw, Parameter::GetInstance().voxel_size);

    // for visualization
    points_turned_odom = points_turned;
    points_turned_odom.transformPoints(pose);

    Mat4d pose_in_submap = pose_odom_.inverse() * pose;
    points_turned.transformPoints(pose_in_submap);

    dividePointsIntoCellInitMap(points_turned);
    initialized_ = true;
    // cells_glb.reserve(5000);

    // if(Parameter::GetInstance().enable_mesh_texture){
    //     pointclouds_for_texture_.push_back(pcl_raw_ptr);
    //     images_for_texture_.push_back(image);
    // }

    // Mat4d T_I_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
    // Mat4d T_c2s = pose_in_submap * T_I_lC;
    // UpdateVertexColors(T_c2s.inverse(), image);
}


void  Submap::dividePointsIntoCellInitMap(PointMatrix & points_raw){
    TicToc t_gp;
    double range_max = (Parameter::GetInstance().range_max>100) ? 100 : Parameter::GetInstance().range_max;
    double grid = Parameter::GetInstance().grid, min_num_to_gp = Parameter::GetInstance().min_points_num_to_gp;
    CHECK(min_num_to_gp >= 3);

    cells_glb.clear();
  
    // 获取点云外包围框
    double max_x = points_raw.point.block(0, 0, 1, points_raw.num_point).maxCoeff();
    double max_y = points_raw.point.block(1, 0, 1, points_raw.num_point).maxCoeff();
    double max_z = points_raw.point.block(2, 0, 1, points_raw.num_point).maxCoeff();
    double min_x = points_raw.point.block(0, 0, 1, points_raw.num_point).minCoeff();
    double min_y = points_raw.point.block(1, 0, 1, points_raw.num_point).minCoeff();
    double min_z = points_raw.point.block(2, 0, 1, points_raw.num_point).minCoeff();
    double  bot_grid_x = grid*floor(min_x/grid),
            bot_grid_y = grid*floor(min_y/grid),
            bot_grid_z = grid*floor(min_z/grid);
    
    //各个方向的cell数
    int num_grid_x = int(ceil(max_x/grid)-floor(min_x/grid)),
        num_grid_y = int(ceil(max_y/grid)-floor(min_y/grid)),
        num_grid_z = int(ceil(max_z/grid)-floor(min_z/grid));
    int num_grid = num_grid_x * num_grid_y * num_grid_z;
    
    PointMatrix init_points_bucket;
    Cell init_cell;
    int num_bucket = num_grid_x * num_grid_y * num_grid_z;
    
    cell_key_set_.clear();
    ary_points_bucket_.clear();
    ary_bucket_cell_.clear();

    //push points into bucket
    TicToc t2;
    for(int i = 0; i < points_raw.num_point; i ++){
        int ix = floor((points_raw.point(0, i) - bot_grid_x) / grid);
        int iy = floor((points_raw.point(1, i) - bot_grid_y) / grid);
        int iz = floor((points_raw.point(2, i) - bot_grid_z) / grid);
        Eigen::Array3i index={ix,iy,iz};
        KeyType k = IndexToKey(index);

        auto it = cell_key_set_.find(k);
        if (it == cell_key_set_.end()) {
            ary_points_bucket_.push_back(PointMatrix());
            ary_bucket_cell_.push_back(init_cell);
            ary_cell_xyz_.push_back(index);
            ary_points_bucket_.back().addPoint(points_raw.point.col(i), points_raw.rgb.col(i));
            cell_key_set_.insert({k, ary_points_bucket_.size()-1});
        }else{
            ary_points_bucket_[it->second].addPoint(points_raw.point.col(i), points_raw.rgb.col(i));
        }        
    }
    // LOG(INFO)<<"allacate cost "<<t2.toc()<<", "<<ary_points_bucket_.size();
    

    //reconstruction
    TicToc t3;
    std::vector<size_t> indices(ary_points_bucket_.size());
    iota(indices.begin(), indices.end(),  0);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](auto &idx){
        PointMatrix& points_raw_cell = ary_points_bucket_[idx];
        if(points_raw_cell.num_point > min_num_to_gp){
            Eigen::Array3i& xyz = ary_cell_xyz_[idx];
            double iz = xyz.z() * grid + bot_grid_z, iy = xyz.y() * grid + bot_grid_y, ix = xyz.x() * grid + bot_grid_x;
            double posi_tmp;
            posi_tmp = getPosiWithTime(ix, iy, iz, grid, 0);
            Region region_tmp(ix, iy, iz, ix + grid, iy + grid,iz + grid);

            Cell tmp_cell(points_raw_cell, this->step, posi_tmp, region_tmp, true, true);
            ary_bucket_cell_[idx] = tmp_cell;
        }
    });

    //because emplace are not suitable to be multi-threaded, so use ary_bucket_cell to store cells first
    for(int i_bucket = 0; i_bucket < ary_bucket_cell_.size(); i_bucket++){
        if(ary_points_bucket_[i_bucket].num_point > min_num_to_gp){
            cells_glb.emplace(ary_bucket_cell_[i_bucket].hash_position, ary_bucket_cell_[i_bucket]);
        }
    }
}


void  Submap::dividePointsIntoCell(PointMatrix& points_raw){
    // clear states of old frame
    cells_keyset_now_.clear();
    ary_points_now_.clear();
    ary_cells_now_.clear();
    ary_cells_xyz_now_.clear();

    TicToc t_gp;
    double range_max = (Parameter::GetInstance().range_max>100) ? 100 : Parameter::GetInstance().range_max;
    double grid = Parameter::GetInstance().grid, min_num_to_gp = Parameter::GetInstance().min_points_num_to_gp;
    CHECK(min_num_to_gp >= 3);

    //point division
    //find a bounding box of current raw points in world frame
    double max_x = points_raw.point.block(0, 0, 1, points_raw.num_point).maxCoeff();
    double max_y = points_raw.point.block(1, 0, 1, points_raw.num_point).maxCoeff();
    double max_z = points_raw.point.block(2, 0, 1, points_raw.num_point).maxCoeff();
    double min_x = points_raw.point.block(0, 0, 1, points_raw.num_point).minCoeff();
    double min_y = points_raw.point.block(1, 0, 1, points_raw.num_point).minCoeff();
    double min_z = points_raw.point.block(2, 0, 1, points_raw.num_point).minCoeff();
    double  bot_grid_x = grid * floor(min_x / grid),
            bot_grid_y = grid * floor(min_y / grid),
            bot_grid_z = grid * floor(min_z / grid);
    int num_grid_x = int(ceil(max_x / grid) - floor(min_x / grid)),
        num_grid_y = int(ceil(max_y / grid) - floor(min_y / grid)),
        num_grid_z = int(ceil(max_z / grid) - floor(min_z / grid));
    int num_grid = num_grid_x * num_grid_y * num_grid_z;    
    int num_bucket = num_grid_x * num_grid_y * num_grid_z;
    //push points into bucket
    TicToc t_push_points;
    PointMatrix init_points_bucket;
    Cell init_cell;
    for(int i = 0; i < points_raw.num_point; i ++){
        int ix = floor((points_raw.point(0, i) - bot_grid_x) / grid);
        int iy = floor((points_raw.point(1, i) - bot_grid_y) / grid);
        int iz = floor((points_raw.point(2, i) - bot_grid_z) / grid);
        Eigen::Array3i index={ix,iy,iz};
        KeyType k = IndexToKey(index);

        auto it = cells_keyset_now_.find(k);
        if (it == cells_keyset_now_.end()) {
            ary_cells_now_.push_back(init_cell);
            ary_cells_xyz_now_.push_back(index);

            ary_points_now_.push_back(init_points_bucket);
            ary_points_now_.back().addPoint(points_raw.point.col(i), points_raw.rgb.col(i));
            cells_keyset_now_.insert({k, ary_points_now_.size()-1});
        }else{
            ary_points_now_[it->second].addPoint(points_raw.point.col(i), points_raw.rgb.col(i));
        }        
    }
    std::vector<size_t> indices(ary_points_now_.size());
    iota(indices.begin(), indices.end(),  0);
    CHECK(ary_points_now_.size() == ary_cells_now_.size());
    CHECK(ary_points_now_.size() == ary_cells_xyz_now_.size());
    /* std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](auto &idx){
        PointMatrix& points_raw_cell = ary_points_now_[idx];
        if(points_raw_cell.num_point > min_num_to_gp){
            Eigen::Array3i& xyz = ary_cells_xyz_now_[idx];
            double iz = xyz.z() * grid + bot_grid_z, iy = xyz.y() * grid + bot_grid_y, ix = xyz.x() * grid + bot_grid_x;
            double posi_tmp;
            posi_tmp = getPosiWithTime(ix, iy, iz, grid, 0);
            Region region_tmp(ix, iy, iz, ix + grid, iy + grid,iz + grid);

            bool map_glb_not_surface;        
            auto cell_glb = cells_glb.find(posi_tmp);
            if(cell_glb != cells_glb.end()){
                map_glb_not_surface = cell_glb->second.not_surface;
            }

            Cell tmp_cell(points_raw_cell, this->step, posi_tmp, region_tmp, true, map_glb_not_surface);
            tmp_cell.time_stamp = this->step;
            ary_cells_now_[idx] = tmp_cell;
        }
    }); */
    for(int idx : indices){
        PointMatrix& points_raw_cell = ary_points_now_[idx];
        if(points_raw_cell.num_point > min_num_to_gp){
            Eigen::Array3i& xyz = ary_cells_xyz_now_[idx];
            double iz = xyz.z() * grid + bot_grid_z, iy = xyz.y() * grid + bot_grid_y, ix = xyz.x() * grid + bot_grid_x;
            double posi_tmp;
            posi_tmp = getPosiWithTime(ix, iy, iz, grid, 0);
            Region region_tmp(ix, iy, iz, ix + grid, iy + grid,iz + grid);
            bool map_glb_not_surface;        
            auto cell_glb = cells_glb.find(posi_tmp);
            if(cell_glb != cells_glb.end()){
                map_glb_not_surface = cell_glb->second.not_surface;
            }
            Cell tmp_cell(points_raw_cell, this->step, posi_tmp, region_tmp, true, map_glb_not_surface);
            tmp_cell.time_stamp = this->step;
            ary_cells_now_[idx] = tmp_cell;
        }
    }
}

// 3D DDA算法遍历射线穿过的体素
std::vector<const Cell*> Submap::getIntersectedCells(const Ray& ray, int max_steps) const {
    double grid = Parameter::GetInstance().grid;
    std::vector<const Cell*> result;
    
    Vec3d current_key = getVoxelKey(ray.origin);
    double hash = getPosiWithTime(current_key(0), current_key(1), current_key(2), grid, 0);
    auto it = cells_glb.find(hash);
    if (it != cells_glb.end()) {
        // Cell& cell = it->second;
        result.push_back(&it->second);
    }
    
    // 2. 初始化步进方向和边界信息
    Vec3d ray_dir = ray.direction;
    Vec3i step;
    Vec3d t_max, t_delta;
    
    for (int i = 0; i < 3; ++i) {
        if (ray_dir[i] != 0.0) {
            step[i] = ray_dir[i] > 0 ? 1 : -1;
            double voxel_boundary = current_key[i] + (step[i] > 0 ? grid : 0);
            t_max[i] = (voxel_boundary - ray.origin[i]) / ray_dir[i];
            t_delta[i] = grid / abs(ray_dir[i]);
        } else {
            step[i] = 0;
            t_max[i] = std::numeric_limits<double>::max();
            t_delta[i] = std::numeric_limits<double>::max();
        }
    }
    
    // 3. DDA主循环
    int steps = 0;
    while (steps++ < max_steps) {
        // 找到最小t_max的轴
        int axis = 0;
        if (t_max.y() < t_max.x()) axis = 1;
        if (t_max.z() < t_max[axis]) axis = 2;
        
        // 沿该轴步进
        current_key[axis] += step[axis] * grid;
        
        // 检查新体素是否存在
        double current_hash = getPosiWithTime(current_key(0), current_key(1), current_key(2), grid, 0);
        it = cells_glb.find(current_hash);
        if (it != cells_glb.end()) {
            result.push_back(&it->second);

        } else {
            // 如果遇到不存在的体素，可以选择继续或停止
            // 这里选择继续，直到max_steps
        }
        
        // 更新t_max
        t_max[axis] += t_delta[axis];
    }
    
    return result;
}

void Submap::RetrieveColors(const pcl::PointCloud<pcl::PointXYZ>& points,
                            const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                            pcl::PointCloud<pcl::PointXYZRGB>& points_rgb){
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    
    // transform points to submap frame
    Mat4d pose_in_submap = pose_odom_.inverse() * pose_guess;
    pcl::PointCloud<pcl::PointXYZ> points_in_submap;
    pcl::transformPointCloud(points, points_in_submap, pose_in_submap);
   
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'
    points_rgb.points.resize(points_in_submap.points.size()); //for parallel
    std::vector<size_t> idx(points_in_submap.points.size());
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        points_rgb.points[i].x = points.points[i].x;
        points_rgb.points[i].y = points.points[i].y;
        points_rgb.points[i].z = points.points[i].z;
        points_rgb.points[i].r = 0;
        points_rgb.points[i].g = 0;
        points_rgb.points[i].b = 0;
        auto& pt = points_in_submap.points[i];
        Vec3d vpt={pt.x, pt.y, pt.z};
        double  gx = floor(pt.x / grid) * grid;
        double  gy = floor(pt.y / grid) * grid;
        double  gz = floor(pt.z / grid) * grid;
        double hash = getPosiWithTime(gx, gy, gz, grid, 0);
        auto cell_glb = this->cells_glb.find(hash);
        if(cell_glb != this->cells_glb.end()){
            const Cell* ptr_cell_glb = &(cell_glb->second);
            if((!ptr_cell_glb->not_surface) && (!ptr_cell_glb->empty)){
                for(int dir=0; dir<3; dir++){
                    if(points_rgb.points[i].r>0 || points_rgb.points[i].g>0 || points_rgb.points[i].b>0) break;
                    const PointMatrix& points_glb = ptr_cell_glb->ary_cell_vertices[dir];
                    if(points_glb.num_point==0) continue;
                    double ix = (pt.x - gx) / resolution, iy = (pt.y - gy) / resolution, iz = (pt.z - gz) / resolution;
                    int ix_min = floor(ix), iy_min = floor(iy), iz_min = floor(iz);
                    int idx1, idx2, idx3, idx4;
                    if(dir==Direction::X){
                        idx1 = iy_min + iz_min * n_row;
                        idx2 = (iy_min+1) + iz_min * n_row;
                        idx3 = iy_min + (iz_min+1) * n_row;
                        idx4 = (iy_min+1) + (iz_min+1) * n_row;
                    }else if(dir==Direction::Y){
                        idx1 = iz_min + ix_min * n_row;
                        idx2 = (iz_min+1) + ix_min * n_row;
                        idx3 = iz_min + (ix_min+1) * n_row;
                        idx4 = (iz_min+1) + (ix_min+1) * n_row;
                    }else if(dir==Direction::Z){
                        idx1 = ix_min + iy_min * n_row;
                        idx2 = (ix_min+1) + iy_min * n_row;
                        idx3 = ix_min + (iy_min+1) * n_row;
                        idx4 = (ix_min+1) + (iy_min+1) * n_row;
                    }
                    
                    // average or max?
                    // double avg_var = (points_glb.variance(idx1) + points_glb.variance(idx2)
                    //                     + points_glb.variance(idx3) + points_glb.variance(idx4)) / 4;
                    
                    Vec3d c1 = points_glb.rgb.col(idx1);
                    Vec3d c2 = points_glb.rgb.col(idx2);
                    Vec3d c3 = points_glb.rgb.col(idx3);
                    Vec3d c4 = points_glb.rgb.col(idx4);
                    Vec3d rgb = (c1+c2+c3+c4) / 4;

                    points_rgb.points[i].r = rgb(0);
                    points_rgb.points[i].g = rgb(1);
                    points_rgb.points[i].b = rgb(2);
                }
            }
        }
    });
}

void Submap::GetAllPoints(pcl::PointCloud<pcl::PointXYZRGB>& xyzrgb){
    xyzrgb.clear();
    xyzrgb.points.reserve(5000);
    for(auto& cell_glb : cells_glb){
        PointMatrix& points = cell_glb.second.cell_raw_points;
        for(int i = 0; i < points.num_point; ++i){
            Vec3d xyz = points.point.col(i);
            Vec3d rgb = points.rgb.col(i);
            Vec3d xyz_odom = pose_odom_.block<3,3>(0,0) * xyz + pose_odom_.block<3,1>(0,3);
            pcl::PointXYZRGB pt;
            pt.x = xyz_odom(0);
            pt.y = xyz_odom(1);
            pt.z = xyz_odom(2);
            pt.r = rgb(0);
            pt.g = rgb(1);
            pt.b = rgb(2);
            xyzrgb.points.push_back(pt);
        }
    }
}

std::vector<bool> Submap::FindCorrespondences(const pcl::PointCloud<pcl::PointXYZRGB>& source,
                             const Mat4d& pose_guess,
                             std::vector<Vec3d>& target_color_gradient,
                             pcl::PointCloud<pcl::PointXYZRGBNormal>& target_xyzrgbn){
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    
    // transform points to submap frame
    Mat4d pose_in_submap = pose_odom_.inverse() * pose_guess;
    pcl::PointCloud<pcl::PointXYZRGB> source_in_submap;
    pcl::transformPointCloud(source, source_in_submap, pose_in_submap);
   
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'
    size_t n = source_in_submap.size();
    target_color_gradient.resize(n);
    target_xyzrgbn.resize(n);
    std::vector<bool> flag(n);
    std::vector<size_t> idx(n);
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        pcl::PointXYZRGB& src = source_in_submap.points[i];
        double  gx = floor(src.x / grid) * grid;
        double  gy = floor(src.y / grid) * grid;
        double  gz = floor(src.z / grid) * grid;
        double hash = getPosiWithTime(gx, gy, gz, grid, 0);
        auto cell_glb = this->cells_glb.find(hash);
        if(cell_glb != this->cells_glb.end()){
            const Cell* ptr_cell_glb = &(cell_glb->second);
            if((!ptr_cell_glb->not_surface) && (!ptr_cell_glb->empty)){
                for(int dir=0; dir<3; dir++){
                    if(flag[i]) continue;

                    const PointMatrix& points_glb = ptr_cell_glb->ary_cell_vertices[dir];
                    if(points_glb.num_point==0) continue;
                    double ix = (src.x - gx) / resolution, iy = (src.y - gy) / resolution, iz = (src.z - gz) / resolution;
                    int ix_min = floor(ix), iy_min = floor(iy), iz_min = floor(iz);
                    int idx1, idx2, idx3, idx4;
                    // double w1, w2, w3, w4; TODO
                    if(dir==Direction::X){
                        idx1 = iy_min + iz_min * n_row;
                        idx2 = (iy_min+1) + iz_min * n_row;
                        idx3 = iy_min + (iz_min+1) * n_row;
                        idx4 = (iy_min+1) + (iz_min+1) * n_row;
                    }else if(dir==Direction::Y){
                        idx1 = iz_min + ix_min * n_row;
                        idx2 = (iz_min+1) + ix_min * n_row;
                        idx3 = iz_min + (ix_min+1) * n_row;
                        idx4 = (iz_min+1) + (ix_min+1) * n_row;
                    }else if(dir==Direction::Z){
                        idx1 = ix_min + iy_min * n_row;
                        idx2 = (ix_min+1) + iy_min * n_row;
                        idx3 = ix_min + (iy_min+1) * n_row;
                        idx4 = (ix_min+1) + (iy_min+1) * n_row;
                    }
                    
                    // average or max?
                    double avg_var = (points_glb.variance(idx1) + points_glb.variance(idx2)
                                        + points_glb.variance(idx3) + points_glb.variance(idx4)) / 4;
                    double max_var = -1.0;
                    std::vector<int> ids={idx1, idx2, idx3, idx4};
                    for(int k : ids){
                        max_var = points_glb.variance(k) > max_var ? points_glb.variance(k) : max_var;
                    }

                    Vec3d p1 = points_glb.point.col(idx1);
                    Vec3d p2 = points_glb.point.col(idx2);
                    Vec3d p3 = points_glb.point.col(idx3);
                    Vec3d p4 = points_glb.point.col(idx4);
                    
                    Vec3d c1 = points_glb.rgb.col(idx1);
                    Vec3d c2 = points_glb.rgb.col(idx2);
                    Vec3d c3 = points_glb.rgb.col(idx3);
                    Vec3d c4 = points_glb.rgb.col(idx4);
                    
                    std::vector<Vec3d> near_points_submap = {p1,p2,p3,p4};
                    std::vector<Vec3d> near_colors = {c1, c2, c3, c4};
                    std::vector<Vec3d> near_points_odom={};
                    
                    Vec3d p0 = Vec3d::Zero(); 
                    Vec3d c0 = Vec3d::Zero(); 
                    for(const Vec3d& p: near_points_submap){
                        Vec4d homo_pt={p(0),p(1),p(2),1.};
                        near_points_odom.push_back((pose_odom_ * homo_pt).head(3));
                        p0 += near_points_odom.back();
                    }
                    

                    Vec4d plane_para;
                    if(estiPlane(near_points_odom, 0.1f, plane_para) && max_var>0.){
                        flag[i] = true;
                        Vec3d np = plane_para.head(3); //TODO: more exactly
                        Vec4d src_homo = {src.x, src.y, src.y, 1.};
                        Vec3d src_odom = (pose_odom_ * src_homo).head(3);
                        
                        p0 = projectPointToPlane(src_odom, plane_para);
                        c0 = (c1+c2+c3+c4) / 4.;

                        Eigen::Matrix3Xd A(3, 5);
                        Eigen::VectorXd b(5);
                        double ip = c0.dot(rgb_to_intensity_weight_) / 255.0;
                        for (int k = 0; k < 4; ++k) {
                            Vec3d pp = near_points_odom[k];
                            Vec3d cpp = near_colors[k];
                            double ipp = cpp.dot(rgb_to_intensity_weight_) / 255.0;

                            // pp在tangent plane上的对应点
                            Vec3d pp_proj = pp - (pp-p0).dot(np)*np;
                            // tangent plane上p指向pp的向量
                            A.block<3, 1>(0, k) = pp_proj - p0;
                            // 近邻点与当前点的颜色差值
                            b(k) = ipp - ip;
                        }
                        // add orthogonal constraint dp.dot(np) = 0 with weight (nn_size-1)
                        A.block<3, 1>(0, 4) = 4 * np;
                        b(4) = 0;

                        // // solve for dp
                        Vec3d X = (A * A.transpose()).ldlt().solve(A * b);// X is the estimated color gradient dp
                        target_color_gradient[i] = X;
                        
                        pcl::PointXYZRGBNormal tgt;
                        tgt.x = p0(0);
                        tgt.y = p0(1);
                        tgt.z = p0(2);
                        tgt.r = c0(0);
                        tgt.g = c0(1);
                        tgt.b = c0(2);
                        tgt.normal_x = np(0);
                        tgt.normal_y = np(1);
                        tgt.normal_z = np(2);
                        target_xyzrgbn[i] = tgt;
                    }
                }
            }
        }
    });
    return flag;
}

// 有效点太少
std::vector<bool> Submap::FindCorrespondencesV1(const pcl::PointCloud<pcl::PointXYZRGB>& source,
                             const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                             std::vector<Vec3d>& target_color_gradient,
                             pcl::PointCloud<pcl::PointXYZRGBNormal>& target_xyzrgbn){
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    
    // transform points to submap frame
    Mat4d pose_in_submap = pose_odom_.inverse() * pose_guess;
    pcl::PointCloud<pcl::PointXYZRGB> source_in_submap;
    pcl::transformPointCloud(source, source_in_submap, pose_in_submap);
   
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'
    size_t n = source_in_submap.size();
    target_color_gradient.resize(n);
    target_xyzrgbn.resize(n);
    std::vector<bool> flag(n);
    std::vector<size_t> idx(n);
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        pcl::PointXYZRGB& src = source_in_submap.points[i];
        double  gx = floor(src.x / grid) * grid;
        double  gy = floor(src.y / grid) * grid;
        double  gz = floor(src.z / grid) * grid;
        double hash = getPosiWithTime(gx, gy, gz, grid, 0);
        auto cell_glb = this->cells_glb.find(hash);
        if(cell_glb != this->cells_glb.end()){
            const Cell* ptr_cell_glb = &(cell_glb->second);
            if((!ptr_cell_glb->not_surface) && (!ptr_cell_glb->empty)){
                for(int dir=0; dir<3; dir++){
                    if(flag[i]) continue;

                    const PointMatrix& points_glb = ptr_cell_glb->ary_cell_vertices[dir];
                    if(points_glb.num_point==0) continue;
                    double ix = (src.x - gx) / resolution, iy = (src.y - gy) / resolution, iz = (src.z - gz) / resolution;
                    int ix_min = floor(ix), iy_min = floor(iy), iz_min = floor(iz);
                    int idx1, idx2, idx3, idx4;
                    // double w1, w2, w3, w4; TODO
                    if(dir==Direction::X){
                        idx1 = iy_min + iz_min * n_row;
                        idx2 = (iy_min+1) + iz_min * n_row;
                        idx3 = iy_min + (iz_min+1) * n_row;
                        idx4 = (iy_min+1) + (iz_min+1) * n_row;
                    }else if(dir==Direction::Y){
                        idx1 = iz_min + ix_min * n_row;
                        idx2 = (iz_min+1) + ix_min * n_row;
                        idx3 = iz_min + (ix_min+1) * n_row;
                        idx4 = (iz_min+1) + (ix_min+1) * n_row;
                    }else if(dir==Direction::Z){
                        idx1 = ix_min + iy_min * n_row;
                        idx2 = (ix_min+1) + iy_min * n_row;
                        idx3 = ix_min + (iy_min+1) * n_row;
                        idx4 = (ix_min+1) + (iy_min+1) * n_row;
                    }
                    
                    // average or max?
                    double avg_var = (points_glb.variance(idx1) + points_glb.variance(idx2)
                                        + points_glb.variance(idx3) + points_glb.variance(idx4)) / 4;
                    double max_var = -1.0;
                    std::vector<int> ids={idx1, idx2, idx3, idx4};
                    for(int k : ids){
                        max_var = points_glb.variance(k) > max_var ? points_glb.variance(k) : max_var;
                    }

                    Vec3d p1 = points_glb.point.col(idx1);
                    Vec3d p2 = points_glb.point.col(idx2);
                    Vec3d p3 = points_glb.point.col(idx3);
                    Vec3d p4 = points_glb.point.col(idx4);
                    
                    Vec3d c1 = points_glb.rgb.col(idx1);
                    Vec3d c2 = points_glb.rgb.col(idx2);
                    Vec3d c3 = points_glb.rgb.col(idx3);
                    Vec3d c4 = points_glb.rgb.col(idx4);

                    Vec3d dp1 = points_glb.color_gradient.col(idx1);
                    Vec3d dp2 = points_glb.color_gradient.col(idx2);
                    Vec3d dp3 = points_glb.color_gradient.col(idx3);
                    Vec3d dp4 = points_glb.color_gradient.col(idx4);

                    

                    std::vector<Vec3d> near_points_submap = {p1,p2,p3,p4};
                    std::vector<Vec3d> near_points_odom={};
                    
                    for(const Vec3d& p: near_points_submap){
                        Vec4d homo_pt={p(0),p(1),p(2),1.};
                        near_points_odom.push_back((pose_odom_ * homo_pt).head(3));
                    }
                    
                    bool dpok = !(dp1.norm() < 0.001 || dp2.norm() < 0.001 || dp3.norm() < 0.001 || dp4.norm() < 0.001);

                    Vec4d plane_para;
                    if(dpok && estiPlane(near_points_odom, 0.1f, plane_para) && max_var>0.){
                        flag[i] = true;
                        Vec3d np = plane_para.head(3);
                        Vec4d src_homo = {src.x, src.y, src.y, 1.};
                        Vec3d src_odom = (pose_odom_ * src_homo).head(3);
                        
                        Vec3d p0 = projectPointToPlane(src_odom, plane_para);
                        double r1 = (p1-p0).norm()+1e-5;
                        double r2 = (p2-p0).norm()+1e-5;
                        double r3 = (p3-p0).norm()+1e-5;
                        double r4 = (p4-p0).norm()+1e-5;
                        double w1 = 1. / r1;
                        double w2 = 1. / r2;
                        double w3 = 1. / r3;
                        double w4 = 1. / r4;
                        double ww = w1+w2+w3+w4;
                        Vec3d c0 = (c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / ww;
                        Vec3d dp0 = (dp1 * w1 + dp2 * w2 + dp3 * w3 + dp4 * w4) / ww;

                        // target_color_gradient[i] = pose_odom_.block<3, 3>(0, 0) * dp0;
                        target_color_gradient[i] = dp0;
                        pcl::PointXYZRGBNormal tgt;
                        tgt.x = p0(0);
                        tgt.y = p0(1);
                        tgt.z = p0(2);
                        tgt.r = c0(0);
                        tgt.g = c0(1);
                        tgt.b = c0(2);
                        tgt.normal_x = np(0);
                        tgt.normal_y = np(1);
                        tgt.normal_z = np(2);
                        target_xyzrgbn[i] = tgt;
                    }
                }
            }
        }
    });
    return flag;
}

void Submap::prepareColorGradient(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target) {
  int cloud_size = static_cast<int> (target->size());
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  kdtree->setInputCloud(target);
  std::vector<int> indices (1);
  std::vector<float> sq_dist (1);
  
  target_color_gradient_.clear();
  target_normal_.clear();
  target_color_gradient_.resize(cloud_size, Eigen::Vector3d::Zero());
  target_normal_.resize(cloud_size, Eigen::Vector3d::Zero());

  // find color gradient for each point in the target cloud
  for (int i = 0; i < cloud_size; ++i) {
    const Eigen::Vector3d p = target->points[i].getVector3fMap().cast<double>();
    
    const Eigen::Vector3d cp = target->points[i].getRGBVector3i().cast<double>();
    double ip = cp.dot(rgb_to_intensity_weight_) / 255.0;

    // search neighbor points for each point in the cloud
    kdtree->radiusSearch(target->points[i], search_radius_, indices, sq_dist);
    int nn_size = static_cast<int> (indices.size());
    if (nn_size < 5) {
    //   LOG(WARNING) << "[WARNING]: not enough points in the neighborhood; size = " << nn_size << "; skipping ... ";
      continue; // TODO: test the influence of leaving color gradient zero --> no contribution from this component
    }

    Vec4d plane;
    std::vector<Vec3d> near_points;
    for(int k = 0; k < 5; ++k) {
      const Eigen::Vector3d pp = target->points[indices[k]].getVector3fMap().cast<double>();
      near_points.push_back(pp);
    }
    if(!estiPlane(near_points, 0.1f, plane)){
        continue;
    } 
    const Eigen::Vector3d np = plane.head<3>().normalized(); // normal vector of the tangent plane at p
    target_normal_[i]= np;
    
    // solve a linear least-squares problem to find the color gradient; see Eq. 10 in the paper
    Eigen::Matrix3Xd A (3, nn_size); // we have transposed A here for the convenience of computation
    Eigen::VectorXd b (nn_size);
    for (int k = 1; k < nn_size; ++k) { // skip index 0, which is the query point itself
      const Eigen::Vector3d pp = target->points[indices[k]].getVector3fMap().cast<double>();
      const Eigen::Vector3d cpp = target->points[indices[k]].getRGBVector3i().cast<double>();
      double ipp = cpp.dot(rgb_to_intensity_weight_) / 255.0;

      // project neighbor points to p's tangent plane
      // pp在tangent plane上的对应点
      const Eigen::Vector3d& pp_proj = pp - (pp-p).dot(np)*np;
      // tangent plane上p指向pp的向量
      A.block<3, 1>(0, k-1) = pp_proj - p;
      // 近邻点与当前点的颜色差值
      b (k-1) = ipp - ip;
    }
    // add orthogonal constraint dp.dot(np) = 0 with weight (nn_size-1)
    A.block<3, 1>(0, nn_size-1) = (nn_size-1) * np;
    b (nn_size-1) = 0; // TODO: test the influence of this constraint --> can make optimization tighter

    // solve for dp
    Eigen::Vector3d X = (A * A.transpose()).ldlt().solve(A * b);
    target_color_gradient_[i] = X;  // X is the estimated color gradient dp
  }
}


void Submap::findMatchPointToVertices(const pcl::PointCloud<pcl::PointXYZ>& points,
                                      const Mat4d& pose_guess, 
                                      std::vector<Point2MeshMatch>& matches){
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    
    // transform points to submap frame
    Mat4d pose_in_submap = pose_odom_.inverse() * pose_guess;
    pcl::PointCloud<pcl::PointXYZ> points_in_submap;
    pcl::transformPointCloud(points, points_in_submap, pose_in_submap);
   
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'
    matches.resize(points_in_submap.points.size()); //for parallel
    std::vector<size_t> idx(points_in_submap.points.size());
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        auto& pt = points_in_submap.points[i];
        Vec3d vpt={pt.x, pt.y, pt.z};
        double  gx = floor(pt.x / grid) * grid;
        double  gy = floor(pt.y / grid) * grid;
        double  gz = floor(pt.z / grid) * grid;
        double hash = getPosiWithTime(gx, gy, gz, grid, 0);
        auto cell_glb = this->cells_glb.find(hash);
        if(cell_glb != this->cells_glb.end()){
            const Cell* ptr_cell_glb = &(cell_glb->second);
            if((!ptr_cell_glb->not_surface) && (!ptr_cell_glb->empty)){
                for(int dir=0; dir<3; dir++){
                    const PointMatrix& points_glb = ptr_cell_glb->ary_cell_vertices[dir];
                    if(points_glb.num_point==0) continue;
                    double ix = (pt.x - gx) / resolution, iy = (pt.y - gy) / resolution, iz = (pt.z - gz) / resolution;
                    int ix_min = floor(ix), iy_min = floor(iy), iz_min = floor(iz);
                    int idx1, idx2, idx3, idx4;
                    if(dir==Direction::X){
                        idx1 = iy_min + iz_min * n_row;
                        idx2 = (iy_min+1) + iz_min * n_row;
                        idx3 = iy_min + (iz_min+1) * n_row;
                        idx4 = (iy_min+1) + (iz_min+1) * n_row;
                    }else if(dir==Direction::Y){
                        idx1 = iz_min + ix_min * n_row;
                        idx2 = (iz_min+1) + ix_min * n_row;
                        idx3 = iz_min + (ix_min+1) * n_row;
                        idx4 = (iz_min+1) + (ix_min+1) * n_row;
                    }else if(dir==Direction::Z){
                        idx1 = ix_min + iy_min * n_row;
                        idx2 = (ix_min+1) + iy_min * n_row;
                        idx3 = ix_min + (iy_min+1) * n_row;
                        idx4 = (ix_min+1) + (iy_min+1) * n_row;
                    }
                    
                    // average or max?
                    double avg_var = (points_glb.variance(idx1) + points_glb.variance(idx2)
                                        + points_glb.variance(idx3) + points_glb.variance(idx4)) / 4;
                    double max_var = -1.0;
                    std::vector<int> ids={idx1, idx2, idx3, idx4};
                    for(int k : ids){
                        max_var = points_glb.variance(k) > max_var ? points_glb.variance(k) : max_var;
                    }

                    Vec3d p1 = points_glb.point.col(idx1);
                    Vec3d p2 = points_glb.point.col(idx2);
                    Vec3d p3 = points_glb.point.col(idx3);
                    Vec3d p4 = points_glb.point.col(idx4);
                    Vec4d plane_para;
                    std::vector<Vec3d> near_points_submap={p1,p2,p3,p4};
                    std::vector<Vec3d> near_points_odom={};
                    for(const Vec3d& p: near_points_submap){
                        Vec4d homo_pt={p(0),p(1),p(2),1.};
                        near_points_odom.push_back((pose_odom_ * homo_pt).head(3));
                    }
                    // estimate the plane parameters in odom frame
                    if(estiPlane(near_points_odom, 0.1f, plane_para) && max_var>0.){
                        Point2MeshMatch match;
                        Vec4d homo_vpt = {vpt(0),vpt(1),vpt(2),1.};
                        match = Point2MeshMatch((pose_odom_ * homo_vpt).head(3), plane_para, max_var, true);
                        const auto& p_raw = points.points[i];
                        match.point_in_body = Vec3d(p_raw.x, p_raw.y, p_raw.z);
                        matches[i]=match;
                    }
                }
            }
        }
    });
}

void Submap::findMatchCellToCell(const pcl::PointCloud<pcl::PointXYZ>& points,
                                      const Mat4d& pose_guess, 
                                      std::vector<Point2MeshMatch>& matches){
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    double variance_thr = Parameter::GetInstance().variance_register;
    
    // LOG(INFO);
    // transform points to submap frame
    PointMatrix pts_now(points);
    // LOG(INFO);
    Mat4d T_b2s = pose_odom_.inverse() * pose_guess;//body to submap
    pts_now.transformPoints(T_b2s);
    dividePointsIntoCell(pts_now);
    
    // LOG(INFO);
    Mat4d pose_guess_inv = pose_guess.inverse();

    int step = 1;
    OverlapCellsRelation overlap = overlapCellsCrossCell(step);
    std::vector<size_t> idx(overlap.cells_now.size());
    iota(idx.begin(), idx.end(),  0);
    matches.resize(overlap.cells_now.size() * 3 * num_test_square);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
    // for(int i=0; i<overlap.cells_now.size();i++){
        if(overlap.cells_now[i]){
            for(int dir = 0; dir < 3; dir++){
                // find closest
                Eigen::Matrix<double, 5, Eigen::Dynamic> closest_point_glb = Eigen::MatrixXd::Zero(5, num_test_square);
                closest_point_glb = Eigen::MatrixXd::Zero(5, num_test_square);
                closest_point_glb.row(4).fill(-1);//distance, -1: not found
                PointMatrix& point_now = (*overlap.cells_now[i]).ary_cell_vertices[dir];
                if(point_now.num_point == 0){
                    continue;
                }
                // LOG(INFO);
                for(int k = -1 * step; k <= step; k++) {
                    int index;
                    if (k == 0) {
                        index = 0;
                    } else if (k < 0) {
                        index = 1 + 2 * step * dir + k + step;
                    } else {
                        index = 1 + 2 * step * dir + k + step / 2;
                    }
                    // LOG(INFO)<<overlap.multi_cells_glb[i].size()<<","<<index;
                    if (overlap.multi_cells_glb[i][index] == nullptr) {
                        continue;
                    }
                    // LOG(INFO)<<index;
                    PointMatrix& points_glb = (*overlap.multi_cells_glb[i][index]).ary_cell_vertices[dir];
                    if (points_glb.num_point == 0) {
                        continue;
                    }
                    // LOG(INFO)<<index;
                    if(index == 0){
                        for (int j = 0; j < point_now.num_point; j++) {
                            double distance_sqrt = (points_glb.point(dir, j) - point_now.point(dir, j)) *
                                                (points_glb.point(dir, j) - point_now.point(dir, j));
                            if (closest_point_glb(4, j) < 0) {
                                closest_point_glb.col(j).topRows(3) = points_glb.point.col(j);
                                closest_point_glb.col(j).row(3) = points_glb.variance.col(j);
                                closest_point_glb(4, j) = distance_sqrt;
                            } else if (distance_sqrt < closest_point_glb(4, j)) {
                                closest_point_glb.col(j).topRows(3) = points_glb.point.col(j);
                                closest_point_glb.col(j).row(3) = points_glb.variance.col(j);
                                closest_point_glb(4, j) = distance_sqrt;
                            }
                        }
                    }else{
                        for (int j = 0; j < point_now.num_point; j++) {
                            if ((points_glb.variance(0, j) < variance_thr)) {
                                double dis_square = (points_glb.point(dir, j) - point_now.point(dir, j)) *
                                                    (points_glb.point(dir, j) - point_now.point(dir, j));
                                if (closest_point_glb(4, j) < 0) {
                                    closest_point_glb.col(j).topRows(3) = points_glb.point.col(j);
                                    closest_point_glb.col(j).row(3) = points_glb.variance.col(j);
                                    closest_point_glb(4, j) = dis_square;
                                } else if (dis_square < closest_point_glb(4, j) || closest_point_glb(3, j) > variance_thr) {
                                    closest_point_glb.col(j).topRows(3) = points_glb.point.col(j);
                                    closest_point_glb.col(j).row(3) = points_glb.variance.col(j);
                                    closest_point_glb(4, j) = dis_square;
                                }
                            }
                        }
                    }
                    // LOG(INFO);
                }

                // LOG(INFO);

                // normal estimation
                PointMatrix normal_cell(num_test_square);
                Eigen::Matrix<int, 2, 4> offset;
                offset << 1, 0,-1, 0,
                        0,-1, 0, 1;
                for(int j = 0; j < num_test_square; j++) {
                    Eigen::Vector3d normal_at_j;
                    normal_at_j.setZero();
                    if(closest_point_glb(4, j) < 0) {
                        continue;
                    }
                    int num_valid_neighboor = 0;
                    int ix_cell = j % (n_row);
                    int iy_cell = j / (n_row);

                    std::vector<Vec3d> near_points_submap;
                    double max_var = -1.;
                    for(int i_offset=0; i_offset<4; i_offset++) {
                        if (ix_cell + offset(0, i_offset) < n_row &&
                            ix_cell + offset(0, i_offset) >= 0 &&
                            iy_cell + offset(1, i_offset) < n_row &&
                            iy_cell + offset(1, i_offset) >= 0){

                            int idx_1 = j + offset(0, i_offset) +           n_row * offset(1, i_offset);
                            if(closest_point_glb(4, idx_1) > 0){
                                Point curr_index_1 = closest_point_glb.col(j).topRows(3) - closest_point_glb.col(idx_1).topRows(3);
                                if(curr_index_1.norm() > 1.0 * grid){
                                    continue;
                                }
                                num_valid_neighboor++;
                                near_points_submap.push_back(closest_point_glb.col(idx_1).topRows(3));
                                max_var = closest_point_glb(3, idx_1) > max_var ? closest_point_glb(3, idx_1) : max_var;
                            }
                        }
                    }
                    if(num_valid_neighboor == 4){
                        // estimate the plane parameters in odom frame
                        Vec4d plane_para;
                        std::vector<Vec3d> near_points_odom={};
                        for(const Vec3d& p: near_points_submap){
                            Vec4d homo_pt={p(0), p(1), p(2), 1.};
                            near_points_odom.push_back((pose_odom_ * homo_pt).head(3));
                        }
                        if(estiPlane(near_points_odom, 0.1f, plane_para) && max_var>0.){
                            Point2MeshMatch match;
                            Vec3d pj = point_now.point.col(j); //in submap frame
                            Vec4d homo_vpt = {pj(0), pj(1), pj(2), 1.};
                            match = Point2MeshMatch((pose_odom_ * homo_vpt).head(3), plane_para, max_var, true);
                            match.point_in_body = (pose_guess_inv * (pose_odom_ * homo_vpt)).head(3);
                            matches[i * num_test_square * 3 + dir * num_test_square + j] = match;
                        }
                    }
                }
                // LOG(INFO);
            }
        }
    });
    // LOG(INFO);
}

void Submap::findMatchVisualPointToVertices(const pcl::PointCloud<pcl::PointXYZ>& points_w,
                                      std::vector<Point2MeshMatch>& matches){
    int num_test_square = Parameter::GetInstance().num_test * Parameter::GetInstance().num_test;
    int n_row =  Parameter::GetInstance().num_test;
    double grid = Parameter::GetInstance().grid;
    
    // transform points to submap frame
    pcl::PointCloud<pcl::PointXYZ> points_in_submap;
    Mat4d T_w2s = pose_odom_.inverse();
    pcl::transformPointCloud(points_w, points_in_submap, T_w2s);
   
    double resolution = grid / (n_row-1); //TODO, check parameter 'full_cover'
    matches.resize(points_in_submap.points.size()); //for parallel
    std::vector<size_t> idx(points_w.points.size());
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par_unseq, idx.begin(), idx.end(), [&](auto &i){
        auto& pt = points_in_submap.points[i];
        Vec3d vpt={pt.x, pt.y, pt.z};
        double  gx = floor(pt.x / grid) * grid;
        double  gy = floor(pt.y / grid) * grid;
        double  gz = floor(pt.z / grid) * grid;
        double hash = getPosiWithTime(gx, gy, gz, grid, 0);
        auto cell_glb = this->cells_glb.find(hash);
        if(cell_glb != this->cells_glb.end()){
            const Cell* ptr_cell_glb = &(cell_glb->second);
            if((!ptr_cell_glb->not_surface) && (!ptr_cell_glb->empty)){
                for(int dir=0; dir<3; dir++){
                    const PointMatrix& points_glb = ptr_cell_glb->ary_cell_vertices[dir];
                    if(points_glb.num_point==0) continue;
                    double ix = (pt.x - gx) / resolution, iy = (pt.y - gy) / resolution, iz = (pt.z - gz) / resolution;
                    int ix_min = floor(ix), iy_min = floor(iy), iz_min = floor(iz);
                    int idx1, idx2, idx3, idx4;
                    if(dir==Direction::X){
                        idx1 = iy_min + iz_min * n_row;
                        idx2 = (iy_min+1) + iz_min * n_row;
                        idx3 = iy_min + (iz_min+1) * n_row;
                        idx4 = (iy_min+1) + (iz_min+1) * n_row;
                    }else if(dir==Direction::Y){
                        idx1 = iz_min + ix_min * n_row;
                        idx2 = (iz_min+1) + ix_min * n_row;
                        idx3 = iz_min + (ix_min+1) * n_row;
                        idx4 = (iz_min+1) + (ix_min+1) * n_row;
                    }else if(dir==Direction::Z){
                        idx1 = ix_min + iy_min * n_row;
                        idx2 = (ix_min+1) + iy_min * n_row;
                        idx3 = ix_min + (iy_min+1) * n_row;
                        idx4 = (ix_min+1) + (iy_min+1) * n_row;
                    }
                    
                    // average or max?
                    double avg_var = (points_glb.variance(idx1) + points_glb.variance(idx2)
                                        + points_glb.variance(idx3) + points_glb.variance(idx4)) / 4;
                    double max_var = -1.0;
                    std::vector<int> ids={idx1, idx2, idx3, idx4};
                    for(int k : ids){
                        max_var = points_glb.variance(k) > max_var ? points_glb.variance(k) : max_var;
                    }

                    Vec3d p1 = points_glb.point.col(idx1);
                    Vec3d p2 = points_glb.point.col(idx2);
                    Vec3d p3 = points_glb.point.col(idx3);
                    Vec3d p4 = points_glb.point.col(idx4);
                    Vec4d plane_para;
                    std::vector<Vec3d> near_points_submap={p1,p2,p3,p4};
                    std::vector<Vec3d> near_points_odom={};
                    for(const Vec3d& p: near_points_submap){
                        Vec4d homo_pt={p(0),p(1),p(2),1.};
                        near_points_odom.push_back((pose_odom_ * homo_pt).head(3));
                    }
                    // estimate the plane parameters in odom frame
                    if(estiPlane(near_points_odom, 0.1f, plane_para) && max_var>0.){
                        Point2MeshMatch match;
                        Vec4d homo_vpt = {vpt(0),vpt(1),vpt(2),1.};
                        match = Point2MeshMatch((pose_odom_ * homo_vpt).head(3), plane_para, max_var, true);
                        matches[i]=match;
                    }
                }
            }
        }
    });
}

OverlapCellsRelation Submap::overlapCells(){
    OverlapCellsRelation overlap_ship;

    for(int i = 0; i < ary_cells_now_.size(); i++){
        Cell& i_cell_now = ary_cells_now_[i];
        if(i_cell_now.empty) continue;

        //don't consider revisit
        auto i_cell_glb = cells_glb.find(i_cell_now.hash_position);
        if(i_cell_glb != cells_glb.end() && (Parameter::GetInstance().num_margin_old_cell < 0 ||
              std::abs(this->step - i_cell_glb->second.time_stamp) < Parameter::GetInstance().num_margin_old_cell)){
            Cell* ptr_cell_glb = &(i_cell_glb->second);
            Cell* ptr_cell_now = &i_cell_now;
            overlap_ship.cells_glb.push_back(ptr_cell_glb);
            overlap_ship.cells_now.push_back(ptr_cell_now);
        }else{
            Cell* ptr_cell_new = &i_cell_now;
            overlap_ship.cells_now_new.push_back(ptr_cell_new);
        }
    }
    return overlap_ship;
}


OverlapCellsRelation Submap::overlapCellsCrossCell(int overlap_length){
    double grid = Parameter::GetInstance().grid;
    OverlapCellsRelation overlap_ship;
    
    for(int i = 0; i < ary_cells_now_.size(); i++){
        Cell& i_cell_now = ary_cells_now_[i];
        std::vector<Cell*> multi_overlap_cell_glb(6 * overlap_length + 1);
        std::vector<double> overlap_index(6 * overlap_length + 1);

        //don't consider revisit
        auto i_cell_glb = cells_glb.find(i_cell_now.hash_position);
        if(i_cell_glb != cells_glb.end() && (Parameter::GetInstance().num_margin_old_cell < 0 ||
              std::abs(this->step - i_cell_glb->second.time_stamp) < Parameter::GetInstance().num_margin_old_cell)){
            Cell* ptr_cell_glb = &(i_cell_glb->second);
            Cell* ptr_cell_now = &i_cell_now;
            overlap_ship.cells_glb.push_back(ptr_cell_glb);
            overlap_ship.cells_now.push_back(ptr_cell_now);
            multi_overlap_cell_glb[0]=(ptr_cell_glb);
            overlap_index[0] = i_cell_now.hash_position;
        }else{
            Cell* ptr_cell_new = &i_cell_now;
            overlap_ship.cells_now_new.push_back(ptr_cell_new);
            multi_overlap_cell_glb[0]=(nullptr);
            overlap_ship.cells_now.push_back(ptr_cell_new);
            overlap_index[0] = i_cell_now.hash_position;
        }

        //cross overlap
        for(int iz = -1 * overlap_length; iz <= overlap_length; iz++){
            for(int iy = -1 * overlap_length; iy <= overlap_length; iy++){
                for(int ix = -1 * overlap_length; ix <= overlap_length; ix++){
                    if((ix * iy == 0) + (ix * iz == 0) + (iz * iy == 0) >=3){
                        int index = 1 + (ix + iy + iz + overlap_length / (1 + (ix + iy + iz > 0))) +
                                  (iy != 0) * overlap_length * 2 +
                                  (iz != 0) * overlap_length * 4;
                        double tmp_posi = getPosiWithTime(i_cell_now.region.x_min + ix * grid,
                                                          i_cell_now.region.y_min + iy * grid,
                                                          i_cell_now.region.z_min + iz * grid, grid, 0);
                        auto neighboor_cell_glb = cells_glb.find(tmp_posi);
                        overlap_index[index] = tmp_posi;
                        if(neighboor_cell_glb != cells_glb.end() && (Parameter::GetInstance().num_margin_old_cell < 0 ||
                                std::abs(this->step - neighboor_cell_glb->second.time_stamp) < Parameter::GetInstance().num_margin_old_cell)){
                            Cell *ptr_cell_glb = &(neighboor_cell_glb->second);
                            multi_overlap_cell_glb[index] = (ptr_cell_glb);
                        }
                        else{
                            multi_overlap_cell_glb[index]=(nullptr);
                        }
                    }
                }
            }
        }
        overlap_ship.multi_cells_glb.push_back(multi_overlap_cell_glb);
    }
    
    return overlap_ship;
}


void  Submap::filterVerticesByVariance(bool global_map, double variance_thr){
    vertices_filted.clear_quick();
    for(int dir = 0; dir < 3; dir++){
        ary_vertices_all_show[dir].clear_quick();
    }

    bool showxyzmap = false;

    //collect point
    if(!global_map){
        for(int i_cell=0; i_cell < ary_cells_now_.size(); i_cell++){
            Cell& cell_now = ary_cells_now_[i_cell];
            for(int dir = 0; dir < 3; dir ++){
                PointMatrix & points_filt = cell_now.ary_cell_vertices[dir];
                for(int i_point=0; i_point < points_filt.num_point; i_point++){
                    if(((points_filt.variance(0, i_point)) < variance_thr)){
                        vertices_filted.addPointWithVariance(points_filt.getPointWithVariance(i_point), points_filt.rgb.col(i_point));
                    }
                }
            }
        }
    }else{//map glb
        for(auto& cell_glb : cells_glb){
            for(int dir = 0; dir < 3; dir++){
                PointMatrix & point_filt = cell_glb.second.ary_cell_vertices[dir];
                for(int i = 0; i < point_filt.num_point; i++){
                    if(((point_filt.variance(0,i)) < variance_thr)){
                        vertices_filted.addPointWithVariance(point_filt.getPointWithVariance(i), point_filt.rgb.col(i));
                        if(showxyzmap){
                            ary_vertices_all_show[dir].addPointWithVariance(point_filt.getPointWithVariance(i), point_filt.rgb.col(i));
                        }
                    }
                }
            }
        }
    }
}

void  Submap::filterVerticesByVarianceOverlap(double variance_thr, Submap & map_now){
    //used to draw map only overlap region, in order to save time
    //clear
    vertices_filted.clear();
    for(int dir = 0; dir < 3; dir++){
        ary_vertices_all_show[dir].clear();
    }
    bool only_pub_overlap_region = true;
    bool showxyzmap = true;

    OverlapCellsRelation overlap_ship = map_now.overlapCells();
    for(auto& i_cell_glb : overlap_ship.cells_glb){
        for(int dir = 0; dir < 3; dir++){
            PointMatrix & points_filt = (*i_cell_glb).ary_cell_vertices[dir];
            for(int i = 0; i < points_filt.num_point; i++){
                if(((points_filt.variance(0, i)) < variance_thr)){
                    vertices_filted.addPointWithVariance(points_filt.getPointWithVariance(i), points_filt.rgb.col(i));
                    if(showxyzmap && name == "MAP_GLB"){
                        ary_vertices_all_show[dir].addPointWithVariance(points_filt.getPointWithVariance(i), points_filt.rgb.col(i));
                    }
                }
            }
        }
    }
}


void Submap::updateMap(size_t frame_id, const Mat4d& opt_pose,  const cv::Mat& image, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_raw_ptr){
    step = frame_id;
    frame_ids_.push_back(frame_id);
    node_poses_.push_back(opt_pose);

    points_turned.clear_quick();
    vertices_filted.clear_quick();
    for(int i = 0; i < 3; i ++){
        ary_vertices_all_show[i].clear_quick();
        ary_overlap_vertices[i].clear_quick();
    }
    
    if(!preprocessScan(pcl_raw_ptr, points_turned, pcl_raw, Parameter::GetInstance().voxel_size)){
        return;
    }
    cv::Mat img_gray;
    if(image.channels() == 3){
        cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
    }else{
        image.copyTo(img_gray);
    }

    // for visualization
    points_turned_odom = points_turned;
    points_turned_odom.transformPoints(opt_pose);

    Mat4d T_si = pose_odom_.inverse() * opt_pose;
    points_turned.transformPoints(T_si);
    dividePointsIntoCell(points_turned);

    Mat4d T_I_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
    Mat4d T_sc = T_si * T_I_lC;
    Mat4d T_cs = T_sc.inverse();
    //find overlapped regions
    OverlapCellsRelation overlap_ship = overlapCells();
    newly_updated_cells.clear();
    //update overlapped region
    for(int i = 0; i < overlap_ship.cells_now.size(); i++){
        if((overlap_ship.cells_glb[i] != nullptr) &&
           (overlap_ship.cells_now[i] != nullptr)){
            Cell & cell_glb = *overlap_ship.cells_glb[i];
            Cell & cell_now = *overlap_ship.cells_now[i];
            // case1: 本不是表面，但新的帧中是表面
            if(cell_glb.not_surface && !cell_now.not_surface){
                cell_glb.cell_raw_points += cell_now.cell_raw_points;
                cell_glb.reconstructSurfaces(true);
            }else{
                for(int i_dir = 0; i_dir < 3; i_dir ++){
                    if(cell_glb.ary_cell_vertices[i_dir].num_point != 0 &&
                       cell_now.ary_cell_vertices[i_dir].num_point != 0 ) {//global mesh内该cell是表面时，对顶点进行更新
                        cell_glb.updateVertices(cell_now, Direction(i_dir));   
                    }else if(cell_glb.ary_cell_vertices[i_dir].num_point == 0 &&
                            cell_now.ary_cell_vertices[i_dir].num_point != 0){
                        //cells_glb.ary_cell_vertices[i_dir] = cells_now.ary_cell_vertices[i_dir];
                    }
                }
            }
            // TODO(wz)
            cell_glb.time_stamp = this->step;
            cell_glb.updateViewedLocation(node_poses_.back());
            // 对Overlap区域内的Cell更新参考图像patch
            // updateReferencePatch(img_gray, T_cs, cell_glb);
            newly_updated_cells.push_back(& cell_glb);
        }
    }

    //add new cell
    if( !overlap_ship.cells_now_new.empty()){
        for(auto& cell_now_new : overlap_ship.cells_now_new){
            Point current_pose, current_viewed_dir;
            current_pose = trans3Dpoint(0, 0, 0, node_poses_.back());
            current_viewed_dir = (current_pose - cell_now_new->center) / (current_pose - cell_now_new->center).norm();
            double tmp_posi = getPosiWithTime(cell_now_new->region.x_min, cell_now_new->region.y_min, cell_now_new->region.z_min, 
                                              Parameter::GetInstance().grid, cell_now_new->revisited);
            Cell tmp_cell((*cell_now_new).cell_raw_points, (*cell_now_new).ary_cell_vertices, frame_ids_.back(),
                           tmp_posi, (*cell_now_new).region);
            tmp_cell.updateViewedLocation(node_poses_.back());
            
            // 新建的Cell赋予参考图像patch
            // updateReferencePatch(img_gray, T_cs, tmp_cell);
            pair<std::unordered_map<double, Cell>::iterator, bool> inserted_cell = cells_glb.emplace(tmp_posi, tmp_cell);

            newly_updated_cells.push_back(& (inserted_cell.first->second));
        }
    }


    // if(Parameter::GetInstance().enable_mesh_texture){
    //     pointclouds_for_texture_.push_back(pcl_raw_ptr);
    //     images_for_texture_.push_back(image);
    // }

    
    // UpdateVertexColors(T_c2s.inverse(), image);
}


void  Submap::filterMeshLocal(){
    //mesh_msg current scan by direct connectin, visualized by mesh_tool
    double variance_point = -1;
    double variance_face = Parameter::GetInstance().variance_map_show;
    int n_row(Parameter::GetInstance().num_test), n_col(Parameter::GetInstance().num_test);
    double grid = Parameter::GetInstance().grid;
    //init msg
    mesh_msgs::MeshGeometryStamped tmp_mesh;
    mesh_msgs::MeshVertexColors tmp_mesh_vertices_colors;
    tmp_mesh.header.frame_id = "map";
    tmp_mesh.header.stamp = ros::Time::now();
    
    Mat4d T_I_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();

    for(int i = 0; i < ary_cells_now_.size(); i++){
        // std::pair<double, Cell> & i_cell = cells_now[index_bucket_enough_point[i]];
        Cell& i_cell = ary_cells_now_[i];
        if(i_cell.empty) continue;

        // in each cell, dir, filter mesh_msg under vatiance threshold
        for(int dir = 0; dir < 3; dir ++){
            PointMatrix & vertices =  i_cell.ary_cell_vertices[dir];
            if(vertices.num_point != 0){
                //inside mesh_msg
                int start_vertex_i = tmp_mesh.mesh_geometry.vertices.size();
                for(int vertex_i = 0; vertex_i < vertices.num_point ; vertex_i++){
                    // transform to odom frame
                    Vec3d vp = vertices.point.col(vertex_i);
                    Vec3d rgb = vertices.rgb.col(vertex_i);
                    vp = pose_odom_.block<3,3>(0, 0) * vp + pose_odom_.block<3,1>(0, 3);

                    geometry_msgs::Point tmp_point;
                    tmp_point.x =  vp(0);
                    tmp_point.y =  vp(1);
                    tmp_point.z =  vp(2);
                    tmp_mesh.mesh_geometry.vertices.push_back(tmp_point);

                    std_msgs::ColorRGBA vtx_rgb;
                    vtx_rgb.a = 1.0;                
                    vtx_rgb.r = rgb[0];
                    vtx_rgb.g = rgb[1];
                    vtx_rgb.b = rgb[2];
                    tmp_mesh_vertices_colors.vertex_colors.push_back(vtx_rgb);
                }
                for(int vertex_i = 0; vertex_i < vertices.num_point; vertex_i++){
                    int ix_cell = vertex_i % (n_row);
                    int iy_cell = vertex_i / (n_row);
                    mesh_msgs::MeshTriangleIndices tmp_face;
                    if(ix_cell + 1 < n_row && iy_cell - 1 >= 0){
                        double variance_face_i = (vertices.variance(0, vertex_i) +
                                                  vertices.variance(0, vertex_i + 1) +
                                                  vertices.variance(0, vertex_i + 1 - n_row)) / 3.0;
                        if(variance_face_i < variance_face){
                            tmp_face.vertex_indices[0] = start_vertex_i + vertex_i;
                            tmp_face.vertex_indices[1] = start_vertex_i + vertex_i + 1;
                            tmp_face.vertex_indices[2] = start_vertex_i + vertex_i + 1 - n_row;
                            tmp_mesh.mesh_geometry.faces.push_back(tmp_face);
                        }
                    }
                    if(ix_cell + 1 < n_row && iy_cell + 1 < n_row ){
                        double variance_face_i = (vertices.variance(0, vertex_i) +
                                                  vertices.variance(0, vertex_i + 1) +
                                                  vertices.variance(0, vertex_i + n_row)) / 3.0;
                        if(variance_face_i < variance_face){
                            tmp_face.vertex_indices[0] = start_vertex_i + vertex_i;
                            tmp_face.vertex_indices[1] = start_vertex_i + vertex_i + 1;
                            tmp_face.vertex_indices[2] = start_vertex_i + vertex_i + n_row;
                            tmp_mesh.mesh_geometry.faces.push_back(tmp_face);
                        }
                    }
                }
            }
        }
    }
    tmp_mesh.mesh_vertex_color = tmp_mesh_vertices_colors;
    mesh_msg_now_ = tmp_mesh;
    mesh_msg_now_.uuid = std::to_string(tmp_mesh.header.stamp.toNSec());
}

void  Submap::filterMeshGlb(bool use_pose_slam){
    //mesh_msg global scan by direct connectin, visualized by mesh_tool, need to traverse all cell in map
    double variance_point = -1;
    double variance_face_show = Parameter::GetInstance().variance_map_show;
    int n_row(Parameter::GetInstance().num_test), n_col(Parameter::GetInstance().num_test);
    double grid = Parameter::GetInstance().grid;
    int min_show_updated_times = 1;
    double max_show_distance = 70;
    //init msg
    mesh_msgs::MeshGeometryStamped tmp_mesh;
    mesh_msgs::MeshVertexColors tmp_mesh_vertices_colors;
    
    Mat4d T_I_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();

    TicToc t_connect_push("t_connect_push");
    for (auto & i_cell : cells_glb){
        // in each cell, dir, build mesh_msg
        // if(i_cell.second.average_viewed_distance > max_show_distance ){
        //     continue;
        // }
        for(int dir = 0; dir < 3; dir ++){
            // if(i_cell.second.updated_times[dir] < min_show_updated_times){
            //     continue;
            // }
            PointMatrix & vertices = i_cell.second.ary_cell_vertices[dir];
            if(vertices.num_point == 0){
                continue;
            }
            //inside mesh_msg
            int start_vertex_i = tmp_mesh.mesh_geometry.vertices.size();
            for(int vertex_i = 0; vertex_i < vertices.num_point ; vertex_i ++){
                // transform to odom frame
                Vec3d vp = vertices.point.col(vertex_i);
                Vec3d rgb = vertices.rgb.col(vertex_i);
                if(use_pose_slam){
                    vp = pose_slam_.block<3,3>(0, 0) * vp + pose_slam_.block<3,1>(0, 3);
                }else{
                    vp = pose_odom_.block<3,3>(0, 0) * vp + pose_odom_.block<3,1>(0, 3);
                }
                
                geometry_msgs::Point tmp_point;
                tmp_point.x =  vp(0);
                tmp_point.y =  vp(1);
                tmp_point.z =  vp(2);
                tmp_mesh.mesh_geometry.vertices.push_back(tmp_point);
                
                std_msgs::ColorRGBA vtx_rgb;
                vtx_rgb.a = 1.0;
                vtx_rgb.r = rgb[0];
                vtx_rgb.g = rgb[1];
                vtx_rgb.b = rgb[2];

                tmp_mesh_vertices_colors.vertex_colors.push_back(vtx_rgb);
            }
            for(int vertex_i = 0; vertex_i < vertices.num_point; vertex_i++){
                int ix_cell = vertex_i % (n_row);
                int iy_cell = vertex_i / (n_row);
                mesh_msgs::MeshTriangleIndices tmp_face;
                if(ix_cell + 1 < n_row && iy_cell - 1 >= 0){
                    double variance_face_i = (vertices.variance(0, vertex_i) +
                                              vertices.variance(0, vertex_i + 1) +
                                              vertices.variance(0, vertex_i + 1 - n_row)) / 3.0;
                    if(1){//variance_face_i < variance_face_show
                        tmp_face.vertex_indices[0] = start_vertex_i + vertex_i;
                        tmp_face.vertex_indices[1] = start_vertex_i + vertex_i + 1;
                        tmp_face.vertex_indices[2] = start_vertex_i + vertex_i + 1 - n_row;
                        tmp_mesh.mesh_geometry.faces.push_back(tmp_face);
                    }
                }
                if(ix_cell + 1 < n_row && iy_cell + 1 < n_row ){
                    double variance_face_i = (vertices.variance(0, vertex_i) +
                                              vertices.variance(0, vertex_i + 1) +
                                              vertices.variance(0, vertex_i + n_row)) / 3.0;
                    if(1){//variance_face_i < variance_face_show
                        tmp_face.vertex_indices[0] = start_vertex_i + vertex_i;
                        tmp_face.vertex_indices[1] = start_vertex_i + vertex_i + 1;
                        tmp_face.vertex_indices[2] = start_vertex_i + vertex_i + n_row;
                        tmp_mesh.mesh_geometry.faces.push_back(tmp_face);
                    }
                }
            }
        }
    }
    tmp_mesh.mesh_vertex_color = tmp_mesh_vertices_colors;
    mesh_msg_full_ = tmp_mesh;
    mesh_msg_full_.header.frame_id = "map";
    mesh_msg_full_.header.stamp = ros::Time::now();
    mesh_msg_full_.uuid = std::to_string(mesh_msg_full_.header.stamp.toNSec());
}

pcl::PointCloud<pcl::PointXYZ> Submap::getAllVertices(){
    // you should call filterMeshGlb first.
    pcl::PointCloud<pcl::PointXYZ> vcloud;
    size_t n = mesh_msg_full_.mesh_geometry.vertices.size();
    if(n==0){
        LOG(WARNING)<<"The mesh vertices of this submap is empty, you should call filterMeshGlb first!";
        return vcloud;
    }
    vcloud.points.resize(n);
    for(size_t i = 0; i < n; ++i){
        pcl::PointXYZ v;
        v.x = mesh_msg_full_.mesh_geometry.vertices[i].x;
        v.y = mesh_msg_full_.mesh_geometry.vertices[i].y;
        v.z = mesh_msg_full_.mesh_geometry.vertices[i].z;
        vcloud.points[i] = v;
    }
    return vcloud;
}

void Submap::UpdateVertexColors(const Mat4d& T_s2c, const cv::Mat& image){
    for (auto & i_cell : cells_glb){
        for(int dir = 0; dir < 3; dir ++){
            PointMatrix & vertices = i_cell.second.ary_cell_vertices[dir];
            if(vertices.num_point == 0){
                continue;
            }
            std::vector<size_t> idx(vertices.num_point);
            iota(idx.begin(), idx.end(),  0);
            std::for_each(idx.begin(), idx.end(), [&](int i){
                Vec3d vp = vertices.point.col(i);
                Vec3d p_cam = T_s2c.block<3,3>(0,0) * vp + T_s2c.block<3,1>(0,3);
                Vec2d pxpy;
                camera_left_->spaceToPlane(p_cam, pxpy);
                if (pxpy.x() >= 0 && pxpy.x() < image.cols && pxpy.y() >= 0 && pxpy.y() < image.rows) {
                    cv::Vec3b color = image.at<cv::Vec3b>(pxpy.y(), pxpy.x());
                    Vec3d rgb(double(color[2])/255.0, double(color[1])/255.0, double(color[0])/255.0);
                    i_cell.second.updateVerticeColor(Direction(dir), i, rgb, p_cam.norm(), p_cam);
                }
            });
        }
    }
}


bool  Submap::outputMeshAsPly(const std::string& filename, const mesh_msgs::MeshGeometryStamped & mesh_msg_to_save) {
    //save the mesh in mesh_tool msg
    ROS_DEBUG("outputMeshAsPly");
    //std::ofstream stream(filename.c_str());
    std::ofstream stream;
    stream.open(filename, std::ios::out);
    if (!stream) {
        ROS_WARN("Can not open outputMeshAsPly file");
        std::cout <<"wrong file location of mesh_msg_to_save to save:" <<filename<<std::endl;
        return false;
    }

    size_t num_points = 3 * mesh_msg_to_save.mesh_geometry.faces.size();
    if(num_points == 0) {
        ROS_WARN("Empty mesh msg");
        return false;
    }
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << num_points << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
/*
    if (mesh_msg_to_save.hasNormals()) {
        stream << "property float normal_x" << std::endl;
        stream << "property float normal_y" << std::endl;
        stream << "property float normal_z" << std::endl;
    }
    if (mesh_msg_to_save.hasColors()) {
        stream << "property uchar red" << std::endl;
        stream << "property uchar green" << std::endl;
        stream << "property uchar blue" << std::endl;
        stream << "property uchar alpha" << std::endl;
    }
*/

    //if (mesh_msg_to_save.hasTriangles()) {
    stream << "element face " << mesh_msg_to_save.mesh_geometry.faces.size() << std::endl;
    stream << "property list uchar int vertex_indices"
           << std::endl;  // pcl-1.7(ros::kinetic) breaks ply convention by not
    // using "vertex_index"
    //}
    stream << "end_header" << std::endl;

    size_t vertices_index = 0;
    for (const mesh_msgs::MeshTriangleIndices& vert : mesh_msg_to_save.mesh_geometry.faces) {
        stream << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[0]].x << " "
               << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[0]].y << " "
               << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[0]].z << std::endl;
        stream << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[1]].x << " "
               << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[1]].y << " "
               << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[1]].z << std::endl;
        stream << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[2]].x << " "
               << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[2]].y << " "
               << mesh_msg_to_save.mesh_geometry.vertices[vert.vertex_indices[2]].z << std::endl;

/*        if (mesh_msg_to_save.hasNormals()) {
            const Point& normal = mesh_msg_to_save.normals[vertices_index];
            stream << " " << normal.x() << " " << normal.y() << " " << normal.z();
        }
        if (mesh_msg_to_save.hasColors()) {
            const Color& color = mesh_msg_to_save.colors[vertices_index];
            int r = static_cast<int>(color.r);
            int g = static_cast<int>(color.g);
            int b = static_cast<int>(color.b);
            int a = static_cast<int>(color.a);
            // Uint8 prints as character otherwise. :(
            stream << " " << r << " " << g << " " << b << " " << a;
        }*/

        vertices_index ++;
    }
    //if (mesh_msg_to_save.hasTriangles()) {
    for (size_t i = 0; i < (3 * mesh_msg_to_save.mesh_geometry.faces.size()); i += 3) {
        stream << "3 ";

        for (int j = 0; j < 3; j++) {
            stream << i + j << " ";
        }
        stream << std::endl;
    }
    //}
    stream.close();
    return true;
}

void Submap::textureAndSaveObj(bool use_pose_slam) {
    // Implement the texture mapping for the submap's mesh
    if(!Parameter::GetInstance().enable_mesh_texture){
        return;
    }

    double variance_point = -1;
    double variance_face_show = Parameter::GetInstance().variance_map_show;
    int n_row(Parameter::GetInstance().num_test), n_col(Parameter::GetInstance().num_test);
    double grid = Parameter::GetInstance().grid;
    int min_show_updated_times = 1;
    double max_show_distance = 70;

    std::vector<Eigen::Vector3d> mesh_vertices = {};
    std::vector<Eigen::Vector3i> mesh_faces = {};
    for (auto & i_cell : cells_glb){
        // in each cell, dir, build mesh_msg
        for(int dir = 0; dir < 3; dir ++){
            PointMatrix & vertices = i_cell.second.ary_cell_vertices[dir];
            if(vertices.num_point == 0){
                continue;
            }
            //inside mesh_msg
            int start_vertex_i = mesh_vertices.size();
            for(int vertex_i = 0; vertex_i < vertices.num_point ; vertex_i ++){
                // transform to odom frame
                Vec3d vp = vertices.point.col(vertex_i);
                if(use_pose_slam){
                    vp = pose_slam_.block<3,3>(0, 0) * vp + pose_slam_.block<3,1>(0, 3);
                }else{
                    vp = pose_odom_.block<3,3>(0, 0) * vp + pose_odom_.block<3,1>(0, 3);
                }
                mesh_vertices.push_back(vp);
            }
            for(int vertex_i = 0; vertex_i < vertices.num_point; vertex_i++){
                int ix_cell = vertex_i % (n_row);
                int iy_cell = vertex_i / (n_row);
                Eigen::Vector3i tmp_face;
                if(ix_cell + 1 < n_row && iy_cell - 1 >= 0){
                    double variance_face_i = (vertices.variance(0, vertex_i) +
                                              vertices.variance(0, vertex_i + 1) +
                                              vertices.variance(0, vertex_i + 1 - n_row)) / 3.0;
                    if(1){//variance_face_i < variance_face_show
                        tmp_face[0] = start_vertex_i + vertex_i;
                        tmp_face[1] = start_vertex_i + vertex_i + 1;
                        tmp_face[2] = start_vertex_i + vertex_i + 1 - n_row;
                        mesh_faces.push_back(tmp_face);
                    }
                }
                if(ix_cell + 1 < n_row && iy_cell + 1 < n_row ){
                    double variance_face_i = (vertices.variance(0, vertex_i) +
                                              vertices.variance(0, vertex_i + 1) +
                                              vertices.variance(0, vertex_i + n_row)) / 3.0;
                    if(1){//variance_face_i < variance_face_show
                        tmp_face[0] = start_vertex_i + vertex_i;
                        tmp_face[1] = start_vertex_i + vertex_i + 1;
                        tmp_face[2] = start_vertex_i + vertex_i + n_row;
                        mesh_faces.push_back(tmp_face);
                    }
                }
            }
        }
    }

    TextureMapper::Mesh mesh;
    mesh.vertices = mesh_vertices;
    mesh.faces = mesh_faces;
    std::vector<TextureMapper::PosedImage> images;
    for (size_t i = 0; i < images_for_texture_.size(); ++i) {
        const Mat4d& T_sb = node_poses_[i];  //转换到子地图坐标系
        Mat4d T_B_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
        Mat4d T_cs = (T_sb * T_B_lC).inverse();
        TextureMapper::PosedImage img;
        img.R = T_cs.topLeftCorner(3,3);
        img.t = T_cs.topRightCorner(3,1);
        img.image = images_for_texture_[i];
        images.push_back(img);
    }

    TextureMapper texture_mapper_(mesh, images);
    texture_mapper_.generateTexturedMesh(
        Parameter::GetInstance().log_slam_mesh_dir+"/"+"mesh.obj", 
        Parameter::GetInstance().log_slam_mesh_dir +"/"+"mesh.jpg",
        Parameter::GetInstance().log_slam_mesh_dir +"/"+"mesh.mtl");

    /*
    Mat4d T_I_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
    Eigen::MatrixXd V = toEigenMatrix(mesh_vertices);
    Eigen::MatrixXi F = toEigenMatrix(mesh_faces);

    LOG(INFO);
    // 1. 找到边界顶点环
    Eigen::VectorXi b;
    igl::boundary_loop(F, b);  // b 是边界顶点索引的列表，例如 [0,1,2,3,...]

    LOG(INFO)<<b.size();
    // 2. 将边界顶点映射到单位圆上（生成初始的 2D UV 布局，避免重叠）
    Eigen::MatrixXd bc;  // bc 是边界顶点在 UV 空间的目标位置
    igl::map_vertices_to_circle(V, b, bc);  // 把边界顶点均匀放在一个圆周上
    LOG(INFO);

    // 3. 执行 LSCM 参数化
    Eigen::MatrixXd UV;
    igl::lscm(V, F, b, bc, UV);
    LOG(INFO);

    baked_texture_ = cv::Mat(2048, 2048, CV_8UC3, cv::Scalar(0, 0, 0));  // 黑色背景
    generate_texture_from_multiview(V, UV, baked_texture_);
    LOG(INFO);
    std::string texture_image_path = Parameter::GetInstance().log_slam_mesh_dir + "/" +"texture.jpg";  // 替换为你的图像路径！
    cv::imwrite(texture_image_path, baked_texture_);
    LOG(INFO);
    int tex_width  = baked_texture_.cols;
    int tex_height = baked_texture_.rows;

    // libigl 需要纹理数据为：高度 × 宽度 × 3，即 [rows, cols, channels]
    // 并且是 RGB（但 OpenCV 是 BGR，所以如果需要可以转换，但 libigl 内部会处理）
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(tex_height, tex_width),
                                                              G(tex_height, tex_width),
                                                              B(tex_height, tex_width);

    for (int y = 0; y < tex_height; ++y) {
        for (int x = 0; x < tex_width; ++x) {
            // OpenCV Mat 是 [y, x]，并且是 BGR 顺序
            cv::Vec3b color = baked_texture_.at<cv::Vec3b>(y, x);
            B(y, x) = color[2];  // Red   (BGR -> RGB)
            G(y, x) = color[1];  // Green
            R(y, x) = color[0];  // Blue
        }
    }

    // libigl 的 set_texture() 需要：RGB 数据排布为 [height, width, 3]
    // 即：先 height（行），再 width（列），每个点是 R G B
    // 所以我们将 R, G, B 拼成 [H x W x 3] 的矩阵形式（libigl 内部会组合）
    // 或者更简单：使用 cv::Mat 转换后直接传入 BGR，libigl 也能接受 BGR

    // 方法 1（推荐）：直接使用 OpenCV 的 BGR 格式，转成 libigl 的 [H x W x 3] = rows x cols x 3
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> texture;
    texture.resize(tex_height, tex_width * 3);

    for (int y = 0; y < tex_height; ++y) {
        for (int x = 0; x < tex_width; ++x) {
            texture(y, 3*x + 0) = B(y, x);  // R
            texture(y, 3*x + 1) = G(y, x);  // G
            texture(y, 3*x + 2) = R(y, x);  // B
        }
    }

    // -------------------------------
    // 3. 使用 libigl 的 GLFW Viewer 显示带纹理的 Mesh
    // -------------------------------

    igl::opengl::glfw::Viewer viewer;

    // 添加 Mesh (V, F)
    viewer.data().set_mesh(V, F);

    // 设置 UV 坐标
    viewer.data().set_uv(UV);

    // 设置纹理
    viewer.data().set_texture(texture, texture, texture);  // 或使用 texture

    // 启用纹理模式
    viewer.data().show_texture = true;

    // 可选：设置白色光照 / 材质，确保纹理可见
    viewer.data().set_colors(Eigen::MatrixXd::Ones(F.rows(), 3));

    // -------------------------------
    // 4. 启动交互式窗口
    // -------------------------------
    std::cout << "按 ESC 或关闭窗口退出。" << std::endl;
    viewer.launch(); */


    /* 
    Mat4d T_I_lC = SensorExtrinsicsManager::GetInstance().GetSensorToBody(SensorType::CAMERA_LEFT).matrix();
    std::vector<float> xatlas_vertices;
    std::vector<uint32_t> xatlas_indices;

    // 假设 V 是 Eigen::MatrixXd (Nv x 3), F 是 Eigen::MatrixXi (Nf x 3)
    for (int v = 0; v < mesh_vertices.size(); v++) {
        xatlas_vertices.push_back(mesh_vertices[v](0));
        xatlas_vertices.push_back(mesh_vertices[v](1));
        xatlas_vertices.push_back(mesh_vertices[v](2));
    }

    for (int f = 0; f < mesh_faces.size(); f++) {
        xatlas_indices.push_back(mesh_faces[f](0));
        xatlas_indices.push_back(mesh_faces[f](1));
        xatlas_indices.push_back(mesh_faces[f](2));
    }

    // 创建 xatlas::Atlas 并添加 Mesh
    xatlas::Atlas* atlas = xatlas::Create();
    if (!atlas) {
        LOG(ERROR) << "Failed to create xatlas";
        return;
    }

    xatlas::MeshDecl meshDecl;
    meshDecl.vertexCount = mesh_vertices.size();
    meshDecl.vertexPositionData = xatlas_vertices.data();
    meshDecl.vertexPositionStride = 3 * sizeof(float);
    meshDecl.indexCount = mesh_faces.size() * 3;
    meshDecl.indexData = xatlas_indices.data();
    meshDecl.indexFormat = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError::Enum meshHandle = xatlas::AddMesh(atlas, meshDecl, 1);
    if (meshHandle != xatlas::AddMeshError::Success) {
        LOG(ERROR) << "Failed to add mesh to atlas";
        xatlas::Destroy(atlas);
        return;
    }
    
    xatlas::Generate(atlas);

    if(atlas->meshCount==0){
        LOG(ERROR) << "No mesh generated by xatlas";
        xatlas::Destroy(atlas);
        return;
    }
    LOG(INFO)<<"Meshes generated for "<<name << ": "<<atlas->meshCount;

    const xatlas::Mesh& mesh = atlas->meshes[0];
    LOG(INFO)<<"Original mesh has vertices: "<<mesh_vertices.size()<<", Xatlas has vertices: "<<mesh.vertexCount;
    LOG(INFO)<<"Original mesh has faces: "<<mesh_faces.size()<<", Xatlas has faces: "<<mesh.indexCount/3;

    // 烘焙纹理（Baking Textures from Images）
    baked_texture_ = cv::Mat(atlas->height, atlas->width, CV_8UC3, cv::Scalar(0, 0, 0));  // 黑色背景 */
    /* pcl::PointCloud<pcl::PointXYZRGB> cloud_temp;
    cloud_temp.points.resize(mesh.vertexCount);
    for (int i = 0; i < mesh.vertexCount; i++) {  // 遍历每个顶点
        const xatlas::Vertex& v = mesh.vertexArray[i];

        // 顶点在原始 Mesh 中的索引（xatlas 可能重排顶点，但通常按顺序）
        Eigen::Vector3d point_3d = mesh_vertices[v.xref];  // 世界坐标
        cloud_temp.points[i].x = point_3d.x();
        cloud_temp.points[i].y = point_3d.y();
        cloud_temp.points[i].z = point_3d.z();

        double u = v.uv[0];
        double v_uv = v.uv[1];

        // LOG(INFO)<<u<<" "<<v_uv;
        // 转换为纹理图上的像素坐标
        int px = static_cast<int>(u);
        int py = static_cast<int>(v_uv);

        // Clamp to texture size
        px = std::clamp(px, 0, int(atlas->width) - 1);
        py = std::clamp(py, 0, int(atlas->height) - 1);

        cv::Vec3b color(0, 0, 0);
        for (size_t img_id = 0; img_id < images_for_texture_.size(); ++img_id) {
            const Mat4d& T_si = node_poses_[img_id];  //转换到子地图坐标系
            Mat4d T_cs = (T_si * T_I_lC).inverse();

            Eigen::Vector3d point_cam = T_cs.block<3,3>(0,0) * point_3d + T_cs.block<3,1>(0,3);

            if (point_cam.z() <= 0) continue; 

            Vec2d pxpy;
            camera_left_->spaceToPlane(point_cam, pxpy);
            const cv::Mat& img = images_for_texture_[img_id];
            if (pxpy.x() < 0 || pxpy.x() >= img.cols || pxpy.y() < 0 || pxpy.y() >= img.rows)
                continue;
            color = img.at<cv::Vec3b>(pxpy.y(), pxpy.x());

            break;//TODO
        }
        cloud_temp.points[i].r = color[2];
        cloud_temp.points[i].g = color[1];
        cloud_temp.points[i].b = color[0];
        baked_texture_.at<cv::Vec3b>(py, px) = color;
    } */

    /* std::vector<size_t> idx(mesh.indexCount/3);
    iota(idx.begin(), idx.end(),  0);
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t f) {
        uint32_t i0 = xatlas_indices[f * 3 + 0];
        uint32_t i1 = xatlas_indices[f * 3 + 1];
        uint32_t i2 = xatlas_indices[f * 3 + 2];
        const xatlas::Vertex& v0 = mesh.vertexArray[i0];
        const xatlas::Vertex& v1 = mesh.vertexArray[i1];
        const xatlas::Vertex& v2 = mesh.vertexArray[i2];

        // 1. 三个 UV 坐标 [0,1]
        double u0 = v0.uv[0], v0_uv = v0.uv[1];
        double u1 = v1.uv[0], v1_uv = v1.uv[1];
        double u2 = v2.uv[0], v2_uv = v2.uv[1];

        // 2. 三个 3D 顶点
        int idx0 = v0.xref, idx1 = v1.xref, idx2 = v2.xref;
        Eigen::Vector3d p0 = mesh_vertices[idx0];
        Eigen::Vector3d p1 = mesh_vertices[idx1];
        Eigen::Vector3d p2 = mesh_vertices[idx2];

        // 3. 三角形在纹理图上的像素范围（Bounding Box）
        int min_u = std::min({u0, u1, u2});
        int max_u = std::max({u0, u1, u2});
        int min_v = std::min({v0_uv, v1_uv, v2_uv});
        int max_v = std::max({v0_uv, v1_uv, v2_uv});

        // LOG(INFO)<<min_u<<" "<<max_u<<" "<<min_v<<" "<<max_v;

        min_u = std::max(0, min_u);
        max_u = std::min(static_cast<int>(atlas->width) - 1, max_u);
        min_v = std::max(0, static_cast<int>(min_v));
        max_v = std::min(static_cast<int>(atlas->height) - 1, max_v);

        LOG(INFO)<<max_u-min_u<<" "<<max_v-min_v;

        // 遍历该三角形 UV 区域内的所有像素
        for (int py = min_v; py <= max_v; ++py) {
            for (int px = min_u; px <= max_u; ++px) {
                double u = static_cast<double>(px);
                double v = static_cast<double>(py);

                // 判断 (u,v) 是否在三角形 (u0,v0), (u1,v1), (u2,v2) 内
                if (pointInTriangle(u, v, u0, v0_uv, u1, v1_uv, u2, v2_uv)) {
                    // 3. 插值得到 3D 点（或使用重心）
                    Eigen::Vector3d point_3d = computeBarycentricInterpolatedPoint(
                        u, v, u0, v0_uv, u1, v1_uv, u2, v2_uv, p0, p1, p2);

                    // 4. 投影到图像，采样颜色
                    cv::Vec3b color(0, 0, 0);
                    for (size_t img_id = 0; img_id < images_for_texture_.size(); ++img_id) {
                        const Mat4d& T_si = node_poses_[img_id];
                        Mat4d T_cs = (T_si * T_I_lC).inverse();

                        Eigen::Vector3d p_cam = T_cs.block<3,3>(0,0) * point_3d + T_cs.block<3,1>(0,3);
                        if (p_cam.z() <= 0) continue;

                        Vec2d pxpy;
                        camera_left_->spaceToPlane(p_cam, pxpy);
                        const cv::Mat& img = images_for_texture_[img_id];
                        if (pxpy.x() < 0 || pxpy.x() >= img.cols || pxpy.y() < 0 || pxpy.y() >= img.rows)
                            continue;

                        color = img.at<cv::Vec3b>(pxpy.y(), pxpy.x());
                        break;
                    }

                    // 5. 填充到纹理图
                    baked_texture_.at<cv::Vec3b>(py, px) = color;
                }
            }
        }
    });
        
    std::string mesh_dir = Parameter::GetInstance().log_slam_mesh_dir;
    std::string obj_filename = mesh_dir + "/" + "textured_mesh.obj";
    std::string texture_filename = mesh_dir + "/" + "textured_mesh.png";
    SaveXatlasMeshToObj(obj_filename.c_str(), mesh, mesh_vertices, atlas->width, atlas->height);
    cv::imwrite(texture_filename, baked_texture_); */
    // cloud_temp.height = 1;
    // cloud_temp.width = mesh.vertexCount;
    // pcl::io::savePCDFileASCII(mesh_dir + "/" + "colored_vertices" + ".pcd", cloud_temp);
    // visualizeTexturedMeshWithLibIGL(xatlas_vertices, xatlas_indices, UV, baked_texture_, mesh_faces);
    // exportTexturedMeshToObj(Parameter::GetInstance().log_slam_mesh_dir, 
    //         name + "_textured_mesh", mesh_vertices, mesh_faces, UV, baked_texture_);
}



// 导出带 UV 的 Mesh 为 OBJ + MTL + PNG 纹理
void Submap::exportTexturedMeshToObj(
    const std::string& mesh_dir,
    const std::string& base_filename,          // 如 "mesh"，会生成 mesh.obj, mesh.mtl, texture.png
    const std::vector<Vec3d>& V,     // 顶点，Nx3
    const std::vector<Vec3i>& F,     // 面片（三角形），Mx3，顶点索引从0开始
    const std::vector<Vec2d>& UV,                 // 每个顶点的 UV，与 V 一一对应
    const cv::Mat& texture_img                 // 纹理图像，CV_8UC3，BGR 格式
) {
    // 1. 保存纹理图为 PNG
    std::string texture_filename = mesh_dir + "/" + base_filename + ".png";
    cv::imwrite(texture_filename, texture_img);

    // 2. 保存 OBJ 文件
    std::string obj_filename = mesh_dir + "/" + base_filename + ".obj";
    std::ofstream obj_file(obj_filename);
    if (!obj_file.is_open()) {
        LOG(ERROR) << "Can't create OBJ file: " << obj_filename;
        return;
    }

    // 写入 OBJ 头，引用 MTL
    obj_file << "# Exported textured mesh with UVs\n";
    obj_file << "mtllib " << base_filename << ".mtl\n";

    // 写入顶点 (v)
    for (const auto& v : V) {
        obj_file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }

    // 写入 UV (vt)
    for (int i = 0; i < UV.size(); i++) {
        Vec2d uv = UV[i];
        obj_file << "vt " << uv(0) << " " << uv(1) << "\n";
    }

    // 写入面片 (f 1/1 2/2 3/3)
    obj_file << "\n";  // 空行分隔
    for (const auto& face : F) {
        // OBJ 面索引从 1 开始！
        int v0 = face[0] + 1;
        int v1 = face[1] + 1;
        int v2 = face[2] + 1;

        // 格式: f v1/vt1 v2/vt2 v3/vt3
        obj_file << "f " << v0 << "/" << v0 << " " 
                        << v1 << "/" << v1 << " " 
                        << v2 << "/" << v2 << "\n";
    }

    obj_file.close();
    LOG(INFO) << "OBJ saved as: " << obj_filename;

    // 3. 保存 MTL 文件
    std::string mtl_filename = mesh_dir + "/" + base_filename + ".mtl";
    std::ofstream mtl_file(mtl_filename);
    if (!mtl_file.is_open()) {
        LOG(ERROR) << "Can't create MTL file: " << mtl_filename;
        return;
    }

    mtl_file << "# Material file for textured mesh\n";
    mtl_file << "newmtl Material\n";
    mtl_file << "Ka 1.0 1.0 1.0      # 环境光\n";
    mtl_file << "Kd 1.0 1.0 1.0      # 漫反射\n";
    mtl_file << "Ks 0.0 0.0 0.0      # 镜面反射\n";
    mtl_file << "d 1.0               # 不透明度\n";
    mtl_file << "illum 1             # 光照模型\n";

    // 引用纹理贴图
    mtl_file << "map_Kd " << base_filename + ".png" << "\n";  // 纹理贴图

    mtl_file.close();
    LOG(INFO) << "MTL 文件已保存为: " << mtl_filename;

    LOG(INFO) << "✅ 导出完成！你可以在 MeshLab 中打开 " << obj_filename << " 查看带纹理的 Mesh";
    LOG(INFO) << "确保 mesh.obj、mesh.mtl、mesh.png 在同一目录下";
}


// submap.cpp

void Submap::aggregateCellFeatures(std::vector<Vec3d>& aggregated_keypoints) {
    std::vector<Vec3d> all_raw_edges;
    
    // 1. 收集所有 Cell 的边缘点
    for (auto& pair : cells_glb) {
        Cell* cell = &pair.second;
        std::vector<FeaturePoint> edge_pts, planar_pts;
        cell->extractLinK3DFeatures(edge_pts, planar_pts); // 调用之前为 Cell 补充的函数
        for (const auto& fp : edge_pts) {
            all_raw_edges.push_back(fp.pos);
        }
    }

    if (all_raw_edges.empty()) return;

    // 2. 在 XOY 平面聚合 (类似非极大值抑制)
    // 使用简单的距离阈值，确保 2D 空间内关键点不至于太密集
    const double aggregate_dist_sq = 0.4 * 0.4; // 40cm 聚合阈值
    for (const auto& pt : all_raw_edges) {
        bool too_close = false;
        for (const auto& agg_pt : aggregated_keypoints) {
            // 仅计算 XOY 平面距离
            double dx = pt.x() - agg_pt.x();
            double dy = pt.y() - agg_pt.y();
            if (dx*dx + dy*dy < aggregate_dist_sq) {
                too_close = true;
                break;
            }
        }
        if (!too_close) {
            aggregated_keypoints.push_back(pt);
        }
    }
}

void Submap::computeDescriptors(const std::vector<Vec3d>& keypoints, 
                               std::vector<LinK3DDescriptor>& out_descriptors) {
    const int sector_num = 180;
    const double search_radius = 15.0; // LinK3D 典型搜索半径

    for (size_t i = 0; i < keypoints.size(); ++i) {
        LinK3DDescriptor desc;
        desc.position = keypoints[i];
        desc.id = i;

        // 遍历其他所有关键点，寻找每个扇区的最近邻
        for (size_t j = 0; j < keypoints.size(); ++j) {
            if (i == j) continue;

            Vec3d diff = keypoints[j] - keypoints[i];
            double dist = diff.norm();
            if (dist > search_radius) continue;

            // 计算在 XOY 平面上的角度 [0, 2pi]
            double angle = atan2(diff.y(), diff.x());
            if (angle < 0) angle += 2.0 * M_PI;

            // 映射到 180 个扇区
            int sector_idx = static_cast<int>(angle / (2.0 * M_PI) * sector_num);
            sector_idx = std::max(0, std::min(sector_num - 1, sector_idx));

            // LinK3D 逻辑：存储该扇区内最近的点的距离
            float current_dist = static_cast<float>(dist);
            if (desc.descriptor[sector_idx] == 0 || current_dist < desc.descriptor[sector_idx]) {
                desc.descriptor[sector_idx] = current_dist;
            }
        }
        out_descriptors.push_back(desc);
    }
}

void Submap::generateLinK3DFeatures(std::vector<LinK3DDescriptor>& out_descriptors) {
    std::vector<Vec3d> keypoints;
    
    // 1. 提取并聚合子地图坐标系下的关键点
    aggregateCellFeatures(keypoints);
    
    
    // 2. 计算子地图局部坐标系下的描述子
    computeDescriptors(keypoints, out_descriptors);
    
    // 注意：如果是为了给 BoW3D 做回环检测，
    // 通常直接使用子地图局部坐标系下的 descriptors 即可，
    // 因为 LinK3D 具有旋转不变性（通过循环移位实现匹配）。
}
