#pragma once
#ifndef MAP_CELL_H_
#define MAP_CELL_H_
//std
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cmath>
#include <unordered_map>
#include <queue>
#include <ctime>
#include <chrono>
//eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>
//ros
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64.h"
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include "geometry_msgs/Quaternion.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include "tf/transform_datatypes.h"
#include "tf/LinearMath/Matrix3x3.h"
#include <tf/transform_broadcaster.h>
//pcl
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
//ceres
#include <ceres/ceres.h>
// meshing_tsdf
#include <mesh_msgs_conversions/conversions.h>
#include <mesh_msgs/MeshTexture.h>
//ours
#include "utility/pointMatrix.h"
#include "utility/tool.h"
#include "utility/parameter.h"

#include "opencv2/core/core.hpp"
#include "omp.h"
#include "vertex_color_predictor.h"

#include "LKtoORBConverter.h"


typedef  Eigen::Matrix <double, 1, Eigen::Dynamic> Matrix1xd;
enum Direction{
    X = 0, Y, Z, Unknown
};
enum Color{
    R = 0, G, B
};
enum MapState{
    INIT_ED, KD_TREE_ED, GP_ED
};

struct FeaturePoint {
    Vec3d pos;
    double curvature;
    int index; // 在 ary_cell_vertices 中的索引
    Direction dir;
};

class Region{
    //define a voxel region, which is cells' borders
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double x_min{0}, y_min{0}, z_min{0}, x_max{0}, y_max{0}, z_max{0};
    Region(){}
    Region(double x_min_, double y_min_, double z_min_, double x_max_, double y_max_, double z_max_)
    : x_min(x_min_), y_min(y_min_), x_max(x_max_), y_max(y_max_), z_max(z_max_), z_min(z_min_){
    }
    double getDirMin(Direction dir){
        if(dir == X){
            return x_min;
        }
        else if (dir == Y){
            return y_min;
        }
        else if (dir == Z){
            return z_min;
        }
    }
    double getDirMax(Direction dir){
        if(dir == X){
            return x_max;
        }
        else if (dir == Y){
            return y_max;
        }
        else if (dir == Z){
            return z_max;
        }
    }
};

struct VisualPoint3D{
    uint64_t id;
    Eigen::Vector3d xyz;
    BRIEF::bitset descriptor;
};

class Cell{
    // define a basic data structure of the map, a voxel cell containing points, several layers of mesh, and so on.
private:
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //state
    bool empty = true, not_surface = true;
    int time_stamp = 0;
    Region region;
    double hash_position = 0;
    Point center {0,0,0};
    int revisited = 0;//if the sensor scan the same region after a certain time, revisited ++
    Point viewed_location {0, 0, 0};//sensor's location when this cell is observed
    int viewed_dir_count = 0;
    double average_viewed_distance = 0;//distance from sensor
    //data
    PointMatrix cell_raw_points;
    PointMatrix ary_cell_vertices[3];//direction X, Y, Z

    
    cv::Mat ref_patch[3]; // 一个Cell最多只存一个顶点（该顶点周围的像素变化足够明显，该顶点处的法向量方差小）处的patch
    int patch_idx[3]; // patch 索引

    size_t updated_times[3];//if object is only observed in several frames, it may be dynamic object and will not shown in the final map
    //slamesh can also build tsdf map, it is optional
    
    // 成熟的3D视觉特征点，可能存在不同ID但是同一个点，需要做归并
    std::map<int64_t, VisualPoint3D> visual_points_;
    
    GPVertexColorPredictor vertex_color_predictor_;

    Vec3d getVertice(Direction dir, int i,  int j) const;
    Vec4d getVerticeAndVariance(Direction dir, int i,  int j) const;

    Vec3d getVertice(Direction dir, int k) const;
    Vec4d getVerticeAndVariance(Direction dir, int k) const;

    // TODO, Remove redundunt codes, ray and cell should be in the same reference frame.
    std::optional<Vec4d> rayCellIntersectionX(const Ray& ray) const;
    std::optional<Vec4d> rayCellIntersectionY(const Ray& ray) const;
    std::optional<Vec4d> rayCellIntersectionZ(const Ray& ray) const;

    void gaussianProcess(enum Direction gp_direction);
    void gaussianProcessRGB(enum Direction gp_direction);
    void gaussianProcessRGB(enum Direction gp_direction, enum Color gp_color);
    void ColorGradient(enum Direction gp_direction);
    void reconstructSurfaces(bool glb_cell_not_surface);
    void updateVertices(Cell & cell_new, enum Direction update_direction);
    void updateVerticeColor(Direction dir, int idx, Vec3d color, double distance, Vec3d view_dir);
    void clearPointsQuick(){
        //may be memorial comsuming
        cell_raw_points.clear_quick();
        for(auto & i_vertices : ary_cell_vertices){
            i_vertices.clear_quick();
        }
        empty = true;
    }
    void updateViewedLocation(const Transf & transf_viewed);
    void initCell(PointMatrix & raw_points, int time_stamp_,
                  double posi_, Region region_,
                  bool reconstruct, bool not_surface_or_unknown_map_glb){
        //try to use it to reduce copy of cells in dividePointsIntoCells()
        cell_raw_points = raw_points;
        region = region_;
        hash_position = posi_;
        time_stamp = time_stamp_;
        not_surface = true;
        revisited = 0;
        viewed_location << 0,0,0;
        viewed_dir_count = 0;
        average_viewed_distance = 0;//distance from sensor
        memset(updated_times, 0, sizeof(updated_times));
        center << (region.x_min + region.x_max) / 2.0, (region.y_min + region.y_max) / 2.0,(region.z_min + region.z_max) / 2.0;
        if(reconstruct){
            reconstructSurfaces(not_surface_or_unknown_map_glb);// if no map_glb info, this value is true
        }
        empty = false;

        for(int i = 0; i < 3; i++){
            // patch_dir[i] = -1;
            patch_idx[i] = -1;
        }
    };


    Cell(PointMatrix & raw_points, int time_stamp_,
         double posi_, Region region_,
         bool reconstruct, bool not_surface_or_unknown_map_glb) :
            cell_raw_points(raw_points), region(region_), hash_position(posi_), time_stamp(time_stamp_){
        //used in dividePointsIntoCell, and dividePointsIntoCellInitMap.
        center << (region.x_min+region.x_max)/2.0, (region.y_min+region.y_max)/2.0,(region.z_min+region.z_max)/2.0;
        if(reconstruct){
            reconstructSurfaces(not_surface_or_unknown_map_glb);// if no map_glb info, this value is true
        }
        empty = false;
        for(int i = 0; i < 3; i++){
            // patch_dir[i] = -1;
            patch_idx[i] = -1;
        }
    };
    
    Cell(PointMatrix & raw_points, PointMatrix ary_cell_vertices_[], int time_stamp_,
         double posi_, Region region_) :
            cell_raw_points(raw_points), region(region_), hash_position(posi_), time_stamp(time_stamp_){
        // used in map_update add new cell, just copy all cell_vertices , not reconstruct again
        center << (region.x_min+region.x_max)/2.0, (region.y_min+region.y_max)/2.0,(region.z_min+region.z_max)/2.0;
        for(int i = 0; i < 3; i++){
            ary_cell_vertices[i] = ary_cell_vertices_[i];
        }
        empty = false;
        for(int i = 0; i < 3; i++){
            // patch_dir[i] = -1;
            patch_idx[i] = -1;
        }
    };
    Cell(){
        for(int i = 0; i < 3; i++){
            // patch_dir[i] = -1;
            patch_idx[i] = -1;
        }
    };

    ///////////////////////////////////////////////////

    

    // 提取当前 Cell 内所有 GP 表面上的边缘点和角点
    void extractLinK3DFeatures(std::vector<FeaturePoint>& edge_points, 
                            std::vector<FeaturePoint>& planar_points);

    // 内部计算单个顶点的几何特征（基于邻域网格）
    void computeVertexCurvature(Direction dir, int r, int c, double& curvature, double& linearity);
};

void VoxelFilter2D(int num_input, int & num_output, double grid,
                   Matrix1xd & f, Matrix1xd & train_x, Matrix1xd & train_y, int num_test);
#endif