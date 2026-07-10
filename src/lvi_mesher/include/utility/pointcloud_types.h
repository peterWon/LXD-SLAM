#ifndef POINTCLOUD_TYPES_H
#define POINTCLOUD_TYPES_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
);
using VeloPointXYZIRT = VelodynePointXYZIRT;

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    // uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint32_t, range, range) //(uint16_t, noise, noise) 
)

struct RsPointXYZIRT
{
  PCL_ADD_POINT4D;
  uint8_t intensity;
  uint16_t ring = 0;
  double timestamp = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(RsPointXYZIRT, 
  (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)
  (uint16_t, ring, ring)(double, timestamp, timestamp))

struct HesaiPointXYZIRT
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    double timestamp = 0;
    uint16_t ring = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(HesaiPointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp)(uint16_t, ring, ring))

struct PointXYZT
{
    PCL_ADD_POINT4D;
    double t = 0.;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZT,(float, x, x)(float, y, y)(float, z, z)(double, t, t))

struct PointXYZRGBT
{
    float x;
    float y;
    float z;
    float r;
    float g;
    float b;
    double t = 0.;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRGBT,
    (float, x, x)(float, y, y)(float, z, z)(float, r, r)(float, g, g)(float, b, b)(double, t, t))
#endif