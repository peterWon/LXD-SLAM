#include "Eigen/Geometry"
#include "Eigen/Core"
#include "glog/logging.h"

int main(){
    /* Eigen::Matrix4d T_Body2CamLeft, T_Body2CamRight, T_CamRight2CamLeft;
    T_Body2CamRight << -0.01916508, -0.01496218,  0.99970437,  0.00519443,
           0.99974371,  0.01176483,  0.01934191,  0.1347802,
          -0.01205075,  0.99981884,  0.01473287,  0.01465067,
           0.00000000,  0.00000000,  0.00000000,  1.00000000;
    T_Body2CamLeft << 0.02183084, -0.01312053,  0.99967558,  0.00552943,
           0.99975965,  0.00230088, -0.02180248, -0.12431302,
          -0.00201407,  0.99991127,  0.01316761,  0.01614686, 
           0.00000000,  0.00000000,  0.00000000,  1.00000000;
    T_Body2CamRight = T_Body2CamRight.inverse();//actually camera in IMU
    T_Body2CamLeft = T_Body2CamLeft.inverse();
    T_CamRight2CamLeft = T_Body2CamLeft.inverse() * T_Body2CamRight;

    Eigen::Matrix4d T_body2hlidar, T_body2vlidar, T_hlidar2camleft, T_vlidar2hlidar;
    T_body2hlidar << 1.0,  0.0,  0.0, -0.050,
           0.0,  1.0,  0.0,  0.000,
           0.0,  0.0,  1.0,  0.055,
           0.0,  0.0,  0.0,  1.000;
    T_body2vlidar << -1.0,  0.0,  0.0, -0.550,
           0.0,  0.0,  1.0,  0.030,
           0.0,  1.0,  0.0,  0.050,
           0.0,  0.0,  0.0,  1.000;
    T_body2hlidar = T_body2hlidar.inverse();
    T_body2vlidar = T_body2vlidar.inverse();
    T_hlidar2camleft = T_Body2CamLeft.inverse() * T_body2hlidar;
    T_vlidar2hlidar = T_body2hlidar.inverse() * T_body2vlidar; 
    
    Eigen::Matrix4d T_camleft2body = T_Body2CamLeft.inverse();
    LOG(INFO)<<"T_camleft2body: "<<T_camleft2body;
    LOG(INFO)<<"T_CamRight2CamLeft: "<<T_CamRight2CamLeft;
    LOG(INFO)<<"T_hlidar2camleft: "<<T_hlidar2camleft;
    LOG(INFO)<<"T_vlidar2hlidar: "<<T_vlidar2hlidar; */

    /* Eigen::Matrix3d R_l2i;
    R_l2i << 0.999958,  0.00830745,  0.00371543, 
           -0.00831234,    0.999964,  0.00130257, 
           -0.00370449,  -0.0013334,    0.999992;
    Eigen::Quaterniond q(R_l2i);
    LOG(INFO)<<q.toRotationMatrix().eulerAngles(2,1,0); */
    Eigen::Matrix4d T_IMU_lidar;
    T_IMU_lidar<<1, 0, 0, 0.04165,
                   0, 1, 0, 0.02326,
                   0, 0, 1, -0.0284,
                     0,          0,          0,           1;

    Eigen::Matrix4d T_CamLeft_Lidar;
    T_CamLeft_Lidar<<0.00516621,-0.999958,-0.00753827, -1.24355e-05,
         -0.00114374,0.00753245,-0.999971, 0.0640151,
          0.999986,0.00517468,-0.00110478, 0.00999183,
                        0,          0,          0,           1;
    Eigen::Isometry3d T_l2i(T_IMU_lidar);                
    Eigen::Isometry3d T_l2c(T_CamLeft_Lidar);
    Eigen::Isometry3d T_c2l = T_l2c.inverse();
    
    LOG(INFO)<<"T_cam2imu: "<< (T_l2i * T_c2l) .matrix();       

//     LOG(INFO)<<"T_lidar2cam"<< T_c2l.inverse().matrix();       
    return 0;
}