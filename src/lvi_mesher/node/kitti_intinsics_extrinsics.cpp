#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <Eigen/Dense>
#include <glog/logging.h>

class KITTIOdometryCalib {
private:
    Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();  // Velodyne -> Camera 0 (外参)
    Eigen::Matrix<float, 3, 4> P0;  // 左相机投影矩阵
    Eigen::Matrix<float, 3, 4> P1;  // 右相机投影矩阵
    Eigen::Matrix3f K0;             // 左相机内参矩阵 (新增)
    Eigen::Matrix3f K1;             // 右相机内参矩阵 (新增)
    float baseline;                 // 基线（米）

    // 从投影矩阵提取内参（私有辅助逻辑）
    void extractIntrinsicFromProjection() {
        // 校正后的投影矩阵前3x3即为内参矩阵K
        K0 = P0.block<3, 3>(0, 0);
        K1 = P1.block<3, 3>(0, 0);
    }

public:
    // 构造函数：解析calib.txt，同时提取内参+外参
    KITTIOdometryCalib(const std::string& calib_file) {
        std::ifstream file(calib_file);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open calib file: " + calib_file);
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            std::getline(iss, key, ':');

            if (key == "Tr") {
                // 读取Velodyne->Camera0外参（4x4，实际存储前3行）
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        iss >> Tr(i, j);
                    }
                }
                // 补全4x4矩阵的最后一行（0,0,0,1）
                Tr(3, 0) = Tr(3, 1) = Tr(3, 2) = 0.0f;
                Tr(3, 3) = 1.0f;
            } else if (key == "P0") {
                // 读取左相机投影矩阵（3x4）
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        iss >> P0(i, j);
                    }
                }
            } else if (key == "P1") {
                // 读取右相机投影矩阵（3x4）
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        iss >> P1(i, j);
                    }
                }
            }
        }
        file.close();

        // 计算基线 + 提取内参（核心融合逻辑）
        float fx = P0(0, 0);
        baseline = -P1(0, 3) / fx;
        extractIntrinsicFromProjection(); // 从P0/P1提取K0/K1
    }

    // 获取相机内参矩阵（左/右）
    Eigen::Matrix3f getIntrinsic(int camera_id = 0) const {
        if (camera_id == 0) {
            return K0;
        } else if (camera_id == 1) {
            return K1;
        } else {
            throw std::runtime_error("Unsupported camera_id: only 0 (left) or 1 (right)");
        }
    }

    // 获取投影矩阵（左/右）
    Eigen::Matrix<float, 3, 4> getProjectionMatrix(int camera_id = 0) const {
        if (camera_id == 0) {
            return P0;
        } else if (camera_id == 1) {
            return P1;
        } else {
            throw std::runtime_error("Unsupported camera_id: only 0 (left) or 1 (right)");
        }
    }

    // 获取相机到激光雷达的变换矩阵（原有逻辑）
    Eigen::Matrix4f getCameraToLidar(int camera_id = 0) const {
        Eigen::Matrix4f T_cam0_to_velo = Tr.inverse();

        if (camera_id == 0) {
            return T_cam0_to_velo;
        } else if (camera_id == 1) {
            // 右相机到左相机的变换（仅x轴平移基线）
            Eigen::Matrix4f T_cam1_to_cam0 = Eigen::Matrix4f::Identity();
            T_cam1_to_cam0(0, 3) = -baseline;

            // 右相机 -> 左相机 -> 激光雷达
            return T_cam0_to_velo * T_cam1_to_cam0;
        } else {
            throw std::runtime_error("Unsupported camera_id: only 0 (left) or 1 (right)");
        }
    }

    // 获取激光雷达到相机的变换矩阵（补充接口，方便使用）
    Eigen::Matrix4f getLidarToCamera(int camera_id = 0) const {
        if (camera_id == 0) {
            return Tr;
        } else if (camera_id == 1) {
            Eigen::Matrix4f T_cam1_to_cam0 = Eigen::Matrix4f::Identity();
            T_cam1_to_cam0(0, 3) = -baseline;
            // 激光雷达 -> 左相机 -> 右相机
            return T_cam1_to_cam0.inverse() * Tr;
        } else {
            throw std::runtime_error("Unsupported camera_id: only 0 (left) or 1 (right)");
        }
    }

    // 获取基线长度
    float getBaseline() const {
        return baseline;
    }

    // 生成指定格式的YAML文件（左/右相机）
    void writeCameraYAML(const std::string& yaml_path, const std::string& seq, int camera_id = 0) const {
        // 写入左相机YAML
        int img_width = 1241; 
        int img_height = 376;
        if(seq == "00" || seq == "01" || seq == "02" ){
            img_width = 1241;
            img_height = 376;
        }else if(seq =="03"){
            img_width = 1242;
            img_height = 375;
        }else if(seq =="04" || seq =="05" || seq =="06" || seq =="07" || seq =="08" || seq =="09" || seq =="10"){
            img_width = 1226;
            img_height = 370;
        }
        
        std::ofstream yaml(yaml_path);
        if (!yaml.is_open()) {
            throw std::runtime_error("Failed to create camera YAML file: " + yaml_path);
        }
        yaml.precision(12);
        if(camera_id==0){
            yaml << "%YAML:1.0\n"
                  << "model_type:   PINHOLE\n"
                  << "camera_name:  left\n"
                  << "image_width:  " << img_width << "\n"
                  << "image_height: " << img_height << "\n\n"
                  << "mask: \"\"\n"
                  << "distortion_model: radial-tangential\n"
                  << "distortion_parameters:\n"
                  << "   k1: 0\n"
                  << "   k2: 0\n"
                  << "   p1: 0\n"
                  << "   p2: 0\n"
                  << "projection_parameters:\n"
                  << "   fx: " << K0(0,0) << "\n"
                  << "   fy: " << K0(1,1) << "\n"
                  << "   cx: " << K0(0,2) << "\n"
                  << "   cy: " << K0(1,2) << "\n";
        }else if(camera_id == 1){
            yaml << "%YAML:1.0\n"
                   << "model_type:   PINHOLE\n"
                   << "camera_name:  right\n"
                   << "image_width:  " << img_width << "\n"
                   << "image_height: " << img_height << "\n\n"
                   << "mask: \"\"\n"
                   << "distortion_model: radial-tangential\n"
                   << "distortion_parameters:\n"
                   << "   k1: 0\n"
                   << "   k2: 0\n"
                   << "   p1: 0\n"
                   << "   p2: 0\n"
                   << "projection_parameters:\n"
                   << "   fx: " << K1(0,0) << "\n"
                   << "   fy: " << K1(1,1) << "\n"
                   << "   cx: " << K1(0,2) << "\n"
                   << "   cy: " << K1(1,2) << "\n";
        } else {
            throw std::runtime_error("Unsupported camera_id: only 0 (left) or 1 (right)");
        }
        yaml.close();
        std::cout << "Camera YAML files generated successfully!" << std::endl;
    }
};

int main(int argc, char**argv){
    if(argc != 2){
        std::cout<<"Error arguments, please provide sequence number.";
        return -1;
    }
    std::string kitti_odometry_dir = "/media/wz/2C96A0A60155E8F8/Dataset/KITTI-Odometry/";
    std::string camera_intrinsic_dir = "/home/wz/codes/ros_ws/livo/lvi_mesher/src/lvi_mesher/config/cameras/kitti/";
    std::string seq = std::string(argv[1]);
    std::string calib_filepath = kitti_odometry_dir + "/data_odometry_velodyne/dataset/sequences/" + seq + "/calib.txt";
    KITTIOdometryCalib calib = KITTIOdometryCalib(calib_filepath);
    LOG(INFO)<<"Left camera in Lidar: "<<calib.getCameraToLidar(0);
    LOG(INFO)<<"Right camera in Lidar: "<<calib.getCameraToLidar(1);

    std::string left_intrinsic_filepath = camera_intrinsic_dir + seq + "_left.yaml";
    std::string right_intrinsic_filepath = camera_intrinsic_dir + seq + "_right.yaml";
    std::string extrinsics_filepath = camera_intrinsic_dir + "extrinsics.yaml";
    
    calib.writeCameraYAML(left_intrinsic_filepath, seq, 0);
    calib.writeCameraYAML(right_intrinsic_filepath, seq, 1);

    std::ofstream ofs(extrinsics_filepath, std::ios::app);
    if(ofs.is_open()){
        ofs << seq << ":\n";
        ofs << "T_Body_CamLeft: "<<calib.getCameraToLidar(0)<<"\n";
        ofs << "T_Body_CamRight: "<<calib.getCameraToLidar(1)<<"\n";
    }
    ofs.close();

    return 0;
}
