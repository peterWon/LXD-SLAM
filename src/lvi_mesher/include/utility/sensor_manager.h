#ifndef __SENSOR_MANAGER_H__
#define __SENSOR_MANAGER_H__
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <Eigen/Geometry>

// 传感器类型枚举
enum SensorType {
    IMU,
    WHEEL,
    CAMERA_LEFT,
    CAMERA_RIGHT,
    LIDAR1,
    LIDAR2,
    GNSS
};


class SensorExtrinsicsManager {
public:
    // 单例获取接口
    static SensorExtrinsicsManager& GetInstance() {
        static SensorExtrinsicsManager instance;
        return instance;
    }

    // 禁止拷贝和赋值
    SensorExtrinsicsManager(const SensorExtrinsicsManager&) = delete;
    SensorExtrinsicsManager& operator=(const SensorExtrinsicsManager&) = delete;

    // 设置传感器到Body系的直接外参（刚体变换：旋转+平移）
    // 语义变更：不再是到父节点，而是直接到Body系
    void SetSensorToParentExtrinsic(SensorType sensor, const Eigen::Isometry3d& extrinsic) {
        if (sensor_to_body_map_.find(sensor) == sensor_to_body_map_.end()) {
            throw std::invalid_argument("Sensor is not in the predefined list!");
        }
        sensor_to_body_map_[sensor] = extrinsic;
        // 标记该传感器已配置（非虚拟）
        sensor_enabled_[sensor] = true;
    }

    // 重载：支持从Matrix4d直接构造Isometry3d（兼容外部传入4x4矩阵的场景）
    void SetSensorToParentExtrinsic(SensorType sensor, const Eigen::Matrix4d& extrinsic_mat) {
        Eigen::Isometry3d extrinsic;
        extrinsic.matrix() = extrinsic_mat; // 从4x4矩阵初始化（自动保证旋转正交性）
        SetSensorToParentExtrinsic(sensor, extrinsic);
    }

    // 获取任意传感器到body坐标系的转换矩阵（刚体变换）
    // 直接返回预存的到Body系的外参，无累积误差
    Eigen::Isometry3d GetSensorToBody(SensorType sensor) {
        // 首先确定body系
        SensorType body_sensor = DetermineBodySensor();
        
        // 如果目标传感器就是body系，返回单位矩阵
        if (sensor == body_sensor) {
            return Eigen::Isometry3d::Identity();
        }

        // 直接返回传感器到Body系的直接外参（虚拟传感器用单位矩阵）
        return GetExtrinsicWithVirtual(sensor);
    }

    // 获取当前选定的body系传感器类型
    SensorType GetBodySensor() {
        return DetermineBodySensor();
    }

    std::string GetBodyFrame() {
        SensorType body_sensor = DetermineBodySensor();
        return SensorTypeToString(body_sensor);
    }

    // 辅助：将Isometry3d转换为4x4矩阵输出（方便打印/存储）
    static Eigen::Matrix4d IsometryToMatrix(const Eigen::Isometry3d& iso) {
        return iso.matrix();
    }

private:
    // 私有构造函数：初始化传感器列表和默认值
    SensorExtrinsicsManager() {
        // 初始化所有传感器的默认外参（到Body系）为单位矩阵
        sensor_to_body_map_[SensorType::IMU] = Eigen::Isometry3d::Identity();
        sensor_to_body_map_[SensorType::WHEEL] = Eigen::Isometry3d::Identity();
        sensor_to_body_map_[SensorType::CAMERA_LEFT] = Eigen::Isometry3d::Identity();
        sensor_to_body_map_[SensorType::CAMERA_RIGHT] = Eigen::Isometry3d::Identity();
        sensor_to_body_map_[SensorType::LIDAR1] = Eigen::Isometry3d::Identity();
        sensor_to_body_map_[SensorType::LIDAR2] = Eigen::Isometry3d::Identity();
        sensor_to_body_map_[SensorType::GNSS] = Eigen::Isometry3d::Identity();

        // 初始化传感器启用状态（默认未启用）
        for (const auto& pair : sensor_to_body_map_) {
            sensor_enabled_[pair.first] = false;
        }
    }

    // 确定body系传感器（按优先级：IMU > 轮速计 > 左相机 > LiDAR1）
    SensorType DetermineBodySensor() {
        // 优先级1：启用IMU则以IMU为body
        if (sensor_enabled_[SensorType::IMU]) {
            return SensorType::IMU;
        }

        // 优先级2：无IMU，有轮速计则以轮速计为body
        if (sensor_enabled_[SensorType::WHEEL]) {
            return SensorType::WHEEL;
        }

        // 优先级3：无IMU和轮速计，有左相机则以左相机为body
        if (sensor_enabled_[SensorType::CAMERA_LEFT]) {
            return SensorType::CAMERA_LEFT;
        }

        // 优先级4：否则以LiDAR1为body
        return SensorType::LIDAR1;
    }

    // 获取传感器到Body系的外参（虚拟传感器返回单位矩阵）
    Eigen::Isometry3d GetExtrinsicWithVirtual(SensorType sensor) {
        if (sensor_enabled_[sensor]) {
            return sensor_to_body_map_[sensor];
        } else {
            return Eigen::Isometry3d::Identity();
        }
    }

    // 方便打印传感器类型的辅助函数
    std::string SensorTypeToString(SensorType type) {
        switch (type) {
            case SensorType::IMU: return "IMU";
            case SensorType::WHEEL: return "WHEEL";
            case SensorType::CAMERA_LEFT: return "CAMERA_LEFT";
            case SensorType::CAMERA_RIGHT: return "CAMERA_RIGHT";
            case SensorType::LIDAR1: return "LIDAR1";
            case SensorType::LIDAR2: return "LIDAR2";
            case SensorType::GNSS: return "GNSS";
            default: return "UNKNOWN";
        }
    }

    // 传感器到Body系的直接外参映射（核心变更：替代原有父子层级）
    std::unordered_map<SensorType, Eigen::Isometry3d> sensor_to_body_map_;
    // 传感器启用状态（true=实际存在，false=虚拟）
    std::unordered_map<SensorType, bool> sensor_enabled_;
};
#endif // __SENSOR_MANAGER_H__