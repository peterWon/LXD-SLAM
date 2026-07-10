#ifndef VERTEX_COLOR_PREDICTOR_H
#define VERTEX_COLOR_PREDICTOR_H

#include <Eigen/Dense>
#include <vector>
#include <map>
// #include <opencv2/opencv.hpp>

// 观测数据结构：使用方向向量替代角度
struct VertexObservation {
    int vertexIndex;               // 顶点索引
    Eigen::Vector3d viewDirection; // 相机坐标系下的观测方向向量（单位向量）
    double distance;               // 观测距离
    Eigen::Vector3d color;         // RGB颜色 (0-1范围)
};

// 高斯过程顶点颜色预测器
class GPVertexColorPredictor {
private:
    // 每个顶点的观测数据
    std::map<int, std::vector<Eigen::VectorXd>> featuresMap;  // 特征: [方向向量(x,y,z), 距离] -> 4维特征
    std::map<int, std::vector<Eigen::Vector3d>> colorsMap;    // 对应的颜色
    
    // 高斯过程超参数
    double lengthScale;  // 长度尺度
    double signalStd;    // 信号标准差
    double noiseStd;     // 噪声标准差
    
    // 预计算的核矩阵（每个顶点单独存储）
    std::map<int, Eigen::MatrixXd> kernelMatrices;
    
    // 单位化方向向量（确保方向向量模长为1）
    Eigen::Vector3d normalizeDirection(const Eigen::Vector3d& dir);
    
    // 核函数：计算两个特征向量的相似度
    double kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2);
    
    // 构建核矩阵
    Eigen::MatrixXd buildKernelMatrix(const std::vector<Eigen::VectorXd>& features);
    
public:
    // 构造函数：初始化超参数
    GPVertexColorPredictor(double lengthScale = 1.0, 
                          double signalStd = 1.0, 
                          double noiseStd = 0.01);
    
    // 添加观测数据
    void addObservations(const std::vector<VertexObservation>& observations);
    
    // 为指定顶点训练模型（构建核矩阵）
    void trainVertex(int vertexIndex);
    
    // 预测指定顶点在新视角下的颜色
    Eigen::Vector3d predictColor(int vertexIndex, 
                                const Eigen::Vector3d& newViewDir,  // 新的观测方向向量
                                double newDistance, 
                                Eigen::Vector3d& variance);         // 输出预测方差
    
    // 获取所有有观测数据的顶点索引
    std::vector<int> getVertexIndices() const;
    
    // 检查顶点是否有足够的观测数据
    bool hasEnoughData(int vertexIndex, int minObservations = 3) const;
    
    // 模型保存和加载
    bool saveModel(const std::string& filename) const;
    bool loadModel(const std::string& filename);
};

#endif // VERTEX_COLOR_PREDICTOR_H
