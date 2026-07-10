#include "vertex_color_predictor.h"
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <glog/logging.h>

// 构造函数
GPVertexColorPredictor::GPVertexColorPredictor(double lengthScale, 
                                             double signalStd, 
                                             double noiseStd)
    : lengthScale(lengthScale), signalStd(signalStd), noiseStd(noiseStd) {}

// 单位化方向向量
Eigen::Vector3d GPVertexColorPredictor::normalizeDirection(const Eigen::Vector3d& dir) {
    double norm = dir.norm();
    if (norm < 1e-9) {
        throw std::invalid_argument("方向向量不能为零向量");
    }
    return dir / norm;
}

// 核函数：平方指数核
double GPVertexColorPredictor::kernel(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) {
    if (x1.size() != x2.size()) {
        throw std::invalid_argument("特征向量维度不匹配");
    }
    
    double squaredDist = (x1 - x2).squaredNorm();
    return signalStd * signalStd * std::exp(-0.5 * squaredDist / (lengthScale * lengthScale));
}

// 构建核矩阵
Eigen::MatrixXd GPVertexColorPredictor::buildKernelMatrix(const std::vector<Eigen::VectorXd>& features) {
    int n = features.size();
    Eigen::MatrixXd K(n, n);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            K(i, j) = kernel(features[i], features[j]);
        }
        // 添加噪声项（对角线）
        K(i, i) += noiseStd * noiseStd;
    }
    
    return K;
}

// 添加观测数据
void GPVertexColorPredictor::addObservations(const std::vector<VertexObservation>& observations) {
    for (const auto& obs : observations) {
        // 确保方向向量是单位向量
        Eigen::Vector3d normalizedDir = normalizeDirection(obs.viewDirection);
        
        // 构建特征向量：[方向向量(x,y,z), 距离] -> 4维特征
        Eigen::VectorXd feature(4);
        feature << normalizedDir.x(), normalizedDir.y(), normalizedDir.z(), obs.distance;
        
        // 存储到对应顶点的数据中
        featuresMap[obs.vertexIndex].push_back(feature);
        colorsMap[obs.vertexIndex].push_back(obs.color);
    }
}

// 为指定顶点训练模型
void GPVertexColorPredictor::trainVertex(int vertexIndex) {
    if (!featuresMap.count(vertexIndex) || !colorsMap.count(vertexIndex)) {
        throw std::invalid_argument("顶点 " + std::to_string(vertexIndex) + " 没有观测数据");
    }
    
    const auto& features = featuresMap[vertexIndex];
    if (features.empty()) {
        throw std::invalid_argument("顶点 " + std::to_string(vertexIndex) + " 观测数据为空");
    }
    
    // 构建并存储核矩阵
    kernelMatrices[vertexIndex] = buildKernelMatrix(features);
}

// 预测颜色
Eigen::Vector3d GPVertexColorPredictor::predictColor(int vertexIndex, 
                                                   const Eigen::Vector3d& newViewDir, 
                                                   double newDistance, 
                                                   Eigen::Vector3d& variance) {
    // 检查顶点数据
    if (!hasEnoughData(vertexIndex)) {
        // LOG(WARNING) << "顶点 " << vertexIndex << " 数据不足";
        return Eigen::Vector3d::Zero();
    }
    
    // 确保方向向量单位化
    Eigen::Vector3d normalizedDir = normalizeDirection(newViewDir);
    
    // 构建新视角的特征向量
    Eigen::VectorXd xStar(4);
    xStar << normalizedDir.x(), normalizedDir.y(), normalizedDir.z(), newDistance;
    
    // 获取训练数据
    const auto& features = featuresMap[vertexIndex];
    const auto& colors = colorsMap[vertexIndex];
    const auto& K = kernelMatrices[vertexIndex];
    int n = features.size();
    
    // 计算k*向量（新特征与所有训练特征的核函数值）
    Eigen::VectorXd kStar(n);
    for (int i = 0; i < n; ++i) {
        kStar(i) = kernel(features[i], xStar);
    }
    
    // 计算K的逆矩阵（使用Cholesky分解提高效率）
    Eigen::MatrixXd KInv = K.llt().solve(Eigen::MatrixXd::Identity(n, n));
    
    // 对RGB三个通道分别进行预测
    Eigen::Vector3d predictedColor;
    for (int c = 0; c < 3; ++c) {
        // 提取当前通道的颜色值
        Eigen::VectorXd y(n);
        for (int i = 0; i < n; ++i) {
            y(i) = colors[i](c);
        }
        
        // 预测均值
        predictedColor(c) = kStar.transpose() * KInv * y;
        
        // 预测方差（不确定性）
        double kStarStar = kernel(xStar, xStar) + noiseStd * noiseStd;  // 新特征自身的核函数值
        variance(c) = kStarStar - kStar.transpose() * KInv * kStar;
    }
    
    // 确保颜色在有效范围内
    predictedColor = predictedColor.cwiseMin(1.0).cwiseMax(0.0);
    
    return predictedColor;
}

// 获取所有顶点索引
std::vector<int> GPVertexColorPredictor::getVertexIndices() const {
    std::vector<int> indices;
    for (const auto& pair : featuresMap) {
        indices.push_back(pair.first);
    }
    return indices;
}

// 检查是否有足够数据
bool GPVertexColorPredictor::hasEnoughData(int vertexIndex, int minObservations) const {
    auto it = featuresMap.find(vertexIndex);
    if (it == featuresMap.end()) return false;
    return it->second.size() >= minObservations;
}

// 保存模型
bool GPVertexColorPredictor::saveModel(const std::string& filename) const {
    /* std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    // 保存超参数
    file.write(reinterpret_cast<const char*>(&lengthScale), sizeof(lengthScale));
    file.write(reinterpret_cast<const char*>(&signalStd), sizeof(signalStd));
    file.write(reinterpret_cast<const char*>(&noiseStd), sizeof(noiseStd));
    
    // 保存顶点数量
    size_t numVertices = featuresMap.size();
    file.write(reinterpret_cast<const char*>(&numVertices), sizeof(numVertices));
    
    // 保存每个顶点的数据
    for (const auto& pair : featuresMap) {
        int vertexIndex = pair.first;
        const auto& features = pair.second;
        const auto& colors = colorsMap.at(vertexIndex);
        
        file.write(reinterpret_cast<const char*>(&vertexIndex), sizeof(vertexIndex));
        
        // 保存特征数量
        size_t numFeatures = features.size();
        file.write(reinterpret_cast<const char*>(&numFeatures), sizeof(numFeatures));
        
        // 保存特征
        for (const auto& feat : features) {
            for (int i = 0; i < feat.size(); ++i) {
                double val = feat(i);
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
        
        // 保存颜色
        for (const auto& color : colors) {
            for (int c = 0; c < 3; ++c) {
                double val = color(c);
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
    }*/
    
    return true; 
}

// 加载模型
bool GPVertexColorPredictor::loadModel(const std::string& filename) {
    /* std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    // 清空现有数据
    featuresMap.clear();
    colorsMap.clear();
    kernelMatrices.clear();
    
    // 加载超参数
    file.read(reinterpret_cast<char*>(&lengthScale), sizeof(lengthScale));
    file.read(reinterpret_cast<char*>(&signalStd), sizeof(signalStd));
    file.read(reinterpret_cast<char*>(&noiseStd), sizeof(noiseStd));
    
    // 加载顶点数量
    size_t numVertices;
    file.read(reinterpret_cast<char*>(&numVertices), sizeof(numVertices));
    
    // 加载每个顶点的数据
    for (size_t v = 0; v < numVertices; ++v) {
        int vertexIndex;
        file.read(reinterpret_cast<char*>(&vertexIndex), sizeof(vertexIndex));
        
        // 加载特征数量
        size_t numFeatures;
        file.read(reinterpret_cast<char*>(&numFeatures), sizeof(numFeatures));
        
        // 加载特征
        std::vector<Eigen::VectorXd> features;
        for (size_t f = 0; f < numFeatures; ++f) {
            Eigen::VectorXd feat(4);
            for (int i = 0; i < 4; ++i) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                feat(i) = val;
            }
            features.push_back(feat);
        }
        
        // 加载颜色
        std::vector<Eigen::Vector3d> colors;
        for (size_t c = 0; c < numFeatures; ++c) {
            Eigen::Vector3d color;
            for (int i = 0; i < 3; ++i) {
                double val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                color(i) = val;
            }
            colors.push_back(color);
        }
        
        featuresMap[vertexIndex] = features;
        colorsMap[vertexIndex] = colors;
        
        // 重建核矩阵
        kernelMatrices[vertexIndex] = buildKernelMatrix(features);
    } */
    
    return true;
}
