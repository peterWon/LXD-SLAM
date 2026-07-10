#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <algorithm>
#include <cmath>

// 假设这些头文件存在，实际使用时需确保路径正确
// 如果没有camodocal，需要替换为简单的针孔投影逻辑
#include "camera_models/Camera.h"
#include "camera_models/CameraFactory.h"
#include "utility/parameter.h" 

// 模拟相机类接口，以便在没有camodocal环境下也能理解逻辑
/*
class Camera {
public:
    virtual void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const = 0;
    virtual int imageWidth() const = 0;
    virtual int imageHeight() const = 0;
};
*/

using namespace camodocal;

class TextureMapper {
public:
    struct Mesh {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<Eigen::Vector3i> faces;
    };

    struct PosedImage {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        cv::Mat image;
        int id;
    };

    TextureMapper(const Mesh& mesh, const std::vector<PosedImage>& cameras) 
        : mesh_(mesh), images_(cameras) {
        // 初始化相机模型
        left_camera_ = CameraFactory::instance()->generateCameraFromYamlFile(Parameter::GetInstance().left_camera_model_path);
    }

    void generateTexturedMesh(const std::string& output_mesh_path, 
                              const std::string& output_texture_path,
                              const std::string& output_mtl_path) {
        std::cout << "1. Generating Atlas Layout..." << std::endl;
        generateAtlasLayout(); // 替代原有的 generateUVCoordinates
        
        std::cout << "2. Allocating Texture Atlas..." << std::endl;
        createTextureAtlas();
        
        std::cout << "3. Projecting Textures..." << std::endl;
        projectTexturesToUV();
        
        std::cout << "4. Blending (Simple Inpainting)..." << std::endl;
        postProcessTexture();
        
        std::cout << "5. Saving Mesh..." << std::endl;
        saveTexturedMesh(output_mesh_path, output_texture_path, output_mtl_path);
    }

private:
    Mesh mesh_;
    std::vector<PosedImage> images_;
    camodocal::CameraPtr left_camera_;
    
    cv::Mat texture_atlas_;
    int atlas_size_ = 4096;
    int padding_ = 2; // 纹理块之间的间隔，防止出血

    // 核心修改：不再为每个顶点存储1个UV，而是为每个面的每个角存储UV
    // face_uvs_[face_idx][0/1/2] 对应 face[0/1/2] 的 UV
    std::vector<std::vector<Eigen::Vector2d>> face_uvs_; 

    // 1. 生成图集布局（简单的网格打包算法）
    // 将每个三角形映射到图集中的独立区域，彻底解决重叠问题
    void generateAtlasLayout() {
        face_uvs_.resize(mesh_.faces.size());
        
        // 计算需要多少个网格
        int num_faces = mesh_.faces.size();
        // 估算每个三角形在图集中占据的方块大小
        // 简单的做法是把图集切成 sqrt(N) x sqrt(N) 的网格
        int grid_cols = std::ceil(std::sqrt((double)num_faces));
        int grid_rows = std::ceil((double)num_faces / grid_cols);
        
        double slot_size_u = 1.0 / grid_cols;
        double slot_size_v = 1.0 / grid_rows;
        
        // 在每个slot内部留一点边距
        double margin_u = slot_size_u * 0.05; 
        double margin_v = slot_size_v * 0.05;

        for (int i = 0; i < num_faces; ++i) {
            int row = i / grid_cols;
            int col = i % grid_cols;

            double u_start = col * slot_size_u + margin_u;
            double v_start = row * slot_size_v + margin_v;
            double u_end = (col + 1) * slot_size_u - margin_u;
            double v_end = (row + 1) * slot_size_v - margin_v;

            // 简单的直角三角形映射，或者根据3D三角形形状展开
            // 这里为了鲁棒性，简单地映射为直角三角形放在Slot左下
            face_uvs_[i].push_back(Eigen::Vector2d(u_start, v_start));     // v0 -> (0,0)
            face_uvs_[i].push_back(Eigen::Vector2d(u_end, v_start));       // v1 -> (1,0)
            face_uvs_[i].push_back(Eigen::Vector2d(u_start, v_end));       // v2 -> (0,1)
            
            // 进阶做法：应该使用最小二乘保角映射(LSCM)来保持形状，
            // 但对于简单的贴图，这样保证了互不重叠。
        }
    }

    void createTextureAtlas() {
        // 初始化为灰色，方便调试未覆盖区域
        texture_atlas_ = cv::Mat(atlas_size_, atlas_size_, CV_8UC3, cv::Scalar(128, 128, 128));
    }

    void projectTexturesToUV() {
        // 计算每个面对应的最佳相机
        std::vector<int> best_cameras(mesh_.faces.size(), -1);
        
        // 并行计算最佳相机（这是只读操作，且写入不同索引，安全）
        #pragma omp parallel for
        for (size_t i = 0; i < mesh_.faces.size(); ++i) {
            best_cameras[i] = computeBestCamera(i);
        }

        // 渲染纹理到图集
        // 注意：这里不能并行，因为不同面可能写入相邻像素，且OpenCV Mat不是线程安全的
        // 由于我们使用了Grid Packing，写入区域理论上不重叠，但为了安全起见，单线程执行
        for (size_t i = 0; i < mesh_.faces.size(); ++i) {
            if (best_cameras[i] != -1) {
                projectFaceTexture(i, best_cameras[i]);
            }
        }
    }

    int computeBestCamera(int face_idx) {
        const auto& face = mesh_.faces[face_idx];
        Eigen::Vector3d v0 = mesh_.vertices[face[0]];
        Eigen::Vector3d v1 = mesh_.vertices[face[1]];
        Eigen::Vector3d v2 = mesh_.vertices[face[2]];
        Eigen::Vector3d face_center = (v0 + v1 + v2) / 3.0;
        Eigen::Vector3d normal = (v1 - v0).cross(v2 - v0).normalized();

        int best_cam = -1;
        double max_score = -1.0;

        for (size_t i = 0; i < images_.size(); ++i) {
            const auto& cam = images_[i];
            Eigen::Vector3d cam_center = cam.t;
            Eigen::Vector3d view_dir = (face_center - cam_center).normalized(); // Camera to Face
            double dist = (face_center - cam_center).norm();

            // 1. 角度检查 (Back-face culling)
            // 法线应指向相机 (dot product < 0 because view_dir is cam->face)
            double cos_angle = normal.dot(-view_dir); 
            
            if (cos_angle > 0.1) { // 稍微放宽一点阈值
                // 2. 投影检查 (是否在FOV内)
                Eigen::Vector3d pt_cam = cam.R.transpose() * (face_center - cam.t);
                if (pt_cam.z() < 0.1) continue; // Behind camera

                Eigen::Vector2d uv;
                left_camera_->spaceToPlane(pt_cam, uv);
                
                if (uv.x() >= 0 && uv.x() < cam.image.cols && 
                    uv.y() >= 0 && uv.y() < cam.image.rows) {
                    
                    // 3. 分数计算
                    // 结合角度和分辨率(距离倒数)
                    // score = cos(theta) / distance^2 (近似投影面积)
                    double score = cos_angle / (dist * dist);
                    
                    // 修正：这里仍然缺少真正的遮挡查询(Raycast/Z-buffer)。
                    // 实际工程中，这里需要根据depth map判断点是否被遮挡。
                    
                    if (score > max_score) {
                        max_score = score;
                        best_cam = i;
                    }
                }
            }
        }
        return best_cam;
    }

    void projectFaceTexture(int face_idx, int cam_idx) {
        const auto& cam = images_[cam_idx];
        const auto& face = mesh_.faces[face_idx];
        
        // 获取图集中的三角形UV坐标
        Eigen::Vector2d uv0 = face_uvs_[face_idx][0];
        Eigen::Vector2d uv1 = face_uvs_[face_idx][1];
        Eigen::Vector2d uv2 = face_uvs_[face_idx][2];

        // 3D 顶点
        Eigen::Vector3d P0 = mesh_.vertices[face[0]];
        Eigen::Vector3d P1 = mesh_.vertices[face[1]];
        Eigen::Vector3d P2 = mesh_.vertices[face[2]];

        // 计算图集中的包围盒 (ROI)
        double min_u = std::min({uv0.x(), uv1.x(), uv2.x()});
        double max_u = std::max({uv0.x(), uv1.x(), uv2.x()});
        double min_v = std::min({uv0.y(), uv1.y(), uv2.y()});
        double max_v = std::max({uv0.y(), uv1.y(), uv2.y()});

        int u_start = std::max(0, (int)(min_u * atlas_size_));
        int u_end = std::min(atlas_size_ - 1, (int)(max_u * atlas_size_) + 1);
        int v_start = std::max(0, (int)(min_v * atlas_size_));
        int v_end = std::min(atlas_size_ - 1, (int)(max_v * atlas_size_) + 1);

        // 遍历图集像素（反向映射）
        for (int v = v_start; v <= v_end; ++v) {
            for (int u = u_start; u <= u_end; ++u) {
                Eigen::Vector2d p_uv((u + 0.5) / atlas_size_, (v + 0.5) / atlas_size_);

                // 计算重心坐标 (Barycentric Coordinates) 针对 UV 三角形
                // 解决原代码中的变量遮蔽问题，使用明确的变量名
                Eigen::Vector2d v0_uv = uv1 - uv0;
                Eigen::Vector2d v1_uv = uv2 - uv0;
                Eigen::Vector2d v2_uv = p_uv - uv0;

                double d00 = v0_uv.dot(v0_uv);
                double d01 = v0_uv.dot(v1_uv);
                double d11 = v1_uv.dot(v1_uv);
                double d20 = v2_uv.dot(v0_uv);
                double d21 = v2_uv.dot(v1_uv);
                
                double denom = d00 * d11 - d01 * d01;
                if (std::abs(denom) < 1e-8) continue;

                double v_b = (d11 * d20 - d01 * d21) / denom;
                double w_b = (d00 * d21 - d01 * d20) / denom;
                double u_b = 1.0 - v_b - w_b;

                // 检查是否在三角形内
                if (v_b >= 0 && w_b >= 0 && u_b >= 0) {
                    // 利用重心坐标插值得到对应的3D点
                    Eigen::Vector3d P_interp = u_b * P0 + v_b * P1 + w_b * P2;

                    // 将3D点投影到源图像
                    Eigen::Vector3d P_cam = cam.R.transpose() * (P_interp - cam.t);
                    
                    if (P_cam.z() > 0.1) {
                        Eigen::Vector2d img_pix;
                        left_camera_->spaceToPlane(P_cam, img_pix);

                        // 双线性插值采样颜色
                        if (img_pix.x() >= 0 && img_pix.x() < cam.image.cols - 1 &&
                            img_pix.y() >= 0 && img_pix.y() < cam.image.rows - 1) {
                            
                            texture_atlas_.at<cv::Vec3b>(v, u) = getBilinearPixel(cam.image, img_pix);
                        }
                    }
                }
            }
        }
    }

    cv::Vec3b getBilinearPixel(const cv::Mat& img, const Eigen::Vector2d& pt) {
        int x = (int)pt.x();
        int y = (int)pt.y();
        double dx = pt.x() - x;
        double dy = pt.y() - y;

        cv::Vec3b c00 = img.at<cv::Vec3b>(y, x);
        cv::Vec3b c10 = img.at<cv::Vec3b>(y, x + 1);
        cv::Vec3b c01 = img.at<cv::Vec3b>(y + 1, x);
        cv::Vec3b c11 = img.at<cv::Vec3b>(y + 1, x + 1);

        cv::Vec3f res = 
            c00 * (1 - dx) * (1 - dy) +
            c10 * dx * (1 - dy) +
            c01 * (1 - dx) * dy +
            c11 * dx * dy;
        
        return (cv::Vec3b)res;
    }
    
    void postProcessTexture() {
        // 简单的膨胀操作，填充三角形边缘的黑缝 (Seam fixing)
        cv::Mat mask;
        cv::cvtColor(texture_atlas_, mask, cv::COLOR_BGR2GRAY);
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY_INV); // 找到未着色区域
        
        cv::Mat dilated;
        cv::dilate(texture_atlas_, dilated, cv::Mat(), cv::Point(-1,-1), 2);
        
        dilated.copyTo(texture_atlas_, mask);
    }

    void saveTexturedMesh(const std::string& mesh_path, const std::string& texture_path, const std::string& mtl_path) {
        cv::imwrite(texture_path, texture_atlas_);

        std::ofstream obj_file(mesh_path);
        if (!obj_file.is_open()) return;

        obj_file << "mtllib texture.mtl\n";
        obj_file << "usemtl Material\n";

        // 1. 写入顶点几何坐标
        for (const auto& v : mesh_.vertices) {
            obj_file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
        }

        // 2. 写入纹理坐标 (所有面的所有角)
        // 注意：face_uvs_ 结构是 vector<vector<Vec2>>, 需展平写入
        for (const auto& face_uv : face_uvs_) {
            for (const auto& uv : face_uv) {
                obj_file << "vt " << uv.x() << " " << (1.0 - uv.y()) << "\n";
            }
        }

        // 3. 写入面索引
        // 顶点索引复用 (1-based)，纹理索引唯一且递增
        int vt_idx_counter = 1;
        for (size_t i = 0; i < mesh_.faces.size(); ++i) {
            const auto& f = mesh_.faces[i];
            obj_file << "f ";
            // OBJ格式: v/vt
            obj_file << (f[0] + 1) << "/" << (vt_idx_counter++) << " ";
            obj_file << (f[1] + 1) << "/" << (vt_idx_counter++) << " ";
            obj_file << (f[2] + 1) << "/" << (vt_idx_counter++) << "\n";
        }
        
        // 生成简单的MTL文件 (同原代码，略)
        // 保存材质文件
        std::ofstream mtl_file(mtl_path);
        if (mtl_file.is_open()) {
            mtl_file << "newmtl Material\n";
            mtl_file << "Ka 1.0 1.0 1.0\n";  // 环境光
            mtl_file << "Kd 1.0 1.0 1.0\n";  // 漫反射
            mtl_file << "Ks 0.0 0.0 0.0\n";  // 高光
            mtl_file << "Ns 10.0\n";         // 高光指数
            mtl_file << "map_Kd " << texture_path << "\n";
            mtl_file.close();
        }
    }
};