#ifndef __SUBMAP_MAP_H__
#define __SUBMAP_MAP_H__

#include "cell.h"
#include "utility/parameter.h"
#include "utility/sensor_manager.h"
#include "camera_models/Camera.h"
#include "camera_models/CameraFactory.h"
#include "camera_models/CataCamera.h"
#include "camera_models/PinholeCamera.h"
#include "texture_mapper.h"

using namespace camodocal;

struct CellCandidate {
    const Cell* cell;
    double depth; // 在相机坐标系下的深度，用于排序
    
    // 按深度从小到大排序
    bool operator<(const CellCandidate& other) const {
        return depth < other.depth;
    }
};

// LinK3D 描述子结构（180 维）
struct LinK3DDescriptor {
    std::vector<float> descriptor; // 180 维向量
    Vec3d position;               // 关键点在子地图局部坐标系下的位置
    int id;                       // 关键点 ID

    LinK3DDescriptor() : descriptor(180, 0.0f) {}
};

class OverlapCellsRelation{
    //store the overlaping correspondences of map cells between current scan and global map
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //for each cell in current scan, it can be overlapped with global map:
    std::vector<Cell*> cells_now;//array of ptr to map cell in map_now
    //the correspondent cell in the global map are store in one-to-one correspondence vector:
    std::vector<Cell*> cells_glb;//array of ptr to map cell in map_glb
    //when cross-cell overlapping is allowed, we look for neighbour cells too, so 1-multi correspondence exist,
    std::vector<std::vector<Cell*>> multi_cells_glb;
    //inside the array, the map cells are store in order: 0,x-1,x+1,y-1,y+1,z-1,z+1 (when overlap legnth is 1)
    // cell in current scan can also have no correspondent cell in the global map, in this way, it is new map cells:
    std::vector<Cell*> cells_now_new;
};

struct Point2MeshMatch{
    Point2MeshMatch(Vec3d pt, Vec4d pl, double var, bool v):point(pt),plane(pl), variance(var), valid(v){}
    Point2MeshMatch(){}
    Vec3d point;
    Vec4d plane;//abcd

    Vec3d point_in_body;//
    double variance;
    bool valid=false;
};

class Submap{
    // Submap class. The current scan and the global map share similar behaviour, so both of them use the "Submap" class,
    // eventhough traditionally you may think only the global map is "Submap".
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Submap(camodocal::CameraPtr camera){camera_left_ = camera;};
    void initializeMap(size_t frame_id, const Mat4d& pose, const cv::Mat& image, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_raw_ptr);
    
    bool preprocessScan(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pcl_raw_ptr, 
                   PointMatrix& points_result, 
                   pcl::PointCloud<pcl::PointXYZRGB> & pcl_got,
                   double voxel_filter_size);

    void dividePointsIntoCell(PointMatrix & points_raw);
    void dividePointsIntoCellInitMap(PointMatrix & points_raw);

    std::vector<const Cell*> getIntersectedCells(const Ray& ray, int max_steps = 1000) const;
    Vec3d getVoxelKey(const Vec3d& point) const {
        double grid = Parameter::GetInstance().grid;
        return (point.array() / grid).floor() * grid;
    }

    pcl::PointCloud<pcl::PointXYZ> getAllVertices();
    

    void findMatchPointToVertices(const pcl::PointCloud<pcl::PointXYZ>& points,
                                  const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                                  std::vector<Point2MeshMatch>& matches);
    void findMatchCellToCell(const pcl::PointCloud<pcl::PointXYZ>& points,
                                  const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                                  std::vector<Point2MeshMatch>& matches);

    void findMatchVisualPointToVertices(const pcl::PointCloud<pcl::PointXYZ>& points_w,
                                  std::vector<Point2MeshMatch>& matches);

    std::vector<bool> FindCorrespondences(const pcl::PointCloud<pcl::PointXYZRGB>& source,
                             const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                             std::vector<Vec3d>& target_color_gradient,
                             pcl::PointCloud<pcl::PointXYZRGBNormal>& target_xyzrgbn);
    std::vector<bool> FindCorrespondencesV1(const pcl::PointCloud<pcl::PointXYZRGB>& source,
                             const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                             std::vector<Vec3d>& target_color_gradient,
                             pcl::PointCloud<pcl::PointXYZRGBNormal>& target_xyzrgbn);

    void prepareColorGradient(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& target);
    std::vector<bool> FindColorCorrespondences(const pcl::PointCloud<pcl::PointXYZRGB>& source,
                             const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                             std::vector<Vec3d>& target_color_gradient,
                             pcl::PointCloud<pcl::PointXYZRGBNormal>& target_xyzrgbn);
    
    

    void RetrieveColors(const pcl::PointCloud<pcl::PointXYZ>& xyz,
                        const Mat4d& pose_guess,/*pose of points in odom frame*/ 
                        pcl::PointCloud<pcl::PointXYZRGB>& xyzrgb);
    
    // for debug only
    void GetAllPoints(pcl::PointCloud<pcl::PointXYZRGB>& xyzrgb);                    

    OverlapCellsRelation overlapCells();
    OverlapCellsRelation overlapCellsCrossCell(int overlap_length);
    
    void updateMap(size_t frame_id,  const Mat4d& opt_pose, const cv::Mat& image, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_raw_ptr);
    void UpdateVertexColors(const Mat4d& T_s2c, const cv::Mat& image);
    
    void filterVerticesByVariance(bool global_map, double variance_thr);
    void filterVerticesByVarianceOverlap(double variance_thr, Submap & map_now);
    
    void filterMeshLocal();
    void filterMeshGlb(bool use_pose_slam = false);
    bool outputMeshAsPly(const std::string & filename, const mesh_msgs::MeshGeometryStamped & mesh_msg_to_save);

    void finish(){
        finished_ = true;
        // clear raw points in cell
        for (auto& pair : cells_glb) {
            Cell* cell = &pair.second;
            cell->cell_raw_points.clear_deep();
        }
    }

    void textureAndSaveObj(bool use_pose_slam);
    void exportTexturedMeshToObj(
        const std::string& mesh_dir,
        const std::string& base_filename,          
        const std::vector<Eigen::Vector3d>& V,     
        const std::vector<Eigen::Vector3i>& F,     
        const std::vector<Vec2d>& UV,    
        const cv::Mat& texture_img                 
    );


    void generateLinK3DFeatures(std::vector<LinK3DDescriptor>& out_descriptors);

    void aggregateCellFeatures(std::vector<Vec3d>& aggregated_keypoints);
    
    void computeDescriptors(const std::vector<Vec3d>& keypoints, 
                            std::vector<LinK3DDescriptor>& out_descriptors);
                            
// private://TODO
    inline Eigen::Vector3d projectPointToPlane(const Eigen::Vector3d& P, const Eigen::Vector4d& plane) {
        Eigen::Vector3d n = plane.head<3>();
        double d = plane(3);
        double t = (n.dot(P) + d) / n.squaredNorm();
        return P - t * n;
    }

    inline bool getColorInLeftCamera(const Eigen::Vector3d& p_s, 
            const Mat4d& T_cs, const cv::Mat& image, Vec3d& rgb) { 
        Vec3d p_c = T_cs.block<3,3>(0,0) * p_s + T_cs.block<3,1>(0,3);
        Vec2d uv;
        camera_left_->spaceToPlane(p_c, uv);
        Vec2i uv_int = uv.cast<int>();
        if (uv_int.x() >= 0 && uv_int.x() < image.cols && uv_int.y() >= 0 && uv_int.y() < image.rows) {
            cv::Vec3b color = image.at<cv::Vec3b>(uv_int.y(), uv_int.x());
            rgb << color[2], color[1], color[0];
            return true;
        }else{
            return false;
        }
    }
    
    inline float getShiTomasScore(const Eigen::Vector3d& p_s,  const Mat4d& T_cs, const cv::Mat& image){
        Vec2d uv;
        Vec3d p_c = T_cs.block<3,3>(0,0) * p_s + T_cs.block<3,1>(0,3);
        camera_left_->spaceToPlane(p_c, uv);
        Vec2i uv_int = uv.cast<int>();
        if(uv_int.x() < 0 || uv_int.x() >= image.cols || uv_int.y() < 0 || uv_int.y() >= image.rows){
            return -1.;
        }

        return shiTomasiScore(image, uv_int.x(), uv_int.y());
    }

    inline bool getPatchInLeftCamera(const Eigen::Vector3d& p_s,  const Mat4d& T_cs, 
            const cv::Mat& img, cv::Mat& patch) {
        const int patch_size = 8;
        Vec2d uv;  
        Vec3d p_c = T_cs.block<3,3>(0,0) * p_s + T_cs.block<3,1>(0,3);
        camera_left_->spaceToPlane(p_c, uv);
        Vec2i uv_int = uv.cast<int>();
        int u = uv_int.x();
        int v = uv_int.y();
        if(u < 0 || u >= img.cols || v < 0 || v >= img.rows)
            return false;
        const int halfbox_size = 4;
        const int box_size = 2*halfbox_size;
        const int box_area = box_size*box_size;
        const int x_min = u-halfbox_size;
        const int x_max = u+halfbox_size;
        const int y_min = v-halfbox_size;
        const int y_max = v+halfbox_size;
        if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
            return false; // patch is too close to the boundary
        
        cv::Range row_range(y_min, y_max);
        cv::Range col_range(x_min, x_max);
        patch = img(row_range, col_range).clone();
        return true;
    }


    float shiTomasiScore(const cv::Mat& img, int u, int v){
        assert(img.type() == CV_8UC1);
        float dXX = 0.0;
        float dYY = 0.0;
        float dXY = 0.0;
        const int halfbox_size = 4;
        const int box_size = 2*halfbox_size;
        const int box_area = box_size*box_size;
        const int x_min = u-halfbox_size;
        const int x_max = u+halfbox_size;
        const int y_min = v-halfbox_size;
        const int y_max = v+halfbox_size;

        if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
            return 0.0; // patch is too close to the boundary

        const int stride = img.step.p[0];
        for( int y=y_min; y<y_max; ++y ){
            const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
            const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
            const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
            const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
            for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom){
                float dx = *ptr_right - *ptr_left;
                float dy = *ptr_bottom - *ptr_top;
                dXX += dx*dx;
                dYY += dy*dy;
                dXY += dx*dy;
            }
        }

        // Find and return smaller eigenvalue:
        dXX = dXX / (2.0 * box_area);
        dYY = dYY / (2.0 * box_area);
        dXY = dXY / (2.0 * box_area);
        return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
    }

    void updateReferencePatch(const cv::Mat& image, const Mat4d& T_cs, Cell& cell){
        for(int dir = 0; dir < 3; dir++){
            size_t npt = cell.ary_cell_vertices[dir].num_point;
            if(npt < 1) continue;
            std::vector<size_t> idx(npt);
            std::vector<float> scores(npt, 0.);
            iota(idx.begin(), idx.end(),  0);
            std::for_each(idx.begin(), idx.end(), [&](int k){
                Vec3d vs = cell.ary_cell_vertices[dir].point.col(k);
                double var = cell.ary_cell_vertices[dir].variance(0, k);
                if(var < 0.2){
                    scores[k] = getShiTomasScore(vs, T_cs, image);
                }
            });
            auto it = std::max_element(scores.begin(), scores.end());
            cell.patch_idx[dir] = std::distance(scores.begin(), it);
            Vec3d ps = cell.ary_cell_vertices[dir].point.col(cell.patch_idx[dir]);
            bool patch_exist = getPatchInLeftCamera(ps, T_cs, image, cell.ref_patch[dir]);
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB> fromPointMatrix(const PointMatrix& cloud){
        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
        pcl_cloud.width = cloud.num_point;
        pcl_cloud.height = 1;
        pcl_cloud.points.resize(pcl_cloud.width * pcl_cloud.height);
        for (size_t i = 0; i < cloud.num_point; ++i) {
            pcl_cloud.points[i].x = cloud.point(0, i);
            pcl_cloud.points[i].y = cloud.point(1, i);
            pcl_cloud.points[i].z = cloud.point(2, i);
            pcl_cloud.points[i].r = static_cast<uint8_t>(cloud.rgb(0, i));
            pcl_cloud.points[i].g = static_cast<uint8_t>(cloud.rgb(1, i));
            pcl_cloud.points[i].b = static_cast<uint8_t>(cloud.rgb(2, i));
        }
        return pcl_cloud;
    }

    void AddVisualMapPoint(const VisualPoint3D& vpt_w){
        VisualPoint3D vpt_s;
        Mat4d T_w2s = pose_odom_.inverse();
        vpt_s.id = vpt_w.id;
        vpt_s.xyz = T_w2s.block<3, 3>(0, 0) * vpt_w.xyz + T_w2s.block<3, 1>(0, 3);
        vpt_s.descriptor = vpt_w.descriptor;
        Vec3d current_key = getVoxelKey(vpt_s.xyz);
        double grid = Parameter::GetInstance().grid;
        double hash = getPosiWithTime(current_key(0), current_key(1), current_key(2), grid, 0);
        auto it = cells_glb.find(hash);
        if (it != cells_glb.end()) {
            it->second.visual_points_[vpt_s.id] = vpt_s;
        } 
    }

    int get_id(){
        return id_;
    }

    std::string name;// map_glb or map_now
    int id_;
    
    bool initialized_ = false;
    bool finished_ = false;
    bool added_to_graph_ = false;
    Mat4d pose_odom_ = Mat4d::Identity();
    Mat4d pose_slam_ = Mat4d::Identity();
    
    std::vector<size_t> frame_ids_;
    std::vector<Mat4d> node_poses_; //in local frame (odom)

    int step;// frame count

    pcl::PointCloud<pcl::PointXYZRGB> pcl_raw;//current new scan
    pcl::PointCloud<pcl::PointXYZRGB> pcl_voxel_filted;//downsampled
    
    // deprecated
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_for_registration_;
    std::vector<Eigen::Vector3d> target_color_gradient_;
    std::vector<Eigen::Vector3d> target_normal_;
    double search_radius_ = 0.3;//TODO

    PointMatrix points_turned;//raw points, store in point matrix, in submap frame
    PointMatrix points_turned_odom;//raw points, store in point matrix, in odom frame, for visualization

    PointMatrix vertices_filted;//for visiualization
    PointMatrix ary_vertices_all_show[3];//for visiualization
    PointMatrix ary_overlap_vertices[3];//for registration
    PointMatrix ary_normal[3];//for point2planeregistration

    mesh_msgs::MeshGeometryStamped mesh_msg_now_;//for mesh visualization
    mesh_msgs::MeshGeometryStamped mesh_msg_full_;//for mesh visualization
    
    std::unordered_map<double, Cell> cells_glb;//store the gped point in map glb
    // std::vector<std::pair<double, Cell>> cells_now;//map now, why vector: easy for multi-thread
    std::vector<int> index_bucket_enough_point;//only cells have enough raw points will be reconstructed
    std::vector<Cell*> newly_updated_cells;//for visualization

    bool first_time_update_ = true;
    bool sc_processed_ = false;

    using KeyType = std::bitset<3 * 32>;
    static KeyType IndexToKey(const Eigen::Array3i& index);
    Eigen::Array3i GetCellIndex(const Eigen::Vector3f& point) const;
    
    // global
    std::unordered_map<KeyType, size_t> cell_key_set_;
    std::vector<PointMatrix> ary_points_bucket_;
    std::vector<Cell> ary_bucket_cell_;
    std::vector<Eigen::Array3i> ary_cell_xyz_;
    
    // for current point cloud
    std::unordered_map<KeyType, size_t> cells_keyset_now_;
    std::vector<PointMatrix> ary_points_now_;
    std::vector<Cell> ary_cells_now_;
    std::vector<Eigen::Array3i> ary_cells_xyz_now_;

    Vec3d rgb_to_intensity_weight_ = Eigen::Vector3d::Ones () / 3.0;


    // for texture mapping
    camodocal::CameraPtr camera_left_;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pointclouds_for_texture_;
    std::vector<cv::Mat> images_for_texture_;
    cv::Mat baked_texture_;
    
    void generate_texture_from_multiview(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& UV,
        cv::Mat& texture_map
    ); 
    bool project_point_to_pixel(
        const Eigen::Vector3d& point_world,
        const Eigen::Matrix4d& T_cw,
        Eigen::Vector2i& pixel,
        Eigen::Vector3d& point_camera,
        bool& is_visible
    ) ;
};

// adapted from cartographer
class ActiveSubmaps{
public:
    ActiveSubmaps(camodocal::CameraPtr camera){
        unique_camera_ = camera;
        AddSubmap();
    }
    ActiveSubmaps(const ActiveSubmaps&) = delete;
    ActiveSubmaps& operator=(const ActiveSubmaps&) = delete;
    
    void AddData(size_t frame_id,  
                const Mat4d& local_submap_pose, 
                const cv::Mat& image,
                const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_raw_ptr){
        for (auto& submap : submaps_) {
            if(!submap->initialized_){
                submap->initializeMap(frame_id, local_submap_pose, image, pcl_raw_ptr);
            }else{
                submap->updateMap(frame_id, local_submap_pose, image, pcl_raw_ptr);
            }
        }
        if (submaps_.back()->node_poses_.size() == Parameter::GetInstance().frames_for_submap / 2) {
            AddSubmap();
        }
    }

    void AddVisualMapPoint(const VisualPoint3D& vpt_w){
        for (auto& submap : submaps_) {
            submap->AddVisualMapPoint(vpt_w);
        }
    }


    std::shared_ptr<Submap> matching_submap(){
        if(submaps_.empty()){
            return nullptr;
        } else {
            return submaps_.front();
        }
    }

    std::vector<std::shared_ptr<Submap>> submaps() const{
        return submaps_;
    }
private:
    void AddSubmap() {
        if (submaps_.size() > 1) {
            submaps_.front()->finish();
            submaps_.erase(submaps_.begin());
        }
        submaps_.emplace_back(new Submap(unique_camera_));
        submaps_.back()->id_ = newest_submap_id_;
        newest_submap_id_++;
    }
    std::vector<std::shared_ptr<Submap>> submaps_;
    camodocal::CameraPtr unique_camera_;
    int newest_submap_id_ = 0;
};
#endif //__SUBMAP_MAP_H__
