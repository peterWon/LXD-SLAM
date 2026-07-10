cmake_minimum_required(VERSION 3.5)
project(lvimesher)

SET(CMAKE_CXX_FLAGS "-std=c++17 -Wno-deprecated-declarations -Wno-dev")
set(CMAKE_BUILD_TYPE "Release")
set(catkin_DIR /opt/ros/noetic/share/catkin/cmake)

#ros
#add_compile_definitions(VOXBLOX)
#set(USE_VOXBLOX ON)
set(USE_VOXBLOX OFF)
if(USE_VOXBLOX)
    message("USE_VOXBLOX")
    find_package(catkin REQUIRED COMPONENTS
            pcl_ros
            roscpp
            rospy
            sensor_msgs
            std_msgs
            mesh_msgs
            mesh_msgs_conversions
            cv_bridge
            image_transport
            livox_ros_driver
            velodyne_msgs
            velodyne_pointcloud
            )
else()
    message("DONT USE_VOXBLOX")
    find_package(catkin REQUIRED COMPONENTS
            pcl_ros
            pcl_ros
            roscpp
            rospy
            sensor_msgs
            std_msgs
            mesh_msgs
            mesh_msgs_conversions
            cv_bridge
            image_transport
            livox_ros_driver
            velodyne_msgs
            velodyne_pointcloud
            )
endif()
catkin_package()

include_directories(/usr/local/include include ${catkin_INCLUDE_DIRS})


# set(OpenCV_DIR "/home/wz/codes/opencv-4.5.0/build/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# include(${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/CMakeLists.txt)
# include(${PROJECT_SOURCE_DIR}/Thirdparty/g2o/CMakeLists.txt)


# add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/libigl)

# add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/xatlas)


#pcl
find_package(PCL REQUIRED)
include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
#ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#gtsam
find_package(GTSAM REQUIRED)
include_directories(${GTSAM_INCLUDE_DIRS})
link_directories(${GTSAM_LIBRARY_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


#mesh_msg
find_package(LVR2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
include_directories(include ${LVR2_INCLUDE_DIRS})
# FIND_PACKAGE(CGAL REQUIRED)
# include_directories(${CGAL_INCLUDE_DIRS})
# add_definitions(${CGAL_DEFINITIONS})
# link_directories(${CGAL_LIBRARY_DIRS})
ADD_DEFINITIONS(-D_USE_BOOST -D_USE_EIGEN -D_USE_OPENMP)
SET(_USE_BOOST TRUE)
SET(_USE_OPENMP TRUE)
SET(_USE_EIGEN TRUE)

add_subdirectory(./Thirdparty/GeographicLib/)

file(GLOB_RECURSE CAMERA_MODELS_CC "Thirdparty/camera_models/*.cc")
file(GLOB_RECURSE VINS_CC "Thirdparty/vins/*.cc")
file(GLOB_RECURSE SOPHUS_HPP "Thirdparty/sophus/sophus/*.hpp")
file(GLOB_RECURSE SCAN_CONTEXT_CPP "Thirdparty/scancontext/*.cpp")
file(GLOB_RECURSE SOURCE_CPP "src/*.cpp")
# file(GLOB_RECURSE ORBSLAM3_SOURCE_CC "src/orb_slam3/*.cc")
file(GLOB_RECURSE MVS_MESHING_SOURCE_CPP "Thirdparty/meshing/MVS/*.cpp")
include_directories(${CMAKE_INCLUDE_DIRECTORIES_BEFORE}
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    "Thirdparty/"
                    "Thirdparty/meshing/"
                    "Thirdparty/libigl/include"
                    "Thirdparty/GeographicLib/include"
                    # "/usr/local/include/"
                    )
                    # ${CMAKE_CURRENT_SOURCE_DIR}/include/orb_slam3)

# add_library(mvs_meshing Thirdparty/meshing/MVS/Common/Common.cpp
#                         Thirdparty/meshing/MVS/Common/Log.cpp
#                         Thirdparty/meshing/MVS/Common/Timer.cpp
#                         Thirdparty/meshing/MVS/Common/Types.cpp
#                         Thirdparty/meshing/MVS/Common/Util.cpp
#                         # MVS                       
#                         Thirdparty/meshing/MVS/Mesh.cpp
#                         Thirdparty/meshing/MVS/PointCloud.cpp
#                         Thirdparty/meshing/MVS/Camera.cpp
#                         Thirdparty/meshing/MVS/Platform.cpp
#                         Thirdparty/meshing/MVS/PLY.cpp
#                         Thirdparty/meshing/MVS/OBJ.cpp
#                         Thirdparty/meshing/MVS/IBFS.cpp)
# target_link_libraries(mvs_meshing 
#             ${Boost_LIBRARIES}
#             ${Boost_FILESYSTEM_LIBRARY}
#             ${Boost_SERIALIZATION_LIBRARY} # serialization
#             ${CGAL_LIBS}
#             ${OpenCV_LIBRARIES}
#             ${JPEG_LIBRARIES} 
#             gmp
#             pcl_common 
#             pcl_io
#             pcl_kdtree)




if(USE_VOXBLOX)
    message("USE_VOXBLOX")
    add_library(lvimesher
            ${CAMERA_MODELS_CC}
            ${SCAN_CONTEXT_CPP}
            ${VINS_CC}
            ${SOURCE_CPP}
            src/voxblox/mesh/mesher.cpp
            src/voxblox/mesh/marching_cubes.cc)
else()
    message("DONT USE_VOXBLOX")
    add_library(lvimesher
            ${CAMERA_MODELS_CC}
            ${SCAN_CONTEXT_CPP}
            ${VINS_CC}
            ${SOURCE_CPP})
endif()



target_link_libraries(lvimesher 
   ${catkin_LIBRARIES} 
   ${PCL_LIBRARIES} 
   ${OpenCV_LIBS} 
#    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
#    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
   ${CERES_LIBRARIES} 
   ${LVR2_LIBRARIES} gtsam glog tbb libGeographic)#xatlas igl::core igl::opengl igl::opengl_glfw 

add_executable(slam_node node/slam_node.cpp)
target_link_libraries(slam_node lvimesher)


add_executable(offline_node node/offline_node.cpp)
target_link_libraries(offline_node lvimesher)

add_executable(offline_node_kitti node/offline_node_kitti.cpp)
target_link_libraries(offline_node_kitti lvimesher)

add_executable(offline_node_ncd node/offline_node_ncd.cpp)
target_link_libraries(offline_node_ncd lvimesher)

add_executable(kitti_helper node/kitti_helper.cpp)
target_link_libraries(kitti_helper lvimesher)


add_executable(extrinsic_transform node/extrinsics_transform.cpp)
target_link_libraries(extrinsic_transform glog) 

# add_executable(kitti_intinsics_extrinsics node/kitti_intinsics_extrinsics.cpp)
# target_link_libraries(kitti_intinsics_extrinsics glog) 