# LXD-SLAM

## 1. System Requirements

### 1.1  Support of C++ 17 Features

The compiler should at least support c++17 standards.

### 1.2  ROS-Noetic System

LXD-SLAM is developed under [ROS-Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) with its default OpenCV4 library. However, LXD-SLAM should be working on ROS-Melodic with OpenCV3. In the future, we may add support to ROS 2.

### 1.3  Eigen Library

Eigen is a fantastic matrix computation library. LXD-SLAM is developed under [Eigen3.3.7](https://eigen.tuxfamily.org/index.php?title=Main_Page). Other Eigen 3 versions should be OK for LXD-SLAM.

### 1.4  PCL Library

We use [PCL-1.14](https://github.com/PointCloudLibrary/pcl) Library for point cloud processing. 

### 1.5  GTSAM Library

We use [GTSAM 4.2.0](https://github.com/borglab/gtsam) for pose graph construction and optimization.

### 1.6 mesh_tools

We use mesh_tools to visualize the mesh map with the `mesh_msgs::MeshGeometryStamped` ROS message. mesh_tools also incorporates navigation functions upon mesh map. [Mesh tool introduction](https://github.com/naturerobots/mesh_tools)

Install mesh_tools by:

1. Install [lvr2](https://github.com/uos/lvr2):

```
sudo apt-get install build-essential \
     cmake cmake-curses-gui libflann-dev \
     libgsl-dev libeigen3-dev libopenmpi-dev \
     openmpi-bin opencl-c-headers ocl-icd-opencl-dev \
     libboost-all-dev \
     freeglut3-dev libhdf5-dev qtbase5-dev \
     qt5-default libqt5opengl5-dev liblz4-dev \
     libopencv-dev libyaml-cpp-dev
```

In Ubuntu 20.04 or higher,

```
sudo apt-get install  libvtk7-dev libvtk7-qt-dev
```

then:

```
cd a_non_ros_dir
```

build:

```
git clone https://github.com/uos/lvr2.git
cd lvr2 
mkdir build && cd build
cmake .. && make
sudo make install
```

It may take you some time.

Install mesh_tools,

```
cd your_ws/src
git clone https://github.com/naturerobots/mesh_tools.git
cd ..
rosdep update
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

## 2. Build LXD-SLAM

Download or clone this repo to your ROS workspace.

```
cd ~/ws_catkin
catkin_make
```

Source the setup file to let ROS recognize the related launch files.

```
source devel/setup.bash
```

## 3. Run with VIRAL Datasets

```
roslaunch lvimesher online_viral.launch bag_filename:=YOUR_ROSBAG_FILEPATH
```

## 4. Run with FusionPortableV2 Datasets

```
roslaunch lvimesher online_fusionportable.launch bag_filename:=YOUR_ROSBAG_FILEPATH
```

### ## 5. License

This software is open-sourced under GPLv3 license. See [GNU GPL v3.0 - GNU](https://www.gnu.org/licenses/gpl-3.0.html).

Please cite our paper if you use either our code or our fixed-wing datasets.
