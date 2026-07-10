source /home/wz/Codes/ros_ws/lvi_mesher/devel/setup.sh

# Get the current directory
CURR_DIR=$(pwd)
# Get the location of the viral package
PACKAGE_DIR=$(pwd)
# Return to the current dir, print the directions
cd $CURR_DIR
echo CURRENT DIR: $CURR_DIR

# # should modified according to your dataset location
export EPOC_DIR=/media/wz/2C96A0A60155E8F8/Dataset/KITTI-Odometry/LOG/LXD-SLAM-lc-vloop/
export DATASET_LOCATION=/media/wz/2C96A0A60155E8F8/Dataset/KITTI-Odometry/
# # wait;

rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '00' 00_to_02;
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '01' '00_to_02';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '02' '00_to_02';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '03' '03';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '04' '04_to_10';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '05' '04_to_10';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '06' '04_to_10';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '07' '04_to_10';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '08' '04_to_10';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '09' '04_to_10';
# rosrun lvimesher run_one_kitti.sh $EPOC_DIR $DATASET_LOCATION '10' '04_to_10';

# # wait;
# poweroff
