source /home/wz/codes/ros_ws/livo/lvi_mesher/devel/setup.sh

# Get the current directory
CURR_DIR=$(pwd)
# Get the location of the viral package
PACKAGE_DIR=$(pwd)
# Return to the current dir, print the directions
cd $CURR_DIR
echo CURRENT DIR: $CURR_DIR
echo PACKAGE DIR:   $PACKAGE_DIR

# should modified according to your dataset location
export EPOC_DIR=/media/wz/2C96A0A60155E8F8/Dataset/NewerCollege/02_long_experiment/LOG/LXD-SLAM-li/


export CAPTURE_SCREEN=false;
export LOG_DATA=true;

# # wait;
export DATASET_LOCATION=/media/wz/2C96A0A60155E8F8/Dataset/NewerCollege/02_long_experiment/rosbag/
rosrun lvimesher run_one_ncd.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR rooster_2020-03-10-11-39-38_0 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;
