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
export EPOC_DIR=/media/wz/2C96A0A60155E8F8/Dataset/FusionPortableV2/LOG/LXD-SLAM-lc/


export CAPTURE_SCREEN=false;
export LOG_DATA=true;

export DATASET_LOCATION=/media/wz/2C96A0A60155E8F8/Dataset/FusionPortableV2/rosbag/handheld/
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR handheld_escalator00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 handheld;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR handheld_escalator01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 handheld;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR handheld_grass00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 handheld;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR handheld_room00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 handheld;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR handheld_room01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 handheld;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR handheld_underground00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 handheld;

export DATASET_LOCATION=/media/wz/2C96A0A60155E8F8/Dataset/FusionPortableV2/rosbag/legged/
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR legged_grass01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 legged;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR legged_room00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 legged;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR legged_underground00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 legged;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR legged_grass00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 legged;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR legged_transition00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 legged;

# # wait;
export DATASET_LOCATION=/media/wz/2C96A0A60155E8F8/Dataset/FusionPortableV2/rosbag/ugv/
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_campus00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_parking00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_parking01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_parking02 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_parking03 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;

# # 以下序列没有wheel encoder
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_campus01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_transition00 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;
rosrun lvimesher run_one_fusionportable.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR ugv_transition01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1 ugv;

