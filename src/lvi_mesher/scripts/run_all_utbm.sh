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
export EPOC_DIR=//media/wz/LiDAR_DATA/dataset/UTBM/LOG/LXD-SLAM/


export CAPTURE_SCREEN=false;
export LOG_DATA=true;

# # wait;
export DATASET_LOCATION=/media/wz/LiDAR_DATA/dataset/UTBM/
rosrun lvimesher run_one_utbm.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR utbm_robocar_dataset_20180502_noimage1 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;

