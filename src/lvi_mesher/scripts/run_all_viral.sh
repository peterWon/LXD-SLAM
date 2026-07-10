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
export EPOC_DIR=/media/wz/2C96A0A60155E8F8/Dataset/NTU-VIRAL/LOG/LXD-SLAM-l-worot/
export DATASET_LOCATION=/media/wz/2C96A0A60155E8F8/Dataset/NTU-VIRAL/

if mkdir -p "$EPOC_DIR" 2>/dev/null; then
    echo "Result directory created (exists): $EPOC_DIR"
else
    echo "ERROR: Failed to create $EPOC_DIR"
    exit 1
fi


export CAPTURE_SCREEN=false;
export LOG_DATA=true;

# wait;

rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR eee_01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR eee_02 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR eee_03 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;

rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR nya_01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR nya_02 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR nya_03 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;

rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR sbs_01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR sbs_02 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR sbs_03 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;

# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR rtp_01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR rtp_02 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR rtp_03 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;

# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR spms_01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR spms_02 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR spms_03 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;

# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR tnp_01 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR tnp_02 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;
# rosrun lvimesher run_one_viral.sh $EPOC_DIR $DATASET_LOCATION $PACKAGE_DIR tnp_03 $CAPTURE_SCREEN $LOG_DATA 450 0 0 0.75 -1;

# wait;
# poweroff
