#!/bin/bash

export EPOC_DIR=$1;
export DATASET_LOCATION=$2;
export ROS_PKG_DIR=$3;
export EXP_NAME=$4;
export CAPTURE_SCREEN=$5;
export LOG_DATA=$6;
export LOG_DUR=$7;
export FUSE_UWB=$8;
export FUSE_VIS=$9;
export UWB_BIAS=${10};
export ANC_ID_MAX=${11};

export BAG_FILE=$DATASET_LOCATION/$EXP_NAME.bag;
export BAG_DUR=$(rosbag info $BAG_FILE | grep 'duration' | sed 's/^.*(//' | sed 's/s)//');
let LOG_DUR=BAG_DUR+20

echo "BAG DURATION:" $BAG_DUR "=> LOG_DUR:" $LOG_DUR;


export EXP_OUTPUT_DIR=$EPOC_DIR/${EXP_NAME};
mkdir -p $EXP_OUTPUT_DIR;
echo OUTPUT DIR: $EXP_OUTPUT_DIR;


mkdir -p $EXP_OUTPUT_DIR/ ;
roslaunch lvimesher offline_utbm.launch \
bag_filename:=$BAG_FILE \
log_dir:=$EXP_OUTPUT_DIR;