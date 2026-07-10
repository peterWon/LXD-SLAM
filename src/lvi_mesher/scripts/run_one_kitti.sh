#!/bin/bash

export EPOC_DIR=$1;
export DATASET_LOCATION=$2;
export SEQUENCE=$3;
export CONFIG=$4;

echo SEQUENCE: $SEQUENCE
echo CONFIG: $CONFIG

export EXP_OUTPUT_DIR=$EPOC_DIR/${SEQUENCE};
mkdir -p $EXP_OUTPUT_DIR;
echo OUTPUT DIR: $EXP_OUTPUT_DIR;


mkdir -p $EXP_OUTPUT_DIR/ ;
roslaunch lvimesher offline_kitti.launch \
dataset_folder:=$DATASET_LOCATION \
sequence:=$SEQUENCE \
log_dir:=$EXP_OUTPUT_DIR \
config:=$CONFIG;