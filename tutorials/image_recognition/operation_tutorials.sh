#!/bin/bash
###############################################
# Tutorial script to train CIFAR-10 image recognition with small data
# usage:
# OPERATION_ID=IR_2D_CNN_V2_l49-c64_20200109-TRAIN
# nohup /var/smalltrain/tutorials/image_recognition/operation_tutorials.sh $OPERATION_ID &

OPERATION_ID=$1

## home dir
export SMALLTRAIN_HOME=/var/smalltrain

## data dir
DATA_HOME_DIR=$SMALLTRAIN_HOME/data
## log dir
LOG_DIR=$SMALLTRAIN_HOME/logs
mkdir -p $LOG_DIR
## tmp dir
mkdir -p /var/tmp/tsp/

# Operation setting file
JSON_SETTING_FILE_PATH=/var/smalltrain/operation/"$OPERATION_ID".json

# Exec operation
cd $SMALLTRAIN_HOME/src
python smalltrain/model/operation.py --setting_file_path="$JSON_SETTING_FILE_PATH" > $LOG_DIR/$OPERATION_ID.log 2>&1
# If exec with nohup
# nohup python smalltrain/model/operation.py --setting_file_path="$JSON_SETTING_FILE_PATH" > $LOG_DIR/$OPERATION_ID.log 2>&1 &

# After-operation script (upload reports, shutdown, etc.) if needes.
# sh/after-operation.sh

###############################################
