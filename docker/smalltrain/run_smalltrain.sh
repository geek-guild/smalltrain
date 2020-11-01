#!/bin/bash

# Run tensorboard
mkdir -p  /var/data/smalltrain/results/logs/
nohup tensorboard --logdir /var/data/smalltrain/results/logs/ &

# Test operation
# 1. Debug with small data set
OPERATION_ID=IR_2D_CNN_V2_l49-c64_TUTORIAL-DEBUG-WITH-SMALLDATASET-20200708-TRAIN
echo "Exec operation id: ${OPERATION_ID}"
# For Sequential processing
/var/smalltrain/tutorials/image_recognition/operation_tutorials.sh $OPERATION_ID

# 2. Training example
# Training from restored model
OPERATION_ID=IR_2D_CNN_V2_l49-c64_TUTORIAL-20200708-TRAIN
# Zero base training
# OPERATION_ID=IR_2D_CNN_V2_l49-c64_TUTORIAL-ZEROBASE-TRAIN
echo "Exec operation id: ${OPERATION_ID}"
# For Parallel processing (Idling after starting operation)
nohup /var/smalltrain/tutorials/image_recognition/operation_tutorials.sh $OPERATION_ID &
