###############################################
# usage: ~/github/geek-guild/smalltrain/src/sh/operation.sh TRAIN_ID

TRAIN_ID=$1

# prepare envs

## activate venv
source ~/venv/venv_python3.6.0/bin/activate

## home dir
TSP_HOME=~/github/geek-guild/smalltrain/src
## data dir
DATA_HOME_DIR=/var/tensorflow/tsp
## log dir
LOG_DIR=$DATA_HOME_DIR/logs
mkdir -p $LOG_DIR
## tmp dir
mkdir -p /var/tmp/tsp/

# exec
JSON_SETTING_FILE_PATH=/var/tensorflow/tsp/operation/"$TRAIN_ID".json

cd $TSP_HOME
python smalltrain/model/operation.py --setting_file_path="$JSON_SETTING_FILE_PATH" > $LOG_DIR/$TRAIN_ID.log 2>&1
sh/after-operation.sh

###############################################
