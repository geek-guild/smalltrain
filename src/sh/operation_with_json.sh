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
TRAIN_JSON_FILE_PATH=/var/tensorflow/tsp/operation/"$TRAIN_ID".json

cd $TSP_HOME
nohup ./sh/train_tsp_with_json.sh $TRAIN_JSON_FILE_PATH > $LOG_DIR/$TRAIN_ID.log 2>&1 &

# run tensorboard
# tensorboard --logdir=/var/tensorflow/tsp/logs &
## access to http://localhost:6006

# exit
###############################################
