# for GPU
# source activate tensorflow_p36
# source $HOME/.bashrc
# export CUDA_HOME=/usr/local/cuda-8.0
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"


TRAIN_ID=$1
DEFAULT_TRAIN_ID=default_train

# read param from json
DATA_HOME_DIR=/var/tensorflow/tsp/sample
JSON_FILE_PATH=$DATA_HOME_DIR/operation/$TRAIN_ID.json

if [ -z "$TRAIN_ID" ]; then
  echo "No TRAIN_ID set. We use DEFAULT_TRAIN_ID:"$DEFAULT_TRAIN_ID
  DEFAULT_JSON_FILE_PATH=/var/tensorflow/tsp/sample/operation/$DEFAULT_TRAIN_ID.json
  JSON_FILE_PATH=$DEFAULT_JSON_FILE_PATH
  TRAIN_ID=$DEFAULT_TRAIN_ID
fi

TMP=`cat $JSON_FILE_PATH`
JSON_PARAM=`echo "$TMP" | tr -d '\r' | tr -d '\n'`
echo "### JSON_PARAM ###"$JSON_PARAM

python smalltrain/model/nn_model.py --json_param="$JSON_PARAM"




sh/after-1001.sh
