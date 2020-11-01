# for GPU
# source activate tensorflow_p36
# source $HOME/.bashrc
# export CUDA_HOME=/usr/local/cuda-8.0
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"


JSON_FILE_PATH=$1

TMP=`cat $JSON_FILE_PATH`
JSON_PARAM=`echo "$TMP" | tr -d '\r' | tr -d '\n'`
echo "### JSON_PARAM ###"$JSON_PARAM

# python smalltrain/model/nn_model.py --json_param="$JSON_PARAM"
python smalltrain/model/operation.py --json_param="$JSON_PARAM"


sh/after-1001.sh
