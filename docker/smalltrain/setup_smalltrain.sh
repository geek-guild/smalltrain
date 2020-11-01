#!/bin/bash

# SmallTrain setup script

# Make directories
mkdir -p /var/smalltrain/logs
mkdir -p /var/smalltrain/data
mkdir -p /var/smalltrain/operation
mkdir -p /var/smalltrain/model/image_recognition/tutorials/tensorflow/
mkdir -p /usr/local/etc/vendor/gg/smalltrain

# Install Python libraries
pip install --upgrade pip
pip install -r /var/work/requirements.txt

# Prepare log directory
# TODO change not to use this log directory
mkdir -p /var/smalltrain/src/log

# Distribute configs
cd /var/smalltrain/src/configs/
cp -rf smalltrain-conf.json /usr/local/etc/vendor/gg/smalltrain/smalltrain-conf.json
cp -rf jwt_secret.key /usr/local/etc/vendor/gg/jwt_secret.key
cp -rf redis_connection_setting.json /usr/local/etc/vendor/gg/redis_connection_setting.json

# Distribute tutorial operation setting file
cd /var/smalltrain/tutorials/
cp -rf image_recognition/operation/* /var/smalltrain/operation/

apt-get update

# Install SmallTrain
# You need git clone before exec docker
# ```
# $ mkdir -p ~/github/geek-guild/
# $ cd  ~/github/geek-guild/
# $ git clone -b release/v0.2.0 https://github.com/geek-guild/smalltrain.git
# ```
cd /var/smalltrain/src
pip install .

# Install GGUtils
# You need git clone before exec docker
# ```
# $ mkdir -p ~/github/geek-guild/
# $ cd  ~/github/geek-guild/
# $ git clone -b release/v0.0.3 https://github.com/geek-guild/ggutils.git
# ```
cd /var/ggutils
pip install .

# Install latest onnx-tensorflow(onnx-tf)
pip install --user https://github.com/onnx/onnx-tensorflow/archive/master.zip

# Prepare test data set for SmallTrain (CIFAR-10)
python /var/smalltrain/tutorials/image_recognition/convert_cifar_data_set.py

exit 0