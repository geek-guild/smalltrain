# SmallTrain as standalone Docker container


# Table of Contents



*	[Install Pre-Requisites](#install-pre-requisites)
	*	[Install Docker package directly from Docker](#install-docker-package-directly-from-docker)
	*	[Install Docker-compose](#install-docker-compose)
	*	[Run docker](#run-docker)
*	[Setup Smalltrain](#setup-smalltrain)
	*	[Clone SmallTrain repository](#clone-smalltrain-repository)
	*	[Clone GGUtils repository](#clone-ggutils-repository)
	*	[Run docker image](#run-docker-image)
	*	[Login SmallTrain container](#login-smalltrain-container)
	*	[Logging](#logging)
*	[Miscellaneous](#miscellaneous)
	*	[Run custom tutorial operation on SmallTrain container](#run-custom-tutorial-operation-on-smalltrain-container)
	*	[Run TensorBoard](#run-tensorboard)
	*	[Check TensorBoard on your browser](#check-tensorboard-on-your-browser)
	*	[Check the result of the tutorial operation](#check-the-result-of-the-tutorial-operation)



## Install Pre-Requisites
Before getting started there are few pre-requisites that need to be satisfied. If its already satisfied you can skip to Setup Smalltrain section, please note all the text that are hightlighted like `this` explains where you should do the operation.


###	Install Docker package directly from Docker

`host machine by host sudoers`

```
$ sudo apt-get update -y
$ curl -sSL https://get.docker.com/ | sh
```
###	Install Docker-compose
`host machine by host sudoers`

```
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.22.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
```

### Run docker
`host machine by host sudoers`
```
$ sudo service docker start
```

## Setup Smalltrain

###	Clone SmallTrain repository

`host machine`
```
$ mkdir -p ~/github/geek-guild/
$ cd  ~/github/geek-guild/
$ git clone https://github.com/geek-guild/smalltrain.git


```

### Clone GGUtils repository
`host machine`
```
$ mkdir -p ~/github/geek-guild/
$ cd  ~/github/geek-guild/
$ git clone -b release/v0.0.3 https://github.com/geek-guild/ggutils.git

```

### Run docker image
`host machine`
```
# SmallTrain
$ cd ~/github/geek-guild/smalltrain/docker/
$ docker-compose up -d

```

### Login SmallTrain container
`host machine`
```
# Check CONTAINER ID of smalltrain container

$ docker ps -a
CONTAINER ID        IMAGE                     COMMAND                  CREATED             STATUS              PORTS                                              NAMES
YYYYYYYYYYYY        docker_smalltrain-redis   "docker-entrypoint.s…"   15 minutes ago      Up 15 minutes       0.0.0.0:6379->6379/tcp, 0.0.0.0:16379->16379/tcp   smalltrain-redis
XXXXXXXXXXXX        docker_smalltrain         "/usr/local/bin/entr…"   15 minutes ago      Up 15 minutes       0.0.0.0:6006->6006/tcp                             smalltrain

# Login the container
$ CONTAINER_ID=XXXXXXXXXXXX
$ docker exec -it $CONTAINER_ID /bin/bash
```
### Logging

`host machine`

```
$ CONTAINER_ID=XXXXXXXXXXXX
$ docker logs $CONTAINER_ID

...
Exec operation id: IR_2D_CNN_V2_l49-c64_20200109-TRAIN
nohup: appending output to 'nohup.out'

```

## Miscellaneous


### Run custom tutorial operation on SmallTrain container
`SmallTrain container`
```
$ OPERATION_ID=IR_2D_CNN_V2_l49-c64_20200109-TRAIN
$ /var/smalltrain/src/sh/operation_tutorials.sh $OPERATION_ID

```


### Run TensorBoard
`SmallTrain container`
```
$ nohup tensorboard --logdir /var/model/image_recognition/tutorials/tensorflow/logs/ &
```
### Check TensorBoard on your browser
`Browser`

```
http://<HOST_IP>:<TENSORBOARD_PORT>
```

### Check the result of the tutorial operation
`SmallTrain container`
```
# Report directory
$ ls -l /var/model/image_recognition/tutorials/tensorflow/report/IR_2D_CNN_V2_l49-c64_20200109-TRAIN/
total 424
-rw-r--r-- 1 root root  38074 Jan 10 08:17 all_variables_names.csv
-rw-r--r-- 1 root root  77687 Jan 10 08:19 prediction_e49_all.csv
-rw-r--r-- 1 root root  77687 Jan 10 08:17 prediction_e9_all.csv
-rw-r--r-- 1 root root     28 Jan 10 08:17 summary_layers_9.json
-rw-r--r-- 1 root root 109285 Jan 10 08:17 test_plot__.png
-rw-r--r-- 1 root root  54706 Jan 10 08:19 test_plot_e49_all.png
-rw-r--r-- 1 root root  55107 Jan 10 08:17 test_plot_e9_all.png
-rw-r--r-- 1 root root   6406 Jan 10 08:17 trainable_variables_names.csv

# Prediction after 49steps of training
$ less /var/model/image_recognition/tutorials/tensorflow/report/IR_2D_CNN_V2_l49-c64_20200109-TRAIN/prediction_e49_all.csv

DateTime,Estimated,MaskedEstimated,True
/var/data/cifar-10-image/test_batch/test_batch_i9_c1.png_0,1,0.0,1
/var/data/cifar-10-image/test_batch/test_batch_i90_c0.png_0,0,0.0,0
/var/data/cifar-10-image/test_batch/test_batch_i91_c3.png_0,6,0.0,3
/var/data/cifar-10-image/test_batch/test_batch_i92_c8.png_0,8,0.0,8

```

