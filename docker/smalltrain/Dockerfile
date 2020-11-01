# SmallTrain setup Dockerfile
# If using CPU
# FROM ubuntu:18.04
# If using NVIDIA GPU
FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

# exec ```sudo docker-compose up -d``` from ```smalltrain/docker``` root dir

MAINTAINER S.Hanamura <labo@geek-guild.jp>

RUN apt-get update
RUN apt-get install -y apt-utils vim curl less telnet git

# Prepare setup shell script
COPY setup_smalltrain.sh /var/work/setup_smalltrain.sh
RUN chmod +x /var/work/setup_smalltrain.sh

# Put requirements file for setup shell script
COPY requirements_nv.txt /var/work/requirements.txt

# Exec setup shell script in ENTRYPOINT
COPY entrypoint_smalltrain.sh /usr/local/bin/
COPY run_smalltrain.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh
ENTRYPOINT ["/usr/local/bin/entrypoint_smalltrain.sh"]
