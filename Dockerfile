FROM	pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN 	apt-get update \
	&& apt-get -y install \
	apt-utils git vim openssh-server

RUN	DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN	pip install --upgrade pip
RUN	pip install setuptools
RUN 	sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR	/workspace
ADD	. .
ENV	PYTHONPATH $PYTHONPATH:/workspace

RUN	apt-get -y install libgl1-mesa-glx
RUN	apt-get install unzip
RUN	apt update

RUN 	chmod -R a+w /workspace
RUN	/bin/bash
