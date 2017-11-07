#!/bin/bash

# install CUDA Toolkit v8.0
# instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb (network))
CUDA_REPO_PKG="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-get update
sudo apt-get -y install gcc cuda-8-0 python3-pip git

# install cuDNN v6.0
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

# set environment variables
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda-8.0/" >> ~/.bashrc
. ~/.bashrc

git clone https://github.com/jorditorresBCN/dlaimet.git
sudo pip3 install virtualenv
virtualenv venv-tf
echo "export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/" >> ~/venv-tf/bin/activate
. venv-tf/bin/activate
pip install --upgrade pip tensorflow-gpu==1.4.0 jupyter keras ipython matplotlib
echo ">>>> GENERATING JUPYTER CONFIG "
jupyter notebook --generate-config
echo ">>>> CREATE A PASSWORD FOR JUPYTER WEB "
jupyter notebook password
