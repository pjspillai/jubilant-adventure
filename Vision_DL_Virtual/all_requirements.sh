#!/bin/bash

sudo apt-get install openmpi-bin

# OpenCV 3.4
#conda install -y -c forge conda-forge opencv
conda install -y -c anaconda opencv 

# PyTorch
conda install -y -c soumith pytorch

# MxNet
conda install -y mxnet

# Caffe2 (with CUDA 8 and CuDNN 7 support:)
conda install -y pytorch-nightly cuda80 -c pytorch

# Microsoft CNTK toolkit
pip3 install -U cntk-gpu

pip3 install --upgrade -U -r pip_requirements.txt

pip3 install --upgrade numpy Cython
