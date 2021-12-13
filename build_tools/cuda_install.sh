#!/bin/bash
set -x

# install dependencies
kernel_version=$(uname -r)
yum clean all
yum install -y kernel
yum install -y kernel-devel-${kernel_version} kernel-headers-${kernel_version} pciutils

# install CUDA toolkit

yum install -y yum-utils epel-release
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum clean all
yum -y install nvidia-driver-latest-dkms cuda
yum -y install cuda-drivers

# post-install actions

# commented to allow outer PATH-setting
# export PATH=/usr/local/cuda-11/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH

# works here, does not escape to build process
nvcc --version
which nvcc
nvidia-smi

set +x
