#!/bin/env bash

# exit on error
#set -e

yum update -y

##############################################################
# System things needed.
##############################################################

yum install -y \
      hwloc-devel \
      libedit-devel \
      libical-devel \
      libtool \
      ncurses-devel \
      openssl-devel \
      python3-devel \
      yum-utils \
      rpm-build 

##############################################################
# Get HTCondor installed.
##############################################################
echo "Installing HTCONDOR"
#yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
yum install -y https://research.cs.wisc.edu/htcondor/repo/8.8/el7/release/htcondor-release-8.8-1.el7.noarch.rpm
yum install -y condor

##############################################################
# Install CUDA
##############################################################
echo "Installing CUDA"
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
yum -y install nvidia-driver-latest-dkms cuda
yum -y install cuda-drivers

##############################################################
# Install gcc 8.3
##############################################################
echo "Installing GCC"
yum install -y centos-release-scl
yum install -y devtoolset-8-gcc devtoolset-8-gcc-c++
source scl_source enable devtoolset-8

##############################################################
# Build qsim
##############################################################
echo "Building QSIM"
export PATH=$PATH:/usr/local/cuda-11.4/bin/ # for some reason nvcc is not in the path. Addded.  Seems to work.
git clone https://github.com/quantumlib/qsim.git
cd qsim
gmake qsim
gmake qsim-cuda
gmake pybind

python3 -m pip install cmake
python3 -m pip install .

##############################################################
# Create Fluentd Config
##############################################################
cat <<EOF > condor.conf
<source>
  @type tail
  format none
  path /var/log/condor/*Log
  pos_file /var/lib/google-fluentd/pos/condor.pos
  read_from_head true
  tag condor
</source>
EOF

mkdir -p /etc/google-fluentd/config.d/
mv condor.conf /etc/google-fluentd/config.d/

if [ "$SERVER_TYPE" == "submit" ]; then
mkdir -p /var/log/condor/jobs
touch /var/log/condor/jobs/stats.log
chmod 666 /var/log/condor/jobs/stats.log
fi

systemctl restart google-fluentd

