#!/bin/bash
set -e  # fail and exit on any command erroring
set -x  # print evaluated commands

sudo apt-get -y install cmake

# Create virtual env
python3 -m venv ./env
source env/bin/activate

# Run cuQuantum tests.
echo ${PWD}
ls -al
export CUQUANTUM_DIR=${PWD}/cuquantum
export CUQUANTUM_ROOT=${CUQUANTUM_DIR}
export LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${LD_LIBRARY_PATH}
make run-custatevec-tests
