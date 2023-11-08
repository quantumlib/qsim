# Support for AMD Instinct™ MI Series Accelerators

qsim provides support for AMD Instinct accelerators.
The implementation covers the native GPU support in qsim
by utilizing [HIP](https://rocm.docs.amd.com/projects/HIP)
(Heterogeneous-Compute Interface for Portability).
The cuQuantum implementation is currently not covered.

## Building

To enable support for AMD Instinct GPUs, qsim needs to be built from sources.
This can be done as follows:

```
conda env list
conda create -y -n CirqDevEnv python=3
conda activate CirqDevEnv
pip install pybind11

git clone https://github.com/quantumlib/qsim.git
cd qsim

make -j qsim      # to build CPU qsim
make -j qsim-hip  # to build HIP qsim
make -j pybind    # to build Python bindings
make -j cxx-tests # to build CPU tests
make -j hip-tests # to build HIP tests

pip install .
```

Note: To avoid problems when building qsim with support for AMD GPUs,
make sure to use the latest version of CMake.

## Testing

### Simulator

To test the qsim simulator:

```
make run-cxx-tests # to run CPU tests
make run-hip-tests # to run HIP tests
```

or

```
cd tests
for file in *.x; do ./"$file"; done          # to run all tests
for file in *_hip_test.x; do ./"$file"; done # to run HIP tests only
```

### Python Bindings

To test the Python bindings:

```
make run-py-tests
```

or

```
cd qsimcirq_tests
python3 -m pytest -v qsimcirq_test.py
```

## Using

Using qsim on AMD Instinct GPUs is identical to using it on NVIDIA GPUs.
I.e., it is done by passing `use_gpu=True` and `gpu_mode=0` as `qsimcirq.QSimOptions`:

```
simulator = qsimcirq.QSimSimulator(qsim_options=qsimcirq.QSimOptions(
        use_gpu=True,
        gpu_mode=0,
        ...
    ))
```

Note: `gpu_mode` has to be set to zero for AMD GPUs, as cuStateVec is not supported.
