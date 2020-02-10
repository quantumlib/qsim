# qsim and qsimh

Quantum circuit simulators qsim and qsimh. These simulators were used for cross
entropy benchmarking in
[[1]](https://www.nature.com/articles/s41586-019-1666-5).

[[1]](https://www.nature.com/articles/s41586-019-1666-5), F. Arute et al,
"Quantum Supremacy Using a Programmable Superconducting Processor",
Nature 574, 505, (2019).

## qsim

qsim is a Schrödinger full state-vector simulator. It computes all the
![2^n](https://render.githubusercontent.com/render/math?math=2%5En)
amplitudes of the state vector, where
![n](https://render.githubusercontent.com/render/math?math=n) is the number of
qubits. Essentially, the simulator performs matrix-vector multiplications
repeatedly. One matrix-vector multiplication corresponds to applying one gate.
The total runtime is proportional to
![g2^n](https://render.githubusercontent.com/render/math?math=g2%5En), where
![g](https://render.githubusercontent.com/render/math?math=g) is the number of
2-qubit gates. To speed up the simulator, we use gate fusion
[[2]](https://arxiv.org/abs/1601.07195), single precision arithmetic, AVX/FMA
instructions for vectorization and OpenMP for multi-threading.

[[2]](https://arxiv.org/abs/1601.07195) M. Smelyanskiy, N. P. Sawaya,
A. Aspuru-Guzik, "qHiPSTER: The Quantum High Performance Software Testing
Environment", arXiv:1601.07195 (2016).

## qsimh

qsimh is a hybrid Schrödinger-Feynman simulator
[[3]](https://arxiv.org/abs/1807.10749). The lattice is split into two parts
and the Schmidt decomposition is used to decompose 2-qubit gates on the
cut. If the Schmidt rank of each gate is
![m](https://render.githubusercontent.com/render/math?math=m) and the number of gates on
the cut is
![k](https://render.githubusercontent.com/render/math?math=k) then there are
![m^k](https://render.githubusercontent.com/render/math?math=m%5Ek) paths. To
simulate a circuit with fidelity one, one needs to simulate all the
![m^k](https://render.githubusercontent.com/render/math?math=m%5Ek) paths and
sum the results. The total runtime is proportional to
![(2^{n_1} + 2^{n_2})m^k](https://render.githubusercontent.com/render/math?math=(2%5E%7Bn_1%7D%20%2B%202%5E%7Bn_2%7D)m%5Ek)
, where
![n_1](https://render.githubusercontent.com/render/math?math=n_1) and
![n_2](https://render.githubusercontent.com/render/math?math=n_2) are the qubit
numbers in the first and second parts. Path simulations are independent of each
other and can be trivially parallelized to run on supercomputers or in data
centers. Note that one can run simulations with fidelity
![F < 1](https://render.githubusercontent.com/render/math?math=F%20%3C%201)
just by summing over a fraction
![F](https://render.githubusercontent.com/render/math?math=F) of all the paths.

A two level checkpointing scheme is used to improve performance. Say, there
are ![k](https://render.githubusercontent.com/render/math?math=k) gates on the
cut. We split those into three parts:
![p+r+s=k](https://render.githubusercontent.com/render/math?math=p%2Br%2Bs%3Dk)
, where
![p](https://render.githubusercontent.com/render/math?math=p) is the number of
"prefix" gates,
![r](https://render.githubusercontent.com/render/math?math=r) is the number of
"root" gates and
![s](https://render.githubusercontent.com/render/math?math=s) is the number of
"suffix" gates. The first checkpoint is executed after applying all the gates
up to and including the prefix gates and the second checkpoint is executed
after applying all the gates up to and including the root gates. The full
summation over all the paths for the root and suffix gates is performed.

If ![p > 0](https://render.githubusercontent.com/render/math?math=p%20%3E%200)
then one such simulation gives
![F\approx 1/m^p](https://render.githubusercontent.com/render/math?math=F%5Capprox%201%2Fm%5Ep)
(for all the prefix gates having the same Schmidt rank m). One needs to run
![m^p](https://render.githubusercontent.com/render/math?math=m%5Ep)
simulations with different prefix paths and sum the results to get
![F = 1](https://render.githubusercontent.com/render/math?math=F%20%3D%201).

[[3]](https://arxiv.org/abs/1807.10749) I. L. Markov, A. Fatima, S. V. Isakov,
S. Boixo, "Quantum Supremacy Is Both Closer and Farther than It Appears",
arXiv:1807.10749 (2018).

## C++ Usage

The code is basically designed as a library. The user can modify sample
aplications in [apps](apps) to meet their own needs. The usage of sample
applications is described in [docs](docs/usage.md).

### Input format

Circuit input format is described in [docs](docs/input_format.md).

### Sample Circuits

A number of sample circuits are provided in
circuits.

### Unit tests

Unit tests are located in [tests](tests). The Google test framework is used.
Paths to Google test include files and to Google test library files should be
edited in [tests/make.sh](tests/make.sh).

## Cirq Usage

[Cirq](https://github.com/quantumlib/cirq) is a framework for modeling and
invoking Noisy Intermediate Scale Quantum (NISQ) circuits.

To run qsim on Google Cirq circuits, or just to call the simulator from Python,
see [docs](docs/cirq_interface.md).

## Authors

Sergei Isakov (Google): qsim and qsimh simulators, Vamsi Krishna Devabathini
(Google): Cirq interface.

## Disclaimer

This is not an officially supported Google product.
