# Usage

qsim and qsimh can be used in many different ways. Sample applications are
provided in [apps](apps). To compile the codes, just run

```
$ g++ -O3 -mavx2 -mfma -fopenmp -lgomp -o qsim1 qsim1.cc
$ g++ -O3 -mavx2 -mfma -fopenmp -lgomp -o qsim2 qsim2.cc
$ g++ -O3 -mavx2 -mfma -fopenmp -lgomp -o qsim3 qsim3.cc
$ g++ -O3 -mavx2 -mfma -fopenmp -lgomp -o qsimh1 qsimh1.cc
$ g++ -O3 -mavx2 -mfma -fopenmp -lgomp -o qsimh2 qsimh2.cc
```

Sample circuits are provided in [circuits](circuits).

## qsim1 usage

```
./qsim1 -c circuit_file -d maxtime -t num_threads -v verbosity
```

`circuit_file` - circuit file to run
`maxtime` - maximum time
`num_threads` - number of threads to use
`verbosity` - verbosity level

qsim1 computes all the amplitudes and just prints the first eight of them
(or a smaller number for 1- or 2-qubit circuits).

Example:
```
./qsim1 -c ../circuits/circuit_q30 -d 16 -t 8 -v 1
```

Note that this particular simulation requires 8 GB of RAM.

## qsim2 usage

```
./qsim2 -c circuit_file -d maxtime -t num_threads -v verbosity
```

`circuit_file` - circuit file to run
`maxtime` - maximum time
`num_threads` - number of threads to use
`verbosity` - verbosity level

qsim2 computes all the amplitudes and calculates the von Neumann entropy.
Note that this can be quite slow for large circuits and small thread numbers
as the calculation of logarithms is slow.

Example:
```
./qsim2 -c ../circuits/circuit_q24 -d 16 -t 4 -v 1
```

## qsim3 usage

```
./qsim3 -c circuit_file -d times_to_save_results -i input_files -o output_files -t num_threads -v verbosity
```

`circuit_file` - circuit file to run
`times_to_save_results` - comma-separated list of circuit times to save results at
`input_files` - comma-separated list of bitstring input files
`output_files` - comma-separated list of amplitude output files
`num_threads` - number of threads to use
`verbosity` - verbosity level

qsim3 reads input files of bitstrings, computes the corresponding amplitudes
at specified times and writes them to output files.

Bitstring files should contain bitstings (one bitstring per line) in text
format.

Example:
```
./qsim3 -c ../circuits/circuit_q24 -t 4 -d 16,24 -i ../circuits/bitsrtings_q24_s1,../circuits/bitsrtings_q24_s2 -o ampl_q24_s1,ampl_q24_s2 -v 1
```

## qsimh1 usage

```
./qsimh1 -c circuit_file -d maxtime -k part1_qubits -w prefix -p num_prefix_gates -r num_root_gates -t num_threads -v verbosity
```

`circuit_file` - circuit file to run
`maxtime` - maximum time
`part1_qubits` - comma-separated qubit indices for part 1
`prefix` - prefix value
`num_prefix_gates` - number of prefix gates
`num_root_gates` - number of root gates
`num_threads` - number of threads to use
`verbosity` - verbosity level

qsimh1 just computes and just prints the first eight amplitudes. The hybrid
Schrödinger-Feynman method is used. The lattice is split into two parts.
A two level checkpointing scheme is used to improve performance. Say, there
are $$k$$ gates on the cut. We split those into three parts: $$p+r+s=k$$, where
$$p$$ is the number of "prefix" gates, $$r$$ is the number of "root" gates and
$$s$$ is the number of "suffix" gates. The first checkpoint is executed after
applying all the gates up to and including the prefix gates and the second
checkpoint is executed after applying all the gates up to and including the
root gates. The full summation over all the paths for the root and suffix gates
is performed. The path for the prefix gates is specified by `prefix`. It is
just a value of bit-shifted path indices in the order of occurrence of prefix
gates in the circuit file.

Example:
```
./qsimh1 -c ../circuits/circuit_q30 -d 16 -k 0,1,2,6,7,8,12,13,14,18,19,20,24,25,26 -t 8 -w 0 -p 0 -r 5 -v 1
```

## qsimh2 usage
```
./qsimh2 -c circuit_file -d maxtime -k part1_qubits -w prefix -p num_prefix_gates -r num_root_gates -i input_file -o output_file -t num_threads -v verbosity
```

`circuit_file` - circuit file to run
`maxtime` - maximum time
`part1_qubits` - comma-separated list of qubit indices for part 1
`prefix` - prefix value
`num_prefix_gates` - number of prefix gates
`num_root_gates` - number of root gates
`input_file` - bitstring input file
`output_file` - amplitude output file
`num_threads` - number of threads to use
`verbosity` - verbosity level

qsimh2 reads the input file of bitstrings, computes the corresponding
amplitudes and writes them to the output file. The hybrid Schrödinger-Feynman
method is used, see above.

Bitstring files should contain bitstings (one bitstring per line) in text
format.

Example:
```
./qsimh2 -c ../circuits/circuit_q40 -d 47 -k 0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,23,24 -t 8 -w 0 -p 9 -r 4 -i ../circuits/bitsrtings_q40_s1 -o ampl_q40_s1_w0 -v 1
./qsimh2 -c ../circuits/circuit_q40 -d 47 -k 0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,23,24 -t 8 -w 1 -p 9 -r 4 -i ../circuits/bitsrtings_q40_s1 -o ampl_q40_s1_w1 -v 1
...
```

# Old computers

SimulatorBasic can be used on old computers that do not support AVX
instructions. In this case, simulator_basic.h should be included instead of
simulator_avx.h and SimulatorAVX should be replaced by
SimulatorBasic<ParallelFor, float>.
