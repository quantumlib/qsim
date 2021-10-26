# Usage of sample applications

qsim and qsimh are designed to be extensible to a variety of different
applications. The base versions of each are `qsim_base` and `qsimh_base`;
sample extensions are provided in
[apps](https://github.com/quantumlib/qsim/blob/master/apps). To compile the
code, just run `make qsim`. Binaries of the form `qsim(h)_*.x` will be added
to the `apps` directory.

Sample circuits are provided in
[circuits](https://github.com/quantumlib/qsim/blob/master/circuits).

## qsim_base usage

```
./qsim_base.x -c circuit_file -d maxtime -t num_threads -f max_fused_size -v verbosity -z
```

| Flag | Description |
|-------|------------|
|`-c circuit_file` | circuit file to run|
|`-d maxtime` | maximum time |
|`-t num_threads` | number of threads to use|
|`-f max_fused_size` | maximum fused gate size|
|`-v verbosity` | verbosity level (0,1,2,3,4,5)|
|`-z` | set flush-to-zero and denormals-are-zeros MXCSR control flags|

qsim_base computes all the amplitudes and just prints the first eight of them
(or a smaller number for 1- or 2-qubit circuits).

Verbosity levels are described in the following table.

| Verbosity level | Description |
|-----------------|-------------|
| 0 | no additional information|
| 1 | add total simulation runtime|
| 2 | add initialization runtime and fuser runtime|
| 3 | add basic fuser statistics|
| 4 | add simulation runtime for each fused gate|
| 5 | additional fuser information (qubit indices for each fused gate)|

Example:
```
./qsim_base.x -c ../circuits/circuit_q24 -d 16 -t 8 -v 1
```

## qsim_von_neumann usage

```
./qsim_von_neumann.x -c circuit_file -d maxtime -t num_threads -f max_fused_size -v verbosity -z
```


| Flag | Description |
|-------|------------|
|`-c circuit_file` | circuit file to run|
|`-d maxtime` | maximum time |
|`-t num_threads` | number of threads to use|
|`-f max_fused_size` | maximum fused gate size|
|`-v verbosity` | verbosity level (0,1,2,3,4,5)|
|`-z` | set flush-to-zero and denormals-are-zeros MXCSR control flags|

qsim_von_neumann computes all the amplitudes and calculates the von Neumann
entropy. Note that this can be quite slow for large circuits and small thread
numbers as the calculation of logarithms is slow.

Example:
```
./qsim_von_neumann.x -c ../circuits/circuit_q24 -d 16 -t 4 -v 1
```

## qsim_amplitudes usage

```
./qsim_amplitudes.x -c circuit_file \
                    -d times_to_save_results \
                    -i input_files \
                    -o output_files \
                    -f max_fused_size \
                    -t num_threads -v verbosity -z
```

| Flag | Description |
|-------|------------|
|`-c circuit_file` | circuit file to run|
|`-d times_to_save_results` | comma-separated list of circuit times to save results at|
|`-i input_files` | comma-separated list of bitstring input files|
|`-o output_files` | comma-separated list of amplitude output files|
|`-t num_threads` | number of threads to use|
|`-f max_fused_size` | maximum fused gate size|
|`-v verbosity` | verbosity level (0,1,2,3,4,5)|
|`-z` | set flush-to-zero and denormals-are-zeros MXCSR control flags|

qsim_amplitudes reads input files of bitstrings, computes the corresponding
amplitudes at specified times and writes them to output files.

Bitstring files should contain bitstings (one bitstring per line) in text
format.

Example:
```
./qsim_amplitudes.x -c ../circuits/circuit_q24 -t 4 -d 16,24 -i ../circuits/bitstrings_q24_s1,../circuits/bitstrings_q24_s2 -o ampl_q24_s1,ampl_q24_s2 -v 1
```

## qsim_qtrajectory_cuda usage

```
./qsim_qtrajectory_cuda.x -c circuit_file \
                          -d times_to_calculate_observables \
                          -a amplitude_damping_const \
                          -p phase_damping_const \
                          -t traj0 -n num_trajectories \
                          -f max_fused_size \
                          -v verbosity
```

| Flag | Description |
|-------|------------|
|`-c circuit_file` | circuit file to run|
|`-d times_to_calculate_observables` | comma-separated list of circuit times to calculate observables at|
|`-a amplitude_damping_const` | amplitude damping constant |
|`-p phase_damping_const` | phase damping constant |
|`-t traj0` | starting trajectory |
|`-n num_trajectories ` | number of trajectories to run starting with `traj0` |
|`-f max_fused_size` | maximum fused gate size|
|`-v verbosity` | verbosity level (0,1,2,3,4,5)|

qsim_qtrajectory_cuda runs on GPUs. qsim_qtrajectory_cuda performs quantum
trajactory simulations with amplitude damping and phase damping noise channels.
qsim_qtrajectory_cuda calculates observables (operator X at each qubit) at
specified times.

Example:
```
./qsim_qtrajectory_cuda.x -c ../circuits/circuit_q24 -d 8,16,32 -a 0.005 -p 0.005 -t 0 -n 100 -f 4 -v 0
```

## qsimh_base usage

```
./qsimh_base.x -c circuit_file \
               -d maxtime \
               -k part1_qubits \
               -w prefix \
               -p num_prefix_gates \
               -r num_root_gates \
               -t num_threads -v verbosity -z
```

| Flag | Description |
|-------|------------|
|`-c circuit_file` | circuit file to run|
|`-d maxtime` | maximum time |
|`-k part1_qubits` |  comma-separated list of qubit indices for part 1|
|`-w prefix`| prefix value |
|`-p num_prefix_gates` | number of prefix gates|
|`-r num_root_gates` | number of root gates|
|`-t num_threads` | number of threads to use|
|`-v verbosity` | verbosity level (0,1,4,5)|
|`-z` | set flush-to-zero and denormals-are-zeros MXCSR control flags|

qsimh_base just computes and just prints the first eight amplitudes. The hybrid
Schrödinger-Feynman method is used. The lattice is split into two parts.
A two level checkpointing scheme is used to improve performance. Say, there
are `N` gates on the cut. We split those into three parts: `p+r+s=N`, where
`p` is the number of "prefix" gates, `r` is the number of "root" gates and
`s` is the number of "suffix" gates. The first checkpoint is executed after
applying all the gates up to and including the prefix gates and the second
checkpoint is executed after applying all the gates up to and including the
root gates. The full summation over all the paths for the root and suffix gates
is performed.

The path for the prefix gates is specified by `prefix`. It is just a value of
bit-shifted path indices in the order of occurrence of prefix gates in the
circuit file. This is primarily used for distributed execution - see the
`Distributed execution` section below for more details.

Example (running on one machine):
```
./qsimh_base.x -c ../circuits/circuit_q30 -d 16 \
               -k 0,1,2,6,7,8,12,13,14,18,19,20,24,25,26 \
               -t 8 -w 0 -p 0 -r 5 -v 1
```

### Choosing flag values for qsimh

**-k** defines how the lattice will be split up. In the examples above, the
lattice has the structure below (cuts are denoted by the `|` symbol):

```
 0    1    2 |  3    4    5

 6    7    8 |  9   10   11

12   13   14 | 15   16   17

18   19   20 | 21   22   23

24   25   26 | 27   28   29
```

Deciding which cuts are optimal for a given circuit is computationally hard.
However, splitting the grid into roughly equal parts with the fewest cuts
possible (as is done for the lattice above) produces a circuit that performs
reasonably well in most cases.

The runtime of an execution is heavily influenced by **-p**, as there is no
summation over the "prefix" gates. The unique "prefix" path is specified by
**-w**; see the "Distributed execution" section below for details on this.

**-r** implicitly specifies the number of the "suffix" gates: the total number
of gates on the cut minus the values specified by **-p** and **-r**. For
performance, the "suffix" gates should typically be the gates on the cut with
maximum "time".


## qsimh_amplitudes usage
```
./qsimh_amplitudes.x -c circuit_file \
                     -d maxtime \
                     -k part1_qubits \
                     -w prefix \
                     -p num_prefix_gates \
                     -r num_root_gates \
                     -i input_file -o output_file \
                     -t num_threads -v verbosity -z
```

| Flag | Description |
|-------|------------|
|`-c circuit_file` | circuit file to run|
|`-d maxtime` | maximum time |
|`-k part1_qubits` | comma-separated list of qubit indices for part 1|
|`-w prefix`| prefix value |
|`-p num_prefix_gates` | number of prefix gates|
|`-r num_root_gates` | number of root gates|
|`-i input_file` | bitstring input file|
|`-o output_file` | amplitude output file|
|`-t num_threads` | number of threads to use|
|`-v verbosity` | verbosity level (0,1,4,5)|
|`-z` | set flush-to-zero and denormals-are-zeros MXCSR control flags|

qsimh_amplitudes reads the input file of bitstrings, computes the corresponding
amplitudes and writes them to the output file. The hybrid Schrödinger-Feynman
method is used, see above.

Bitstring files should contain bitstrings (one bitstring per line) in text
format.

Example (do not execute - see below):
```
./qsimh_amplitudes.x -c ../circuits/circuit_q40 -d 47 -k 0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,23,24 -t 8 -w 0 -p 0 -r 13 -i ../circuits/bitstrings_q40_s1 -o ampl_q40_s1 -v 1
```

This command could take weeks to run, since parallelism on a single machine is
limited by the -t flag and the available cores on the device. For large
circuits like this, distributed execution is recommended.

### Distributed execution

By setting -p to be greater than zero, the workload of qsimh_amplitudes can be
distributed across multiple machines. Each machine should use the same
arguments to `./qsimh_amplitudes.x`, with the exception of the -w flag, which specifies the path that machine will evaluate.

Example:
```
# Machine 1
./qsimh_amplitudes.x -c ../circuits/circuit_q40 -d 47 -k 0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,23,24 -t 8 -w 0 -p 9 -r 4 -i ../circuits/bitstrings_q40_s1 -o ampl_q40_s1_w0 -v 1

# Machine 2
./qsimh_amplitudes.x -c ../circuits/circuit_q40 -d 47 -k 0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,23,24 -t 8 -w 1 -p 9 -r 4 -i ../circuits/bitstrings_q40_s1 -o ampl_q40_s1_w1 -v 1

# ...additional executions...
```

Each execution above computes a portion of the overall amplitude for the
specified bitstrings. Summing across these results will give the final
amplitudes, with fidelity dependent on the number of paths executed.
