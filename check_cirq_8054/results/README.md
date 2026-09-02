# Check of memory in qsim simulation before and after cirq PR 8054

2026-05-27

Smallest number of qubits causing OOM crash, checked with
`oom_qsim_simulation.py` on a Debian box with 65 GB of memory.

- baseline qsim c5bc02394a and cirq f4c35b02090b - OOM for 34 qubits
- patched qsim 7d05583a00e0 and cirq #8054 (bca9c5c617de) - OOM for 34 qubits

# 2026-05-27 - RUN LIST ------------------------------------------------------

Memory use during `qsim_simulation.py` was collected using
```
valgrind --tool=massif python qsim_simulation.py
```
The output massif.out.NNNNNN files were converted to memtotal.NNNNNN.dat
using `get-total-memory` script.

## baseline qsim and cirq ----------------------------------------------------

### nqubits=20 (repeated to check reproducibility)

- memtotal.733604.dat
- memtotal.736557.dat

### nqubits=24 (repeated to check reproducibility)

- memtotal.737548.dat
- memtotal.738662.dat

## patched qsim and cirq -----------------------------------------------------

### nqubits=20 (repeated to check reproducibility)

- memtotal.742794.dat
- memtotal.743107.dat

### nqubits=24 (repeated to check reproducibility)

- memtotal.743990.dat
- memtotal.745873.dat
