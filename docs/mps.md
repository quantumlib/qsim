# The MPS Simulator

qsim includes a **Matrix Product State (MPS)** simulator alongside its standard
state-vector and hybrid Schrödinger-Feynman simulators. While the full
state-vector simulator (`qsim`) stores the entire quantum state in memory — which
grows exponentially with the number of qubits — the MPS simulator takes a
different approach that can be much more memory-efficient for certain kinds of
circuits.

## What is a Matrix Product State?

A Matrix Product State is a way of representing a quantum state as a chain of
tensors, one per qubit, connected together. Instead of storing one giant
exponentially-large vector, you store a sequence of small matrices. The "bond
dimension" (often written as χ or `bond_dim`) controls how much entanglement the
representation can capture: a higher bond dimension is more accurate but uses more
memory and takes longer to simulate.

The catch, and the reason this is so useful, is that many quantum circuits of
practical interest don't generate a lot of entanglement. For those circuits, a
small bond dimension works great, and you can simulate far more qubits than a
full state-vector simulator could handle.

The trade-off is that MPS simulation is **approximate for highly entangled circuits**.
When a gate creates entanglement that exceeds the bond dimension, the simulator
truncates it (using SVD). For low-entanglement circuits (e.g., 1D nearest-neighbor
circuits, QAOA with shallow depth), the results are exact or very close to exact.

## Where to find the implementation

The MPS simulator lives in two C++ header files:

- [`lib/mps_simulator.h`](https://github.com/quantumlib/qsim/blob/main/lib/mps_simulator.h)
  — the `MPSSimulator` class, which applies gates to an MPS.
- [`lib/mps_statespace.h`](https://github.com/quantumlib/qsim/blob/main/lib/mps_statespace.h)
  — the `MPSStateSpace` class, which manages MPS memory, sampling, and inner products.

Both live in the `qsim::mps` namespace.

## Using MPSStateSpace

`MPSStateSpace` handles everything related to creating and manipulating the MPS
object itself.

### Creating a state

```cpp
// Requires num_qubits >= 2 and bond_dim >= 2.
auto state = MPSStateSpace::Create(num_qubits, bond_dim);
```

### Initializing to |0⟩

```cpp
MPSStateSpace::SetStateZero(state);
```

### Copying a state

```cpp
MPSStateSpace::Copy(src_state, dest_state);
```

### Computing inner products

```cpp
// Full complex inner product <state1|state2>
auto ip = MPSStateSpace::InnerProduct(state1, state2);

// Real part only
auto rip = MPSStateSpace::RealInnerProduct(state1, state2);
```

### Sampling

```cpp
// Draw one sample
std::vector<bool> sample;
MPSStateSpace::SampleOnce(state, scratch, scratch2, &rng, &sample);

// Draw multiple samples
std::vector<std::vector<bool>> results(num_samples);
MPSStateSpace::Sample(state, scratch, scratch2, num_samples, seed, &results);
```

Note that sampling requires two additional scratch MPS objects of the same size
as the state. These are used as working memory during the sequential sampling
process.

### Reduced density matrices

You can compute the 2×2 reduced density matrix (1-RDM) for any single qubit:

```cpp
float rdm[8]; // 2x2 complex matrix = 8 floats
MPSStateSpace::ReduceDensityMatrix(state, scratch, qubit_index, rdm);
```

## Using MPSSimulator

`MPSSimulator` applies quantum gates to an MPS state.

### Applying gates

```cpp
MPSSimulator sim(/* ForArgs */);

// Apply a 1-qubit gate
sim.ApplyGate({qubit_index}, gate_matrix, state);

// Apply a 2-qubit gate (must be adjacent)
sim.ApplyGate({qubit_a, qubit_b}, gate_matrix, state);
```

When a 2-qubit gate is applied, the simulator:
1. Contracts the two neighboring MPS tensors into one combined tensor.
2. Applies the gate matrix.
3. Uses **Singular Value Decomposition (SVD)** to split the result back into two
   tensors.
4. Keeps only the top `bond_dim` singular values, truncating the rest.

This is the key step where approximation happens. If the true quantum state
needs more entanglement than `bond_dim` allows, some information is lost.

## Current limitations

The MPS simulator is actively developed but not yet complete. As of now:

- **Only 1-qubit and 2-qubit gates are supported.** Support for 3+ qubit gates
  is not yet implemented (commented placeholders exist in the source).
- **Controlled gates are not yet implemented.** `ApplyControlledGate` exists but
  is a stub (`// TODO`).
- **Expectation values are not yet implemented.** `ExpectationValue` currently
  returns a placeholder value of `(-10, -10)`.
- **No Python interface.** The MPS simulator is currently only accessible through
  C++. It is not yet exposed via the `qsimcirq` Python package.
- **2-qubit gates must act on adjacent qubits.** Non-adjacent 2-qubit gates
  require SWAP networks (not handled automatically).

## When to use MPS vs. state-vector simulation

| Situation | Recommended simulator |
|---|---|
| Circuit has low entanglement (1D, shallow QAOA, etc.) | MPS |
| You want exact results for any circuit | qsim (state-vector) |
| Many qubits, low depth | MPS |
| Deep random circuits | qsim or qsimh |
| You need a Python interface | qsim or qsimh (via qsimcirq) |

## Further reading

- [Vidal, G. "Efficient Simulation of One-Dimensional Quantum Many-Body Systems"](https://arxiv.org/abs/quant-ph/0310089)
  — the foundational paper on MPS simulation of quantum circuits.
- [qsim overview](./overview.md) — description of all simulators in qsim.
- [C++ template reference](./type_reference.md) — description of template
  parameters like `For`, `fp_type`, etc., used throughout qsim including MPS.
