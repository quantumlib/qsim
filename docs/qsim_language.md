# qsim Circuit Description Language

This document describes the circuit description format for the `qsim` application (`qsim/qsim/`). A subset of this format is supported by the example applications in `qsim/apps/` (where noise, variables, and classical control structures are ignored).

## Expressions, Variables, Constants, and Scopes

### Expressions

Circuit descriptions can include expressions for gate parameters, loop/branch conditions, and variable assignments. For more details, see [Expressions](classical_control.md%23Expressions).

### Delimiters & Comments

* Comments start with `#` and extend to the end of the line.
* Statements are delimited by semicolons `;` or newlines `\n`.

### Constants & Variables

Constants are read-only at runtime and evaluated during parsing. Variables are mutable and updated dynamically during simulation.

```text
# Constants
const int a = 2           # Integer scalar
const float b = 2.3 * a   # Floating-point scalar
const int c(3) = 1 a 3    # Integer vector of size 3 (c[0]=1, c[1]=a, c[2]=3)
const float d(3)          # Uninitialized float vector (defaults to 0.0, 1.0, 2.0)

# Variables
int i = 0                 # Integer variable
float x = 1.5             # Float variable
int arr(4) = 1 2 3 i + 1  # Integer vector

```

**Notes:**

* **Vector Initialization:** Expression elements on the right-hand side of a vector assignment are delimited by spaces (and can be enclosed in parentheses for readability). If a vector declaration omits an explicit assignment, its elements default to consecutive integers from `0` to `size - 1`.

### Scopes

Nested control blocks (`if`, `repeat`, `do ... while`) define isolated symbol scopes. Variable resolution traverses outward through parent scopes.

## Quantum Circuit Specification & Gates

### Number of Qubits

The first entry in a circuit definition specifies the total number of qubits. It is **optional** and must evaluate to a constant integer expression (which can reference existing integer constants in the symbol table).

```text
5      # Explicit total circuit qubits

```

How the parser determines qubit count:

* **If present:** The parser evaluates the expression, sets the total qubit count, and stores or updates `nq` in the symbol table.
* **If omitted:** The parser checks the symbol table for `nq`:
* **`nq` exists:** Sets the qubit count to the value of `nq`.
* **`nq` is missing:** Defaults the qubit count to `1` and inserts `nq = 1` into the symbol table.



### Optional Noise

Noise can be specified at the top of a circuit file—either as the **second entry** (immediately following the qubit count) or as the **first entry** (if the qubit count is omitted).

#### Syntax

```text
noise channel parameters

```

* **`channel`**: Name of the one-qubit noise channel.
* **`parameters`**: One or more space-separated floating-point expressions (can include existing constants in the symbol table).

#### Supported Channels

| Channel | Parameters | Description |
| --- | --- | --- |
| `amplitude_damp` | 1 | Amplitude damping |
| `phase_damp` | 1 | Phase damping |
| `phase_flip` | 1 | Phase flip |
| `bit_flip` | 1 | Bit flip |
| `depolarize` | 1 | Depolarization |
| `generalized_amplitude_damp` | 2 | Generalized amplitude damping |
| `asymmetric_depolarize` | 3 | Asymmetric depolarization |

#### Example

```text
noise amplitude_damp 0.01

```

When specified, the noise channel is automatically applied to **every qubit** after **every moment**.

### Supported Quantum Gates

Quantum gates are defined using the following format:

```text
[time] gate_name qubit_indices... [parameters...]

```

The optional `[time]` prefix defines the explicit moment/time step for the gate. If omitted, the time step is assigned automatically. Qubit indices can involve integer constant expressions, and gate parameters can involve runtime expressions (except for controlled gates).

The supported gates are listed below:

| Gate | Format | Example Usage |
| --- | --- | --- |
| Global Phase ($\phi$) | `p phi` | `p 3.14` |
| Hadamard | `h qi` | `h 0` |
| T | `t qi` | `t 1` |
| X | `x qi` | `x 2` |
| Y | `y qi` | `y 3` |
| Z | `z qi` | `z 4` |
| $\sqrt{X}$ | `x_1_2 qi` | `x_1_2 5` |
| $\sqrt{Y}$ | `y_1_2 qi` | `y_1_2 6` |
| $R_x(\phi) = \exp(-i \phi X / 2)$ | `rx qi phi` | `rx 7 0.79` |
| $R_y(\phi) = \exp(-i \phi Y / 2)$ | `ry qi phi` | `ry 8 1.05` |
| $R_z(\phi) = \exp(-i \phi Z / 2)$ | `rz qi phi` | `rz 9 0.79` |
| $R_{x,y}(\theta, \phi) = \exp(-i \phi[\cos(\theta) X + \sin(\theta) Y ] / 2)$ | `rxy qi theta phi` | `rxy 0 1.05 0.79` |
| $\sqrt{W} = \sqrt{i} R_{x,y}(\pi/4, \pi/2)$ | `hz_1_2 qi` | `hz_1_2 1` |
| S | `s qi` | `s 1` |
| CZ | `cz qi1 qi2` | `cz 2 3` |
| CNOT | `cnot qi1 qi2` (or `cx`) | `cnot 4 5` |
| SWAP | `sw qi1 qi2` | `sw 6 7` |
| iSWAP | `is qi1 qi2` | `is 6 7` |
| fSim($\theta, \phi$) | `fs qi1 qi2 theta phi` | `fs 6 7 3.14 1.57` |
| CPhase($\phi$) | `cp qi1 qi2 phi` | `cp 0 1 0.78` |
| Identity (1-qubit) | `id1 qi` | `id1 0` |
| Identity (2-qubit) | `id2 qi1 qi2` | `id2 0 1` |
| Measurement ($n$-qubit) | `m qi1 qi2 ... [tag]` | `m 0 1 2` or `m 0 1 2 m1` |
| Controlled Gate | `c cqi1 cqi2 ... gate` | `c 0 1 rx 4 0.5` |

For the measurement gate, gn optional measurement tag can be specified following the qubit indices. This tag serves as the measurement variable name and can be referenced in subsequent expressions.

```text
const int qs(nq)
m qs[2] m1
fs qs[0] qs[1] pi * m1 / 2 pi / 4

```

## Classical Control Constructs

Classical control flow in `qsim` consists of six core constructs (the `[time]` prefix is optional for all constructs):

### 1. Conditional Execution (`if / elsif / else`)

Branches execution based on boolean expressions. `elsif` and `else` blocks are optional; `elsif` cannot follow `else`.

```text
[time] if condition1
    # Executed if condition1 is true
elsif condition2
    # Executed if condition1 is false and condition2 is true
elsif condition3
    # Executed if condition1 is false and condition3 is true
# ...
else
    # Executed if all preceding conditions are false
end

```

### 2. Pre-Condition Loop (`repeat`)

Repeats the enclosed block while `condition` evaluates to `true` (evaluated prior to each iteration).

```text
[time] repeat condition
    # Block executed while condition is true
end

```

### 3. Post-Condition Loop (`do ... while`)

Executes the enclosed block at least once, evaluating `condition` at the end of each iteration.

```text
[time] do
    # Block executed at least once
while condition

```

### 4. Runtime Variable Assignment

Mutates classical variables or vector elements during circuit execution.

```text
a = 2                        # Integer scalar assignment
b = 2.3 + a                  # Float scalar assignment
c(3) = 2 * a 3 * a 4 * a     # Vector assignment
vec[index] = expr            # Single-element update

```

### 5. Trajectory Filtering (`discard`)

Immediately halts and aborts the current quantum trajectory simulation if `condition` evaluates to `true`. Discarded trajectories are excluded from observable calculations.

```text
m 0 m1
discard m1 == 0

```

### 6. Console Debug Output (`println`)

Prints evaluated classical expressions or formatted strings (using C++20 formatting syntax) during simulation. Supports up to four expressions.

```text
println 'Hello, World!'
println "Index = {}, Value = {}" i (i * 2)

```

## Circuit Moments & Automatic Timing

Each operation is assigned an integer `time` tag specifying its execution moment:

1. **Explicit Timing:** Operations can be explicitly prefixed with a time tag (e.g., `0 h 0`).
2. **Automatic Timing:** If time tags are omitted, `qsim` assigns the earliest possible time step such that no qubit undergoes multiple operations within the same moment.
3. **Monotonic Ordering:** Time indices within a single scope must be non-decreasing.
4. **Control Structure Timing:** Automatic timing for `if`, `repeat`, and `do ... while` blocks accounts for measurement tags referenced in their conditional expressions.
5. **Control Scope Timing:** Time counters reset to `0` inside nested classical control blocks (`if`, `repeat`, `do ... while`) and resume tracking outer-scope time upon block exit.

### Gate Ordering Rules

Operations are executed in literal program order without reordering. In the following example:

* Both `h 0` and `h 1` are assigned time `0`.
* Measurement `m 0 m1` is assigned time `1`.
* The `if` statement and `y 1` gate are assigned time `2` (the `if` block and `y 1` do not swap, and `y 1` cannot share time `1` with the measurement).
* The `x 0` gate inside the `if` block resets to time `0` within its local scope.

```text
2
h 0
h 1
m 0 m1
if m1
 x 0
end
y 1

```

## Observables

### Measurement Histograms

Histograms accumulate output bitstring distributions across simulated trajectories:

```text
m 0 1 2 m1
histogram m1

```

* All measurement tags referenced in `histogram` statements must be unique, even across different scopes.
* Generating histograms for a large number of measured qubits may increase simulation runtime.

## Examples

### 1. Basic Output

```text
println 'Hello, World!'

```

### 2. Iterative Factorial Computation ($12!$)

```text
const int n = 12
int f = 1
int i = 1

repeat i <= n
  f = f * i
  i = i + 1
end

println f

```

### 3. Noisy Single-Qubit Circuit with Histogram

```text
1
noise amplitude_damp 0.1

x 0
m 0 m1

histogram m1

```