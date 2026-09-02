# qsim Classical Control

`qsim` natively supports classical control (feed-forward) in quantum circuits. Subcircuits and conditional gates can be executed dynamically based on boolean expressions that depend on classical variables, constants, and measurement outcomes. Both `while`-style and `do-while`-style loops are supported. In addition, quantum gate angles can be parameterized by arithmetic expressions evaluated at runtime.

## Variables & Symbols

`qsim` supports classical constants and variables (symbols) registered in a symbol table. Variables are assignable and updated at runtime. Supported types are:

* **`int`**: 64-bit signed integer.
* **`float`**: Double-precision floating-point number.
* **`vector<int>`** and **`vector<float>`**: Dynamic arrays of scalars.

Quantum measurement registers are represented as integers (`int`) and are updated dynamically via measurement operations.

## Expressions

Assignment statements, conditions, and gate parameterizations rely on expressions that compute numerical values from literals, classical symbols, and quantum measurement outcomes. There are two types of expressions:

* **Constant Expressions:** If an expression consists entirely of literal constants or symbols marked as constant, it is evaluated and folded **at compile time**, producing a scalar literal.
* **Runtime Expressions:** If an expression involves mutable variables or measurement results, the parser yields an executable closure evaluated dynamically during execution.

All constants, variables, and measurement outcomes referenced in expressions must be defined in the symbol table.

### Types & Implicit Conversions

`qsim` expressions support three evaluation types: `int`, `float`, and `bool`. Type conversions adhere to the following rules:

* **`bool` -> `int`:** `false` becomes `0`, `true` becomes `1`.
* **`int` -> `bool`:** `0` becomes `false`, any non-zero value becomes `true`.
* **`int` / `bool` -> `float`:** Implicitly promoted when floating-point operations or operands are present.
* **`float` -> `int` / `bool`:** **Forbidden.** Floating-point values cannot be implicitly narrowed to integers or booleans.

### Constant Literals

Integer literals can be specified in decimal (`42`), hexadecimal (`0x2A`), or binary (`0b101010`) format. Floating-point literals must follow standard decimal notation (e.g., `1.23` or `1.2e-3`).

### Indexing

Individual elements of vectors and specific bits of measurement registers are accessed using bracket notation:

```text
vec[index]
m1[bit_index]

```

* **Unindexed Registers:** Unindexed measurement registers evaluate to their full integer bitstring value (e.g., if qubits measure $\vert{}11\rangle$, `m1` evaluates to `3`).
* **Index Types:** Any expression convertible to `int` is a valid index.
* **Bounds Checking:** Occurs at compile time for constant indices and at runtime for dynamic indices.

### Operators & Precedence

Operators are listed below in **decreasing order of precedence**:

| Operator | Type | Description | Type Rules / Restrictions |
| --- | --- | --- | --- |
| `!` | Logical | Unary NOT | Requires operand convertible to `bool`. |
| `-` | Arithmetic | Unary Negation | Preserves `int` or `float`. |
| `~` | Bitwise | Unary Bitwise NOT | Inverts register bits for measurements. |
| `**` | Arithmetic | Power ($a^b$) | Always returns `float`. |
| `*`, `/`, `%` | Arithmetic | Multiplication, Division, Modulo | `%` requires integer operands. |
| `+`, `-` | Arithmetic | Addition, Subtraction | Mixed types promoted (`int` + `float` -> `float`). |
| `<<`, `>>` | Bitwise | Left / Right Shift | Requires integer operands. |
| `<`, `<=`, `>`, `>=` | Comparison | Relational | Yields `bool`. |
| `==`, `!=` | Comparison | Equality | Yields `bool`. |
| `&` | Bitwise | Bitwise AND | Requires integer operands. |
| `^` | Bitwise | Bitwise XOR | Requires integer operands. |
| `|` | Bitwise | Bitwise OR | Requires integer operands. |
| `&&` | Logical | Logical AND | Requires operands convertible to `bool`. |
| `^^` | Logical | Logical XOR | Requires operands convertible to `bool` (low-precedence version of `!=`). |
| `||` | Logical | Logical OR | Requires operands convertible to `bool`. |

Parentheses (`(` and `)`) can be used to override default operator precedence, e.g., `(2 + 3) * 4`.

### Parser Stop Rules

The expression parser consumes tokens continuously until operators are exhausted. For example, in the string `"2 * i  3 * j"`, a single parser call parses `2 * i` and stops before token `3`.

### Integer Expression Parser

A specialized parser variant handles integer and boolean expressions exclusively. It throws an error if it encounters a floating-point number or the power operator (`**`).

### Expression Examples

```text
2 * 3 + 4
0x0F & 0b1010
3.14159 * 2.0

m0[0] ^ m1[1]
(v[i] + 1) % 8

2.0 ** theta
phase + 3.14159 / 2.0

(m0 != 0) && (m1[0] == 1)

```

## C++ API Overview

`qsim` provides a unified C++20 API for quantum circuit execution with classical control. It supports dynamic feed-forward control flow, measurement-driven gate parameterization, runtime symbol assignments, and trajectory filtering.

### Classical Control Structures

Classical control flow in `qsim` consists of six primary construct types:

**1. Conditional Execution (`if / elsif / else`)**

Branches execution based on conditional boolean expressions evaluated at runtime:

```text
if condition1
    # Executed if condition1 is true
elsif condition2
    # Executed if condition1 is false and condition2 is true
else
    # Executed if all preceding conditions are false
end

```

**2. Pre-Condition Loop (`repeat`)**

Repeats the inner block while the loop condition evaluates to true (evaluated before each iteration):

```text
repeat condition
    # Block executed while condition is true
end

```

**3. Post-Condition Loop (`do ... while`)**

Executes the inner block at least once, evaluating the loop condition at the end of each iteration:

```text
do
    # Block executed at least once
while condition

```

**4. Runtime Symbol Assignment (`assign`)**

Mutates classical variables or vector elements during execution based on evaluated expressions. Supports scalar, vector, and indexed vector assignments.

**5. Quantum Trajectory Filtering (`discard`)**

Immediately halts and aborts the current quantum trajectory simulation if the specified condition evaluates to true.

**6. Debug Output (`println`)**

Prints formatted strings or evaluated classical expression values to `stdout` during circuit execution.

### C++ API Architecture & Types

The API abstracts circuit execution into variant-based operations and parameterized containers:

```text
                   ┌───────────────────────────────┐
                   │            Circuit            │
                   └───────────────┬───────────────┘
                                   │ ops
                                   ▼
                   ┌───────────────────────────────┐
                   │           Operation           │
                   └───────────────┬───────────────┘
                                   │ (variant)
    ┌────────────────┬─────────────┼───────────────────┬────────────────┐
    ▼                ▼             ▼                   ▼                ▼
┌───────┐  ┌──────────────┐ ┌─────────────┐ ┌────────────────────┐ ┌─────────┐
│ Gate  │  │ControlledGate│ │ Measurement │ │RuntimeResolvedGate │ │   CCO   │
└───────┘  └──────────────┘ └─────────────┘ └────────────────────┘ └─────────┘

```

*(Note: `CCO` stands for `ClassicallyControlledOperation`.)*

### Key API Components

`FP` below represents the floating-point precision type (typically `float`, or `double` depending on simulator configuration).

1. **`Circuit<Operation<FP>>`**: Top-level container representing a quantum program. Holds `num_qubits` and an ordered sequence of `Operation` instances.
2. **`Operation<FP>`**: `std::variant` type representing any executable circuit node:
* **`Gate<FP>`**: Standard fixed-matrix quantum gate (e.g., $H$, $X$, $CZ$).
* **`ControlledGate<FP>`**: Quantum gate controlled by specified control qubits (`controlled_by`).
* **`RuntimeResolvedGate<FP>`**: Dynamic gate whose parameters (`param_exprs`) depend on classical symbols or measurements, re-evaluated immediately prior to gate execution.
* **`Measurement`**: Measures a set of qubits at a given time step and writes results to the symbol table under `id`.
* **`Channel<FP>`**: Quantum noise channel modeled by a collection of `KrausOperator<FP>` objects.
* **`ClassicallyControlledOperation<FP>`**: Encapsulates classical control flow constructs, nested operations, and variable updates.
3. **`Symbol`**: Polymorphic container holding a single symbol.
4. **`SymbolTable`**: Scoped symbol table that holds measurement outcomes, constants, and variables.
5. **`Expr`**: Variant type holding compile-time constants or runtime-evaluated expressions.
6. **`ExprParser`**: Expression parser that parses an input string and returns an `Expr` instance.
7. **`QSimRunner<Fuser, RGen>`**: Main driver class executing quantum circuits (both clean and noisy) with optional classical control flow.