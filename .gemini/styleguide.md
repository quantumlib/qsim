# qsim style guide

This style guide outlines the coding conventions for this project. There are
separate subsections for Python, C++, and other file types below.

## General guidance

### Overall principles

*   _Readability_: Code should be easy to understand for all team members.

*   _Maintainability_: Code should be easy to modify and extend.

*   _Consistency_: Adhering to a consistent style across all projects improves
    collaboration and reduces errors.

*   _Performance_: While readability is paramount, code should be efficient.

### Overall development approach

*   Use test-driven development (TDD) whenever possible. Write a failing test
    _before_ writing new code. This ensures testability and clarifies
    requirements.

*   Isolate tests. Tests must be independent and must not rely on the state of
    other tests. Use setup and teardown functions to create a clean environment
    for each test run.

*   When new functions, classes, and files are introduced, they should also have
    corresponding tests. Existing tests must continue to pass (or be updated)
    when changes are introduced, and code should be covered by tests.

*   Make sure to cover edge cases: Test for invalid inputs, null values, empty
    arrays, zero values, and off-by-one errors.

*   Use mocking for dependencies. In unit tests, external dependencies (e.g.,
    databases, network services, file system) must be mocked to ensure the test
    is isolated to the unit under test.

*   Use asserts intelligently. Test assertions should be specific. Instead of
    just asserting `true`, assert that a specific value equals an expected
    value. Provide meaningful failure messages.

*   Test coverage must be high. We don't require 100% coverage, but any
    uncovered code must be annotated with `# pragma: no cover`. To ignore
    coverage of a single line, place `# pragma: no cover` at the end of the
    line. To ignore coverage for an entire block, start the block with a `#
    pragma: no cover` comment on its own line.

### Overall code format conventions

This project generally follows Google coding conventions, with a few changes
that are mostly defined in the various configuration files at the top level of
the source tree. The following Google style guides are the starting points:

*   [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

*   [Google Python Style
    Guide](https://google.github.io/styleguide/pyguide.html)

*   [Google Markdown Style
    Guide](https://google.github.io/styleguide/docguide/style.html)

*   [Google Shell Style
    Guide](https://google.github.io/styleguide/shellguide.html)

To learn the conventions for line length, indentation, and other style
characteristics, please inspect the following configuration files (if present at
the top level of this project repository):

*   `.editorconfig` for basic code editor configuration for indentation and line
    length.

*   `.clang-format` for C++ code and also protobuf (Protocol Buffers) data
    structure definitions.

*   `.hadolint.yaml` for `Dockerfile`s.

*   `.jsonlintrc.yaml` for JSON files.

*   `.markdownlintrc` for Markdown files.

*   `.pylintrc` for Python code.

*   `.yamllint.yaml` for YAML files.

### Overall code commenting conventions

Every source file must begin with a header comment with the copyright and
license. We use the Apache 2.0 license, and copyright by Google LLC. Here is an
example of the required file header for a Python language code file:

```python
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

License headers are necessary on Python, C++, Bash/shell, and similar
programming language files, as well as configuration files in YAML, TOML, ini,
and other configuration file formats. They are not necessary in Markdown,
reStructuredText, or plain text files.

For comments in other parts of the files, follow these guidelines:

*   _Write clear and concise comments_: Comments must explain the "why", not the
    "what". The code itself shows what it's doing. The comments should explain
    the intent, the trade-offs, and the reasoning behind the implementation.

*   _Comment sparingly_: Well-written code should be self-documenting where
    possible. It's not necessary to add comments for code fragments that can
    reasonably be assumed to be self-explanatory.

*   _Use complete sentences_: Start comments with a capital letter and use
    proper punctuation.

### Overall nomenclature conventions

This project follows some nomenclature rules and conventions in order to
maintain a consistent interface that is easy to use. By using consistent naming,
we can reduce cognitive load on human users and developers. Please try to use
these terms when writing code.

*   Use `state_vector` to describe a pure state.  **Do not** use `wavefunction`,
    `wave_function`, or `state` for this object (`state` is too overloaded).

*   If the object is an array or possibly a computational basis state
    (given by an `int`), use `state_rep` or, if it is the initial state of
    a system `initial_state`.

*   A function argument (`state_vector`, `state_rep`, or `initial_state`)
    should permit any of the possible representations of a state: A NumPy
    array, a NumPy tensor, an integer representing a qubit-system's
    computational basis state, a sequence of _n_ integers representing a
    qudit's basis state, or a `cirq.ProductState`. The type annotation should
    be `cirq.STATE_VECTOR_LIKE` and you should use `cirq.to_valid_state_vector`
    to canonicalize as a NumPy array of amplitudes. If a function expects a
    NumPy array of amplitudes, its type annotation should be `np.ndarray`.

*   Use `density_matrix` to describe a mixed state that is passed in as a NumPy
    matrix or NumPy tensor.  **Do not** used `mixed_state`, `density_operator`,
    or `state`.

## Python-specific guidance

This section outlines the coding conventions for Python code in this project.

### Python naming conventions

*   _Variables_: Use lowercase with underscores (snake_case). Examples:
    `qsim_op`, `qubit_to_index_dict`, `gate_kind`.

*   _Constants_: Use uppercase with underscores. Examples: `MAX_VALUE`,
    `DATABASE_NAME`.

*   _Functions_: Use lowercase with underscores (snake_case). Examples:
    `calculate_total()`, `process_data()`. Internal functions are prefixed with
    an understore (`_`).

*   _Classes_: Use CapWords (CamelCase). Examples: `UserManager`,
    `PaymentProcessor`.

*   _Modules_: Use lowercase with underscores (snake_case). Examples:
    `user_utils`, `payment_gateway`.

###  Docstrings and documentation

This project uses [Google style doc
strings](http://google.github.io/styleguide/pyguide.html#381-docstrings) with a
Markdown flavor and support for LaTeX. Docstrings use tripe double quotes, and
the first line should be a concise one-line summary of the function or object.

Here is an example docstring:

```python
def some_method(a: int, b: str) -> float:
    r"""One-line summary of method.

    Additional information about the method, perhaps with some sort of LaTeX
    equation to make it clearer:

        $$
        M = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}
        $$

    Notice that this docstring is an r-string, since the LaTeX has backslashes.
    We can also include example code:

        print(cirq_google.Sycamore)

    You can also do inline LaTeX like $y = x^2$ and inline code like
    `cirq.unitary(cirq.X)`.

    And of course there's the standard sections.

    Args:
        a: The first argument.
        b: Another argument.

    Returns:
        An important value.

    Raises:
        ValueError: The value of `a` wasn't quite right.
    """
```

### Python formatting

Note: the Python code uses 88-column line widths, which is the default used by
code the formatters `black` and `flynt`. (The C++ files use 80.)

The following programs can be used to perform some automated formatting.

*   `black --diff --check PATH/TO/FILE` will report whether the file
    `PATH/TO/FILE` conforms to the project style conventions.

*   `black PATH/TO/FILE` will reformat the file.

*   `isort PATH/TO/FILE` will sort the `import` statements into a consistent
    form. We follow the [import
    standards](https://www.python.org/dev/peps/pep-0008/#imports) of PEP 8.

*   `check/format-incremental` will check and optionally reformat files that
    have been changed since the last commit.

### Python type annotations

This project makes extensive use of type annotations as defined by [PEP
484](https://peps.python.org/pep-0484/). All new code should use type
annotations where possible, especially on public classes and functions to serve
as documentation, but also on internal code so that the `mypy` type checker can
help catch coding errors.

### Python linting

Python files:

*   `pylint PATH/TO/FILE` will run `pylint` on the file `PATH/TO/FILE`. This is
    useful to use after editing a single file and before committing changes to
    the git repository.

*   `pylint -j0 .` will run `pylint` on all Python files.

### Python testing

Unit and integration tests can be run for the Python portions of this project
using the following command:

```shell
make run-py-tests
```

## C++-specific guidance

This section outlines the coding conventions for C++ code in this project.

### C++ naming conventions

#### Functions, variables, classes, and similar code elements

*   _Variables_: Use lowercase with underscores (snake_case). Examples:
    `user_name`, `total_count`.

*   _Constants_: Use uppercase with underscores. Examples: `MAX_VALUE`,
    `DATABASE_NAME`.

*   _Functions_: Use CapWords (CamelCase). Examples: `PrintAmplitudes`,
    `FillIndices,`.

*   _Classes_/_Structs_: Use CapWords (CamelCase). Examples: `UserManager`,
    `PaymentProcessor`.

*   _Namespaces_: Use snake_case.

*   _Enum Members_: Usually prefixed with `k` followed by a CamelCase name.
    Examples: `kX`, `kY`, as seen from Python.

#### Template names

| Name                    | Example Type                                             |
|-------------------------|----------------------------------------------------------|
| `Args` (`parfor.h`)     | Arguments to a `Function` object.                        |
| `Bitstring`             | Alias for `uint64_t`.                                    |
| `Circuit`               | [`Circuit`](../lib/circuit.h)                            |
| `Container` (`util.h`)  | A vector of strings, or `Op`'s output type.              |
| `Ctype`                 | A complex type, e.g., `std::complex<float>`.             |
| `For`                   | `for` loop abstractions, see below.                      |
| `ForArgs`               | Arguments for constructing `For` objects, see below.     |
| `FP`                    | (In `simulator_basic.h`.) Same as `fp_type`.             |
| `fp_type`               | A floating-point type, i.e., `float` or `double`.        |
| `Function` (`parfor.h`) | Any function; args are specified with `Args`.            |
| `FuserT`                | [`BasicGateFuser`](../lib/fuser_basic.h)                 |
| `Gate`                  | [`Gate`](../lib/gate.h)                                  |
| `GateCirq`              | [`GateCirq`](../lib/gates_cirq.h)                        |
| `GateDef`               | [`GateRX`](../lib/gate.h)                                |
| `GateQSim`              | [`GateQSim`](../lib/gates_qsim.h)                        |
| `GK` / `GateKind`       | [`GateKind`](../lib/gate.h)                              |
| `HybridSimulator`       | [`HybridSimulator`](../lib/hybrid.h)                     |
| `IO`                    | [`IO`](../lib/io.h)                                      |
| `IOFile`                | [`IOFile`](../lib/io_file.h)                             |
| `Matrix`                | A numeric C++ vector representing a matrix.              |
| `MeasurementFunc`       | [`measure` (in `PrintAmplitudes`)](../apps/qsim_base.cc) |
| `Op` (`util.h`)         | [`to_int` (in `Options`)](../apps/qsim_amplitudes.cc)    |
| `ParallelFor`           | [`ParallelFor`](../lib/parfor.h)                         |
| `Params`                | Vector of `fp_type`.                                     |
| `SequentialFor`         | [`SequentialFor`](../lib/seqfor.h)                       |
| `Simulator`             | [`SimulatorAVX`](../lib/simulator_avx.h)                 |
| `State`                 | Unique pointer to `fp_type`.                             |
| `StateSpace`            | [`StateSpace`](../lib/statespace.h)                      |
| `Stream`                | A valid input for `std::getline()`.                      |

#### `For` and `ForArgs`

The `For` type represents a `for` loop. It is a template parameter of the
`StateSpace*` (lib/statespace*.h) and `Simulator*` (lib/simulator*.h) classes.
`For` objects in these classes are utilized to iterate over quantum state
arrays. `ForArgs` is a variadic template parameter pack of the constructors of
`StateSpace*` and `Simulator*`. It is utilized to pass arguments to the
constructors of `For` objects.

The qsim library provides `ParallelFor` (lib/parfor.h) and `SequentialFor`
(lib/seqfor.h). The user can also use custom `For` types. Examples of usage
follow.

```C++
// ParallelFor(unsigned num_threads) constructor
SimulatorAVX<ParallelFor> simulator(num_qubits, num_threads);
```

```C++
// copy constructor
ParallelFor parallel_for(num_threads);
SimulatorAVX<ParallelFor> simulator(num_qubits, parallel_for);
```

```C++
ParallelFor parallel_for(num_threads);
// const reference to parallel_for in simulator
SimulatorAVX<const ParallelFor&> simulator(num_qubits, parallel_for);
```

In the following, we assume a custom `MyFor` type that has
`MyFor(unsigned num_threads, const Context& context)` and copy constructors.

```C++
// MyFor(unsigned num_threads, const Context& context) constructor
SimulatorAVX<MyFor> simulator(num_qubits, num_threads, context);
```

```C++
// copy constructor
MyFor my_for(num_threads, context);
SimulatorAVX<MyFor> simulator(num_qubits, my_for);
```

```C++
// const reference to my_for in simulator
MyFor my_for(num_threads, context);
SimulatorAVX<const MyFor&> simulator(num_qubits, my_for);
```

### C++ formatting

Note: the C++ code files use 80-column line widths. (The Python files use 88.)

*   `clang-format -n PATH/TO/FILE` will report whether the file `PATH/TO/FILE`
    conforms to the project style conventions.

*   `clang-format -i PATH/TO/FILE` will reformat the file.

#### C++ testing

If you only want to run tests for the core C++ libraries, use this command:

```shell
make run-cxx-tests
```

To build tests without running them, instead use:

```shell
make cxx-tests
```
