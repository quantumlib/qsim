# qsim style guide

This style guide for Gemini outlines the coding conventions for this project.

## Introduction and goals

### Overall goals

*   _Readability_: Code must be easy for humans to understand. Prefer clarity
    over cleverness. Write elegant, well-structured code.

*   _Maintainability_: Code must be easy to modify and extend.

*   _Consistency_: Be consistent with naming convention, style, and other
    patterns in the code base.

*   _Performance_: While readability is paramount, code should be efficient.

### Overall software engineering principles

*   Write modular code that follows language idioms and best practices.

*   Prioritize numerical stability and accuracy. Keep in mind the limitations of
    floating-point arithmetic (e.g., float, double).

*   Strive for performance through memory efficiency. Watch out for bottlenecks
    due to memory access. Design data structures and algorithms to promote data
    locality. Access memory sequentially (e.g., iterating through a
    `std::vector`) to maximize cache hits.

*   Design for vectorization and parallelism. Structure loops and data access
    patterns to be friendly to compiler vectorization (SIMD). Avoid complex
    control flow (if/else) inside tight loops where possible.

*   Validate all user and file-based data to guard against security
    vulnerabilities.

*   New code requires new tests. Ensure existing tests continue to pass or are
    updated when making changes.

*   Thoroughly test for edge cases, including invalid inputs, null values, empty
    arrays, and off-by-one errors.

*   Keep tests independent. Use setup and teardown functions for clean
    environments and mock all external dependencies.

## Development workflow

### Before you start

*   **CRITICAL**: before changing any file, first ask the user if they want to
    create a git branch for the work you are about to do.

### File naming conventions

*   Regular file names should be in all lower case, using underscores as word
    separators as needed. The names should indicate the purpose of the code in
    the file, while also be kept as short as possible without compromising
    understandability.

*   Test files are usually named after the file they test but with a name
    ending in `_test.py` or `_test.cc`. For example, `something.py` would have
    tests in a file named `something_test.py`.

### File structure conventions

*   Files must end with a final newline, unless they are special files that do
    not normally have ending newlines.

### Git commit conventions

*   Use `git commit` to commit changes to files as you work. Each commit should
    encompass a subportion of your work that is conceptually related.

*   Each commit must have a title and a description.

### Code style conventions

Follow these Google coding conventions, with some project-specific conventions
noted below.

*   [Google C++ Style Guide](
    https://google.github.io/styleguide/cppguide.html)

*   [Google Python Style Guide](
    https://google.github.io/styleguide/pyguide.html)

*   [Google Markdown Style Guide](
    https://google.github.io/styleguide/docguide/style.html)

*   [Google Shell Style Guide](
    https://google.github.io/styleguide/shellguide.html)

To learn this project's conventions for line length, indentation, and other
details of coding style, please inspect the following configuration files:

*   [`.editorconfig`](../.editorconfig) for basic code editor configuration
    (e.g., indentation and line length) specified using the
    [EditorConfig](https://editorconfig.org/) format.

*   [`.clang-format`](../.clang-format) for C++ code and also protobuf (Protocol
    Buffers) data structure definitions.

*   [`.hadolint.yaml`](../.hadolint.yaml) for `Dockerfile`s.

*   [`.jsonlintrc.yaml`](../.jsonlintrc.yaml) for JSON files.

*   [`.markdownlintrc`](../.markdownlintrc) for Markdown files.

*   [`.pylintrc`](../.pylintrc) for Python code.

*   [`.yamllint.yaml`](../.yamllint.yaml) for YAML files.

### Code comment conventions

Every source code file longer than 2 lines must begin with a header comment with
the copyright and license. We use the Apache 2.0 license. Here is an example for
Python code:

```python
# Copyright 2026 Google LLC
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

License headers are necessary in Python, C++, Bash/shell, and other programming
language files, as well as configuration files in YAML, TOML, ini, and other
config file formats. They are not necessary in Markdown or plain text files.

For comments in other parts of the files, follow these guidelines:

*   _Write clear and concise comments_: Comments should explain the "why", not
    the "what". The comments should explain the intent, trade-offs, and
    reasoning behind the implementation.

*   _Comment sparingly_: Well-written code should be self-documenting where
    possible. It's not necessary to add comments for code fragments that can
    reasonably be assumed to be self-explanatory.

## Python-specific guidance

This section outlines coding conventions for Python code in this project.

### Setting up a Python environment for development

To set up a Python development environment, do the following:

```shell
python3 -m venv venv
source venv/bin/activate
pip install -U pip
```

If there is a `dev-requirements.txt` file in the directory, run

```shell
pip install -r requirements.txt -r dev-requirements.txt
```

otherwise, run

```shell
pip install -r requirements.txt --group dev
```

### Python naming conventions

This project follows some nomenclature rules and conventions in order to
maintain a consistent interface that is easy to use. By using consistent
naming, we can reduce cognitive load on human users and developers.

*   _Variables_: Use lowercase with underscores (snake_case). Examples:
    `qsim_op`, `qubit_to_index_dict`, `gate_kind`.

*   _Constants_: Use uppercase with underscores. Examples: `MAX_VALUE`,
    `DATABASE_NAME`.

*   _Functions_: Use lowercase with underscores (snake_case). Examples:
    `calculate_total()`, `process_data()`. Internal functions are prefixed with
    an underscore (`_`).

*   _Classes_: Use CapWords (CamelCase). Examples: `UserManager`,
    `PaymentProcessor`.

*   _Modules_: Use lowercase with underscores (snake_case). Examples:
    `user_utils`, `payment_gateway`.

*   _Domain-specific terms_:

    *   Use `state_vector` to describe a pure state; do not use
        `wavefunction` or `wave_function`.

    *   If an object is an array or a computational basis state (given by an
        `int`), use the term `state_rep`. If it is the initial state of a
        system, use `initial_state`.

    *   A function argument (`state_vector`, `state_rep`, or `initial_state`)
        should permit any of the possible representations of a state: A NumPy
        array, a NumPy tensor, an integer representing a qubit-system's
        computational basis state, a sequence of _n_ integers representing a
        qudit's basis state, or a `cirq.ProductState`. The type annotation
        should be `cirq.STATE_VECTOR_LIKE` and you should use
        `cirq.to_valid_state_vector` to canonicalize as a NumPy array of
        amplitudes. If a function expects a NumPy array of amplitudes, its type
        annotation should be `np.ndarray`.

### Python docstrings and documentation

This project uses [Google style doc strings](
http://google.github.io/styleguide/pyguide.html#381-docstrings) with a Markdown
flavor and support for LaTeX. Docstrings use tripe double quotes, and the first
line should be a concise one-line summary of the function or object.

Here is an example docstring:

```python
def some_function(a: int, b: str) -> float:
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
code the formatter `black` but not the [Google Python Style Guide](
https://google.github.io/styleguide/pyguide.html). (The latter use 80.)

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

This project uses type annotations as defined by [PEP 484](
https://peps.python.org/pep-0484/). All new code should use type annotations
where possible, especially on public classes and functions to serve as
documentation, but also on internal code so that the `mypy` type checker can
help catch coding errors.

### Linting Python code

Python files:

*   `pylint PATH/TO/FILE` will run `pylint` on the file `PATH/TO/FILE`. This is
    useful to use after editing a single file and before committing changes to
    the git repository.

*   `pylint -j0 .` will run `pylint` on all Python files.

### Testing Python code

Unit and integration tests can be run for the Python portions of this project
using the following command:

```shell
make -j run-py-tests
```

## C++-specific guidance

This section outlines coding conventions for C++ code in this project.

### C++ naming conventions

*   _Variables_: Use lowercase with underscores (snake_case). Examples:
    `user_name`, `total_count`.

*   _Functions_: Use CapWords (CamelCase). Examples: `PrintAmplitudes`,
    `FillIndices,`.

*   _Classes_/_Structs_: Use CapWords (CamelCase). Examples: `UserManager`,
    `PaymentProcessor`.

*   _Namespaces_: Use snake_case.

*   _Domain-specific terms_:

    *   In the C++ code, `state` is used everywhere for state vectors.

    *   A computational basis state (say, $|0000\rangle$) is typically
        referred to as a `bitstring`.

### Formatting C++ code

Note: the C++ code files use 80-column line widths. (The Python files use 88.)

*   `clang-format -n PATH/TO/FILE` will report whether the file `PATH/TO/FILE`
    conforms to the project style conventions.

*   `clang-format -i PATH/TO/FILE` will reformat the file.

#### Testing C++ code

If you only want to run tests for the core C++ libraries, use this command:

```shell
bazel test tests:all
```

To build tests without running them, instead use:

```shell
bazel build --config=verbose tests:all
```

## Shell script-specific guidance

Shell scripts should use Bash.

### Formatting shell scripts

Use the [Google Shell Style Guide](
https://google.github.io/styleguide/shellguide.html) with the following changes:

*   Use indentation of 4 spaces, not 2.

## TOML file-specific guidance

### Formatting `.toml` files

We use indentation of 4 spaces, not 2.
