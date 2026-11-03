// Copyright 2026 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPERATION_H_
#define OPERATION_H_

#include <optional>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "channel.h"
#include "classical_control_expr.h"
#include "gate.h"
#include "operation_base.h"

namespace qsim {

// Forward declaration of classically controlled operation.
template <typename>
struct ClassicallyControlledOperation;

/**
 * Polymorphic variant representing any executable circuit operation.
 * @tparam FP Floating-point precision type (`float` or `double`).
 */
template <typename FP>
using Operation = std::variant<Gate<FP>, RuntimeResolvedGate<FP>,
                               ControlledGate<FP>, Measurement, Channel<FP>,
                               ClassicallyControlledOperation<FP>>;

/**
 * Represents a classical control flow structure, runtime variable assignment,
 * or I/O operation.
 *
 * A `ClassicallyControlledOperation` encapsulates `if/elsif/else` branching,
 * loops, variable assignments, trajectory filtering, and printing.
 * The exact role of struct fields depends on `kind`:
 *
 * - **`kIfElse`**:
 *   - `sub_ops`: Operations for each branch (`sub_ops[0]` for `if`,
 *     `sub_ops[1..n-1]` for `elsif`, last for `else`).
 *   - `exprs`: Condition expressions (`exprs[0]` for `if`, `exprs[1..n-1]`
 *     for `elsif`).
 *   - `scope_indices`: Symbol table scope indices for each branch block.
 *
 * - **`kDoWhile`**:
 *   - `sub_ops[0]`: Loop blody operations.
 *   - `exprs[0]`: Condition expression evaluated *after* each iteration.
 *   - `scope_indices[0]`: Scope index for loop block.
 *
 * - **`kRepeat`**:
 *   - `sub_ops[0]`: Loop block operations.
 *   - `exprs[0]`: Condition expression evaluated *before* each iteration.
 *   - `scope_indices[0]`: Scope index for loop block.
 *
 * - **`kAssign`**:
 *   - `str[0]`: Identifier string of the target variable/symbol.
 *   - `exprs`: Expression(s) assigned to the symbol (scalar or vector values).
 *   - `indices`: Single index expression (`indices[0]`) if updating
 *     a specific element (e.g. `vec[i] = x`).
 *
 * - **`kPrintLn`**:
 *   - `str[0]`: Optional string literal or format specifier.
 *   - `exprs`: Expressions to evaluate and print (up to 4 expressions).
 *
 * - **`kDiscard`**:
 *   - `exprs[0]`: Condition expression. If `true`, the simulator aborts
 *     current simulation trajectory and the return value indicates that this
 *     trajectory should be discarded.
 *
 * @tparam FP Floating-point precision type (`float` or `double`).
 */
template <typename FP>
struct ClassicallyControlledOperation : public BaseOperation {
  enum Kind {
    /** Conditional branching block (`if / elsif / else`). */
    kIfElse = 1,
    /** Loop block executed at least once (`do ... while`). */
    kDoWhile,
    /** Pre-condition loop block (`repeat condition`). */
    kRepeat,
    /** Runtime symbol/variable assignment (`var = expr`). */
    kAssign,
    /** Console debugging/printing output (`println`). */
    kPrintLn,
    /** Trajectory filtering (`discard condition`). */
    kDiscard,
  };

  /** Nested operations associated with control branches/blocks. */
  std::vector<std::vector<Operation<FP>>> sub_ops;
  /** Expressions used for conditions, assignment values, or printing. */
  std::vector<cc::Expr> exprs;
  /** Array indices for vector element updates. */
  std::vector<cc::Index> indices;
  /** String parameters (variable names or print format strings). */
  std::vector<std::string_view> str;
  /** Symbol table scope indices corresponding to sub-blocks. */
  std::vector<unsigned> scope_indices;
  /** Type classification of this classical operation. */
  Kind kind;
};

namespace detail {

template <typename T>
struct op_fp_type {
  using type = typename T::fp_type;
};

template <typename T>
struct op_fp_type<T*> {
  using type = typename T::fp_type;
};

template <typename... Ts>
struct op_fp_type<std::variant<Ts...>> {
  using T = std::variant_alternative_t<0, std::variant<Ts...>>;
  using type = typename op_fp_type<T>::type;
};

}  // namespace detail

template <typename Operation>
using OpFpType = typename detail::op_fp_type<std::decay_t<Operation>>::type;

}  // namespace qsim

#endif  // OPERATION_H_
