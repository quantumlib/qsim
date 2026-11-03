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

#ifndef CLASSICAL_CONTROL_EXPR_H_
#define CLASSICAL_CONTROL_EXPR_H_

#include <cstdint>
#include <functional>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "error.h"

namespace qsim::cc {

struct SymTable;

// The prefix V stands for Value.
using VInt = int64_t;
using VFloat = double;
using VBool = bool;

using EvalIntFn = std::function<VInt(const SymTable&)>;
using EvalFloatFn = std::function<double(const SymTable&)>;
using EvalBoolFn = std::function<bool(const SymTable&)>;

/**
 * Variant representing an index expression used to access elements in vectors
 * or measurements.
 */
using Index = std::variant<VInt, VBool, EvalIntFn, EvalBoolFn>;

/** Represents a symbol reference. */
using SymbolKey = std::string_view;

/**
 * Represents an indexed symbol reference, such as a vector or measurement bit
 * lookup (e.g., `v[i]`).
 */
using SymbolKeyInd = std::pair<SymbolKey, Index>;

/**
 * Abstract representation of compile-time constants or runtime-evaluated
 * expressions.
 *
 * Expressions can be:
 * - Direct scalar constants (`VInt`, `VFloat`, `VBool`)
 * - Runtime functional closures (`EvalIntFn`, `EvalFloatFn`, `EvalBoolFn`)
 * - Plain or indexed symbol lookup identifiers (`SymbolKey`, `SymbolKeyInd`)
 */
using Expr = std::variant<VInt, VFloat, VBool, EvalIntFn,
                          EvalFloatFn, EvalBoolFn, SymbolKey, SymbolKeyInd>;

/**
 * Computes the dimensionality/size of an Index.
 * @return Returns 1 for scalar index expressions.
 */
inline unsigned IndexSize(const Index&) {
  return 1;
}

/**
 * Checks if an index expression can be evaluated at compile-time.
 * @param i The index expression to check.
 * @return True if the index is a static constant (`VInt` or `VBool`).
 */
inline bool IsConstIndex(const Index& i) {
  auto f = [](auto&& i) -> bool {
    using I = std::decay_t<decltype(i)>;
    return std::is_same_v<I, VInt> || std::is_same_v<I, VBool>;
  };

  return std::visit(f, i);
}

/**
 * Evaluates an Index variant to a concrete array/vector position index
 * (`std::size_t`).
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param index Index variant to evaluate.
 * @return Non-negative target index position.
 * @throws std::runtime_error If evaluated index is negative.
 */
template <typename SymTable>
inline std::size_t EvalIndex(const SymTable& symtab, const Index& index) {
  auto f = [&symtab](auto&& index) -> VInt {
    using I = std::decay_t<decltype(index)>;

    if constexpr (std::is_same_v<I, VInt>) {
      return index;
    } else if constexpr (std::is_same_v<I, VBool>) {
      return index;
    } else if constexpr (std::is_same_v<I, EvalIntFn>) {
      return index(symtab);
    } else if constexpr (std::is_same_v<I, EvalBoolFn>) {
      return index(symtab);
    }
  };

  VInt i = std::visit(f, index);

  if (i < 0) {
    Error::Throw("negative index");
  }

  return static_cast<std::size_t>(i);
}

/**
 * Checks if an expression can be evaluated at compile-time.
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to check.
 * @return True for literals (`VInt`, `VFloat`, `VBool`) and read-only symbols;
 *   false for closures.
 * @throws std::runtime_error If symbol lookup fails.
 */
template <typename SymTable>
inline bool IsConstExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> bool {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, VInt>) {
      return true;
    } else if constexpr (std::is_same_v<E, VFloat>) {
      return true;
    } else if constexpr (std::is_same_v<E, VBool>) {
      return true;
    } else if constexpr (std::is_same_v<E, EvalIntFn>) {
      return false;
    } else if constexpr (std::is_same_v<E, EvalFloatFn>) {
      return false;
    } else if constexpr (std::is_same_v<E, EvalBoolFn>) {
      return false;
    } else if constexpr (std::is_same_v<E, SymbolKey>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->IsReadOnly();
    } else if constexpr (std::is_same_v<E, SymbolKeyInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->IsReadOnly() && IsConstIndex(e.second);
    }
  };

  return std::visit(f, e);
}

/**
 * Checks if an expression can be converted/evaluated to an integer.
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Expression to check.
 * @return True for int, bool, int-closures, or int-convertible symbols;
 *   false for floats.
 * @throws std::runtime_error If symbol lookup fails.
 */
template <typename SymTable>
inline bool IsConvertibleToInt(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> bool {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, VInt>) {
      return true;
    } else if constexpr (std::is_same_v<E, VFloat>) {
      return false;
    } else if constexpr (std::is_same_v<E, VBool>) {
      return true;
    } else if constexpr (std::is_same_v<E, EvalIntFn>) {
      return true;
    } else if constexpr (std::is_same_v<E, EvalFloatFn>) {
      return false;
    } else if constexpr (std::is_same_v<E, EvalBoolFn>) {
      return true;
    } else if constexpr (std::is_same_v<E, SymbolKey>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->IsConvertibleToInt();
    } else if constexpr (std::is_same_v<E, SymbolKeyInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->IsConvertibleToInt();
    }
  };

  return std::visit(f, e);
}

/**
 * Evaluates an expression and returns its scalar value widened/converted
 * to VFloat for integer types.
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to evaluate.
 * @return Result as double-precision float (`VFloat`).
 * @throws std::runtime_error If symbol lookup fails.
 */
template <typename SymTable>
inline VFloat EvalFloatExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> VFloat {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, VInt>) {
      return e;
    } else if constexpr (std::is_same_v<E, VFloat>) {
      return e;
    } else if constexpr (std::is_same_v<E, VBool>) {
      return e;
    } else if constexpr (std::is_same_v<E, EvalIntFn>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, EvalFloatFn>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, EvalBoolFn>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, SymbolKey>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->GetFloat();
    } else if constexpr (std::is_same_v<E, SymbolKeyInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->GetFloat(EvalIndex(symtab, e.second));
    }
  };

  return std::visit(f, e);
}

/**
 * Evaluates an expression as an integer (`VInt`).
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to evaluate.
 * @return Integer result.
 * @throws std::runtime_error If expression is a `float` type or if symbol
 *   lookup fails.
 */
template <typename SymTable>
inline VInt EvalIntExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> VInt {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, VInt>) {
      return e;
    } else if constexpr (std::is_same_v<E, VBool>) {
      return e;
    } else if constexpr (std::is_same_v<E, EvalIntFn>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, EvalBoolFn>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, SymbolKey>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->GetInt();
    } else if constexpr (std::is_same_v<E, SymbolKeyInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->GetInt(EvalIndex(symtab, e.second));
    } else {
      Error::Throw("'float' is not convertible to 'int'");
      return 0;
    }
  };

  return std::visit(f, e);
}

/**
 * Evaluates an expression as a boolean condition (`VBool`).
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to evaluate.
 * @return True if integer value is non-zero or boolean condition holds.
 * @throws std::runtime_error if expression evaluates from a `float` type or
 *   if symbol lookup fails.
 */
template <typename SymTable>
inline VBool EvalCondExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> VBool {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, VInt>) {
      return e != 0;
    } else if constexpr (std::is_same_v<E, VBool>) {
      return e;
    } else if constexpr (std::is_same_v<E, EvalIntFn>) {
      return e(symtab) != 0;
    } else if constexpr (std::is_same_v<E, EvalBoolFn>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, SymbolKey>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->GetInt() != 0;
    } else if constexpr (std::is_same_v<E, SymbolKeyInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->GetInt(EvalIndex(symtab, e.second)) != 0;
    } else {
      Error::Throw("'float' is not convertible to 'bool'");
      return 0.0;
    }
  };

  return std::visit(f, e);
}

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_EXPR_H_
