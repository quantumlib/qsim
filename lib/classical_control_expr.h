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

using TInt = int64_t;
using TFloat = double;
using TBool = bool;
using TFuncI = std::function<TInt(const SymTable&)>;
using TFuncF = std::function<double(const SymTable&)>;
using TFuncB = std::function<bool(const SymTable&)>;

/**
 * Variant representing an index expression used to access elements in vectors
 * or measurements.
 */
using Index = std::variant<TInt, TBool, TFuncI, TFuncB>;

/** Represents a symbol reference. */
using TSymbol = std::string_view;

/**
 * Represents an indexed symbol reference, such as a vector or measurement bit
 * lookup (e.g., `v[i]`).
 */
using TSymbolInd = std::pair<TSymbol, Index>;

/**
 * Abstract representation of compile-time constants or runtime-evaluated
 * expressions.
 *
 * Expressions can be:
 * - Direct scalar constants (`TInt`, `TFloat`, `TBool`)
 * - Runtime functional closures (`TFuncI`, `TFuncF`, `TFuncB`)
 * - Plain or indexed symbol lookup identifiers (`TSymbol`, `TSymbolInd`)
 */
using Expr = std::variant<TInt, TFloat, TBool,
                          TFuncI, TFuncF, TFuncB, TSymbol, TSymbolInd>;

/**
 * Computes the dimensionality/size of an Index.
 * @return Returns 1 for scalar index expressions.
 */
inline unsigned IndexSize(const Index&) {
  return 1;
}

/**
 * Checks if an index expression can be evaluated at compile-time.
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param i The index expression to check.
 * @return True if the index is a static constant (`TInt` or `TBool`).
 */
template <typename SymTable>
inline bool IsConstIndex(const SymTable& symtab, const Index& i) {
  auto f = [&symtab](auto&& i) -> bool {
    using I = std::decay_t<decltype(i)>;
    return std::is_same_v<I, TInt> || std::is_same_v<I, TBool>;
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
 */
template <typename SymTable>
inline std::size_t EvalIndex(const SymTable& symtab, const Index& index) {
  auto f = [&symtab](auto&& index) -> std::size_t {
    using I = std::decay_t<decltype(index)>;

    if constexpr (std::is_same_v<I, TInt>) {
      return index;
    } else if constexpr (std::is_same_v<I, TBool>) {
      return index;
    } else if constexpr (std::is_same_v<I, TFuncI>) {
      return index(symtab);
    } else if constexpr (std::is_same_v<I, TFuncB>) {
      return index(symtab);
    }
  };

  return std::visit(f, index);
}

/**
 * Checks if an index expression can be evaluated at compile-time.
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to check.
 * @return True for literals (`TInt`, `TFloat`, `TBool`) and read-only symbols;
 *   false for closures.
 * @throws std::runtime_error If symbol lookup fails.
 */
template <typename SymTable>
inline bool IsConstExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> bool {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, TInt>) {
      return true;
    } else if constexpr (std::is_same_v<E, TFloat>) {
      return true;
    } else if constexpr (std::is_same_v<E, TBool>) {
      return true;
    } else if constexpr (std::is_same_v<E, TFuncI>) {
      return false;
    } else if constexpr (std::is_same_v<E, TFuncF>) {
      return false;
    } else if constexpr (std::is_same_v<E, TFuncB>) {
      return false;
    } else if constexpr (std::is_same_v<E, TSymbol>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->IsReadOnly();
    } else if constexpr (std::is_same_v<E, TSymbolInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->IsReadOnly();
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

    if constexpr (std::is_same_v<E, TInt>) {
      return true;
    } else if constexpr (std::is_same_v<E, TFloat>) {
      return false;
    } else if constexpr (std::is_same_v<E, TBool>) {
      return true;
    } else if constexpr (std::is_same_v<E, TFuncI>) {
      return true;
    } else if constexpr (std::is_same_v<E, TFuncF>) {
      return false;
    } else if constexpr (std::is_same_v<E, TFuncB>) {
      return true;
    } else if constexpr (std::is_same_v<E, TSymbol>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->IsConvertibleToInt();
    } else if constexpr (std::is_same_v<E, TSymbolInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->IsConvertibleToInt();
    }
  };

  return std::visit(f, e);
}

/**
 * Evaluates an expression and returns its scalar value widened/converted
 * to TFloat for integer types.
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to evaluate.
 * @return Result as double-precision float (`TFloat`).
 * @throws std::runtime_error If symbol lookup fails.
 */
template <typename SymTable>
inline TFloat EvalExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> TFloat {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, TInt>) {
      return e;
    } else if constexpr (std::is_same_v<E, TFloat>) {
      return e;
    } else if constexpr (std::is_same_v<E, TBool>) {
      return e;
    } else if constexpr (std::is_same_v<E, TFuncI>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, TFuncF>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, TFuncB>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, TSymbol>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->GetFloat();
    } else if constexpr (std::is_same_v<E, TSymbolInd>) {
      const auto* sym = symtab.LookupOrError(
          e.first, "identifier '{}' is not defined", e.first);
      return sym->GetFloat(EvalIndex(symtab, e.second));
    }
  };

  return std::visit(f, e);
}

/**
 * Evaluates an expression as an integer (`TInt`).
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to evaluate.
 * @return Integer result.
 * @throws std::runtime_error If expression is a `float` type or if symbol
 *   lookup fails.
 */
template <typename SymTable>
inline TInt EvalIntExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> TInt {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, TInt>) {
      return e;
    } else if constexpr (std::is_same_v<E, TBool>) {
      return e;
    } else if constexpr (std::is_same_v<E, TFuncI>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, TFuncB>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, TSymbol>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->GetInt();
    } else if constexpr (std::is_same_v<E, TSymbolInd>) {
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
 * Evaluates an expression as a boolean condition (`TBool`).
 * @tparam SymTable Symbol table type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param e Target expression to evaluate.
 * @return True if integer value is non-zero or boolean condition holds.
 * @throws std::runtime_error If expression evaluates from a `float` type or
 *   if symbol lookup fails.
 */
template <typename SymTable>
inline TBool EvalCondExpr(const SymTable& symtab, const Expr& e) {
  auto f = [&symtab](auto&& e) -> TBool {
    using E = std::decay_t<decltype(e)>;

    if constexpr (std::is_same_v<E, TInt>) {
      return e != 0;
    } else if constexpr (std::is_same_v<E, TBool>) {
      return e;
    } else if constexpr (std::is_same_v<E, TFuncI>) {
      return e(symtab) != 0;
    } else if constexpr (std::is_same_v<E, TFuncB>) {
      return e(symtab);
    } else if constexpr (std::is_same_v<E, TSymbol>) {
      const auto* sym = symtab.LookupOrError(
          e, "identifier '{}' is not defined", e);
      return sym->GetInt() != 0;
    } else if constexpr (std::is_same_v<E, TSymbolInd>) {
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
