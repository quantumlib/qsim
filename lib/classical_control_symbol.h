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

#ifndef CLASSICAL_CONTROL_SYMBOL_H_
#define CLASSICAL_CONTROL_SYMBOL_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "classical_control_expr.h"
#include "error.h"

namespace qsim::cc {

/**
 * Represents a typed value container holding scalars, vectors, or quantum
 * measurements. A Symbol encapsulates values used in quantum circuit
 * simulation or classical computation, supporting typed access, indexing,
 * conversions, assignment and and read-only flags.
 */
struct Symbol {
  using Int = TInt;
  using Float = TFloat;
  using IntVector = std::vector<Int>;
  using FloatVector = std::vector<Float>;

  /**
   * Holds the result of a quantum measurement over a specified set of qubits.
   */
  struct Mea {
    /** Bitfield storing measurement outcomes. */
    uint64_t bits = 0;
    /** Number of measured qubits. */
    unsigned num_bits = 0;
    /** Circuit moment/time tag when the measurement occurred. */
    unsigned time = 0;
  };

  /**  Variant over all supported concrete value types. */
  using Val = std::variant<Mea, Int, Float, IntVector, FloatVector>;

  /** Constructs an uninitialized/default Symbol. */
  Symbol() {}

  /**
   * Constructs a Symbol inferring read-only status based on the value type.
   * Mea instances default to read_only = false; all other types default to
   * read_only = true.
   * @tparam V Forwarded value type.
   * @param v Value to store in the variant.
   */
  template <typename V>
  Symbol(V&& v) : val(std::forward<V>(v)) {
    if constexpr (std::is_same_v<std::decay_t<V>, Mea>) {
      read_only = false;
    }
  }

  /**
   * Constructs a Symbol with an explicit read-only specification.
   * @tparam V Forwarded value type.
   * @param v Value to store in the variant.
   * @param read_only Whether the symbol should be treated as
   *   constant/read-only.
   */
  template <typename V>
  Symbol(V&& v, bool read_only)
      : val(std::forward<V>(v)), read_only(read_only) {}

  /** Checks if the underlying value is a quantum measurement (Mea). */
  bool HoldsMea() const {
    return std::holds_alternative<Mea>(val);
  }

  /**
   * Checks if the underlying value is a vector type (IntVector or
   * FloatVector).
   */
  bool HoldsVector() const {
    return std::holds_alternative<IntVector>(val)
        || std::holds_alternative<FloatVector>(val);
  }

  /**
   * Checks if the underlying value uses floating-point precision (Float or
   * FloatVector).
   */
  bool IsFloat() const {
    return std::holds_alternative<Float>(val)
        || std::holds_alternative<FloatVector>(val);
  }

  /**
   * Checks if the underlying value can be implicitly or explicitly converted
   * to Int.
   */
  bool IsConvertibleToInt() const {
    return std::holds_alternative<Mea>(val) || std::holds_alternative<Int>(val)
        || std::holds_alternative<IntVector>(val);
  }

  /** Returns whether this symbol is marked read-only. */
  bool IsReadOnly() const {
    return read_only;
  }

  /**
   * Returns a mutable reference to the underlying Mea object.
   * @throws std::bad_variant_access if variant does not hold a Mea.
   */
  Mea& GetMea() {
    return std::get<Mea>(val);
  }

  /**
   * Returns a const reference to the underlying Mea object.
   * @throws std::bad_variant_access if variant does not hold a Mea.
   */
  const Mea& GetMea() const {
    return std::get<Mea>(val);
  }

  /**
   * Returns the size or element count of the stored value.
   * @return Number of bits for Mea, number of elements for vectors,
   *   or 1 for scalars.
   */
  unsigned Size() const {
    auto f = [this](auto&& v) -> unsigned {
      using V = std::decay_t<decltype(v)>;

      if constexpr (std::is_same_v<V, Mea>) {
        return v.num_bits;
      } else if constexpr (std::is_same_v<V, Int>) {
        return 1;
      } else if constexpr (std::is_same_v<V, Float>) {
        return 1;
      } else if constexpr (std::is_same_v<V, IntVector>) {
        return v.size();
      } else if constexpr (std::is_same_v<V, FloatVector>) {
        return v.size();
      }
    };

    return std::visit(f, val);
  }

  /**
   * @return The underlying integer value or raw bitfield of a measurement.
   * @throws std::runtime_error If the stored value is a Float or Vector.
   */
  Int GetInt() const {
    auto f = [this](auto&& v) -> Int {
      using V = std::decay_t<decltype(v)>;

      if constexpr (std::is_same_v<V, Mea>) {
        return GetMeaInt(v);
      } else if constexpr (std::is_same_v<V, Int>) {
        return v;
      } else if constexpr (std::is_same_v<V, Float>) {
        Error::Throw("'float' is not convertible to 'int'");
        return 0;
      } else {
        Error::Throw("'vector' is not convertible to 'int'");
        return 0;
      }
    };

    return std::visit(f, val);
  }

  /**
   * @return The scalar float value or integer promoted to float.
   * @throws std::runtime_error If the stored value is a Vector.
   */
  Float GetFloat() const {
    auto f = [this](auto&& v) -> Float {
      using V = std::decay_t<decltype(v)>;

      if constexpr (std::is_same_v<V, Mea>) {
        return GetMeaInt(v);
      } else if constexpr (std::is_same_v<V, Int>) {
        return v;
      } else if constexpr (std::is_same_v<V, Float>) {
        return v;
      } else {
        Error::Throw("'vector' is not convertible to 'int'");
        return 0.0;
      }
    };

    return std::visit(f, val);
  }

  /**
   * Fetches an integer value at index i.
   * @param i Zero-based index position.
   * @return The bit at i for Mea, or element at i for IntVector.
   * @throws std::runtime_error If out of range, or called on FloatVector
   *   or non-indexable scalars.
   */
  Int GetInt(std::size_t i) const {
    auto f = [this, &i](auto&& v) -> Int {
      using V = std::decay_t<decltype(v)>;

      if constexpr (std::is_same_v<V, Mea>) {
        return GetMeaInt(v, i);
      } else if constexpr (std::is_same_v<V, IntVector>) {
        if (i >= v.size()) {
          Error::Throw("index {} is out of range", i);
        }
        return v[i];
      } else if constexpr (std::is_same_v<V, FloatVector>) {
        Error::Throw("'float' is not convertible to 'int'");
        return 0;
      } else {
        Error::Throw("scalar identifier cannot be indexed");
        return 0;
      }
    };

    return std::visit(f, val);
  }

  /**
   * Fetches a float value at index i.
   * @param i Zero-based index position.
   * @return Value at i promoted/converted to Float.
   * @throws std::runtime_error If out of range or called on non-indexable
   *   scalars.
   */
  Float GetFloat(std::size_t i) const {
    auto f = [this, &i](auto&& v) -> Float {
      using V = std::decay_t<decltype(v)>;

      if constexpr (std::is_same_v<V, Mea>) {
        return GetMeaInt(v, i);
      } else if constexpr (std::is_same_v<V, IntVector>) {
        if (i >= v.size()) {
          Error::Throw("index {} is out of range", i);
        }
        return v[i];
      } else if constexpr (std::is_same_v<V, FloatVector>) {
        if (i >= v.size()) {
          Error::Throw("index {} is out of range", i);
        }
        return v[i];
      } else {
        Error::Throw("scalar identifier cannot be indexed");
        return 0.0;
      }
    };

    return std::visit(f, val);
  }

  /**
   * Extracts the raw bitfield representation of a measurement outcome.
   * @param mea Measurement object.
   * @return Full bitfield value as Int.
   */
  static Int GetMeaInt(const Mea& mea) {
    return mea.bits;
  }

  /**
   * Extracts a single bit from a measurement outcome at position i.
   * @param mea Measurement object.
   * @param i Zero-based bit position index.
   * @return 0 or 1 corresponding to the target bit state.
   * @throws std::runtime_error If i is out of range (`i >= mea.num_bits`).
   */
  static Int GetMeaInt(const Mea& mea, std::size_t i) {
    if (i >= mea.num_bits) {
      Error::Throw("index {} is out of range", i);
    }

    return (mea.bits >> i) & uint64_t{1};
  }

  /**
   * Assigns a sequence of evaluated expressions to the symbol at runtime.
   * For scalar types (`Int` or `Float`), assigns the first expression `es[0]`.
   * For vector types (`IntVector` or `FloatVector`), evaluates and updates
   * elements element-wise up to `std::min(es.size(), val.size())`.
   * @tparam SymTable Symbol table type used for evaluating expressions.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param es Vector of expressions to evaluate and assign.
   * @throws std::runtime_error If trying to assign a float to an integer
   *   symbol/vector or if attempting to assign to a `Mea` symbol.
   */
  template <typename SymTable>
  void Assign(const SymTable& symtab, const std::vector<Expr>& es) {
    auto f = [this, &symtab, &es](auto&& val) {
      using V = std::decay_t<decltype(val)>;

      if constexpr (std::is_same_v<V, Int>) {
        if (!cc::IsConvertibleToInt(symtab, es[0])) {
          Error::Throw("'float' cannot be assigned to an integer identifer");
        }
        val = EvalIntExpr(symtab, es[0]);
      } else if constexpr (std::is_same_v<V, Float>) {
        val = EvalExpr(symtab, es[0]);
      } else if constexpr (std::is_same_v<V, IntVector>) {
        auto size = std::min(es.size(), val.size());
        for (unsigned i = 0; i < size; ++i) {
          if (!cc::IsConvertibleToInt(symtab, es[i])) {
            Error::Throw("'float' cannot be assigned to an integer identifer");
          }
          val[i] = EvalIntExpr(symtab, es[i]);
        }
      } else if constexpr (std::is_same_v<V, FloatVector>) {
        auto size = std::min(es.size(), val.size());
        for (unsigned i = 0; i < size; ++i) {
          val[i] = EvalExpr(symtab, es[i]);
        }
      } else {
        Error::Throw("cannot assign to a measurement identifier");
      }
    };

    std::visit(f, val);
  }

  /**
   * Assigns an evaluated expression to a specific indexed element of a vector.
   * Evaluates the index expression first to determine the target position,
   * then evaluates the provided expression e and assigns it to `IntVector[i]`
   * or `FloatVector[i]`.
   * @tparam SymTable Symbol table type used for evaluation.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param e Expression to evaluate and assigne.
   * @param index Target index expression.
   * @throws std::runtime_error If the evaluated index is out of bounds,
   *   if assigning float to an integer vector, or if called on non-indexable
   *   scalars, or if attempting to assign to a `Mea` symbol.
   */
  template <typename SymTable>
  void Assign(const SymTable& symtab, const Expr& e, const Index& index) {
    std::size_t i = EvalIndex(symtab, index);

    auto g = [this, &symtab, &e, i](auto&& val) {
      using V = std::decay_t<decltype(val)>;

      if constexpr (std::is_same_v<V, IntVector>) {
        if (!cc::IsConvertibleToInt(symtab, e)) {
          Error::Throw("'float' cannot be assigned to an integer identifer");
        }

        if (i >= val.size()) {
          Error::Throw("index {} is out of range", i);
        }

        val[i] = EvalIntExpr(symtab, e);
      } else if constexpr (std::is_same_v<V, FloatVector>) {
        if (i >= val.size()) {
          Error::Throw("index {} is out of range", i);
        }

        val[i] = EvalExpr(symtab, e);
      } else {
        Error::Throw("cannot assign to a measurement identifier");
      }
    };

    std::visit(g, val);
  }

  /**
   * Replaces the stored scalar Int value.
   * @note Caller must ensure the variant currently holds an Int.
   */
  void Assign(Int v) {
    std::get<Int>(val) = v;
  }

  /**
   * Replaces the stored scalar Float value.
   * @note Caller must ensure the variant currently holds a Float.
   */
  void Assign(Float v) {
    std::get<Float>(val) = v;
  }

  /**
   * Mutates an element at index i in an IntVector.
   * @note Caller must ensure variant currently holds an IntVector and i
   *   is valid.
   */
  void Assign(Int v, std::size_t i) {
    std::get<IntVector>(val)[i] = v;
  }

  /**
   * Mutates an element at index @p i in a FloatVector.
   * @note Caller must ensure variant currently holds a FloatVector and i
   *   is valid.
   */
  void Assign(Float v, std::size_t i) {
    std::get<FloatVector>(val)[i] = v;
  }

 private:
  Val val;
  bool read_only = true;
};

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_SYMBOL_H_
