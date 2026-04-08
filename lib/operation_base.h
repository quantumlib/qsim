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

#ifndef OPERATION_BASE_H_
#define OPERATION_BASE_H_

#include <cstdio>
#include <exception>
#include <type_traits>
#include <variant>
#include <vector>

namespace qsim {

using Qubits = std::vector<unsigned>;

/**
 * A base operation acting on a specific set of qubits at a given time step.
 * Note: The `kind` field is currently utilized only in qsimh.
 */
struct BaseOperation {
  unsigned kind;
  unsigned time;
  Qubits qubits;
};

namespace detail {

template <typename V, typename A>
struct append_to_variant {};

template <typename... T, typename A>
struct append_to_variant<std::variant<T...>, A> {
  using type = std::variant<T..., A>;
};

template <typename V, typename A>
using append_to_variant_t = typename append_to_variant<V, A>::type;

template <typename V, typename A>
struct is_type_in_variant {};

template <typename... T, typename A>
struct is_type_in_variant<std::variant<T...>, A>
    : std::disjunction<std::is_same<T, A>...> {};

template <typename V, typename A>
inline constexpr bool is_type_in_variant_v = is_type_in_variant<V, A>::value;

template <typename T>
struct is_variant : std::false_type {};

template <typename... T>
struct is_variant<std::variant<T...>> : std::true_type {};

template <typename T>
inline constexpr bool is_variant_v = is_variant<T>::value;

}  // namespace detail

/**
 * Recursively searches for a value of type T within a (potentially nested)
 * variant. Traverses the provided `op` to find an alternative that matches
 * type T. This function handles recursive `std::variant` structures and can
 * transparently dereference pointers (e.g., `std::variant<A, B*>`;
 * note, however, that `std::variant<A, const B*>` does not work) to
 * find the target type.
 *
 * @tparam T The target type to extract.
 * @tparam Operation One of gate types, an `std::variant`, or a pointer
 *   to one of these.
 * @param op The input operation, variant container, or pointer to search.
 * @return A pointer to the value of type T if found; otherwise, nullptr.
 */
template <typename T, typename Operation>
T* OpGetAlternative(Operation& op) {
  using O = std::decay_t<decltype(op)>;

  auto f = [](auto& v) -> T* {
    return OpGetAlternative<T>(v);
  };

  if constexpr (std::is_pointer_v<O>) {
    if constexpr (std::is_const_v<std::remove_pointer_t<O>>) {
      return nullptr;
    } else {
      return OpGetAlternative<T>(*op);
    }
  } else if constexpr (std::is_same_v<T, O>) {
    return &op;
  } else if constexpr (detail::is_variant_v<O>) {
    if constexpr (detail::is_type_in_variant_v<O, T>) {
      if (auto* p = std::get_if<T>(&op)) {
        return p;
      } else {
        return std::visit(f, op);
      }
    } else if constexpr (detail::is_type_in_variant_v<O, T*>) {
      if (auto* p = std::get_if<T*>(&op)) {
        return *p;
      } else {
        return std::visit(f, op);
      }
    } else {
      return std::visit(f, op);
    }
  } else {
    return nullptr;
  }
}

/**
 * Recursively searches for a value of type T within a (potentially nested)
 * variant. Traverses the provided `op` to find an alternative that matches
 * type T. This function handles recursive `std::variant` structures and can
 * transparently dereference pointers (e.g., `std::variant<A, const B*>` or
 * `std::variant<A, B*>`) to find the target type.
 *
 * @tparam T The target type to extract.
 * @tparam Operation One of gate types, an `std::variant`, or a pointer
 *   to one of these.
 * @param op The input operation, variant container, or pointer to search.
 * @return A const pointer to the value of type T if found; otherwise, nullptr.
 */
template <typename T, typename Operation>
const T* OpGetAlternative(const Operation& op) {
  using O = std::decay_t<decltype(op)>;

  auto f = [](const auto& v) -> const T* {
    return OpGetAlternative<T>(v);
  };

  if constexpr (std::is_pointer_v<O>) {
    return OpGetAlternative<T>(*op);
  } else if constexpr (std::is_same_v<T, O>) {
    return &op;
  } else if constexpr (detail::is_variant_v<O>) {
    if constexpr (detail::is_type_in_variant_v<O, T>) {
      if (const auto* p = std::get_if<T>(&op)) {
        return p;
      } else {
        return std::visit(f, op);
      }
    } else if constexpr (detail::is_type_in_variant_v<O, const T*>) {
      if (const auto* p = std::get_if<const T*>(&op)) {
        return *p;
      } else {
        return std::visit(f, op);
      }
    } else {
      return std::visit(f, op);
    }
  } else {
    return nullptr;
  }
}

/**
 * Recursively retrieves the time step from a (nested) variant.
 * This function traverses the provided `op` (concrete type, pointer,
 * or variant) to access the `time` memeber in the underlying `BaseOperation`.
 * This function handles recursive `std::variant` structures and can
 * transparently dereference pointers (e.g., `std::variant<A, const B*>`
 * or `std::variant<A, B*>`) to find the target value.
 *
 * Note: this function assumes that the input contains a type that is
 * derived from `BaseOperation`.
 *
 * @tparam Operation A type derived from `BaseOperation`, an `std::variant`,
 *   or a pointer to one of these.
 * @param op The input operation, variant container, or pointer to search.
 * @return The time step of the underlying `BaseOperation`.
 */
template <typename Operation>
inline unsigned OpTime(const Operation& op) {
  using O = std::decay_t<decltype(op)>;

  if constexpr (std::is_pointer_v<O>) {
    return OpTime(*op);
  } else if constexpr (std::is_base_of_v<BaseOperation, O>) {
    return op.time;
  } else if constexpr (detail::is_variant_v<O>) {
    auto f = [](const auto& v) {
      return OpTime(v);
    };

    return std::visit(f, op);
  } else {
    static_assert(0, "OpQubits encountered an invalid type");
  }
}

/**
 * Recursively retrieves the qubit indices from a (nested) variant.
 * This function traverses the provided `op` (concrete type, pointer,
 * or variant) to access the `qubits` memeber in the underlying `BaseOperation`.
 * This function handles recursive `std::variant` structures and can
 * transparently dereference pointers (e.g., `std::variant<A, const B*>`
 * or `std::variant<A, B*>`) to find the target value.
 *
 * Note: this function assumes that the input contains a type that is
 * derived from `BaseOperation`.
 *
 * @tparam Operation A type derived from `BaseOperation`, an `std::variant`,
 *   or a pointer to one of these.
 * @param op The input operation, variant container, or pointer to search.
 * @return A const reference to qubit indices of the underlying `BaseOperation`.
 */
template <typename Operation>
inline const Qubits& OpQubits(const Operation& op) {
  using O = std::decay_t<decltype(op)>;

  if constexpr (std::is_pointer_v<O>) {
    return OpQubits(*op);
  } else if constexpr (std::is_base_of_v<BaseOperation, O>) {
    return op.qubits;
  } else if constexpr (detail::is_variant_v<O>) {
    auto f = [](const auto& v) -> const Qubits& {
      return OpQubits(v);
    };

    return std::visit(f, op);
  } else {
    static_assert(0, "OpQubits encountered an invalid type");
  }
}

/**
 * Recursively retrieves the BaseOperation from a (nested) variant.
 * This function traverses the provided `op` (concrete type, pointer,
 * or variant) to access the underlying BaseOperation.
 * This function handles recursive `std::variant` structures and can
 * transparently dereference pointers (e.g., `std::variant<A, B*>`;
 * note, however, that `std::variant<A, const B*>` does not work) to
 * find the target value.
 *
 * Note: this function assumes that the input contains a type that is
 * derived from `BaseOperation`.
 *
 * @tparam Operation A type derived from `BaseOperation`, an `std::variant`,
 *   or a pointer to one of these.
 * @param op The input operation, variant container, or pointer to search.
 * @return A reference to the underlying `BaseOperation`.
 */
template <typename Operation>
inline BaseOperation& OpBaseOperation(Operation& op) {
  using O = std::decay_t<decltype(op)>;

  if constexpr (std::is_pointer_v<O>) {
    return OpBaseOperation(*op);
  } else if constexpr (std::is_base_of_v<BaseOperation, O>) {
    return op;
  } else if constexpr (detail::is_variant_v<O>) {
    auto f = [](auto& v) -> BaseOperation& {
      return OpBaseOperation(v);
    };

    return std::visit(f, op);
  } else {
    static_assert(0, "OpBaseOperation encountered an invalid type");
  }
}

/**
 * Recursively retrieves the BaseOperation from a (nested) variant.
 * This function traverses the provided `op` (concrete type, pointer,
 * or variant) to access the underlying BaseOperation.
 * This function handles recursive `std::variant` structures and can
 * transparently dereference pointers (e.g., `std::variant<A, const B*>`
 * or `std::variant<A, B*>`) to find the target value.
 *
 * Note: this function assumes that the input contains a type that is
 * derived from `BaseOperation`.
 *
 * @tparam Operation A type derived from `BaseOperation`, an `std::variant`,
 *   or a pointer to one of these.
 * @param op The input operation, variant container, or pointer to search.
 * @return A const reference to the underlying `BaseOperation`.
 */
template <typename Operation>
inline const BaseOperation& OpBaseOperation(const Operation& op) {
  using O = std::decay_t<decltype(op)>;

  if constexpr (std::is_pointer_v<O>) {
    return OpBaseOperation(*op);
  } else if constexpr (std::is_base_of_v<BaseOperation, O>) {
    return op;
  } else if constexpr (detail::is_variant_v<O>) {
    auto f = [](const auto& v) -> const BaseOperation& {
      return OpBaseOperation(v);
    };

    return std::visit(f, op);
  } else {
    static_assert(0, "OpBaseOperation encountered an invalid type");
  }
}

}  // namespace qsim

#endif  // OPERATION_BASE_H_
