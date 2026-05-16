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

#include <type_traits>
#include <variant>
#include <vector>

#include "channel.h"
#include "gate.h"
#include "operation_base.h"

namespace qsim {

// Forward declaration of classically controlled operation.
template <typename>
struct ClassicallyControlledOperation;

/**
 * A generic operation.
 */
template <typename FP>
using Operation = std::variant<Gate<FP>, ControlledGate<FP>, Measurement,
                               Channel<FP>, ClassicallyControlledOperation<FP>>;

/**
 * A classically controlled operation. Not implemented yet.
 */
template <typename FP>
struct ClassicallyControlledOperation : public BaseOperation {
  std::vector<Operation<FP>> sub_ops;
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
