// Copyright 2019 Google LLC. All Rights Reserved.
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

#ifndef CIRCUIT_H_
#define CIRCUIT_H_

#include <vector>

namespace qsim {

/**
 * A collection of operations.
 */
template <typename Operation>
struct Circuit {
  unsigned num_qubits;
  /**
   * The set of operations to be run. Operation time steps should be ordered.
   */
  std::vector<Operation> ops;
};

/**
 * An adapter for vectors of operations.
 */
template <typename Circuit>
struct Operations;

template <typename Operation>
struct Operations<qsim::Circuit<Operation>> {
  static const std::vector<Operation>& get(
      const qsim::Circuit<Operation>& circuit) {
    return circuit.ops;
  }
};

template <typename Operation>
struct Operations<std::vector<Operation>> {
  static const std::vector<Operation>& get(const std::vector<Operation>& ops) {
    return ops;
  }
};

template <typename Operation>
struct Operations<std::vector<Operation*>> {
  static const std::vector<Operation*>& get(
      const std::vector<Operation*>& ops) {
    return ops;
  }
};

}  // namespace qsim

#endif  // CIRCUIT_H_
