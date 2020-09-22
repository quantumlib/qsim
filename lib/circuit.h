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
 * A collection of gates. This object is consumed by `QSim[h]Runner.Run()`.
 */
template <typename Gate>
struct Circuit {
  unsigned num_qubits;
  /**
   * The set of gates to be run. Gate times should be ordered.
   */
  std::vector<Gate> gates;
};

}  // namespace qsim

#endif  // CIRCUIT_H_
