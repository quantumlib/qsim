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

#ifndef QUBIT_MAPPED_STATE_H_
#define QUBIT_MAPPED_STATE_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "qubit_remap.h"

namespace qsim {

// Bidirectional logical<->physical qubit map; the two arrays are inverse
// permutations of each other at all times. "Physical position p" means
// amplitude-index bit p of the state storage.
class QubitLayout {
 public:
  explicit QubitLayout(unsigned num_qubits)
      : logical_to_physical_(num_qubits),
        physical_to_logical_(num_qubits) {
    for (unsigned i = 0; i < num_qubits; ++i) {
      logical_to_physical_[i] = i;
      physical_to_logical_[i] = i;
    }
  }

  std::size_t NumQubits() const { return logical_to_physical_.size(); }

  unsigned PhysicalPositionOf(unsigned logical_qubit) const {
    return logical_to_physical_[logical_qubit];
  }

  unsigned LogicalQubitAt(unsigned physical_position) const {
    return physical_to_logical_[physical_position];
  }

  const std::vector<unsigned>& LogicalToPhysical() const {
    return logical_to_physical_;
  }

  uint64_t PhysicalIndexOf(uint64_t logical_index) const {
    uint64_t physical_index = 0;
    for (unsigned q = 0; q < NumQubits(); ++q) {
      if ((logical_index >> q) & 1) {
        physical_index |= uint64_t{1} << PhysicalPositionOf(q);
      }
    }
    return physical_index;
  }

  bool IsIdentityAt(unsigned position) const {
    return physical_to_logical_[position] == position;
  }

  void SwapPositions(unsigned position_a, unsigned position_b) {
    const auto qubit_a = physical_to_logical_[position_a];
    const auto qubit_b = physical_to_logical_[position_b];
    std::swap(physical_to_logical_[position_a],
              physical_to_logical_[position_b]);
    std::swap(logical_to_physical_[qubit_a],
              logical_to_physical_[qubit_b]);
  }

  // Builds one disjoint restoration pass and updates this layout to match.
  bool BuildIdentityRestorationSwaps(std::vector<QubitSwap>& swaps) {
    swaps.clear();
    std::vector<char> position_touched(NumQubits(), 0);

    for (unsigned position = 0; position < NumQubits(); ++position) {
      if (IsIdentityAt(position)) continue;
      const auto paired_position = PhysicalPositionOf(position);
      if (position_touched[position] ||
          position_touched[paired_position]) {
        continue;
      }

      swaps.emplace_back(position, paired_position);
      position_touched[position] = 1;
      position_touched[paired_position] = 1;
      SwapPositions(position, paired_position);
    }
    return !swaps.empty();
  }

 private:
  std::vector<unsigned> logical_to_physical_;
  std::vector<unsigned> physical_to_logical_;
};

// Owns physical state storage together with its logical-qubit interpretation.
// A newly created mapped state starts in canonical one-to-one qubit order.
template <typename State>
struct QubitMappedState {
  explicit QubitMappedState(State&& state_storage)
      : state(std::move(state_storage)), layout(state.num_qubits()) {}

  template <typename StateSpace>
  auto GetAmpl(const StateSpace& state_space, uint64_t logical_index) const {
    return state_space.GetAmpl(state, layout.PhysicalIndexOf(logical_index));
  }

  State state;
  QubitLayout layout;
};

}  // namespace qsim

#endif  // QUBIT_MAPPED_STATE_H_
