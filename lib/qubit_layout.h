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

#ifndef QUBIT_LAYOUT_H_
#define QUBIT_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace qsim {

struct PositionSwap {
  unsigned pos_a;
  unsigned pos_b;
};

// Bidirectional mapping between logical qubits and state storage bit positions.
class QubitLayout {
 public:
  explicit QubitLayout(unsigned num_qubits)
      : logical_to_physical_(num_qubits), physical_to_logical_(num_qubits) {
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

  uint64_t PhysicalAmplitudeIndex(uint64_t logical_index) const {
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

  void SwapPhysicalPositions(unsigned position_a, unsigned position_b) {
    const auto qubit_a = physical_to_logical_[position_a];
    const auto qubit_b = physical_to_logical_[position_b];
    std::swap(
        physical_to_logical_[position_a], physical_to_logical_[position_b]);
    std::swap(logical_to_physical_[qubit_a], logical_to_physical_[qubit_b]);
  }

  // Returns one parallel round of disjoint swaps by reversing each cycle.
  // Calling this in a loop restores the identity layout in at most 2 calls.
  std::vector<PositionSwap> StepTowardIdentity() {
    std::vector<PositionSwap> swaps;
    std::vector<char> visited(NumQubits(), 0);

    for (unsigned start = 0; start < NumQubits(); ++start) {
      if (visited[start] || IsIdentityAt(start)) continue;
      ReverseCycle(TraceCycle(start, visited), swaps);
    }
    return swaps;
  }

 private:
  std::vector<unsigned> TraceCycle(
      unsigned start, std::vector<char>& visited) const {
    std::vector<unsigned> cycle;
    for (unsigned pos = start; !visited[pos]; pos = PhysicalPositionOf(pos)) {
      visited[pos] = 1;
      cycle.push_back(pos);
    }
    return cycle;
  }

  void ReverseCycle(
      const std::vector<unsigned>& cycle,
      std::vector<PositionSwap>& swaps) {
    for (std::size_t i = 0; i < cycle.size() / 2; ++i) {
      const unsigned pos_a = cycle[i];
      const unsigned pos_b = cycle[cycle.size() - 1 - i];
      swaps.push_back({pos_a, pos_b});
      SwapPhysicalPositions(pos_a, pos_b);
    }
  }

  std::vector<unsigned> logical_to_physical_;
  std::vector<unsigned> physical_to_logical_;
};

template <typename State>
struct QubitMappedState {
  explicit QubitMappedState(State&& state_storage)
      : state(std::move(state_storage)), layout(state.num_qubits()) {}

  template <typename StateSpace>
  auto GetAmpl(const StateSpace& state_space, uint64_t logical_index) const {
    return state_space.GetAmpl(
        state, layout.PhysicalAmplitudeIndex(logical_index));
  }

  State state;
  QubitLayout layout;
};

}  // namespace qsim

#endif  // QUBIT_LAYOUT_H_
