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

// In-place application of disjoint qubit transpositions to a SIMD state space.

#ifndef QUBIT_SWAP_H_
#define QUBIT_SWAP_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "qubit_layout.h"

namespace qsim {

namespace qubit_swap_internal {

struct GroupBitSwap {
  unsigned lower_bit;
  unsigned upper_bit;
  uint64_t flip_mask;
};

inline bool IsDisjointSwapSet(
    const std::vector<PositionSwap>& swaps, unsigned num_qubits,
    unsigned lane_qubits) {
  std::vector<char> is_swapped(num_qubits, 0);
  for (const auto& [pos_a, pos_b] : swaps) {
    if (pos_a == pos_b) return false;
    for (unsigned position : {pos_a, pos_b}) {
      if (position < lane_qubits || position >= num_qubits) return false;
      if (is_swapped[position]) return false;
      is_swapped[position] = 1;
    }
  }
  return true;
}

inline uint64_t PartnerGroup(
    uint64_t group, const std::vector<GroupBitSwap>& bit_swaps) {
  auto partner = group;
  for (const GroupBitSwap& bit_swap : bit_swaps) {
    if (((group >> bit_swap.lower_bit) ^ (group >> bit_swap.upper_bit)) & 1) {
      partner ^= bit_swap.flip_mask;
    }
  }
  return partner;
}

// Keep the __restrict loop: std::swap_ranges measured consistently 1-3% slower.
inline void SwapGroupSpans(
    float* __restrict first_span, float* __restrict second_span,
    uint64_t num_floats) {
  for (uint64_t i = 0; i < num_floats; ++i) {
    const auto temporary = first_span[i];
    first_span[i] = second_span[i];
    second_span[i] = temporary;
  }
}

}  // namespace qubit_swap_internal

// Swaps disjoint pairs of bit positions at or above `lane_qubits`.
// Positions `[0, lane_qubits)` select lanes inside each SIMD lane group
// (`floats_per_group` floats), while positions `[lane_qubits, num_qubits)` are
// bits `[0, num_qubits - lane_qubits)` of the lane-group index.
inline void ApplyBitPairSwaps(
    float* state, unsigned num_qubits, unsigned lane_qubits,
    const std::vector<PositionSwap>& swaps, unsigned num_threads) {
  if (swaps.empty()) return;
  assert(
      qubit_swap_internal::IsDisjointSwapSet(swaps, num_qubits, lane_qubits));

  std::vector<qubit_swap_internal::GroupBitSwap> bit_swaps;
  bit_swaps.reserve(swaps.size());
  auto span_bits = num_qubits - lane_qubits;
  for (const auto& [pos_a, pos_b] : swaps) {
    const auto lower = std::min(pos_a, pos_b) - lane_qubits;
    const auto upper = std::max(pos_a, pos_b) - lane_qubits;
    bit_swaps.push_back(
        {lower, upper, (uint64_t{1} << lower) | (uint64_t{1} << upper)});
    span_bits = std::min(span_bits, lower);
  }

  // Beyond the SIMD lane bits, no swap in this pass touches the lowest
  // `span_bits` bits of the group index, so `2^span_bits` consecutive lane
  // groups (`floats_per_span` contiguous floats) map to a contiguous partner
  // block and can be swapped together.
  const auto floats_per_group = uint64_t{2} << lane_qubits;
  const auto floats_per_span = floats_per_group << span_bits;
  const int64_t num_spans = int64_t{1}
                            << (num_qubits - lane_qubits - span_bits);

  // PartnerGroup is symmetric over disjoint bit swaps: if group A maps to B,
  // then B maps back to A. Checking `partner > group` swaps each pair once at
  // its lower index and skips groups that map to themselves.
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int64_t span = 0; span < num_spans; ++span) {
    const auto group = uint64_t(span) << span_bits;
    const auto partner = qubit_swap_internal::PartnerGroup(group, bit_swaps);

    if (partner > group) {
      qubit_swap_internal::SwapGroupSpans(
          state + group * floats_per_group, state + partner * floats_per_group,
          floats_per_span);
    }
  }
}

}  // namespace qsim

#endif  // QUBIT_SWAP_H_
