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

// Qubit remapping for cache-tiled simulation.
//
// In-place, single-pass application of a set of DISJOINT qubit-position
// transpositions (an involution) to a state stored in a SIMD lane-group
// layout. Disjoint transpositions commute, so one pass over the state
// (1 read + 1 write of the moved amplitudes) applies them all.
//
// Layout: a SIMD state space stores amplitudes as lane groups of
//
//   group = [re0 .. re(N-1), im0 .. im(N-1)]     N = 2^lane_qubits
//
// i.e. the low `lane_qubits` amplitude-index bits select the SIMD lane inside
// a group (NEON/SSE: lane_qubits=2, N=4; AVX2: 3, N=8; AVX512: 4, N=16; a
// fully interleaved scalar layout is the degenerate lane_qubits=0 case,
// N=1). All swapped positions must therefore be >= lane_qubits: the
// permutation then moves whole groups and never has to touch lanes.
// `lane_qubits` is an explicit property of the caller's state space.
//
// Execution shape: for the lowest swapped amplitude bit b_min, amplitudes
// move as contiguous spans of 2^(b_min - lane_qubits) groups. The pass
// enumerates group-span starts, computes each span's partner by flipping
// the bit pairs whose two bits differ, and swaps the two spans (each span
// is swapped exactly once via the partner > self check; spans whose pair
// bits all match are fixed points and are skipped).

#ifndef QUBIT_REMAP_H_
#define QUBIT_REMAP_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

#include "qubit_layout.h"

namespace qsim {

namespace remap_internal {

// One amplitude-bit swap translated to group-index bit positions.
struct GroupBitSwap {
  unsigned lower_bit;
  unsigned upper_bit;
  uint64_t flip_mask;  // Both bits set.
};

// True when the swaps are pairwise-disjoint transpositions of distinct
// positions in [lane_qubits, num_qubits).
inline bool IsDisjointSwapSet(const std::vector<QubitSwap>& qubit_swaps,
                              unsigned num_qubits, unsigned lane_qubits) {
  std::vector<char> is_swapped(num_qubits, 0);
  for (const auto& [first, second] : qubit_swaps) {
    if (first == second) return false;
    for (unsigned position : {first, second}) {
      if (position < lane_qubits || position >= num_qubits) return false;
      if (is_swapped[position]) return false;
      is_swapped[position] = 1;
    }
  }
  return true;
}

// The group each group exchanges with: every swapped bit pair whose two bits
// differ is flipped. A group whose paired bits all match maps to itself.
inline uint64_t PartnerGroup(uint64_t group,
                             const std::vector<GroupBitSwap>& bit_swaps) {
  auto partner = group;
  for (const GroupBitSwap& bit_swap : bit_swaps) {
    if (((group >> bit_swap.lower_bit) ^ (group >> bit_swap.upper_bit)) & 1) {
      partner ^= bit_swap.flip_mask;
    }
  }
  return partner;
}

// Swap two contiguous lane-group spans. Compilers vectorize this to SIMD-width
// loads and stores; the loop is generally memory-bandwidth-bound.
// Keep the __restrict loop: std::swap_ranges measured consistently 1-3%
// slower swap passes.
inline void SwapGroupSpans(float* __restrict first_span,
                           float* __restrict second_span,
                           uint64_t num_floats) {
  for (uint64_t i = 0; i < num_floats; ++i) {
    const auto temporary = first_span[i];
    first_span[i] = second_span[i];
    second_span[i] = temporary;
  }
}

}  // namespace remap_internal

// Applies all `qubit_swaps` transpositions of amplitude-bit positions to the
// state, in place, in a single pass. The pairs must be disjoint and every
// position must be >= lane_qubits (see layout comment above).
inline void ApplyBitPairSwaps(float* state, unsigned num_qubits,
                              unsigned lane_qubits,
                              const std::vector<QubitSwap>& qubit_swaps,
                              unsigned num_threads) {
  namespace ri = remap_internal;

  if (qubit_swaps.empty()) return;
  assert(ri::IsDisjointSwapSet(qubit_swaps, num_qubits, lane_qubits));

  // Amplitudes move in contiguous spans of 2^span_bits groups, where
  // span_bits is the lowest swapped group-index bit.
  std::vector<ri::GroupBitSwap> bit_swaps;
  bit_swaps.reserve(qubit_swaps.size());
  auto span_bits = num_qubits - lane_qubits;
  for (const auto& [first, second] : qubit_swaps) {
    const auto lower = std::min(first, second) - lane_qubits;
    const auto upper = std::max(first, second) - lane_qubits;
    bit_swaps.push_back(
        {lower, upper, (uint64_t{1} << lower) | (uint64_t{1} << upper)});
    span_bits = std::min(span_bits, lower);
  }

  const auto floats_per_group = uint64_t{2} << lane_qubits;
  const auto floats_per_span = floats_per_group << span_bits;
  const int64_t num_spans =
      int64_t{1} << (num_qubits - lane_qubits - span_bits);

  // Static scheduling divides spans deterministically across threads, avoiding
  // atomic dispatch lock contention and preserving hardware prefetch streams.
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int64_t span = 0; span < num_spans; ++span) {
    const auto group = uint64_t(span) << span_bits;
    const auto partner = ri::PartnerGroup(group, bit_swaps);

    // Each moved pair is visited from both sides; act on one of them
    // (fixed points have partner == group and fall through).
    if (partner > group) {
      ri::SwapGroupSpans(state + group * floats_per_group,
                         state + partner * floats_per_group, floats_per_span);
    }
  }
}

}  // namespace qsim

#endif  // QUBIT_REMAP_H_
