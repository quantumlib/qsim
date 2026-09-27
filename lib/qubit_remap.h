// Qubit remapping for cache-blocked simulation.
//
// In-place, single-pass application of a set of DISJOINT qubit-position
// transpositions (an involution) to a state stored in a chunked SIMD
// layout. Disjoint transpositions commute, so one pass over the state
// (1 read + 1 write of the moved amplitudes) applies them all.
//
// Layout: a chunked SIMD state space stores amplitudes as chunks of
//
//   chunk = [re0 .. re(N-1), im0 .. im(N-1)]     N = 2^chunk_qubits
//
// i.e. the low `chunk_qubits` amplitude-index bits select the lane inside a
// chunk (NEON/SSE: chunk_qubits=2, N=4; AVX2: 3, N=8; AVX512: 4, N=16; a
// non-chunked/fully-interleaved layout is the degenerate chunk_qubits=0
// case, N=1). All swapped positions must therefore be >= chunk_qubits: the
// permutation then moves whole chunks and never has to touch lanes.
// `chunk_qubits` is an explicit property of the caller's state space.
//
// Execution shape: for the lowest swapped amplitude bit b_min, amplitudes
// move as contiguous spans of 2^(b_min - chunk_qubits) chunks. The pass
// enumerates chunk-span starts, computes each span's partner by flipping
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

namespace qsim {

// A transposition of two amplitude-index bit positions.
using QubitSwap = std::pair<unsigned, unsigned>;

namespace remap_internal {

// One amplitude-bit swap translated to chunk-index bit positions.
struct ChunkBitSwap {
  unsigned lower_bit;
  unsigned upper_bit;
  uint64_t flip_mask;  // Both bits set.
};

// True when the swaps are pairwise-disjoint transpositions of distinct
// positions in [chunk_qubits, num_qubits).
inline bool IsDisjointSwapSet(const std::vector<QubitSwap>& qubit_swaps,
                              unsigned num_qubits, unsigned chunk_qubits) {
  std::vector<char> is_swapped(num_qubits, 0);
  for (const auto& [first, second] : qubit_swaps) {
    if (first == second) return false;
    for (unsigned position : {first, second}) {
      if (position < chunk_qubits || position >= num_qubits) return false;
      if (is_swapped[position]) return false;
      is_swapped[position] = 1;
    }
  }
  return true;
}

// The chunk each chunk exchanges with: every swapped bit pair whose two bits
// differ is flipped. A chunk whose paired bits all match maps to itself.
inline uint64_t PartnerChunk(uint64_t chunk,
                             const std::vector<ChunkBitSwap>& bit_swaps) {
  auto partner = chunk;
  for (const ChunkBitSwap& bit_swap : bit_swaps) {
    if (((chunk >> bit_swap.lower_bit) ^ (chunk >> bit_swap.upper_bit)) & 1) {
      partner ^= bit_swap.flip_mask;
    }
  }
  return partner;
}

// Swap two contiguous chunk spans. Compilers vectorize this to SIMD-width
// loads and stores; the loop is generally memory-bandwidth-bound.
// Keep the __restrict loop: std::swap_ranges measured consistently 1-3%
// slower swap passes.
inline void SwapChunkSpans(float* __restrict first_span,
                           float* __restrict second_span,
                           uint64_t num_floats) {
  for (uint64_t i = 0; i < num_floats; ++i) {
    float temporary = first_span[i];
    first_span[i] = second_span[i];
    second_span[i] = temporary;
  }
}

}  // namespace remap_internal

// Applies all `qubit_swaps` transpositions of amplitude-bit positions to the
// state, in place, in a single pass. The pairs must be disjoint and every
// position must be >= chunk_qubits (see layout comment above).
inline void ApplyBitPairSwaps(float* state, unsigned num_qubits,
                              unsigned chunk_qubits,
                              const std::vector<QubitSwap>& qubit_swaps,
                              unsigned num_threads) {
  namespace ri = remap_internal;

  if (qubit_swaps.empty()) return;
  assert(ri::IsDisjointSwapSet(qubit_swaps, num_qubits, chunk_qubits));

  // Amplitudes move in contiguous spans of 2^span_bits chunks, where
  // span_bits is the lowest swapped chunk-index bit.
  std::vector<ri::ChunkBitSwap> bit_swaps;
  bit_swaps.reserve(qubit_swaps.size());
  unsigned span_bits = num_qubits - chunk_qubits;
  for (const auto& [first, second] : qubit_swaps) {
    const unsigned lower = std::min(first, second) - chunk_qubits;
    const unsigned upper = std::max(first, second) - chunk_qubits;
    bit_swaps.push_back(
        {lower, upper, (uint64_t{1} << lower) | (uint64_t{1} << upper)});
    span_bits = std::min(span_bits, lower);
  }

  const uint64_t floats_per_chunk = uint64_t{2} << chunk_qubits;
  const uint64_t floats_per_span = floats_per_chunk << span_bits;
  const int64_t num_spans =
      int64_t{1} << (num_qubits - chunk_qubits - span_bits);

  // Static scheduling divides spans deterministically across threads, avoiding
  // atomic dispatch lock contention and preserving hardware prefetch streams.
#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int64_t span = 0; span < num_spans; ++span) {
    const uint64_t chunk = uint64_t(span) << span_bits;
    const uint64_t partner = ri::PartnerChunk(chunk, bit_swaps);

    // Each moved pair is visited from both sides; act on one of them
    // (fixed points have partner == chunk and fall through).
    if (partner > chunk) {
      ri::SwapChunkSpans(state + chunk * floats_per_chunk,
                         state + partner * floats_per_chunk, floats_per_span);
    }
  }
}

}  // namespace qsim

#endif  // QUBIT_REMAP_H_
