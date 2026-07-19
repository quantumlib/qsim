// Qubit remapping for cache-blocked simulation (refactoring plan, Step 1).
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
//
// Reading guide: ApplyBitPairSwaps (the only public entry point) is
// BitSwapPlan construction followed by a parallel loop of
//   FirstChunkOfSpan -> PartnerChunk -> SwapChunkSpans.

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
class ChunkBitSwap {
 public:
  ChunkBitSwap(unsigned first_amplitude_bit, unsigned second_amplitude_bit,
               unsigned chunk_qubits) {
    assert(first_amplitude_bit >= chunk_qubits);
    assert(second_amplitude_bit >= chunk_qubits);
    lower_chunk_bit_ =
        std::min(first_amplitude_bit, second_amplitude_bit) - chunk_qubits;
    upper_chunk_bit_ =
        std::max(first_amplitude_bit, second_amplitude_bit) - chunk_qubits;
    flip_mask_ = (uint64_t{1} << lower_chunk_bit_) |
                 (uint64_t{1} << upper_chunk_bit_);
  }

  unsigned LowerChunkBit() const { return lower_chunk_bit_; }

  // A bit-position swap moves the chunk only when the two bits differ.
  bool ChangesChunkIndex(uint64_t chunk_index) const {
    return ((chunk_index >> lower_chunk_bit_) & 1) !=
           ((chunk_index >> upper_chunk_bit_) & 1);
  }

  uint64_t FlipMask() const { return flip_mask_; }

 private:
  unsigned lower_chunk_bit_;
  unsigned upper_chunk_bit_;
  uint64_t flip_mask_;
};

// Precomputed chunk geometry for applying a set of disjoint qubit swaps.
class BitSwapPlan {
 public:
  BitSwapPlan(unsigned num_qubits, unsigned chunk_qubits,
              const std::vector<QubitSwap>& qubit_swaps)
      : floats_per_chunk_(uint64_t{2} << chunk_qubits),
        chunk_index_bits_(num_qubits - chunk_qubits),
        chunk_span_bits_(chunk_index_bits_) {
    assert(num_qubits >= chunk_qubits);
#ifndef NDEBUG
    std::vector<char> is_qubit_swapped(num_qubits, 0);
#endif
    chunk_bit_swaps_.reserve(qubit_swaps.size());
    for (const QubitSwap& qubit_swap : qubit_swaps) {
      assert(qubit_swap.first < num_qubits);
      assert(qubit_swap.second < num_qubits);
      assert(qubit_swap.first != qubit_swap.second);
#ifndef NDEBUG
      assert(!is_qubit_swapped[qubit_swap.first]);
      assert(!is_qubit_swapped[qubit_swap.second]);
      is_qubit_swapped[qubit_swap.first] = 1;
      is_qubit_swapped[qubit_swap.second] = 1;
#endif
      AddQubitSwap(qubit_swap, chunk_qubits);
    }
  }

  // Floats in one chunk (2^chunk_qubits amplitudes, 2 floats each).
  uint64_t FloatsPerChunk() const { return floats_per_chunk_; }

  // Amplitudes move in contiguous spans of 2^chunk_span_bits chunks.
  uint64_t FloatsPerChunkSpan() const {
    return floats_per_chunk_ << chunk_span_bits_;
  }

  int64_t NumChunkSpans() const {
    return int64_t{1} << (chunk_index_bits_ - chunk_span_bits_);
  }

  uint64_t FirstChunkOfSpan(int64_t span_index) const {
    return uint64_t(span_index) << chunk_span_bits_;
  }

  // Applies every chunk-bit swap to find the involution partner. A chunk
  // whose paired bits all match maps to itself.
  uint64_t PartnerChunk(uint64_t chunk_index) const {
    auto partner_chunk = chunk_index;
    for (const ChunkBitSwap& bit_swap : chunk_bit_swaps_) {
      if (bit_swap.ChangesChunkIndex(chunk_index)) {
        partner_chunk ^= bit_swap.FlipMask();
      }
    }
    return partner_chunk;
  }

 private:
  void AddQubitSwap(const QubitSwap& qubit_swap, unsigned chunk_qubits) {
    chunk_bit_swaps_.emplace_back(qubit_swap.first, qubit_swap.second,
                                  chunk_qubits);
    chunk_span_bits_ = std::min(
        chunk_span_bits_, chunk_bit_swaps_.back().LowerChunkBit());
  }

  uint64_t floats_per_chunk_;
  std::vector<ChunkBitSwap> chunk_bit_swaps_;
  unsigned chunk_index_bits_;
  unsigned chunk_span_bits_;
};

// First float of the chunk with this index.
inline float* FirstFloatOfChunk(float* state, uint64_t floats_per_chunk,
                                uint64_t chunk_index) {
  return state + floats_per_chunk * chunk_index;
}

// Swap two contiguous chunk spans. Compilers vectorize this to SIMD-width
// loads and stores; the loop is generally memory-bandwidth-bound.
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
inline void ApplyBitPairSwaps(
    float* state, unsigned num_qubits, unsigned chunk_qubits,
    const std::vector<QubitSwap>& qubit_swaps) {
  namespace ri = remap_internal;

  if (qubit_swaps.empty()) {
    return;
  }

  const auto plan = ri::BitSwapPlan(num_qubits, chunk_qubits, qubit_swaps);
  const auto floats_per_chunk = plan.FloatsPerChunk();
  const auto floats_per_chunk_span = plan.FloatsPerChunkSpan();
  const auto num_chunk_spans = plan.NumChunkSpans();

#pragma omp parallel for schedule(dynamic, 32)
  for (int64_t span_index = 0; span_index < num_chunk_spans; ++span_index) {
    const auto first_chunk = plan.FirstChunkOfSpan(span_index);
    const auto partner_chunk = plan.PartnerChunk(first_chunk);

    // Each moved pair is visited from both sides; act on one of them
    // (fixed points have partner == chunk and fall through).
    if (partner_chunk > first_chunk) {
      ri::SwapChunkSpans(
          ri::FirstFloatOfChunk(state, floats_per_chunk, first_chunk),
          ri::FirstFloatOfChunk(state, floats_per_chunk, partner_chunk),
          floats_per_chunk_span);
    }
  }
}

}  // namespace qsim

#endif  // QUBIT_REMAP_H_
