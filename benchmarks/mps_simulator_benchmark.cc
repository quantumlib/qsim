// Copyright 2025 Google LLC
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

// Micro-benchmarks for MPSSimulator::ApplyGate*.

#include <cstdint>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"

#include "../lib/formux.h"
#include "../lib/mps_simulator.h"

namespace qsim {
namespace mps {

namespace {

using Simulator = MPSSimulator<For, float>;
using StateSpace = Simulator::MPSStateSpace_;

// Hadamard (row-major, interleaved real/imaginary).
const float kHadamard[8] = {
    0.70710678f, 0, 0.70710678f,  0,
    0.70710678f, 0, -0.70710678f, 0,
};

// (H x H) * CZ, a 4x4 unitary (row-major, interleaved real/imaginary).
const float kHHCZ[32] = {
    0.5f, 0, 0.5f,  0, 0.5f,  0, -0.5f, 0,
    0.5f, 0, -0.5f, 0, 0.5f,  0, 0.5f,  0,
    0.5f, 0, 0.5f,  0, -0.5f, 0, 0.5f,  0,
    0.5f, 0, -0.5f, 0, -0.5f, 0, -0.5f, 0,
};

// Creates a state filled with small random values, so that the bond
// dimensions are fully populated (as opposed to a product state, which
// would make the SVD in ApplyGate2 unrealistically cheap).
StateSpace::MPS RandomState(unsigned num_qubits, unsigned bond_dim) {
  auto state = StateSpace::Create(num_qubits, bond_dim);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  const auto size = StateSpace::Size(state);
  for (unsigned i = 0; i < size; ++i) {
    state.get()[i] = dist(rng);
  }
  return state;
}

// Args: {num_qubits, bond_dim, target qubit}.
void BM_ApplyGate1(benchmark::State& bm_state) {
  const unsigned num_qubits = bm_state.range(0);
  const unsigned bond_dim = bm_state.range(1);
  const unsigned target = bm_state.range(2);

  Simulator sim(1);
  auto state = RandomState(num_qubits, bond_dim);
  std::vector<unsigned> qs = {target};

  for (auto _ : bm_state) {
    sim.ApplyGate(qs, kHadamard, state);
    benchmark::ClobberMemory();
  }
}

// Args: {num_qubits, bond_dim, first target qubit}. The second target is
// always the next qubit.
void BM_ApplyGate2(benchmark::State& bm_state) {
  const unsigned num_qubits = bm_state.range(0);
  const unsigned bond_dim = bm_state.range(1);
  const unsigned target = bm_state.range(2);

  Simulator sim(1);
  auto state = RandomState(num_qubits, bond_dim);
  std::vector<unsigned> qs = {target, target + 1};

  // The gate is applied repeatedly, in place. For the gate and sizes used
  // here, this gives the same timings as restoring the initial state before
  // every iteration, so the benchmark does not reset the state.
  for (auto _ : bm_state) {
    sim.ApplyGate(qs, kHHCZ, state);
    benchmark::ClobberMemory();
  }
}

constexpr int kNumQubits = 10;

// One-qubit gates: left edge, interior and right edge, for several bond dims.
auto Gate1Args = [](auto* b) {
  for (int bond_dim : {4, 16, 64}) {
    for (int target : {0, kNumQubits / 2, kNumQubits - 1}) {
      b->Args({kNumQubits, bond_dim, target});
    }
  }
};

// Two-qubit gates: left edge, interior and right edge.
auto Gate2Args = [](auto* b) {
  for (int bond_dim : {4, 16, 64}) {
    for (int target : {0, kNumQubits / 2, kNumQubits - 2}) {
      b->Args({kNumQubits, bond_dim, target});
    }
  }
};

BENCHMARK(BM_ApplyGate1)->Apply(Gate1Args);
BENCHMARK(BM_ApplyGate2)->Apply(Gate2Args);

}  // namespace

}  // namespace mps
}  // namespace qsim

BENCHMARK_MAIN();
