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

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>

#include "gtest/gtest.h"

#include "../lib/circuit_reader.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_def.h"
#include "../lib/io.h"
#include "../lib/parfor.h"
#include "../lib/run_qsim.h"
#include "../lib/simulator_avx.h"
#include "../lib/statespace_avx.h"

namespace qsim {

TEST(StateSpaceAVXTest, SamplingBasic) {
  unsigned num_qubits = 3;
  uint64_t num_samples = 10000000;
  constexpr uint64_t size = 8;

  using StateSpace = StateSpaceAVX<ParallelFor>;
  using State = StateSpace::State;

  StateSpace state_space(num_qubits, 1);
  State state = state_space.CreateState();

  EXPECT_EQ(state_space.IsNull(state), false);
  EXPECT_EQ(state_space.Size(state), size);

  std::array<float, size> ps = {0.1, 0.2, 0.13, 0.12, 0.18, 0.15, 0.07, 0.05};

  for (uint64_t i = 0; i < size; ++i) {
    auto r = std::sqrt(ps[i]);
    state_space.SetAmpl(state, i, r * std::cos(i), r * std::sin(i));
  }

  auto samples = state_space.Sample(state, num_samples, 1);

  EXPECT_EQ(samples.size(), num_samples);
  std::array<double, size> bins = {0, 0, 0, 0, 0, 0, 0, 0};

  for (auto sample : samples) {
    ASSERT_LT(sample, size);
    bins[sample] += 1;
  }

  for (uint64_t i = 0; i < size; ++i) {
    EXPECT_NEAR(bins[i] / num_samples, ps[i], 2e-4);
  }
}

TEST(StateSpaceAVXTest, SamplingCrossEntropyDifference) {
  unsigned depth = 30;
  std::string circuit_file = "../circuits/circuit_q24";
  uint64_t num_samples = 10000000;

  Circuit<GateQSim<float>> circuit;
  EXPECT_EQ(CircuitReader<IO>::FromFile(depth, circuit_file, circuit), true);

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<GateQSim<float>>, Simulator>;

  StateSpace state_space(circuit.num_qubits, 1);
  State state = state_space.CreateState();

  EXPECT_EQ(state_space.IsNull(state), false);

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.num_threads = 1;
  param.verbosity = 0;

  EXPECT_EQ(Runner::Run(param, depth, circuit, state), true);

  auto bitstrings = state_space.Sample(state, num_samples, 1);
  EXPECT_EQ(bitstrings.size(), num_samples);

  double sum = 0;
  for (uint64_t i = 0; i < num_samples; ++i) {
    double p = std::norm(state_space.GetAmpl(state, bitstrings[i]));
    sum += std::log(p);
  }

  double gamma = 0.5772156649;
  double ced = circuit.num_qubits * std::log(2) + gamma + sum / num_samples;

  EXPECT_NEAR(ced, 1.0, 1e-3);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
