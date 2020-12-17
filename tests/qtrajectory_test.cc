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

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_cirq.h"
#include "../lib/io.h"
#include "../lib/qtrajectory.h"
#include "../lib/simmux.h"

namespace qsim {

namespace {

namespace types {

using StateSpace = Simulator<For>::StateSpace;
using State = StateSpace::State;
using fp_type = StateSpace::fp_type;
using Gate = Cirq::GateCirq<fp_type>;
using QTSimulaltor = QuantumTrajectorySimulator<IO, Gate, MultiQubitGateFuser,
                                                Simulator<For>>;
using NoisyCircuit = NoisyCircuit<Gate>;

}  // namespace types

void AddBitFlipNoise(unsigned time, double p, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - p;
  double p2 = p;

  auto normal = KrausOperator<types::Gate>::kNormal;

  ncircuit.push_back({{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, 0)}},
                      {normal, 1, p2, {Cirq::X<fp_type>::Create(time, 0)}}});
  ncircuit.push_back({{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, 1)}},
                      {normal, 1, p2, {Cirq::X<fp_type>::Create(time, 1)}}});
}

void AddPhaseDumpNoise(unsigned time, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {0, 0, 0, 0, 0, 0, s, 0})}}});
  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {0, 0, 0, 0, 0, 0, s, 0})}}});
}

void AddAmplDumpNoise(unsigned time, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {0, 0, s, 0, 0, 0, 0, 0})}}});
  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {0, 0, s, 0, 0, 0, 0, 0})}}});
}

void AddGenAmplDumpNoise(
    unsigned time, double g, types::NoisyCircuit& ncircuit) {
  using fp_type = types::Gate::fp_type;

  // Probability of exchanging energy with the environment.
  double p = 0.5;

  double p1 = p * (1 - g);
  double p2 = (1 - p) * (1 - g);
  double p3 = 0;

  fp_type t1 = std::sqrt(p);
  fp_type r1 = std::sqrt(p * (1 - g));
  fp_type s1 = std::sqrt(p * g);
  fp_type t2 = std::sqrt(1 - p);
  fp_type r2 = std::sqrt((1 - p) * (1 - g));
  fp_type s2 = std::sqrt((1 - p) * g);

  auto normal = KrausOperator<types::Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, 0, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, 0, {0, 0, 0, 0, s2, 0, 0, 0})}}});
  ncircuit.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, 1, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, 1, {0, 0, 0, 0, s2, 0, 0, 0})}}});
}

template <typename AddNoise>
types::NoisyCircuit GenerateNoisyCirquit(double p, AddNoise&& add_noise) {
  using fp_type = types::Gate::fp_type;

  types::NoisyCircuit ncircuit;
  ncircuit.reserve(24);

  using Hd = Cirq::H<fp_type>;
  using IS = Cirq::ISWAP<fp_type>;
  using Rx = Cirq::rx<fp_type>;
  using Ry = Cirq::ry<fp_type>;

  auto normal = KrausOperator<types::Gate>::kNormal;

  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 0)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Hd::Create(0, 1)}}});
  add_noise(1, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {IS::Create(2, 0, 1)}}});
  add_noise(3, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {Rx::Create(4, 0, 0.7)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Ry::Create(4, 1, 0.1)}}});
  add_noise(5, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {IS::Create(6, 0, 1)}}});
  add_noise(7, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {Ry::Create(8, 0, 0.4)}}});
  ncircuit.push_back({{normal, 1, 1.0, {Rx::Create(8, 1, 0.7)}}});
  add_noise(9, p, ncircuit);
  ncircuit.push_back({{normal, 1, 1.0, {IS::Create(10, 0, 1)}}});
  add_noise(11, p, ncircuit);
  ncircuit.push_back({{KrausOperator<types::Gate>::kMeasurement, 1, 1.0,
                       {gate::Measurement<types::Gate>::Create(12, {0, 1})}}});
  add_noise(13, p, ncircuit);

  return ncircuit;
}

void Run1(const types::NoisyCircuit& ncircuit,
          const std::vector<double>& expected_results) {
  unsigned num_qubits = 2;
  unsigned num_reps = 20000;

  auto measure = [](uint64_t r, const types::State& state,
                    const std::vector<uint64_t>& stat,
                    std::vector<unsigned>& histogram) {
    ASSERT_EQ(stat.size(), 1);
    ++histogram[stat[0]];
  };

  std::vector<unsigned> histogram(1 << num_qubits, 0);

  types::QTSimulaltor::Parameter param;
  param.collect_mea_stat = true;

  EXPECT_TRUE(types::QTSimulaltor::Run(param, num_qubits, ncircuit,
                                       0, num_reps, measure, histogram));

  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EXPECT_NEAR(double(histogram[i]) / num_reps, expected_results[i], 0.005);
  }
}

void Run2(const types::NoisyCircuit& ncircuit,
          const std::vector<double>& expected_results) {
  unsigned num_qubits = 2;
  unsigned num_reps = 20000;

  types::StateSpace state_space(1);

  types::State scratch = state_space.Null();
  types::State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  std::vector<uint64_t> stat;

  std::vector<unsigned> histogram(1 << num_qubits, 0);

  types::QTSimulaltor::Parameter param;

  for (unsigned i = 0; i < num_reps; ++i) {
    state_space.SetStateZero(state);
    param.collect_mea_stat = true;

    EXPECT_TRUE(types::QTSimulaltor::Run(param, num_qubits, ncircuit, i,
                                         scratch, state, stat));

    ASSERT_EQ(stat.size(), 1);
    ++histogram[stat[0]];
  }

  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EXPECT_NEAR(double(histogram[i]) / num_reps, expected_results[i], 0.005);
  }
}

}  // namespace

TEST(QTrajectoryTest, BitFlip) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

circuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
)

ncircuit = circuit.with_noise(cirq.bit_flip(0.01))

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.389352, 0.242790, 0.081009, 0.286850,
  };

  auto ncircuit1 = GenerateNoisyCirquit(0.01, AddBitFlipNoise);
  Run1(ncircuit1, expected_results);
}

TEST(QTrajectoryTest, PhaseDump) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

channel = cirq.phase_damp(0.02)

ncircuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
  channel.on(qs[0]),
  channel.on(qs[1]),
)

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.412300, 0.230500, 0.057219, 0.299982,
  };

  auto ncircuit = GenerateNoisyCirquit(0.02, AddPhaseDumpNoise);
  Run2(ncircuit, expected_results);
}

TEST(QTrajectoryTest, AmplDump) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

channel = cirq.amplitude_damp(0.05)

ncircuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
  channel.on(qs[0]),
  channel.on(qs[1]),
)

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.500494, 0.235273, 0.090879, 0.173354,
  };

  auto ncircuit = GenerateNoisyCirquit(0.05, AddAmplDumpNoise);
  Run1(ncircuit, expected_results);
}

TEST(QTrajectoryTest, GenDump) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

channel = cirq.generalized_amplitude_damp(0.5, 0.1)

ncircuit = cirq.Circuit(
  cirq.H(qs[0]),
  cirq.H(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.rx(0.7)(qs[0]),
  cirq.ry(0.1)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ry(0.4)(qs[0]),
  cirq.rx(0.7)(qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.ISWAP(qs[0], qs[1]),
  channel.on(qs[0]),
  channel.on(qs[1]),
  cirq.measure(*[qs[0], qs[1]], key='m'),
  channel.on(qs[0]),
  channel.on(qs[1]),
)

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.318501, 0.260538, 0.164616, 0.256345,
  };

  auto ncircuit = GenerateNoisyCirquit(0.1, AddGenAmplDumpNoise);
  Run2(ncircuit, expected_results);
}


}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
