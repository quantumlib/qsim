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

#include "../lib/channels_cirq.h"
#include "../lib/circuit.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_cirq.h"
#include "../lib/io.h"
#include "../lib/qtrajectory.h"
#include "../lib/simmux.h"

namespace qsim {

namespace {

using StateSpace = Simulator<For>::StateSpace;
using State = StateSpace::State;
using fp_type = StateSpace::fp_type;
using Gate = Cirq::GateCirq<fp_type>;
using QTSimulator = QuantumTrajectorySimulator<IO, Gate, MultiQubitGateFuser,
                                               Simulator<For>>;

Circuit<Gate> CleanCircuit() {
  using Hd = Cirq::H<fp_type>;
  using IS = Cirq::ISWAP<fp_type>;
  using Rx = Cirq::rx<fp_type>;
  using Ry = Cirq::ry<fp_type>;

  return {
    2,
    {
      Hd::Create(0, 0),
      Hd::Create(0, 1),
      IS::Create(1, 0, 1),
      Rx::Create(2, 0, 0.7),
      Ry::Create(2, 1, 0.1),
      IS::Create(3, 0, 1),
      Ry::Create(4, 0, 0.4),
      Rx::Create(4, 1, 0.7),
      IS::Create(5, 0, 1),
      gate::Measurement<Gate>::Create(6, {0, 1}),
    },
  };
}

void RunBatch(const NoisyCircuit<Gate>& ncircuit,
              const std::vector<double>& expected_results,
              unsigned num_reps = 25000) {
  unsigned num_qubits = 2;
  unsigned num_threads = 1;

  auto measure = [](uint64_t r, const State& state,
                    const QTSimulator::Stat& stat,
                    std::vector<unsigned>& histogram) {
    ASSERT_EQ(stat.samples.size(), 1);
    ++histogram[stat.samples[0]];
  };

  std::vector<unsigned> histogram(1 << num_qubits, 0);

  QTSimulator::Parameter param;
  param.collect_mea_stat = true;

  Simulator<For> simulator(num_threads);
  Simulator<For>::StateSpace state_space(num_threads);

  EXPECT_TRUE(QTSimulator::RunBatch(param, ncircuit, 0, num_reps, state_space,
                                    simulator, measure, histogram));

  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EXPECT_NEAR(double(histogram[i]) / num_reps, expected_results[i], 0.005);
  }
}

template <typename ChannelFactory>
inline NoisyCircuit<Gate> MakeNoisy2(
    const std::vector<Gate>& gates, const ChannelFactory& channel_factory) {
  NoisyCircuit<Gate> ncircuit;

  ncircuit.num_qubits = 2;
  ncircuit.channels.reserve(2 * gates.size());

  unsigned prev_time = 0;

  for (const auto& gate : gates) {
    if (gate.time > prev_time) {
      ncircuit.channels.push_back(
          channel_factory.Create(2 * prev_time + 1, {0, 1}));
      prev_time = gate.time;
    }

    ncircuit.channels.push_back(MakeChannelFromGate(2 * gate.time, gate));
  }

  return ncircuit;
}

}  // namespace

TEST(ChannelsCirqTest, AsymmetricDepolarizingChannel) {
/* The expected results are obtained with the following Cirq code.

import cirq

qs = cirq.LineQubit.range(2)

channel = cirq.asymmetric_depolarize(0.01, 0.02, 0.05)

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
    0.3249745, 0.2416734, 0.1637337, 0.2696184,
  };

  auto channel = Cirq::asymmetric_depolarize<fp_type>(0.01, 0.02, 0.05);
  auto circuit = CleanCircuit();

  auto ncircuit = MakeNoisy(circuit, channel);
  RunBatch(ncircuit, expected_results);

  auto ncircuit2 = MakeNoisy2(circuit.gates, channel);
  RunBatch(ncircuit2, expected_results);
}

TEST(ChannelsCirqTest, DepolarizingChannel) {
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

ncircuit = circuit.with_noise(cirq.depolarize(0.01))

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.3775343, 0.2423451, 0.0972634, 0.2828572,
  };

  auto channel = Cirq::depolarize<fp_type>(0.02);
  auto circuit = CleanCircuit();

  auto ncircuit = MakeNoisy(circuit, channel);
  RunBatch(ncircuit, expected_results);

  auto ncircuit2 = MakeNoisy2(circuit.gates, channel);
  RunBatch(ncircuit2, expected_results);
}

TEST(ChannelsCirqTest, GeneralizedAmplitudeDampingChannel) {
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

  auto circuit = CleanCircuit();
  auto ncircuit = MakeNoisy(
      circuit, Cirq::generalized_amplitude_damp<fp_type>(0.5, 0.1));

  RunBatch(ncircuit, expected_results);
}

TEST(ChannelsCirqTest, AmplitudeDampingChannel) {
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

  auto circuit = CleanCircuit();
  auto ncircuit = MakeNoisy(circuit, Cirq::amplitude_damp<fp_type>(0.05));

  RunBatch(ncircuit, expected_results);
}

TEST(ChannelsCirqTest, PhaseDampingChannel) {
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

  auto circuit = CleanCircuit();
  auto ncircuit = MakeNoisy(circuit, Cirq::phase_damp<fp_type>(0.02));

  RunBatch(ncircuit, expected_results);
}

TEST(ChannelsCirqTest, ResetChannel) {
  std::vector<double> expected_results = {
    1, 0, 0, 0,
  };

  auto circuit = CleanCircuit();
  auto ncircuit = MakeNoisy(circuit, Cirq::reset<fp_type>());

  RunBatch(ncircuit, expected_results, 1000);
}

TEST(ChannelsCirqTest, PhaseFlipChannel) {
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

ncircuit = circuit.with_noise(cirq.phase_flip(0.01))

reps = 10000000

sim = cirq.Simulator()
res = sim.run(ncircuit, repetitions=reps)

for key, val in sorted(res.histogram(key='m').items()):
  print(f'{key} {float(val) / reps}')

*/

  std::vector<double> expected_results = {
    0.3790965, 0.2183726, 0.1037091, 0.2988218,
  };

  auto circuit = CleanCircuit();
  auto ncircuit = MakeNoisy(circuit, Cirq::phase_flip<fp_type>(0.05));

  RunBatch(ncircuit, expected_results);
}

TEST(ChannelsCirqTest, BitFlipChannel) {
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

  auto circuit = CleanCircuit();
  auto ncircuit = MakeNoisy(circuit, Cirq::bit_flip<fp_type>(0.01));

  RunBatch(ncircuit, expected_results);
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
