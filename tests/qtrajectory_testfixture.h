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

#ifndef QTRAJECTORY_TESTFIXTURE_H_
#define QTRAJECTORY_TESTFIXTURE_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/channel.h"
#include "../lib/channels_cirq.h"
#include "../lib/circuit_noisy.h"
#include "../lib/expect.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_cirq.h"
#include "../lib/io.h"
#include "../lib/qtrajectory.h"

namespace qsim {

template <typename Gate>
void AddBitFlipNoise1(
    unsigned time, unsigned q, double p, NoisyCircuit<Gate>& ncircuit) {
  using fp_type = typename Gate::fp_type;

  double p1 = 1 - p;
  double p2 = p;

  auto normal = KrausOperator<Gate>::kNormal;

  ncircuit.channels.push_back(
      {{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, q)}},
       {normal, 1, p2, {Cirq::X<fp_type>::Create(time, q)}}});
}

template <typename Gate>
void AddBitFlipNoise2(unsigned time, double p, NoisyCircuit<Gate>& ncircuit) {
  using fp_type = typename Gate::fp_type;

  double p1 = 1 - p;
  double p2 = p;

  auto normal = KrausOperator<Gate>::kNormal;

  ncircuit.channels.push_back({
    {normal, 1, p1 * p1, {Cirq::I1<fp_type>::Create(time, 0),
                          Cirq::I1<fp_type>::Create(time, 1)}},
    {normal, 1, p1 * p2, {Cirq::I1<fp_type>::Create(time, 0),
                          Cirq::X<fp_type>::Create(time, 1)}},
    {normal, 1, p2 * p1, {Cirq::X<fp_type>::Create(time, 0),
                          Cirq::I1<fp_type>::Create(time, 1)}},
    {normal, 1, p2 * p2, {Cirq::X<fp_type>::Create(time, 0),
                          Cirq::X<fp_type>::Create(time, 1)}},
  });

//  This can also be imnplemented as the following.
//
//  ncircuit.channels.push_back(
//    {{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, 0)}},
//     {normal, 1, p2, {Cirq::X<fp_type>::Create(time, 0)}}});
//  ncircuit.channels.push_back(
//    {{normal, 1, p1, {Cirq::I1<fp_type>::Create(time, 1)}},
//     {normal, 1, p2, {Cirq::X<fp_type>::Create(time, 1)}}});
}

template <typename Gate>
void AddGenAmplDumpNoise1(
    unsigned time, unsigned q, double g, NoisyCircuit<Gate>& ncircuit) {
  using fp_type = typename Gate::fp_type;

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

  auto normal = KrausOperator<Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.channels.push_back(
      {{normal, 0, p1, {M::Create(time, q, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, q, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, q, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, q, {0, 0, 0, 0, s2, 0, 0, 0})}}});

  for (auto& kop : ncircuit.channels.back()) {
    kop.CalculateKdKMatrix();
  }
}

template <typename Gate>
void AddGenAmplDumpNoise2(
    unsigned time, double g, NoisyCircuit<Gate>& ncircuit) {
  using fp_type = typename Gate::fp_type;

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

  auto normal = KrausOperator<Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.channels.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, 0, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, 0, {0, 0, 0, 0, s2, 0, 0, 0})}}});

  for (auto& kop : ncircuit.channels.back()) {
    kop.CalculateKdKMatrix();
  }

  ncircuit.channels.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3, {M::Create(time, 1, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3, {M::Create(time, 1, {0, 0, 0, 0, s2, 0, 0, 0})}}});

  for (auto& kop : ncircuit.channels.back()) {
    kop.CalculateKdKMatrix();
  }
}

// Adds the same channel as in AddGenAmplDumpNoise2 above.
template <typename Gate>
void AddGenAmplDumpNoise2Alt(
    unsigned time, double g, NoisyCircuit<Gate>& ncircuit) {
  using fp_type = typename Gate::fp_type;

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

  auto normal = KrausOperator<Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.channels.push_back(
      {{normal, 0, p1 * p1,
        {M::Create(time, 0, {t1, 0, 0, 0, 0, 0, r1, 0}),
         M::Create(time, 1, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p1 * p2,
        {M::Create(time, 0, {t1, 0, 0, 0, 0, 0, r1, 0}),
         M::Create(time, 1, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p1 * p3,
        {M::Create(time, 0, {t1, 0, 0, 0, 0, 0, r1, 0}),
         M::Create(time, 1, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p1 * p3,
        {M::Create(time, 0, {t1, 0, 0, 0, 0, 0, r1, 0}),
         M::Create(time, 1, {0, 0, 0, 0, s2, 0, 0, 0})}},

       {normal, 0, p2 * p1,
        {M::Create(time, 0, {r2, 0, 0, 0, 0, 0, t2, 0}),
         M::Create(time, 1, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p2 * p2,
        {M::Create(time, 0, {r2, 0, 0, 0, 0, 0, t2, 0}),
         M::Create(time, 1, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p2 * p3,
        {M::Create(time, 0, {r2, 0, 0, 0, 0, 0, t2, 0}),
         M::Create(time, 1, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p2 * p3,
        {M::Create(time, 0, {r2, 0, 0, 0, 0, 0, t2, 0}),
         M::Create(time, 1, {0, 0, 0, 0, s2, 0, 0, 0})}},

       {normal, 0, p3 * p1,
        {M::Create(time, 0, {0, 0, s1, 0, 0, 0, 0, 0}),
         M::Create(time, 1, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p3 * p2,
        {M::Create(time, 0, {0, 0, s1, 0, 0, 0, 0, 0}),
         M::Create(time, 1, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3 * p3,
        {M::Create(time, 0, {0, 0, s1, 0, 0, 0, 0, 0}),
         M::Create(time, 1, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3 * p3,
        {M::Create(time, 0, {0, 0, s1, 0, 0, 0, 0, 0}),
         M::Create(time, 1, {0, 0, 0, 0, s2, 0, 0, 0})}},

       {normal, 0, p3 * p1,
        {M::Create(time, 0, {0, 0, 0, 0, s2, 0, 0, 0}),
         M::Create(time, 1, {t1, 0, 0, 0, 0, 0, r1, 0})}},
       {normal, 0, p3 * p2,
        {M::Create(time, 0, {0, 0, 0, 0, s2, 0, 0, 0}),
         M::Create(time, 1, {r2, 0, 0, 0, 0, 0, t2, 0})}},
       {normal, 0, p3 * p3,
        {M::Create(time, 0, {0, 0, 0, 0, s2, 0, 0, 0}),
         M::Create(time, 1, {0, 0, s1, 0, 0, 0, 0, 0})}},
       {normal, 0, p3 * p3,
        {M::Create(time, 0, {0, 0, 0, 0, s2, 0, 0, 0}),
         M::Create(time, 1, {0, 0, 0, 0, s2, 0, 0, 0})}},
      });

  for (auto& kop : ncircuit.channels.back()) {
    kop.CalculateKdKMatrix();
  }
}

template <typename Gate>
void AddAmplDumpNoise1(
    unsigned time, unsigned q, double g, NoisyCircuit<Gate>& ncircuit) {
  using fp_type = typename Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.channels.push_back(
      {{normal, 0, p1, {M::Create(time, q, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, q, {0, 0, s, 0, 0, 0, 0, 0})}}});

  for (auto& kop : ncircuit.channels.back()) {
    kop.CalculateKdKMatrix();
  }
}

template <typename Gate>
void AddAmplDumpNoise2(
    unsigned time, double g, NoisyCircuit<Gate>& ncircuit) {
  using fp_type = typename Gate::fp_type;

  double p1 = 1 - g;
  double p2 = 0;

  fp_type r = std::sqrt(p1);
  fp_type s = std::sqrt(g);

  auto normal = KrausOperator<Gate>::kNormal;

  using M = Cirq::MatrixGate1<fp_type>;

  ncircuit.channels.push_back(
      {{normal, 0, p1, {M::Create(time, 0, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 0, {0, 0, s, 0, 0, 0, 0, 0})}}});

  for (auto& kop : ncircuit.channels.back()) {
    kop.CalculateKdKMatrix();
  }

  ncircuit.channels.push_back(
      {{normal, 0, p1, {M::Create(time, 1, {1, 0, 0, 0, 0, 0, r, 0})}},
       {normal, 0, p2, {M::Create(time, 1, {0, 0, s, 0, 0, 0, 0, 0})}}});

  for (auto& kop : ncircuit.channels.back()) {
    kop.CalculateKdKMatrix();
  }
}

template <typename Gate, typename AddNoise1, typename AddNoise2>
NoisyCircuit<Gate> GenerateNoisyCircuit(
    double p, AddNoise1&& add_noise1, AddNoise2&& add_noise2,
    bool add_measurement = true) {
  using fp_type = typename Gate::fp_type;

  NoisyCircuit<Gate> ncircuit;

  ncircuit.num_qubits = 2;
  ncircuit.channels.reserve(24);

  using Hd = Cirq::H<fp_type>;
  using IS = Cirq::ISWAP<fp_type>;
  using Rx = Cirq::rx<fp_type>;
  using Ry = Cirq::ry<fp_type>;

  auto normal = KrausOperator<Gate>::kNormal;

  ncircuit.channels.push_back({{normal, 1, 1.0, {Hd::Create(0, 0)}}});
  add_noise1(1, 0, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {Hd::Create(0, 1)}}});
  add_noise1(1, 1, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {IS::Create(2, 0, 1)}}});
  add_noise2(3, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {Rx::Create(4, 0, 0.7)}}});
  add_noise1(5, 0, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {Ry::Create(4, 1, 0.1)}}});
  add_noise1(5, 1, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {IS::Create(6, 0, 1)}}});
  add_noise2(7, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {Ry::Create(8, 0, 0.4)}}});
  add_noise1(9, 0, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {Rx::Create(8, 1, 0.7)}}});
  add_noise1(9, 1, p, ncircuit);
  ncircuit.channels.push_back({{normal, 1, 1.0, {IS::Create(10, 0, 1)}}});
  add_noise2(11, p, ncircuit);
  if (add_measurement) {
    ncircuit.channels.push_back(
        {{KrausOperator<Gate>::kMeasurement, 1, 1.0,
          {gate::Measurement<Gate>::Create(12, {0, 1})}}});
    add_noise2(13, p, ncircuit);
  }

  return ncircuit;
}

template <typename Factory, typename Gate>
void RunBatch(const Factory& factory, const NoisyCircuit<Gate>& ncircuit,
              const std::vector<double>& expected_results) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using QTSimulator = QuantumTrajectorySimulator<IO, Gate, MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 2;
  unsigned num_reps = 25000;

  auto measure = [](uint64_t r, const State& state,
                    const typename QTSimulator::Stat& stat,
                    std::vector<unsigned>& histogram) {
    ASSERT_EQ(stat.samples.size(), 1);
    ++histogram[stat.samples[0]];
  };

  std::vector<unsigned> histogram(1 << num_qubits, 0);

  typename QTSimulator::Parameter param;
  param.collect_mea_stat = true;

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  EXPECT_TRUE(QTSimulator::RunBatch(param, ncircuit, 0, num_reps, state_space,
                                    simulator, measure, histogram));

  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EXPECT_NEAR(double(histogram[i]) / num_reps, expected_results[i], 0.005);
  }
}

template <typename Factory, typename Gate>
void RunOnceRepeatedly(const Factory& factory,
                       const NoisyCircuit<Gate>& ncircuit,
                       const std::vector<double>& expected_results) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using QTSimulator = QuantumTrajectorySimulator<IO, Gate, MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 2;
  unsigned num_reps = 25000;

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  typename QTSimulator::Stat stat;

  std::vector<unsigned> histogram(1 << num_qubits, 0);

  typename QTSimulator::Parameter param;
  param.collect_mea_stat = true;

  for (unsigned i = 0; i < num_reps; ++i) {
    state_space.SetStateZero(state);

    EXPECT_TRUE(QTSimulator::RunOnce(
        param, ncircuit, i, state_space, simulator, state, stat));

    ASSERT_EQ(stat.samples.size(), 1);
    ++histogram[stat.samples[0]];
  }

  for (std::size_t i = 0; i < histogram.size(); ++i) {
    EXPECT_NEAR(double(histogram[i]) / num_reps, expected_results[i], 0.005);
  }
}

template <typename Factory, typename Gate>
std::vector<std::complex<double>> ExpValsRunBatch(
    const Factory& factory, const NoisyCircuit<Gate>& ncircuit,
    bool reuse_results) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using Fuser = MultiQubitGateFuser<IO, Gate>;
  using QTSimulator = QuantumTrajectorySimulator<IO, Gate, MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 2;
  unsigned num_reps = 25000;

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  typename QTSimulator::Stat stat;

  typename QTSimulator::Parameter param;
  param.apply_last_deferred_ops = !reuse_results;

  using Observables = std::vector<std::vector<qsim::OpString<Gate>>>;
  Observables observables;
  observables.reserve(num_qubits);

  using rx = qsim::Cirq::rx<typename Gate::fp_type>;

  for (unsigned q = 0; q < num_qubits; ++q) {
    observables.push_back({{{1.0, 0.0}, {rx::Create(0, q, 1.7 + 0.6 * q)}}});
  }

  using TrajResults = std::vector<std::vector<std::complex<double>>>;
  TrajResults traj_results(observables.size());

  for (std::size_t k = 0; k < observables.size(); ++k) {
    traj_results[k].reserve(num_reps);
  }

  std::vector<std::complex<double>> primary_results;
  primary_results.reserve(observables.size());

  auto measure = [](uint64_t r, const State& state,
                    const typename QTSimulator::Stat& stat,
                    const Simulator& simulator, bool reuse_results,
                    const Observables& observables,
                    std::vector<std::complex<double>>& primary_results,
                    TrajResults& traj_results) {
    if (reuse_results && stat.primary && !primary_results.empty()) {
      for (std::size_t k = 0; k < observables.size(); ++k) {
        traj_results[k].push_back(primary_results[k]);
      }
    } else {
      for (std::size_t k = 0; k < observables.size(); ++k) {
        const auto& obs = observables[k];
        auto result = ExpectationValue<IO, Fuser>(obs, simulator, state);
        traj_results[k].push_back(result);

        if (reuse_results && stat.primary) {
          primary_results.push_back(result);
        }
      }
    }
  };

  EXPECT_TRUE(QTSimulator::RunBatch(param, ncircuit, 0, num_reps, state_space,
                                    simulator, measure, simulator,
                                    reuse_results, observables, primary_results,
                                    traj_results));

  std::vector<std::complex<double>> results;
  results.reserve(observables.size());

  double inverse_num_reps = 1.0 / num_reps;

  for (std::size_t k = 0; k < observables.size(); ++k) {
    std::complex<double> sum = 0;
    for (unsigned i = 0; i < num_reps; ++i) {
      sum += traj_results[k][i];
    }

    results.push_back(inverse_num_reps * sum);
  }

  return results;
}

template <typename Factory, typename Gate>
std::vector<std::complex<double>> ExpValsRunOnceRepeatedly(
    const Factory& factory, const NoisyCircuit<Gate>& ncircuit,
    bool reuse_results) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using Fuser = MultiQubitGateFuser<IO, Gate>;
  using QTSimulator = QuantumTrajectorySimulator<IO, Gate, MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 2;
  unsigned num_reps = 25000;

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  typename QTSimulator::Stat stat;

  typename QTSimulator::Parameter param;
  param.apply_last_deferred_ops = true;

  std::vector<std::vector<qsim::OpString<Gate>>> observables;
  observables.reserve(num_qubits);

  using rx = qsim::Cirq::rx<typename Gate::fp_type>;

  for (unsigned q = 0; q < num_qubits; ++q) {
    observables.push_back({{{1.0, 0.0}, {rx::Create(0, q, 1.7 + 0.6 * q)}}});
  }

  using TrajResults = std::vector<std::vector<std::complex<double>>>;
  TrajResults traj_results(observables.size());

  for (std::size_t k = 0; k < observables.size(); ++k) {
    traj_results[k].reserve(num_reps);
  }

  std::vector<std::complex<double>> primary_results;
  primary_results.reserve(observables.size());

  for (unsigned i = 0; i < num_reps; ++i) {
    state_space.SetStateZero(state);

    EXPECT_TRUE(QTSimulator::RunOnce(
        param, ncircuit, i, state_space, simulator, state, stat));

    if (reuse_results && stat.primary && !primary_results.empty()) {
      for (std::size_t k = 0; k < observables.size(); ++k) {
        traj_results[k].push_back(primary_results[k]);
      }
    } else {
      for (std::size_t k = 0; k < observables.size(); ++k) {
        const auto& obs = observables[k];
        auto result = ExpectationValue<IO, Fuser>(obs, simulator, state);
        traj_results[k].push_back(result);

        if (reuse_results && stat.primary) {
          primary_results.push_back(result);
          param.apply_last_deferred_ops = false;
        }
      }
    }
  }

  std::vector<std::complex<double>> results;
  results.reserve(observables.size());

  double inverse_num_reps = 1.0 / num_reps;

  for (std::size_t k = 0; k < observables.size(); ++k) {
    std::complex<double> sum = 0;
    for (unsigned i = 0; i < num_reps; ++i) {
      sum += traj_results[k][i];
    }

    results.push_back(inverse_num_reps * sum);
  }

  return results;
}

template <typename Factory>
void TestBitFlip(const Factory& factory) {
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

  using Gate = Cirq::GateCirq<typename Factory::fp_type>;

  auto ncircuit = GenerateNoisyCircuit<Gate>(0.01, AddBitFlipNoise1<Gate>,
                                             AddBitFlipNoise2<Gate>);
  RunBatch(factory, ncircuit, expected_results);
}

template <typename Factory>
void TestGenDump(const Factory& factory) {
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

  using Gate = Cirq::GateCirq<typename Factory::fp_type>;

  {
    auto ncircuit = GenerateNoisyCircuit<Gate>(0.1, AddGenAmplDumpNoise1<Gate>,
                                               AddGenAmplDumpNoise2<Gate>);
    RunOnceRepeatedly(factory, ncircuit, expected_results);
  }

  {
    auto ncircuit = GenerateNoisyCircuit<Gate>(0.1, AddGenAmplDumpNoise1<Gate>,
                                               AddGenAmplDumpNoise2Alt<Gate>);
    RunOnceRepeatedly(factory, ncircuit, expected_results);
  }
}

template <typename Factory>
void TestReusingResults(const Factory& factory) {
  using Gate = Cirq::GateCirq<typename Factory::fp_type>;

  auto ncircuit = GenerateNoisyCircuit<Gate>(0.02, AddAmplDumpNoise1<Gate>,
                                             AddAmplDumpNoise2<Gate>, false);

  auto results1 = ExpValsRunOnceRepeatedly(factory, ncircuit, false);
  auto results2 = ExpValsRunOnceRepeatedly(factory, ncircuit, true);
  auto results3 = ExpValsRunBatch(factory, ncircuit, false);
  auto results4 = ExpValsRunBatch(factory, ncircuit, true);

  for (std::size_t k = 0; k < results1.size(); ++k) {
    EXPECT_NEAR(std::real(results1[k]), std::real(results2[k]), 1e-8);
    EXPECT_NEAR(std::imag(results1[k]), std::imag(results2[k]), 1e-8);
    EXPECT_NEAR(std::real(results1[k]), std::real(results3[k]), 1e-8);
    EXPECT_NEAR(std::imag(results1[k]), std::imag(results3[k]), 1e-8);
    EXPECT_NEAR(std::real(results1[k]), std::real(results4[k]), 1e-8);
    EXPECT_NEAR(std::imag(results1[k]), std::imag(results4[k]), 1e-8);
  }
}

template <typename Factory>
void TestCollectKopStat(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using GateCirq = Cirq::GateCirq<fp_type>;
  using QTSimulator = QuantumTrajectorySimulator<IO, GateCirq,
                                                 MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 4;
  unsigned num_reps = 20000;

  double p = 0.1;
  double p1 = 1 - p;
  double p2 = p;

  using Hd = Cirq::H<fp_type>;
  using I = Cirq::I1<fp_type>;
  using X = Cirq::X<fp_type>;

  auto normal = KrausOperator<GateCirq>::kNormal;

  NoisyCircuit<GateCirq> ncircuit;

  ncircuit.num_qubits = num_qubits;
  ncircuit.channels.reserve(8);

  ncircuit.channels.push_back({{normal, 1, 1.0, {Hd::Create(0, 0)}}});
  ncircuit.channels.push_back({{normal, 1, 1.0, {Hd::Create(0, 1)}}});
  ncircuit.channels.push_back({{normal, 1, 1.0, {Hd::Create(0, 2)}}});
  ncircuit.channels.push_back({{normal, 1, 1.0, {Hd::Create(0, 3)}}});

  // Add bit flip noise.
  ncircuit.channels.push_back({{normal, 1, p1, {I::Create(1, 0)}},
                      {normal, 1, p2, {X::Create(1, 0)}}});
  ncircuit.channels.push_back({{normal, 1, p1, {I::Create(1, 1)}},
                      {normal, 1, p2, {X::Create(1, 1)}}});
  ncircuit.channels.push_back({{normal, 1, p1, {I::Create(1, 2)}},
                      {normal, 1, p2, {X::Create(1, 2)}}});
  ncircuit.channels.push_back({{normal, 1, p1, {I::Create(1, 3)}},
                      {normal, 1, p2, {X::Create(1, 3)}}});

  auto measure = [](uint64_t r, const State& state,
                    const typename QTSimulator::Stat& stat,
                    std::vector<std::vector<unsigned>>& histogram) {
    ASSERT_EQ(stat.samples.size(), histogram.size());
    for (std::size_t i = 0; i < histogram.size(); ++i) {
      ++histogram[i][stat.samples[i]];
    }
  };

  std::vector<std::vector<unsigned>> histogram(8, std::vector<unsigned>(2, 0));

  typename QTSimulator::Parameter param;
  param.collect_kop_stat = true;

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  EXPECT_TRUE(QTSimulator::RunBatch(param, ncircuit, 0, num_reps, state_space,
                                    simulator, measure, histogram));

  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(histogram[i][0], num_reps);
    EXPECT_EQ(histogram[i][1], 0);
  }

  for (std::size_t i = 4; i < 8; ++i) {
    EXPECT_NEAR(double(histogram[i][0]) / num_reps, p1, 0.005);
    EXPECT_NEAR(double(histogram[i][1]) / num_reps, p2, 0.005);
  }
}

template <typename Factory>
void TestCleanCircuit(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using GateCirq = Cirq::GateCirq<fp_type>;
  using QTSimulator = QuantumTrajectorySimulator<IO, GateCirq,
                                                 MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 4;
  auto size = uint64_t{1} << num_qubits;

  std::vector<GateCirq> circuit;
  circuit.reserve(20);

  circuit.push_back(Cirq::H<fp_type>::Create(0, 0));
  circuit.push_back(Cirq::H<fp_type>::Create(0, 1));
  circuit.push_back(Cirq::H<fp_type>::Create(0, 2));
  circuit.push_back(Cirq::H<fp_type>::Create(0, 3));

  circuit.push_back(Cirq::T<fp_type>::Create(1, 0));
  circuit.push_back(Cirq::T<fp_type>::Create(1, 1));
  circuit.push_back(Cirq::T<fp_type>::Create(1, 2));
  circuit.push_back(Cirq::T<fp_type>::Create(1, 3));

  circuit.push_back(Cirq::CX<fp_type>::Create(2, 0, 2));
  circuit.push_back(Cirq::CZ<fp_type>::Create(2, 1, 3));

  circuit.push_back(Cirq::XPowGate<fp_type>::Create(3, 0, 0.3, 1.1));
  circuit.push_back(Cirq::YPowGate<fp_type>::Create(3, 1, 0.4, 1.0));
  circuit.push_back(Cirq::ZPowGate<fp_type>::Create(3, 2, 0.5, 0.9));
  circuit.push_back(Cirq::HPowGate<fp_type>::Create(3, 3, 0.6, 0.8));

  circuit.push_back(Cirq::CZPowGate<fp_type>::Create(4, 0, 1, 0.7, 0.2));
  circuit.push_back(Cirq::CXPowGate<fp_type>::Create(4, 2, 3, 1.2, 0.4));

  circuit.push_back(Cirq::HPowGate<fp_type>::Create(5, 0, 0.7, 0.2));
  circuit.push_back(Cirq::XPowGate<fp_type>::Create(5, 1, 0.8, 0.3));
  circuit.push_back(Cirq::YPowGate<fp_type>::Create(5, 2, 0.9, 0.4));
  circuit.push_back(Cirq::ZPowGate<fp_type>::Create(5, 3, 1.0, 0.5));

  NoisyCircuit<GateCirq> ncircuit;

  ncircuit.num_qubits = num_qubits;
  ncircuit.channels.reserve(circuit.size());

  auto normal = KrausOperator<GateCirq>::kNormal;

  for (std::size_t i = 0; i < circuit.size(); ++i) {
    ncircuit.channels.push_back({{normal, 1, 1.0, {circuit[i]}}});
  }

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(num_qubits);
  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  // Run clean-circuit simulator.
  for (const auto& gate : circuit) {
    ApplyGate(simulator, gate, state);
  }

  State nstate = state_space.Create(num_qubits);

  EXPECT_FALSE(state_space.IsNull(nstate));

  typename QTSimulator::Stat stat;

  typename QTSimulator::Parameter param;

  state_space.SetStateZero(nstate);

  // Run quantum trajectory simulator.
  EXPECT_TRUE(QTSimulator::RunOnce(param, num_qubits, ncircuit.channels.begin(),
                                   ncircuit.channels.end(), 0, state_space,
                                   simulator, nstate, stat));

  EXPECT_EQ(stat.samples.size(), 0);

  for (uint64_t i = 0; i < size; ++i) {
    auto a1 = state_space.GetAmpl(state, i);
    auto a2 = state_space.GetAmpl(nstate, i);
    EXPECT_NEAR(std::real(a1), std::real(a2), 1e-6);
    EXPECT_NEAR(std::imag(a1), std::imag(a2), 1e-6);
  }
}

// Test that QTSimulator::Run does not overwrite initial states.
template <typename Factory>
void TestInitialState(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using GateCirq = Cirq::GateCirq<fp_type>;
  using QTSimulator = QuantumTrajectorySimulator<IO, GateCirq,
                                                 MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 3;

  NoisyCircuit<GateCirq> ncircuit;

  ncircuit.num_qubits = num_qubits;
  ncircuit.channels.reserve(3);

  auto normal = KrausOperator<GateCirq>::kNormal;

  ncircuit.channels.push_back(
      {{normal, 1, 1.0, {Cirq::X<fp_type>::Create(0, 0)}}});
  ncircuit.channels.push_back(
      {{normal, 1, 1.0, {Cirq::X<fp_type>::Create(0, 1)}}});
  ncircuit.channels.push_back(
      {{normal, 1, 1.0, {Cirq::X<fp_type>::Create(0, 2)}}});

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(num_qubits);

  EXPECT_FALSE(state_space.IsNull(state));

  typename QTSimulator::Parameter param;
  typename QTSimulator::Stat stat;

  for (unsigned i = 0; i < 8; ++i) {
    state_space.SetAmpl(state, i, 1 + i, 0);
  }

  EXPECT_TRUE(QTSimulator::RunOnce(
      param, ncircuit, 0, state_space, simulator, state, stat));

  // Expect reversed order of amplitudes.
  for (unsigned i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(std::real(state_space.GetAmpl(state, i)), 8 - i);
  }
}

template <typename Factory>
void TestUncomputeFinalState(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using GateCirq = Cirq::GateCirq<fp_type>;
  using QTSimulator = QuantumTrajectorySimulator<IO, GateCirq,
                                                 MultiQubitGateFuser,
                                                 Simulator>;

  unsigned num_qubits = 4;

  std::vector<GateCirq> circuit = {
    Cirq::H<fp_type>::Create(0, 0),
    Cirq::H<fp_type>::Create(0, 1),
    Cirq::H<fp_type>::Create(0, 2),
    Cirq::H<fp_type>::Create(0, 3),
    Cirq::ISWAP<fp_type>::Create(1, 0, 1),
    Cirq::ISWAP<fp_type>::Create(1, 2, 3),
    Cirq::rx<fp_type>::Create(2, 0, 0.1),
    Cirq::ry<fp_type>::Create(2, 1, 0.2),
    Cirq::rz<fp_type>::Create(2, 2, 0.3),
    Cirq::rx<fp_type>::Create(2, 3, 0.4),
    Cirq::ISWAP<fp_type>::Create(3, 0, 3),
    Cirq::ISWAP<fp_type>::Create(3, 1, 2),
    Cirq::ry<fp_type>::Create(4, 0, 0.5),
    Cirq::rz<fp_type>::Create(4, 1, 0.6),
    Cirq::rx<fp_type>::Create(4, 2, 0.7),
    Cirq::ry<fp_type>::Create(4, 3, 0.8),
  };

  // Works only with mixtures.
  auto channel = Cirq::bit_flip<fp_type>(0.3);
  auto ncircuit = MakeNoisy(num_qubits, circuit, channel);

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(num_qubits);

  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  typename QTSimulator::Parameter param;
  param.collect_kop_stat = true;

  typename QTSimulator::Stat stat;

  // Run one trajectory.
  EXPECT_TRUE(QTSimulator::RunOnce(
      param, ncircuit, 0, state_space, simulator, state, stat));

  EXPECT_EQ(ncircuit.channels.size(), stat.samples.size());

  // Uncompute the final state back to |0000> (up to round-off errors).
  for (std::size_t i = 0; i < ncircuit.channels.size(); ++i) {
    auto k = ncircuit.channels.size() - 1 - i;

    const auto& ops = ncircuit.channels[k][stat.samples[k]].ops;

    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
      ApplyGateDagger(simulator, *it, state);
    }
  }

  unsigned size = 1 << num_qubits;

  for (unsigned i = 0; i < size; ++i) {
    auto a = state_space.GetAmpl(state, i);
    EXPECT_NEAR(std::real(a), i == 0 ? 1 : 0, 1e-6);
    EXPECT_NEAR(std::imag(a), 0, 1e-7);
  }
}

}  // namespace qsim

#endif  // QTRAJECTORY_TESTFIXTURE_H_
