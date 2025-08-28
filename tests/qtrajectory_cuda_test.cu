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

#include "qtrajectory_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/fuser_mqubit.h"
#include "../lib/gates_cirq.h"
#include "../lib/io.h"
#include "../lib/run_qsim.h"
#include "../lib/simulator_cuda.h"

namespace qsim {

template <typename FP>
struct Factory {
  using fp_type = FP;
  using Simulator = qsim::SimulatorCUDA<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  Factory(const typename StateSpace::Parameter& param) : param(param) {}

  StateSpace CreateStateSpace() const {
    return StateSpace(param);
  }

  Simulator CreateSimulator() const {
    return Simulator();
  }

  typename StateSpace::Parameter param;
};

TEST(QTrajectoryCUDATest, BitFlip) {
  using Factory = qsim::Factory<float>;
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory>;
  Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestBitFlip<Runner>(factory);
}

TEST(QTrajectoryCUDATest, GenDump) {
  using Factory = qsim::Factory<float>;
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory>;
  Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestGenDump<Runner>(factory);
}

TEST(QTrajectoryCUDATest, ReusingResults) {
  using Factory = qsim::Factory<float>;
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory>;
  Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestReusingResults<Runner>(factory);
}

TEST(QTrajectoryCUDATest, CollectKopStat) {
  using Factory = qsim::Factory<float>;
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory>;
  Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestCollectKopStat<Runner>(factory);
}

TEST(QTrajectoryCUDATest, CleanCircuit) {
  using Factory = qsim::Factory<float>;
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory>;
  Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestCleanCircuit<Runner>(factory);
}

TEST(QTrajectoryCUDATest, InitialState) {
  using Factory = qsim::Factory<float>;
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory>;
  Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestInitialState<Runner>(factory);
}

TEST(QTrajectoryCUDATest, UncomputeFinalState) {
  using Factory = qsim::Factory<float>;
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory>;
  Factory::StateSpace::Parameter param;
  Factory factory(param);
  TestUncomputeFinalState<Runner>(factory);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
