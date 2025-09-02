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
#include "../lib/seqfor.h"
#include "../lib/simulator_avx.h"

namespace qsim {

template <typename For>
struct Factory {
  using Simulator = qsim::SimulatorAVX<For>;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace CreateStateSpace() const {
    return StateSpace(1);
  }

  Simulator CreateSimulator() const {
    return Simulator(1);
  }
};

TEST(QTrajectoryAVXTest, BitFlip) {
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory<SequentialFor>>;
  TestBitFlip<Runner>(Factory<SequentialFor>());
}

TEST(QTrajectoryAVXTest, GenDump) {
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory<SequentialFor>>;
  TestGenDump<Runner>(Factory<SequentialFor>());
}

TEST(QTrajectoryAVXTest, ReusingResults) {
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory<SequentialFor>>;
  TestReusingResults<Runner>(Factory<SequentialFor>());
}

TEST(QTrajectoryAVXTest, CollectKopStat) {
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory<SequentialFor>>;
  TestCollectKopStat<Runner>(Factory<SequentialFor>());
}

TEST(QTrajectoryAVXTest, CleanCircuit) {
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory<SequentialFor>>;
  TestCleanCircuit<Runner>(Factory<SequentialFor>());
}

TEST(QTrajectoryAVXTest, InitialState) {
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory<SequentialFor>>;
  TestInitialState<Runner>(Factory<SequentialFor>());
}

TEST(QTrajectoryAVXTest, UncomputeFinalState) {
  using Fuser = MultiQubitGateFuser<IO, const Cirq::GateCirq<float>*>;
  using Runner = QSimRunner<IO, Fuser, Factory<SequentialFor>>;
  TestUncomputeFinalState<Runner>(Factory<SequentialFor>());
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
