// Copyright 2025 Google LLC. All Rights Reserved.
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

#include <custatevecEx.h>

#include "gtest/gtest.h"

#include "../lib/io.h"
#include "../lib/multiprocess_custatevecex.h"
#include "../lib/run_custatevecex.h"
#include "../lib/simulator_custatevecex.h"

namespace qsim {

MultiProcessCuStateVecEx mp;

template <typename FP>
struct Factory {
  using fp_type = FP;
  using Simulator = qsim::SimulatorCuStateVecEx<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace CreateStateSpace() const {
    return StateSpace{mp};
  }

  Simulator CreateSimulator() const {
    return Simulator{};
  }
};

TEST(QTrajectoryCuStateVecExTest, BitFlip) {
  using Runner = CuStateVecExRunner<IO, Factory<float>>;
  TestBitFlip<Runner>(Factory<float>());
}

TEST(QTrajectoryCuStateVecExTest, GenDump) {
  using Runner = CuStateVecExRunner<IO, Factory<float>>;
  TestGenDump<Runner>(Factory<float>());
}

TEST(QTrajectoryCuStateVecExTest, ReusingResults) {
  using Runner = CuStateVecExRunner<IO, Factory<float>>;
  TestReusingResults<Runner>(Factory<float>());
}

TEST(QTrajectoryCuStateVecExTest, CollectKopStat) {
  using Runner = CuStateVecExRunner<IO, Factory<float>>;
  TestCollectKopStat<Runner>(Factory<float>());
}

TEST(QTrajectoryCuStateVecExTest, CleanCircuit) {
  using Runner = CuStateVecExRunner<IO, Factory<float>>;
  TestCleanCircuit<Runner>(Factory<float>());
}

TEST(QTrajectoryCuStateVecExTest, InitialState) {
  using Runner = CuStateVecExRunner<IO, Factory<float>>;
  TestInitialState<Runner>(Factory<float>());
}

TEST(QTrajectoryCuStateVecExTest, UncomputeFinalState) {
  using Runner = CuStateVecExRunner<IO, Factory<float>>;
  TestUncomputeFinalState<Runner>(Factory<float>());
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  qsim::mp.initialize(qsim::MultiProcessCuStateVecEx::Parameter{});

  return RUN_ALL_TESTS();
}
