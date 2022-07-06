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

#include <cublas_v2.h>
#include <custatevec.h>

#include "gtest/gtest.h"

#include "../lib/simulator_custatevec.h"

namespace qsim {

template <typename FP>
struct Factory {
  using fp_type = FP;
  using Simulator = qsim::SimulatorCuStateVec<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  Factory() {
    ErrorCheck(cublasCreate(&cublas_handle));
    ErrorCheck(custatevecCreate(&custatevec_handle));
  }

  ~Factory() {
    ErrorCheck(cublasDestroy(cublas_handle));
    ErrorCheck(custatevecDestroy(custatevec_handle));
  }

  StateSpace CreateStateSpace() const {
    return StateSpace(cublas_handle, custatevec_handle);
  }

  Simulator CreateSimulator() const {
    return Simulator(custatevec_handle);
  }

  cublasHandle_t cublas_handle;
  custatevecHandle_t custatevec_handle;
};

TEST(QTrajectoryCuStateVecTest, BitFlip) {
  TestBitFlip(qsim::Factory<float>());
}

TEST(QTrajectoryCuStateVecTest, GenDump) {
  TestGenDump(qsim::Factory<float>());
}

TEST(QTrajectoryCuStateVecTest, ReusingResults) {
  TestReusingResults(qsim::Factory<float>());
}

TEST(QTrajectoryCuStateVecTest, CollectKopStat) {
  TestCollectKopStat(qsim::Factory<float>());
}

TEST(QTrajectoryCuStateVecTest, CleanCircuit) {
  TestCleanCircuit(qsim::Factory<float>());
}

TEST(QTrajectoryCuStateVecTest, InitialState) {
  TestInitialState(qsim::Factory<float>());
}

TEST(QTrajectoryCuStateVecTest, UncomputeFinalState) {
  TestUncomputeFinalState(qsim::Factory<float>());
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
