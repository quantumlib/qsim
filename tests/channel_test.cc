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

#include "../lib/channel.h"
#include "../lib/formux.h"
#include "../lib/gates_cirq.h"
#include "../lib/matrix.h"
#include "../lib/simmux.h"

namespace qsim {

namespace {

template <typename KrausOperator>
void TestUnitaryKrausOparator(const KrausOperator& kop) {
  // Should be an identity matrix.

  unsigned m = 1 << kop.qubits.size();

  for (unsigned i = 0; i < m; ++i) {
    for (unsigned j = 0; j < m; ++j) {
      auto re = kop.kd_k.data()[2 * m * i + 2 * j];
      auto im = kop.kd_k.data()[2 * m * i + 2 * j + 1];

      if (i == j) {
        EXPECT_NEAR(re, 1, 1e-6);
        EXPECT_NEAR(im, 0, 1e-7);
      } else {
        EXPECT_NEAR(re, 0, 1e-7);
        EXPECT_NEAR(im, 0, 1e-7);
      }
    }
  }
}

template <typename KrausOperator, typename StateSpace, typename Simulator,
          typename State>
void TestNonUnitaryKrausOparator(const KrausOperator& kop,
                                 const StateSpace& state_space,
                                 const Simulator& simulator,
                                 State& state0, State& state1) {
  state_space.SetStateUniform(state0);
  state_space.SetStateUniform(state1);

  for (const auto&op : kop.ops) {
    simulator.ApplyGate(op.qubits, op.matrix.data(), state0);
  }

  for (auto it = kop.ops.rbegin(); it != kop.ops.rend(); ++it) {
    auto md = it->matrix;
    MatrixDagger(1 << it->qubits.size(), md);
    simulator.ApplyGate(it->qubits, md.data(), state0);
  }

  simulator.ApplyGate(kop.qubits, kop.kd_k.data(), state1);

  unsigned size = unsigned{1} << (state0.num_qubits() + 1);

  for (unsigned i = 0; i < size; ++i) {
    EXPECT_NEAR(state0.get()[i], state1.get()[i], 1e-7);
  }
}

}  // namespace

TEST(ChannelTest, UnitaryKdKMatrix) {
  using fp_type = Simulator<For>::fp_type;
  using Gate = Cirq::GateCirq<fp_type>;

  auto normal = KrausOperator<Gate>::kNormal;

  Channel<Gate> channel = {
    {
      normal, 1, 0.2, {
                        Cirq::FSimGate<fp_type>::Create(0, 3, 4, 0.1, 1.4),
                      }
    },
    {
      normal, 1, 0.2, {
                        Cirq::rx<fp_type>::Create(0, 0, 0.1),
                        Cirq::ry<fp_type>::Create(0, 1, 0.2),
                        Cirq::FSimGate<fp_type>::Create(1, 0, 1, 0.2, 1.3),
                      }
    },
    {
      normal, 1, 0.2, {
                        Cirq::rz<fp_type>::Create(0, 3, 0.3),
                        Cirq::rx<fp_type>::Create(0, 1, 0.4),
                        Cirq::ry<fp_type>::Create(0, 4, 0.5),
                        Cirq::rz<fp_type>::Create(0, 0, 0.6),
                      }
    },
    {
      normal, 1, 0.2, {
                        Cirq::rx<fp_type>::Create(0, 4, 0.7),
                        Cirq::ry<fp_type>::Create(0, 3, 0.8),
                        Cirq::rz<fp_type>::Create(0, 1, 0.9),
                        Cirq::rx<fp_type>::Create(0, 0, 1.0),
                        Cirq::FSimGate<fp_type>::Create(1, 1, 3, 0.3, 1.2),
                        Cirq::FSimGate<fp_type>::Create(1, 0, 4, 0.4, 1.1),
                      }
    },
    {
      normal, 1, 0.2, {
                        Cirq::ry<fp_type>::Create(0, 7, 1.1),
                        Cirq::rz<fp_type>::Create(0, 5, 1.2),
                        Cirq::rx<fp_type>::Create(0, 1, 1.3),
                        Cirq::ry<fp_type>::Create(0, 3, 1.4),
                        Cirq::rz<fp_type>::Create(0, 2, 1.5),
                        Cirq::rx<fp_type>::Create(0, 4, 1.6),
                        Cirq::FSimGate<fp_type>::Create(1, 4, 5, 0.5, 1.0),
                        Cirq::FSimGate<fp_type>::Create(1, 1, 3, 0.6, 0.9),
                        Cirq::FSimGate<fp_type>::Create(1, 2, 7, 0.7, 0.8),
                      }
    },
  };

  channel[0].CalculateKdKMatrix();
  ASSERT_EQ(channel[0].kd_k.size(), 32);
  ASSERT_EQ(channel[0].qubits.size(), 2);
  EXPECT_EQ(channel[0].qubits[0], 3);
  EXPECT_EQ(channel[0].qubits[1], 4);
  TestUnitaryKrausOparator(channel[0]);

  channel[1].CalculateKdKMatrix();
  ASSERT_EQ(channel[1].kd_k.size(), 32);
  ASSERT_EQ(channel[1].qubits.size(), 2);
  EXPECT_EQ(channel[1].qubits[0], 0);
  EXPECT_EQ(channel[1].qubits[1], 1);
  TestUnitaryKrausOparator(channel[1]);

  channel[2].CalculateKdKMatrix();
  ASSERT_EQ(channel[2].kd_k.size(), 512);
  ASSERT_EQ(channel[2].qubits.size(), 4);
  EXPECT_EQ(channel[2].qubits[0], 0);
  EXPECT_EQ(channel[2].qubits[1], 1);
  EXPECT_EQ(channel[2].qubits[2], 3);
  EXPECT_EQ(channel[2].qubits[3], 4);
  TestUnitaryKrausOparator(channel[2]);

  channel[3].CalculateKdKMatrix();
  ASSERT_EQ(channel[3].kd_k.size(), 512);
  ASSERT_EQ(channel[3].qubits.size(), 4);
  EXPECT_EQ(channel[3].qubits[0], 0);
  EXPECT_EQ(channel[3].qubits[1], 1);
  EXPECT_EQ(channel[3].qubits[2], 3);
  EXPECT_EQ(channel[3].qubits[3], 4);
  TestUnitaryKrausOparator(channel[3]);

  channel[4].CalculateKdKMatrix();
  ASSERT_EQ(channel[4].kd_k.size(), 8192);
  ASSERT_EQ(channel[4].qubits.size(), 6);
  EXPECT_EQ(channel[4].qubits[0], 1);
  EXPECT_EQ(channel[4].qubits[1], 2);
  EXPECT_EQ(channel[4].qubits[2], 3);
  EXPECT_EQ(channel[4].qubits[3], 4);
  EXPECT_EQ(channel[4].qubits[4], 5);
  EXPECT_EQ(channel[4].qubits[5], 7);
  TestUnitaryKrausOparator(channel[4]);
}

TEST(ChannelTest, NonUnitaryKdKMatrix) {
  using StateSpace = Simulator<For>::StateSpace;
  using State = StateSpace::State;
  using fp_type = StateSpace::fp_type;
  using Gate = Cirq::GateCirq<fp_type>;
  using M1 = Cirq::MatrixGate1<fp_type>;
  using M2 = Cirq::MatrixGate2<fp_type>;

  unsigned  num_qubits = 8;
  auto normal = KrausOperator<Gate>::kNormal;

  Channel<Gate> channel = {
    {
      normal, 0, 0, {
                      M1::Create(0, 0,
                                 {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4}),
                      M1::Create(0, 1,
                                 {0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1}),
                      M2::Create(0, 0, 1,
                                 {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
                                  0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1,
                                  0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2,
                                  0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3}),
                    }
    },
    {
      normal, 0, 0, {
                      M1::Create(0, 4,
                                 {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4}),
                      M1::Create(0, 3,
                                 {0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1}),
                      M1::Create(0, 1,
                                 {0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2}),
                      M1::Create(0, 0,
                                 {0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3}),
                      M2::Create(0, 0, 4,
                                 {0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
                                  0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1,
                                  0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2,
                                  0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3}),
                    }
    },
  };

  StateSpace state_space(1);
  Simulator<For> simulator(1);

  State state0 = state_space.Create(num_qubits);
  ASSERT_FALSE(state_space.IsNull(state0));

  State state1 = state_space.Create(num_qubits);
  ASSERT_FALSE(state_space.IsNull(state1));

  channel[0].CalculateKdKMatrix();
  ASSERT_EQ(channel[0].kd_k.size(), 32);
  ASSERT_EQ(channel[0].qubits.size(), 2);
  EXPECT_EQ(channel[0].qubits[0], 0);
  EXPECT_EQ(channel[0].qubits[1], 1);
  TestNonUnitaryKrausOparator(
      channel[0], state_space, simulator, state0, state1);

  channel[1].CalculateKdKMatrix();
  ASSERT_EQ(channel[1].kd_k.size(), 512);
  ASSERT_EQ(channel[1].qubits.size(), 4);
  EXPECT_EQ(channel[1].qubits[0], 0);
  EXPECT_EQ(channel[1].qubits[1], 1);
  EXPECT_EQ(channel[1].qubits[2], 3);
  EXPECT_EQ(channel[1].qubits[3], 4);
  TestNonUnitaryKrausOparator(
      channel[1], state_space, simulator, state0, state1);
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
