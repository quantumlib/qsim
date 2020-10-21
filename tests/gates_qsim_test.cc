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

#include "gtest/gtest.h"

#include "../lib/gates_qsim.h"

namespace qsim {

TEST(GatesQsimTest, GateRX) {
  float phi = 0.42;
  unsigned time = 7;
  unsigned qubit = 11;
  auto gate = GateRX<float>::Create(time, qubit, phi);

  float phi2 = -0.5 * phi;
  float c = std::cos(phi2);
  float s = std::sin(phi2);

  EXPECT_EQ(gate.time, time);
  EXPECT_EQ(gate.qubits.size(), 1);
  EXPECT_EQ(gate.qubits[0], qubit);

  EXPECT_FLOAT_EQ(gate.matrix[0], c);
  EXPECT_FLOAT_EQ(gate.matrix[1], 0);
  EXPECT_FLOAT_EQ(gate.matrix[2], 0);
  EXPECT_FLOAT_EQ(gate.matrix[3], s);
  EXPECT_FLOAT_EQ(gate.matrix[4], 0);
  EXPECT_FLOAT_EQ(gate.matrix[5], s);
  EXPECT_FLOAT_EQ(gate.matrix[6], c);
  EXPECT_FLOAT_EQ(gate.matrix[7], 0);
}

TEST(GatesQsimTest, GateRY) {
  float phi = 0.42;
  unsigned time = 7;
  unsigned qubit = 11;
  auto gate = GateRY<float>::Create(time, qubit, phi);

  float phi2 = -0.5 * phi;
  float c = std::cos(phi2);
  float s = std::sin(phi2);

  EXPECT_EQ(gate.time, time);
  EXPECT_EQ(gate.qubits.size(), 1);
  EXPECT_EQ(gate.qubits[0], qubit);

  EXPECT_FLOAT_EQ(gate.matrix[0], c);
  EXPECT_FLOAT_EQ(gate.matrix[1], 0);
  EXPECT_FLOAT_EQ(gate.matrix[2], s);
  EXPECT_FLOAT_EQ(gate.matrix[3], 0);
  EXPECT_FLOAT_EQ(gate.matrix[4], -s);
  EXPECT_FLOAT_EQ(gate.matrix[5], 0);
  EXPECT_FLOAT_EQ(gate.matrix[6], c);
  EXPECT_FLOAT_EQ(gate.matrix[7], 0);
}

TEST(GatesQsimTest, GateRZ) {
  float phi = 0.42;
  unsigned time = 7;
  unsigned qubit = 11;
  auto gate = GateRZ<float>::Create(time, qubit, phi);

  float phi2 = -0.5 * phi;
  float c = std::cos(phi2);
  float s = std::sin(phi2);

  EXPECT_EQ(gate.time, time);
  EXPECT_EQ(gate.qubits.size(), 1);
  EXPECT_EQ(gate.qubits[0], qubit);

  EXPECT_FLOAT_EQ(gate.matrix[0], c);
  EXPECT_FLOAT_EQ(gate.matrix[1], s);
  EXPECT_FLOAT_EQ(gate.matrix[2], 0);
  EXPECT_FLOAT_EQ(gate.matrix[3], 0);
  EXPECT_FLOAT_EQ(gate.matrix[4], 0);
  EXPECT_FLOAT_EQ(gate.matrix[5], 0);
  EXPECT_FLOAT_EQ(gate.matrix[6], c);
  EXPECT_FLOAT_EQ(gate.matrix[7], -s);
}

TEST(GatesQsimTest, GateRXY) {
  float theta = 0.84;
  float phi = 0.42;
  unsigned time = 7;
  unsigned qubit = 11;
  auto gate = GateRXY<float>::Create(time, qubit, theta, phi);

  float phi2 = -0.5 * phi;
  float cp = std::cos(phi2);
  float sp = std::sin(phi2);
  float ct = std::cos(theta) * sp;
  float st = std::sin(theta) * sp;

  EXPECT_EQ(gate.time, time);
  EXPECT_EQ(gate.qubits.size(), 1);
  EXPECT_EQ(gate.qubits[0], qubit);

  EXPECT_FLOAT_EQ(gate.matrix[0], cp);
  EXPECT_FLOAT_EQ(gate.matrix[1], 0);
  EXPECT_FLOAT_EQ(gate.matrix[2], st);
  EXPECT_FLOAT_EQ(gate.matrix[3], ct);
  EXPECT_FLOAT_EQ(gate.matrix[4], -st);
  EXPECT_FLOAT_EQ(gate.matrix[5], ct);
  EXPECT_FLOAT_EQ(gate.matrix[6], cp);
  EXPECT_FLOAT_EQ(gate.matrix[7], 0);
}

TEST(GatesQsimTest, GateFS) {
  float theta = 0.84;
  float phi = 0.42;
  unsigned time = 7;
  unsigned qubit0 = 11;
  unsigned qubit1 = 12;
  auto gate = GateFS<float>::Create(time, qubit0, qubit1, theta, phi);

  float ct = std::cos(theta);
  float st = std::sin(theta);
  float cp = std::cos(phi);
  float sp = std::sin(phi);

  EXPECT_EQ(gate.time, time);
  EXPECT_EQ(gate.qubits.size(), 2);
  EXPECT_EQ(gate.qubits[0], qubit0);
  EXPECT_EQ(gate.qubits[1], qubit1);

  EXPECT_FLOAT_EQ(gate.matrix[0], 1);
  EXPECT_FLOAT_EQ(gate.matrix[1], 0);
  EXPECT_FLOAT_EQ(gate.matrix[2], 0);
  EXPECT_FLOAT_EQ(gate.matrix[3], 0);
  EXPECT_FLOAT_EQ(gate.matrix[4], 0);
  EXPECT_FLOAT_EQ(gate.matrix[5], 0);
  EXPECT_FLOAT_EQ(gate.matrix[6], 0);
  EXPECT_FLOAT_EQ(gate.matrix[7], 0);
  EXPECT_FLOAT_EQ(gate.matrix[8], 0);
  EXPECT_FLOAT_EQ(gate.matrix[9], 0);
  EXPECT_FLOAT_EQ(gate.matrix[10], ct);
  EXPECT_FLOAT_EQ(gate.matrix[11], 0);
  EXPECT_FLOAT_EQ(gate.matrix[12], 0);
  EXPECT_FLOAT_EQ(gate.matrix[13], -st);
  EXPECT_FLOAT_EQ(gate.matrix[14], 0);
  EXPECT_FLOAT_EQ(gate.matrix[15], 0);
  EXPECT_FLOAT_EQ(gate.matrix[16], 0);
  EXPECT_FLOAT_EQ(gate.matrix[17], 0);
  EXPECT_FLOAT_EQ(gate.matrix[18], 0);
  EXPECT_FLOAT_EQ(gate.matrix[19], -st);
  EXPECT_FLOAT_EQ(gate.matrix[20], ct);
  EXPECT_FLOAT_EQ(gate.matrix[21], 0);
  EXPECT_FLOAT_EQ(gate.matrix[22], 0);
  EXPECT_FLOAT_EQ(gate.matrix[23], 0);
  EXPECT_FLOAT_EQ(gate.matrix[24], 0);
  EXPECT_FLOAT_EQ(gate.matrix[25], 0);
  EXPECT_FLOAT_EQ(gate.matrix[26], 0);
  EXPECT_FLOAT_EQ(gate.matrix[27], 0);
  EXPECT_FLOAT_EQ(gate.matrix[28], 0);
  EXPECT_FLOAT_EQ(gate.matrix[29], 0);
  EXPECT_FLOAT_EQ(gate.matrix[30], cp);
  EXPECT_FLOAT_EQ(gate.matrix[31], -sp);

  auto schmidt_decomp = GateFS<float>::SchmidtDecomp(theta, phi);

  EXPECT_EQ(schmidt_decomp.size(), 4);

  EXPECT_NEAR(schmidt_decomp[0][0][0], 0.90986878, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][0][1], 0.038230576, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][0][2], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][0][3], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][0][4], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][0][5], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][0][6], 0.89784938, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][0][7], -0.15228048, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][0], 0.90986878, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][1], 0.038230576, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][2], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][3], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][4], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][5], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][6], 0.89784938, 1e-6);
  EXPECT_NEAR(schmidt_decomp[0][1][7], -0.15228048, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][0], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][1], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][2], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][3], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][4], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][5], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][6], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][0][7], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][0], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][1], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][2], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][3], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][4], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][5], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][6], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[1][1][7], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][0], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][1], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][2], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][3], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][4], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][5], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][6], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][0][7], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][0], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][1], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][2], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][3], -0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][4], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][5], 0.43146351, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][6], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[2][1][7], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][0], 0.42463028, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][1], -0.081917875, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][2], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][3], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][4], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][5], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][6], -0.39822495, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][0][7], 0.16863659, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][0], 0.42463028, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][1], -0.081917875, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][2], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][3], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][4], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][5], 0, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][6], -0.39822495, 1e-6);
  EXPECT_NEAR(schmidt_decomp[3][1][7], 0.16863659, 1e-6);
}

TEST(GatesQsimTest, GateCP) {
  float phi = 0.42;
  unsigned time = 7;
  unsigned qubit0 = 11;
  unsigned qubit1 = 12;
  auto gate = GateCP<float>::Create(time, qubit0, qubit1, phi);

  float cp = std::cos(phi);
  float sp = std::sin(phi);

  EXPECT_EQ(gate.time, time);
  EXPECT_EQ(gate.qubits.size(), 2);
  EXPECT_EQ(gate.qubits[0], qubit0);
  EXPECT_EQ(gate.qubits[1], qubit1);

  EXPECT_FLOAT_EQ(gate.matrix[0], 1);
  EXPECT_FLOAT_EQ(gate.matrix[1], 0);
  EXPECT_FLOAT_EQ(gate.matrix[2], 0);
  EXPECT_FLOAT_EQ(gate.matrix[3], 0);
  EXPECT_FLOAT_EQ(gate.matrix[4], 0);
  EXPECT_FLOAT_EQ(gate.matrix[5], 0);
  EXPECT_FLOAT_EQ(gate.matrix[6], 0);
  EXPECT_FLOAT_EQ(gate.matrix[7], 0);
  EXPECT_FLOAT_EQ(gate.matrix[8], 0);
  EXPECT_FLOAT_EQ(gate.matrix[9], 0);
  EXPECT_FLOAT_EQ(gate.matrix[10], 1);
  EXPECT_FLOAT_EQ(gate.matrix[11], 0);
  EXPECT_FLOAT_EQ(gate.matrix[12], 0);
  EXPECT_FLOAT_EQ(gate.matrix[13], 0);
  EXPECT_FLOAT_EQ(gate.matrix[14], 0);
  EXPECT_FLOAT_EQ(gate.matrix[15], 0);
  EXPECT_FLOAT_EQ(gate.matrix[16], 0);
  EXPECT_FLOAT_EQ(gate.matrix[17], 0);
  EXPECT_FLOAT_EQ(gate.matrix[18], 0);
  EXPECT_FLOAT_EQ(gate.matrix[19], 0);
  EXPECT_FLOAT_EQ(gate.matrix[20], 1);
  EXPECT_FLOAT_EQ(gate.matrix[21], 0);
  EXPECT_FLOAT_EQ(gate.matrix[22], 0);
  EXPECT_FLOAT_EQ(gate.matrix[23], 0);
  EXPECT_FLOAT_EQ(gate.matrix[24], 0);
  EXPECT_FLOAT_EQ(gate.matrix[25], 0);
  EXPECT_FLOAT_EQ(gate.matrix[26], 0);
  EXPECT_FLOAT_EQ(gate.matrix[27], 0);
  EXPECT_FLOAT_EQ(gate.matrix[28], 0);
  EXPECT_FLOAT_EQ(gate.matrix[29], 0);
  EXPECT_FLOAT_EQ(gate.matrix[30], cp);
  EXPECT_FLOAT_EQ(gate.matrix[31], -sp);

  auto schmidt_decomp = GateCP<float>::SchmidtDecomp(phi);

  EXPECT_EQ(schmidt_decomp.size(), 2);

  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][0], 1);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][1], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][2], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][3], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][4], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][5], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][6], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][0][7], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][0], 1);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][1], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][2], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][3], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][4], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][5], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][6], 1);
  EXPECT_FLOAT_EQ(schmidt_decomp[0][1][7], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][0], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][1], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][2], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][3], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][4], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][5], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][6], 1);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][0][7], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][0], 1);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][1], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][2], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][3], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][4], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][5], 0);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][6], cp);
  EXPECT_FLOAT_EQ(schmidt_decomp[1][1][7], -sp);
}

TEST(GatesQsimTest, GateMeasurement) {
  unsigned time = 5;
  std::vector<unsigned> qubits = {3, 2, 4, 0, 7, 5, 1};
  auto gate = gate::Measurement<GateQSim<float>>::Create(time, qubits);

  EXPECT_EQ(gate.time, time);
  EXPECT_EQ(gate.qubits.size(), qubits.size());

  for (std::size_t i = 0; i < qubits.size(); ++i) {
    EXPECT_EQ(gate.qubits[i], qubits[i]);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
