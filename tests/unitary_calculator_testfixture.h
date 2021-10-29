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

#ifndef UNITARY_CALCULTOR_TESTFIXTURE_H_
#define UNITARY_CALCULTOR_TESTFIXTURE_H_

#include <algorithm>
#include <complex>
#include <cmath>
#include <vector>

#include "../lib/fuser.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_cirq.h"
#include "gtest/gtest.h"

namespace qsim {

namespace unitary {

namespace {

template <typename UnitarySpace, typename Unitary>
void FillMatrix(UnitarySpace& us, Unitary& u, int n) {
  // Intentionally create non-unitary matrix with ascending elements.
  for (int i = 0; i < (1 << n); i++) {
    for (int j = 0; j < (1 << n); j++) {
      auto val = 2 * j * (1 << n) + 2 * i;
      us.SetEntry(u, i, j, val, val + 1);
    }
  }
}

template <typename UnitarySpace, typename Unitary>
void FillMatrix2(UnitarySpace& us, Unitary& u, uint64_t size, uint64_t delta) {
  // Intentionally create non-unitary matrix with ascending elements.
  for (uint64_t i = 0; i < size; ++i) {
    for (uint64_t j = 0; j < size; ++j) {
      auto val = 2 * i * delta + 2 * j;
      us.SetEntry(u, i, j, val, (val + 1));
    }
  }
}

template <typename UnitarySpace, typename Unitary>
void EUnitaryEQ(UnitarySpace& us, Unitary& u, int n, float* expected) {
  for (int i = 0; i < (1 << n); i++) {
    for (int j = 0; j < (1 << n); j++) {
      int ind = 2 * j * (1 << n) + 2 * i;
      auto out = us.GetEntry(u, i, j);
      std::complex<float> e_val =
          std::complex<float>(expected[ind], expected[ind + 1]);
      EXPECT_EQ(out, e_val) << "Mismatch in unitary at: " << i << "," << j
                            << " Expected: " << e_val << " Got: " << out;
    }
  }
}

}  // namespace

template <typename UC>
void TestApplyGate1() {
  const int num_qubits = 3;

  using UnitarySpace = typename UC::UnitarySpace;
  using Unitary = typename UC::Unitary;

  UC uc(1);
  UnitarySpace us(1);
  Unitary u = us.CreateUnitary(num_qubits);

  float ref_gate[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test applying on qubit 0.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_0[] = {
    -22,116,-26,136,-30,156,-34,176,-38,196,-42,216,-46,236,-50,256,
    -30,252,-34,304,-38,356,-42,408,-46,460,-50,512,-54,564,-58,616,
    -86,436,-90,456,-94,476,-98,496,-102,516,-106,536,-110,556,-114,576,
    -94,1084,-98,1136,-102,1188,-106,1240,-110,1292,-114,1344,-118,1396,-122,1448,
    -150,756,-154,776,-158,796,-162,816,-166,836,-170,856,-174,876,-178,896,
    -158,1916,-162,1968,-166,2020,-170,2072,-174,2124,-178,2176,-182,2228,-186,2280,
    -214,1076,-218,1096,-222,1116,-226,1136,-230,1156,-234,1176,-238,1196,-242,1216,
    -222,2748,-226,2800,-230,2852,-234,2904,-238,2956,-242,3008,-246,3060,-250,3112
  };
  // clang-format on
  uc.ApplyGate({0}, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_0);

  // Test applying on qubit 1.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_1[] = {
    -38,228,-42,248,-46,268,-50,288,-54,308,-58,328,-62,348,-66,368,
    -70,388,-74,408,-78,428,-82,448,-86,468,-90,488,-94,508,-98,528,
    -46,492,-50,544,-54,596,-58,648,-62,700,-66,752,-70,804,-74,856,
    -78,908,-82,960,-86,1012,-90,1064,-94,1116,-98,1168,-102,1220,-106,1272,
    -166,868,-170,888,-174,908,-178,928,-182,948,-186,968,-190,988,-194,1008,
    -198,1028,-202,1048,-206,1068,-210,1088,-214,1108,-218,1128,-222,1148,-226,1168,
    -174,2156,-178,2208,-182,2260,-186,2312,-190,2364,-194,2416,-198,2468,-202,2520,
    -206,2572,-210,2624,-214,2676,-218,2728,-222,2780,-226,2832,-230,2884,-234,2936
  };
  // clang-format on
  uc.ApplyGate({1}, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_1);

  // Test applying on qubit 2.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_2[] = {
    -70,452,-74,472,-78,492,-82,512,-86,532,-90,552,-94,572,-98,592,
    -102,612,-106,632,-110,652,-114,672,-118,692,-122,712,-126,732,-130,752,
    -134,772,-138,792,-142,812,-146,832,-150,852,-154,872,-158,892,-162,912,
    -166,932,-170,952,-174,972,-178,992,-182,1012,-186,1032,-190,1052,-194,1072,
    -78,972,-82,1024,-86,1076,-90,1128,-94,1180,-98,1232,-102,1284,-106,1336,
    -110,1388,-114,1440,-118,1492,-122,1544,-126,1596,-130,1648,-134,1700,-138,1752,
    -142,1804,-146,1856,-150,1908,-154,1960,-158,2012,-162,2064,-166,2116,-170,2168,
    -174,2220,-178,2272,-182,2324,-186,2376,-190,2428,-194,2480,-198,2532,-202,2584
  };
  // clang-format on
  uc.ApplyGate({2}, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_2);
}

template <typename UC>
void TestApplyControlledGate1() {
  const int num_qubits = 3;

  using UnitarySpace = typename UC::UnitarySpace;
  using Unitary = typename UC::Unitary;

  UC uc(1);
  UnitarySpace us(1);
  Unitary u = us.CreateUnitary(num_qubits);

  float ref_gate[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test applying on qubit 0 controlling 1.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_0[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    -86.0,436.0,-90.0,456.0,-94.0,476.0,-98.0,496.0,-102.0,516.0,-106.0,536.0,-110.0,556.0,-114.0,576.0,
    -94.0,1084.0,-98.0,1136.0,-102.0,1188.0,-106.0,1240.0,-110.0,1292.0,-114.0,1344.0,-118.0,1396.0,-122.0,1448.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,
    -214.0,1076.0,-218.0,1096.0,-222.0,1116.0,-226.0,1136.0,-230.0,1156.0,-234.0,1176.0,-238.0,1196.0,-242.0,1216.0,
    -222.0,2748.0,-226.0,2800.0,-230.0,2852.0,-234.0,2904.0,-238.0,2956.0,-242.0,3008.0,-246.0,3060.0,-250.0,3112.0,
  };
  // clang-format on
  uc.ApplyControlledGate({0}, {1}, 1, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_0);

  // Test applying on qubit 0 controlling 2.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_1[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,
    48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,
    -150.0,756.0,-154.0,776.0,-158.0,796.0,-162.0,816.0,-166.0,836.0,-170.0,856.0,-174.0,876.0,-178.0,896.0,
    -158.0,1916.0,-162.0,1968.0,-166.0,2020.0,-170.0,2072.0,-174.0,2124.0,-178.0,2176.0,-182.0,2228.0,-186.0,2280.0,
    -214.0,1076.0,-218.0,1096.0,-222.0,1116.0,-226.0,1136.0,-230.0,1156.0,-234.0,1176.0,-238.0,1196.0,-242.0,1216.0,
    -222.0,2748.0,-226.0,2800.0,-230.0,2852.0,-234.0,2904.0,-238.0,2956.0,-242.0,3008.0,-246.0,3060.0,-250.0,3112.0,
  };
  // clang-format on
  uc.ApplyControlledGate({0}, {2}, 1, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_1);

  // Test applying on qubit 0 controlling on 1 and 2.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_2[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,
    48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,
    -214.0,1076.0,-218.0,1096.0,-222.0,1116.0,-226.0,1136.0,-230.0,1156.0,-234.0,1176.0,-238.0,1196.0,-242.0,1216.0,
    -222.0,2748.0,-226.0,2800.0,-230.0,2852.0,-234.0,2904.0,-238.0,2956.0,-242.0,3008.0,-246.0,3060.0,-250.0,3112.0,
  };
  // clang-format on
  uc.ApplyControlledGate({0}, {1, 2}, 3, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_2);

  // Test applying on qubit 1 controlling on 0.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_3[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    -70.0,388.0,-74.0,408.0,-78.0,428.0,-82.0,448.0,-86.0,468.0,-90.0,488.0,-94.0,508.0,-98.0,528.0,
    32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,
    -78.0,908.0,-82.0,960.0,-86.0,1012.0,-90.0,1064.0,-94.0,1116.0,-98.0,1168.0,-102.0,1220.0,-106.0,1272.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    -198.0,1028.0,-202.0,1048.0,-206.0,1068.0,-210.0,1088.0,-214.0,1108.0,-218.0,1128.0,-222.0,1148.0,-226.0,1168.0,
    96.0,97.0,98.0,99.0,100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0,110.0,111.0,
    -206.0,2572.0,-210.0,2624.0,-214.0,2676.0,-218.0,2728.0,-222.0,2780.0,-226.0,2832.0,-230.0,2884.0,-234.0,2936.0,
  };
  // clang-format on
  uc.ApplyControlledGate({1}, {0}, 1, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_3);

  // Test applying on qubit 1 controlling on 0 and 2.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_4[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,
    48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    -198.0,1028.0,-202.0,1048.0,-206.0,1068.0,-210.0,1088.0,-214.0,1108.0,-218.0,1128.0,-222.0,1148.0,-226.0,1168.0,
    96.0,97.0,98.0,99.0,100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0,110.0,111.0,
    -206.0,2572.0,-210.0,2624.0,-214.0,2676.0,-218.0,2728.0,-222.0,2780.0,-226.0,2832.0,-230.0,2884.0,-234.0,2936.0,
  };
  // clang-format on
  uc.ApplyControlledGate({1}, {0, 2}, 3, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_4);

  // Test applying on qubit 2 controlling on 1.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_5[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    -134.0,772.0,-138.0,792.0,-142.0,812.0,-146.0,832.0,-150.0,852.0,-154.0,872.0,-158.0,892.0,-162.0,912.0,
    -166.0,932.0,-170.0,952.0,-174.0,972.0,-178.0,992.0,-182.0,1012.0,-186.0,1032.0,-190.0,1052.0,-194.0,1072.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,
    -142.0,1804.0,-146.0,1856.0,-150.0,1908.0,-154.0,1960.0,-158.0,2012.0,-162.0,2064.0,-166.0,2116.0,-170.0,2168.0,
    -174.0,2220.0,-178.0,2272.0,-182.0,2324.0,-186.0,2376.0,-190.0,2428.0,-194.0,2480.0,-198.0,2532.0,-202.0,2584.0,
  };
  // clang-format on
  uc.ApplyControlledGate({2}, {1}, 1, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_5);

  // Test applying on qubit 2 controlling on 0 and 1.
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_6[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,
    -166.0,932.0,-170.0,952.0,-174.0,972.0,-178.0,992.0,-182.0,1012.0,-186.0,1032.0,-190.0,1052.0,-194.0,1072.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,
    96.0,97.0,98.0,99.0,100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0,110.0,111.0,
    -174.0,2220.0,-178.0,2272.0,-182.0,2324.0,-186.0,2376.0,-190.0,2428.0,-194.0,2480.0,-198.0,2532.0,-202.0,2584.0,
  };
  // clang-format on
  uc.ApplyControlledGate({2}, {0, 1}, 3, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_6);
}

template <typename UC>
void TestApplyGate2() {
  const int num_qubits = 3;

  using UnitarySpace = typename UC::UnitarySpace;
  using Unitary = typename UC::Unitary;

  UC uc(1);
  UnitarySpace us(1);
  Unitary u = us.CreateUnitary(num_qubits);

  // clang-format off
  float ref_gate[] = {1,2,3,4,5,6,7,8,
                      9,10,11,12,13,14,15,16,
                      17,18,19,20,21,22,23,24,
                      25,26,27,28,29,30,31,32};
  // clang-format on

  // Test applying on qubit 0, 1
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_01[] = {
    -116,1200,-124,1272,-132,1344,-140,1416,-148,1488,-156,1560,-164,1632,-172,1704,
    -148,2768,-156,2968,-164,3168,-172,3368,-180,3568,-188,3768,-196,3968,-204,4168,
    -180,4336,-188,4664,-196,4992,-204,5320,-212,5648,-220,5976,-228,6304,-236,6632,
    -212,5904,-220,6360,-228,6816,-236,7272,-244,7728,-252,8184,-260,8640,-268,9096,
    -372,3504,-380,3576,-388,3648,-396,3720,-404,3792,-412,3864,-420,3936,-428,4008,
    -404,9168,-412,9368,-420,9568,-428,9768,-436,9968,-444,10168,-452,10368,-460,10568,
    -436,14832,-444,15160,-452,15488,-460,15816,-468,16144,-476,16472,-484,16800,-492,17128,
    -468,20496,-476,20952,-484,21408,-492,21864,-500,22320,-508,22776,-516,23232,-524,23688
  };
  // clang-format on
  uc.ApplyGate({0, 1}, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_01);

  // Test applying on qubit 1, 2
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_12[] = {
    -212,2384,-220,2456,-228,2528,-236,2600,-244,2672,-252,2744,-260,2816,-268,2888,
    -276,2960,-284,3032,-292,3104,-300,3176,-308,3248,-316,3320,-324,3392,-332,3464,
    -244,5488,-252,5688,-260,5888,-268,6088,-276,6288,-284,6488,-292,6688,-300,6888,
    -308,7088,-316,7288,-324,7488,-332,7688,-340,7888,-348,8088,-356,8288,-364,8488,
    -276,8592,-284,8920,-292,9248,-300,9576,-308,9904,-316,10232,-324,10560,-332,10888,
    -340,11216,-348,11544,-356,11872,-364,12200,-372,12528,-380,12856,-388,13184,-396,13512,
    -308,11696,-316,12152,-324,12608,-332,13064,-340,13520,-348,13976,-356,14432,-364,14888,
    -372,15344,-380,15800,-388,16256,-396,16712,-404,17168,-412,17624,-420,18080,-428,18536,
  };
  // clang-format on
  uc.ApplyGate({1, 2}, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_12);

  // Test applying on qubit 0, 2
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_02[] = {
    -180,2032,-188,2104,-196,2176,-204,2248,-212,2320,-220,2392,-228,2464,-236,2536,
    -212,4624,-220,4824,-228,5024,-236,5224,-244,5424,-252,5624,-260,5824,-268,6024,
    -308,3184,-316,3256,-324,3328,-332,3400,-340,3472,-348,3544,-356,3616,-364,3688,
    -340,7824,-348,8024,-356,8224,-364,8424,-372,8624,-380,8824,-388,9024,-396,9224,
    -244,7216,-252,7544,-260,7872,-268,8200,-276,8528,-284,8856,-292,9184,-300,9512,
    -276,9808,-284,10264,-292,10720,-300,11176,-308,11632,-316,12088,-324,12544,-332,13000,
    -372,12464,-380,12792,-388,13120,-396,13448,-404,13776,-412,14104,-420,14432,-428,14760,
    -404,17104,-412,17560,-420,18016,-428,18472,-436,18928,-444,19384,-452,19840,-460,20296,
  };
  // clang-format on
  uc.ApplyGate({0, 2}, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_02);
}

template <typename UC>
void TestApplyControlledGate2() {
  const int num_qubits = 3;

  using UnitarySpace = typename UC::UnitarySpace;
  using Unitary = typename UC::Unitary;

  UC uc(1);
  UnitarySpace us(1);
  Unitary u = us.CreateUnitary(num_qubits);

  // clang-format off
  float ref_gate[] = {1,2,3,4,5,6,7,8,
                      9,10,11,12,13,14,15,16,
                      17,18,19,20,21,22,23,24,
                      25,26,27,28,29,30,31,32};
  // clang-format on

  // Test applying on qubit 0, 1
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_01[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,
    48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,
    -372.0,3504.0,-380.0,3576.0,-388.0,3648.0,-396.0,3720.0,-404.0,3792.0,-412.0,3864.0,-420.0,3936.0,-428.0,4008.0,
    -404.0,9168.0,-412.0,9368.0,-420.0,9568.0,-428.0,9768.0,-436.0,9968.0,-444.0,10168.0,-452.0,10368.0,-460.0,10568.0,
    -436.0,14832.0,-444.0,15160.0,-452.0,15488.0,-460.0,15816.0,-468.0,16144.0,-476.0,16472.0,-484.0,16800.0,-492.0,17128.0,
    -468.0,20496.0,-476.0,20952.0,-484.0,21408.0,-492.0,21864.0,-500.0,22320.0,-508.0,22776.0,-516.0,23232.0,-524.0,23688.0,
  };
  // clang-format on
  uc.ApplyControlledGate({0, 1}, {2}, 1, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_01);

  // Test applying on qubit 1, 2
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_12[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    -276.0,2960.0,-284.0,3032.0,-292.0,3104.0,-300.0,3176.0,-308.0,3248.0,-316.0,3320.0,-324.0,3392.0,-332.0,3464.0,
    32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,
    -308.0,7088.0,-316.0,7288.0,-324.0,7488.0,-332.0,7688.0,-340.0,7888.0,-348.0,8088.0,-356.0,8288.0,-364.0,8488.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    -340.0,11216.0,-348.0,11544.0,-356.0,11872.0,-364.0,12200.0,-372.0,12528.0,-380.0,12856.0,-388.0,13184.0,-396.0,13512.0,
    96.0,97.0,98.0,99.0,100.0,101.0,102.0,103.0,104.0,105.0,106.0,107.0,108.0,109.0,110.0,111.0,
    -372.0,15344.0,-380.0,15800.0,-388.0,16256.0,-396.0,16712.0,-404.0,17168.0,-412.0,17624.0,-420.0,18080.0,-428.0,18536.0,
  };
  // clang-format on
  uc.ApplyControlledGate({1, 2}, {0}, 1, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_12);

  // Test applying on qubit 0, 2
  FillMatrix(us, u, num_qubits);
  // clang-format off
  float expected_mat_02[] = {
    0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,
    16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,
    -308.0,3184.0,-316.0,3256.0,-324.0,3328.0,-332.0,3400.0,-340.0,3472.0,-348.0,3544.0,-356.0,3616.0,-364.0,3688.0,
    -340.0,7824.0,-348.0,8024.0,-356.0,8224.0,-364.0,8424.0,-372.0,8624.0,-380.0,8824.0,-388.0,9024.0,-396.0,9224.0,
    64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,
    80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,
    -372.0,12464.0,-380.0,12792.0,-388.0,13120.0,-396.0,13448.0,-404.0,13776.0,-412.0,14104.0,-420.0,14432.0,-428.0,14760.0,
    -404.0,17104.0,-412.0,17560.0,-420.0,18016.0,-428.0,18472.0,-436.0,18928.0,-444.0,19384.0,-452.0,19840.0,-460.0,20296.0,
  };
  // clang-format on
  uc.ApplyControlledGate({0, 2}, {1}, 1, ref_gate, u);
  EUnitaryEQ(us, u, num_qubits, expected_mat_02);
}

template <typename UnitaryCalculator>
void TestApplyFusedGate() {
  using UnitarySpace = typename UnitaryCalculator::UnitarySpace;
  using Unitary = typename UnitaryCalculator::Unitary;
  using fp_type = typename UnitaryCalculator::fp_type;
  using Gate = Cirq::GateCirq<fp_type>;

  unsigned num_qubits = 1;

  UnitaryCalculator uc(1);
  UnitarySpace us(1);

  Unitary u = us.CreateUnitary(num_qubits);
  us.SetIdentity(u);

  std::vector<Gate> gates = {Cirq::H<fp_type>::Create(0, 0),
                             Cirq::H<fp_type>::Create(1, 0)};

  GateFused<Gate> fgate {Cirq::kH, 0, {0}, &gates[0],
                         {&gates[0], &gates[1]}, {}};

  CalculateFusedMatrix(fgate);
  ApplyFusedGate(uc, fgate, u);

  unsigned size = 1 << num_qubits;
  for (unsigned i = 0; i < size; ++i) {
    for (unsigned j = 0; j < size; ++j) {
      auto a = us.GetEntry(u, i, j);
      if (i == j) {
        EXPECT_NEAR(std::real(a), 1, 1e-6);
      } else {
        EXPECT_NEAR(std::real(a), 0, 1e-6);
      }

      EXPECT_NEAR(std::imag(a), 0, 1e-6);
    }
  }
}

template <typename UnitaryCalculator>
void TestApplyGates(bool test_double) {
  using UnitarySpace = typename UnitaryCalculator::UnitarySpace;
  using fp_type = typename UnitaryCalculator::fp_type;

  unsigned max_minq = std::log2(UnitaryCalculator::SIMDRegisterSize());
  unsigned max_gate_qubits = 6;
  unsigned num_qubits = max_gate_qubits + max_minq;

  UnitarySpace unitary_space(1);
  UnitaryCalculator simulator(1);

  auto unitary = unitary_space.CreateUnitary(num_qubits);

  std::vector<fp_type> matrix;
  matrix.reserve(1 << (2 * max_gate_qubits + 1));

  std::vector<unsigned> qubits;
  qubits.reserve(max_gate_qubits);

  uint64_t size = 1 << num_qubits;

  // Test 1-, 2-, ..., max_gate_qubits- qubit gates.
  for (unsigned q = 1; q <= max_gate_qubits; ++q) {
    uint64_t size1 = 1 << q;
    uint64_t size2 = size1 * size1;

    matrix.resize(0);

    for (unsigned i = 0; i < 2 * size2; ++i) {
      matrix.push_back(i + 1);
    }

    // k is the first gate qubit.
    for (unsigned k = 0; k <= max_minq; ++k) {
      qubits.resize(0);

      // Gate qbuits are consecuitive from k to k + q - 1.
      for (unsigned i = 0; i < q; ++i) {
        qubits.push_back(i + k);
      }

      uint64_t delta = 42;  // Some random value.
      uint64_t mask = ((size - 1) ^ (((1 << q) - 1) << k));

      FillMatrix2(unitary_space, unitary, size, delta);
      simulator.ApplyGate(qubits, matrix.data(), unitary);

      for (uint64_t i = 0; i < size; ++i) {
        uint64_t a0 = 2 * i * delta;

        for (uint64_t j = 0; j < size; ++j) {
          uint64_t s = j & mask;
          uint64_t l = (j ^ s) >> k;

          // Expected results are calculated analytically.
          fp_type expected_real =
              -fp_type(2 * size2 * l + size1 * (2 * s + a0 + 2)
                       + (1 + (1 << k)) * (size2 - size1));
          fp_type expected_imag = -expected_real - size1
              + 2 * (size1 * (1 << k) * (size1 - 1)
                           * (1 + 2 * size1 * (2 + 3 * l))
                     + 3 * size2 * (1 + 2 * l) * (a0 + 2 * s)) / 3;

          auto a = unitary_space.GetEntry(unitary, i, j);

          if (test_double) {
            EXPECT_NEAR(std::real(a), expected_real, 1e-6);
            EXPECT_NEAR(std::imag(a), expected_imag, 1e-6);
          } else {
            // float does not have enough precision to test as above.
            EXPECT_NEAR(1, std::real(a) / expected_real, 5e-4);
            EXPECT_NEAR(1, std::imag(a) / expected_imag, 5e-4);
          }
        }
      }
    }
  }
}

template <typename UnitaryCalculator>
void TestApplyControlledGates(bool test_double) {
  using UnitarySpace = typename UnitaryCalculator::UnitarySpace;
  using fp_type = typename UnitaryCalculator::fp_type;

  unsigned max_qubits = 3 + std::log2(UnitaryCalculator::SIMDRegisterSize());
  unsigned max_target_qubits = 3;
  unsigned max_control_qubits = 2;

  UnitarySpace unitary_space(1);
  UnitaryCalculator simulator(1);

  auto unitary = unitary_space.CreateUnitary(max_qubits);

  std::vector<unsigned> qubits;
  qubits.reserve(max_qubits);

  std::vector<unsigned> cqubits;
  cqubits.reserve(max_qubits);

  std::vector<fp_type> matrix;
  matrix.reserve(1 << (2 * max_target_qubits + 1));

  // Iterate over circuit size.
  for (unsigned num_qubits = 2; num_qubits <= max_qubits; ++num_qubits) {
    unsigned size = 1 << num_qubits;
    unsigned nmask = size - 1;

    // Iterate over control qubits (as a binary mask).
    for (unsigned cmask = 0; cmask <= nmask; ++cmask) {
      cqubits.resize(0);

      for (unsigned q = 0; q < num_qubits; ++q) {
        if (((cmask >> q) & 1) != 0) {
          cqubits.push_back(q);
        }
      }

      if (cqubits.size() == 0
          || cqubits.size() > std::min(max_control_qubits, num_qubits - 1)) {
        continue;
      }

      // Iterate over target qubits (as a binary mask).
      for (unsigned mask = 0; mask <= nmask; ++mask) {
        unsigned qmask = mask & (cmask ^ nmask);

        qubits.resize(0);

        for (unsigned q = 0; q < num_qubits; ++q) {
          if (((qmask >> q) & 1) > 0) {
            qubits.push_back(q);
          }
        }

        if (cmask != (mask & cmask)) continue;

        unsigned num_available = num_qubits - cqubits.size();
        if (qubits.size() == 0
            || qubits.size() > std::min(max_target_qubits, num_available)) {
          continue;
        }

        // Target qubits are consecuitive.
        std::size_t i = 1;
        for (; i < qubits.size(); ++i) {
          if (qubits[i - 1] + 1 != qubits[i]) break;
        }
        if (i < qubits.size()) continue;

        unsigned k = qubits[0];

        unsigned size1 = 1 << qubits.size();
        unsigned size2 = size1 * size1;

        matrix.resize(0);
        // Non-unitary gate matrix.
        for (unsigned i = 0; i < 2 * size2; ++i) {
          matrix.push_back(i + 1);
        }

        unsigned zmask = nmask ^ qmask;

        // Iterate over control values (all zeros or all ones).
        std::vector<unsigned> cvals = {0, (1U << cqubits.size()) - 1};
        for (unsigned cval : cvals) {
          unsigned cvmask = cval == 0 ? 0 : cmask;

          // Starting unitary.
          uint64_t delta = 42;  // Some random value.
          FillMatrix2(unitary_space, unitary, size, delta);

          simulator.ApplyControlledGate(
              qubits, cqubits, cval, matrix.data(), unitary);

          // Test results.
          for (uint64_t i = 0; i < size; ++i) {
            uint64_t a0 = 2 * i * delta;

            for (unsigned j = 0; j < size; ++j) {
              auto a = unitary_space.GetEntry(unitary, i, j);

              if ((j & cmask) == cvmask) {
                // The target matrix is applied.

                unsigned s = j & zmask;
                unsigned l = (j ^ s) >> k;

                // Expected results are calculated analytically.
                fp_type expected_real =
                    -fp_type(2 * size2 * l + size1 * (2 * s + a0 + 2)
                             + (1 + (1 << k)) * (size2 - size1));
                fp_type expected_imag = -expected_real - size1
                    + 2 * (size1 * (1 << k) * (size1 - 1)
                                 * (1 + 2 * size1 * (2 + 3 * l))
                           + 3 * size2 * (1 + 2 * l) * (a0 + 2 * s)) / 3;

                if (test_double) {
                  EXPECT_NEAR(std::real(a), expected_real, 1e-6);
                  EXPECT_NEAR(std::imag(a), expected_imag, 1e-6);
                } else {
                  // float does not have enough precision to test as above.
                  EXPECT_NEAR(1, std::real(a) / expected_real, 3e-5);
                  EXPECT_NEAR(1, std::imag(a) / expected_imag, 3e-5);
                }
              } else {
                // The target matrix is not applied. Unmodified entries.

                fp_type expected_real = 2 * i * delta + 2 * j;
                fp_type expected_imag = expected_real + 1;

                EXPECT_NEAR(std::real(a), expected_real, 1e-6);
                EXPECT_NEAR(std::imag(a), expected_imag, 1e-6);
              }
            }
          }
        }
      }
    }
  }
}

template <typename UnitaryCalculator>
void TestSmallCircuits() {
  using UnitarySpace = typename UnitaryCalculator::UnitarySpace;
  using fp_type = typename UnitaryCalculator::fp_type;

  unsigned max_num_qubits = 5;

  UnitarySpace unitary_space(1);
  UnitaryCalculator simulator(1);

  std::vector<fp_type> matrix;
  matrix.reserve(1 << (2 * max_num_qubits + 1));

  std::vector<unsigned> qubits;
  qubits.reserve(max_num_qubits);

  for (unsigned num_qubits = 1; num_qubits <= max_num_qubits; ++num_qubits) {
    auto unitary = unitary_space.CreateUnitary(num_qubits);

    unsigned maxq = std::min(num_qubits, 3U);
    uint64_t size = 1 << num_qubits;

    for (unsigned q = 1; q <= maxq; ++q) {
      unsigned max_minq = num_qubits - q;

      uint64_t size1 = 1 << q;
      uint64_t size2 = size1 * size1;

      matrix.resize(0);

      for (unsigned i = 0; i < 2 * size2; ++i) {
        matrix.push_back(i + 1);
      }

      // k is the first gate qubit.
      for (unsigned k = 0; k <= max_minq; ++k) {
        qubits.resize(0);

        // Gate qbuits are consecuitive from k to k + q - 1.
        for (unsigned i = 0; i < q; ++i) {
          qubits.push_back(i + k);
        }

        uint64_t delta = 42;  // Some random value.
        uint64_t mask = ((size - 1) ^ (((1 << q) - 1) << k));

        FillMatrix2(unitary_space, unitary, size, delta);
        simulator.ApplyGate(qubits, matrix.data(), unitary);

        for (uint64_t i = 0; i < size; ++i) {
          uint64_t a0 = 2 * i * delta;

          for (uint64_t j = 0; j < size; ++j) {
            uint64_t s = j & mask;
            uint64_t l = (j ^ s) >> k;

            // Expected results are calculated analytically.
            fp_type expected_real =
                -fp_type(2 * size2 * l + size1 * (2 * s + a0 + 2)
                         + (1 + (1 << k)) * (size2 - size1));
            fp_type expected_imag = -expected_real - size1
                + 2 * (size1 * (1 << k) * (size1 - 1)
                             * (1 + 2 * size1 * (2 + 3 * l))
                       + 3 * size2 * (1 + 2 * l) * (a0 + 2 * s)) / 3;

            auto a = unitary_space.GetEntry(unitary, i, j);

            EXPECT_NEAR(std::real(a), expected_real, 1e-6);
            EXPECT_NEAR(std::imag(a), expected_imag, 1e-6);
          }
        }
      }
    }
  }
}

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_TESTFIXTURE_H_
