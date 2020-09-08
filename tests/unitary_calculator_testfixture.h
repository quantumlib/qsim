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

#include <complex>

#include "gtest/gtest.h"

namespace qsim {

namespace unitary {

namespace {

template <typename UnitarySpace, typename Unitary>
void FillMatrix(UnitarySpace& us, Unitary& u, int n) {
  // Intentionally create non-unitary matrix with ascending elements.
  for(int i =0; i < (1 << n); i++){
    for(int j =0;j < (1 << n); j++) {
      us.SetEntry(u, i, j, 2 * i * (1 << n) + 2 * j,
        2 * i * (1 << n) + 2 * j + 1);
    }
  }
}

}  // namespace

constexpr char provider[] = "unitary_calculator_test";

template <typename UC>
void TestApplyGate1() {
  const int n_qubits = 3;
  UC uc(n_qubits, 1);
  using UnitarySpace = typename UC::UnitarySpace;
  using Unitary = typename UC::Unitary;

  UnitarySpace us(n_qubits, 1);
  Unitary u = us.CreateUnitary();

  float ref_gate[] = {1,2,3,4,5,6,7,8};

  // Test applying on qubit 0.
  FillMatrix(us, u, n_qubits);
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

  for(int i =0;i<8;i++){
    for(int j = 0;j<8;j++) {
      EXPECT_EQ(us.GetEntry(u, i, j),
        std::complex<float>(
          expected_mat_0[2 * i * 8 + 2 * j],
          expected_mat_0[2 * i * 8 + 2 * j + 1]));
    }
  }

  // Test applying on qubit 1.
  FillMatrix(us, u, n_qubits);
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

  for(int i =0;i<8;i++){
    for(int j = 0;j<8;j++) {
      EXPECT_EQ(us.GetEntry(u, i, j),
        std::complex<float>(
          expected_mat_1[2 * i * 8 + 2 * j],
          expected_mat_1[2 * i * 8 + 2 * j + 1]));
    }
  }

  // Test applying on qubit 2.
  FillMatrix(us, u, n_qubits);
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

  for(int i =0;i<8;i++){
    for(int j = 0;j<8;j++) {
      EXPECT_EQ(us.GetEntry(u, i, j),
        std::complex<float>(
          expected_mat_2[2 * i * 8 + 2 * j],
          expected_mat_2[2 * i * 8 + 2 * j + 1]));
    }
  }
}

template <typename UC>
void TestApplyGate2() {

  const int n_qubits = 3;
  UC uc(n_qubits, 1);
  using UnitarySpace = typename UC::UnitarySpace;
  using Unitary = typename UC::Unitary;

  UnitarySpace us(n_qubits, 1);
  Unitary u = us.CreateUnitary();

  // clang-format off
  float ref_gate[] = {1,2,3,4,5,6,7,8,
                      9,10,11,12,13,14,15,16,
                      17,18,19,20,21,22,23,24,
                      25,26,27,28,29,30,31,32};
  // clang-format on

  // Test applying on qubit 0, 1
  FillMatrix(us, u, n_qubits);
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

  for(int i =0;i<8;i++){
    for(int j = 0;j<8;j++) {
      EXPECT_EQ(us.GetEntry(u, i, j),
        std::complex<float>(
          expected_mat_01[2 * i * 8 + 2 * j],
          expected_mat_01[2 * i * 8 + 2 * j + 1]));
    }
  }

  // Test applying on qubit 1, 2
  FillMatrix(us, u, n_qubits);
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

  for(int i =0;i<8;i++){
    for(int j = 0;j<8;j++) {
      EXPECT_EQ(us.GetEntry(u, i, j),
        std::complex<float>(
          expected_mat_12[2 * i * 8 + 2 * j],
          expected_mat_12[2 * i * 8 + 2 * j + 1]));
    }
  }

  // Test applying on qubit 0, 2
  FillMatrix(us, u, n_qubits);
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

  for(int i =0;i<8;i++){
    for(int j = 0;j<8;j++) {
      EXPECT_EQ(us.GetEntry(u, i, j),
        std::complex<float>(
          expected_mat_02[2 * i * 8 + 2 * j],
          expected_mat_02[2 * i * 8 + 2 * j + 1]));
    }
  }
}

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_TESTFIXTURE_H_
