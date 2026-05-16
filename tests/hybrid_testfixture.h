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

#ifndef HYBRID_TESTFIXTURE_H_
#define HYBRID_TESTFIXTURE_H_

#include <cmath>
#include <complex>
#include <cstdint>
#include <sstream>

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_basic.h"
#include "../lib/hybrid.h"
#include "../lib/io.h"
#include "../lib/operation.h"

namespace qsim {

template <typename Factory>
void TestHybrid2(const Factory& factory) {
  constexpr char provider[] = "hybrid_test";
  constexpr char circuit_string[] =
R"(2
0 h 0
0 h 1
1 cz 0 1
2 t 0
2 t 1
3 cz 0 1
4 x_1_2 0
4 y_1_2 1
5 cz 0 1
6 t 0
6 t 1
7 cz 0 1
8 rx 0 0.7
8 ry 1 0.3
9 cz 0 1
10 t 0
10 t 1
11 x_1_2 0
11 y_1_2 1
12 cz 0 1
13 x_1_2 0
13 y_1_2 1
14 sw 0 1
)";

  using fp_type = typename Factory::fp_type;

  std::stringstream ss(circuit_string);
  Circuit<Operation<fp_type>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 2);
  EXPECT_EQ(circuit.ops.size(), 23);

  using Fuser = BasicGateFuser<IO>;
  using HybridSimulator = HybridSimulator<IO, For>;

  std::vector<unsigned> parts = {0, 1};

  typename HybridSimulator::HybridData<fp_type> hd;
  EXPECT_TRUE(HybridSimulator::SplitLattice(parts, circuit.ops, hd));

  EXPECT_EQ(hd.ops0.size(), 15);
  EXPECT_EQ(hd.ops1.size(), 15);
  EXPECT_EQ(hd.gatexs.size(), 7);
  EXPECT_EQ(hd.qubit_map.size(), 2);
  EXPECT_EQ(hd.num_qubits0, 1);
  EXPECT_EQ(hd.num_qubits1, 1);
  EXPECT_EQ(hd.num_gatexs, 7);

  typename HybridSimulator::Parameter<Fuser::Parameter> param;
  param.prefix = 1;
  param.num_prefix_gatexs = 0;
  param.num_root_gatexs = 0;
  param.num_threads = 1;
  param.verbosity = 0;

  auto fops0 = Fuser::FuseGates(param, hd.num_qubits0, hd.ops0);
  auto fops1 = Fuser::FuseGates(param, hd.num_qubits1, hd.ops1);

  EXPECT_EQ(fops0.size(), 7);
  EXPECT_EQ(fops1.size(), 7);

  std::vector<uint64_t> bitstrings;
  bitstrings.reserve(4);
  for (std::size_t i = 0; i < 4; ++i) {
    bitstrings.push_back(i);
  }

  std::vector<std::complex<fp_type>> results(4, 0);

  std::complex<typename Factory::fp_type> zero(0, 0);

  EXPECT_TRUE(HybridSimulator(1).Run(
      param, factory, hd, parts, fops0, fops1, bitstrings, results));

  EXPECT_NEAR(std::real(results[0]), -0.16006945, 1e-6);
  EXPECT_NEAR(std::imag(results[0]), -0.04964612, 1e-6);
  EXPECT_NEAR(std::real(results[1]), 0.22667059, 1e-6);
  EXPECT_NEAR(std::imag(results[1]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::real(results[2]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::imag(results[2]), 0.56567556, 1e-6);
  EXPECT_NEAR(std::real(results[3]), 0.28935891, 1e-6);
  EXPECT_NEAR(std::imag(results[3]), 0.71751291, 1e-6);

  std::fill(results.begin(), results.end(), zero);
  param.num_prefix_gatexs = 0;
  param.num_root_gatexs = 1;

  EXPECT_TRUE(HybridSimulator(1).Run(
      param, factory, hd, parts, fops0, fops1, bitstrings, results));

  EXPECT_NEAR(std::real(results[0]), -0.16006945, 1e-6);
  EXPECT_NEAR(std::imag(results[0]), -0.04964612, 1e-6);
  EXPECT_NEAR(std::real(results[1]), 0.22667059, 1e-6);
  EXPECT_NEAR(std::imag(results[1]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::real(results[2]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::imag(results[2]), 0.56567556, 1e-6);
  EXPECT_NEAR(std::real(results[3]), 0.28935891, 1e-6);
  EXPECT_NEAR(std::imag(results[3]), 0.71751291, 1e-6);

  std::fill(results.begin(), results.end(), zero);
  param.num_prefix_gatexs = 0;
  param.num_root_gatexs = 2;

  EXPECT_TRUE(HybridSimulator(1).Run(
      param, factory, hd, parts, fops0, fops1, bitstrings, results));

  EXPECT_NEAR(std::real(results[0]), -0.16006945, 1e-6);
  EXPECT_NEAR(std::imag(results[0]), -0.04964612, 1e-6);
  EXPECT_NEAR(std::real(results[1]), 0.22667059, 1e-6);
  EXPECT_NEAR(std::imag(results[1]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::real(results[2]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::imag(results[2]), 0.56567556, 1e-6);
  EXPECT_NEAR(std::real(results[3]), 0.28935891, 1e-6);
  EXPECT_NEAR(std::imag(results[3]), 0.71751291, 1e-6);

  std::fill(results.begin(), results.end(), zero);
  param.num_prefix_gatexs = 0;
  param.num_root_gatexs = 5;

  EXPECT_TRUE(HybridSimulator(1).Run(
      param, factory, hd, parts, fops0, fops1, bitstrings, results));

  EXPECT_NEAR(std::real(results[0]), -0.16006945, 1e-6);
  EXPECT_NEAR(std::imag(results[0]), -0.04964612, 1e-6);
  EXPECT_NEAR(std::real(results[1]), 0.22667059, 1e-6);
  EXPECT_NEAR(std::imag(results[1]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::real(results[2]), -0.03155057, 1e-6);
  EXPECT_NEAR(std::imag(results[2]), 0.56567556, 1e-6);
  EXPECT_NEAR(std::real(results[3]), 0.28935891, 1e-6);
  EXPECT_NEAR(std::imag(results[3]), 0.71751291, 1e-6);
}

template <typename Factory>
void TestHybrid4(const Factory& factory) {
  constexpr char provider[] = "hybrid_test";
  constexpr char circuit_string[] =
R"(4
0 h 0
0 h 1
0 h 2
0 h 3
1 t 0
1 t 1
1 t 2
1 t 3
2 cz 0 1
2 cz 2 3
3 x_1_2 0
3 y_1_2 1
3 t 2
3 x_1_2 3
4 cz 1 2
5 t 0
5 x_1_2 1
5 y_1_2 2
5 t 3
6 cz 0 1
6 cz 2 3
7 y_1_2 0
7 t 1
7 t 2
7 x_1_2 3
8 cp 1 2 0.7
9 t 0
9 x_1_2 1
9 y_1_2 2
9 x_1_2 3
10 cz 0 1
10 cz 2 3
11 t 0
11 y_1_2 1
11 y_1_2 2
11 t 3
12 is 1 2
13 x_1_2 0
13 t 1
13 x_1_2 2
13 t 3
14 cz 0 1
14 cz 2 3
15 t 0
15 y_1_2 1
15 x_1_2 2
15 y_1_2 3
16 cnot 1 2
17 t 0
17 x_1_2 1
17 y_1_2 2
17 x_1_2 3
18 cz 0 1
18 cz 2 3
19 x_1_2 0
19 t 1
19 t 2
19 y_1_2 3
20 fs 1 2 0.9 0.5
21 h 0
21 h 1
21 h 2
21 h 3
)";

  using fp_type = typename Factory::fp_type;

  std::stringstream ss(circuit_string);
  Circuit<Operation<fp_type>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.ops.size(), 63);

  using Fuser = BasicGateFuser<IO>;
  using HybridSimulator = HybridSimulator<IO, For>;

  std::vector<unsigned> parts = {0, 0, 1, 1};

  typename HybridSimulator::HybridData<fp_type> hd;
  EXPECT_TRUE(HybridSimulator::SplitLattice(parts, circuit.ops, hd));

  EXPECT_EQ(hd.ops0.size(), 34);
  EXPECT_EQ(hd.ops1.size(), 34);
  EXPECT_EQ(hd.gatexs.size(), 5);
  EXPECT_EQ(hd.qubit_map.size(), 4);
  EXPECT_EQ(hd.num_qubits0, 2);
  EXPECT_EQ(hd.num_qubits1, 2);
  EXPECT_EQ(hd.num_gatexs, 5);

  typename HybridSimulator::Parameter<Fuser::Parameter> param;
  param.prefix = 1;
  param.num_prefix_gatexs = 2;
  param.num_root_gatexs = 1;
  param.num_threads = 1;
  param.verbosity = 0;

  auto fops0 = Fuser::FuseGates(param, hd.num_qubits0, hd.ops0);
  auto fops1 = Fuser::FuseGates(param, hd.num_qubits1, hd.ops1);

  EXPECT_EQ(fops0.size(), 10);
  EXPECT_EQ(fops1.size(), 10);

  std::vector<uint64_t> bitstrings;
  bitstrings.reserve(8);
  for (std::size_t i = 0; i < 8; ++i) {
    bitstrings.push_back(i);
  }

  std::vector<std::complex<typename Factory::fp_type>> results(8, 0);

  EXPECT_TRUE(HybridSimulator(1).Run(
      param, factory, hd, parts, fops0, fops1, bitstrings, results));

  EXPECT_NEAR(std::real(results[0]), -0.02852439, 1e-6);
  EXPECT_NEAR(std::imag(results[0]), -0.05243438, 1e-6);
  EXPECT_NEAR(std::real(results[1]), -0.09453446, 1e-6);
  EXPECT_NEAR(std::imag(results[1]), 0.03033427, 1e-6);
  EXPECT_NEAR(std::real(results[2]), 0.08091709, 1e-6);
  EXPECT_NEAR(std::imag(results[2]), 0.13259856, 1e-6);
  EXPECT_NEAR(std::real(results[3]), 0.13379499, 1e-6);
  EXPECT_NEAR(std::imag(results[3]), 0.13946741, 1e-6);
}

}  // namespace qsim

#endif  // HYBRID_TESTFIXTURE_H_
