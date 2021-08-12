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
#include <complex>
#include <cstdint>
#include <sstream>

#include "gates_cirq_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/run_qsimh.h"
#include "../lib/simmux.h"

namespace qsim {

constexpr char provider[] = "run_qsimh_test";

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

struct Factory {
  using Simulator = qsim::Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using fp_type = Simulator::fp_type;

  static StateSpace CreateStateSpace() {
    return StateSpace(1);
  }

  static Simulator CreateSimulator() {
    return Simulator(1);
  }
};

TEST(RunQSimHTest, QSimHRunner) {
  std::stringstream ss(circuit_string);
  Circuit<GateQSim<float>> circuit;

  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, provider, ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 4);
  EXPECT_EQ(circuit.gates.size(), 63);

  using HybridSimulator = HybridSimulator<IO, GateQSim<float>, BasicGateFuser,
                                          For>;
  using Runner = QSimHRunner<IO, HybridSimulator>;

  Runner::Parameter param;
  param.prefix = 0;
  param.num_prefix_gatexs = 0;
  param.num_root_gatexs = 5;
  param.num_threads = 1;
  param.verbosity = 0;

  std::vector<uint64_t> bitstrings;
  bitstrings.reserve(8);

  for (std::size_t i = 0; i < 8; ++i) {
    bitstrings.push_back(i);
  }

  Factory factory;

  {
    std::vector<std::complex<Factory::fp_type>> results(8, 0);
    std::vector<unsigned> parts = {0, 0, 1, 1};

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    EXPECT_NEAR(std::real(results[0]), -0.08102149, 1e-6);
    EXPECT_NEAR(std::imag(results[0]), 0.08956901, 1e-6);
    EXPECT_NEAR(std::real(results[1]), 0.11983117, 1e-6);
    EXPECT_NEAR(std::imag(results[1]), 0.14673762, 1e-6);
    EXPECT_NEAR(std::real(results[2]), 0.14810989, 1e-6);
    EXPECT_NEAR(std::imag(results[2]), 0.31299597, 1e-6);
    EXPECT_NEAR(std::real(results[3]), 0.12226092, 1e-6);
    EXPECT_NEAR(std::imag(results[3]), 0.26690706, 1e-6);
  }

  {
    std::vector<std::complex<Factory::fp_type>> results(8, 0);
    std::vector<unsigned> parts = {1, 1, 0, 0};

    param.num_root_gatexs = 3;

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    EXPECT_NEAR(std::real(results[0]), -0.08102149, 1e-6);
    EXPECT_NEAR(std::imag(results[0]), 0.08956903, 1e-6);
    EXPECT_NEAR(std::real(results[1]), 0.11983119, 1e-6);
    EXPECT_NEAR(std::imag(results[1]), 0.14673763, 1e-6);
    EXPECT_NEAR(std::real(results[2]), 0.14810986, 1e-6);
    EXPECT_NEAR(std::imag(results[2]), 0.31299597, 1e-6);
    EXPECT_NEAR(std::real(results[3]), 0.12226093, 1e-6);
    EXPECT_NEAR(std::imag(results[3]), 0.26690706, 1e-6);
  }
}

TEST(RunQSimHTest, CirqGates) {
  auto circuit = CirqCircuit1::GetCircuit<float>(false);
  const auto& expected_results = CirqCircuit1::expected_results0;

  using HybridSimulator = HybridSimulator<IO, Cirq::GateCirq<float>,
                                          BasicGateFuser, For>;
  using Runner = QSimHRunner<IO, HybridSimulator>;

  Runner::Parameter param;
  param.prefix = 0;
  param.num_prefix_gatexs = 0;
  param.num_root_gatexs = 0;
  param.num_threads = 1;
  param.verbosity = 0;

  uint64_t num_bitstrings = expected_results.size();

  std::vector<uint64_t> bitstrings;
  bitstrings.reserve(num_bitstrings);

  for (std::size_t i = 0; i < num_bitstrings; ++i) {
    bitstrings.push_back(i);
  }

  Factory factory;

  {
    std::vector<std::complex<Factory::fp_type>> results(num_bitstrings, 0);
    std::vector<unsigned> parts = {1, 1, 0, 0};

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    for (uint64_t i = 0; i < num_bitstrings; ++i) {
      EXPECT_NEAR(std::real(results[i]), std::real(expected_results[i]), 2e-6);
      EXPECT_NEAR(std::imag(results[i]), std::imag(expected_results[i]), 2e-6);
    }
  }

  {
    std::vector<std::complex<Factory::fp_type>> results(num_bitstrings, 0);
    std::vector<unsigned> parts = {1, 0, 1, 0};

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    for (uint64_t i = 0; i < num_bitstrings; ++i) {
      EXPECT_NEAR(std::real(results[i]), std::real(expected_results[i]), 2e-6);
      EXPECT_NEAR(std::imag(results[i]), std::imag(expected_results[i]), 2e-6);
    }
  }

  {
    std::vector<std::complex<Factory::fp_type>> results(num_bitstrings, 0);
    std::vector<unsigned> parts = {1, 0, 0, 0};

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    for (uint64_t i = 0; i < num_bitstrings; ++i) {
      EXPECT_NEAR(std::real(results[i]), std::real(expected_results[i]), 2e-6);
      EXPECT_NEAR(std::imag(results[i]), std::imag(expected_results[i]), 2e-6);
    }
  }

  {
    std::vector<std::complex<Factory::fp_type>> results(num_bitstrings, 0);
    std::vector<unsigned> parts = {0, 1, 0, 0};

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    for (uint64_t i = 0; i < num_bitstrings; ++i) {
      EXPECT_NEAR(std::real(results[i]), std::real(expected_results[i]), 2e-6);
      EXPECT_NEAR(std::imag(results[i]), std::imag(expected_results[i]), 2e-6);
    }
  }

  {
    std::vector<std::complex<Factory::fp_type>> results(num_bitstrings, 0);
    std::vector<unsigned> parts = {0, 0, 1, 0};

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    for (uint64_t i = 0; i < num_bitstrings; ++i) {
      EXPECT_NEAR(std::real(results[i]), std::real(expected_results[i]), 2e-6);
      EXPECT_NEAR(std::imag(results[i]), std::imag(expected_results[i]), 2e-6);
    }
  }

  {
    std::vector<std::complex<Factory::fp_type>> results(num_bitstrings, 0);
    std::vector<unsigned> parts = {0, 0, 0, 1};

    EXPECT_TRUE(Runner::Run(
        param, factory, circuit, parts, bitstrings, results));

    for (uint64_t i = 0; i < num_bitstrings; ++i) {
      EXPECT_NEAR(std::real(results[i]), std::real(expected_results[i]), 2e-6);
      EXPECT_NEAR(std::imag(results[i]), std::imag(expected_results[i]), 2e-6);
    }
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
