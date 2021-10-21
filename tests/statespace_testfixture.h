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

#ifndef STATESPACE_TESTFIXTURE_H_
#define STATESPACE_TESTFIXTURE_H_

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/run_qsim.h"

namespace qsim {

constexpr char provider[] = "statespace_test";

constexpr char circuit_string[] =
R"(20
0 h 0
0 h 1
0 h 2
0 h 3
0 h 4
0 h 5
0 h 6
0 h 7
0 h 8
0 h 9
0 h 10
0 h 11
0 h 12
0 h 13
0 h 14
0 h 15
0 h 16
0 h 17
0 h 18
0 h 19
1 cz 0 1
1 cz 7 8
1 cz 10 11
1 cz 17 18
2 cz 2 3
2 cz 5 6
2 cz 12 13
2 cz 15 16
2 t 0
2 t 1
2 t 7
2 t 8
2 t 10
2 t 11
2 t 17
2 t 18
3 cz 5 10
3 cz 7 12
3 cz 9 14
3 t 2
3 t 3
3 t 6
3 t 13
3 t 15
3 t 16
4 cz 6 11
4 cz 8 13
4 t 5
4 x_1_2 7
4 t 9
4 y_1_2 10
4 t 12
4 t 14
5 cz 1 2
5 cz 8 9
5 cz 11 12
5 cz 18 19
5 y_1_2 6
5 x_1_2 13
6 cz 3 4
6 cz 6 7
6 cz 13 14
6 cz 16 17
6 y_1_2 1
6 y_1_2 2
6 y_1_2 8
6 y_1_2 9
6 y_1_2 11
6 y_1_2 12
6 y_1_2 18
6 t 19
7 cz 0 5
7 cz 2 7
7 cz 4 9
7 cz 11 16
7 cz 13 18
7 x_1_2 3
7 x_1_2 6
7 y_1_2 14
7 x_1_2 17
8 cz 1 6
8 cz 3 8
8 cz 10 15
8 cz 12 17
8 cz 14 19
8 x_1_2 0
8 x_1_2 2
8 t 4
8 x_1_2 5
8 y_1_2 7
8 t 9
8 x_1_2 11
8 t 13
8 y_1_2 16
8 x_1_2 18
9 cz 0 1
9 cz 7 8
9 cz 10 11
9 cz 17 18
9 y_1_2 3
9 t 6
9 t 12
9 t 14
9 y_1_2 15
9 x_1_2 19
10 cz 2 3
10 cz 5 6
10 cz 12 13
10 cz 15 16
10 t 0
10 x_1_2 1
10 t 7
10 x_1_2 8
10 t 10
10 t 11
10 y_1_2 17
10 t 18
11 cz 5 10
11 cz 7 12
11 cz 9 14
11 t 2
11 x_1_2 3
11 x_1_2 6
11 y_1_2 13
11 x_1_2 15
11 t 16
12 cz 6 11
12 cz 8 13
12 t 5
12 y_1_2 7
12 y_1_2 9
12 y_1_2 10
12 x_1_2 12
12 y_1_2 14
13 cz 1 2
13 cz 8 9
13 cz 11 12
13 cz 18 19
13 y_1_2 6
13 t 13
14 cz 3 4
14 cz 6 7
14 cz 13 14
14 cz 16 17
14 t 1
14 y_1_2 2
14 t 8
14 x_1_2 9
14 y_1_2 11
14 y_1_2 12
14 x_1_2 18
14 t 19
15 cz 0 5
15 cz 2 7
15 cz 4 9
15 cz 11 16
15 cz 13 18
15 t 3
15 x_1_2 6
15 t 14
15 x_1_2 17
16 cz 1 6
16 cz 3 8
16 cz 10 15
16 cz 12 17
16 cz 14 19
16 y_1_2 0
16 x_1_2 2
16 x_1_2 4
16 x_1_2 5
16 x_1_2 7
16 y_1_2 9
16 t 11
16 y_1_2 13
16 x_1_2 16
16 y_1_2 18
17 cz 0 1
17 cz 7 8
17 cz 10 11
17 cz 17 18
17 x_1_2 3
17 t 6
17 t 12
17 x_1_2 14
17 t 15
17 x_1_2 19
18 cz 2 3
18 cz 5 6
18 cz 12 13
18 cz 15 16
18 x_1_2 0
18 y_1_2 1
18 y_1_2 7
18 y_1_2 8
18 t 10
18 y_1_2 11
18 t 17
18 t 18
19 cz 5 10
19 cz 7 12
19 cz 9 14
19 t 2
19 y_1_2 3
19 y_1_2 6
19 t 13
19 x_1_2 15
19 y_1_2 16
20 cz 6 11
20 cz 8 13
20 t 5
20 x_1_2 7
20 x_1_2 9
20 y_1_2 10
20 y_1_2 12
20 y_1_2 14
21 cz 1 2
21 cz 8 9
21 cz 11 12
21 cz 18 19
21 t 6
21 x_1_2 13
22 cz 3 4
22 cz 6 7
22 cz 13 14
22 cz 16 17
22 x_1_2 1
22 y_1_2 2
22 x_1_2 8
22 y_1_2 9
22 x_1_2 11
22 t 12
22 y_1_2 18
22 y_1_2 19
23 cz 0 5
23 cz 2 7
23 cz 4 9
23 cz 11 16
23 cz 13 18
23 t 3
23 x_1_2 6
23 x_1_2 14
23 x_1_2 17
24 cz 1 6
24 cz 3 8
24 cz 10 15
24 cz 12 17
24 cz 14 19
24 y_1_2 0
24 x_1_2 2
24 t 4
24 x_1_2 5
24 t 7
24 x_1_2 9
24 t 11
24 t 13
24 t 16
24 t 18
25 cz 0 1
25 cz 7 8
25 cz 10 11
25 cz 17 18
25 y_1_2 3
25 y_1_2 6
25 y_1_2 12
25 t 14
25 t 15
25 t 19
26 cz 2 3
26 cz 5 6
26 cz 12 13
26 cz 15 16
26 x_1_2 0
26 t 1
26 y_1_2 7
26 y_1_2 8
26 x_1_2 10
26 y_1_2 11
26 y_1_2 17
26 x_1_2 18
27 cz 5 10
27 cz 7 12
27 cz 9 14
27 y_1_2 2
27 x_1_2 3
27 t 6
27 y_1_2 13
27 x_1_2 15
27 x_1_2 16
28 cz 6 11
28 cz 8 13
28 t 5
28 x_1_2 7
28 t 9
28 t 10
28 t 12
28 y_1_2 14
29 cz 1 2
29 cz 8 9
29 cz 11 12
29 cz 18 19
29 x_1_2 6
29 x_1_2 13
30 cz 3 4
30 cz 6 7
30 cz 13 14
30 cz 16 17
30 x_1_2 1
30 t 2
30 x_1_2 8
30 y_1_2 9
30 t 11
30 y_1_2 12
30 y_1_2 18
30 y_1_2 19
)";

template <typename Factory>
void TestAdd(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  constexpr unsigned num_qubits = 2;
  StateSpace state_space = factory.CreateStateSpace();

  State state1 = state_space.Create(num_qubits);
  state_space.SetAllZeros(state1);
  state_space.SetAmpl(state1, 0, 1, 2);
  state_space.SetAmpl(state1, 1, 3, 4);
  state_space.SetAmpl(state1, 2, 5, 6);
  state_space.SetAmpl(state1, 3, 7, 8);

  State state2 = state_space.Create(num_qubits);
  state_space.SetAllZeros(state2);
  state_space.SetAmpl(state2, 0, 1, 2);
  state_space.SetAmpl(state2, 1, 3, 4);
  state_space.SetAmpl(state2, 2, 5, 6);
  state_space.SetAmpl(state2, 3, 7, 8);

  EXPECT_TRUE(state_space.Add(state1, state2));
  EXPECT_EQ(state_space.GetAmpl(state2, 0), std::complex<fp_type>(2, 4));
  EXPECT_EQ(state_space.GetAmpl(state2, 1), std::complex<fp_type>(6, 8));
  EXPECT_EQ(state_space.GetAmpl(state2, 2), std::complex<fp_type>(10, 12));
  EXPECT_EQ(state_space.GetAmpl(state2, 3), std::complex<fp_type>(14, 16));

  EXPECT_EQ(state_space.GetAmpl(state1, 0), std::complex<fp_type>(1, 2));
  EXPECT_EQ(state_space.GetAmpl(state1, 1), std::complex<fp_type>(3, 4));
  EXPECT_EQ(state_space.GetAmpl(state1, 2), std::complex<fp_type>(5, 6));
  EXPECT_EQ(state_space.GetAmpl(state1, 3), std::complex<fp_type>(7, 8));
}

template <typename Factory>
void TestNormSmall(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;

  StateSpace state_space = factory.CreateStateSpace();

  for (unsigned num_qubits : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    State state = state_space.Create(num_qubits);
    state_space.SetStateZero(state);
    EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);
    state_space.SetStateUniform(state);
    EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);
  }
}

template <typename Factory>
void TestNormAndInnerProductSmall(const Factory& factory) {
  constexpr unsigned num_qubits = 2;
  constexpr uint64_t size = uint64_t{1} << num_qubits;

  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();

  State state1 = state_space.Create(num_qubits);
  State state2 = state_space.Create(num_qubits);

  EXPECT_FALSE(state_space.IsNull(state1));
  EXPECT_FALSE(state_space.IsNull(state2));

  state_space.SetAllZeros(state1);
  state_space.SetAllZeros(state2);

  std::array<fp_type, size> values1 = {0.25, 0.3, 0.35, 0.1};
  std::array<fp_type, size> values2 = {0.4, 0.15, 0.2, 0.25};

  std::complex<double> inner_product0 = 0;

  for (uint64_t i = 0; i < size; ++i) {
    fp_type c = std::cos(i);
    fp_type s = std::sin(i);
    fp_type r1 = std::sqrt(values1[i]);
    fp_type r2 = std::sqrt(values2[i]);
    std::complex<fp_type> cvalue1 = {r1 * c, -r1 * s};
    std::complex<fp_type> cvalue2 = {r2 * c, r2 * s};

    state_space.SetAmpl(state1, i, cvalue1);
    state_space.SetAmpl(state2, i, cvalue2);

    inner_product0 += std::conj(cvalue1) * cvalue2;
  }

  EXPECT_NEAR(state_space.Norm(state1), 1, 1e-6);
  EXPECT_NEAR(state_space.Norm(state2), 1, 1e-6);

  auto inner_product = state_space.InnerProduct(state1, state2);
  EXPECT_FALSE(std::isnan(std::real(inner_product)));
  EXPECT_NEAR(std::real(inner_product), std::real(inner_product0), 1e-6);
  EXPECT_NEAR(std::imag(inner_product), std::imag(inner_product0), 1e-6);

  auto real_inner_product = state_space.RealInnerProduct(state1, state2);
  EXPECT_FALSE(std::isnan(real_inner_product));
  EXPECT_NEAR(real_inner_product, std::real(inner_product0), 1e-6);

  state_space.Multiply(std::sqrt(1.3), state1);
  EXPECT_NEAR(state_space.Norm(state1), 1.3, 1e-6);
  state_space.Multiply(std::sqrt(0.8), state2);
  EXPECT_NEAR(state_space.Norm(state2), 0.8, 1e-6);
}

template <typename Factory>
void TestNormAndInnerProduct(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, GateQSim<fp_type>>, Factory>;

  unsigned depth = 8;

  std::stringstream ss(circuit_string);
  Circuit<GateQSim<fp_type>> circuit;
  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(depth, provider, ss, circuit));
  circuit.gates.emplace_back(GateT<fp_type>::Create(depth + 1, 0));

  StateSpace state_space = factory.CreateStateSpace();
  State state0 = state_space.Create(circuit.num_qubits);

  EXPECT_FALSE(state_space.IsNull(state0));

  auto measure = [&state0](unsigned k,
                           const StateSpace& state_space, const State& state) {
    if (k == 0) {
      EXPECT_NEAR(state_space.Norm(state), 1, 1e-5);
      EXPECT_TRUE(state_space.Copy(state, state0));
    } else if (k == 1) {
      auto inner_product = state_space.InnerProduct(state0, state);
      EXPECT_FALSE(std::isnan(std::real(inner_product)));
      EXPECT_NEAR(std::real(inner_product), 0.5 + 0.5 / std::sqrt(2), 1e-5);
      EXPECT_NEAR(std::imag(inner_product), 0.5 / std::sqrt(2), 1e-5);

      auto real_inner_product = state_space.RealInnerProduct(state0, state);
      EXPECT_FALSE(std::isnan(real_inner_product));
      EXPECT_NEAR(real_inner_product, 0.5 + 0.5 / std::sqrt(2), 1e-5);
    }
  };

  typename Runner::Parameter param;
  param.seed = 1;
  param.verbosity = 0;

  std::vector<unsigned> times{depth, depth + 1};
  EXPECT_TRUE(Runner::Run(param, factory, times, circuit, measure));

  state_space.Multiply(std::sqrt(1.2), state0);
  EXPECT_NEAR(state_space.Norm(state0), 1.2, 1e-5);
}

template <typename Factory>
void TestSamplingSmall(const Factory& factory) {
  uint64_t num_samples = 2000000;
  constexpr unsigned num_qubits = 6;
  constexpr uint64_t size = uint64_t{1} << num_qubits;

  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;

  StateSpace state_space = factory.CreateStateSpace();
  State state = state_space.Create(num_qubits);

  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetAllZeros(state);

  std::vector<float> ps = {
    0.0046, 0.0100, 0.0136, 0.0239, 0.0310, 0.0263, 0.0054, 0.0028,
    0.0129, 0.0128, 0.0319, 0.0023, 0.0281, 0.0185, 0.0284, 0.0175,
    0.0148, 0.0307, 0.0017, 0.0319, 0.0185, 0.0304, 0.0133, 0.0229,
    0.0190, 0.0213, 0.0197, 0.0028, 0.0288, 0.0084, 0.0052, 0.0147,
    0.0164, 0.0084, 0.0154, 0.0093, 0.0215, 0.0059, 0.0115, 0.0039,
    0.0111, 0.0101, 0.0314, 0.0003, 0.0173, 0.0075, 0.0116, 0.0091,
    0.0289, 0.0005, 0.0027, 0.0293, 0.0063, 0.0181, 0.0043, 0.0063,
    0.0169, 0.0101, 0.0295, 0.0227, 0.0275, 0.0218, 0.0131, 0.0172,
  };

  for (uint64_t i = 0; i < size; ++i) {
    auto r = std::sqrt(ps[i]);
    state_space.SetAmpl(state, i, r * std::cos(i), r * std::sin(i));
  }

  auto samples = state_space.Sample(state, num_samples, 1);

  EXPECT_EQ(samples.size(), num_samples);
  std::vector<double> bins(size, 0);

  for (auto sample : samples) {
    ASSERT_LT(sample, size);
    bins[sample] += 1;
  }

  for (uint64_t i = 0; i < size; ++i) {
    EXPECT_NEAR(bins[i] / num_samples, ps[i], 4e-4);
  }
}

template <typename Factory>
void TestSamplingCrossEntropyDifference(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, GateQSim<fp_type>>, Factory>;

  unsigned depth = 30;
  uint64_t num_samples = 2000000;

  std::stringstream ss(circuit_string);
  Circuit<GateQSim<fp_type>> circuit;
  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(depth, provider, ss, circuit));

  StateSpace state_space = factory.CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  typename Runner::Parameter param;
  param.seed = 1;
  param.verbosity = 0;

  EXPECT_TRUE(Runner::Run(param, factory, circuit, state));

  auto bitstrings = state_space.Sample(state, num_samples, 1);
  EXPECT_EQ(bitstrings.size(), num_samples);

  std::vector<fp_type> vec(state_space.MinSize(circuit.num_qubits));

  state_space.InternalToNormalOrder(state);
  state_space.Copy(state, vec.data());

  double sum = 0;
  for (uint64_t i = 0; i < num_samples; ++i) {
    auto k = bitstrings[i];
    auto re = vec[2 * k];
    auto im = vec[2 * k + 1];
    double p = re * re + im * im;
    sum += std::log(p);
  }

  double gamma = 0.5772156649;
  double ced = circuit.num_qubits * std::log(2) + gamma + sum / num_samples;

  EXPECT_NEAR(ced, 1.0, 2e-3);
}

template <typename Factory>
void TestOrdering(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();

  std::vector<fp_type> vec;
  vec.reserve(state_space.MinSize(9));

  for (unsigned num_qubits : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
    uint64_t size = uint64_t{1} << num_qubits;

    State state = state_space.Create(num_qubits);
    state_space.SetAllZeros(state);

    for (uint64_t i = 0; i < size; ++i) {
      state_space.SetAmpl(state, i, std::complex<fp_type>(i + 1, size + i + 1));
    }

    vec.resize(state_space.MinSize(num_qubits));

    state_space.InternalToNormalOrder(state);
    state_space.Copy(state, vec.data());

    for (uint64_t i = 0; i < size; ++i) {
      EXPECT_NEAR(vec[2 * i], fp_type(i + 1), 1e-8);
      EXPECT_NEAR(vec[2 * i + 1], fp_type(size + i + 1), 1e-8);
    }

    state_space.NormalToInternalOrder(state);

    for (uint64_t i = 0; i < size; ++i) {
      auto a = state_space.GetAmpl(state, i);
      EXPECT_NEAR(std::real(a), fp_type(i + 1), 1e-8);
      EXPECT_NEAR(std::imag(a), fp_type(size + i + 1), 1e-8);
    }
  }
}

template <typename Factory, typename RGen>
void MeasureSmall(const Factory& factory, unsigned num_measurements,
                  double max_error, unsigned num_qubits,
                  const std::vector<unsigned>& qubits_to_measure,
                  const std::vector<double>& ps, RGen& rgen) {
  uint64_t size = uint64_t{1} << num_qubits;

  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();

  State state0 = state_space.Create(num_qubits);
  State state = state_space.Create(num_qubits);

  EXPECT_FALSE(state_space.IsNull(state0));
  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state0);

  std::vector<double> bins(size, 0);

  std::vector<fp_type> vec(state_space.MinSize(num_qubits));

  for (unsigned i = 0; i < size; ++i) {
    fp_type r = std::sqrt(ps[i]);
    vec[2 * i] = r * std::cos(i);
    vec[2 * i + 1] = r * std::sin(i);
  }

  state_space.Copy(vec.data(), state0);
  state_space.NormalToInternalOrder(state0);

  std::vector<unsigned> measured_bits;

  for (unsigned m = 0; m < num_measurements; ++m) {
    state_space.Copy(state0, state);

    auto result = state_space.Measure(qubits_to_measure, rgen, state);
    ASSERT_TRUE(result.valid);

    if (m % 16 == 0) {
      ASSERT_NEAR(state_space.Norm(state), 1, 1e-6);
    }

    uint64_t bin = 0;
    for (std::size_t k = 0; k < qubits_to_measure.size(); ++k) {
      bin |= uint64_t{result.bitstring[k]} << qubits_to_measure[k];
    }

    EXPECT_EQ(bin, result.bits);

    bins[bin] += 1;
  }

  uint64_t mask = 0;
  for (std::size_t k = 0; k < qubits_to_measure.size(); ++k) {
    mask |= uint64_t{1} << qubits_to_measure[k];
  }

  std::vector<double> expected_ps(size, 0);

  for (uint64_t i = 0; i < size; ++i) {
    expected_ps[i & mask] += ps[i];
  }

  for (uint64_t i = 0; i < size; ++i) {
    if (expected_ps[i] == 0) {
      EXPECT_EQ(bins[i], 0);
    } else {
      double p = bins[i] / num_measurements;
      double rel_error = (p - expected_ps[i]) / expected_ps[i];
      EXPECT_LE(rel_error, max_error);
    }
  }
}

template <typename Factory>
void TestMeasurementSmall(const Factory& factory, bool cuda = false) {
  unsigned num_measurements = cuda ? 50000 : 200000;
  double max_error = cuda ? 0.06 : 0.02;

  std::mt19937 rgen(1);

  std::vector<double> ps1 = {0.37, 0.63};
  MeasureSmall(factory, num_measurements, max_error, 1, {0}, ps1, rgen);

  std::vector<double> ps2 = {0.22, 0.42, 0.15, 0.21};
  MeasureSmall(factory, num_measurements, max_error, 2, {1}, ps2, rgen);

  std::vector<double> ps3 = {0.1, 0.2, 0.13, 0.12, 0.18, 0.15, 0.07, 0.05};
  MeasureSmall(factory, num_measurements, max_error, 3, {2}, ps3, rgen);
  MeasureSmall(factory, num_measurements, max_error, 3, {0, 1, 2}, ps3, rgen);

  std::vector<double> ps4 = {
    0.06, 0.10, 0.07, 0.06, 0.09, 0.08, 0.03, 0.03,
    0.03, 0.07, 0.11, 0.04, 0.05, 0.06, 0.07, 0.05
  };
  MeasureSmall(factory, num_measurements, max_error, 4, {0, 3}, ps4, rgen);

  std::vector<double> ps5 = {
    0.041, 0.043, 0.028, 0.042, 0.002, 0.008, 0.039, 0.020,
    0.017, 0.030, 0.020, 0.048, 0.020, 0.044, 0.032, 0.048,
    0.025, 0.050, 0.030, 0.001, 0.039, 0.045, 0.005, 0.051,
    0.030, 0.039, 0.012, 0.049, 0.034, 0.029, 0.050, 0.029
  };
  MeasureSmall(factory, num_measurements, max_error, 5, {1, 3}, ps5, rgen);
  MeasureSmall(
      factory, num_measurements, max_error, 5, {1, 2, 3, 4}, ps5, rgen);

  std::vector<double> ps6 = {
    0.0046, 0.0100, 0.0136, 0.0239, 0.0310, 0.0263, 0.0054, 0.0028,
    0.0129, 0.0128, 0.0319, 0.0023, 0.0281, 0.0185, 0.0284, 0.0175,
    0.0148, 0.0307, 0.0017, 0.0319, 0.0185, 0.0304, 0.0133, 0.0229,
    0.0190, 0.0213, 0.0197, 0.0028, 0.0288, 0.0084, 0.0052, 0.0147,
    0.0164, 0.0084, 0.0154, 0.0093, 0.0215, 0.0059, 0.0115, 0.0039,
    0.0111, 0.0101, 0.0314, 0.0003, 0.0173, 0.0075, 0.0116, 0.0091,
    0.0289, 0.0005, 0.0027, 0.0293, 0.0063, 0.0181, 0.0043, 0.0063,
    0.0169, 0.0101, 0.0295, 0.0227, 0.0275, 0.0218, 0.0131, 0.0172,
  };
  MeasureSmall(factory, num_measurements, max_error, 6, {2, 4, 5}, ps6, rgen);
}

template <typename Factory>
void TestMeasurementLarge(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, GateQSim<fp_type>>, Factory>;

  unsigned depth = 20;

  std::stringstream ss(circuit_string);
  Circuit<GateQSim<fp_type>> circuit;
  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(depth, provider, ss, circuit));

  StateSpace state_space = factory.CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  EXPECT_FALSE(state_space.IsNull(state));

  state_space.SetStateZero(state);

  typename Runner::Parameter param;
  param.seed = 1;
  param.verbosity = 0;

  EXPECT_TRUE(Runner::Run(param, factory, circuit, state));

  std::mt19937 rgen(1);
  auto result = state_space.Measure({0, 4}, rgen, state);

  EXPECT_TRUE(result.valid);
  EXPECT_EQ(result.mask, 17);
  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  auto ampl0 = state_space.GetAmpl(state, 0);
  EXPECT_NEAR(std::real(ampl0), -0.00208748, 1e-6);
  EXPECT_NEAR(std::imag(ampl0), -0.00153427, 1e-6);

  auto ampl2 = state_space.GetAmpl(state, 2);
  EXPECT_NEAR(std::real(ampl2), -0.00076403, 1e-6);
  EXPECT_NEAR(std::imag(ampl2), 0.00123912, 1e-6);

  auto ampl4 = state_space.GetAmpl(state, 4);
  EXPECT_NEAR(std::real(ampl4), -0.00349379, 1e-6);
  EXPECT_NEAR(std::imag(ampl4), 0.00110578, 1e-6);

  auto ampl6 = state_space.GetAmpl(state, 6);
  EXPECT_NEAR(std::real(ampl6), -0.00180432, 1e-6);
  EXPECT_NEAR(std::imag(ampl6), 0.00153727, 1e-6);

  for (uint64_t i = 0; i < 8; ++i) {
    auto ampl = state_space.GetAmpl(state, 2 * i + 1);
    EXPECT_EQ(std::real(ampl), 0);
    EXPECT_EQ(std::imag(ampl), 0);
  }

  for (uint64_t i = 16; i < 32; ++i) {
    auto ampl = state_space.GetAmpl(state, 2 * i + 1);
    EXPECT_EQ(std::real(ampl), 0);
    EXPECT_EQ(std::imag(ampl), 0);
  }
}

template <typename Factory>
void TestCollapse(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using MeasurementResult = typename StateSpace::MeasurementResult;

  MeasurementResult mr;
  StateSpace state_space = factory.CreateStateSpace();

  std::vector<fp_type> vec;
  vec.reserve(state_space.MinSize(6));

  for (unsigned num_qubits = 2; num_qubits <= 6; ++num_qubits) {
    State state = state_space.Create(num_qubits);

    vec.resize(state_space.MinSize(num_qubits));

    unsigned size = 1 << num_qubits;

    for (unsigned mask = 1; mask < size; ++mask) {
      for (unsigned bits = 0; bits < size; ++bits) {
        if ((mask | bits) != mask) continue;

        mr.mask = mask;
        mr.bits = bits;

        unsigned num_measured_qubits = 0;
        for (unsigned i = 0; i < num_qubits; ++i) {
          if (((mask >> i) & 1) == 1) {
            ++num_measured_qubits;
          }
        }

        fp_type r = 1.0 / std::sqrt(1 << (num_qubits - num_measured_qubits));

        state_space.SetStateUniform(state);
        state_space.Collapse(mr, state);

        EXPECT_NEAR(state_space.Norm(state), 1.0, 1e-6);

        state_space.InternalToNormalOrder(state);
        state_space.Copy(state, vec.data());

        for (unsigned i = 0; i < size; ++i) {
          if ((i & mask) != bits) {
            EXPECT_NEAR(vec[2 * i], 0, 1e-6);
            EXPECT_NEAR(vec[2 * i + 1], 0, 1e-6);
          } else {
            EXPECT_NEAR(vec[2 * i], r, 1e-6);
            EXPECT_NEAR(vec[2 * i + 1], 0, 1e-6);
          }
        }
      }
    }
  }
}

template <typename Factory>
void TestInvalidStateSize(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;

  std::mt19937 rgen(1);

  unsigned num_qubits1 = 3;
  unsigned num_qubits2 = 6;

  StateSpace state_space = factory.CreateStateSpace();

  State state1 = state_space.Create(num_qubits1);
  State state2 = state_space.Create(num_qubits2);

  EXPECT_FALSE(state_space.Copy(state1, state2));
  EXPECT_FALSE(state_space.Add(state1, state2));
  EXPECT_FALSE(
      !std::isnan(std::real(state_space.InnerProduct(state1, state2))));
  EXPECT_FALSE(!std::isnan(state_space.RealInnerProduct(state1, state2)));
}

template <typename Factory>
void TestBulkSetAmplitude(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();

  std::vector<fp_type> vec;
  vec.reserve(state_space.MinSize(6));

  for (unsigned num_qubits = 2; num_qubits <= 6; ++num_qubits) {
    State state = state_space.Create(num_qubits);
    state_space.SetAllZeros(state);

    vec.resize(state_space.MinSize(num_qubits));

    unsigned size = 1 << num_qubits;

    for (unsigned mask = 1; mask < size; ++mask) {
      for (unsigned bits = 0; bits < size; ++bits) {
        if ((mask | bits) != mask) continue;

        for (unsigned i = 0; i < size; ++i) {
          vec[2 * i] = 1;
          vec[2 * i + 1] = 1;
        }

        state_space.Copy(vec.data(), state);
        state_space.NormalToInternalOrder(state);

        state_space.BulkSetAmpl(state, mask, bits, 0, 0, false);

        state_space.InternalToNormalOrder(state);
        state_space.Copy(state, vec.data());

        for (unsigned i = 0; i < size; ++i) {
          if ((i & mask) == bits) {
            EXPECT_EQ(vec[2 * i], 0);
            EXPECT_EQ(vec[2 * i + 1], 0);
          } else {
            EXPECT_EQ(vec[2 * i], 1);
            EXPECT_EQ(vec[2 * i + 1], 1);
          }
        }
      }
    }
  }
}

template <typename Factory>
void TestBulkSetAmplitudeExclusion(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();

  std::vector<fp_type> vec;
  vec.reserve(state_space.MinSize(6));

  for (unsigned num_qubits = 2; num_qubits <= 6; ++num_qubits) {
    State state = state_space.Create(num_qubits);
    state_space.SetAllZeros(state);

    vec.resize(state_space.MinSize(num_qubits));

    unsigned size = 1 << num_qubits;

    for (unsigned mask = 1; mask < size; ++mask) {
      for (unsigned bits = 0; bits < size; ++bits) {
        if ((mask | bits) != mask) continue;

        for (unsigned i = 0; i < size; ++i) {
          vec[2 * i] = 1;
          vec[2 * i + 1] = 1;
        }

        state_space.Copy(vec.data(), state);
        state_space.NormalToInternalOrder(state);

        state_space.BulkSetAmpl(state, mask, bits, 0, 0, true);

        state_space.InternalToNormalOrder(state);
        state_space.Copy(state, vec.data());

        for (unsigned i = 0; i < size; ++i) {
          if ((i & mask) != bits) {
            EXPECT_EQ(vec[2 * i], 0);
            EXPECT_EQ(vec[2 * i + 1], 0);
          } else {
            EXPECT_EQ(vec[2 * i], 1);
            EXPECT_EQ(vec[2 * i + 1], 1);
          }
        }
      }
    }
  }
}

template <typename Factory>
void TestBulkSetAmplitudeDefault(const Factory& factory) {
  using StateSpace = typename Factory::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  unsigned num_qubits = 3;

  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(num_qubits);
  state_space.SetAllZeros(state);

  for (unsigned i = 0; i < 8; ++i) {
    state_space.SetAmpl(state, i, 1, 1);
  }
  state_space.BulkSetAmpl(state, 4 | 1, 4, 0, 0);
  EXPECT_EQ(state_space.GetAmpl(state, 0), std::complex<fp_type>(1, 1));
  EXPECT_EQ(state_space.GetAmpl(state, 1), std::complex<fp_type>(1, 1));
  EXPECT_EQ(state_space.GetAmpl(state, 2), std::complex<fp_type>(1, 1));
  EXPECT_EQ(state_space.GetAmpl(state, 3), std::complex<fp_type>(1, 1));
  EXPECT_EQ(state_space.GetAmpl(state, 4), std::complex<fp_type>(0, 0));
  EXPECT_EQ(state_space.GetAmpl(state, 5), std::complex<fp_type>(1, 1));
  EXPECT_EQ(state_space.GetAmpl(state, 6), std::complex<fp_type>(0, 0));
  EXPECT_EQ(state_space.GetAmpl(state, 7), std::complex<fp_type>(1, 1));
}

template <typename StateSpace>
void TestThreadThrashing() {
  using State = typename StateSpace::State;

  StateSpace state_space(1024);

  unsigned num_qubits = 13;

  State state = state_space.Create(num_qubits);  // must be larger than MIN_SIZE.
  for (unsigned i = 0; i < 100; ++i) {
    state_space.SetStateZero(state);
  }

  EXPECT_EQ(state_space.GetAmpl(state, 0), std::complex<float>(1, 0));

  unsigned size = 1 << num_qubits;
  for (unsigned i = 1; i < size; ++i) {
    EXPECT_EQ(state_space.GetAmpl(state, i), std::complex<float>(0, 0));
  }
}

}  // namespace qsim

#endif  // STATESPACE_TESTFIXTURE_H_
