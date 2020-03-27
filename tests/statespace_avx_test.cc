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

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>

#include "gtest/gtest.h"

#include "../lib/circuit_reader.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_def.h"
#include "../lib/io.h"
#include "../lib/parfor.h"
#include "../lib/run_qsim.h"
#include "../lib/simulator_avx.h"
#include "../lib/statespace_avx.h"

namespace qsim {

TEST(StateSpaceAVXTest, SamplingBasic) {
  unsigned num_qubits = 3;
  uint64_t num_samples = 2000000;
  constexpr uint64_t size = 8;

  using StateSpace = StateSpaceAVX<ParallelFor>;
  using State = StateSpace::State;

  StateSpace state_space(num_qubits, 1);
  State state = state_space.CreateState();

  EXPECT_EQ(state_space.IsNull(state), false);
  EXPECT_EQ(state_space.Size(state), size);

  std::array<float, size> ps = {0.1, 0.2, 0.13, 0.12, 0.18, 0.15, 0.07, 0.05};

  for (uint64_t i = 0; i < size; ++i) {
    auto r = std::sqrt(ps[i]);
    state_space.SetAmpl(state, i, r * std::cos(i), r * std::sin(i));
  }

  auto samples = state_space.Sample(state, num_samples, 1);

  EXPECT_EQ(samples.size(), num_samples);
  std::array<double, size> bins = {0, 0, 0, 0, 0, 0, 0, 0};

  for (auto sample : samples) {
    ASSERT_LT(sample, size);
    bins[sample] += 1;
  }

  for (uint64_t i = 0; i < size; ++i) {
    EXPECT_NEAR(bins[i] / num_samples, ps[i], 4e-4);
  }
}

constexpr char provider[] = "statespace_avx_test";

constexpr char circuit_string[] =
R"(24
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
0 h 20
0 h 21
0 h 22
0 h 23
1 cz 0 1
1 cz 4 5
1 cz 8 9
1 cz 12 13
1 cz 16 17
1 cz 20 21
2 cz 2 3
2 cz 6 7
2 cz 10 11
2 cz 14 15
2 cz 18 19
2 cz 22 23
2 t 0
2 t 1
2 t 4
2 t 5
2 t 8
2 t 9
2 t 12
2 t 13
2 t 16
2 t 17
2 t 20
2 t 21
3 cz 6 12
3 cz 8 14
3 cz 10 16
3 t 2
3 t 3
3 t 7
3 t 11
3 t 15
3 t 18
3 t 19
3 t 22
3 t 23
4 cz 7 13
4 cz 9 15
4 cz 11 17
4 t 6
4 x_1_2 8
4 t 10
4 y_1_2 12
4 t 14
4 y_1_2 16
5 cz 1 2
5 cz 9 10
5 cz 13 14
5 cz 21 22
5 x_1_2 7
5 y_1_2 11
5 y_1_2 15
5 y_1_2 17
6 cz 3 4
6 cz 7 8
6 cz 15 16
6 cz 19 20
6 y_1_2 1
6 y_1_2 2
6 y_1_2 9
6 y_1_2 10
6 x_1_2 13
6 x_1_2 14
6 y_1_2 21
6 x_1_2 22
7 cz 0 6
7 cz 2 8
7 cz 4 10
7 cz 13 19
7 cz 15 21
7 cz 17 23
7 x_1_2 3
7 y_1_2 7
7 x_1_2 16
7 x_1_2 20
8 cz 1 7
8 cz 3 9
8 cz 5 11
8 cz 12 18
8 cz 14 20
8 cz 16 22
8 y_1_2 0
8 x_1_2 2
8 y_1_2 4
8 y_1_2 6
8 y_1_2 8
8 x_1_2 10
8 t 13
8 t 15
8 t 17
8 y_1_2 19
8 x_1_2 21
8 y_1_2 23
9 cz 0 1
9 cz 4 5
9 cz 8 9
9 cz 12 13
9 cz 16 17
9 cz 20 21
9 y_1_2 3
9 t 7
9 x_1_2 11
9 t 14
9 y_1_2 18
9 y_1_2 22
10 cz 2 3
10 cz 6 7
10 cz 10 11
10 cz 14 15
10 cz 18 19
10 cz 22 23
10 t 0
10 t 1
10 x_1_2 4
10 x_1_2 5
10 t 8
10 x_1_2 9
10 t 12
10 y_1_2 13
10 t 16
10 y_1_2 17
10 t 20
10 y_1_2 21
11 cz 6 12
11 cz 8 14
11 cz 10 16
11 t 2
11 x_1_2 3
11 y_1_2 7
11 t 11
11 y_1_2 15
11 t 18
11 x_1_2 19
11 t 22
11 x_1_2 23
12 cz 7 13
12 cz 9 15
12 cz 11 17
12 x_1_2 6
12 y_1_2 8
12 t 10
12 x_1_2 12
12 y_1_2 14
12 x_1_2 16
13 cz 1 2
13 cz 9 10
13 cz 13 14
13 cz 21 22
13 t 7
13 x_1_2 11
13 x_1_2 15
13 x_1_2 17
14 cz 3 4
14 cz 7 8
14 cz 15 16
14 cz 19 20
14 x_1_2 1
14 x_1_2 2
14 t 9
14 y_1_2 10
14 x_1_2 13
14 x_1_2 14
14 x_1_2 21
14 y_1_2 22
15 cz 0 6
15 cz 2 8
15 cz 4 10
15 cz 13 19
15 cz 15 21
15 cz 17 23
15 t 3
15 x_1_2 7
15 t 16
15 x_1_2 20
16 cz 1 7
16 cz 3 9
16 cz 5 11
16 cz 12 18
16 cz 14 20
16 cz 16 22
16 x_1_2 0
16 t 2
16 y_1_2 4
16 t 6
16 t 8
16 t 10
16 t 13
16 t 15
16 t 17
16 y_1_2 19
16 t 21
16 t 23
17 cz 0 1
17 cz 4 5
17 cz 8 9
17 cz 12 13
17 cz 16 17
17 cz 20 21
17 x_1_2 3
17 y_1_2 7
17 t 11
17 y_1_2 14
17 x_1_2 18
17 t 22
18 cz 2 3
18 cz 6 7
18 cz 10 11
18 cz 14 15
18 cz 18 19
18 cz 22 23
18 t 0
18 y_1_2 1
18 t 4
18 y_1_2 5
18 x_1_2 8
18 y_1_2 9
18 y_1_2 12
18 x_1_2 13
18 x_1_2 16
18 y_1_2 17
18 t 20
18 x_1_2 21
19 cz 6 12
19 cz 8 14
19 cz 10 16
19 y_1_2 2
19 y_1_2 3
19 x_1_2 7
19 x_1_2 11
19 x_1_2 15
19 y_1_2 18
19 t 19
19 x_1_2 22
19 y_1_2 23
20 cz 7 13
20 cz 9 15
20 cz 11 17
20 x_1_2 6
20 t 8
20 y_1_2 10
20 t 12
20 t 14
20 t 16
21 cz 1 2
21 cz 9 10
21 cz 13 14
21 cz 21 22
21 y_1_2 7
21 t 11
21 t 15
21 t 17
22 cz 3 4
22 cz 7 8
22 cz 15 16
22 cz 19 20
22 t 1
22 x_1_2 2
22 t 9
22 t 10
22 y_1_2 13
22 x_1_2 14
22 t 21
22 y_1_2 22
23 cz 0 6
23 cz 2 8
23 cz 4 10
23 cz 13 19
23 cz 15 21
23 cz 17 23
23 x_1_2 3
23 x_1_2 7
23 x_1_2 16
23 y_1_2 20
24 cz 1 7
24 cz 3 9
24 cz 5 11
24 cz 12 18
24 cz 14 20
24 cz 16 22
24 y_1_2 0
24 y_1_2 2
24 x_1_2 4
24 t 6
24 x_1_2 8
24 y_1_2 10
24 t 13
24 y_1_2 15
24 y_1_2 17
24 x_1_2 19
24 x_1_2 21
24 x_1_2 23
25 cz 0 1
25 cz 4 5
25 cz 8 9
25 cz 12 13
25 cz 16 17
25 cz 20 21
25 t 3
25 y_1_2 7
25 y_1_2 11
25 t 14
25 t 18
25 x_1_2 22
26 cz 2 3
26 cz 6 7
26 cz 10 11
26 cz 14 15
26 cz 18 19
26 cz 22 23
26 t 0
26 x_1_2 1
26 y_1_2 4
26 t 5
26 y_1_2 8
26 y_1_2 9
26 y_1_2 12
26 x_1_2 13
26 y_1_2 16
26 t 17
26 x_1_2 20
26 t 21
27 cz 6 12
27 cz 8 14
27 cz 10 16
27 x_1_2 2
27 y_1_2 3
27 x_1_2 7
27 t 11
27 x_1_2 15
27 y_1_2 18
27 y_1_2 19
27 y_1_2 22
27 y_1_2 23
28 cz 7 13
28 cz 9 15
28 cz 11 17
28 y_1_2 6
28 x_1_2 8
28 t 10
28 x_1_2 12
28 y_1_2 14
28 x_1_2 16
29 cz 1 2
29 cz 9 10
29 cz 13 14
29 cz 21 22
29 y_1_2 7
29 x_1_2 11
29 y_1_2 15
29 x_1_2 17
30 cz 3 4
30 cz 7 8
30 cz 15 16
30 cz 19 20
30 t 1
30 y_1_2 2
30 x_1_2 9
30 y_1_2 10
30 y_1_2 13
30 x_1_2 14
30 x_1_2 21
30 t 22
)";

TEST(StateSpaceAVXTest, SamplingCrossEntropyDifference) {
  unsigned depth = 30;
  uint64_t num_samples = 2000000;

  std::stringstream ss(circuit_string);
  Circuit<GateQSim<float>> circuit;
  EXPECT_EQ(CircuitReader<IO>::FromStream(30, provider, ss, circuit), true);

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<GateQSim<float>>, Simulator>;

  StateSpace state_space(circuit.num_qubits, 1);
  State state = state_space.CreateState();

  EXPECT_EQ(state_space.IsNull(state), false);

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.num_threads = 1;
  param.verbosity = 0;

  EXPECT_EQ(Runner::Run(param, depth, circuit, state), true);

  auto bitstrings = state_space.Sample(state, num_samples, 1);
  EXPECT_EQ(bitstrings.size(), num_samples);

  double sum = 0;
  for (uint64_t i = 0; i < num_samples; ++i) {
    double p = std::norm(state_space.GetAmpl(state, bitstrings[i]));
    sum += std::log(p);
  }

  double gamma = 0.5772156649;
  double ced = circuit.num_qubits * std::log(2) + gamma + sum / num_samples;

  EXPECT_NEAR(ced, 1.0, 1e-3);
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
