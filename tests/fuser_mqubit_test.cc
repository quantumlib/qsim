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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "fuser_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/fuser.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gate.h"
#include "../lib/gate_appl.h"
#include "../lib/matrix.h"
#include "../lib/operation.h"
#include "../lib/simmux.h"

namespace qsim {

struct IO {
  static void errorf(const char* format, ...) {}
  static void messagef(const char* format, ...) {}
};

namespace {

Gate<float> CreateGate(unsigned time, std::vector<unsigned>&& qubits) {
  return {0, time, std::move(qubits), {}, {}, false};
}

ControlledGate<float> CreateControlledGate(
    unsigned time, std::vector<unsigned>&& qubits,
    std::vector<unsigned>&& controlled_by) {
  return Gate<float>{0, time, std::move(qubits), {}, {}, false}.ControlledBy(
      std::move(controlled_by));
}

std::vector<unsigned> GenQubits(unsigned num_qubits, std::mt19937& rgen,
                                unsigned& n, std::vector<unsigned>& available) {
  std::vector<unsigned> qubits;
  qubits.reserve(num_qubits);

  for (unsigned i = 0; i < num_qubits; ++i) {
    unsigned k = rgen() % n--;
    qubits.push_back(available[k]);
    std::swap(available[k], available[n]);
  }

  return qubits;
}

constexpr double p0 = 0.02;
constexpr double p1 = 0.18 + p0;
constexpr double p2 = 0.6 + p1;
constexpr double p3 = 0.08 + p2;
constexpr double p4 = 0.05 + p3;
constexpr double p5 = 0.035 + p4;
constexpr double p6 = 0.02 + p5;
constexpr double pc = 0.0075 + p6;
constexpr double pm = 0.0075 + pc;

constexpr double pd = 0.002;

template <typename OperationD>
void AddToCircuit(unsigned time,
                  std::uniform_real_distribution<double>& distr,
                  std::mt19937& rgen, unsigned& n,
                  std::vector<unsigned>& available,
                  std::vector<OperationD>& circuit) {
  double r = distr(rgen);

  if (r < p0) {
    circuit.push_back(CreateGate(time, {}));
  } else if (r < p1) {
    circuit.push_back(CreateGate(time, GenQubits(1, rgen, n, available)));

    if (distr(rgen) < pd) {
      using Gate = qsim::Gate<float>;
      using DecomposedGate = qsim::DecomposedGate<float>;
      circuit.back() = DecomposedGate{*OpGetAlternative<Gate>(circuit.back())};
    }
  } else if (r < p2) {
    circuit.push_back(CreateGate(time, GenQubits(2, rgen, n, available)));
  } else if (r < p3) {
    circuit.push_back(CreateGate(time, GenQubits(3, rgen, n, available)));
  } else if (r < p4) {
    circuit.push_back(CreateGate(time, GenQubits(4, rgen, n, available)));
  } else if (r < p5) {
    circuit.push_back(CreateGate(time, GenQubits(5, rgen, n, available)));
  } else if (r < p6) {
    circuit.push_back(CreateGate(time, GenQubits(6, rgen, n, available)));
  } else if (r < pc) {
    auto qs = GenQubits(1 + rgen() % 3, rgen, n, available);
    auto cqs = GenQubits(1 + rgen() % 3, rgen, n, available);
    circuit.push_back(
        CreateControlledGate(time, std::move(qs), std::move(cqs)));
  } else if (r < pm) {
    unsigned num_mea_gates = 0;
    unsigned max_num_mea_gates = 1 + rgen() % 5;

    while (n > 0 && num_mea_gates < max_num_mea_gates) {
      unsigned k = 1 + rgen() % 12;
      if (k > n) k = n;
      circuit.push_back(
          CreateMeasurement(time, GenQubits(k, rgen, n, available)));
      ++num_mea_gates;
    }
  }

  auto& op = circuit.back();

  if (!OpGetAlternative<Measurement>(op)) {
    auto& base_op = OpBaseOperation(op);
    std::sort(base_op.qubits.begin(), base_op.qubits.end());
  }
}

auto GenerateRandomCircuit1(unsigned num_qubits, unsigned depth) {
  using DecomposedGate = qsim::DecomposedGate<float>;
  using Operation = qsim::Operation<float>;
  using OperationD = detail::append_to_variant_t<Operation, DecomposedGate>;

  std::vector<OperationD> circuit;
  circuit.reserve(depth);

  std::mt19937 rgen(1);
  std::uniform_real_distribution<double> distr(0, 1);

  std::vector<unsigned> available(num_qubits, 0);

  for (unsigned time = 0; time < depth; ++time) {
    for (unsigned k = 0; k < num_qubits; ++k) {
      available[k] = k;
    }

    unsigned n = num_qubits;

    AddToCircuit(time, distr, rgen, n, available, circuit);
  }

  return circuit;
}

auto GenerateRandomCircuit2(unsigned num_qubits, unsigned depth,
                            unsigned max_fused_gate_size) {
  using DecomposedGate = qsim::DecomposedGate<float>;
  using Operation = qsim::Operation<float>;
  using OperationD = detail::append_to_variant_t<Operation, DecomposedGate>;

  std::vector<OperationD> circuit;
  circuit.reserve(num_qubits * depth);

  std::mt19937 rgen(2);
  std::uniform_real_distribution<double> distr(0, 1);

  std::vector<unsigned> available(num_qubits, 0);

  for (unsigned time = 0; time < depth; ++time) {
    for (unsigned k = 0; k < num_qubits; ++k) {
      available[k] = k;
    }

    unsigned n = num_qubits;

    while (n > max_fused_gate_size) {
      AddToCircuit(time, distr, rgen, n, available, circuit);
    }
  }

  return circuit;
}

}  // namespace

TEST(FuserMultiQubitTest, RandomCircuit1) {
  using Fuser = MultiQubitGateFuser<IO>;

  unsigned num_qubits = 30;
  unsigned depth = 100000;

  // Random circuit of 100000 gates.
  auto circuit = GenerateRandomCircuit1(num_qubits, depth);

  Fuser::Parameter param;
  param.verbosity = 0;

  using Operation = decltype(circuit)::value_type;

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(),
        {5000, 7000, 25000, 37000}, false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }
}

TEST(FuserMultiQubitTest, RandomCircuit2) {
  using Fuser = MultiQubitGateFuser<IO>;

  unsigned num_qubits = 40;
  unsigned depth = 6400;

  // Random circuit of approximately 100000 gates.
  auto circuit = GenerateRandomCircuit2(num_qubits, depth, 6);

  using Operation = decltype(circuit)::value_type;

  // Vector of pointers to gates.
  std::vector<const Operation*> pcircuit;
  pcircuit.reserve(circuit.size());

  for (const auto& gate : circuit) {
    pcircuit.push_back(&gate);
  }

  Fuser::Parameter param;
  param.verbosity = 0;

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates<const Operation*>(
        param, num_qubits, pcircuit.begin(), pcircuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates<const Operation*>(
        param, num_qubits, pcircuit.begin(), pcircuit.end(),
        {300, 700, 2400}, false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }
}

TEST(FuserMultiQubitTest, Simulation) {
  using Gate = qsim::Gate<float>;
  using FusedGate = qsim::FusedGate<float>;
  using ControlledGate = ControlledGate<float>;
  using DecomposedGate = qsim::DecomposedGate<float>;
  using Fuser = MultiQubitGateFuser<IO>;

  unsigned num_qubits = 12;
  unsigned depth = 200;

  auto circuit = GenerateRandomCircuit2(num_qubits, depth, 6);

  std::mt19937 rgen(1);
  std::uniform_real_distribution<double> distr(0, 1);

  for (auto& op : circuit) {
    if (OpGetAlternative<Measurement>(op)) continue;

    if (auto* pg = OpGetAlternative<Gate>(op)) {
      unsigned size = unsigned{1} << (2 * pg->qubits.size() + 1);
      pg->matrix.reserve(size);

      // Random gate matrices.
      for (unsigned i = 0; i < size; ++i) {
        pg->matrix.push_back(2 * distr(rgen) - 1);
      }
    } else if (auto* pg = OpGetAlternative<DecomposedGate>(op)) {
      unsigned size = unsigned{1} << (2 * pg->qubits.size() + 1);
      pg->matrix.reserve(size);

      // Random gate matrices.
      for (unsigned i = 0; i < size; ++i) {
        pg->matrix.push_back(2 * distr(rgen) - 1);
      }
    } else if (auto* pg = OpGetAlternative<ControlledGate>(op)) {
      unsigned size = unsigned{1} << (2 * pg->qubits.size() + 1);
      pg->matrix.reserve(size);

      // Random gate matrices.
      for (unsigned i = 0; i < size; ++i) {
        pg->matrix.push_back(2 * distr(rgen) - 1);
      }

      pg->cmask = (uint64_t{1} << pg->controlled_by.size()) - 1;
    }
  }

  using StateSpace = typename Simulator<For>::StateSpace;

  Simulator<For> simulator(1);
  StateSpace state_space(1);

  auto state0 = state_space.Create(num_qubits);
  state_space.SetStateZero(state0);

  // Simulate unfused gates.
  for (const auto& gate : circuit) {
    ApplyGate(simulator, gate, state0);
    // Renormalize the state to prevent floating point overflow.
    state_space.Multiply(1.0 / std::sqrt(state_space.Norm(state0)), state0);
  }

  Fuser::Parameter param;
  param.verbosity = 0;

  auto state1 = state_space.Create(num_qubits);

  for (unsigned q = 2; q <= 6; ++q) {
    state_space.SetStateZero(state1);

    using Operation = decltype(circuit)::value_type;

    param.max_fused_size = q;
    auto fused_ops = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_ops));

    // Simulate fused gates.
    for (const auto& op : fused_ops) {
      if (const auto* pg = OpGetAlternative<FusedGate>(op)) {
        if (!pg->ParentIsDecomposed()) {
          ApplyGate(simulator, op, state1);
        } else {
          auto fgate = *pg;
          CalculateFusedMatrix(fgate);
          ApplyGate(simulator, fgate, state1);
        }
      } else {
        ApplyGate(simulator, op, state1);
      }

      // Renormalize the state to prevent floating point overflow.
      state_space.Multiply(1.0 / std::sqrt(state_space.Norm(state1)), state1);
    }

    unsigned size = 1 << (num_qubits + 1);
    for (unsigned i = 0; i < size; ++i) {
      EXPECT_NEAR(state0.get()[i], state1.get()[i], 1e-6);
    }
  }
}

TEST(FuserMultiQubitTest, SmallCircuits) {
  using Gate = qsim::Gate<float>;
  using FusedGate = qsim::FusedGate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = MultiQubitGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(1, {1, 2}),
      CreateGate(2, {0, 1}),
      CreateGate(2, {2, 3}),
      CreateGate(3, {1, 2}),
    };

    param.max_fused_size = 4;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(1, {1, 2}),
      CreateControlledGate(1, {0}, {3}),
      CreateGate(3, {0, 1}),
      CreateGate(3, {2, 3}),
      CreateGate(4, {1, 2}),
    };

    param.max_fused_size = 4;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 3);
  }

  {
    unsigned num_qubits = 6;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(0, {4, 5}),
      CreateGate(1, {1, 2}),
      CreateGate(1, {3, 4}),
      CreateGate(2, {0, 1}),
      CreateGate(2, {2, 3}),
      CreateGate(2, {4, 5}),
      CreateGate(3, {1, 2}),
      CreateGate(3, {3, 4}),
    };

    param.max_fused_size = 6;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 6;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(0, {4, 5}),
      CreateGate(1, {1, 2}),
      CreateGate(1, {3, 4}),
      CreateMeasurement(2, {0, 1, 2}),
      CreateMeasurement(2, {4, 5}),
      CreateGate(3, {0, 1}),
      CreateGate(3, {2, 3}),
      CreateGate(3, {4, 5}),
      CreateGate(4, {1, 2}),
      CreateGate(4, {3, 4}),
    };

    param.max_fused_size = 6;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 3);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1, 2}),
      CreateGate(1, {2, 3}),
      CreateGate(2, {1, 2}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 2);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(1, {1, 2}),
      CreateGate(2, {2, 3}),
      CreateGate(3, {0, 1, 2}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 3);
  }

  {
    unsigned num_qubits = 5;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(1, {1, 2}),
      CreateGate(2, {2, 3}),
      CreateGate(3, {3, 4}),
      CreateGate(4, {0, 1, 2, 3}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 3);
  }

  {
    unsigned num_qubits = 6;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(0, {4, 5}),
      CreateGate(1, {1, 2}),
      CreateGate(1, {3, 4}),
      CreateGate(1, {5, 0}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 4);
  }

  {
    unsigned num_qubits = 8;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(0, {4, 5}),
      CreateGate(0, {6, 7}),
      CreateGate(1, {1, 2}),
      CreateGate(1, {3, 4}),
      CreateGate(1, {5, 6}),
      CreateGate(1, {7, 0}),
    };

    param.max_fused_size = 5;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 3);
  }

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      CreateGate(0, {1, 2}),
      CreateGate(1, {0, 1}),
      CreateGate(2, {1, 2}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(1, {0, 2}),
      CreateGate(2, {1, 2}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 6;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(1, {2, 3}),
      CreateGate(2, {4, 5}),
      CreateGate(3, {1, 2}),
      CreateGate(4, {2, 3, 4}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 4);
  }

  {
    unsigned num_qubits = 7;
    std::vector<Gate> circuit = {
      CreateGate(0, {1, 6}),
      CreateGate(1, {0, 1}),
      CreateGate(1, {4, 5}),
      CreateGate(2, {1, 2, 3}),
      CreateGate(2, {5, 6}),
      CreateGate(3, {3, 4}),
    };

    {
      param.max_fused_size = 3;
      auto fused_gates = Fuser::FuseGates<Gate>(
          param, num_qubits, circuit.begin(), circuit.end(), false);

      EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
      EXPECT_EQ(fused_gates.size(), 4);
    }

    {
      param.max_fused_size = 5;
      auto fused_gates = Fuser::FuseGates<Gate>(
          param, num_qubits, circuit.begin(), circuit.end(), false);

      EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));

      // The number of fused gates is not optimal here.
      // This may change in the future.
      EXPECT_EQ(fused_gates.size(), 4);
    }
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      CreateGate(0, {}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      CreateGate(0, {}),
      CreateGate(0, {}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Operation> circuit = {
      CreateControlledGate(0, {0}, {1}),
      CreateGate(1, {0, 1}),
      CreateGate(1, {}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 2);

    const auto* fgate1 = OpGetAlternative<FusedGate>(fused_gates[1]);
    ASSERT_NE(fgate1, nullptr);
    const auto* gate11 = OpGetAlternative<Gate>(fgate1->gates[1]);
    ASSERT_NE(gate11, nullptr);
    EXPECT_EQ(gate11, OpGetAlternative<Gate>(circuit[2]));
  }
}

TEST(FuserMultiQubitTest, ValidTimeOrder) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = MultiQubitGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 8;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateControlledGate(1, {1}, {2}),
      CreateGate(0, {4, 5}),
      CreateGate(2, {0, 1}),
      CreateGate(1, {3, 4}),
      CreateGate(2, {2, 3}),
      CreateGate(3, {1, 2}),
      CreateControlledGate(2, {4}, {5}),
      CreateGate(3, {3, 4}),
      CreateGate(5, {0, 1}),
      CreateGate(4, {2, 3}),
      CreateGate(5, {4, 5}),
      CreateGate(4, {6, 7}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 14);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 6;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(1, {1, 2}),
      CreateGate(0, {4, 5}),
      CreateGate(1, {3, 4}),
      CreateMeasurement(2, {0, 1, 2}),
      CreateMeasurement(2, {4, 5}),
      CreateGate(3, {0, 1}),
      CreateGate(3, {2, 3}),
      CreateGate(4, {1, 2}),
      CreateGate(3, {4, 5}),
      CreateGate(4, {3, 4}),
    };

    param.max_fused_size = 6;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 3);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 6;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(1, {1, 2}),
      CreateGate(0, {4, 5}),
      CreateGate(1, {3, 4}),
      CreateGate(1, {5, 0}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 4);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 8;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(0, {4, 5}),
      CreateGate(1, {1, 2}),
      CreateGate(1, {3, 4}),
      CreateGate(0, {6, 7}),
      CreateGate(1, {5, 6}),
      CreateGate(1, {7, 0}),
    };

    param.max_fused_size = 5;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 3);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 6;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(1, {2, 3}),
      CreateGate(3, {1, 2}),
      CreateGate(2, {4, 5}),
      CreateGate(4, {2, 3, 4}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 4);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 7;
    std::vector<Gate> circuit = {
      CreateGate(0, {1, 6}),
      CreateGate(1, {0, 1}),
      CreateGate(2, {1, 2, 3}),
      CreateGate(1, {4, 5}),
      CreateGate(3, {3, 4}),
      CreateGate(2, {5, 6}),
    };

    {
      param.max_fused_size = 3;
      auto fused_gates = Fuser::FuseGates<Gate>(
          param, num_qubits, circuit.begin(), circuit.end(), false);

      EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    }

    {
      param.max_fused_size = 5;
      auto fused_gates = Fuser::FuseGates<Gate>(
          param, num_qubits, circuit.begin(), circuit.end(), false);

      EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    }
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(2, {1, 2}),
      CreateGate(1, {0, 3}),
      CreateGate(3, {1, 2}),
      CreateGate(3, {0, 3}),
      CreateGate(4, {0, 1}),
      CreateGate(4, {2, 3}),
      CreateGate(6, {0, 3}),
      CreateGate(5, {1, 2}),
    };

    param.max_fused_size = 4;
    std::vector<unsigned> time_boundary = {3};
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(),
        time_boundary, false);

    EXPECT_EQ(fused_gates.size(), 2);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(2, {1, 2}),
      CreateGate(1, {0, 3}),
      CreateMeasurement(3, {1, 2}),
      CreateGate(3, {0, 3}),
      CreateGate(4, {0, 1}),
      CreateGate(4, {2, 3}),
      CreateGate(6, {0, 3}),
      CreateGate(5, {1, 2}),
    };

    param.max_fused_size = 4;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 3);
    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      CreateGate(1, {0, 1}),
      CreateGate(2, {}),
    };

    param.max_fused_size = 2;
    std::vector<unsigned> time_boundary = {1};
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(),
        time_boundary, false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 2);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      CreateGate(1, {0, 1}),
      CreateGate(0, {}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
    EXPECT_EQ(fused_gates.size(), 1);
  }
}

TEST(FuserMultiQubitTest, InvalidTimeOrder) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = MultiQubitGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {1, 2}),
    };

    param.max_fused_size = 3;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(2, {1, 2}),
      CreateGate(1, {0, 2}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(2, {0, 3}),
      CreateGate(1, {1, 2}),
    };

    param.max_fused_size = 2;
    std::vector<unsigned> time_boundary = {1};
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(),
        time_boundary, false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(2, {0, 3}),
      CreateGate(1, {1, 2}),
    };

    param.max_fused_size = 2;
    std::vector<unsigned> time_boundary = {2};
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(),
        time_boundary, false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(2, {0, 3}),
      CreateMeasurement(1, {1, 2}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateMeasurement(2, {0, 3}),
      CreateGate(1, {1, 2}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateGate(2, {0, 3}),
      CreateControlledGate(1, {1}, {3}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateGate(0, {2, 3}),
      CreateControlledGate(2, {1}, {3}),
      CreateGate(1, {0, 3}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 2;
    std::vector<Gate> circuit = {
      CreateGate(1, {0, 1}),
      CreateGate(0, {}),
    };

    param.max_fused_size = 2;
    std::vector<unsigned> time_boundary = {1};
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(),
        time_boundary, false);

    EXPECT_EQ(fused_gates.size(), 0);
  }
}

TEST(FuserMultiQubitTest, QubitsOutOfRange) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = MultiQubitGateFuser<IO>;

  Fuser::Parameter param;
  param.verbosity = 0;

  {
    unsigned num_qubits = 3;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 3}),
      CreateGate(0, {1, 2}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }

  {
    unsigned num_qubits = 3;
    std::vector<Operation> circuit = {
      CreateGate(0, {0, 1}),
      CreateControlledGate(0, {2}, {3}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 0);
  }
}

TEST(FuserMultiQubitTest, OrphanedGates) {
  using Gate = qsim::Gate<float>;
  using Operation = qsim::Operation<float>;
  using Fuser = MultiQubitGateFuser<IO>;

  std::vector<Gate> circuit;
  circuit.reserve(6);

  Fuser::Parameter param;
  param.verbosity = 0;

  for (unsigned num_qubits = 2; num_qubits <= 6; ++ num_qubits) {
    circuit.resize(0);

    for (unsigned q = 0; q < num_qubits; ++q) {
      circuit.push_back(CreateGate(0, {q}));
    }

    for (unsigned f = 2; f <= num_qubits; ++f) {
      param.max_fused_size = f;
      auto fused_gates = Fuser::FuseGates<Gate>(
          param, num_qubits, circuit.begin(), circuit.end(), false);

      EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
      EXPECT_EQ(fused_gates.size(), (num_qubits - 1) / f + 1);
    }
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0}),
      CreateGate(0, {1}),
      CreateGate(0, {2}),
      CreateGate(0, {3}),
      CreateGate(1, {0, 3}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 2);
  }

  {
    unsigned num_qubits = 4;
    std::vector<Gate> circuit = {
      CreateGate(0, {0, 3}),
      CreateGate(1, {0}),
      CreateGate(1, {1}),
      CreateGate(1, {2}),
      CreateGate(1, {3}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Gate>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 2);
  }

  {
    unsigned num_qubits = 3;
    std::vector<Operation> circuit = {
      CreateGate(0, {0}),
      CreateControlledGate(0, {1}, {2}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 2);
  }

  {
    unsigned num_qubits = 3;
    std::vector<Operation> circuit = {
      CreateGate(0, {0}),
      CreateMeasurement(0, {2}),
    };

    param.max_fused_size = 2;
    auto fused_gates = Fuser::FuseGates<Operation>(
        param, num_qubits, circuit.begin(), circuit.end(), false);

    EXPECT_EQ(fused_gates.size(), 2);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
