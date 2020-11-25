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

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gate.h"
#include "../lib/gate_appl.h"
#include "../lib/io.h"
#include "../lib/matrix.h"
#include "../lib/simmux.h"

namespace qsim {

namespace {

enum DummyGateKind {
  kGateOther = 0,
  kMeasurement = gate::kMeasurement,
};

struct DummyGate {
  using GateKind = DummyGateKind;

  GateKind kind;
  unsigned time;
  uint64_t cmask;
  std::vector<unsigned> qubits;
  std::vector<unsigned> controlled_by;
  Matrix<float> matrix;
  bool unfusible;
};

DummyGate CreateDummyGate(unsigned time, std::vector<unsigned>&& qubits) {
  return {kGateOther, time, 0, std::move(qubits), {}, {}, false};
}

DummyGate CreateDummyMeasurementGate(unsigned time,
                                     std::vector<unsigned>&& qubits) {
  return {kMeasurement, time, 0, std::move(qubits), {}, {}, false};
}

DummyGate CreateDummyControlledGate(unsigned time,
                                    std::vector<unsigned>&& qubits,
                                    std::vector<unsigned>&& controlled_by) {
  return
  {kGateOther, time, 0, std::move(qubits), std::move(controlled_by), {}, false};
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

constexpr double p1 = 0.2;
constexpr double p2 = 0.6 + p1;
constexpr double p3 = 0.08 + p2;
constexpr double p4 = 0.05 + p3;
constexpr double p5 = 0.035 + p4;
constexpr double p6 = 0.02 + p5;
constexpr double pc = 0.0075 + p6;
constexpr double pm = 0.0075 + pc;

constexpr double pu = 0.002;

void AddToCircuit(unsigned time,
                  std::uniform_real_distribution<double>& distr,
                  std::mt19937& rgen, unsigned& n,
                  std::vector<unsigned>& available,
                  std::vector<DummyGate>& circuit) {
  double r = distr(rgen);

  if (r < p1) {
    circuit.push_back(CreateDummyGate(time, GenQubits(1, rgen, n, available)));
  } else if (r < p2) {
    circuit.push_back(CreateDummyGate(time, GenQubits(2, rgen, n, available)));
  } else if (r < p3) {
    circuit.push_back(CreateDummyGate(time, GenQubits(3, rgen, n, available)));
  } else if (r < p4) {
    circuit.push_back(CreateDummyGate(time, GenQubits(4, rgen, n, available)));
  } else if (r < p5) {
    circuit.push_back(CreateDummyGate(time, GenQubits(5, rgen, n, available)));
  } else if (r < p6) {
    circuit.push_back(CreateDummyGate(time, GenQubits(6, rgen, n, available)));
  } else if (r < pc) {
    auto qs = GenQubits(1 + rgen() % 3, rgen, n, available);
    auto cqs = GenQubits(1 + rgen() % 3, rgen, n, available);
    circuit.push_back(
        CreateDummyControlledGate(time, std::move(qs), std::move(cqs)));
  } else if (r < pm) {
    unsigned num_mea_gates = 0;
    unsigned max_num_mea_gates = 1 + rgen() % 5;

    while (n > 0 && num_mea_gates < max_num_mea_gates) {
      unsigned k = 1 + rgen() % 12;
      if (k > n) k = n;
      circuit.push_back(
          CreateDummyMeasurementGate(time, GenQubits(k, rgen, n, available)));
      ++num_mea_gates;
    }
  }

  if (r < p6 && distr(rgen) < pu) {
    circuit.back().unfusible = true;
  }

  auto& gate = circuit.back();

  if (gate.kind != gate::kMeasurement) {
    std::sort(gate.qubits.begin(), gate.qubits.end());
  }
}

std::vector<DummyGate> GenerateRandomCircuit1(unsigned num_qubits,
                                              unsigned depth) {
  std::vector<DummyGate> circuit;
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

std::vector<DummyGate> GenerateRandomCircuit2(unsigned num_qubits,
                                              unsigned depth,
                                              unsigned max_fused_gate_size) {
  std::vector<DummyGate> circuit;
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

template <typename FusedGate>
bool TestFusedGates(unsigned num_qubits,
                    const std::vector<DummyGate>& gates,
                    const std::vector<FusedGate>& fused_gates) {
  std::vector<unsigned> times(num_qubits, 0);
  std::vector<unsigned> gate_map(gates.size(), 0);

  // Test if gate times are ordered correctly.
  for (auto g : fused_gates) {
    if (g.parent->controlled_by.size() > 0 && g.gates.size() > 1) {
      return false;
    }

    for (auto p : g.gates) {
      auto k = (std::size_t(p) - std::size_t(gates.data())) / sizeof(*p);

      if (k >= gate_map.size()) {
        return false;
      }

      ++gate_map[k];

      if (p->kind == gate::kMeasurement) {
        if (g.parent->kind != gate::kMeasurement || g.parent->time != p->time) {
          return false;
        }
      }

      for (auto q : p->qubits) {
        if (p->time < times[q]) {
          return false;
        }
        times[q] = p->time;
      }

      for (auto q : p->controlled_by) {
        if (p->time < times[q]) {
          return false;
        }
        times[q] = p->time;
      }
    }
  }

  // Test if all gates are present only once.
  for (auto m : gate_map) {
    if (m != 1) {
      return false;
    }
  }

  return true;
}

}  // namespace

TEST(FuserMultiQubitTest, RandomCircuit1) {
  using Fuser = MultiQubitGateFuser<IO, DummyGate>;

  unsigned num_qubits = 30;
  unsigned depth = 100000;

  // Random circuit of 100000 gates.
  auto circuit = GenerateRandomCircuit1(num_qubits, depth);

  Fuser::Parameter param;
  param.verbosity = 0;

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end(),
        {5000, 7000, 25000, 37000});

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }
}

TEST(FuserMultiQubitTest, RandomCircuit2) {
  using Fuser = MultiQubitGateFuser<IO, DummyGate>;

  unsigned num_qubits = 40;
  unsigned depth = 6400;

  // Random circuit of approximately 100000 gates.
  auto circuit = GenerateRandomCircuit2(num_qubits, depth, 6);

  Fuser::Parameter param;
  param.verbosity = 0;

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }

  for (unsigned q = 2; q <= 6; ++q) {
    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end(), {300, 700, 2400});

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));
  }
}

TEST(FuserMultiQubitTest, Simulation) {
  using Fuser = MultiQubitGateFuser<IO, DummyGate>;

  unsigned num_qubits = 12;
  unsigned depth = 200;

  auto circuit = GenerateRandomCircuit2(num_qubits, depth, 6);

  std::mt19937 rgen(1);
  std::uniform_real_distribution<double> distr(0, 1);

  for (auto& gate : circuit) {
    if (gate.controlled_by.size() > 0) {
      gate.cmask = (uint64_t{1} << gate.controlled_by.size()) - 1;
    }

    unsigned size = unsigned{1} << (2 * gate.qubits.size() + 1);
    gate.matrix.reserve(size);

    // Random gate matrices.
    for (unsigned i = 0; i < size; ++i) {
      gate.matrix.push_back(2 * distr(rgen) - 1);
    }
  };

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

    param.max_fused_size = q;
    auto fused_gates = Fuser::FuseGates(
        param, num_qubits, circuit.begin(), circuit.end());

    EXPECT_TRUE(TestFusedGates(num_qubits, circuit, fused_gates));

    // Simulate fused gates.
    for (const auto& gate : fused_gates) {
      ApplyFusedGate(simulator, gate, state1);
      // Renormalize the state to prevent floating point overflow.
      state_space.Multiply(1.0 / std::sqrt(state_space.Norm(state1)), state1);
    }

    unsigned size = 1 << (num_qubits + 1);
    for (unsigned i = 0; i < size; ++i) {
      EXPECT_NEAR(state0.get()[i], state1.get()[i], 1e-6);
    }
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
