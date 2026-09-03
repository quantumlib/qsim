// Copyright 2026 Google LLC. All Rights Reserved.
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
#include <random>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/fuser.h"
#include "../lib/gate.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_qsim.h"
#include "../lib/matrix.h"
#include "../lib/operation.h"
#include "../lib/operation_base.h"
#include "../lib/seqfor.h"
#include "../lib/simulator_basic.h"

namespace qsim {

template <typename FP>
class GateApplTest : public ::testing::Test {
 protected:
  using fp_type = FP;
  using Simulator = SimulatorBasic<SequentialFor, fp_type>;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using MeasurementResult = typename StateSpace::MeasurementResult;

  static constexpr fp_type kEps = std::is_same_v<fp_type, float> ? 1e-6 : 1e-12;
};

using FPTestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GateApplTest, FPTestTypes);

TYPED_TEST(GateApplTest, ApplyGateDirect) {
  using fp_type = typename TestFixture::fp_type;
  using Simulator = typename TestFixture::Simulator;
  using StateSpace = typename TestFixture::StateSpace;

  StateSpace state_space(1);
  Simulator simulator(1);

  // Test single-qubit Gate.
  {
    auto state = state_space.Create(1);
    state_space.SetStateZero(state);

    auto gate_x = GateX<fp_type>::Create(0, 0);
    ApplyGate(simulator, gate_x, state);

    EXPECT_NEAR(std::abs(StateSpace::GetAmpl(state, 0)), 0, TestFixture::kEps);
    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 1) - fp_type(1)), 0,
        TestFixture::kEps);

    auto gate_h = GateHd<fp_type>::Create(1, 0);
    ApplyGate(simulator, gate_h, state);

    fp_type is2 = GateHd<fp_type>::is2;
    EXPECT_NEAR(
        std::real(StateSpace::GetAmpl(state, 0)), is2, TestFixture::kEps);
    EXPECT_NEAR(
        std::real(StateSpace::GetAmpl(state, 1)), -is2, TestFixture::kEps);
  }

  // Test two-qubit Gate.
  {
    auto state = state_space.Create(2);
    state_space.SetStateZero(state);

    auto gate_x0 = GateX<fp_type>::Create(0, 0);
    ApplyGate(simulator, gate_x0, state);

    auto gate_cnot = GateCNot<fp_type>::Create(1, 0, 1);
    ApplyGate(simulator, gate_cnot, state);

    // Initial |00> -> X(0) -> |01> (qubit 0 = 1) -> CNOT(0, 1) -> |11> (index
    // 3).
    EXPECT_NEAR(std::abs(StateSpace::GetAmpl(state, 0)), 0, TestFixture::kEps);
    EXPECT_NEAR(std::abs(StateSpace::GetAmpl(state, 1)), 0, TestFixture::kEps);
    EXPECT_NEAR(std::abs(StateSpace::GetAmpl(state, 2)), 0, TestFixture::kEps);
    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 3) - fp_type(1)), 0,
        TestFixture::kEps);
  }

  // Test ControlledGate with control value 1 (default).
  {
    auto state = state_space.Create(2);
    state_space.SetStateZero(state);

    auto cgate = GateX<fp_type>::Create(0, 1).ControlledBy({0});

    // Qubit 0 is 0, control condition not met, state remains |00>.
    ApplyGate(simulator, cgate, state);
    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 0) - fp_type(1)), 0,
        TestFixture::kEps);

    // Flip qubit 0 to 1 -> state is |01> (index 1).
    auto gate_x0 = GateX<fp_type>::Create(1, 0);
    ApplyGate(simulator, gate_x0, state);

    // Qubit 0 is 1, control condition met, flips qubit 1 -> state becomes |11>
    // (index 3).
    ApplyGate(simulator, cgate, state);
    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 3) - fp_type(1)), 0,
        TestFixture::kEps);
  }

  // Test ControlledGate with custom control value 0.
  {
    auto state = state_space.Create(2);
    state_space.SetStateZero(state);

    // Control on qubit 0 == 0.
    auto cgate0 = GateX<fp_type>::Create(0, 1).ControlledBy({0}, {0});

    // Qubit 0 is 0, so qubit 1 should flip -> state becomes |10> (index 2).
    ApplyGate(simulator, cgate0, state);
    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 2) - fp_type(1)), 0,
        TestFixture::kEps);

    // Apply again -> qubit 1 flips back -> state becomes |00> (index 0).
    ApplyGate(simulator, cgate0, state);
    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 0) - fp_type(1)), 0,
        TestFixture::kEps);
  }

  // Test FusedGate.
  {
    auto state_fused = state_space.Create(2);
    auto state_seq = state_space.Create(2);
    state_space.SetStateZero(state_fused);
    state_space.SetStateZero(state_seq);

    auto gate1 = GateHd<fp_type>::Create(0, 0);
    auto gate2 = GateCZ<fp_type>::Create(1, 0, 1);
    auto gate3 = GateT<fp_type>::Create(2, 0);
    auto gate4 = GateRX<fp_type>::Create(3, 1, 0.6);

    FusedGate<fp_type> fgate{
        kGateCZ, 1, {0, 1}, &gate2, {&gate1, &gate2, &gate3, &gate4}, {}};
    CalculateFusedMatrix(fgate);

    ApplyGate(simulator, fgate, state_fused);

    ApplyGate(simulator, gate1, state_seq);
    ApplyGate(simulator, gate2, state_seq);
    ApplyGate(simulator, gate3, state_seq);
    ApplyGate(simulator, gate4, state_seq);

    for (unsigned i = 0; i < 4; ++i) {
      auto a_fused = StateSpace::GetAmpl(state_fused, i);
      auto a_seq = StateSpace::GetAmpl(state_seq, i);
      EXPECT_NEAR(std::real(a_fused), std::real(a_seq), TestFixture::kEps);
      EXPECT_NEAR(std::imag(a_fused), std::imag(a_seq), TestFixture::kEps);
    }
  }

  // Test pointer dereferencing through OpGetAlternative.
  {
    auto state = state_space.Create(1);
    state_space.SetStateZero(state);

    auto gate_x = GateX<fp_type>::Create(0, 0);
    const auto* pgate = &gate_x;
    ApplyGate(simulator, pgate, state);

    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 1) - fp_type(1)), 0,
        TestFixture::kEps);
  }
}

TYPED_TEST(GateApplTest, ApplyGateOperationVariant) {
  using fp_type = typename TestFixture::fp_type;
  using Simulator = typename TestFixture::Simulator;
  using StateSpace = typename TestFixture::StateSpace;
  using Operation = qsim::Operation<fp_type>;

  StateSpace state_space(1);
  Simulator simulator(1);

  auto state = state_space.Create(2);
  state_space.SetStateZero(state);

  // Operation holding Gate.
  Operation op1 = GateX<fp_type>::Create(0, 0);
  ApplyGate(simulator, op1, state);
  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 1) - fp_type(1)), 0,
      TestFixture::kEps);

  // Operation holding ControlledGate.
  Operation op2 = GateX<fp_type>::Create(1, 1).ControlledBy({0});
  ApplyGate(simulator, op2, state);
  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 3) - fp_type(1)), 0,
      TestFixture::kEps);

  // Operation holding Measurement should be a no-op in 3-argument ApplyGate.
  Measurement meas;
  meas.time = 2;
  meas.qubits = {0};
  Operation op3 = meas;
  ApplyGate(simulator, op3, state);
  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 3) - fp_type(1)), 0,
      TestFixture::kEps);

  // Custom variant containing FusedGate.
  auto gate1 = GateHd<fp_type>::Create(3, 0);
  auto gate2 = GateHd<fp_type>::Create(4, 0);
  FusedGate<fp_type> fgate{kGateHd, 3, {0}, &gate1, {&gate1, &gate2}, {}};
  CalculateFusedMatrix(fgate);

  std::variant<Gate<fp_type>, FusedGate<fp_type>, Measurement> fop = fgate;
  ApplyGate(simulator, fop, state);
  // Two Hadamards on qubit 0 cancel out, state remains |11> (index 3).
  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 3) - fp_type(1)), 0,
      TestFixture::kEps);
}

TYPED_TEST(GateApplTest, ApplyGateDaggerDirect) {
  using fp_type = typename TestFixture::fp_type;
  using Simulator = typename TestFixture::Simulator;
  using StateSpace = typename TestFixture::StateSpace;

  StateSpace state_space(1);
  Simulator simulator(1);

  // Single-qubit non-Hermitian gate: T gate (T != T^dagger).
  {
    auto state = state_space.Create(1);
    state_space.SetStateZero(state);

    auto gate_h = GateHd<fp_type>::Create(0, 0);
    ApplyGate(simulator, gate_h, state);

    auto gate_t = GateT<fp_type>::Create(1, 0);
    ApplyGate(simulator, gate_t, state);

    // Apply dagger of T, should recover |+> state.
    ApplyGateDagger(simulator, gate_t, state);

    fp_type is2 = GateHd<fp_type>::is2;
    EXPECT_NEAR(
        std::real(StateSpace::GetAmpl(state, 0)), is2, TestFixture::kEps);
    EXPECT_NEAR(std::imag(StateSpace::GetAmpl(state, 0)), 0, TestFixture::kEps);
    EXPECT_NEAR(
        std::real(StateSpace::GetAmpl(state, 1)), is2, TestFixture::kEps);
    EXPECT_NEAR(std::imag(StateSpace::GetAmpl(state, 1)), 0, TestFixture::kEps);
  }

  // Two-qubit non-Hermitian gate: ISWAP gate.
  {
    auto state = state_space.Create(2);
    state_space.SetStateZero(state);

    auto gate_x0 = GateX<fp_type>::Create(0, 0);
    ApplyGate(simulator, gate_x0, state);

    auto gate_is = GateIS<fp_type>::Create(1, 0, 1);
    ApplyGate(simulator, gate_is, state);

    // State after ISWAP on |01> is i|10> (index 2).
    EXPECT_NEAR(std::imag(StateSpace::GetAmpl(state, 2)), 1, TestFixture::kEps);

    // Apply dagger, should recover |01> (index 1).
    ApplyGateDagger(simulator, gate_is, state);

    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 1) - fp_type(1)), 0,
        TestFixture::kEps);
    EXPECT_NEAR(std::abs(StateSpace::GetAmpl(state, 2)), 0, TestFixture::kEps);
  }

  // ControlledGate dagger with control values 1 (default).
  {
    auto state = state_space.Create(2);
    state_space.SetStateZero(state);

    // Prepare |01> (qubit 0 = 1, qubit 1 = 0).
    auto gate_x0 = GateX<fp_type>::Create(0, 0);
    ApplyGate(simulator, gate_x0, state);

    auto gate_h1 = GateHd<fp_type>::Create(1, 1);
    ApplyGate(simulator, gate_h1, state);

    // Controlled T gate on qubit 1 controlled by qubit 0.
    auto cgate = GateT<fp_type>::Create(2, 1).ControlledBy({0});
    ApplyGate(simulator, cgate, state);

    // Apply dagger of controlled gate.
    ApplyGateDagger(simulator, cgate, state);

    // Apply H on qubit 1 again to verify qubit 1 returned to |0>.
    ApplyGate(simulator, gate_h1, state);

    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 1) - fp_type(1)), 0,
        TestFixture::kEps);
    for (unsigned i = 0; i < 4; ++i) {
      if (i != 1) {
        EXPECT_NEAR(
            std::abs(StateSpace::GetAmpl(state, i)), 0, TestFixture::kEps);
      }
    }
  }

  // ControlledGate dagger with custom control value 0.
  {
    auto state = state_space.Create(2);
    state_space.SetStateZero(state);

    // Qubit 0 is 0.
    auto gate_h1 = GateHd<fp_type>::Create(0, 1);
    ApplyGate(simulator, gate_h1, state);

    auto cgate0 = GateT<fp_type>::Create(1, 1).ControlledBy({0}, {0});
    ApplyGate(simulator, cgate0, state);
    ApplyGateDagger(simulator, cgate0, state);

    ApplyGate(simulator, gate_h1, state);

    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 0) - fp_type(1)), 0,
        TestFixture::kEps);
  }

  // FusedGate dagger: U^dagger * U = I.
  {
    auto state = state_space.Create(2);
    state_space.SetStateZero(state);

    auto gate1 = GateHd<fp_type>::Create(0, 0);
    auto gate2 = GateHd<fp_type>::Create(0, 1);
    auto gate3 = GateT<fp_type>::Create(1, 0);
    auto gate4 = GateRX<fp_type>::Create(1, 1, 0.7);
    auto gate5 = GateCZ<fp_type>::Create(2, 0, 1);
    auto gate6 = GateIS<fp_type>::Create(3, 0, 1);

    FusedGate<fp_type> fgate{kGateIS,
                             3,
                             {0, 1},
                             &gate6,
                             {&gate1, &gate2, &gate3, &gate4, &gate5, &gate6},
                             {}};
    CalculateFusedMatrix(fgate);

    ApplyGate(simulator, fgate, state);
    EXPECT_NEAR(state_space.Norm(state), 1, TestFixture::kEps);

    ApplyGateDagger(simulator, fgate, state);

    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 0) - fp_type(1)), 0,
        TestFixture::kEps);
    for (unsigned i = 1; i < 4; ++i) {
      EXPECT_NEAR(
          std::abs(StateSpace::GetAmpl(state, i)), 0, TestFixture::kEps);
    }
  }

  // Measurement passed to ApplyGateDagger should be a no-op.
  {
    auto state = state_space.Create(1);
    state_space.SetStateZero(state);

    Measurement meas;
    meas.time = 0;
    meas.qubits = {0};

    ApplyGateDagger(simulator, meas, state);
    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 0) - fp_type(1)), 0,
        TestFixture::kEps);
  }

  // Pointer passed to ApplyGateDagger.
  {
    auto state = state_space.Create(1);
    state_space.SetStateZero(state);

    auto gate_x = GateX<fp_type>::Create(0, 0);
    ApplyGate(simulator, gate_x, state);

    const auto* pgate = &gate_x;
    ApplyGateDagger(simulator, pgate, state);

    EXPECT_NEAR(
        std::abs(StateSpace::GetAmpl(state, 0) - fp_type(1)), 0,
        TestFixture::kEps);
  }
}

TYPED_TEST(GateApplTest, ApplyGateDaggerOperationVariant) {
  using fp_type = typename TestFixture::fp_type;
  using Simulator = typename TestFixture::Simulator;
  using StateSpace = typename TestFixture::StateSpace;
  using Operation = qsim::Operation<fp_type>;

  StateSpace state_space(1);
  Simulator simulator(1);

  auto state = state_space.Create(2);
  state_space.SetStateZero(state);

  std::vector<Operation> ops;
  ops.push_back(GateHd<fp_type>::Create(0, 0));
  ops.push_back(GateT<fp_type>::Create(1, 0));
  ops.push_back(GateRX<fp_type>::Create(2, 1, 0.5));
  ops.push_back(GateT<fp_type>::Create(3, 1).ControlledBy({0}));
  ops.push_back(Measurement{4, 0, {0}});

  for (const auto& op : ops) {
    ApplyGate(simulator, op, state);
  }

  for (int i = ops.size() - 1; i >= 0; --i) {
    ApplyGateDagger(simulator, ops[i], state);
  }

  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 0) - fp_type(1)), 0,
      TestFixture::kEps);
  for (unsigned i = 1; i < 4; ++i) {
    EXPECT_NEAR(std::abs(StateSpace::GetAmpl(state, i)), 0, TestFixture::kEps);
  }
}

TYPED_TEST(GateApplTest, ApplyGateWithMeasurementAndResults) {
  using fp_type = typename TestFixture::fp_type;
  using Simulator = typename TestFixture::Simulator;
  using StateSpace = typename TestFixture::StateSpace;
  using MeasurementResult = typename TestFixture::MeasurementResult;
  using Operation = qsim::Operation<fp_type>;

  StateSpace state_space(1);
  Simulator simulator(1);
  std::mt19937 rgen(42);

  auto state = state_space.Create(2);
  state_space.SetStateZero(state);

  std::vector<MeasurementResult> mresults;

  // 1. Non-measurement gate types should apply the gate and leave mresults
  // untouched.
  Operation op_gate = GateX<fp_type>::Create(0, 0);
  EXPECT_TRUE(
      ApplyGate(state_space, simulator, op_gate, rgen, state, mresults));
  EXPECT_TRUE(mresults.empty());
  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 1) - fp_type(1)), 0,
      TestFixture::kEps);

  Operation op_cgate = GateX<fp_type>::Create(1, 1).ControlledBy({0});
  EXPECT_TRUE(
      ApplyGate(state_space, simulator, op_cgate, rgen, state, mresults));
  EXPECT_TRUE(mresults.empty());
  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 3) - fp_type(1)), 0,
      TestFixture::kEps);

  // 2. Deterministic measurement on state |11>.
  Measurement meas0;
  meas0.time = 2;
  meas0.qubits = {0};
  EXPECT_TRUE(ApplyGate(state_space, simulator, meas0, rgen, state, mresults));
  ASSERT_EQ(mresults.size(), 1);
  EXPECT_TRUE(mresults[0].valid);
  EXPECT_EQ(mresults[0].mask, 1);
  EXPECT_EQ(mresults[0].bits, 1);
  ASSERT_EQ(mresults[0].bitstring.size(), 1);
  EXPECT_EQ(mresults[0].bitstring[0], 1);
  EXPECT_NEAR(state_space.Norm(state), 1, TestFixture::kEps);

  // Multi-qubit measurement on |11>.
  Measurement meas01;
  meas01.time = 3;
  meas01.qubits = {0, 1};
  EXPECT_TRUE(ApplyGate(state_space, simulator, meas01, rgen, state, mresults));
  ASSERT_EQ(mresults.size(), 2);
  EXPECT_TRUE(mresults[1].valid);
  EXPECT_EQ(mresults[1].mask, 3);
  EXPECT_EQ(mresults[1].bits, 3);
  ASSERT_EQ(mresults[1].bitstring.size(), 2);
  EXPECT_EQ(mresults[1].bitstring[0], 1);
  EXPECT_EQ(mresults[1].bitstring[1], 1);

  // 3. Measurement on superposition state.
  state_space.SetStateZero(state);
  auto gate_h = GateHd<fp_type>::Create(4, 0);
  ApplyGate(simulator, gate_h, state);

  Measurement meas_superpos;
  meas_superpos.time = 5;
  meas_superpos.qubits = {0};
  EXPECT_TRUE(
      ApplyGate(state_space, simulator, meas_superpos, rgen, state, mresults));
  ASSERT_EQ(mresults.size(), 3);
  EXPECT_TRUE(mresults[2].valid);
  EXPECT_NEAR(state_space.Norm(state), 1, TestFixture::kEps);

  // Post-measurement state must have collapsed to either |0> or |1>.
  auto a0 = StateSpace::GetAmpl(state, 0);
  auto a1 = StateSpace::GetAmpl(state, 1);
  if (mresults[2].bitstring[0] == 0) {
    EXPECT_NEAR(std::abs(a0 - fp_type(1)), 0, TestFixture::kEps);
    EXPECT_NEAR(std::abs(a1), 0, TestFixture::kEps);
  } else {
    EXPECT_NEAR(std::abs(a0), 0, TestFixture::kEps);
    EXPECT_NEAR(std::abs(a1 - fp_type(1)), 0, TestFixture::kEps);
  }

  // 4. Invalid measurement (qubit out of bounds).
  Measurement invalid_meas;
  invalid_meas.time = 6;
  invalid_meas.qubits = {5};
  EXPECT_FALSE(
      ApplyGate(state_space, simulator, invalid_meas, rgen, state, mresults));
  EXPECT_EQ(mresults.size(), 3);
}

TYPED_TEST(GateApplTest, ApplyGateWithMeasurementDiscardResults) {
  using fp_type = typename TestFixture::fp_type;
  using Simulator = typename TestFixture::Simulator;
  using StateSpace = typename TestFixture::StateSpace;
  using Operation = qsim::Operation<fp_type>;

  StateSpace state_space(1);
  Simulator simulator(1);
  std::mt19937 rgen(123);

  auto state = state_space.Create(2);
  state_space.SetStateZero(state);

  // Non-measurement operations return true.
  Operation op_gate = GateX<fp_type>::Create(0, 0);
  EXPECT_TRUE(ApplyGate(state_space, simulator, op_gate, rgen, state));
  EXPECT_NEAR(
      std::abs(StateSpace::GetAmpl(state, 1) - fp_type(1)), 0,
      TestFixture::kEps);

  // Superposition measurement returns true and collapses the state.
  state_space.SetStateZero(state);
  auto gate_h = GateHd<fp_type>::Create(1, 0);
  ApplyGate(simulator, gate_h, state);

  Measurement meas;
  meas.time = 2;
  meas.qubits = {0};
  EXPECT_TRUE(ApplyGate(state_space, simulator, meas, rgen, state));
  EXPECT_NEAR(state_space.Norm(state), 1, TestFixture::kEps);

  auto a0 = StateSpace::GetAmpl(state, 0);
  auto a1 = StateSpace::GetAmpl(state, 1);
  bool is_zero = std::abs(a0 - fp_type(1)) < TestFixture::kEps &&
                 std::abs(a1) < TestFixture::kEps;
  bool is_one = std::abs(a0) < TestFixture::kEps &&
                std::abs(a1 - fp_type(1)) < TestFixture::kEps;
  EXPECT_TRUE(is_zero || is_one);

  // Invalid measurement returns false.
  Measurement invalid_meas;
  invalid_meas.time = 3;
  invalid_meas.qubits = {10};
  EXPECT_FALSE(ApplyGate(state_space, simulator, invalid_meas, rgen, state));
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
