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

#include <complex>

#include "gtest/gtest.h"

#include "../lib/fuser.h"
#include "../lib/gates_appl.h"
#include "../lib/gates_def.h"
#include "../lib/parfor.h"
#include "../lib/simulator_avx.h"

namespace qsim {

TEST(SimulatorAVXTest, ApplyGate1) {
  unsigned num_qubits = 1;
  unsigned num_threads = 1;

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<float>::Create(0, 0);
  auto gate2 = GateT<float>::Create(1, 0);
  auto gate3 = GateRX<float>::Create(2, 0, 0.5);
  auto gate4 = GateRXY<float>::Create(3, 0, 0.4, 0.3);
  auto gate5 = GateZ<float>::Create(4, 0);
  auto gate6 = GateHZ2<float>::Create(5, 0);

  ApplyGate(simulator, gate1, state);
  ApplyGate(simulator, gate2, state);
  ApplyGate(simulator, gate3, state);
  ApplyGate(simulator, gate4, state);
  ApplyGate(simulator, gate5, state);
  ApplyGate(simulator, gate6, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.37798857, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), 0.66353267, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.41492094, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.49466114, 1e-6);
  }
}

TEST(SimulatorAVXTest, ApplyGate2) {
  unsigned num_qubits = 2;
  unsigned num_threads = 1;

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<float>::Create(0, 0);
  auto gate2 = GateHd<float>::Create(0, 1);
  auto gate3 = GateT<float>::Create(1, 0);
  auto gate4 = GateT<float>::Create(1, 1);
  auto gate5 = GateRX<float>::Create(2, 0, 0.7);
  auto gate6 = GateRXY<float>::Create(2, 1, 0.2, 0.8);
  auto gate7 = GateZ<float>::Create(3, 0);
  auto gate8 = GateHZ2<float>::Create(3, 1);
  auto gate9 = GateCZ<float>::Create(1, 0, 1);

  ApplyGate(simulator, gate1, state);
  ApplyGate(simulator, gate2, state);
  ApplyGate(simulator, gate3, state);
  ApplyGate(simulator, gate4, state);
  ApplyGate(simulator, gate5, state);
  ApplyGate(simulator, gate6, state);
  ApplyGate(simulator, gate7, state);
  ApplyGate(simulator, gate8, state);
  ApplyGate(simulator, gate9, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.53100818, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.17631586, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), -0.32348031, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.11164886, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), 0.64307469, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.03410439, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.29973805, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), 0.25551257, 1e-6);
  }
}

TEST(SimulatorAVXTest, ApplyGate3) {
  unsigned num_qubits = 3;
  unsigned num_threads = 1;

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<float>::Create(0, 0);
  auto gate2 = GateHd<float>::Create(0, 1);
  auto gate3 = GateHd<float>::Create(0, 2);
  auto gate4 = GateT<float>::Create(1, 0);
  auto gate5 = GateT<float>::Create(1, 1);
  auto gate6 = GateT<float>::Create(1, 2);
  auto gate7 = GateCZ<float>::Create(2, 0, 1);
  auto gate8 = GateS<float>::Create(3, 0);
  auto gate9 = GateRX<float>::Create(3, 1, 0.7);
  auto gate10 = GateIS<float>::Create(4, 1, 2);
  auto gate11 = GateRY<float>::Create(5, 1, 0.4);
  auto gate12 = GateT<float>::Create(5, 2);

  ApplyGate(simulator, gate1, state);
  ApplyGate(simulator, gate2, state);
  ApplyGate(simulator, gate3, state);
  ApplyGate(simulator, gate4, state);
  ApplyGate(simulator, gate5, state);
  ApplyGate(simulator, gate6, state);
  ApplyGate(simulator, gate7, state);
  ApplyGate(simulator, gate8, state);
  ApplyGate(simulator, gate9, state);
  ApplyGate(simulator, gate10, state);
  ApplyGate(simulator, gate11, state);
  ApplyGate(simulator, gate12, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.45616995, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.15475702, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), -0.24719276, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), 0.16029677, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.14714938, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.33194723, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.037359533, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.21891155, 1e-6);
  }
}

TEST(SimulatorAVXTest, ApplyGate5) {
  unsigned num_qubits = 5;
  unsigned num_threads = 1;

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<float>::Create(0, 0);
  auto gate2 = GateHd<float>::Create(0, 1);
  auto gate3 = GateHd<float>::Create(0, 2);
  auto gate4 = GateHd<float>::Create(0, 3);
  auto gate5 = GateHd<float>::Create(0, 4);
  auto gate6 = GateT<float>::Create(1, 0);
  auto gate7 = GateT<float>::Create(1, 1);
  auto gate8 = GateT<float>::Create(1, 2);
  auto gate9 = GateT<float>::Create(1, 3);
  auto gate10 = GateT<float>::Create(1, 4);
  auto gate11 = GateCZ<float>::Create(2, 0, 1);
  auto gate12 = GateX<float>::Create(3, 0);
  auto gate13 = GateY<float>::Create(3, 1);
  auto gate14 = GateIS<float>::Create(4, 1, 2);
  auto gate15 = GateX2<float>::Create(5, 1);
  auto gate16 = GateY2<float>::Create(5, 2);
  auto gate17 = GateCNot<float>::Create(6, 2, 3);
  auto gate18 = GateRX<float>::Create(7, 2, 0.9);
  auto gate19 = GateRY<float>::Create(7, 3, 0.2);
  auto gate20 = GateFS<float>::Create(8, 3, 4, 0.4, 0.6);
  auto gate21 = GateRXY<float>::Create(9, 3, 0.8, 0.1);
  auto gate22 = GateRZ<float>::Create(9, 4, 0.4);
  auto gate23 = GateCP<float>::Create(10, 0, 1, 0.7);
  auto gate24 = GateY2<float>::Create(11, 3);
  auto gate25 = GateRX<float>::Create(11, 4, 0.3);

  GateFused<GateQSim<float>> fgate1{kGateCZ, 2, 2, {0, 1}, &gate11,
      {&gate1, &gate2, &gate6, &gate7, &gate11, &gate12, &gate13}};
  ApplyFusedGate(simulator, fgate1, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), -0.5, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), 0, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.35355335, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.35355335, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.35355335, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.35355335, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), 0.5, 1e-6);
  }

  GateFused<GateQSim<float>> fgate2{kGateIS, 4, 2, {1, 2}, &gate14,
      {&gate3, &gate8, &gate14, &gate15, &gate16}};
  ApplyFusedGate(simulator, fgate2, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.17677667, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.17677667, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.24999997, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), 0.35355339, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), 0.07322330, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.42677669, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.25, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), 0, 1e-6);
  }


  GateFused<GateQSim<float>> fgate3{kGateCNot, 6, 2, {2, 3}, &gate17,
      {&gate4, &gate9, &gate17, &gate18, &gate19}};
  ApplyFusedGate(simulator, fgate3, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.03269903, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.21794335, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.20945998, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), 0.25095809, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.00879907, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.24247472, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.02421566, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), 0.00031570, 1e-6);
  }

  GateFused<GateQSim<float>> fgate4{kGateFS, 8, 2, {3, 4}, &gate20,
      {&gate5, &gate10, &gate20, &gate21, &gate22}};
  ApplyFusedGate(simulator, fgate4, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), -0.00938794, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.15174214, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.18047242, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), 0.13247597, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), 0.03860849, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.16120625, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.01934232, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.00987822, 1e-6);
  }

  GateFused<GateQSim<float>> fgate5{kGateCP, 10, 2, {0, 1}, &gate23, {&gate23}};
  ApplyFusedGate(simulator, fgate5, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), -0.00938794, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.15174214, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.18047242, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), 0.13247597, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), 0.03860849, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.16120625, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.00843010, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.02001594, 1e-6);
  }

  ApplyGate(simulator, gate24, state);
  ApplyGate(simulator, gate25, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.05261526, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.03246338, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.02790548, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.02198864, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), 0.10250939, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), -0.02654653, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), -0.03221833, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.11284899, 1e-6);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
