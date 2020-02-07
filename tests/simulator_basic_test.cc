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
#include "../lib/simulator_basic.h"

namespace qsim {

using fp_type = double;

TEST(SimulatorBasicTest, ApplyGate1) {
  unsigned num_qubits = 1;
  unsigned num_threads = 1;

  using Simulator = SimulatorBasic<ParallelFor, fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<fp_type>::Create(0, 0);
  auto gate2 = GateT<fp_type>::Create(1, 0);
  auto gate3 = GateRX<fp_type>::Create(2, 0, 0.5);
  auto gate4 = GateRXY<fp_type>::Create(3, 0, 0.4, 0.3);
  auto gate5 = GateZ<fp_type>::Create(4, 0);
  auto gate6 = GateHZ2<fp_type>::Create(5, 0);

  ApplyGate(simulator, gate1, state);
  ApplyGate(simulator, gate2, state);
  ApplyGate(simulator, gate3, state);
  ApplyGate(simulator, gate4, state);
  ApplyGate(simulator, gate5, state);
  ApplyGate(simulator, gate6, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.37798857, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), 0.66353267, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.41492094, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.49466114, 1e-6);
  }
}

TEST(SimulatorBasicTest, ApplyGate2) {
  unsigned num_qubits = 2;
  unsigned num_threads = 1;

  using Simulator = SimulatorBasic<ParallelFor, fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<fp_type>::Create(0, 0);
  auto gate2 = GateHd<fp_type>::Create(0, 1);
  auto gate3 = GateT<fp_type>::Create(1, 0);
  auto gate4 = GateT<fp_type>::Create(1, 1);
  auto gate5 = GateRX<fp_type>::Create(2, 0, 0.7);
  auto gate6 = GateRXY<fp_type>::Create(2, 1, 0.2, 0.8);
  auto gate7 = GateZ<fp_type>::Create(3, 0);
  auto gate8 = GateHZ2<fp_type>::Create(3, 1);
  auto gate9 = GateCZ<fp_type>::Create(4, 0, 1);

  ApplyGate(simulator, gate1, state);
  ApplyGate(simulator, gate2, state);
  ApplyGate(simulator, gate3, state);
  ApplyGate(simulator, gate4, state);
  ApplyGate(simulator, gate5, state);
  ApplyGate(simulator, gate6, state);
  ApplyGate(simulator, gate7, state);
  ApplyGate(simulator, gate8, state);
  ApplyGate(simulator, gate9, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

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

TEST(SimulatorBasicTest, ApplyGate3) {
  unsigned num_qubits = 3;
  unsigned num_threads = 1;

  using Simulator = SimulatorBasic<ParallelFor, fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<fp_type>::Create(0, 0);
  auto gate2 = GateHd<fp_type>::Create(0, 1);
  auto gate3 = GateHd<fp_type>::Create(0, 2);
  auto gate4 = GateT<fp_type>::Create(1, 0);
  auto gate5 = GateT<fp_type>::Create(1, 1);
  auto gate6 = GateT<fp_type>::Create(1, 2);
  auto gate7 = GateCZ<fp_type>::Create(2, 0, 1);
  auto gate8 = GateS<fp_type>::Create(3, 0);
  auto gate9 = GateRX<fp_type>::Create(3, 1, 0.7);
  auto gate10 = GateIS<fp_type>::Create(4, 1, 2);
  auto gate11 = GateRY<fp_type>::Create(5, 1, 0.4);
  auto gate12 = GateT<fp_type>::Create(5, 2);

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

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.36285768, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.013274317, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), -0.21313113, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), 0.06239493, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), 0.31317451, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), -0.36600887, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), -0.13067181, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), 0.26405340, 1e-6);
  }
}

TEST(SimulatorBasicTest, ApplyGate5) {
  unsigned num_qubits = 5;
  unsigned num_threads = 1;

  using Simulator = SimulatorBasic<ParallelFor, fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  auto gate1 = GateHd<fp_type>::Create(0, 0);
  auto gate2 = GateHd<fp_type>::Create(0, 1);
  auto gate3 = GateHd<fp_type>::Create(0, 2);
  auto gate4 = GateHd<fp_type>::Create(0, 3);
  auto gate5 = GateHd<fp_type>::Create(0, 4);
  auto gate6 = GateT<fp_type>::Create(1, 0);
  auto gate7 = GateT<fp_type>::Create(1, 1);
  auto gate8 = GateT<fp_type>::Create(1, 2);
  auto gate9 = GateT<fp_type>::Create(1, 3);
  auto gate10 = GateT<fp_type>::Create(1, 4);
  auto gate11 = GateCZ<fp_type>::Create(2, 0, 1);
  auto gate12 = GateX<fp_type>::Create(3, 0);
  auto gate13 = GateY<fp_type>::Create(3, 1);
  auto gate14 = GateIS<fp_type>::Create(4, 1, 2);
  auto gate15 = GateX2<fp_type>::Create(5, 1);
  auto gate16 = GateY2<fp_type>::Create(5, 2);
  auto gate17 = GateCNot<fp_type>::Create(6, 2, 3);
  auto gate18 = GateRX<fp_type>::Create(7, 2, 0.9);
  auto gate19 = GateRY<fp_type>::Create(7, 3, 0.2);
  auto gate20 = GateFS<fp_type>::Create(8, 3, 4, 0.4, 0.6);
  auto gate21 = GateRXY<fp_type>::Create(9, 3, 0.8, 0.1);
  auto gate22 = GateRZ<fp_type>::Create(9, 4, 0.4);
  auto gate23 = GateCP<fp_type>::Create(10, 0, 1, 0.7);
  auto gate24 = GateY2<fp_type>::Create(11, 3);
  auto gate25 = GateRX<fp_type>::Create(11, 4, 0.3);

  GateFused<Gate<fp_type>> fgate1{kGateCZ, 2, 2, {0, 1}, &gate11,
      {&gate1, &gate2, &gate6, &gate7, &gate11, &gate12, &gate13}};
  ApplyFusedGate(simulator, fgate1, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

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

  GateFused<Gate<fp_type>> fgate2{kGateIS, 4, 2, {1, 2}, &gate14,
      {&gate3, &gate8, &gate14, &gate15, &gate16}};
  ApplyFusedGate(simulator, fgate2, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.17677667, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.17677667, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.24999997, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.35355335, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.42677662, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), -0.07322330, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.25, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), 0, 1e-6);
  }

  GateFused<Gate<fp_type>> fgate3{kGateCNot, 6, 2, {2, 3}, &gate17,
      {&gate4, &gate9, &gate17, &gate18, &gate19}};
  ApplyFusedGate(simulator, fgate3, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.01734632, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.01689983, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.17767739, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.16523364, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.32556468, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), -0.02934359, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.17723089, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.13098748, 1e-6);
  }

  GateFused<Gate<fp_type>> fgate4{kGateFS, 8, 2, {3, 4}, &gate20,
      {&gate5, &gate10, &gate20, &gate21, &gate22}};
  ApplyFusedGate(simulator, fgate4, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.00669215, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.02066205, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.08668970, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.14129037, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.22128790, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.03393859, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.10065960, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.11393616, 1e-6);
  }

  GateFused<Gate<fp_type>> fgate5{kGateCP, 10, 2, {0, 1}, &gate23, {&gate23}};
  ApplyFusedGate(simulator, fgate5, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.00669215, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.02066205, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.08668970, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.14129037, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.22128790, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.03393859, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.00358902, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.15198988, 1e-6);
  }

  ApplyGate(simulator, gate24, state);
  ApplyGate(simulator, gate25, state);

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-12);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), -0.04172631, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.10968970, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), -0.09125633, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), -0.05371386, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.00418384, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), 0.03528048, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.05519247, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), -0.02785729, 1e-6);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
