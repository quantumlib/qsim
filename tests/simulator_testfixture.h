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

#ifndef SIMULATOR_TESTFIXTURE_H_
#define SIMULATOR_TESTFIXTURE_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/expect.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/util_cpu.h"

namespace qsim {

template <typename Factory>
void TestApplyGate1(const Factory& factory) {
  unsigned num_qubits = 1;

  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(num_qubits);
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

template <typename Factory>
void TestApplyGate2(const Factory& factory) {
  unsigned num_qubits = 2;

  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(num_qubits);
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
  auto gate10 = GateSwap<fp_type>::Create(5, 0, 1);

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

  EXPECT_NEAR(state_space.Norm(state), 1, 1e-6);

  {
    auto ampl0 = state_space.GetAmpl(state, 0);
    EXPECT_NEAR(std::real(ampl0), 0.53100818, 1e-6);
    EXPECT_NEAR(std::imag(ampl0), -0.17631586, 1e-6);
    auto ampl1 = state_space.GetAmpl(state, 1);
    EXPECT_NEAR(std::real(ampl1), 0.64307469, 1e-6);
    EXPECT_NEAR(std::imag(ampl1), 0.03410439, 1e-6);
    auto ampl2 = state_space.GetAmpl(state, 2);
    EXPECT_NEAR(std::real(ampl2), -0.32348031, 1e-6);
    EXPECT_NEAR(std::imag(ampl2), -0.11164886, 1e-6);
    auto ampl3 = state_space.GetAmpl(state, 3);
    EXPECT_NEAR(std::real(ampl3), 0.29973805, 1e-6);
    EXPECT_NEAR(std::imag(ampl3), 0.25551257, 1e-6);
  }
}

template <typename Factory>
void TestApplyGate3(const Factory& factory) {
  unsigned num_qubits = 3;

  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(num_qubits);
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

template <typename Factory>
void TestApplyGate5(const Factory& factory) {
  unsigned num_qubits = 5;

  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(num_qubits);
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

  GateFused<GateQSim<fp_type>> fgate1{kGateCZ, 2, {0, 1},  &gate11,
      {&gate1, &gate2, &gate6, &gate7, &gate11, &gate12, &gate13}, {}};
  CalculateFusedMatrix(fgate1);
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

  GateFused<GateQSim<fp_type>> fgate2{kGateIS, 4, {1, 2}, &gate14,
      {&gate3, &gate8, &gate14, &gate15, &gate16}, {}};
  CalculateFusedMatrix(fgate2);
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

  GateFused<GateQSim<fp_type>> fgate3{kGateCNot, 6, {2, 3}, &gate17,
      {&gate4, &gate9, &gate17, &gate18, &gate19},{}};
  CalculateFusedMatrix(fgate3);
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

  GateFused<GateQSim<fp_type>> fgate4{kGateFS, 8, {3, 4}, &gate20,
      {&gate5, &gate10, &gate20, &gate21, &gate22}, {}};
  CalculateFusedMatrix(fgate4);
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

  GateFused<GateQSim<fp_type>> fgate5{kGateCP, 10, {0, 1}, &gate23,
      {&gate23}, {}};
  CalculateFusedMatrix(fgate5);
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

template <typename Factory>
void TestCircuitWithControlledGates(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;
  using Gate = GateQSim<fp_type>;

  unsigned num_qubits = 6;

  std::vector<Gate> gates;
  gates.reserve(128);

  gates.push_back(GateHd<fp_type>::Create(0, 0));
  gates.push_back(GateHd<fp_type>::Create(0, 1));
  gates.push_back(GateHd<fp_type>::Create(0, 2));
  gates.push_back(GateHd<fp_type>::Create(0, 3));
  gates.push_back(GateHd<fp_type>::Create(0, 4));
  gates.push_back(GateHd<fp_type>::Create(0, 5));
  gates.push_back(GateT<fp_type>::Create(1, 1).ControlledBy({0}));
  gates.push_back(GateRX<fp_type>::Create(2, 0, 0.1));
  gates.push_back(GateRY<fp_type>::Create(2, 1, 0.2));
  gates.push_back(GateRZ<fp_type>::Create(2, 2, 0.3));
  gates.push_back(GateRX<fp_type>::Create(2, 3, 0.4));
  gates.push_back(GateRY<fp_type>::Create(2, 4, 0.5));
  gates.push_back(GateRZ<fp_type>::Create(2, 5, 0.6));
  gates.push_back(GateHd<fp_type>::Create(3, 2).ControlledBy({1, 0}, {1, 0}));
  gates.push_back(GateRY<fp_type>::Create(4, 0, 0.7));
  gates.push_back(GateRZ<fp_type>::Create(4, 1, 0.8));
  gates.push_back(GateRX<fp_type>::Create(4, 2, 0.9));
  gates.push_back(GateRY<fp_type>::Create(4, 3, 1.0));
  gates.push_back(GateRZ<fp_type>::Create(4, 4, 1.1));
  gates.push_back(GateRX<fp_type>::Create(4, 5, 1.2));
  gates.push_back(GateT<fp_type>::Create(5, 3).ControlledBy({0, 1, 2}, {1, 1, 0}));
  gates.push_back(GateRZ<fp_type>::Create(6, 0, 1.3));
  gates.push_back(GateRX<fp_type>::Create(6, 1, 1.4));
  gates.push_back(GateRY<fp_type>::Create(6, 2, 1.5));
  gates.push_back(GateRZ<fp_type>::Create(6, 3, 1.6));
  gates.push_back(GateRX<fp_type>::Create(6, 4, 1.7));
  gates.push_back(GateRY<fp_type>::Create(6, 5, 1.8));
  gates.push_back(GateT<fp_type>::Create(7, 4).ControlledBy({0, 2, 3, 1}, {0, 1, 1, 0}));
  gates.push_back(GateRX<fp_type>::Create(8, 0, 1.9));
  gates.push_back(GateRY<fp_type>::Create(8, 1, 2.0));
  gates.push_back(GateRZ<fp_type>::Create(8, 2, 2.1));
  gates.push_back(GateRX<fp_type>::Create(8, 3, 2.2));
  gates.push_back(GateRY<fp_type>::Create(8, 4, 2.3));
  gates.push_back(GateRZ<fp_type>::Create(8, 5, 2.4));
  gates.push_back(GateIS<fp_type>::Create(9, 1, 2).ControlledBy({0}, {0}));
  gates.push_back(GateRY<fp_type>::Create(10, 0, 2.5));
  gates.push_back(GateRZ<fp_type>::Create(10, 1, 2.6));
  gates.push_back(GateRX<fp_type>::Create(10, 2, 2.7));
  gates.push_back(GateRY<fp_type>::Create(10, 3, 2.8));
  gates.push_back(GateRZ<fp_type>::Create(10, 4, 2.9));
  gates.push_back(GateRX<fp_type>::Create(10, 5, 3.0));
  gates.push_back(GateIS<fp_type>::Create(11, 2, 3).ControlledBy({1, 0}));
  gates.push_back(GateRZ<fp_type>::Create(12, 0, 3.1));
  gates.push_back(GateRX<fp_type>::Create(12, 1, 3.2));
  gates.push_back(GateRY<fp_type>::Create(12, 2, 3.3));
  gates.push_back(GateRZ<fp_type>::Create(12, 3, 3.4));
  gates.push_back(GateRX<fp_type>::Create(12, 4, 3.5));
  gates.push_back(GateRY<fp_type>::Create(12, 5, 3.6));
  gates.push_back(GateCNot<fp_type>::Create(13, 3, 4).ControlledBy({0, 2, 1}, {1, 1, 0}));
  gates.push_back(GateRX<fp_type>::Create(14, 0, 3.7));
  gates.push_back(GateRY<fp_type>::Create(14, 1, 3.8));
  gates.push_back(GateRZ<fp_type>::Create(14, 2, 3.9));
  gates.push_back(GateRX<fp_type>::Create(14, 3, 4.0));
  gates.push_back(GateRY<fp_type>::Create(14, 4, 4.1));
  gates.push_back(GateRZ<fp_type>::Create(14, 5, 4.2));
  gates.push_back(GateIS<fp_type>::Create(15, 4, 5).ControlledBy({3, 1, 0, 2}, {1, 1, 0, 0}));
  gates.push_back(GateRY<fp_type>::Create(16, 0, 4.3));
  gates.push_back(GateRZ<fp_type>::Create(16, 1, 4.4));
  gates.push_back(GateRX<fp_type>::Create(16, 2, 4.5));
  gates.push_back(GateRY<fp_type>::Create(16, 3, 4.6));
  gates.push_back(GateRZ<fp_type>::Create(16, 4, 4.7));
  gates.push_back(GateRX<fp_type>::Create(16, 5, 4.8));
  gates.push_back(GateCNot<fp_type>::Create(17, 5, 4).ControlledBy({3}, {0}));
  gates.push_back(GateRZ<fp_type>::Create(18, 0, 4.9));
  gates.push_back(GateRX<fp_type>::Create(18, 1, 5.0));
  gates.push_back(GateRY<fp_type>::Create(18, 2, 5.1));
  gates.push_back(GateRZ<fp_type>::Create(18, 3, 5.2));
  gates.push_back(GateRX<fp_type>::Create(18, 4, 5.3));
  gates.push_back(GateRY<fp_type>::Create(18, 5, 5.4));
  gates.push_back(GateIS<fp_type>::Create(19, 0, 1).ControlledBy({4}));
  gates.push_back(GateRX<fp_type>::Create(20, 0, 5.5));
  gates.push_back(GateRY<fp_type>::Create(20, 1, 5.6));
  gates.push_back(GateRZ<fp_type>::Create(20, 2, 5.7));
  gates.push_back(GateRX<fp_type>::Create(20, 3, 5.8));
  gates.push_back(GateRY<fp_type>::Create(20, 4, 5.9));
  gates.push_back(GateRZ<fp_type>::Create(20, 5, 6.0));
  gates.push_back(GateIS<fp_type>::Create(21, 0, 2).ControlledBy({4}));
  gates.push_back(GateRY<fp_type>::Create(22, 0, 6.1));
  gates.push_back(GateRZ<fp_type>::Create(22, 1, 6.2));
  gates.push_back(GateRX<fp_type>::Create(22, 2, 6.3));
  gates.push_back(GateRY<fp_type>::Create(22, 3, 6.4));
  gates.push_back(GateRZ<fp_type>::Create(22, 4, 6.5));
  gates.push_back(GateRX<fp_type>::Create(22, 5, 6.6));
  gates.push_back(GateIS<fp_type>::Create(23, 0, 5).ControlledBy({4}, {0}));
  gates.push_back(GateRZ<fp_type>::Create(24, 0, 6.7));
  gates.push_back(GateRX<fp_type>::Create(24, 1, 6.8));
  gates.push_back(GateRY<fp_type>::Create(24, 2, 6.9));
  gates.push_back(GateRZ<fp_type>::Create(24, 3, 7.0));
  gates.push_back(GateRX<fp_type>::Create(24, 4, 7.1));
  gates.push_back(GateRY<fp_type>::Create(24, 5, 7.2));
  gates.push_back(GateHd<fp_type>::Create(25, 5).ControlledBy({4}));
  gates.push_back(GateRX<fp_type>::Create(26, 0, 7.3));
  gates.push_back(GateRY<fp_type>::Create(26, 1, 7.4));
  gates.push_back(GateRZ<fp_type>::Create(26, 2, 7.5));
  gates.push_back(GateRX<fp_type>::Create(26, 3, 7.6));
  gates.push_back(GateRY<fp_type>::Create(26, 4, 7.7));
  gates.push_back(GateRZ<fp_type>::Create(26, 5, 7.8));

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state1 = state_space.Create(num_qubits);
  state_space.SetStateZero(state1);

  for (const auto& gate : gates) {
    ApplyGate(simulator, gate, state1);
  }

/*
The results are obtained with the following Cirq code:

import cirq

def main():
  q0 = cirq.LineQubit(5)
  q1 = cirq.LineQubit(4)
  q2 = cirq.LineQubit(3)
  q3 = cirq.LineQubit(2)
  q4 = cirq.LineQubit(1)
  q5 = cirq.LineQubit(0)

  circuit = cirq.Circuit(
    cirq.Moment([
      cirq.H(q0),
      cirq.H(q1),
      cirq.H(q2),
      cirq.H(q3),
      cirq.H(q4),
      cirq.H(q5),
    ]),
    cirq.Moment([
      cirq.T(q1).controlled_by(q0),
    ]),
    cirq.Moment([
       cirq.rx(0.1)(q0),
       cirq.ry(0.2)(q1),
       cirq.rz(0.3)(q2),
       cirq.rx(0.4)(q3),
       cirq.ry(0.5)(q4),
       cirq.rz(0.6)(q5),
    ]),
    cirq.Moment([
      cirq.H(q2).controlled_by(q1, q0, control_values=[1, 0]),
    ]),
    cirq.Moment([
       cirq.ry(0.7)(q0),
       cirq.rz(0.8)(q1),
       cirq.rx(0.9)(q2),
       cirq.ry(1.0)(q3),
       cirq.rz(1.1)(q4),
       cirq.rx(1.2)(q5),
    ]),
    cirq.Moment([
      cirq.T(q3).controlled_by(q0, q1, q2, control_values=[1, 1, 0]),
    ]),
    cirq.Moment([
       cirq.rz(1.3)(q0),
       cirq.rx(1.4)(q1),
       cirq.ry(1.5)(q2),
       cirq.rz(1.6)(q3),
       cirq.rx(1.7)(q4),
       cirq.ry(1.8)(q5),
    ]),
    cirq.Moment([
      cirq.T(q4).controlled_by(q0, q2, q3, q1, control_values=[0, 1, 1, 0]),
    ]),
    cirq.Moment([
       cirq.rx(1.9)(q0),
       cirq.ry(2.0)(q1),
       cirq.rz(2.1)(q2),
       cirq.rx(2.2)(q3),
       cirq.ry(2.3)(q4),
       cirq.rz(2.4)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q1, q2).controlled_by(q0, control_values=[0]),
    ]),
    cirq.Moment([
       cirq.ry(2.5)(q0),
       cirq.rz(2.6)(q1),
       cirq.rx(2.7)(q2),
       cirq.ry(2.8)(q3),
       cirq.rz(2.9)(q4),
       cirq.rx(3.0)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q2, q3).controlled_by(q1, q0),
    ]),
    cirq.Moment([
       cirq.rz(3.1)(q0),
       cirq.rx(3.2)(q1),
       cirq.ry(3.3)(q2),
       cirq.rz(3.4)(q3),
       cirq.rx(3.5)(q4),
       cirq.ry(3.6)(q5),
    ]),
    cirq.Moment([
      cirq.CNOT(q3, q4).controlled_by(q0, q1, q2, control_values=[1, 0, 1]),
    ]),
    cirq.Moment([
       cirq.rx(3.7)(q0),
       cirq.ry(3.8)(q1),
       cirq.rz(3.9)(q2),
       cirq.rx(4.0)(q3),
       cirq.ry(4.1)(q4),
       cirq.rz(4.2)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q4, q5).controlled_by(q3, q1, q0, q2,
                                       control_values=[1, 1, 0, 0]),
    ]),
    cirq.Moment([
       cirq.ry(4.3)(q0),
       cirq.rz(4.4)(q1),
       cirq.rx(4.5)(q2),
       cirq.ry(4.6)(q3),
       cirq.rz(4.7)(q4),
       cirq.rx(4.8)(q5),
    ]),
    cirq.Moment([
      cirq.CNOT(q5, q4).controlled_by(q3, control_values=[0]),
    ]),
    cirq.Moment([
       cirq.rz(4.9)(q0),
       cirq.rx(5.0)(q1),
       cirq.ry(5.1)(q2),
       cirq.rz(5.2)(q3),
       cirq.rx(5.3)(q4),
       cirq.ry(5.4)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q1).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.rx(5.5)(q0),
       cirq.ry(5.6)(q1),
       cirq.rz(5.7)(q2),
       cirq.rx(5.8)(q3),
       cirq.ry(5.9)(q4),
       cirq.rz(6.0)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q2).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.ry(6.1)(q0),
       cirq.rz(6.2)(q1),
       cirq.rx(6.3)(q2),
       cirq.ry(6.4)(q3),
       cirq.rz(6.5)(q4),
       cirq.rx(6.6)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q5).controlled_by(q4, control_values=[0]),
    ]),
    cirq.Moment([
       cirq.rz(6.7)(q0),
       cirq.rx(6.8)(q1),
       cirq.ry(6.9)(q2),
       cirq.rz(7.0)(q3),
       cirq.rx(7.1)(q4),
       cirq.ry(7.2)(q5),
    ]),
    cirq.Moment([
      cirq.H(q5).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.rx(7.3)(q0),
       cirq.ry(7.4)(q1),
       cirq.rz(7.5)(q2),
       cirq.rx(7.6)(q3),
       cirq.ry(7.7)(q4),
       cirq.rz(7.8)(q5),
    ]),
  )

  simulator = cirq.Simulator()
  result = simulator.simulate(circuit)

  for i in range(len(result.state_vector())):
    print(i, result.state_vector()[i])


if __name__ == '__main__':
  main()

*/

  std::vector<std::vector<fp_type>> expected_results = {
    {0.04056215, 0.11448385},
    {0.04013729, -0.061976265},
    {0.05715254, 0.06587616},
    {-0.0999089, -0.0551068},
    {-0.020135913, -0.0017108098},
    {-0.056598634, -0.011147065},
    {-0.05639626, 0.09074731},
    {-0.057448477, 0.040516872},
    {0.0344304, 0.016834},
    {-0.0556134, -0.006876275},
    {-0.036210306, -0.045713138},
    {0.106739536, 0.04557059},
    {0.0042791665, 0.071074575},
    {-0.025317883, 0.06527158},
    {0.003052316, -0.002724175},
    {-0.027759908, 0.082198195},
    {-0.10696569, 0.009430081},
    {-0.03781139, 0.11874371},
    {-0.020180658, -0.07570377},
    {0.05576851, -0.022236263},
    {-0.06552034, 0.058305625},
    {-0.0484216, -0.1268896},
    {-0.088334806, -0.2118823},
    {-0.058212772, -0.10756658},
    {0.06811757, -0.10867228},
    {-0.006912032, -0.056490533},
    {0.14454205, -0.08358974},
    {0.09103435, 0.15097837},
    {-0.023433153, -0.11143835},
    {0.019963266, -0.0008750437},
    {0.25689512, -0.13761702},
    {0.060466085, -0.083674595},
    {-0.10356863, -0.031856094},
    {0.05267005, -0.040480673},
    {0.0017506611, -0.057084523},
    {-0.049090747, 0.0076575093},
    {-0.05804465, 0.048070334},
    {-0.037869103, 0.007335903},
    {-0.13274089, -0.1556583},
    {-0.013423506, -0.10376227},
    {0.063333265, -0.20126863},
    {-0.1259143, 0.07443194},
    {0.13821091, 0.045418564},
    {0.034076303, 0.054569334},
    {-0.09922538, -0.09469399},
    {0.09066829, -0.064125836},
    {0.235489, -0.19617496},
    {0.15996316, -0.036261443},
    {-0.02887804, -0.047851864},
    {0.046452887, -0.05820565},
    {0.015137469, 0.07583993},
    {-0.09476741, -0.054346137},
    {0.015158612, 0.08472719},
    {-0.03694186, 0.0070148334},
    {-0.025821798, 0.08404015},
    {0.061565418, -0.012411967},
    {-0.078881726, 0.12779479},
    {-0.05464944, 0.056015424},
    {-0.16184065, -0.009010859},
    {0.12749553, -0.12438276},
    {0.019615382, 0.092316},
    {-0.04924332, 0.044155773},
    {-0.24133444, -0.033628717},
    {-0.18774915, 0.12311842},
  };

  unsigned size = 1 << num_qubits;

  for (unsigned i = 0; i < size; ++i) {
    auto a = StateSpace::GetAmpl(state1, i);
    EXPECT_NEAR(std::real(a), expected_results[i][0], 1e-6);
    EXPECT_NEAR(std::imag(a), expected_results[i][1], 1e-6);
  }

  SetFlushToZeroAndDenormalsAreZeros();

  auto state2 = state_space.Create(num_qubits);
  state_space.SetStateZero(state2);

  for (const auto& gate : gates) {
    ApplyGate(simulator, gate, state2);
  }

  for (unsigned i = 0; i < size; ++i) {
    auto a1 = StateSpace::GetAmpl(state1, i);
    auto a2 = StateSpace::GetAmpl(state2, i);
    EXPECT_EQ(std::real(a1), std::real(a2));
    EXPECT_EQ(std::imag(a1), std::imag(a2));
  }

  ClearFlushToZeroAndDenormalsAreZeros();

  auto state3 = state_space.Create(num_qubits);
  state_space.SetStateZero(state3);

  for (const auto& gate : gates) {
    ApplyGate(simulator, gate, state3);
  }

  for (unsigned i = 0; i < size; ++i) {
    auto a1 = StateSpace::GetAmpl(state1, i);
    auto a2 = StateSpace::GetAmpl(state3, i);
    EXPECT_EQ(std::real(a1), std::real(a2));
    EXPECT_EQ(std::imag(a1), std::imag(a2));
  }
}

template <typename Factory>
void TestCircuitWithControlledGatesDagger(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;
  using Gate = GateQSim<fp_type>;

  unsigned num_qubits = 6;
  unsigned size = 1 << num_qubits;

  std::vector<Gate> gates;
  gates.reserve(128);

  gates.push_back(GateHd<fp_type>::Create(0, 0));
  gates.push_back(GateHd<fp_type>::Create(0, 1));
  gates.push_back(GateHd<fp_type>::Create(0, 2));
  gates.push_back(GateHd<fp_type>::Create(0, 3));
  gates.push_back(GateHd<fp_type>::Create(0, 4));
  gates.push_back(GateHd<fp_type>::Create(0, 5));
  gates.push_back(GateT<fp_type>::Create(1, 1).ControlledBy({0}));
  gates.push_back(GateRX<fp_type>::Create(2, 0, 0.1));
  gates.push_back(GateRY<fp_type>::Create(2, 1, 0.2));
  gates.push_back(GateRZ<fp_type>::Create(2, 2, 0.3));
  gates.push_back(GateRX<fp_type>::Create(2, 3, 0.4));
  gates.push_back(GateRY<fp_type>::Create(2, 4, 0.5));
  gates.push_back(GateRZ<fp_type>::Create(2, 5, 0.6));
  gates.push_back(GateHd<fp_type>::Create(3, 2).ControlledBy({1, 0}, {1, 0}));
  gates.push_back(GateRY<fp_type>::Create(4, 0, 0.7));
  gates.push_back(GateRZ<fp_type>::Create(4, 1, 0.8));
  gates.push_back(GateRX<fp_type>::Create(4, 2, 0.9));
  gates.push_back(GateRY<fp_type>::Create(4, 3, 1.0));
  gates.push_back(GateRZ<fp_type>::Create(4, 4, 1.1));
  gates.push_back(GateRX<fp_type>::Create(4, 5, 1.2));
  gates.push_back(GateT<fp_type>::Create(5, 3).ControlledBy({0, 1, 2}, {1, 1, 0}));
  gates.push_back(GateRZ<fp_type>::Create(6, 0, 1.3));
  gates.push_back(GateRX<fp_type>::Create(6, 1, 1.4));
  gates.push_back(GateRY<fp_type>::Create(6, 2, 1.5));
  gates.push_back(GateRZ<fp_type>::Create(6, 3, 1.6));
  gates.push_back(GateRX<fp_type>::Create(6, 4, 1.7));
  gates.push_back(GateRY<fp_type>::Create(6, 5, 1.8));
  gates.push_back(GateT<fp_type>::Create(7, 4).ControlledBy({0, 2, 3, 1}, {0, 1, 1, 0}));
  gates.push_back(GateRX<fp_type>::Create(8, 0, 1.9));
  gates.push_back(GateRY<fp_type>::Create(8, 1, 2.0));
  gates.push_back(GateRZ<fp_type>::Create(8, 2, 2.1));
  gates.push_back(GateRX<fp_type>::Create(8, 3, 2.2));
  gates.push_back(GateRY<fp_type>::Create(8, 4, 2.3));
  gates.push_back(GateRZ<fp_type>::Create(8, 5, 2.4));
  gates.push_back(GateIS<fp_type>::Create(9, 1, 2).ControlledBy({0}, {0}));
  gates.push_back(GateRY<fp_type>::Create(10, 0, 2.5));
  gates.push_back(GateRZ<fp_type>::Create(10, 1, 2.6));
  gates.push_back(GateRX<fp_type>::Create(10, 2, 2.7));
  gates.push_back(GateRY<fp_type>::Create(10, 3, 2.8));
  gates.push_back(GateRZ<fp_type>::Create(10, 4, 2.9));
  gates.push_back(GateRX<fp_type>::Create(10, 5, 3.0));
  gates.push_back(GateIS<fp_type>::Create(11, 2, 3).ControlledBy({1, 0}));
  gates.push_back(GateRZ<fp_type>::Create(12, 0, 3.1));
  gates.push_back(GateRX<fp_type>::Create(12, 1, 3.2));
  gates.push_back(GateRY<fp_type>::Create(12, 2, 3.3));
  gates.push_back(GateRZ<fp_type>::Create(12, 3, 3.4));
  gates.push_back(GateRX<fp_type>::Create(12, 4, 3.5));
  gates.push_back(GateRY<fp_type>::Create(12, 5, 3.6));
  gates.push_back(GateCNot<fp_type>::Create(13, 3, 4).ControlledBy({0, 2, 1}, {1, 1, 0}));
  gates.push_back(GateRX<fp_type>::Create(14, 0, 3.7));
  gates.push_back(GateRY<fp_type>::Create(14, 1, 3.8));
  gates.push_back(GateRZ<fp_type>::Create(14, 2, 3.9));
  gates.push_back(GateRX<fp_type>::Create(14, 3, 4.0));
  gates.push_back(GateRY<fp_type>::Create(14, 4, 4.1));
  gates.push_back(GateRZ<fp_type>::Create(14, 5, 4.2));
  gates.push_back(GateIS<fp_type>::Create(15, 4, 5).ControlledBy({3, 1, 0, 2}, {1, 1, 0, 0}));
  gates.push_back(GateRY<fp_type>::Create(16, 0, 4.3));
  gates.push_back(GateRZ<fp_type>::Create(16, 1, 4.4));
  gates.push_back(GateRX<fp_type>::Create(16, 2, 4.5));
  gates.push_back(GateRY<fp_type>::Create(16, 3, 4.6));
  gates.push_back(GateRZ<fp_type>::Create(16, 4, 4.7));
  gates.push_back(GateRX<fp_type>::Create(16, 5, 4.8));
  gates.push_back(GateCNot<fp_type>::Create(17, 5, 4).ControlledBy({3}, {0}));
  gates.push_back(GateRZ<fp_type>::Create(18, 0, 4.9));
  gates.push_back(GateRX<fp_type>::Create(18, 1, 5.0));
  gates.push_back(GateRY<fp_type>::Create(18, 2, 5.1));
  gates.push_back(GateRZ<fp_type>::Create(18, 3, 5.2));
  gates.push_back(GateRX<fp_type>::Create(18, 4, 5.3));
  gates.push_back(GateRY<fp_type>::Create(18, 5, 5.4));
  gates.push_back(GateIS<fp_type>::Create(19, 0, 1).ControlledBy({4}));
  gates.push_back(GateRX<fp_type>::Create(20, 0, 5.5));
  gates.push_back(GateRY<fp_type>::Create(20, 1, 5.6));
  gates.push_back(GateRZ<fp_type>::Create(20, 2, 5.7));
  gates.push_back(GateRX<fp_type>::Create(20, 3, 5.8));
  gates.push_back(GateRY<fp_type>::Create(20, 4, 5.9));
  gates.push_back(GateRZ<fp_type>::Create(20, 5, 6.0));
  gates.push_back(GateIS<fp_type>::Create(21, 0, 2).ControlledBy({4}));
  gates.push_back(GateRY<fp_type>::Create(22, 0, 6.1));
  gates.push_back(GateRZ<fp_type>::Create(22, 1, 6.2));
  gates.push_back(GateRX<fp_type>::Create(22, 2, 6.3));
  gates.push_back(GateRY<fp_type>::Create(22, 3, 6.4));
  gates.push_back(GateRZ<fp_type>::Create(22, 4, 6.5));
  gates.push_back(GateRX<fp_type>::Create(22, 5, 6.6));
  gates.push_back(GateIS<fp_type>::Create(23, 0, 5).ControlledBy({4}, {0}));
  gates.push_back(GateRZ<fp_type>::Create(24, 0, 6.7));
  gates.push_back(GateRX<fp_type>::Create(24, 1, 6.8));
  gates.push_back(GateRY<fp_type>::Create(24, 2, 6.9));
  gates.push_back(GateRZ<fp_type>::Create(24, 3, 7.0));
  gates.push_back(GateRX<fp_type>::Create(24, 4, 7.1));
  gates.push_back(GateRY<fp_type>::Create(24, 5, 7.2));
  gates.push_back(GateHd<fp_type>::Create(25, 5).ControlledBy({4}));
  gates.push_back(GateRX<fp_type>::Create(26, 0, 7.3));
  gates.push_back(GateRY<fp_type>::Create(26, 1, 7.4));
  gates.push_back(GateRZ<fp_type>::Create(26, 2, 7.5));
  gates.push_back(GateRX<fp_type>::Create(26, 3, 7.6));
  gates.push_back(GateRY<fp_type>::Create(26, 4, 7.7));
  gates.push_back(GateRZ<fp_type>::Create(26, 5, 7.8));

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(num_qubits);
  std::vector<std::vector<fp_type>> final_amplitudes = {
    {0.04056215, 0.11448385},
    {0.04013729, -0.061976265},
    {0.05715254, 0.06587616},
    {-0.0999089, -0.0551068},
    {-0.020135913, -0.0017108098},
    {-0.056598634, -0.011147065},
    {-0.05639626, 0.09074731},
    {-0.057448477, 0.040516872},
    {0.0344304, 0.016834},
    {-0.0556134, -0.006876275},
    {-0.036210306, -0.045713138},
    {0.106739536, 0.04557059},
    {0.0042791665, 0.071074575},
    {-0.025317883, 0.06527158},
    {0.003052316, -0.002724175},
    {-0.027759908, 0.082198195},
    {-0.10696569, 0.009430081},
    {-0.03781139, 0.11874371},
    {-0.020180658, -0.07570377},
    {0.05576851, -0.022236263},
    {-0.06552034, 0.058305625},
    {-0.0484216, -0.1268896},
    {-0.088334806, -0.2118823},
    {-0.058212772, -0.10756658},
    {0.06811757, -0.10867228},
    {-0.006912032, -0.056490533},
    {0.14454205, -0.08358974},
    {0.09103435, 0.15097837},
    {-0.023433153, -0.11143835},
    {0.019963266, -0.0008750437},
    {0.25689512, -0.13761702},
    {0.060466085, -0.083674595},
    {-0.10356863, -0.031856094},
    {0.05267005, -0.040480673},
    {0.0017506611, -0.057084523},
    {-0.049090747, 0.0076575093},
    {-0.05804465, 0.048070334},
    {-0.037869103, 0.007335903},
    {-0.13274089, -0.1556583},
    {-0.013423506, -0.10376227},
    {0.063333265, -0.20126863},
    {-0.1259143, 0.07443194},
    {0.13821091, 0.045418564},
    {0.034076303, 0.054569334},
    {-0.09922538, -0.09469399},
    {0.09066829, -0.064125836},
    {0.235489, -0.19617496},
    {0.15996316, -0.036261443},
    {-0.02887804, -0.047851864},
    {0.046452887, -0.05820565},
    {0.015137469, 0.07583993},
    {-0.09476741, -0.054346137},
    {0.015158612, 0.08472719},
    {-0.03694186, 0.0070148334},
    {-0.025821798, 0.08404015},
    {0.061565418, -0.012411967},
    {-0.078881726, 0.12779479},
    {-0.05464944, 0.056015424},
    {-0.16184065, -0.009010859},
    {0.12749553, -0.12438276},
    {0.019615382, 0.092316},
    {-0.04924332, 0.044155773},
    {-0.24133444, -0.033628717},
    {-0.18774915, 0.12311842},
  };

  for (unsigned i = 0; i < size; ++i) {
    state_space.SetAmpl(state, i, final_amplitudes[i][0], final_amplitudes[i][1]);
  }

  for (int i = gates.size() - 1; i >= 0; --i) {
    ApplyGateDagger(simulator, gates[i], state);
  }

/*
The results are obtained with the following Cirq code:

import cirq

def main():
  q0 = cirq.LineQubit(5)
  q1 = cirq.LineQubit(4)
  q2 = cirq.LineQubit(3)
  q3 = cirq.LineQubit(2)
  q4 = cirq.LineQubit(1)
  q5 = cirq.LineQubit(0)

  circuit = cirq.Circuit(
    cirq.Moment([
      cirq.H(q0),
      cirq.H(q1),
      cirq.H(q2),
      cirq.H(q3),
      cirq.H(q4),
      cirq.H(q5),
    ]),
    cirq.Moment([
      cirq.T(q1).controlled_by(q0),
    ]),
    cirq.Moment([
       cirq.rx(0.1)(q0),
       cirq.ry(0.2)(q1),
       cirq.rz(0.3)(q2),
       cirq.rx(0.4)(q3),
       cirq.ry(0.5)(q4),
       cirq.rz(0.6)(q5),
    ]),
    cirq.Moment([
      cirq.H(q2).controlled_by(q1, q0, control_values=[1, 0]),
    ]),
    cirq.Moment([
       cirq.ry(0.7)(q0),
       cirq.rz(0.8)(q1),
       cirq.rx(0.9)(q2),
       cirq.ry(1.0)(q3),
       cirq.rz(1.1)(q4),
       cirq.rx(1.2)(q5),
    ]),
    cirq.Moment([
      cirq.T(q3).controlled_by(q0, q1, q2, control_values=[1, 1, 0]),
    ]),
    cirq.Moment([
       cirq.rz(1.3)(q0),
       cirq.rx(1.4)(q1),
       cirq.ry(1.5)(q2),
       cirq.rz(1.6)(q3),
       cirq.rx(1.7)(q4),
       cirq.ry(1.8)(q5),
    ]),
    cirq.Moment([
      cirq.T(q4).controlled_by(q0, q2, q3, q1, control_values=[0, 1, 1, 0]),
    ]),
    cirq.Moment([
       cirq.rx(1.9)(q0),
       cirq.ry(2.0)(q1),
       cirq.rz(2.1)(q2),
       cirq.rx(2.2)(q3),
       cirq.ry(2.3)(q4),
       cirq.rz(2.4)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q1, q2).controlled_by(q0, control_values=[0]),
    ]),
    cirq.Moment([
       cirq.ry(2.5)(q0),
       cirq.rz(2.6)(q1),
       cirq.rx(2.7)(q2),
       cirq.ry(2.8)(q3),
       cirq.rz(2.9)(q4),
       cirq.rx(3.0)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q2, q3).controlled_by(q1, q0),
    ]),
    cirq.Moment([
       cirq.rz(3.1)(q0),
       cirq.rx(3.2)(q1),
       cirq.ry(3.3)(q2),
       cirq.rz(3.4)(q3),
       cirq.rx(3.5)(q4),
       cirq.ry(3.6)(q5),
    ]),
    cirq.Moment([
      cirq.CNOT(q3, q4).controlled_by(q0, q1, q2, control_values=[1, 0, 1]),
    ]),
    cirq.Moment([
       cirq.rx(3.7)(q0),
       cirq.ry(3.8)(q1),
       cirq.rz(3.9)(q2),
       cirq.rx(4.0)(q3),
       cirq.ry(4.1)(q4),
       cirq.rz(4.2)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q4, q5).controlled_by(q3, q1, q0, q2,
                                       control_values=[1, 1, 0, 0]),
    ]),
    cirq.Moment([
       cirq.ry(4.3)(q0),
       cirq.rz(4.4)(q1),
       cirq.rx(4.5)(q2),
       cirq.ry(4.6)(q3),
       cirq.rz(4.7)(q4),
       cirq.rx(4.8)(q5),
    ]),
    cirq.Moment([
      cirq.CNOT(q5, q4).controlled_by(q3, control_values=[0]),
    ]),
    cirq.Moment([
       cirq.rz(4.9)(q0),
       cirq.rx(5.0)(q1),
       cirq.ry(5.1)(q2),
       cirq.rz(5.2)(q3),
       cirq.rx(5.3)(q4),
       cirq.ry(5.4)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q1).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.rx(5.5)(q0),
       cirq.ry(5.6)(q1),
       cirq.rz(5.7)(q2),
       cirq.rx(5.8)(q3),
       cirq.ry(5.9)(q4),
       cirq.rz(6.0)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q2).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.ry(6.1)(q0),
       cirq.rz(6.2)(q1),
       cirq.rx(6.3)(q2),
       cirq.ry(6.4)(q3),
       cirq.rz(6.5)(q4),
       cirq.rx(6.6)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q5).controlled_by(q4, control_values=[0]),
    ]),
    cirq.Moment([
       cirq.rz(6.7)(q0),
       cirq.rx(6.8)(q1),
       cirq.ry(6.9)(q2),
       cirq.rz(7.0)(q3),
       cirq.rx(7.1)(q4),
       cirq.ry(7.2)(q5),
    ]),
    cirq.Moment([
      cirq.H(q5).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.rx(7.3)(q0),
       cirq.ry(7.4)(q1),
       cirq.rz(7.5)(q2),
       cirq.rx(7.6)(q3),
       cirq.ry(7.7)(q4),
       cirq.rz(7.8)(q5),
    ]),
  )

  simulator = cirq.Simulator()
  result = simulator.simulate(circuit)

  for i in range(len(result.state_vector())):
    print(i, result.state_vector()[i])


if __name__ == '__main__':
  main()

*/

  EXPECT_NEAR(std::real(StateSpace::GetAmpl(state, 0)), 1, 1e-6);
  EXPECT_NEAR(std::imag(StateSpace::GetAmpl(state, 0)), 0, 1e-6);
  for (unsigned i = 1; i < size; ++i) {
    auto a = StateSpace::GetAmpl(state, i);
    EXPECT_NEAR(std::real(a), 0, 1e-6);
    EXPECT_NEAR(std::imag(a), 0, 1e-6);
  }
}

template <typename Factory>
void TestMultiQubitGates(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  unsigned max_minq = 4;
  unsigned max_gate_qubits = 6;
  unsigned num_qubits = max_gate_qubits + max_minq;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(num_qubits);

  std::vector<fp_type> matrix;
  matrix.reserve(1 << (2 * max_gate_qubits + 1));

  std::vector<unsigned> qubits;
  qubits.reserve(max_gate_qubits);

  std::vector<fp_type> vec(state_space.MinSize(num_qubits));

  unsigned size = 1 << num_qubits;
  fp_type inorm = std::sqrt(1.0 / (1 << num_qubits));

  for (unsigned q = 1; q <= max_gate_qubits; ++q) {
    unsigned size1 = 1 << q;
    unsigned size2 = size1 * size1;

    matrix.resize(0);

    for (unsigned i = 0; i < 2 * size2; ++i) {
      matrix.push_back(i + 1);
    }

    unsigned mask = (1 << q) - 1;

    for (unsigned k = 0; k <= max_minq; ++k) {
      qubits.resize(0);

      for (unsigned i = 0; i < q; ++i) {
        qubits.push_back(i + k);
      }

      state_space.SetStateUniform(state);
      simulator.ApplyGate(qubits, matrix.data(), state);

      state_space.InternalToNormalOrder(state);
      state_space.Copy(state, vec.data());

      for (unsigned i = 0; i < size; ++i) {
        unsigned j = (i >> k) & mask;

        // Expected results are calculated analytically.
        fp_type expected_real = size2 * (1 + 2 * j) * inorm;
        fp_type expected_imag = expected_real + size1 * inorm;

        EXPECT_NEAR(vec[2 * i], expected_real, 1e-6);
        EXPECT_NEAR(vec[2 * i + 1], expected_imag, 1e-6);
      }
    }
  }
}

template <typename Factory>
void TestControlledGates(const Factory& factory, bool high_precision) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  unsigned max_qubits = 5 + std::log2(Simulator::SIMDRegisterSize());
  unsigned max_target_qubits = 4;
  unsigned max_control_qubits = 3;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(max_qubits);

  std::vector<unsigned> qubits;
  qubits.reserve(max_qubits);

  std::vector<unsigned> cqubits;
  cqubits.reserve(max_qubits);

  std::vector<fp_type> matrix;
  matrix.reserve(1 << (2 * max_target_qubits + 1));

  std::vector<fp_type> vec(state_space.MinSize(max_qubits));

  // Iterate over circuit size.
  for (unsigned num_qubits = 2; num_qubits <= max_qubits; ++num_qubits) {
    unsigned size = 1 << num_qubits;
    unsigned nmask = size - 1;

    // Iterate over control qubits (as a binary mask).
    for (unsigned cmask = 0; cmask <= nmask; ++cmask) {
      cqubits.resize(0);

      for (unsigned q = 0; q < num_qubits; ++q) {
        if (((cmask >> q) & 1) != 0) {
          cqubits.push_back(q);
        }
      }

      if (cqubits.size() == 0
          || cqubits.size() > std::min(max_control_qubits, num_qubits - 1)) {
        continue;
      }

      // Iterate over target qubits (as a binary mask).
      for (unsigned mask = 0; mask <= nmask; ++mask) {
        unsigned qmask = mask & (cmask ^ nmask);

        qubits.resize(0);

        for (unsigned q = 0; q < num_qubits; ++q) {
          if (((qmask >> q) & 1) > 0) {
            qubits.push_back(q);
          }
        }

        if (cmask != (mask & cmask)) continue;

        unsigned num_available = num_qubits - cqubits.size();
        if (qubits.size() == 0
            || qubits.size() > std::min(max_target_qubits, num_available)) {
          continue;
        }

        // Target qubits are consecuitive.
        std::size_t i = 1;
        for (; i < qubits.size(); ++i) {
          if (qubits[i - 1] + 1 != qubits[i]) break;
        }
        if (i < qubits.size()) continue;

        unsigned k = qubits[0];

        unsigned size1 = 1 << qubits.size();
        unsigned size2 = size1 * size1;

        matrix.resize(0);
        // Non-unitary gate matrix.
        for (unsigned i = 0; i < 2 * size2; ++i) {
          matrix.push_back(i + 1);
        }

        unsigned zmask = nmask ^ qmask;

        // Iterate over control values (all zeros or all ones).
        std::vector<unsigned> cvals = {0, (1U << cqubits.size()) - 1};
        for (unsigned cval : cvals) {
          unsigned cvmask = cval == 0 ? 0 : cmask;

          // Starting state.
          for (unsigned j = 0; j < size; ++j) {
            auto val = 2 * j;
            vec[2 * j] = val;
            vec[2 * j + 1] = val + 1;
          }

          state_space.Copy(vec.data(), state);
          state_space.NormalToInternalOrder(state);

          simulator.ApplyControlledGate(
              qubits, cqubits, cval, matrix.data(), state);

          state_space.InternalToNormalOrder(state);
          state_space.Copy(state, vec.data());

          // Test results.
          for (unsigned j = 0; j < size; ++j) {
            if ((j & cmask) == cvmask) {
              // The target matrix is applied.

              unsigned s = j & zmask;
              unsigned l = (j ^ s) >> k;

              // Expected results are calculated analytically.
              fp_type expected_real =
                  -fp_type(2 * size2 * l + size1 * (2 * s + 2)
                           + (1 + (1 << k)) * (size2 - size1));
              fp_type expected_imag = -expected_real - size1
                  + 2 * (size1 * (1 << k) * (size1 - 1)
                               * (1 + 2 * size1 * (2 + 3 * l))
                         + 6 * size2 * (1 + 2 * l) * s) / 3;

              if (high_precision) {
                EXPECT_NEAR(vec[2 * j], expected_real, 1e-6);
                EXPECT_NEAR(vec[2 * j + 1], expected_imag, 1e-6);
              } else {
                EXPECT_NEAR(vec[2 * j] / expected_real, 1.0, 1e-6);
                EXPECT_NEAR(vec[2 * j + 1] / expected_imag, 1.0, 1e-6);
              }
            } else {
              // The target matrix is not applied. Unmodified entries.

              fp_type expected_real = 2 * j;
              fp_type expected_imag = expected_real + 1;

              EXPECT_NEAR(vec[2 * j], expected_real, 1e-6);
              EXPECT_NEAR(vec[2 * j + 1], expected_imag, 1e-6);
            }
          }
        }
      }
    }
  }
}

template <typename Factory>
void TestExpectationValue1(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  unsigned max_minq = 4;
  unsigned max_gate_qubits = 6;
  unsigned num_qubits = max_gate_qubits + max_minq;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  auto state = state_space.Create(num_qubits);

  std::vector<fp_type> matrix;
  matrix.reserve(1 << (2 * max_gate_qubits + 1));

  std::vector<unsigned> qubits;
  qubits.reserve(max_gate_qubits);

  for (unsigned q = 1; q <= max_gate_qubits; ++q) {
    unsigned size1 = 1 << q;
    unsigned size2 = size1 * size1;

    // Expected results are calculated analytically.
    fp_type expected_real = size2 * size1;
    fp_type expected_imag = expected_real + size1;

    matrix.resize(0);

    for (unsigned i = 0; i < 2 * size2; ++i) {
      matrix.push_back(i + 1);
    }

    for (unsigned k = 0; k <= max_minq; ++k) {
      qubits.resize(0);

      for (unsigned i = 0; i < q; ++i) {
        qubits.push_back(i + k);
      }

      state_space.SetStateUniform(state);
      auto eval = simulator.ExpectationValue(qubits, matrix.data(), state);

      EXPECT_NEAR(std::real(eval), expected_real, 1e-6);
      EXPECT_NEAR(std::imag(eval), expected_imag, 1e-6);
    }
  }
}

template <typename Factory>
void TestExpectationValue2(const Factory& factory) {
  using Simulator = typename Factory::Simulator;
  using StateSpace = typename Simulator::StateSpace;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;
  using Fuser = MultiQubitGateFuser<IO, GateQSim<fp_type>>;

  unsigned num_qubits = 16;
  unsigned depth = 16;

  StateSpace state_space = factory.CreateStateSpace();
  Simulator simulator = factory.CreateSimulator();

  State state = state_space.Create(num_qubits);
  state_space.SetStateZero(state);

  std::vector<GateQSim<fp_type>> circuit;
  circuit.reserve(num_qubits * (3 * depth / 2 + 1));

  for (unsigned k = 0; k < num_qubits; ++k) {
    circuit.push_back(GateHd<fp_type>::Create(0, k));
  }

  unsigned t = 1;

  for (unsigned i = 0; i < depth / 2; ++i) {
    for (unsigned k = 0; k < num_qubits; ++k) {
      circuit.push_back(GateRX<fp_type>::Create(t, k, 0.1 * k));
    }

    ++t;

    for (unsigned k = 0; k < num_qubits / 2; ++k) {
      circuit.push_back(GateIS<fp_type>::Create(t, 2 * k, 2 * k + 1));
    }

    ++t;

    for (unsigned k = 0; k < num_qubits; ++k) {
      circuit.push_back(GateRY<fp_type>::Create(t, k, 0.1 * k));
    }

    ++t;

    for (unsigned k = 0; k < num_qubits / 2; ++k) {
      circuit.push_back(GateIS<fp_type>::Create(
          t, 2 * k + 1, (2 * k + 2) % num_qubits));
    }

    ++t;
  }

  for (const auto& gate : circuit) {
    ApplyGate(simulator, gate, state);
  }

/*

The expected results are obtained with the following Cirq code.

import cirq

num_qubits = 16
depth = 16

num_qubits2 = num_qubits // 2

qubits = cirq.LineQubit.range(num_qubits)
qubits.reverse()

circuit = cirq.Circuit()

gates = [cirq.H(qubits[k]) for k in range(num_qubits)]
circuit.append(cirq.Moment(gates))

for i in range(depth // 2):
  gates = [cirq.rx(0.1 * k)(qubits[k]) for k in range(num_qubits)]
  circuit.append(cirq.Moment(gates))

  gates = [cirq.ISWAP(qubits[2 * k], qubits[2 * k + 1])
           for k in range(num_qubits2)]
  circuit.append(cirq.Moment(gates))

  gates = [cirq.ry(0.1 * k)(qubits[k]) for k in range(len(qubits))]
  circuit.append(cirq.Moment(gates))

  gates = [cirq.ISWAP(qubits[2 * k + 1], qubits[(2 * k + 2) % num_qubits])
           for k in range(num_qubits2)]
  circuit.append(cirq.Moment(gates))

simulator = cirq.Simulator()
results = simulator.simulate(circuit)

state_vector = results.state_vector()

qubit_map = {qubits[k]: num_qubits - 1 - k for k in range(num_qubits)}

def op(j):
  return [cirq.X, cirq.Y, cirq.Z][j % 3]

for k in range(1, 7):
  ps = [cirq.PauliString(0.1 + 0.2 * i, [op(j)(qubits[i + j])
                                         for j in range(k)])
        for i in range(num_qubits - k + 1)]
  p = cirq.PauliSum.from_pauli_strings(ps);

  expectation = p.expectation_from_state_vector(state_vector, qubit_map)
  print(expectation)

*/

  fp_type expected_real[6] = {
    0.014314421865856278,
    0.021889885055134076,
    -0.006954622792545706,
    0.013091871136566622,
    0.004322795104235413,
    -0.008040613483171907,
  };

  for (unsigned k = 1; k <= 6; ++k) {
    std::vector<OpString<GateQSim<fp_type>>> strings;
    strings.reserve(num_qubits);

    for (unsigned i = 0; i <= num_qubits - k; ++i) {
      strings.push_back({{0.1 + 0.2 * i, 0}, {}});

      strings.back().ops.reserve(k);

      for (unsigned j = 0; j < k; ++j) {
        switch (j % 3) {
        case 0:
          strings.back().ops.push_back(GateX<fp_type>::Create(0, i + j));
          break;
        case 1:
          strings.back().ops.push_back(GateY<fp_type>::Create(0, i + j));
          break;
        case 2:
          strings.back().ops.push_back(GateZ<fp_type>::Create(0, i + j));
          break;
        }
      }
    }

    auto eval = ExpectationValue<IO, Fuser>(strings, simulator, state);

    EXPECT_NEAR(std::real(eval), expected_real[k - 1], 1e-6);
    EXPECT_NEAR(std::imag(eval), 0, 1e-8);
  }
}

}  // namespace qsim

#endif  // SIMULATOR_TESTFIXTURE_H_
