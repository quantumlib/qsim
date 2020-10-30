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

#include <complex>
#include <vector>

#include "gtest/gtest.h"

#include "../lib/fuser.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_qsim.h"

namespace qsim {

template <typename Simulator>
void TestApplyGate1() {
  unsigned num_qubits = 1;
  unsigned num_threads = 1;

  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space(num_threads);
  Simulator simulator(num_threads);

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

template <typename Simulator>
void TestApplyGate2() {
  unsigned num_qubits = 2;
  unsigned num_threads = 1;

  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space(num_threads);
  Simulator simulator(num_threads);

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

template <typename Simulator>
void TestApplyGate3() {
  unsigned num_qubits = 3;
  unsigned num_threads = 1;

  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space(num_threads);
  Simulator simulator(num_threads);

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

template <typename Simulator>
void TestApplyGate5() {
  unsigned num_qubits = 5;
  unsigned num_threads = 1;

  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  StateSpace state_space(num_threads);
  Simulator simulator(num_threads);

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

  GateFused<GateQSim<fp_type>> fgate2{kGateIS, 4, {1, 2}, &gate14,
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

  GateFused<GateQSim<fp_type>> fgate3{kGateCNot, 6, {2, 3}, &gate17,
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

  GateFused<GateQSim<fp_type>> fgate4{kGateFS, 8, {3, 4}, &gate20,
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

  GateFused<GateQSim<fp_type>> fgate5{kGateCP, 10, {0, 1}, &gate23, {&gate23}};
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

template <typename Simulator>
void TestApplyControlGate() {
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

  StateSpace state_space(1);
  Simulator simulator(1);

  auto state = state_space.Create(num_qubits);
  state_space.SetStateZero(state);

  for (const auto& gate : gates) {
    ApplyGate(simulator, gate, state);
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
    auto a = StateSpace::GetAmpl(state, i);
    EXPECT_NEAR(std::real(a), expected_results[i][0], 1e-6);
    EXPECT_NEAR(std::imag(a), expected_results[i][1], 1e-6);
  }
}

template <typename Simulator>
void TestMultiQubitGates() {
  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  for (unsigned q = 1; q < 6; ++q) {
    unsigned size1 = 1 << q;
    unsigned size2 = size1 * size1;

    std::vector<fp_type> matrix;
    matrix.reserve(2 * size2);

    for (unsigned i = 0; i < 2 * size2; ++i) {
      matrix.push_back(i + 1);
    }

    StateSpace state_space(1);
    Simulator simulator(1);

    unsigned num_qubits = 2 * q;
    auto state = state_space.Create(num_qubits);

    std::vector<unsigned> qubits;
    qubits.reserve(q);

    for (unsigned i = 0; i < q; ++i) {
      qubits.push_back(i);
    }

    state_space.SetStateUniform(state);

    // Apply q-qubit gate to the first q qubits.
    simulator.ApplyGate(qubits, matrix.data(), state);

    for (unsigned i = 0; i < size1; ++i) {
      for (unsigned j = 0; j < size1; ++j) {
        // Expected results are calculated analytically.
        fp_type expected_real = size1 * (1 + 2 * j);
        fp_type expected_imag = expected_real + 1;

        auto a = state_space.GetAmpl(state, i * size1 + j);

        EXPECT_NEAR(std::real(a), expected_real, 1e-6);
        EXPECT_NEAR(std::imag(a), expected_imag, 1e-6);
      }
    }

    for (unsigned i = 0; i < q; ++i) {
      qubits[i] = q + i;
    }

    state_space.SetStateUniform(state);

    // Apply q-qubit gate to the last q qubits.
    simulator.ApplyGate(qubits, matrix.data(), state);

    for (unsigned i = 0; i < size1; ++i) {
      for (unsigned j = 0; j < size1; ++j) {
        // Expected results are calculated analytically.
        fp_type expected_real = (1 + 2 * j) * size1;
        fp_type expected_imag = expected_real + 1;

        auto a = state_space.GetAmpl(state, j * size1 + i);

        EXPECT_NEAR(std::real(a), expected_real, 1e-6);
        EXPECT_NEAR(std::imag(a), expected_imag, 1e-6);
      }
    }
  }
}

}  // namespace qsim

#endif  // SIMULATOR_TESTFIXTURE_H_
