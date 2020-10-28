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

#include "gtest/gtest.h"

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser.h"
#include "../lib/gate_appl.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"

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
    constexpr char circuit_str[] =
R"(6
0 h 0
0 h 1
0 h 2
0 h 3
0 h 4
0 h 5
1 c 0 t 1
2 rx 0 0.1
2 ry 1 0.2
2 rz 2 0.3
2 rx 3 0.4
2 ry 4 0.5
2 rz 5 0.6
3 c 0 1 h 2
4 rz 0 0.7
4 rx 1 0.8
4 ry 2 0.9
4 rz 3 1.0
4 rx 4 1.1
4 rx 5 1.2
5 c 0 1 2 t 3
6 ry 0 1.3
6 rz 1 1.4
6 rx 2 1.5
6 ry 3 1.6
6 rz 4 1.7
6 rx 5 1.8
7 c 0 1 2 3 t 4
8 rz 0 1.9
8 rx 1 2.0
8 ry 2 2.1
8 rz 3 2.2
8 rx 4 2.3
8 ry 5 2.4
9 c 0 is 1 2
10 rx 0 2.5
10 ry 1 2.6
10 rz 2 2.7
10 rx 3 2.8
10 ry 4 2.9
10 rz 5 3.0
11 c 0 1 is 2 3
12 ry 0 3.1
12 rz 1 3.2
12 rx 2 3.3
12 ry 3 3.4
12 rz 4 3.5
12 rx 5 3.6
13 c 0 1 2 is 3 4
14 rz 0 3.7
15 rx 1 3.8
15 ry 2 3.9
15 rz 3 4.0
15 rx 4 4.1
15 rx 5 4.2
16 c 0 1 2 3 is 4 5
17 rx 0 4.3
17 ry 1 4.4
17 rz 2 4.5
17 rx 3 4.6
17 ry 4 4.7
17 rz 5 4.8
18 c 3 is 4 5
19 ry 0 4.9
19 rz 1 5.0
19 rx 2 5.1
19 ry 3 5.2
19 rz 4 5.3
19 rx 5 5.4
20 c 4 is 0 1
21 rz 0 5.5
21 rx 1 5.6
21 ry 2 5.7
21 rz 3 5.8
21 rx 4 5.9
21 ry 5 6.0
22 c 4 is 0 2
23 rx 0 6.1
23 ry 1 6.2
23 rz 2 6.3
23 rx 3 6.4
23 ry 4 6.5
23 rz 5 6.6
24 c 4 is 0 5
25 ry 0 6.7
25 rz 1 6.8
25 rx 2 6.9
25 ry 3 7.0
25 rz 4 7.1
25 rx 5 7.2
26 c 4 h 5
27 rz 0 7.3
27 rx 1 7.4
27 ry 2 7.5
27 rz 3 7.6
27 rx 4 7.7
27 ry 5 7.8
)";

  using StateSpace = typename Simulator::StateSpace;
  using fp_type = typename StateSpace::fp_type;

  Circuit<GateQSim<fp_type>> circuit;
  std::stringstream ss(circuit_str);
  EXPECT_TRUE(CircuitQsimParser<IO>::FromStream(99, "string", ss, circuit));
  EXPECT_EQ(circuit.num_qubits, 6);
  EXPECT_EQ(circuit.gates.size(), 97);

  StateSpace state_space(1);
  Simulator simulator(1);

  auto state = state_space.Create(circuit.num_qubits);
  state_space.SetStateZero(state);

  for (const auto& gate : circuit.gates) {
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
      cirq.H(q2).controlled_by(q0, q1),
    ]),
    cirq.Moment([
       cirq.rz(0.7)(q0),
       cirq.rx(0.8)(q1),
       cirq.ry(0.9)(q2),
       cirq.rz(1.0)(q3),
       cirq.rx(1.1)(q4),
       cirq.rx(1.2)(q5),
    ]),
    cirq.Moment([
      cirq.T(q3).controlled_by(q0, q1, q2),
    ]),
    cirq.Moment([
       cirq.ry(1.3)(q0),
       cirq.rz(1.4)(q1),
       cirq.rx(1.5)(q2),
       cirq.ry(1.6)(q3),
       cirq.rz(1.7)(q4),
       cirq.rx(1.8)(q5),
    ]),
    cirq.Moment([
      cirq.T(q4).controlled_by(q0, q1, q2, q3),
    ]),
    cirq.Moment([
       cirq.rz(1.9)(q0),
       cirq.rx(2.0)(q1),
       cirq.ry(2.1)(q2),
       cirq.rz(2.2)(q3),
       cirq.rx(2.3)(q4),
       cirq.ry(2.4)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q1, q2).controlled_by(q0),
    ]),
    cirq.Moment([
       cirq.rx(2.5)(q0),
       cirq.ry(2.6)(q1),
       cirq.rz(2.7)(q2),
       cirq.rx(2.8)(q3),
       cirq.ry(2.9)(q4),
       cirq.rz(3.0)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q2, q3).controlled_by(q0, q1),
    ]),
    cirq.Moment([
       cirq.ry(3.1)(q0),
       cirq.rz(3.2)(q1),
       cirq.rx(3.3)(q2),
       cirq.ry(3.4)(q3),
       cirq.rz(3.5)(q4),
       cirq.rx(3.6)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q3, q4).controlled_by(q0, q1, q2),
    ]),
    cirq.Moment([
       cirq.rz(3.7)(q0),
    ]),
    cirq.Moment([
       cirq.rx(3.8)(q1),
       cirq.ry(3.9)(q2),
       cirq.rz(4.0)(q3),
       cirq.rx(4.1)(q4),
       cirq.rx(4.2)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q4, q5).controlled_by(q0, q1, q2, q3),
    ]),
    cirq.Moment([
       cirq.rx(4.3)(q0),
       cirq.ry(4.4)(q1),
       cirq.rz(4.5)(q2),
       cirq.rx(4.6)(q3),
       cirq.ry(4.7)(q4),
       cirq.rz(4.8)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q4, q5).controlled_by(q3),
    ]),
    cirq.Moment([
       cirq.ry(4.9)(q0),
       cirq.rz(5.0)(q1),
       cirq.rx(5.1)(q2),
       cirq.ry(5.2)(q3),
       cirq.rz(5.3)(q4),
       cirq.rx(5.4)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q1).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.rz(5.5)(q0),
       cirq.rx(5.6)(q1),
       cirq.ry(5.7)(q2),
       cirq.rz(5.8)(q3),
       cirq.rx(5.9)(q4),
       cirq.ry(6.0)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q2).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.rx(6.1)(q0),
       cirq.ry(6.2)(q1),
       cirq.rz(6.3)(q2),
       cirq.rx(6.4)(q3),
       cirq.ry(6.5)(q4),
       cirq.rz(6.6)(q5),
    ]),
    cirq.Moment([
      cirq.ISWAP(q0, q5).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.ry(6.7)(q0),
       cirq.rz(6.8)(q1),
       cirq.rx(6.9)(q2),
       cirq.ry(7.0)(q3),
       cirq.rz(7.1)(q4),
       cirq.rx(7.2)(q5),
    ]),
    cirq.Moment([
      cirq.H(q5).controlled_by(q4),
    ]),
    cirq.Moment([
       cirq.rz(7.3)(q0),
       cirq.rx(7.4)(q1),
       cirq.ry(7.5)(q2),
       cirq.rz(7.6)(q3),
       cirq.rx(7.7)(q4),
       cirq.ry(7.8)(q5),
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
    {-0.024000414, 0.13359392},
    {-0.010777686, 0.040734157},
    {0.09360839, 0.10015345},
    {-0.017626993, -0.037382595},
    {0.16321257, 0.18193814},
    {0.110039294, 0.00031051738},
    {0.2403549, -0.14277315},
    {-0.003620103, 0.021327903},
    {-0.03628681, 0.035105858},
    {0.010516882, 0.028288865},
    {0.02837902, -0.09398348},
    {0.015269253, -0.06789709},
    {-0.042230412, -0.023053035},
    {0.093636535, -0.031277604},
    {-0.0024242331, 0.018563256},
    {0.010324602, 0.05955182},
    {-0.17416054, -0.044389807},
    {-0.00607755, -0.07658397},
    {-0.14872268, -0.080746755},
    {-0.040476087, -0.02440174},
    {-0.008808218, 0.13825473},
    {-0.030034762, 0.089864776},
    {0.14024268, 0.14761701},
    {-0.00045500696, 0.09647506},
    {-0.014829321, 0.09739353},
    {0.024183076, 0.05714892},
    {0.021111023, 0.046445612},
    {-0.005038852, -0.04069536},
    {0.052329402, -0.012001134},
    {-0.014491424, 0.038514502},
    {-0.0062525272, -0.05639422},
    {-0.0823572, -0.014202977},
    {-0.15058956, 0.023767654},
    {0.06630518, -0.08713264},
    {-0.049538083, -0.09937088},
    {-0.0689518, -0.09209575},
    {0.0022153333, -0.1445844},
    {0.016036857, -0.003312569},
    {-0.05356656, 0.11282002},
    {-0.06705719, 0.050860442},
    {0.034315012, -0.094269216},
    {0.0027618187, 0.03822149},
    {0.04699982, 0.0855632},
    {-0.11269423, -0.030068845},
    {0.06262903, 0.09884508},
    {-0.10035621, 0.07141959},
    {-0.0031322166, -0.019698868},
    {0.033788137, -0.02274291},
    {-0.15133715, -0.2433885},
    {0.011551354, -0.031001069},
    {-0.16904244, -0.3048535},
    {0.024375454, 0.09429265},
    {0.2454747, -0.18158379},
    {0.07559521, 0.07178006},
    {-0.20359893, -0.053257704},
    {0.1096768, -0.028451644},
    {-0.0577899, 0.07780492},
    {0.07624379, -0.02042614},
    {0.022697305, 0.037892006},
    {-0.05274996, 0.1001262},
    {-0.05086846, -0.019555436},
    {0.014091074, 0.0069812844},
    {0.049313433, 0.016418802},
    {-0.10714137, 0.026649015},
  };

  unsigned size = 1 << circuit.num_qubits;

  for (unsigned i = 0; i < size; ++i) {
    auto a = StateSpace::GetAmpl(state, i);
    EXPECT_NEAR(std::real(a), expected_results[i][0], 1e-6);
    EXPECT_NEAR(std::imag(a), expected_results[i][1], 1e-6);
  }
}

}  // namespace qsim

#endif  // SIMULATOR_TESTFIXTURE_H_
