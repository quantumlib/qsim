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

#ifndef GATES_CIRQ_TESTFIXTURE_H_
#define GATES_CIRQ_TESTFIXTURE_H_

#include <array>
#include <complex>
#include <vector>

#include "../lib/circuit.h"
#include "../lib/gates_cirq.h"

namespace qsim {

namespace CirqCircuit1 {

template <typename fp_type>
Circuit<Cirq::GateCirq<fp_type>> GetCircuit(bool qsim) {
  Circuit<Cirq::GateCirq<fp_type>> circuit{4, {}};
  circuit.gates.reserve(128);

  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(0, 0));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(0, 1));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(0, 2));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(0, 3));

  circuit.gates.emplace_back(Cirq::T<fp_type>::Create(1, 0));
  circuit.gates.emplace_back(Cirq::T<fp_type>::Create(1, 1));
  circuit.gates.emplace_back(Cirq::T<fp_type>::Create(1, 2));
  circuit.gates.emplace_back(Cirq::T<fp_type>::Create(1, 3));

  circuit.gates.emplace_back(
      Cirq::CZPowGate<fp_type>::Create(2, 0, 1, 0.7, 0.2));
  circuit.gates.emplace_back(
      Cirq::CXPowGate<fp_type>::Create(2, 2, 3, 1.2, 0.4));

  circuit.gates.emplace_back(Cirq::XPowGate<fp_type>::Create(3, 0, 0.3, 1.1));
  circuit.gates.emplace_back(Cirq::YPowGate<fp_type>::Create(3, 1, 0.4, 1.0));
  circuit.gates.emplace_back(Cirq::ZPowGate<fp_type>::Create(3, 2, 0.5, 0.9));
  circuit.gates.emplace_back(Cirq::HPowGate<fp_type>::Create(3, 3, 0.6, 0.8));

  circuit.gates.emplace_back(Cirq::CX<fp_type>::Create(4, 0, 2));
  circuit.gates.emplace_back(Cirq::CZ<fp_type>::Create(4, 1, 3));

  circuit.gates.emplace_back(Cirq::X<fp_type>::Create(5, 0));
  circuit.gates.emplace_back(Cirq::Y<fp_type>::Create(5, 1));
  circuit.gates.emplace_back(Cirq::Z<fp_type>::Create(5, 2));
  circuit.gates.emplace_back(Cirq::S<fp_type>::Create(5, 3));

  circuit.gates.emplace_back(
      Cirq::XXPowGate<fp_type>::Create(6, 0, 1, 0.4, 0.7));
  circuit.gates.emplace_back(
      Cirq::YYPowGate<fp_type>::Create(6, 2, 3, 0.8, 0.5));

  circuit.gates.emplace_back(Cirq::I1<fp_type>::Create(7, 0));
  circuit.gates.emplace_back(Cirq::I1<fp_type>::Create(7, 1));
  circuit.gates.emplace_back(Cirq::I2<fp_type>::Create(7, 2, 3));

  circuit.gates.emplace_back(Cirq::rx<fp_type>::Create(8, 0, 0.7));
  circuit.gates.emplace_back(Cirq::ry<fp_type>::Create(8, 1, 0.2));
  circuit.gates.emplace_back(Cirq::rz<fp_type>::Create(8, 2, 0.4));
  circuit.gates.emplace_back(
      Cirq::PhasedXPowGate<fp_type>::Create(8, 3, 0.8, 0.6, 0.3));

  circuit.gates.emplace_back(
      Cirq::ZZPowGate<fp_type>::Create(9, 0, 2, 0.3, 1.3));
  circuit.gates.emplace_back(
      Cirq::ISwapPowGate<fp_type>::Create(9, 1, 3, 0.6, 1.2));

  circuit.gates.emplace_back(Cirq::XPowGate<fp_type>::Create(10, 0, 0.1, 0.9));
  circuit.gates.emplace_back(Cirq::YPowGate<fp_type>::Create(10, 1, 0.2, 1.0));
  circuit.gates.emplace_back(Cirq::ZPowGate<fp_type>::Create(10, 2, 0.3, 1.1));
  circuit.gates.emplace_back(Cirq::HPowGate<fp_type>::Create(10, 3, 0.4, 1.2));

  circuit.gates.emplace_back(
      Cirq::SwapPowGate<fp_type>::Create(11, 0, 1, 0.2, 0.9));
  circuit.gates.emplace_back(
      Cirq::PhasedISwapPowGate<fp_type>::Create(11, 2, 3, 0.8, 0.6));

  circuit.gates.emplace_back(
      Cirq::PhasedXZGate<fp_type>::Create(12, 0, 0.2, 0.3, 1.4));
  circuit.gates.emplace_back(Cirq::T<fp_type>::Create(12, 1));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(12, 2));
  circuit.gates.emplace_back(Cirq::S<fp_type>::Create(12, 3));

  circuit.gates.emplace_back(Cirq::SWAP<fp_type>::Create(13, 0, 2));
  circuit.gates.emplace_back(Cirq::XX<fp_type>::Create(13, 1, 3));

  circuit.gates.emplace_back(Cirq::rx<fp_type>::Create(14, 0, 0.8));
  circuit.gates.emplace_back(Cirq::ry<fp_type>::Create(14, 1, 0.9));
  circuit.gates.emplace_back(Cirq::rz<fp_type>::Create(14, 2, 1.2));
  circuit.gates.emplace_back(Cirq::T<fp_type>::Create(14, 3));

  circuit.gates.emplace_back(Cirq::YY<fp_type>::Create(15, 0, 1));
  circuit.gates.emplace_back(Cirq::ISWAP<fp_type>::Create(15, 2, 3));

  circuit.gates.emplace_back(Cirq::T<fp_type>::Create(16, 0));
  circuit.gates.emplace_back(Cirq::Z<fp_type>::Create(16, 1));
  circuit.gates.emplace_back(Cirq::Y<fp_type>::Create(16, 2));
  circuit.gates.emplace_back(Cirq::X<fp_type>::Create(16, 3));

  circuit.gates.emplace_back(
      Cirq::FSimGate<fp_type>::Create(17, 0, 2, 0.3, 1.7));
  circuit.gates.emplace_back(Cirq::ZZ<fp_type>::Create(17, 1, 3));

  if (qsim) {
    circuit.gates.emplace_back(Cirq::ry<fp_type>::Create(18, 0, 1.3));
    circuit.gates.emplace_back(Cirq::rz<fp_type>::Create(18, 1, 0.4));
    circuit.gates.emplace_back(Cirq::rx<fp_type>::Create(18, 2, 0.7));
    circuit.gates.emplace_back(Cirq::S<fp_type>::Create(18, 3));

    circuit.gates.emplace_back(Cirq::I<fp_type>::Create(19, {0, 1, 2, 3}));

    circuit.gates.emplace_back(
        Cirq::CCZPowGate<fp_type>::Create(20, 2, 0, 1, 0.7, 0.3));

    circuit.gates.emplace_back(Cirq::CCXPowGate<fp_type>::Create(
        21, 3, 1, 0, 0.4, 0.6).ControlledBy({2}, {0}));

    circuit.gates.emplace_back(Cirq::rx<fp_type>::Create(22, 0, 0.3));
    circuit.gates.emplace_back(Cirq::ry<fp_type>::Create(22, 1, 0.5));
    circuit.gates.emplace_back(Cirq::rz<fp_type>::Create(22, 2, 0.7));
    circuit.gates.emplace_back(Cirq::rx<fp_type>::Create(22, 3, 0.9));

    circuit.gates.emplace_back(
        Cirq::TwoQubitDiagonalGate<fp_type>::Create(23, 0, 1,
                                                    {0.1f, 0.2f, 0.3f, 0.4f}));

    circuit.gates.emplace_back(
        Cirq::ThreeQubitDiagonalGate<fp_type>::Create(
            24, 1, 2, 3, {0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3}));

    circuit.gates.emplace_back(Cirq::CSwapGate<fp_type>::Create(25, 0, 3, 1));

    circuit.gates.emplace_back(Cirq::rz<fp_type>::Create(26, 0, 0.6));
    circuit.gates.emplace_back(Cirq::rx<fp_type>::Create(26, 1, 0.7));
    circuit.gates.emplace_back(Cirq::ry<fp_type>::Create(26, 2, 0.8));
    circuit.gates.emplace_back(Cirq::rz<fp_type>::Create(26, 3, 0.9));

    circuit.gates.emplace_back(Cirq::TOFFOLI<fp_type>::Create(27, 3, 2, 0));

    circuit.gates.emplace_back(Cirq::FREDKIN<fp_type>::Create(28, 1, 3, 2));

    Matrix<fp_type> m40 = {0, 0, -0.5, -0.5, -0.5, -0.5, 0, 0,
                           0.5, -0.5, 0, 0, 0, 0, -0.5, 0.5,
                           0.5, -0.5, 0, 0, 0, 0, 0.5, -0.5,
                           0, 0, -0.5, -0.5, 0.5, 0.5, 0, 0};

    circuit.gates.emplace_back(
        Cirq::MatrixGate2<fp_type>::Create(51, 0, 1, m40));
    circuit.gates.emplace_back(
        Cirq::MatrixGate<fp_type>::Create(51, {2, 3},
                                          {0.5, -0.5, 0, 0, 0, 0, -0.5, 0.5,
                                           0, 0, 0.5, -0.5, -0.5, 0.5, 0, 0,
                                           0, 0, -0.5, 0.5, -0.5, 0.5, 0, 0,
                                           0.5, -0.5, 0, 0, 0, 0, 0.5, -0.5}));
  }

  Matrix<fp_type> m20 = {1, 0, 0, 0, 0, 0, 0, 1};
  Matrix<fp_type> m21 = {0, 0, 0, -1, 0, 1, 0, 0};
  Matrix<fp_type> m22 = {0, 0, 1, 0, 1, 0, 0, 0};
  circuit.gates.emplace_back(Cirq::MatrixGate1<fp_type>::Create(52, 0, m20));
  circuit.gates.emplace_back(Cirq::MatrixGate1<fp_type>::Create(52, 1, m21));
  circuit.gates.emplace_back(Cirq::MatrixGate1<fp_type>::Create(52, 2, m22));
  circuit.gates.emplace_back(
      Cirq::MatrixGate1<fp_type>::Create(52, 3, {1, 0, 0, 0, 0, 0, -1, 0}));

  circuit.gates.emplace_back(Cirq::riswap<fp_type>::Create(53, 0, 1, 0.7));
  circuit.gates.emplace_back(Cirq::givens<fp_type>::Create(53, 2, 3, 1.2));

  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(54, 0));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(54, 1));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(54, 2));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(54, 3));

  return circuit;
}

std::vector<std::complex<double>> expected_results0 = {
  {0.12549974, 0.21873295},
  {-0.09108202, 0.042387843},
  {0.11101487, -0.1457827},
  {0.07818171, 0.06796847},
  {-0.12215838, -0.01435155},
  {0.19018647, -0.2773067},
  {0.07645161, 0.16945378},
  {-0.13923888, -0.23470248},
  {-0.15811542, 0.18569943},
  {0.12532015, -0.2430008},
  {0.011194898, -0.43144885},
  {-0.29667473, 0.0515977},
  {0.017342143, 0.35579148},
  {0.21502252, 0.082056835},
  {-0.04769493, 0.004258407},
  {-0.21426316, -0.074324496},
};

std::vector<std::complex<double>> expected_results1 = {
  {-0.014675243, 0.05654204},
  {-0.0075858636, 0.28545904},
  {0.044140648, 0.053896483},
  {-0.033529136, 0.32497203},
  {-0.13991567, -0.13084067},
  {0.234054, -0.07352882},
  {-0.14253256, -0.022177307},
  {-0.09260284, -0.13516076},
  {-0.061443992, -0.14103678},
  {0.25451535, 0.22917412},
  {-0.34202546, -0.27581766},
  {0.0010748552, 0.1542618},
  {0.07094702, -0.21318978},
  {0.06633715, 0.37584817},
  {0.2312484, 0.09549438},
  {-0.18656375, -0.08693269},
};

/*

These results were obtained by the following Cirq code, which simulates
the same circuit as above.

import cirq
import numpy as np

def main():
  q0 = cirq.GridQubit(1, 1) # 3
  q1 = cirq.GridQubit(1, 0) # 2
  q2 = cirq.GridQubit(0, 1) # 1
  q3 = cirq.GridQubit(0, 0) # 0

  circuit = cirq.Circuit(
    cirq.Moment([
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.H(q3),
    ]),
    cirq.Moment([
        cirq.T(q0),
        cirq.T(q1),
        cirq.T(q2),
        cirq.T(q3),
    ]),
    cirq.Moment([
        cirq.CZPowGate(exponent=0.7, global_shift=0.2)(q0, q1),
        cirq.CXPowGate(exponent=1.2, global_shift=0.4)(q2, q3),
    ]),
    cirq.Moment([
        cirq.XPowGate(exponent=0.3, global_shift=1.1)(q0),
        cirq.YPowGate(exponent=0.4, global_shift=1)(q1),
        cirq.ZPowGate(exponent=0.5, global_shift=0.9)(q2),
        cirq.HPowGate(exponent=0.6, global_shift=0.8)(q3),
    ]),
    cirq.Moment([
        cirq.CX(q0, q2),
        cirq.CZ(q1, q3),
    ]),
    cirq.Moment([
        cirq.X(q0),
        cirq.Y(q1),
        cirq.Z(q2),
        cirq.S(q3),
    ]),
    cirq.Moment([
        cirq.XXPowGate(exponent=0.4, global_shift=0.7)(q0, q1),
        cirq.YYPowGate(exponent=0.8, global_shift=0.5)(q2, q3),
    ]),
    cirq.Moment([
        cirq.I(q0),
        cirq.I(q1),
        cirq.IdentityGate(2)(q2, q3)
    ]),
    cirq.Moment([
        cirq.rx(0.7)(q0),
        cirq.ry(0.2)(q1),
        cirq.rz(0.4)(q2),
        cirq.PhasedXPowGate(
            phase_exponent=0.8, exponent=0.6, global_shift=0.3)(q3),
    ]),
    cirq.Moment([
        cirq.ZZPowGate(exponent=0.3, global_shift=1.3)(q0, q2),
        cirq.ISwapPowGate(exponent=0.6, global_shift=1.2)(q1, q3),
    ]),
    cirq.Moment([
        cirq.XPowGate(exponent=0.1, global_shift=0.9)(q0),
        cirq.YPowGate(exponent=0.2, global_shift=1)(q1),
        cirq.ZPowGate(exponent=0.3, global_shift=1.1)(q2),
        cirq.HPowGate(exponent=0.4, global_shift=1.2)(q3),
    ]),
    cirq.Moment([
        cirq.SwapPowGate(exponent=0.2, global_shift=0.9)(q0, q1),
        cirq.PhasedISwapPowGate(phase_exponent = 0.8, exponent=0.6)(q2, q3),
    ]),
    cirq.Moment([
        cirq.PhasedXZGate(
            x_exponent=0.2, z_exponent=0.3, axis_phase_exponent=1.4)(q0),
        cirq.T(q1),
        cirq.H(q2),
        cirq.S(q3),
    ]),
    cirq.Moment([
        cirq.SWAP(q0, q2),
        cirq.XX(q1, q3),
    ]),
    cirq.Moment([
        cirq.rx(0.8)(q0),
        cirq.ry(0.9)(q1),
        cirq.rz(1.2)(q2),
        cirq.T(q3),
    ]),
    cirq.Moment([
        cirq.YY(q0, q1),
        cirq.ISWAP(q2, q3),
    ]),
    cirq.Moment([
        cirq.T(q0),
        cirq.Z(q1),
        cirq.Y(q2),
        cirq.X(q3),
    ]),
    cirq.Moment([
        cirq.FSimGate(0.3, 1.7)(q0, q2),
        cirq.ZZ(q1, q3),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.ry(1.3)(q0),
        cirq.rz(0.4)(q1),
        cirq.rx(0.7)(q2),
        cirq.S(q3),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.IdentityGate(4).on(q0, q1, q2, q3),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.CCZPowGate(exponent=0.7, global_shift=0.3)(q2, q0, q1),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.CCXPowGate(exponent=0.4, global_shift=0.6)(
            q3, q1, q0).controlled_by(q2, control_values=[0]),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.rx(0.3)(q0),
        cirq.ry(0.5)(q1),
        cirq.rz(0.7)(q2),
        cirq.rx(0.9)(q3),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.TwoQubitDiagonalGate([0.1, 0.2, 0.3, 0.4])(q0, q1),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.ThreeQubitDiagonalGate([0.5, 0.6, 0.7, 0.8,
                                     0.9, 1, 1.2, 1.3])(q1, q2, q3),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.CSwapGate()(q0, q3, q1),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.rz(0.6)(q0),
        cirq.rx(0.7)(q1),
        cirq.ry(0.8)(q2),
        cirq.rz(0.9)(q3),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.TOFFOLI(q3, q2, q0),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.FREDKIN(q1, q3, q2),
    ]),
    # The following moment should not be included if qsim is false above.
    cirq.Moment([
        cirq.MatrixGate(np.array([[0, -0.5 - 0.5j, -0.5 - 0.5j, 0],
                                  [0.5 - 0.5j, 0, 0, -0.5 + 0.5j],
                                  [0.5 - 0.5j, 0, 0, 0.5 - 0.5j],
                                  [0, -0.5 - 0.5j, 0.5 + 0.5j, 0]]))(q0, q1),
        cirq.MatrixGate(np.array([[0.5 - 0.5j, 0, 0, -0.5 + 0.5j],
                                  [0, 0.5 - 0.5j, -0.5 + 0.5j, 0],
                                  [0, -0.5 + 0.5j, -0.5 + 0.5j, 0],
                                  [0.5 - 0.5j, 0, 0, 0.5 - 0.5j]]))(q2, q3),
    ]),
    cirq.Moment([
        cirq.MatrixGate(np.array([[1, 0], [0, 1j]]))(q0),
        cirq.MatrixGate(np.array([[0, -1j], [1j, 0]]))(q1),
        cirq.MatrixGate(np.array([[0, 1], [1, 0]]))(q2),
        cirq.MatrixGate(np.array([[1, 0], [0, -1]]))(q3),
    ]),
    cirq.Moment([
        cirq.riswap(0.7)(q0, q1),
        cirq.givens(1.2)(q2, q3),
    ]),
    cirq.Moment([
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.H(q3),
    ]),
  )

  simulator = cirq.Simulator()
  result = simulator.simulate(circuit)

  for i in range(len(result.state_vector())):
      print(result.state_vector()[i])

if __name__ == '__main__':
  main()

*/

}  // namespace CirqCircuit1

}  // namespace qsim

#endif  // GATES_CIRQ_TESTFIXTURE_H_
