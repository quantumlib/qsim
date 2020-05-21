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
Circuit<Cirq::GateCirq<fp_type>> GetCircuit() {
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

  circuit.gates.emplace_back(Cirq::I<fp_type>::Create(7, 0));
  circuit.gates.emplace_back(Cirq::I<fp_type>::Create(7, 1));
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

  using C = std::complex<fp_type>;
  using A = std::array<C, 2>;
  Cirq::Matrix1q<fp_type> m0 = {A{C{1, 0}, C{0, 0}}, A{C{0, 0}, C{0, 1}}};
  Cirq::Matrix1q<fp_type> m1 = {A{C{0, 0}, C{0, -1}}, A{C{0, 1}, C{0, 0}}};
  Cirq::Matrix1q<fp_type> m2 = {A{C{0, 0}, C{1, 0}}, A{C{1, 0}, C{0, 0}}};
  Cirq::Matrix1q<fp_type> m3 = {A{C{1, 0}, C{0, 0}}, A{C{0, 0}, C{-1, 0}}};
  circuit.gates.emplace_back(Cirq::MatrixGate1<fp_type>::Create(18, 0, m0));
  circuit.gates.emplace_back(Cirq::MatrixGate1<fp_type>::Create(18, 1, m1));
  circuit.gates.emplace_back(Cirq::MatrixGate1<fp_type>::Create(18, 2, m2));
  circuit.gates.emplace_back(Cirq::MatrixGate1<fp_type>::Create(18, 3, m3));

  circuit.gates.emplace_back(Cirq::riswap<fp_type>::Create(19, 0, 1, 0.7));
  circuit.gates.emplace_back(Cirq::givens<fp_type>::Create(19, 2, 3, 1.2));

  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(20, 0));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(20, 1));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(20, 2));
  circuit.gates.emplace_back(Cirq::H<fp_type>::Create(20, 3));

  return circuit;
}

std::vector<std::complex<double>> expected_results = {
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
