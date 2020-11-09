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

#ifndef GATES_QSIM_H_
#define GATES_QSIM_H_

#include <array>
#include <cmath>
#include <vector>

#include "gate.h"

namespace qsim {

// Gate set implemented in qsim contains the following gates.
enum GateKind {
  kGateId1 = 0, // one-qubit Id
  kGateHd,      // Hadamard
  kGateT,       // T
  kGateX,       // X
  kGateY,       // Y
  kGateZ,       // Z
  kGateX2,      // sqrt(X)
  kGateY2,      // sqrt(Y)
  kGateRX,      // X-rotation
  kGateRY,      // Y-rotation
  kGateRZ,      // Z-rotation
  kGateRXY,     // XY-rotation (rotation around arbitrary axis in the XY plane)
  kGateHZ2,     // pi / 2 rotation around the X + Y axis
  kGateS,       // S
  kGateId2,     // two-qubit Id
  kGateCZ,      // CZ
  kGateCNot,    // CNOT (CX)
  kGateSwap,    // swap
  kGateIS,      // iSwap
  kGateFS,      // fSim
  kGateCP,      // control phase
  kDecomp = gate::kDecomp,
  kMeasurement = gate::kMeasurement,
};

// Specialization of Gate (defined in gate.h) for the qsim gate set.
template <typename fp_type>
using GateQSim = Gate<fp_type, GateKind>;

constexpr double h_double = 0.5;
constexpr double is2_double = 0.7071067811865475;

// One-qubit gates:

/**
 * The one-qubit identity gate.
 */
template <typename fp_type>
struct GateId1 {
  static constexpr GateKind kind = kGateId1;
  static constexpr char name[] = "id1";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateId1>(
        time, {q0}, {1, 0, 0, 0, 0, 0, 1, 0});
  }
};

/**
 * The Hadamard gate.
 */
template <typename fp_type>
struct GateHd {
  static constexpr GateKind kind = kGateHd;
  static constexpr char name[] = "h";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateHd>(
        time, {q0}, {is2, 0, is2, 0, is2, 0, -is2, 0});
  }
};

/**
 * The T gate, equivalent to `Z ^ 0.25`.
 */
template <typename fp_type>
struct GateT {
  static constexpr GateKind kind = kGateT;
  static constexpr char name[] = "t";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateT>(
        time, {q0}, {1, 0, 0, 0, 0, 0, is2, is2});
  }
};

/**
 * The Pauli X (or "NOT") gate.
 */
template <typename fp_type>
struct GateX {
  static constexpr GateKind kind = kGateX;
  static constexpr char name[] = "x";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateX>(
        time, {q0}, {0, 0, 1, 0, 1, 0, 0, 0});
  }
};

/**
 * The Pauli Y gate.
 */
template <typename fp_type>
struct GateY {
  static constexpr GateKind kind = kGateY;
  static constexpr char name[] = "y";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateY>(
        time, {q0}, {0, 0, 0, -1, 0, 1, 0, 0});
  }
};

/**
 * The Pauli Z gate.
 */
template <typename fp_type>
struct GateZ {
  static constexpr GateKind kind = kGateZ;
  static constexpr char name[] = "z";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateZ>(
        time, {q0}, {1, 0, 0, 0, 0, 0, -1, 0});
  }
};

/**
 * The "square root of X" gate.
 */
template <typename fp_type>
struct GateX2 {
  static constexpr GateKind kind = kGateX2;
  static constexpr char name[] = "x_1_2";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateX2>(
        time, {q0}, {h, h, h, -h, h, -h, h, h});
  }
};

/**
 * The "square root of Y" gate.
 */
template <typename fp_type>
struct GateY2 {
  static constexpr GateKind kind = kGateY2;
  static constexpr char name[] = "y_1_2";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateY2>(
        time, {q0}, {h, h, -h, -h, h, h, h, h});
  }
};

/**
 * A gate that rotates around the X axis of the Bloch sphere.
 * This is a generalization of the X gate.
 */
template <typename fp_type>
struct GateRX {
  static constexpr GateKind kind = kGateRX;
  static constexpr char name[] = "rx";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type c = std::cos(phi2);
    fp_type s = std::sin(phi2);

    return CreateGate<GateQSim<fp_type>, GateRX>(
        time, {q0}, {c, 0, 0, s, 0, s, c, 0}, {phi});
  }
};

/**
 * A gate that rotates around the Y axis of the Bloch sphere.
 * This is a generalization of the Y gate.
 */
template <typename fp_type>
struct GateRY {
  static constexpr GateKind kind = kGateRY;
  static constexpr char name[] = "ry";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type c = std::cos(phi2);
    fp_type s = std::sin(phi2);

    return CreateGate<GateQSim<fp_type>, GateRY>(
        time, {q0}, {c, 0, s, 0, -s, 0, c, 0}, {phi});
  }
};

/**
 * A gate that rotates around the Z axis of the Bloch sphere.
 * This is a generalization of the Z gate.
 */
template <typename fp_type>
struct GateRZ {
  static constexpr GateKind kind = kGateRZ;
  static constexpr char name[] = "rz";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type c = std::cos(phi2);
    fp_type s = std::sin(phi2);

    return CreateGate<GateQSim<fp_type>, GateRZ>(
        time, {q0}, {c, s, 0, 0, 0, 0, c, -s}, {phi});
  }
};

/**
 * A gate that rotates around an arbitrary axis in the XY-plane.
 */
template <typename fp_type>
struct GateRXY {
  static constexpr GateKind kind = kGateRXY;
  static constexpr char name[] = "rxy";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(
      unsigned time, unsigned q0, fp_type theta, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type cp = std::cos(phi2);
    fp_type sp = std::sin(phi2);
    fp_type ct = std::cos(theta) * sp;
    fp_type st = std::sin(theta) * sp;

    return CreateGate<GateQSim<fp_type>, GateRXY>(
        time, {q0}, {cp, 0, st, ct, -st, ct, cp, 0}, {theta, phi});
  }
};

/**
 * A pi / 2 rotation around the X + Y axis.
 */
template <typename fp_type>
struct GateHZ2 {
  static constexpr GateKind kind = kGateHZ2;
  static constexpr char name[] = "hz_1_2";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateHZ2>(
        time, {q0}, {h, h, 0, -is2, is2, 0, h, h});
  }
};

/**
 * The S gate, equivalent to "square root of Z".
 */
template <typename fp_type>
struct GateS {
  static constexpr GateKind kind = kGateS;
  static constexpr char name[] = "s";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateQSim<fp_type>, GateS>(
        time, {q0}, {1, 0, 0, 0, 0, 0, 0, 1});
  }
};

// Two-qubit gates:

/**
 * The two-qubit identity gate.
 */
template <typename fp_type>
struct GateId2 {
  static constexpr GateKind kind = kGateId2;
  static constexpr char name[] = "id2";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateQSim<fp_type>, GateId2>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 0, 0, 1, 0}},
    };
  }
};

/**
 * The controlled-Z (CZ) gate.
 */
template <typename fp_type>
struct GateCZ {
  static constexpr GateKind kind = kGateCZ;
  static constexpr char name[] = "cz";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateQSim<fp_type>, GateCZ>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, -1, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 1, 0}},
      {{0, 0, 0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 0, 0, -1, 0}},
    };
  }
};

/**
 * The controlled-X (CX or CNOT) gate.
 */
template <typename fp_type>
struct GateCNot {
  static constexpr GateKind kind = kGateCNot;
  static constexpr char name[] = "cnot";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = false;

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateQSim<fp_type>, GateCNot>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 1, 0}},
      {{0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 1, 0, 1, 0, 0, 0}},
    };
  }
};

/**
 * The SWAP gate. Exchanges two qubits.
 */
template <typename fp_type>
struct GateSwap {
  static constexpr GateKind kind = kGateSwap;
  static constexpr char name[] = "sw";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateQSim<fp_type>, GateSwap>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{is2, 0, 0, 0, 0, 0, is2, 0}, {is2, 0, 0, 0, 0, 0, is2, 0}},
      {{0, 0, is2, 0, is2, 0, 0, 0}, {0, 0, is2, 0, is2, 0, 0, 0}},
      {{0, 0, 0, -is2, 0, is2, 0, 0}, {0, 0, 0, -is2, 0, is2, 0, 0}},
      {{is2, 0, 0, 0, 0, 0, -is2, 0}, {is2, 0, 0, 0, 0, 0, -is2, 0}},
    };
  }
};

/**
 * The ISWAP gate.
 */
template <typename fp_type>
struct GateIS {
  static constexpr GateKind kind = kGateIS;
  static constexpr char name[] = "is";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type h = static_cast<fp_type>(h_double);
  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateQSim<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateQSim<fp_type>, GateIS>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{is2, 0, 0, 0, 0, 0, is2, 0}, {is2, 0, 0, 0, 0, 0, is2, 0}},
      {{0, 0, h, h, h, h, 0, 0}, {0, 0, h, h, h, h, 0, 0}},
      {{0, 0, h, -h, -h, h, 0, 0}, {0, 0, h, -h, -h, h, 0, 0}},
      {{is2, 0, 0, 0, 0, 0, -is2, 0}, {is2, 0, 0, 0, 0, 0, -is2, 0}},
    };
  }
};

/**
 * The fermionic simulation (FSim) gate family. Contains all two-qubit
 * interactions that preserve excitations, up to single-qubit rotations and
 * global phase.
 */
template <typename fp_type>
struct GateFS {
  static constexpr GateKind kind = kGateFS;
  static constexpr char name[] = "fs";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateQSim<fp_type> Create(
      unsigned time, unsigned q0, unsigned q1, fp_type theta, fp_type phi) {
    if (phi < 0) {
      phi += 2 * 3.141592653589793;
    }

    fp_type ct = std::cos(theta);
    fp_type st = std::sin(theta);
    fp_type cp = std::cos(phi);
    fp_type sp = std::sin(phi);

    return CreateGate<GateQSim<fp_type>, GateFS>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, ct, 0, 0, -st, 0, 0,
                         0, 0, 0, -st, ct, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, cp, -sp}, {theta, phi});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type theta, fp_type phi) {
    fp_type ct = std::cos(theta);
    fp_type st = std::sin(theta);

    fp_type cp2 = std::cos(0.5 * phi);
    fp_type sp2 = std::sin(0.5 * phi);
    fp_type cp4 = std::cos(0.25 * phi);
    fp_type sp4 = std::sin(0.25 * phi);

    fp_type a0 = std::sqrt(std::sqrt(1 + 2 * ct * cp2 + ct * ct));
    fp_type a1 = std::sqrt(std::sqrt(1 - 2 * ct * cp2 + ct * ct));

    fp_type p0 = 0.5 * std::atan2(-sp2, cp2 + ct);
    fp_type p1 = 0.5 * std::atan2(-sp2, cp2 - ct);

    fp_type c0 = is2 * a0 * std::cos(p0);
    fp_type s0 = is2 * a0 * std::sin(p0);

    fp_type c1 = is2 * a1 * std::cos(p1);
    fp_type s1 = is2 * a1 * std::sin(p1);

    fp_type st2 = 0.5 * std::sqrt(st);

    fp_type a = cp4 * c0 - sp4 * s0;
    fp_type b = cp4 * s0 + sp4 * c0;
    fp_type c = cp4 * c0 + sp4 * s0;
    fp_type d = cp4 * s0 - sp4 * c0;

    fp_type e = cp4 * c1 - sp4 * s1;
    fp_type f = cp4 * s1 + sp4 * c1;
    fp_type g = -(cp4 * c1 + sp4 * s1);
    fp_type h = -(cp4 * s1 - sp4 * c1);

    return schmidt_decomp_type<fp_type>{
      {{a, b, 0, 0, 0, 0, c, d}, {a, b, 0, 0, 0, 0, c, d}},
      {{0, 0, st2, -st2, st2, -st2, 0, 0}, {0, 0, st2, -st2, st2, -st2, 0, 0}},
      {{0, 0, -st2, -st2, st2, st2, 0, 0}, {0, 0, -st2, -st2, st2, st2, 0, 0}},
      {{e, f, 0, 0, 0, 0, g, h}, {e, f, 0, 0, 0, 0, g, h}},
    };
  }
};

/**
 * The controlled phase gate. A generalized version of GateCZ.
 */
template <typename fp_type>
struct GateCP {
  static constexpr GateKind kind = kGateCP;
  static constexpr char name[] = "cp";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateQSim<fp_type> Create(
      unsigned time, unsigned q0, unsigned q1, fp_type phi) {
    fp_type cp = std::cos(phi);
    fp_type sp = std::sin(phi);

    return CreateGate<GateQSim<fp_type>, GateCP>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, cp, -sp}, {phi});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(fp_type phi) {
    fp_type cp = std::cos(phi);
    fp_type sp = std::sin(phi);

    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 1, 0}},
      {{0, 0, 0, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 0, 0, cp, -sp}},
    };
  }
};

template <typename fp_type>
inline schmidt_decomp_type<fp_type> GetSchmidtDecomp(
    GateKind kind, const std::vector<fp_type>& params) {
  switch (kind) {
  case kGateId2:
    return GateId2<fp_type>::SchmidtDecomp();
  case kGateCZ:
    return GateCZ<fp_type>::SchmidtDecomp();
  case kGateCNot:
    return GateCNot<fp_type>::SchmidtDecomp();
  case kGateSwap:
    return GateSwap<fp_type>::SchmidtDecomp();
  case kGateIS:
    return GateIS<fp_type>::SchmidtDecomp();
  case kGateFS:
    return GateFS<fp_type>::SchmidtDecomp(params[0], params[1]);
  case kGateCP:
    return GateCP<fp_type>::SchmidtDecomp(params[0]);
  default:
    // Single qubit gates: empty Schmidt decomposition.
    return schmidt_decomp_type<fp_type>{};
  }
}

}  // namespace qsim

#endif  // GATES_QSIM_H_
