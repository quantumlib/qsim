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

#ifndef GATES_CIRQ_H_
#define GATES_CIRQ_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "gate.h"
#include "matrix.h"

namespace qsim {

namespace Cirq {

enum GateKind {
  kI1 = 0,     // One-qubit identity gate.
  kI2,         // Two-qubit identity gate.
  kI,          // Multi-qubit identity gate.
  kXPowGate,
  kYPowGate,
  kZPowGate,
  kHPowGate,
  kCZPowGate,
  kCXPowGate,
  krx,
  kry,
  krz,
  kH,
  kS,
  kCZ,
  kCX,
  kT,
  kX,
  kY,
  kZ,
  kPhasedXPowGate,
  kPhasedXZGate,
  kXXPowGate,
  kYYPowGate,
  kZZPowGate,
  kXX,
  kYY,
  kZZ,
  kSwapPowGate,
  kISwapPowGate,
  kriswap,
  kSWAP,
  kISWAP,
  kPhasedISwapPowGate,
  kgivens,
  kFSimGate,
  kTwoQubitDiagonalGate,
  kThreeQubitDiagonalGate,
  kCCZPowGate,
  kCCXPowGate,
  kCSwapGate,
  kCCZ,
  kCCX,
  kMatrixGate1,  // One-qubit matrix gate.
  kMatrixGate2,  // Two-qubit matrix gate.
  kMatrixGate,   // Multi-qubit matrix gate.
  kDecomp = gate::kDecomp,
  kMeasurement = gate::kMeasurement,
};

template <typename fp_type>
using GateCirq = Gate<fp_type, GateKind>;

constexpr double h_double = 0.5;
constexpr double pi_double = 3.14159265358979323846264338327950288;
constexpr double is2_double = 0.7071067811865475;

// Gates from cirq/ops/identity.py:

/**
 * A one-qubit identity gate.
 */
template <typename fp_type>
struct I1 {
  static constexpr GateKind kind = kI1;
  static constexpr char name[] = "I1";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateCirq<fp_type>, I1>(
        time, {q0}, {1, 0, 0, 0, 0, 0, 1, 0});
  }
};

/**
 * A two-qubit identity gate.
 */
template <typename fp_type>
struct I2 {
  static constexpr GateKind kind = kI2;
  static constexpr char name[] = "I2";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateCirq<fp_type>, I2>(
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
 * A multi-qubit identity gate.
 */
template <typename fp_type>
struct I {
  static constexpr GateKind kind = kI;
  static constexpr char name[] = "I";
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time,
                                  const std::vector<unsigned>& qubits) {
    Matrix<fp_type> matrix;
    MatrixIdentity(1 << qubits.size(), matrix);
    return CreateGate<GateCirq<fp_type>, I>(time, qubits, std::move(matrix));
  }
};

// Gates form cirq/ops/common_gates.py:

/**
 * A gate that rotates around the X axis of the Bloch sphere.
 * This is a generalization of the X gate.
 */
template <typename fp_type>
struct XPowGate {
  static constexpr GateKind kind = kXPowGate;
  static constexpr char name[] = "XPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type gc = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type gs = std::sin(pi * exponent * (0.5 + global_shift));

    return CreateGate<GateCirq<fp_type>, XPowGate>(
        time, {q0}, {c * gc, c * gs, s * gs, -s * gc,
                     s * gs, -s * gc, c * gc, c * gs},
        {exponent, global_shift});
  }
};

/**
 * A gate that rotates around the Y axis of the Bloch sphere.
 * This is a generalization of the Y gate.
 */
template <typename fp_type>
struct YPowGate {
  static constexpr GateKind kind = kYPowGate;
  static constexpr char name[] = "YPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type gc = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type gs = std::sin(pi * exponent * (0.5 + global_shift));

    return CreateGate<GateCirq<fp_type>, YPowGate>(
        time, {q0}, {c * gc, c * gs, -s * gc, -s * gs,
                     s * gc, s * gs, c * gc, c * gs}, {exponent, global_shift});
  }
};

/**
 * A gate that rotates around the Z axis of the Bloch sphere.
 * This is a generalization of the Z gate.
 */
template <typename fp_type>
struct ZPowGate {
  static constexpr GateKind kind = kZPowGate;
  static constexpr char name[] = "ZPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type c = std::cos(pi * exponent);
    fp_type s = std::sin(pi * exponent);
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);

    return CreateGate<GateCirq<fp_type>, ZPowGate>(
        time, {q0}, {gc, gs, 0, 0, 0, 0, c * gc - s * gs, c * gs + s * gc},
        {exponent, global_shift});
  }
};

/**
 * A gate that rotates around the X+Z axis of the Bloch sphere.
 * This is a generalization of the Hadamard gate.
 */
template <typename fp_type>
struct HPowGate {
  static constexpr GateKind kind = kHPowGate;
  static constexpr char name[] = "HPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);
  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type gc = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type gs = std::sin(pi * exponent * (0.5 + global_shift));

    fp_type a = s * gs * is2;
    fp_type b = s * gc * is2;

    return CreateGate<GateCirq<fp_type>, HPowGate>(
        time, {q0}, {c * gc + a, c * gs - b, a, -b,
                     a, -b, c * gc - a, c * gs + b}, {exponent, global_shift});
  }
};

/**
 * A gate that applies a phase to the |11⟩ state of two qubits.
 * This is a generalization of the CZ gate.
 */
template <typename fp_type>
struct CZPowGate {
  static constexpr GateKind kind = kCZPowGate;
  static constexpr char name[] = "CZPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type ec = std::cos(pi * exponent * (1 + global_shift));
    fp_type es = std::sin(pi * exponent * (1 + global_shift));

    return CreateGate<GateCirq<fp_type>, CZPowGate>(
        time, {q0, q1}, {gc, gs, 0, 0, 0, 0, 0, 0,
                         0, 0, gc, gs, 0, 0, 0, 0,
                         0, 0, 0, 0, gc, gs, 0, 0,
                         0, 0, 0, 0, 0, 0, ec, es}, {exponent, global_shift});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type exponent, fp_type global_shift) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type ec = std::cos(pi * exponent * (1 + global_shift));
    fp_type es = std::sin(pi * exponent * (1 + global_shift));

    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 0, 0}, {gc, gs, 0, 0, 0, 0, gc, gs}},
      {{0, 0, 0, 0, 0, 0, 1, 0}, {gc, gs, 0, 0, 0, 0, ec, es}},
    };
  }
};

/**
 * A gate that applies a controlled power of an X gate.
 * This is a generalization of the CX (or CNOT) gate.
 */
template <typename fp_type>
struct CXPowGate {
  static constexpr GateKind kind = kCXPowGate;
  static constexpr char name[] = "CXPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type ec = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type es = std::sin(pi * exponent * (0.5 + global_shift));

    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, CXPowGate>(
        time, {q0, q1}, {gc, gs, 0, 0, 0, 0, 0, 0,
                         0, 0, c * ec, c * es, 0, 0, s * es, -s * ec,
                         0, 0, 0, 0, gc, gs, 0, 0,
                         0, 0, s * es, -s * ec, 0, 0, c * ec, c * es},
        {exponent, global_shift});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type exponent, fp_type global_shift) {
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type ec = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type es = std::sin(pi * exponent * (0.5 + global_shift));

    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 0, 0}, {gc, gs, 0, 0, 0, 0, gc, gs}},
      {{0, 0, 0, 0, 0, 0, 1, 0}, {c * ec, c * es, s * es, -s * ec,
                                  s * es, -s * ec, c * ec, c * es}},
    };
  }
};

/**
 * The `(exponent = phi/pi, global_shift = -0.5)` instance of XPowGate.
 * This is a generalization of the X gate with a fixed global phase.
 * This is a function in Cirq.
 */
template <typename fp_type>
struct rx {
  static constexpr GateKind kind = krx;
  static constexpr char name[] = "rx";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type c = std::cos(-0.5 * phi);
    fp_type s = std::sin(-0.5 * phi);

    return CreateGate<GateCirq<fp_type>, rx>(
        time, {q0}, {c, 0, 0, s, 0, s, c, 0}, {phi});
  }
};

/**
 * The `(exponent = phi/pi, global_shift = -0.5)` instance of YPowGate.
 * This is a generalization of the Y gate with a fixed global phase.
 * This is a function in Cirq.
 */
template <typename fp_type>
struct ry {
  static constexpr GateKind kind = kry;
  static constexpr char name[] = "ry";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type c = std::cos(-0.5 * phi);
    fp_type s = std::sin(-0.5 * phi);

    return CreateGate<GateCirq<fp_type>, ry>(
        time, {q0}, {c, 0, s, 0, -s, 0, c, 0}, {phi});
  }
};

/**
 * The `(exponent = phi/pi, global_shift = -0.5)` instance of ZPowGate.
 * This is a generalization of the Z gate with a fixed global phase.
 * This is a function in Cirq.
 */
template <typename fp_type>
struct rz {
  static constexpr GateKind kind = krz;
  static constexpr char name[] = "rz";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type c = std::cos(-0.5 * phi);
    fp_type s = std::sin(-0.5 * phi);

    return CreateGate<GateCirq<fp_type>, rz>(
        time, {q0}, {c, s, 0, 0, 0, 0, c, -s}, {phi});
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of HPowGate.
 * This is the canonical Hadamard (or H) gate.
 */
template <typename fp_type>
struct H {
  static constexpr GateKind kind = kH;
  static constexpr char name[] = "H";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateCirq<fp_type>, H>(
        time, {q0}, {is2, 0, is2, 0, is2, 0, -is2, 0});
  }
};

/**
 * The `(exponent = 0.5, global_shift = 0)` instance of ZPowGate.
 * This is the canonical S gate.
 */
template <typename fp_type>
struct S {
  static constexpr GateKind kind = kS;
  static constexpr char name[] = "S";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateCirq<fp_type>, S>(
        time, {q0}, {1, 0, 0, 0, 0, 0, 0, 1});
  }
};

/**
 * The `(exponent = 0.25, global_shift = 0)` instance of ZPowGate.
 * This is the canonical T gate.
 */
template <typename fp_type>
struct T {
  static constexpr GateKind kind = kT;
  static constexpr char name[] = "T";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateCirq<fp_type>, T>(
        time, {q0}, {1, 0, 0, 0, 0, 0, is2, is2});
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of CZPowGate.
 * This is the canonical CZ gate.
 */
template <typename fp_type>
struct CZ {
  static constexpr GateKind kind = kCZ;
  static constexpr char name[] = "CZ";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateCirq<fp_type>, CZ>(
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

template <typename fp_type>
using CNotPowGate = CXPowGate<fp_type>;

/**
 * The `(exponent = 1, global_shift = 0)` instance of CXPowGate.
 * This is the canonical CX (or CNOT) gate.
 */
template <typename fp_type>
struct CX {
  static constexpr GateKind kind = kCX;
  static constexpr char name[] = "kCX";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = false;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, CX>(
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

template <typename fp_type>
using CNOT = CX<fp_type>;

// Gates from cirq/ops/pauli_gates.py:

/**
 * The `(exponent = 1, global_shift = 0)` instance of XPowGate.
 * This is the canonical Pauli X gate.
 */
template <typename fp_type>
struct X : public XPowGate<fp_type> {
  static constexpr GateKind kind = kX;
  static constexpr char name[] = "X";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateCirq<fp_type>, X>(
        time, {q0}, {0, 0, 1, 0, 1, 0, 0, 0});
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of YPowGate.
 * This is the canonical Pauli Y gate.
 */
template <typename fp_type>
struct Y : public YPowGate<fp_type> {
  static constexpr GateKind kind = kY;
  static constexpr char name[] = "Y";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateCirq<fp_type>, Y>(
        time, {q0}, {0, 0, 0, -1, 0, 1, 0, 0});
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of ZPowGate.
 * This is the canonical Pauli Z gate.
 */
template <typename fp_type>
struct Z : public ZPowGate<fp_type> {
  static constexpr GateKind kind = kZ;
  static constexpr char name[] = "Z";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0) {
    return CreateGate<GateCirq<fp_type>, Z>(
        time, {q0}, {1, 0, 0, 0, 0, 0, -1, 0});
  }
};

// Gates from cirq/ops/phased_x_gate.py:

/**
 * An XPowGate conjugated by ZPowGate%s.
 * Equivalent to the circuit `───Z^-p───X^t───Z^p───`.
 */
template <typename fp_type>
struct PhasedXPowGate {
  static constexpr GateKind kind = kPhasedXPowGate;
  static constexpr char name[] = "PhasedXPowGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type phase_exponent, fp_type exponent = 1,
                                  fp_type global_shift = 0) {
    fp_type pc = std::cos(pi * phase_exponent);
    fp_type ps = std::sin(pi * phase_exponent);
    fp_type ec = std::cos(pi * exponent);
    fp_type es = std::sin(pi * exponent);
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);

    fp_type ar = 0.5 * ((1 + ec) * gc - es * gs);
    fp_type ai = 0.5 * ((1 + ec) * gs + es * gc);
    fp_type br = -0.5 * ((-1 + ec) * gc - es * gs);
    fp_type bi = -0.5 * ((-1 + ec) * gs + es * gc);

    return CreateGate<GateCirq<fp_type>, PhasedXPowGate>(
        time, {q0}, {ar, ai, pc * br + ps * bi, pc * bi - ps * br,
                     pc * br - ps * bi, pc * bi + ps * br, ar, ai},
        {phase_exponent, exponent, global_shift});
  }
};

// Gates from cirq/ops/phased_x_z_gate.py:

/**
 * A PhasedXPowGate followed by a ZPowGate.
 * Equivalent to the circuit `───Z^(-a)──X^x──Z^a───Z^z───`.
 */
template <typename fp_type>
struct PhasedXZGate {
  static constexpr GateKind kind = kPhasedXZGate;
  static constexpr char name[] = "PhasedXZGate";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  fp_type x_exponent, fp_type z_exponent,
                                  fp_type axis_phase_exponent) {
    fp_type xc = std::cos(pi * x_exponent);
    fp_type xs = std::sin(pi * x_exponent);
    fp_type zc = std::cos(pi * z_exponent);
    fp_type zs = std::sin(pi * z_exponent);
    fp_type ac = std::cos(pi * axis_phase_exponent);
    fp_type as = std::sin(pi * axis_phase_exponent);

    fp_type br = 0.5 * (1 + xc);
    fp_type bi = 0.5 * xs;
    fp_type cr = -0.5 * (-1 + xc);
    fp_type ci = -0.5 * xs;
    fp_type dr = ac * zc - as * zs;
    fp_type di = ac * zs + as * zc;

    return CreateGate<GateCirq<fp_type>, PhasedXZGate>(
        time, {q0}, {br, bi, ac * cr + as * ci, ac * ci - as * cr,
                     dr * cr - di * ci, dr * ci + di * cr,
                     zc * br - zs * bi, zc * bi + zs * br},
        {x_exponent, z_exponent, axis_phase_exponent});
  }
};

// Gates from cirq/ops/parity_gates.py:

/**
 * The tensor product of two X gates, possibly raised to an exponent.
 */
template <typename fp_type>
struct XXPowGate {
  static constexpr GateKind kind = kXXPowGate;
  static constexpr char name[] = "XXPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent);
    fp_type s = std::sin(pi * exponent);
    fp_type ic = 0.5 * ((1 + c) * gc - s * gs);
    fp_type is = 0.5 * ((1 + c) * gs + s * gc);
    fp_type xc = 0.5 * ((1 - c) * gc + s * gs);
    fp_type xs = 0.5 * ((1 - c) * gs - s * gc);

    return CreateGate<GateCirq<fp_type>, XXPowGate>(
        time, {q0, q1}, {ic, is, 0, 0, 0, 0, xc, xs,
                         0, 0, ic, is, xc, xs, 0, 0,
                         0, 0, xc, xs, ic, is, 0, 0,
                         xc, xs, 0, 0, 0, 0, ic, is}, {exponent, global_shift});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type exponent, fp_type global_shift) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent);
    fp_type s = std::sin(pi * exponent);
    fp_type ic = 0.5 * ((1 + c) * gc - s * gs);
    fp_type is = 0.5 * ((1 + c) * gs + s * gc);
    fp_type xc = 0.5 * ((1 - c) * gc + s * gs);
    fp_type xs = 0.5 * ((1 - c) * gs - s * gc);

    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 1, 0}, {ic, is, 0, 0, 0, 0, ic, is}},
      {{0, 0, 1, 0, 1, 0, 0, 0}, {0, 0, xc, xs, xc, xs, 0, 0}},
    };
  }
};

/**
 * The tensor product of two Y gates, possibly raised to an exponent.
 */
template <typename fp_type>
struct YYPowGate {
  static constexpr GateKind kind = kYYPowGate;
  static constexpr char name[] = "YYPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent);
    fp_type s = std::sin(pi * exponent);
    fp_type ic = 0.5 * ((1 + c) * gc - s * gs);
    fp_type is = 0.5 * ((1 + c) * gs + s * gc);
    fp_type yc = 0.5 * ((1 - c) * gc + s * gs);
    fp_type ys = 0.5 * ((1 - c) * gs - s * gc);

    return CreateGate<GateCirq<fp_type>, YYPowGate>(
        time, {q0, q1}, {ic, is, 0, 0, 0, 0, -yc, -ys,
                         0, 0, ic, is, yc, ys, 0, 0,
                         0, 0, yc, ys, ic, is, 0, 0,
                         -yc, -ys, 0, 0, 0, 0, ic, is},
        {exponent, global_shift});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type exponent, fp_type global_shift) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent);
    fp_type s = std::sin(pi * exponent);
    fp_type ic = 0.5 * ((1 + c) * gc - s * gs);
    fp_type is = 0.5 * ((1 + c) * gs + s * gc);
    fp_type yc = 0.5 * ((1 - c) * gc + s * gs);
    fp_type ys = 0.5 * ((1 - c) * gs - s * gc);

    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 1, 0}, {ic, is, 0, 0, 0, 0, ic, is}},
      {{0, 0, 0, -1, 0, 1, 0, 0}, {0, 0, ys, -yc, -ys, yc, 0, 0}},
    };
  }
};

/**
 * The tensor product of two Z gates, possibly raised to an exponent.
 */
template <typename fp_type>
struct ZZPowGate {
  static constexpr GateKind kind = kZZPowGate;
  static constexpr char name[] = "ZZPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type zc = std::cos(pi * exponent * (1 + global_shift));
    fp_type zs = std::sin(pi * exponent * (1 + global_shift));

    return CreateGate<GateCirq<fp_type>, ZZPowGate>(
        time, {q0, q1}, {gc, gs, 0, 0, 0, 0, 0, 0,
                         0, 0, zc, zs, 0, 0, 0, 0,
                         0, 0, 0, 0, zc, zs, 0, 0,
                         0, 0, 0, 0, 0, 0, gc, gs}, {exponent, global_shift});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type exponent, fp_type global_shift) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent);
    fp_type s = std::sin(pi * exponent);
    fp_type ic = 0.5 * ((1 + c) * gc - s * gs);
    fp_type is = 0.5 * ((1 + c) * gs + s * gc);
    fp_type zc = 0.5 * ((1 - c) * gc + s * gs);
    fp_type zs = 0.5 * ((1 - c) * gs - s * gc);

    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, 1, 0}, {ic, is, 0, 0, 0, 0, ic, is}},
      {{1, 0, 0, 0, 0, 0, -1, 0}, {zc, zs, 0, 0, 0, 0, -zc, -zs}},
    };
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of XXPowGate.
 * This is the tensor product of two X gates.
 */
template <typename fp_type>
struct XX {
  static constexpr GateKind kind = kXX;
  static constexpr char name[] = "XX";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateCirq<fp_type>, XX>(
        time, {q0, q1}, {0, 0, 0, 0, 0, 0, 1, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0,
                         1, 0, 0, 0, 0, 0, 0, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{0, 0, 1, 0, 1, 0, 0, 0}, {0, 0, 1, 0, 1, 0, 0, 0}},
    };
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of YYPowGate.
 * This is the tensor product of two Y gates.
 */
template <typename fp_type>
struct YY {
  static constexpr GateKind kind = kYY;
  static constexpr char name[] = "YY";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateCirq<fp_type>, YY>(
        time, {q0, q1}, {0, 0, 0, 0, 0, 0, -1, 0,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0,
                         -1, 0, 0, 0, 0, 0, 0, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{0, 0, 0, -1, 0, 1, 0, 0}, {0, 0, 0, -1, 0, 1, 0, 0}},
    };
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of ZZPowGate.
 * This is the tensor product of two Z gates.
 */
template <typename fp_type>
struct ZZ {
  static constexpr GateKind kind = kZZ;
  static constexpr char name[] = "ZZ";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateCirq<fp_type>, ZZ>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, -1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, -1, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    return schmidt_decomp_type<fp_type>{
      {{1, 0, 0, 0, 0, 0, -1, 0}, {1, 0, 0, 0, 0, 0, -1, 0}},
    };
  }
};

// Gates from cirq/ops/swap_gates.py:

/**
 * The SWAP gate, possibly raised to a power. Exchanges qubits.
 */
template <typename fp_type>
struct SwapPowGate {
  static constexpr GateKind kind = kSwapPowGate;
  static constexpr char name[] = "SwapPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);
  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type ec = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type es = std::sin(pi * exponent * (0.5 + global_shift));

    return CreateGate<GateCirq<fp_type>, SwapPowGate>(
        time, {q0, q1}, {gc, gs, 0, 0, 0, 0, 0, 0,
                         0, 0, c * ec, c * es, s * es, -s * ec, 0, 0,
                         0, 0, s * es, -s * ec, c * ec, c * es, 0, 0,
                         0, 0, 0, 0, 0, 0, gc, gs}, {exponent, global_shift});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type exponent, fp_type global_shift) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type ec = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type es = std::sin(pi * exponent * (0.5 + global_shift));

    return schmidt_decomp_type<fp_type>{
      {{h, 0, 0, 0, 0, 0, h, 0}, {gc + c * ec, gs + c * es, 0, 0,
                                  0, 0, gc + c * ec, gs + c * es}},
      {{0, 0, h, 0, h, 0, 0, 0}, {0, 0, s * es, -s * ec,
                                  s * es, -s * ec, 0, 0}},
      {{0, 0, 0, -h, 0, h, 0, 0}, {0, 0, -s * ec, -s * es,
                                   s * ec, s * es, 0, 0}},
      {{h, 0, 0, 0, 0, 0, -h, 0}, {gc - c * ec, gs - c * es, 0, 0,
                                   0, 0, -gc + c * ec, -gs + c * es}},
    };
  }
};

/**
 * Rotates the |01⟩ vs |10⟩ subspace of two qubits around its Bloch X-axis.
 * This is a generalization of the ISWAP gate.
 */
template <typename fp_type>
struct ISwapPowGate {
  static constexpr GateKind kind = kISwapPowGate;
  static constexpr char name[] = "ISwapPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);
  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);

    return CreateGate<GateCirq<fp_type>, ISwapPowGate>(
        time, {q0, q1}, {gc, gs, 0, 0, 0, 0, 0, 0,
                         0, 0, c * gc, c * gs, -s * gs, s * gc, 0, 0,
                         0, 0, -s * gs, s * gc, c * gc, c * gs, 0, 0,
                         0, 0, 0, 0, 0, 0, gc, gs}, {exponent, global_shift});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type exponent, fp_type global_shift) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);

    return schmidt_decomp_type<fp_type>{
      {{h, 0, 0, 0, 0, 0, h, 0}, {gc + c * gc, gs + c * gs, 0, 0,
                                  0, 0, gc + c * gc, gs + c * gs}},
      {{0, 0, h, 0, h, 0, 0, 0}, {0, 0, -s * gs, s * gc,
                                  -s * gs, s * gc, 0, 0}},
      {{0, 0, 0, -h, 0, h, 0, 0}, {0, 0, s * gc, s * gs,
                                   -s * gc, -s * gs, 0, 0}},
      {{h, 0, 0, 0, 0, 0, -h, 0}, {gc - c * gc, gs - c * gs, 0, 0,
                                   0, 0, -gc + c * gc, -gs + c * gs}},
    };
  }
};

/**
 * The `(exponent = 2*phi/pi, global_shift = 0)` instance of ISwapPowGate.
 * This is a generalization of the ISWAP gate with a fixed global phase of zero.
 * This is a function in Cirq.
 */
template <typename fp_type>
struct riswap {
  static constexpr GateKind kind = kriswap;
  static constexpr char name[] = "riswap";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);
  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type phi) {
    fp_type c = std::cos(phi);
    fp_type s = std::sin(phi);

    return CreateGate<GateCirq<fp_type>, riswap>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, c, 0, 0, s, 0, 0,
                         0, 0, 0, s, c, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0}, {phi});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(fp_type phi) {
    fp_type c = std::cos(phi);
    fp_type s = std::sin(phi);

    return schmidt_decomp_type<fp_type>{
      {{h, 0, 0, 0, 0, 0, h, 0}, {1 + c, 0, 0, 0, 0, 0, 1 + c, 0}},
      {{0, 0, h, 0, h, 0, 0, 0}, {0, 0, 0, s, 0, s, 0, 0}},
      {{0, 0, 0, -h, 0, h, 0, 0}, {0, 0, s, 0, -s, 0, 0, 0}},
      {{h, 0, 0, 0, 0, 0, -h, 0}, {1 - c, 0, 0, 0, 0, 0, -1 + c, 0}},
    };
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of SwapPowGate.
 * This is the canonical SWAP gate.
 */
template <typename fp_type>
struct SWAP {
  static constexpr GateKind kind = kSWAP;
  static constexpr char name[] = "SWAP";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateCirq<fp_type>, SWAP>(
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
 * The `(exponent = 1, global_shift = 0)` instance of ISwapPowGate.
 * This is the canonical ISWAP gate.
 */
template <typename fp_type>
struct ISWAP {
  static constexpr GateKind kind = kISWAP;
  static constexpr char name[] = "ISWAP";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type h = static_cast<fp_type>(h_double);
  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return CreateGate<GateCirq<fp_type>, ISWAP>(
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

// Gates from cirq/ops/phased_iswap_gate.py:

/**
 * An ISwapPowGate conjugated by ZPowGate%s.
 * Equivalent to the composition `(Z^-p ⊗ Z^p) ISWAP^t (Z^p ⊗ Z^-p)`.
 */
template <typename fp_type>
struct PhasedISwapPowGate {
  static constexpr GateKind kind = kPhasedISwapPowGate;
  static constexpr char name[] = "PhasedISwapPowGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);
  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type phase_exponent = 0.25,
                                  fp_type exponent = 1.0) {
    fp_type fc = std::cos(2 * pi * phase_exponent);
    fp_type fs = std::sin(2 * pi * phase_exponent);
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);

    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, PhasedISwapPowGate>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, c, 0, s * fs, s * fc, 0, 0,
                         0, 0, -s * fs, s * fc, c, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0}, {phase_exponent, exponent});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(
      fp_type phase_exponent, fp_type exponent) {
    fp_type fc = std::cos(2 * pi * phase_exponent);
    fp_type fs = std::sin(2 * pi * phase_exponent);
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);

    return schmidt_decomp_type<fp_type>{
      {{h, 0, 0, 0, 0, 0, h, 0}, {1 + c, 0, 0, 0, 0, 0, 1 + c, 0}},
      {{0, 0, h, 0, h, 0, 0, 0}, {0, 0, s * fs, s * fc, -s * fs, s * fc, 0, 0}},
      {{0, 0, 0, -h, 0, h, 0, 0}, {0, 0, s * fc, -s * fs,
                                   -s * fc, -s * fs, 0, 0}},
      {{h, 0, 0, 0, 0, 0, -h, 0}, {1 - c, 0, 0, 0, 0, 0, -1 + c, 0}},
    };
  }
};

/**
 * The `(phase_exponent = 0.25, exponent = 2*phi/pi)` instance of
 * PhasedISwapPowGate.
 * This is the "Givens rotation" from numerical linear algebra.
 * This is a function in Cirq.
 */
template <typename fp_type>
struct givens {
  static constexpr GateKind kind = kgivens;
  static constexpr char name[] = "givens";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);
  static constexpr fp_type h = static_cast<fp_type>(h_double);

  static GateCirq<fp_type> Create(unsigned time, unsigned q0, unsigned q1,
                                  fp_type phi) {
    fp_type c = std::cos(phi);
    fp_type s = std::sin(phi);

    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, givens>(
        time, {q0, q1}, {1, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, c, 0, s, 0, 0, 0,
                         0, 0, -s, 0, c, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0}, {phi});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(fp_type phi) {
    fp_type c = std::cos(phi);
    fp_type s = std::sin(phi);

    return schmidt_decomp_type<fp_type>{
      {{h, 0, 0, 0, 0, 0, h, 0}, {1 + c, 0, 0, 0, 0, 0, 1 + c, 0}},
      {{0, 0, h, 0, h, 0, 0, 0}, {0, 0, s, 0, -s, 0, 0, 0}},
      {{0, 0, 0, -h, 0, h, 0, 0}, {0, 0, 0, -s, 0, -s, 0, 0}},
      {{h, 0, 0, 0, 0, 0, -h, 0}, {1 - c, 0, 0, 0, 0, 0, -1 + c, 0}},
    };
  }
};

// Gates from cirq/ops/fsim_gate.py:

/**
 * The fermionic simulation gate family. Contains all two-qubit interactions
 * that preserve excitations, up to single-qubit rotations and global phase.
 */
template <typename fp_type>
struct FSimGate {
  static constexpr GateKind kind = kFSimGate;
  static constexpr char name[] = "FSimGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = true;

  static constexpr fp_type is2 = static_cast<fp_type>(is2_double);

  static GateCirq<fp_type> Create(
      unsigned time, unsigned q0, unsigned q1, fp_type theta, fp_type phi) {
    if (phi < 0) {
      phi += 2 * 3.141592653589793;
    }

    fp_type ct = std::cos(theta);
    fp_type st = std::sin(theta);
    fp_type cp = std::cos(phi);
    fp_type sp = std::sin(phi);

    return CreateGate<GateCirq<fp_type>, FSimGate>(
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

// Gates from cirq/ops/two_qubit_diagonal_gate.py:

/**
 * A two-qubit diagonal gate.
 */
template <typename fp_type>
struct TwoQubitDiagonalGate {
  static constexpr GateKind kind = kTwoQubitDiagonalGate;
  static constexpr char name[] = "TwoQubitDiagonalGate";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time,
                                  unsigned q0, unsigned q1,
                                  const std::vector<fp_type>& angles) {
    std::vector<fp_type> cs;
    std::vector<fp_type> ss;
    cs.reserve(4);
    ss.reserve(4);

    for (std::size_t i = 0; i < angles.size(); ++i) {
      cs.push_back(std::cos(angles[i]));
      ss.push_back(std::sin(angles[i]));
    }

    for (std::size_t i = angles.size(); i < 4; ++i) {
      cs.push_back(1);
      ss.push_back(0);
    }

    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, TwoQubitDiagonalGate>(
        time, {q0, q1}, {cs[0], ss[0], 0, 0, 0, 0, 0, 0,
                         0, 0, cs[2], ss[2], 0, 0, 0, 0,
                         0, 0, 0, 0, cs[1], ss[1], 0, 0,
                         0, 0, 0, 0, 0, 0, cs[3], ss[3]});
  }
};

// Gates from cirq/ops/three_qubit_gates.py:

/**
 * A three-qubit diagonal gate.
 */
template <typename fp_type>
struct ThreeQubitDiagonalGate {
  static constexpr GateKind kind = kThreeQubitDiagonalGate;
  static constexpr char name[] = "ThreeQubitDiagonalGate";
  static constexpr unsigned num_qubits = 3;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time,
                                  unsigned q0, unsigned q1, unsigned q2,
                                  const std::vector<fp_type>& angles) {
    std::vector<fp_type> cs;
    std::vector<fp_type> ss;
    cs.reserve(8);
    ss.reserve(8);

    for (std::size_t i = 0; i < angles.size(); ++i) {
      cs.push_back(std::cos(angles[i]));
      ss.push_back(std::sin(angles[i]));
    }

    for (std::size_t i = angles.size(); i < 8; ++i) {
      cs.push_back(1);
      ss.push_back(0);
    }

    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, ThreeQubitDiagonalGate>(
        time, {q0, q1, q2},
        {cs[0], ss[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, cs[4], ss[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, cs[2], ss[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, cs[6], ss[6], 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, cs[1], ss[1], 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cs[5], ss[5], 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cs[3], ss[3], 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, cs[7], ss[7]});
  }
};

/**
 * A gate that applies a phase to the |111⟩ state of three qubits.
 * This is a generalization of the CCZ gate.
 */
template <typename fp_type>
struct CCZPowGate {
  static constexpr GateKind kind = kCCZPowGate;
  static constexpr char name[] = "CCZPowGate";
  static constexpr unsigned num_qubits = 3;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time,
                                  unsigned q0, unsigned q1, unsigned q2,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type ec = std::cos(pi * exponent * (1 + global_shift));
    fp_type es = std::sin(pi * exponent * (1 + global_shift));

    return CreateGate<GateCirq<fp_type>, CCZPowGate>(
        time, {q0, q1, q2}, {gc, gs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, gc, gs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, gc, gs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, gc, gs, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, gc, gs, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gc, gs, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gc, gs, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ec, es},
                            {exponent, global_shift});
  }
};

/**
 * A gate that applies a doubly-controlled power of an X gate.
 * This is a generalization of the CCX (or CCNOT) gate.
 */
template <typename fp_type>
struct CCXPowGate {
  static constexpr GateKind kind = kCCXPowGate;
  static constexpr char name[] = "CCXPowGate";
  static constexpr unsigned num_qubits = 3;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time,
                                  unsigned q0, unsigned q1, unsigned q2,
                                  fp_type exponent, fp_type global_shift = 0) {
    fp_type c = std::cos(pi * exponent * 0.5);
    fp_type s = std::sin(pi * exponent * 0.5);
    fp_type gc = std::cos(pi * exponent * global_shift);
    fp_type gs = std::sin(pi * exponent * global_shift);
    fp_type ec = std::cos(pi * exponent * (0.5 + global_shift));
    fp_type es = std::sin(pi * exponent * (0.5 + global_shift));

    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, CCXPowGate>(
        time, {q0, q1, q2},
        {gc, gs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, gc, gs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, gc, gs, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, c * ec, c * es, 0, 0, 0, 0, 0, 0, s * es, -s * ec,
         0, 0, 0, 0, 0, 0, 0, 0, gc, gs, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gc, gs, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gc, gs, 0, 0,
         0, 0, 0, 0, 0, 0, s * es, -s * ec, 0, 0, 0, 0, 0, 0, c * ec, c * es},
        {exponent, global_shift});
  }
};

/**
 * A controlled swap gate (the Fredkin gate).
 */
template <typename fp_type>
struct CSwapGate {
  static constexpr GateKind kind = kCSwapGate;
  static constexpr char name[] = "CSwapGate";
  static constexpr unsigned num_qubits = 3;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time,
                                  unsigned q0, unsigned q1, unsigned q2) {
    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, CSwapGate>(
        time, {q0, q1, q2}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of CCZPowGate.
 * This is the canonical doubly-controlled Z gate.
 */
template <typename fp_type>
struct CCZ {
  static constexpr GateKind kind = kCCZ;
  static constexpr char name[] = "CCZ";
  static constexpr unsigned num_qubits = 3;
  static constexpr bool symmetric = true;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time,
                                  unsigned q0, unsigned q1, unsigned q2) {
    return CreateGate<GateCirq<fp_type>, CCZ>(
        time, {q0, q1, q2}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0});
  }
};

/**
 * The `(exponent = 1, global_shift = 0)` instance of CCXPowGate.
 * This is the canonical doubly-controlled X gate (the TOFFOLI gate).
 */
template <typename fp_type>
struct CCX {
  static constexpr GateKind kind = kCCX;
  static constexpr char name[] = "CCX";
  static constexpr unsigned num_qubits = 3;
  static constexpr bool symmetric = false;

  static constexpr fp_type pi = static_cast<fp_type>(pi_double);

  static GateCirq<fp_type> Create(unsigned time,
                                  unsigned q0, unsigned q1, unsigned q2) {
    // Matrix is in this form because the simulator uses inverse qubit order.
    return CreateGate<GateCirq<fp_type>, CCX>(
        time, {q0, q1, q2}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  }
};

template <typename fp_type>
using CCNotPowGate = CCXPowGate<fp_type>;

template <typename fp_type>
using TOFFOLI = CCX<fp_type>;

template <typename fp_type>
using CCNOT = CCX<fp_type>;

template <typename fp_type>
using CSWAP = CSwapGate<fp_type>;

template <typename fp_type>
using FREDKIN = CSwapGate<fp_type>;

// Gates from cirq/ops/matrix_gates.py:

/**
 * A one-qubit gate defined entirely by its matrix.
 */
template <typename fp_type>
struct MatrixGate1 {
  static constexpr GateKind kind = kMatrixGate1;
  static constexpr char name[] = "MatrixGate1";
  static constexpr unsigned num_qubits = 1;
  static constexpr bool symmetric = true;

  static GateCirq<fp_type> Create(unsigned time, unsigned q0,
                                  const Matrix<fp_type>& m) {
    auto m2 = m;
    return
        CreateGate<GateCirq<fp_type>, MatrixGate1>(time, {q0}, std::move(m2));
  }
};

/**
 * A two-qubit gate defined entirely by its matrix.
 */
template <typename fp_type>
struct MatrixGate2 {
  static constexpr GateKind kind = kMatrixGate2;
  static constexpr char name[] = "MatrixGate2";
  static constexpr unsigned num_qubits = 2;
  static constexpr bool symmetric = false;

  template <typename M = Matrix<fp_type>>
  static GateCirq<fp_type> Create(
      unsigned time, unsigned q0, unsigned q1, M&& m) {
    return CreateGate<GateCirq<fp_type>, MatrixGate2>(time, {q1, q0},
                                                      std::forward<M>(m));
  }
};

/**
 * A multi-qubit gate defined entirely by its matrix.
 */
template <typename fp_type>
struct MatrixGate {
  static constexpr GateKind kind = kMatrixGate;
  static constexpr char name[] = "MatrixGate";
  static constexpr bool symmetric = false;

  template <typename M = Matrix<fp_type>>
  static GateCirq<fp_type> Create(unsigned time,
                                  std::vector<unsigned> qubits, M&& m) {
    std::reverse(qubits.begin(), qubits.end());
    return CreateGate<GateCirq<fp_type>, MatrixGate>(time, std::move(qubits),
                                                     std::forward<M>(m));
  }
};

}  // namesapce Cirq

template <typename fp_type>
inline schmidt_decomp_type<fp_type> GetSchmidtDecomp(
    Cirq::GateKind kind, const std::vector<fp_type>& params) {
  switch (kind) {
  case Cirq::kI2:
    return Cirq::I2<fp_type>::SchmidtDecomp();
  case Cirq::kCZPowGate:
    return Cirq::CZPowGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  case Cirq::kCXPowGate:
    return Cirq::CXPowGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  case Cirq::kCZ:
    return Cirq::CZ<fp_type>::SchmidtDecomp();
  case Cirq::kCX:
    return Cirq::CX<fp_type>::SchmidtDecomp();
  case Cirq::kXXPowGate:
    return Cirq::XXPowGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  case Cirq::kYYPowGate:
    return Cirq::YYPowGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  case Cirq::kZZPowGate:
    return Cirq::ZZPowGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  case Cirq::kXX:
    return Cirq::XX<fp_type>::SchmidtDecomp();
  case Cirq::kYY:
    return Cirq::YY<fp_type>::SchmidtDecomp();
  case Cirq::kZZ:
    return Cirq::ZZ<fp_type>::SchmidtDecomp();
  case Cirq::kSwapPowGate:
    return Cirq::SwapPowGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  case Cirq::kISwapPowGate:
    return Cirq::ISwapPowGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  case Cirq::kriswap:
    return Cirq::riswap<fp_type>::SchmidtDecomp(params[0]);
  case Cirq::kSWAP:
    return Cirq::SWAP<fp_type>::SchmidtDecomp();
  case Cirq::kISWAP:
    return Cirq::ISWAP<fp_type>::SchmidtDecomp();
  case Cirq::kPhasedISwapPowGate:
    return Cirq::PhasedISwapPowGate<fp_type>::SchmidtDecomp(
        params[0], params[1]);
  case Cirq::kgivens:
    return Cirq::givens<fp_type>::SchmidtDecomp(params[0]);
  case Cirq::kFSimGate:
    return Cirq::FSimGate<fp_type>::SchmidtDecomp(params[0], params[1]);
  default:
    // Single qubit gates of gates with unimplemented Schmidt decomposition.
    return schmidt_decomp_type<fp_type>{};
  }
}

}  // namespace qsim

#endif  // GATES_CIRQ_H_
