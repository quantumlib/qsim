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

#ifndef GATES_DEF_H_
#define GATES_DEF_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "gate.h"
#include "matrix.h"

namespace qsim {

namespace detail {

template <typename Gate, typename GateDef, typename Params, typename Matrix>
inline Gate CreateGate(
    unsigned time, unsigned q0, Params&& params, Matrix&& matrix) {
  return Gate{GateDef::kind, time, GateDef::num_qubits, {q0}, false, false,
              std::move(params), std::move(matrix)};
}

template <typename Gate, typename GateDef>
inline Gate CreateStaticGate(unsigned time, unsigned q0) {
  Gate gate = {GateDef::kind, time, GateDef::num_qubits, {q0}, false, false};
  auto begin = GateDef::matrix.begin();
  std::copy(begin, begin + GateDef::matrix.size(), gate.matrix.begin());
  return gate;
}

template <typename Gate, typename GateDef, typename Params, typename Matrix>
inline Gate CreateGate(unsigned time, unsigned q0, unsigned q1,
                       Params&& params, Matrix&& matrix) {
  Gate gate = {GateDef::kind, time, GateDef::num_qubits, {q0, q1}, false,
               false, std::move(params), std::move(matrix)};
  if (q0 > q1) {
    gate.inverse = true;
    std::swap(gate.qubits[0], gate.qubits[1]);
    Matrix4Permute(gate.matrix);
  }
  return gate;
}

template <typename Gate, typename GateDef>
inline Gate CreateStaticGate(unsigned time, unsigned q0, unsigned q1) {
  Gate gate = {GateDef::kind, time, GateDef::num_qubits, {q0, q1}, false,
               false};
  auto begin = GateDef::matrix.begin();
  std::copy(begin, begin + GateDef::matrix.size(), gate.matrix.begin());
  if (q0 > q1) {
    gate.inverse = true;
    std::swap(gate.qubits[0], gate.qubits[1]);
    Matrix4Permute(gate.matrix);
  }
  return gate;
}

}  // namespace detail

constexpr double h = 0.5;
constexpr double is2 = 0.7071067811865475;

template <typename fp_type>
using Matrix1q = std::array<fp_type, 8>;

template <typename fp_type>
using Matrix2q = std::array<fp_type, 32>;

template <typename fp_type>
using schmidt_decomp_type = std::vector<std::array<Matrix1q<fp_type>, 2>>;

// 1-qubit gates.

template <typename fp_type>
struct GateId1 {
  static constexpr GateKind kind = kGateId1;
  static constexpr char name[] = "id1";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateId1>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateId1<fp_type>::matrix = {1, 0, 0, 0, 0, 0, 1, 0};

template <typename fp_type>
struct GateHd {
  static constexpr GateKind kind = kGateHd;
  static constexpr char name[] = "h";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateHd>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateHd<fp_type>::matrix = {is2, 0, is2, 0, is2, 0, -is2, 0};

template <typename fp_type>
struct GateT {
  static constexpr GateKind kind = kGateT;
  static constexpr char name[] = "t";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateT>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateT<fp_type>::matrix = {1, 0, 0, 0, 0, 0, is2, is2};

template <typename fp_type>
struct GateX {
  static constexpr GateKind kind = kGateX;
  static constexpr char name[] = "x";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateX>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateX<fp_type>::matrix = {0, 0, 1, 0, 1, 0, 0, 0};

template <typename fp_type>
struct GateY {
  static constexpr GateKind kind = kGateY;
  static constexpr char name[] = "y";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateY>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateY<fp_type>::matrix = {0, 0, 0, -1, 0, 1, 0, 0};

template <typename fp_type>
struct GateZ {
  static constexpr GateKind kind = kGateZ;
  static constexpr char name[] = "z";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateZ>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateZ<fp_type>::matrix = {1, 0, 0, 0, 0, 0, -1, 0};

template <typename fp_type>
struct GateX2 {
  static constexpr GateKind kind = kGateX2;
  static constexpr char name[] = "x_1_2";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateX2>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateX2<fp_type>::matrix = {h, h, h, -h, h, -h, h, h};

template <typename fp_type>
struct GateY2 {
  static constexpr GateKind kind = kGateY2;
  static constexpr char name[] = "y_1_2";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateY2>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateY2<fp_type>::matrix = {h, h, -h, -h, h, h, h, h};

template <typename fp_type>
struct GateRX {
  static constexpr GateKind kind = kGateRX;
  static constexpr char name[] = "rx";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type c = std::cos(phi2);
    fp_type s = std::sin(phi2);
    return detail::CreateGate<Gate<fp_type>, GateRX>(
        time, q0, std::vector<fp_type>{phi},
        std::array<fp_type, 32>{c, 0, 0, s, 0, s, c, 0});
  }
};

template <typename fp_type>
struct GateRY {
  static constexpr GateKind kind = kGateRY;
  static constexpr char name[] = "ry";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type c = std::cos(phi2);
    fp_type s = std::sin(phi2);
    return detail::CreateGate<Gate<fp_type>, GateRY>(
        time, q0, std::vector<fp_type>{phi},
        std::array<fp_type, 32>{c, 0, s, 0, -s, 0, c, 0});
  }
};

template <typename fp_type>
struct GateRZ {
  static constexpr GateKind kind = kGateRZ;
  static constexpr char name[] = "rz";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type c = std::cos(phi2);
    fp_type s = std::sin(phi2);
    return detail::CreateGate<Gate<fp_type>, GateRZ>(
        time, q0, std::vector<fp_type>{phi},
        std::array<fp_type, 32>{c, s, 0, 0, 0, 0, c, -s});
  }
};

template <typename fp_type>
struct GateRXY {
  static constexpr GateKind kind = kGateRXY;
  static constexpr char name[] = "rxy";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(
      unsigned time, unsigned q0, fp_type theta, fp_type phi) {
    fp_type phi2 = -0.5 * phi;
    fp_type cp = std::cos(phi2);
    fp_type sp = std::sin(phi2);
    fp_type ct = std::cos(theta) * sp;
    fp_type st = std::sin(theta) * sp;
    return detail::CreateGate<Gate<fp_type>, GateRXY>(
        time, q0, std::vector<fp_type>{phi},
        std::array<fp_type, 32>{cp, 0, st, ct, -st, ct, cp, 0});
  }
};

template <typename fp_type>
struct GateHZ2 {
  static constexpr GateKind kind = kGateHZ2;
  static constexpr char name[] = "hz_1_2";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateHZ2>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateHZ2<fp_type>::matrix = {h, h, 0, -is2, is2, 0, h, h};

template <typename fp_type>
struct GateS {
  static constexpr GateKind kind = kGateS;
  static constexpr char name[] = "s";
  static constexpr unsigned num_qubits = 1;

  static Gate<fp_type> Create(unsigned time, unsigned q0) {
    return detail::CreateStaticGate<Gate<fp_type>, GateS>(time, q0);
  }

  static Matrix1q<fp_type> matrix;
};

template <typename fp_type>
Matrix1q<fp_type> GateS<fp_type>::matrix = {1, 0, 0, 0, 0, 0, 0, 1};

// 2-qubit gates.

template <typename fp_type>
struct GateId2 {
  static constexpr GateKind kind = kGateId2;
  static constexpr char name[] = "id2";
  static constexpr unsigned num_qubits = 2;

  static Gate<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return detail::CreateStaticGate<Gate<fp_type>, GateId2>(time, q0, q1);
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    schmidt_decomp_type<fp_type> schmidt_decomp(1);
    schmidt_decomp[0][0] = {1, 0, 0, 0, 0, 0, 1, 0};
    schmidt_decomp[0][1] = {1, 0, 0, 0, 0, 0, 1, 0};
    return schmidt_decomp;
  }

  static Matrix2q<fp_type> matrix;
};

template <typename fp_type>
Matrix2q<fp_type> GateId2<fp_type>::matrix = {1, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 1, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 1, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 1, 0};

template <typename fp_type>
struct GateCZ {
  static constexpr GateKind kind = kGateCZ;
  static constexpr char name[] = "cz";
  static constexpr unsigned num_qubits = 2;

  static Gate<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return detail::CreateStaticGate<Gate<fp_type>, GateCZ>(time, q0, q1);
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    schmidt_decomp_type<fp_type> schmidt_decomp(2);
    schmidt_decomp[0][0] = {1, 0, 0, 0, 0, 0, 0, 0};
    schmidt_decomp[0][1] = {1, 0, 0, 0, 0, 0, 1, 0};
    schmidt_decomp[1][0] = {0, 0, 0, 0, 0, 0, 1, 0};
    schmidt_decomp[1][1] = {1, 0, 0, 0, 0, 0, -1, 0};
    return schmidt_decomp;
  }

  static Matrix2q<fp_type> matrix;
};

template <typename fp_type>
Matrix2q<fp_type> GateCZ<fp_type>::matrix = {1, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 1, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 1, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, -1, 0};

template <typename fp_type>
struct GateCNot {
  static constexpr GateKind kind = kGateCNot;
  static constexpr char name[] = "cnot";
  static constexpr unsigned num_qubits = 2;

  static Gate<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return detail::CreateStaticGate<Gate<fp_type>, GateCNot>(time, q0, q1);
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    schmidt_decomp_type<fp_type> schmidt_decomp(2);
    schmidt_decomp[0][0] = {1, 0, 0, 0, 0, 0, 0, 0};
    schmidt_decomp[0][1] = {1, 0, 0, 0, 0, 0, 1, 0};
    schmidt_decomp[1][0] = {0, 0, 0, 0, 0, 0, 1, 0};
    schmidt_decomp[1][1] = {0, 0, 1, 0, 1, 0, 0, 0};
    return schmidt_decomp;
  }

  static Matrix2q<fp_type> matrix;
};

// The matrix is in this form because the simulator uses the inverse order of
// qubits.
template <typename fp_type>
Matrix2q<fp_type> GateCNot<fp_type>::matrix = {1, 0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 1, 0,
                                               0, 0, 0, 0, 1, 0, 0, 0,
                                               0, 0, 1, 0, 0, 0, 0, 0};

template <typename fp_type>
struct GateIS {
  static constexpr GateKind kind = kGateIS;
  static constexpr char name[] = "is";
  static constexpr unsigned num_qubits = 2;

  static Gate<fp_type> Create(unsigned time, unsigned q0, unsigned q1) {
    return detail::CreateStaticGate<Gate<fp_type>, GateIS>(time, q0, q1);
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp() {
    schmidt_decomp_type<fp_type> schmidt_decomp(4);
    schmidt_decomp[0][0] = {is2, 0, 0, 0, 0, 0, is2, 0};
    schmidt_decomp[0][1] = {is2, 0, 0, 0, 0, 0, is2, 0};
    schmidt_decomp[1][0] = {0, 0, 0.5, -0.5, 0.5, -0.5, 0, 0};
    schmidt_decomp[1][1] = {0, 0, 0.5, -0.5, 0.5, -0.5, 0, 0};
    schmidt_decomp[2][0] = {0, 0, -0.5, -0.5, 0.5, 0.5, 0, 0};
    schmidt_decomp[2][1] = {0, 0, -0.5, -0.5, 0.5, 0.5, 0, 0};
    schmidt_decomp[3][0] = {is2, 0, 0, 0, 0, 0, -is2, 0};
    schmidt_decomp[3][1] = {is2, 0, 0, 0, 0, 0, -is2, 0};
    return schmidt_decomp;
  }

  static Matrix2q<fp_type> matrix;
};

template <typename fp_type>
Matrix2q<fp_type> GateIS<fp_type>::matrix = {1, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, -1, 0, 0,
                                             0, 0, 0, -1, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 1, 0};

template <typename fp_type>
struct GateFS {
  static constexpr GateKind kind = kGateFS;
  static constexpr char name[] = "fs";
  static constexpr unsigned num_qubits = 2;

  static Gate<fp_type> Create(
      unsigned time, unsigned q0, unsigned q1, fp_type theta, fp_type phi) {
    if (phi < 0) {
      phi += 2 * 3.141592653589793;
    }

    fp_type ct = std::cos(theta);
    fp_type st = std::sin(theta);
    fp_type cp = std::cos(phi);
    fp_type sp = std::sin(phi);

    return detail::CreateGate<Gate<fp_type>, GateFS>(
        time, q0, q1, std::vector<fp_type>{theta, phi},
        std::array<fp_type, 32>{1, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, ct, 0, 0, -st, 0, 0,
                                0, 0, 0, -st, ct, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, cp, -sp});
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

    schmidt_decomp_type<fp_type> schmidt_decomp(4);
    schmidt_decomp[0][0] = {a, b, 0, 0, 0, 0, c, d};
    schmidt_decomp[0][1] = {a, b, 0, 0, 0, 0, c, d};
    schmidt_decomp[1][0] = {0, 0, st2, -st2, st2, -st2, 0, 0};
    schmidt_decomp[1][1] = {0, 0, st2, -st2, st2, -st2, 0, 0};
    schmidt_decomp[2][0] = {0, 0, -st2, -st2, st2, st2, 0, 0};
    schmidt_decomp[2][1] = {0, 0, -st2, -st2, st2, st2, 0, 0};
    schmidt_decomp[3][0] = {e, f, 0, 0, 0, 0, g, h};
    schmidt_decomp[3][1] = {e, f, 0, 0, 0, 0, g, h};
    return schmidt_decomp;
  }
};

template <typename fp_type>
struct GateCP {
  static constexpr GateKind kind = kGateCP;
  static constexpr char name[] = "cp";
  static constexpr unsigned num_qubits = 2;

  static Gate<fp_type> Create(
      unsigned time, unsigned q0, unsigned q1, fp_type phi) {
    fp_type cp = std::cos(phi);
    fp_type sp = std::sin(phi);
    return detail::CreateGate<Gate<fp_type>, GateCP>(
        time, q0, q1, std::vector<fp_type>{phi},
        std::array<fp_type, 32>{1, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 1, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 1, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, cp, -sp});
  }

  static schmidt_decomp_type<fp_type> SchmidtDecomp(fp_type phi) {
    fp_type cp = std::cos(phi);
    fp_type sp = std::sin(phi);

    schmidt_decomp_type<fp_type> schmidt_decomp(2);
    schmidt_decomp[0][0] = {1, 0, 0, 0, 0, 0, 0, 0};
    schmidt_decomp[0][1] = {1, 0, 0, 0, 0, 0, 1, 0};
    schmidt_decomp[1][0] = {0, 0, 0, 0, 0, 0, 1, 0};
    schmidt_decomp[1][1] = {1, 0, 0, 0, 0, 0, cp, -sp};
    return schmidt_decomp;
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

#endif  // GATES_DEF_H_
