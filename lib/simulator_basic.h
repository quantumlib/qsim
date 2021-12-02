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

#ifndef SIMULATOR_BASIC_H_
#define SIMULATOR_BASIC_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

#include "simulator.h"
#include "statespace_basic.h"

namespace qsim {

/**
 * Quantum circuit simulator without vectorization.
 */
template <typename For, typename FP = float>
class SimulatorBasic final : public SimulatorBase {
 public:
  using StateSpace = StateSpaceBasic<For, FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  template <typename... ForArgs>
  explicit SimulatorBasic(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      ApplyGateH<1>(qs, matrix, state);
      break;
    case 2:
      ApplyGateH<2>(qs, matrix, state);
      break;
    case 3:
      ApplyGateH<3>(qs, matrix, state);
      break;
    case 4:
      ApplyGateH<4>(qs, matrix, state);
      break;
    case 5:
      ApplyGateH<5>(qs, matrix, state);
      break;
    case 6:
      ApplyGateH<6>(qs, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Applies a controlled gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cvals Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
    case 1:
      ApplyControlledGateH<1>(qs, cqs, cvals, matrix, state);
      break;
    case 2:
      ApplyControlledGateH<2>(qs, cqs, cvals, matrix, state);
      break;
    case 3:
      ApplyControlledGateH<3>(qs, cqs, cvals, matrix, state);
      break;
    case 4:
      ApplyControlledGateH<4>(qs, cqs, cvals, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Computes the expectation value of an operator using non-vectorized
   * instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      return ExpectationValueH<1>(qs, matrix, state);
      break;
    case 2:
      return ExpectationValueH<2>(qs, matrix, state);
      break;
    case 3:
      return ExpectationValueH<3>(qs, matrix, state);
      break;
    case 4:
      return ExpectationValueH<4>(qs, matrix, state);
      break;
    case 5:
      return ExpectationValueH<5>(qs, matrix, state);
      break;
    case 6:
      return ExpectationValueH<6>(qs, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }

    return 0;
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 1;
  }

 private:
  template <unsigned H>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      fp_type rs[hsize], is[hsize];

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = *(p0 + xss[k]);
        is[k] = *(p0 + xss[k] + 1);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn += rs[l] * v[j] - is[l] * v[j + 1];
          in += rs[l] * v[j + 1] + is[l] * v[j];

          j += 2;
        }

        *(p0 + xss[k]) = rn;
        *(p0 + xss[k] + 1) = in;
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, state.get());
  }

  template <unsigned H>
  void ApplyControlledGateH(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs,
                            uint64_t cvals, const fp_type* matrix,
                            State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t cvalsh, uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      fp_type rs[hsize], is[hsize];

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) == cvalsh) {
        auto p0 = rstate + 2 * ii;

        for (unsigned k = 0; k < hsize; ++k) {
          rs[k] = *(p0 + xss[k]);
          is[k] = *(p0 + xss[k] + 1);
        }

        uint64_t j = 0;

        for (unsigned k = 0; k < hsize; ++k) {
          rn = rs[0] * v[j] - is[0] * v[j + 1];
          in = rs[0] * v[j + 1] + is[0] * v[j];

          j += 2;

          for (unsigned l = 1; l < hsize; ++l) {
            rn += rs[l] * v[j] - is[l] * v[j + 1];
            in += rs[l] * v[j + 1] + is[l] * v[j];

            j += 2;
          }

          *(p0 + xss[k]) = rn;
          *(p0 + xss[k] + 1) = in;
        }
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    auto m = GetMasks7(state.num_qubits(), qs, cqs, cvals);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, m.cvalsh, m.cmaskh, state.get());
  }

  template <unsigned H>
  std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      fp_type rs[hsize], is[hsize];

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = *(p0 + xss[k]);
        is[k] = *(p0 + xss[k] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn += rs[l] * v[j] - is[l] * v[j + 1];
          in += rs[l] * v[j + 1] + is[l] * v[j];

          j += 2;
        }

        re += rs[k] * rn + is[k] * in;
        im += rs[k] * in - is[k] * rn;
      }

      return std::complex<double>{re, im};
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, state.get());
  }

  For for_;
};

}  // namespace qsim

#endif  // SIMULATOR_BASIC_H_
