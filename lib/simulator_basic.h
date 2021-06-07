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


#include <algorithm>
#include <complex>
#include <cstdint>

#include "bits.h"
#include "statespace_basic.h"

namespace qsim {

/**
 * Quantum circuit simulator without vectorization.
 */
template <typename For, typename FP = float>
class SimulatorBasic final {
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
      ApplyGate1H(qs, matrix, state);
      break;
    case 2:
      ApplyGate2H(qs, matrix, state);
      break;
    case 3:
      ApplyGate3H(qs, matrix, state);
      break;
    case 4:
      ApplyGate4H(qs, matrix, state);
      break;
    case 5:
      ApplyGate5H(qs, matrix, state);
      break;
    case 6:
      ApplyGate6H(qs, matrix, state);
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
   * @param cmask Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cmask,
                           const fp_type* matrix, State& state) const {
    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
    case 1:
      ApplyControlledGate1H(qs, cqs, cmask, matrix, state);
      break;
    case 2:
      ApplyControlledGate2H(qs, cqs, cmask, matrix, state);
      break;
    case 3:
      ApplyControlledGate3H(qs, cqs, cmask, matrix, state);
      break;
    case 4:
      ApplyControlledGate4H(qs, cqs, cmask, matrix, state);
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
      return ExpectationValue1H(qs, matrix, state);
      break;
    case 2:
      return ExpectationValue2H(qs, matrix, state);
      break;
    case 3:
      return ExpectationValue3H(qs, matrix, state);
      break;
    case 4:
      return ExpectationValue4H(qs, matrix, state);
      break;
    case 5:
      return ExpectationValue5H(qs, matrix, state);
      break;
    case 6:
      return ExpectationValue6H(qs, matrix, state);
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
  void ApplyGate1H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[2], is[2];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 1;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, rstate);
  }

  void ApplyGate2H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[4], is[4];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 2;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, rstate);
  }

  void ApplyGate3H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[8], is[8];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 8; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, rstate);
  }

  void ApplyGate4H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[16], is[16];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]) | (16 * i & ms[4]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 16; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, rstate);
  }

  void ApplyGate5H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[32], is[32];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]) | (16 * i & ms[4]) | (32 * i & ms[5]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 32; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 32; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 32; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, rstate);
  }

  void ApplyGate6H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, State& state) const {
    uint64_t xs[6];
    uint64_t ms[7];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 6; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[6] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[5] - 1);

    uint64_t xss[64];
    for (unsigned i = 0; i < 64; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 6; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[64], is[64];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]) | (16 * i & ms[4]) | (32 * i & ms[5])
          | (64 * i & ms[6]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 64; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 64; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 64; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, rstate);
  }

  void ApplyControlledGate1H(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs,
                             uint64_t cmask, const fp_type* matrix,
                             State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[2], is[2];

      uint64_t k = bits::ExpandBits(i, num_qubits, emaskh) | cmaskh;
      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 1 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, rstate);
  }

  void ApplyControlledGate2H(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs,
                             uint64_t cmask, const fp_type* matrix,
                             State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[4], is[4];

      uint64_t k = bits::ExpandBits(i, num_qubits, emaskh) | cmaskh;
      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 2 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, rstate);
  }

  void ApplyControlledGate3H(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs,
                             uint64_t cmask, const fp_type* matrix,
                             State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[8], is[8];

      uint64_t k = bits::ExpandBits(i, num_qubits, emaskh) | cmaskh;
      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 8; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 3 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, rstate);
  }

  void ApplyControlledGate4H(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs,
                             uint64_t cmask, const fp_type* matrix,
                             State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    uint64_t emaskh = 0;

    for (auto q : cqs) {
      emaskh |= uint64_t{1} << q;
    }

    uint64_t cmaskh = bits::ExpandBits(cmask, state.num_qubits(), emaskh);

    for (auto q : qs) {
      emaskh |= uint64_t{1} << q;
    }

    emaskh = ~emaskh;

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                unsigned num_qubits, uint64_t cmaskh, uint64_t emaskh,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[16], is[16];

      uint64_t k = bits::ExpandBits(i, num_qubits, emaskh) | cmaskh;
      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 16; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        *(p0 + xss[l]) = rn;
        *(p0 + xss[l] + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 4 + cqs.size();
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss,
             state.num_qubits(), cmaskh, emaskh, rstate);
  }

  std::complex<double> ExpectationValue1H(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[0] - 1);

    uint64_t xss[2];
    for (unsigned i = 0; i < 2; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 1; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[2], is[2];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned l = 0; l < 2; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 2; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        re += rs[l] * rn + is[l] * in;
        im += rs[l] * in - is[l] * rn;
      }

      return std::complex<double>{re, im};
    };

    const fp_type* rstate = state.get();

    unsigned k = 1;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, rstate);
  }

  std::complex<double> ExpectationValue2H(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[1] - 1);

    uint64_t xss[4];
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 2; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[4], is[4];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned l = 0; l < 4; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 4; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        re += rs[l] * rn + is[l] * in;
        im += rs[l] * in - is[l] * rn;
      }

      return std::complex<double>{re, im};
    };

    const fp_type* rstate = state.get();

    unsigned k = 2;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, rstate);
  }

  std::complex<double> ExpectationValue3H(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    uint64_t xs[3];
    uint64_t ms[4];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 3; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[3] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[2] - 1);

    uint64_t xss[8];
    for (unsigned i = 0; i < 8; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 3; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[8], is[8];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 8; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned l = 0; l < 8; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 8; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        re += rs[l] * rn + is[l] * in;
        im += rs[l] * in - is[l] * rn;
      }

      return std::complex<double>{re, im};
    };

    const fp_type* rstate = state.get();

    unsigned k = 3;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, rstate);
  }

  std::complex<double> ExpectationValue4H(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    uint64_t xs[4];
    uint64_t ms[5];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 4; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[4] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[3] - 1);

    uint64_t xss[16];
    for (unsigned i = 0; i < 16; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 4; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[16], is[16];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]) | (16 * i & ms[4]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 16; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned l = 0; l < 16; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 16; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        re += rs[l] * rn + is[l] * in;
        im += rs[l] * in - is[l] * rn;
      }

      return std::complex<double>{re, im};
    };

    const fp_type* rstate = state.get();

    unsigned k = 4;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, rstate);
  }

  std::complex<double> ExpectationValue5H(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    uint64_t xs[5];
    uint64_t ms[6];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 5; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[5] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[4] - 1);

    uint64_t xss[32];
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 5; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[32], is[32];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]) | (16 * i & ms[4]) | (32 * i & ms[5]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 32; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned l = 0; l < 32; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 32; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        re += rs[l] * rn + is[l] * in;
        im += rs[l] * in - is[l] * rn;
      }

      return std::complex<double>{re, im};
    };

    const fp_type* rstate = state.get();

    unsigned k = 5;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, rstate);
  }

  std::complex<double> ExpectationValue6H(const std::vector<unsigned>& qs,
                                          const fp_type* matrix,
                                          const State& state) const {
    uint64_t xs[6];
    uint64_t ms[7];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 6; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[6] = ((uint64_t{1} << state.num_qubits()) - 1) ^ (xs[5] - 1);

    uint64_t xss[64];
    for (unsigned i = 0; i < 64; ++i) {
      uint64_t a = 0;
      for (uint64_t k = 0; k < 6; ++k) {
        if (((i >> k) & 1) == 1) {
          a += xs[k];
        }
      }
      xss[i] = a;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[64], is[64];

      uint64_t k = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2])
          | (8 * i & ms[3]) | (16 * i & ms[4]) | (32 * i & ms[5])
          | (64 * i & ms[6]);

      auto p0 = rstate + 2 * k;

      for (unsigned l = 0; l < 64; ++l) {
        rs[l] = *(p0 + xss[l]);
        is[l] = *(p0 + xss[l] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned l = 0; l < 64; ++l) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned n = 1; n < 64; ++n) {
          rn += rs[n] * v[j] - is[n] * v[j + 1];
          in += rs[n] * v[j + 1] + is[n] * v[j];

          j += 2;
        }

        re += rs[l] * rn + is[l] * in;
        im += rs[l] * in - is[l] * rn;
      }

      return std::complex<double>{re, im};
    };

    const fp_type* rstate = state.get();

    unsigned k = 6;
    unsigned n = state.num_qubits() > k ? state.num_qubits() - k : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, rstate);
  }

  For for_;
};

}  // namespace qsim

#endif  // SIMULATOR_BASIC_H_
