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

#ifndef UNITARY_CALCULATOR_BASIC_H_
#define UNITARY_CALCULATOR_BASIC_H_


#include <algorithm>
#include <complex>
#include <cstdint>

#include "bits.h"
#include "unitaryspace_basic.h"

namespace qsim {
namespace unitary {

/**
 * Quantum circuit unitary calculator without vectorization.
 */
template <typename For, typename FP = float>
class UnitaryCalculatorBasic final {
 public:
  using UnitarySpace = UnitarySpaceBasic<For, FP>;
  using Unitary = typename UnitarySpace::Unitary;
  using fp_type = typename UnitarySpace::fp_type;

  using StateSpace = UnitarySpace;
  using State = Unitary;

  template <typename... ForArgs>
  explicit UnitaryCalculatorBasic(unsigned num_qubits, ForArgs&&... args)
      : for_(args...), num_qubits_(num_qubits) {}

  /**
   * Applies a gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, Unitary& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      ApplyGate1H(qs, matrix, state);
      break;
    case 2:
      ApplyGate2H(qs, matrix, state);
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
                           const fp_type* matrix, Unitary& state) const {
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
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  unsigned SIMDRegisterSize() {
    return 1;
  }

 private:
  void ApplyGate1H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << num_qubits_) - 1) ^ (xs[0] - 1);

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
                unsigned num_qubits, fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[2], is[2];

      uint64_t row_size = uint64_t{1} << num_qubits;

      uint64_t ii = i / row_size;
      uint64_t c = i % row_size;
      uint64_t r = (1 * ii & ms[0]) | (2 * ii & ms[1]);

      auto p0 = rstate + 2 * row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = *(p0 + xss[l] * row_size);
        is[l] = *(p0 + xss[l] * row_size + 1);
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

        *(p0 + xss[l] * row_size) = rn;
        *(p0 + xss[l] * row_size + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 1;
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << num_qubits_;

    for_.Run(size * size2, f, matrix, ms, xss, num_qubits_, rstate);
  }

  void ApplyGate2H(const std::vector<unsigned>& qs,
                   const fp_type* matrix, Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << num_qubits_) - 1) ^ (xs[1] - 1);

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
                unsigned num_qubits, fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[4], is[4];

      uint64_t row_size = uint64_t{1} << num_qubits;

      uint64_t ii = i / row_size;
      uint64_t c = i % row_size;
      uint64_t r = (1 * ii & ms[0]) | (2 * ii & ms[1]) | (4 * ii & ms[2]);

      auto p0 = rstate + 2 * row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = *(p0 + xss[l] * row_size);
        is[l] = *(p0 + xss[l] * row_size + 1);
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

        *(p0 + xss[l] * row_size) = rn;
        *(p0 + xss[l] * row_size + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 2;
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << num_qubits_;

    for_.Run(size * size2, f, matrix, ms, xss, num_qubits_, rstate);
  }

  void ApplyControlledGate1H(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs,
                             uint64_t cmask, const fp_type* matrix,
                             Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    ms[1] = ((uint64_t{1} << num_qubits_) - 1) ^ (xs[0] - 1);

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

    uint64_t cmaskh = bits::ExpandBits(cmask, num_qubits_, emaskh);

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

      uint64_t row_size = uint64_t{1} << num_qubits;

      uint64_t ii = i / row_size;
      uint64_t c = i % row_size;
      uint64_t r = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + 2 * row_size * r + 2 * c;

      for (unsigned l = 0; l < 2; ++l) {
        rs[l] = *(p0 + xss[l] * row_size);
        is[l] = *(p0 + xss[l] * row_size + 1);
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

        *(p0 + xss[l] * row_size) = rn;
        *(p0 + xss[l] * row_size + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 1 + cqs.size();
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << num_qubits_;

    for_.Run(size * size2, f, matrix, ms, xss,
             num_qubits_, cmaskh, emaskh, rstate);
  }

  void ApplyControlledGate2H(const std::vector<unsigned>& qs,
                             const std::vector<unsigned>& cqs,
                             uint64_t cmask, const fp_type* matrix,
                             Unitary& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (qs[0] + 1);
    ms[0] = (uint64_t{1} << qs[0]) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (qs[i + 0] + 1);
      ms[i] = ((uint64_t{1} << qs[i + 0]) - 1) ^ (xs[i - 1] - 1);
    }
    ms[2] = ((uint64_t{1} << num_qubits_) - 1) ^ (xs[1] - 1);

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

    uint64_t cmaskh = bits::ExpandBits(cmask, num_qubits_, emaskh);

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

      uint64_t row_size = uint64_t{1} << num_qubits;

      uint64_t ii = i / row_size;
      uint64_t c = i % row_size;
      uint64_t r = bits::ExpandBits(ii, num_qubits, emaskh) | cmaskh;

      auto p0 = rstate + 2 * row_size * r + 2 * c;

      for (unsigned l = 0; l < 4; ++l) {
        rs[l] = *(p0 + xss[l] * row_size);
        is[l] = *(p0 + xss[l] * row_size + 1);
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

        *(p0 + xss[l] * row_size) = rn;
        *(p0 + xss[l] * row_size + 1) = in;
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 2 + cqs.size();
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;
    uint64_t size2 = uint64_t{1} << num_qubits_;

    for_.Run(size * size2, f, matrix, ms, xss,
             num_qubits_, cmaskh, emaskh, rstate);
  }

  For for_;
  unsigned num_qubits_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_BASIC_H_
