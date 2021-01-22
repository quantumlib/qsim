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

#include <cstdint>
#include <vector>

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
  void ApplyGate(const std::vector<unsigned>& qs, const fp_type* matrix,
                 Unitary& state) const {
    if (qs.size() == 1) {
      ApplyGate1(qs[0], matrix, state);
    } else if (qs.size() == 2) {
      // Assume qs[0] < qs[1].
      ApplyGate2(qs[0], qs[1], matrix, state);
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
    if (qs.size() == 1) {
      ApplyControlledGate1(qs[0], cqs, cmask, matrix, state);
    } else if (qs.size() == 2) {
      ApplyControlledGate2(qs[0], qs[1], cqs, cmask, matrix, state);
    }
  }

 private:
  void ApplyControlledGate1(unsigned q0, const std::vector<unsigned>& cqs,
                            uint64_t cmask, const fp_type* matrix,
                            State& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (q0 + 1);
    ms[0] = (uint64_t{1} << q0) - 1;
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

    emaskh |= uint64_t{1} << q0;

    emaskh = ~emaskh;

    auto f = [](unsigned n, unsigned m, uint64_t ii, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, unsigned n_qb,
                unsigned sqrt_size, uint64_t cmaskh, uint64_t emaskh,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[4], is[4];

      auto row_size = uint64_t{1} << n_qb;

      uint64_t i = ii % sqrt_size;
      uint64_t j = ii / sqrt_size;

      uint64_t col_loc = (1 * i & ms[0]) | (2 * i & ms[1]);
      uint64_t row_loc = bits::ExpandBits(j, n_qb, emaskh) | cmaskh;

      auto p0 = rstate + row_size * 2 * row_loc + 2 * col_loc;

      for (unsigned l = 0; l < 2; ++l) {
        for (unsigned k = 0; k < 2; ++k) {
          rs[2 * l + k] = *(p0 + xss[l] * row_size + xss[k]);
          is[2 * l + k] = *(p0 + xss[l] * row_size + xss[k] + 1);
        }
      }

      for (unsigned l = 0; l < 2; l++) {
        uint64_t j = 0;
        for (unsigned k = 0; k < 2; ++k) {
          rn = rs[l] * v[j] - is[l] * v[j + 1];
          in = rs[l] * v[j + 1] + is[l] * v[j];
          j += 2;

          for (unsigned p = 1; p < 2; ++p) {
            rn += rs[2 * p + l] * v[j] - is[2 * p + l] * v[j + 1];
            in += rs[2 * p + l] * v[j + 1] + is[2 * p + l] * v[j];

            j += 2;
          }
          *(p0 + xss[k] * row_size + xss[l]) = rn;
          *(p0 + xss[k] * row_size + xss[l] + 1) = in;
        }
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 1 + cqs.size();
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned kk = 1;
    unsigned nn = num_qubits_ > kk ? num_qubits_ - kk : 0;
    uint64_t size2 = uint64_t{1} << nn;

    for_.Run(size * size2, f, matrix, ms, xss, num_qubits_, size2, cmaskh,
             emaskh, rstate);
  }

  void ApplyControlledGate2(unsigned q0, unsigned q1,
                            const std::vector<unsigned>& cqs, uint64_t cmask,
                            const fp_type* matrix, State& state) const {
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (q0 + 1);
    ms[0] = (uint64_t{1} << q0) - 1;

    xs[1] = uint64_t{1} << (q1 + 1);
    ms[1] = ((uint64_t{1} << q1) - 1) ^ (xs[0] - 1);

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

    emaskh |= uint64_t{1} << q0;
    emaskh |= uint64_t{1} << q1;

    emaskh = ~emaskh;

    auto f = [](unsigned n, unsigned m, uint64_t ii, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, unsigned n_qb,
                unsigned sqrt_size, uint64_t cmaskh, uint64_t emaskh,
                fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[16], is[16];

      auto row_size = uint64_t{1} << n_qb;

      uint64_t i = ii % sqrt_size;
      uint64_t j = ii / sqrt_size;

      uint64_t col_loc = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2]);
      uint64_t row_loc = bits::ExpandBits(j, n_qb, emaskh) | cmaskh;

      auto p0 = rstate + row_size * 2 * row_loc + 2 * col_loc;

      for (unsigned l = 0; l < 4; ++l) {
        for (unsigned k = 0; k < 4; ++k) {
          rs[4 * l + k] = *(p0 + xss[l] * row_size + xss[k]);
          is[4 * l + k] = *(p0 + xss[l] * row_size + xss[k] + 1);
        }
      }

      for (unsigned l = 0; l < 4; l++) {
        uint64_t j = 0;
        for (unsigned k = 0; k < 4; ++k) {
          rn = rs[l] * v[j] - is[l] * v[j + 1];
          in = rs[l] * v[j + 1] + is[l] * v[j];
          j += 2;

          for (unsigned p = 1; p < 4; ++p) {
            rn += rs[4 * p + l] * v[j] - is[4 * p + l] * v[j + 1];
            in += rs[4 * p + l] * v[j + 1] + is[4 * p + l] * v[j];

            j += 2;
          }
          *(p0 + xss[k] * row_size + xss[l]) = rn;
          *(p0 + xss[k] * row_size + xss[l] + 1) = in;
        }
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 2 + cqs.size();
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;

    unsigned kk = 2;
    unsigned nn = num_qubits_ > kk ? num_qubits_ - kk : 0;
    uint64_t size2 = uint64_t{1} << nn;

    for_.Run(size * size2, f, matrix, ms, xss, num_qubits_, size2, cmaskh,
             emaskh, rstate);
  }

  void ApplyGate1(unsigned q0, const fp_type* matrix, Unitary& state) const {
    uint64_t xs[1];
    uint64_t ms[2];

    xs[0] = uint64_t{1} << (q0 + 1);
    ms[0] = (uint64_t{1} << q0) - 1;
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

    auto f = [](unsigned n, unsigned m, uint64_t ii, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, unsigned n_qb,
                unsigned sqrt_size, fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[4], is[4];

      auto row_size = uint64_t{1} << n_qb;

      uint64_t i = ii % sqrt_size;
      uint64_t j = ii / sqrt_size;

      uint64_t col_loc = (1 * i & ms[0]) | (2 * i & ms[1]);
      uint64_t row_loc = (1 * j & ms[0]) | (2 * j & ms[1]);

      auto p0 = rstate + row_size * 2 * row_loc + 2 * col_loc;

      for (unsigned l = 0; l < 2; ++l) {
        for (unsigned k = 0; k < 2; ++k) {
          rs[2 * l + k] = *(p0 + xss[l] * row_size + xss[k]);
          is[2 * l + k] = *(p0 + xss[l] * row_size + xss[k] + 1);
        }
      }

      for (unsigned l = 0; l < 2; l++) {
        uint64_t j = 0;
        for (unsigned k = 0; k < 2; ++k) {
          rn = rs[l] * v[j] - is[l] * v[j + 1];
          in = rs[l] * v[j + 1] + is[l] * v[j];
          j += 2;

          for (unsigned p = 1; p < 2; ++p) {
            rn += rs[2 * p + l] * v[j] - is[2 * p + l] * v[j + 1];
            in += rs[2 * p + l] * v[j + 1] + is[2 * p + l] * v[j];

            j += 2;
          }
          *(p0 + xss[k] * row_size + xss[l]) = rn;
          *(p0 + xss[k] * row_size + xss[l] + 1) = in;
        }
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 1;
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size * size, f, matrix, ms, xss, num_qubits_, size, rstate);
  }

  void ApplyGate2(unsigned q0, unsigned q1, const fp_type* matrix,
                  Unitary& state) const {
    // Assume q0 < q1.
    uint64_t xs[2];
    uint64_t ms[3];

    xs[0] = uint64_t{1} << (q0 + 1);
    ms[0] = (uint64_t{1} << q0) - 1;
    for (unsigned i = 1; i < 2; ++i) {
      xs[i] = uint64_t{1} << (q1 + 1);
      ms[i] = ((uint64_t{1} << q1) - 1) ^ (xs[i - 1] - 1);
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

    auto f = [](unsigned n, unsigned m, uint64_t ii, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, unsigned n_qb,
                unsigned sqrt_size, fp_type* rstate) {
      fp_type rn, in;
      fp_type rs[16], is[16];

      auto row_size = uint64_t{1} << n_qb;
      uint64_t i = ii % sqrt_size;
      uint64_t j = ii / sqrt_size;

      uint64_t col_loc = (1 * i & ms[0]) | (2 * i & ms[1]) | (4 * i & ms[2]);
      uint64_t row_loc = (1 * j & ms[0]) | (2 * j & ms[1]) | (4 * j & ms[2]);

      auto p0 = rstate + row_size * 2 * row_loc + 2 * col_loc;

      for (unsigned l = 0; l < 4; ++l) {
        for (unsigned k = 0; k < 4; ++k) {
          rs[4 * l + k] = *(p0 + xss[l] * row_size + xss[k]);
          is[4 * l + k] = *(p0 + xss[l] * row_size + xss[k] + 1);
        }
      }

      for (unsigned l = 0; l < 4; l++) {
        uint64_t j = 0;
        for (unsigned k = 0; k < 4; ++k) {
          rn = rs[l] * v[j] - is[l] * v[j + 1];
          in = rs[l] * v[j + 1] + is[l] * v[j];
          j += 2;

          for (unsigned p = 1; p < 4; ++p) {
            rn += rs[4 * p + l] * v[j] - is[4 * p + l] * v[j + 1];
            in += rs[4 * p + l] * v[j + 1] + is[4 * p + l] * v[j];

            j += 2;
          }
          *(p0 + xss[k] * row_size + xss[l]) = rn;
          *(p0 + xss[k] * row_size + xss[l] + 1) = in;
        }
      }
    };

    fp_type* rstate = state.get();

    unsigned k = 2;
    unsigned n = num_qubits_ > k ? num_qubits_ - k : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size * size, f, matrix, ms, xss, num_qubits_, size, rstate);
  }

  For for_;
  unsigned num_qubits_;
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARY_CALCULATOR_BASIC_H_
