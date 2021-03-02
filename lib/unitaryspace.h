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

#ifndef UNITARYSPACE_H_
#define UNITARYSPACE_H_

#include <cstdint>

#include "vectorspace.h"

namespace qsim {

namespace unitary {

/**
 * Abstract class containing routines for general unitary matrix manipulations.
 * "AVX", "Basic", and "SSE" implementations are provided.
 */
template <typename Impl, typename For, typename FP>
class UnitarySpace : public VectorSpace<Impl, For, FP> {
 private:
  using Base = VectorSpace<Impl, For, FP>;

 public:
  using fp_type = typename Base::fp_type;
  using Unitary = typename Base::Vector;

  template <typename... ForArgs>
  UnitarySpace(ForArgs&&... args) : Base(args...) {}

  static Unitary CreateUnitary(unsigned num_qubits) {
    return Base::Create(num_qubits);
  }

  static Unitary CreateUnitary(fp_type* p, unsigned num_qubits) {
    return Base::Create(p, num_qubits);
  }

  static Unitary NullUnitary() {
    return Base::Null();
  }

  static uint64_t Size(unsigned num_qubits) {
    return uint64_t{1} << num_qubits;
  };

  void CopyUnitary(const Unitary& src, Unitary& dest) const {
    Base::Copy(src, dest);
  }
};

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARYSPACE_H_
