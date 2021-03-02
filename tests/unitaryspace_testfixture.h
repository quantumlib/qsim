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

#ifndef UNITARYSPACE_TESTFIXTURE_H_
#define UNITARYSPACE_TESTFIXTURE_H_

#include <complex>

#include "gtest/gtest.h"

namespace qsim {

namespace unitary {

template <typename UnitarySpace>
void TestSetZeros() {
  using Unitary = typename UnitarySpace::Unitary;

  for (unsigned nq = 1; nq <= 5; ++nq) {
    UnitarySpace us(1);
    Unitary u = us.CreateUnitary(nq);

    us.SetAllZeros(u);

    unsigned size = 1 << nq;
    for (unsigned i = 0; i < size; ++i) {
      for (unsigned j = 0; j < size; ++j) {
        EXPECT_EQ(us.GetEntry(u, i, j), std::complex<float>(0, 0));
      }
    }
  }
}

template <typename UnitarySpace>
void TestSetIdentity() {
  using Unitary = typename UnitarySpace::Unitary;

  for (unsigned nq = 1; nq <= 5; ++nq) {
    UnitarySpace us(1);
    Unitary u = us.CreateUnitary(nq);

    us.SetIdentity(u);

    unsigned size = 1 << nq;
    for (unsigned i = 0; i < size; ++i) {
      for (unsigned j = 0; j < size; ++j) {
        if (i == j) {
          EXPECT_EQ(us.GetEntry(u, i, j), std::complex<float>(1, 0));
        } else {
          EXPECT_EQ(us.GetEntry(u, i, j), std::complex<float>(0, 0));
        }
      }
    }
  }
}

template <typename UnitarySpace>
void TestSetEntry() {
  using Unitary = typename UnitarySpace::Unitary;

  for (unsigned nq = 1; nq <= 5; ++nq) {
    UnitarySpace us(1);
    Unitary u = us.CreateUnitary(nq);

    unsigned size = 1 << nq;

    for (unsigned i = 0; i < size; ++i) {
      for (unsigned j = 0; j < size; ++j) {
        unsigned val = i * size + j;
        us.SetEntry(u, i, j, 2 * val, 2 * val + 1);
      }
    }

    for (unsigned i = 0; i < size; ++i) {
      for (unsigned j = 0; j < size; ++j) {
        unsigned val = i * size + j;
        EXPECT_EQ(
          us.GetEntry(u, i, j), std::complex<float>(2 * val, 2 * val + 1));
      }
    }
  }
}

}  // namespace unitary
}  // namespace qsim

#endif  // UNITARYSPACE_TESTFIXTURE_H_
