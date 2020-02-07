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

#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>

namespace qsim {

// Routines for 2x2 complex matrices.
// Matrices are arrays of floating-point numbers.
// There are no checks for validaty of arguments.
// We do not care about performance here.

template <typename Array2>
inline void Matrix2SetZero(Array2& matrix) {
  for (unsigned i = 0; i < 8; ++i) {
    matrix[i] = 0;
  }
}

template <typename Array2>
inline void Matrix2SetId(Array2& matrix) {
  Matrix2SetZero(matrix);

  matrix[0] = 1;
  matrix[6] = 1;
}

template <typename Array1, typename Array2>
inline void Matrix2Set(const Array1& u, Array2& matrix) {
  for (unsigned i = 0; i < 8; ++i) {
    matrix[i] = u[i];
  }
}

// Multiply two 2x2 matrices.
template <typename Array1, typename Array2>
inline void Matrix2Multiply(const Array1& u, Array2& matrix) {
  typename Array1::value_type matrix0[8];
  for (unsigned i = 0; i < 8; ++i) {
    matrix0[i] = matrix[i];
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      typename Array1::value_type tr = 0;
      typename Array1::value_type ti = 0;

      for (unsigned k = 0; k < 2; ++k) {
        auto mr0 = matrix0[4 * k + 2 * j + 0];
        auto mi0 = matrix0[4 * k + 2 * j + 1];

        auto uik = &u[4 * i + 2 * k];

        tr += uik[0] * mr0 - uik[1] * mi0;
        ti += uik[0] * mi0 + uik[1] * mr0;
      }

      matrix[4 * i + 2 * j + 0] = tr;
      matrix[4 * i + 2 * j + 1] = ti;
    }
  }
}

// Routines for 4x4 complex matrices.
// Matrices are arrays of floating-point numbers.
// There are no checks for validaty of arguments.
// We do not care about performance here.

template <typename Array2>
inline void Matrix4SetZero(Array2& matrix) {
  for (unsigned i = 0; i < 32; ++i) {
    matrix[i] = 0;
  }
}

template <typename Array2>
inline void Matrix4SetId(Array2& matrix) {
  Matrix4SetZero(matrix);

  matrix[ 0] = 1;
  matrix[10] = 1;
  matrix[20] = 1;
  matrix[30] = 1;
}

template <typename Array1, typename Array2>
inline void Matrix4Set(const Array1& u, Array2& matrix) {
  for (unsigned i = 0; i < 32; ++i) {
    matrix[i] = u[i];
  }
}

// Multiply 4x4 matrix by one qubit matrix corresponding to qubit 0.
template <typename Array1, typename Array2>
inline void Matrix4Multiply20(const Array1& u, Array2& matrix) {
  auto u00 = &u[0];
  auto u01 = &u[2];
  auto u10 = &u[4];
  auto u11 = &u[6];

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      auto mr0 = matrix[16 * j + 0 + 2 * i];
      auto mi0 = matrix[16 * j + 1 + 2 * i];
      auto mr1 = matrix[16 * j + 8 + 2 * i];
      auto mi1 = matrix[16 * j + 9 + 2 * i];

      matrix[16 * j + 0 + 2 * i] =
          u00[0] * mr0 - u00[1] * mi0 + u01[0] * mr1 - u01[1] * mi1;
      matrix[16 * j + 1 + 2 * i] =
          u00[0] * mi0 + u00[1] * mr0 + u01[0] * mi1 + u01[1] * mr1;
      matrix[16 * j + 8 + 2 * i] =
          u10[0] * mr0 - u10[1] * mi0 + u11[0] * mr1 - u11[1] * mi1;
      matrix[16 * j + 9 + 2 * i] =
          u10[0] * mi0 + u10[1] * mr0 + u11[0] * mi1 + u11[1] * mr1;
    }
  }
}

// Multiply 4x4 matrix by one qubit matrix corresponding to qubit 1.
template <typename Array1, typename Array2>
inline void Matrix4Multiply21(const Array1& u, Array2& matrix) {
  auto u00 = &u[0];
  auto u01 = &u[2];
  auto u10 = &u[4];
  auto u11 = &u[6];

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 2; ++j) {
      auto mr0 = matrix[8 * j +  0 + 2 * i];
      auto mi0 = matrix[8 * j +  1 + 2 * i];
      auto mr1 = matrix[8 * j + 16 + 2 * i];
      auto mi1 = matrix[8 * j + 17 + 2 * i];

      matrix[8 * j +  0 + 2 * i] =
          u00[0] * mr0 - u00[1] * mi0 + u01[0] * mr1 - u01[1] * mi1;
      matrix[8 * j +  1 + 2 * i] =
          u00[0] * mi0 + u00[1] * mr0 + u01[0] * mi1 + u01[1] * mr1;
      matrix[8 * j + 16 + 2 * i] =
          u10[0] * mr0 - u10[1] * mi0 + u11[0] * mr1 - u11[1] * mi1;
      matrix[8 * j + 17 + 2 * i] =
          u10[0] * mi0 + u10[1] * mr0 + u11[0] * mi1 + u11[1] * mr1;
    }
  }
}

// Multiply two 4x4 matrices.
template <typename Array1, typename Array2>
inline void Matrix4Multiply(const Array1& u, Array2& matrix) {
  typename Array1::value_type matrix0[32];
  for (unsigned i = 0; i < 32; ++i) {
    matrix0[i] = matrix[i];
  }

  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      typename Array1::value_type tr = 0;
      typename Array1::value_type ti = 0;

      for (unsigned k = 0; k < 4; ++k) {
        auto mr0 = matrix0[8 * k + 2 * j + 0];
        auto mi0 = matrix0[8 * k + 2 * j + 1];

        auto uik = &u[8 * i + 2 * k];

        tr += uik[0] * mr0 - uik[1] * mi0;
        ti += uik[0] * mi0 + uik[1] * mr0;
      }

      matrix[8 * i + 2 * j + 0] = tr;
      matrix[8 * i + 2 * j + 1] = ti;
    }
  }
}

// Permute 4x4 matrix to switch between two qubits.
template <typename Array2>
inline void Matrix4Permute(Array2& matrix) {
  std::swap(matrix[ 2], matrix[ 4]); std::swap(matrix[ 3], matrix[ 5]);
  std::swap(matrix[ 8], matrix[16]); std::swap(matrix[ 9], matrix[17]);
  std::swap(matrix[10], matrix[20]); std::swap(matrix[11], matrix[21]);
  std::swap(matrix[12], matrix[18]); std::swap(matrix[13], matrix[19]);
  std::swap(matrix[14], matrix[22]); std::swap(matrix[15], matrix[23]);
  std::swap(matrix[26], matrix[28]); std::swap(matrix[27], matrix[29]);
}

}  // namespace qsim

#endif  // MATRIX_H_
