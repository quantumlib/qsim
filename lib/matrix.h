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

#include <algorithm>
#include <utility>
#include <vector>

#include "bits.h"

namespace qsim {

/**
 * Gate matrix type. Matrices are stored as vectors. The matrix elements are
 * accessed as real(m[i][j]) <- vector[2 * (n * i + j)] and
 * imag(m[i][j]) <- vector[2 * (n * i + j) + 1], where n is the number of rows
 * or columns (n = 2^q, where q is the number of gate qubits).
 */
template <typename fp_type>
using Matrix = std::vector<fp_type>;

/**
 * Sets all matrix elements to zero.
 * @m Matrix to be cleared.
 */
template <typename fp_type>
inline void MatrixClear(Matrix<fp_type>& m) {
  for (unsigned i = 0; i < m.size(); ++i) {
    m[i] = 0;
  }
}

/**
 * Sets an identity matrix.
 * @n Number of matrix rows (columns).
 * @m Output identity matrix.
 */
template <typename fp_type>
inline void MatrixIdentity(unsigned n, Matrix<fp_type>& m) {
  m.resize(2 * n * n);

  MatrixClear(m);

  for (unsigned i = 0; i < n; ++i) {
    m[2 * (n * i + i)] = 1;
  }
}

/**
 * Multiplies two gate matrices of equal size: m2 = m1 m2.
 * @q Number of gate qubits. The number of matrix rows (columns) is 2^q.
 * @m1 Matrix m1.
 * @m2 Input matrix m2. Output product of matrices m2 = m1 m2.
 */
template <typename fp_type1, typename fp_type2>
inline void MatrixMultiply(
    unsigned q, const Matrix<fp_type1>& m1, Matrix<fp_type2>& m2) {
  Matrix<fp_type2> mt = m2;
  unsigned n = unsigned{1} << q;

  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      fp_type2 re = 0;
      fp_type2 im = 0;

      for (unsigned k = 0; k < n; ++k) {
        fp_type2 r1 = m1[2 * (n * i + k)];
        fp_type2 i1 = m1[2 * (n * i + k) + 1];
        fp_type2 r2 = mt[2 * (n * k + j)];
        fp_type2 i2 = mt[2 * (n * k + j) + 1];

        re += r1 * r2 - i1 * i2;
        im += r1 * i2 + i1 * r2;
      }

      m2[2 * (n * i + j)] = re;
      m2[2 * (n * i + j) + 1] = im;
    }
  }
}

/**
 * Multiplies two gate matrices: m2 = m1 m2. The size of m1 should not exceed
 *   the size of m2.
 * @mask1 Qubit mask that specifies the subset of qubits m1 acts on.
 * @q1 Number of gate qubits. The number of matrix rows (columns) is 2^q1.
 * @m1 Matrix m1.
 * @q2 Number of gate qubits. The number of matrix rows (columns) is 2^q2.
 * @m2 Input matrix m2. Output product of matrices m2 = m1 m2.
 */
template <typename fp_type1, typename fp_type2>
inline void MatrixMultiply(unsigned mask1,
                           unsigned q1, const Matrix<fp_type1>& m1,
                           unsigned q2, Matrix<fp_type2>& m2) {
  if (q1 == q2) {
    MatrixMultiply(q1, m1, m2);
  } else {
    Matrix<fp_type2> mt = m2;
    unsigned n1 = unsigned{1} << q1;
    unsigned n2 = unsigned{1} << q2;

    for (unsigned i = 0; i < n2; ++i) {
      unsigned si = bits::CompressBits(i, q2, mask1);

      for (unsigned j = 0; j < n2; ++j) {
        fp_type2 re = 0;
        fp_type2 im = 0;

        for (unsigned k = 0; k < n1; ++k) {
          unsigned ek = bits::ExpandBits(k, q2, mask1) + (i & ~mask1);

          fp_type2 r1 = m1[2 * (n1 * si + k)];
          fp_type2 i1 = m1[2 * (n1 * si + k) + 1];
          fp_type2 r2 = mt[2 * (n2 * ek + j)];
          fp_type2 i2 = mt[2 * (n2 * ek + j) + 1];

          re += r1 * r2 - i1 * i2;
          im += r1 * i2 + i1 * r2;
        }

        m2[2 * (n2 * i + j)] = re;
        m2[2 * (n2 * i + j) + 1] = im;
      }
    }
  }
}

/**
 * Multiply a matrix by a scalar value.
 * @c Scalar value.
 * @m Input matrix to be multiplied. Output matrix.
 */
template <typename fp_type1, typename fp_type2>
inline void MatrixScalarMultiply(fp_type1 c, Matrix<fp_type2>& m) {
  for (unsigned i = 0; i < m.size(); ++i) {
    m[i] *= c;
  }
}

/**
 * Daggers a matrix.
 * @n Number of matrix rows (columns).
 * @m Input matrix. Output matrix.
 */
template <typename fp_type>
inline void MatrixDagger(unsigned n, Matrix<fp_type>& m) {
  for (unsigned i = 0; i < n; ++i) {
    m[2 * (n * i + i) + 1] = -m[2 * (n * i + i) + 1];

    for (unsigned j = i + 1; j < n; ++j) {
      std::swap(m[2 * (n * i + j)], m[2 * (n * j + i)]);
      fp_type t = m[2 * (n * i + j) + 1];
      m[2 * (n * i + j) + 1] = -m[2 * (n * j + i) + 1];
      m[2 * (n * j + i) + 1] = -t;
    }
  }
}

/**
 * Gets a permutation to rearrange qubits from "normal" order to "gate"
 *   order. Qubits are ordered in increasing order for "normal" order.
 *   Qubits are ordered arbitrarily for "gate" order. Returns an empty vector
 *   if the qubits are in "normal" order.
 * @qubits Qubit indices in "gate" order.
 * @return Permutation as a vector.
 */
inline std::vector<unsigned> NormalToGateOrderPermutation(
    const std::vector<unsigned>& qubits) {
  std::vector<unsigned> perm;

  bool normal_order = true;

  for (std::size_t i = 1; i < qubits.size(); ++i) {
    if (qubits[i] < qubits[i - 1]) {
      normal_order = false;
      break;
    }
  }

  if (!normal_order) {
    struct QI {
      unsigned q;
      unsigned index;
    };

    std::vector<QI> qis;
    qis.reserve(qubits.size());

    for (std::size_t i = 0; i < qubits.size(); ++i) {
      qis.push_back({qubits[i], unsigned(i)});
    }

    std::sort(qis.begin(), qis.end(), [](const QI& l, const QI& r) {
                                        return l.q < r.q;
                                      });

    perm.reserve(qubits.size());

    for (std::size_t i = 0; i < qubits.size(); ++i) {
      perm.push_back(qis[i].index);
    }
  }

  return perm;
}

/**
 * Shuffles the gate matrix elements to get the matrix that acts on qubits
 *   that are in "normal" order (in increasing orger).
 * @perm Permutation to rearrange qubits from "normal" order to "gate" order.
 * @q Number of gate qubits. The number of matrix rows (columns) is 2^q.
 * @m Input matrix. Output shuffled matrix.
 */
template <typename fp_type>
inline void MatrixShuffle(const std::vector<unsigned>& perm,
                          unsigned q, Matrix<fp_type>& m) {
  Matrix<fp_type> mt = m;
  unsigned n = unsigned{1} << q;

  for (unsigned i = 0; i < n; ++i) {
    unsigned pi = bits::PermuteBits(i, q, perm);
    for (unsigned j = 0; j < n; ++j) {
      unsigned pj = bits::PermuteBits(j, q, perm);

      m[2 * (n * i + j)] = mt[2 * (n * pi + pj)];
      m[2 * (n * i + j) + 1] = mt[2 * (n * pi + pj) + 1];
    }
  }
}

}  // namespace qsim

#endif  // MATRIX_H_
