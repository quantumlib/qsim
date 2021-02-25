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

#ifndef FUSER_H_
#define FUSER_H_

#include <cstdint>
#include <vector>

#include "gate.h"
#include "matrix.h"

namespace qsim {

/**
 * A collection of "fused" gates which can be multiplied together before being
 * applied to the state vector.
 */
template <typename Gate>
struct GateFused {
  /**
   * Kind of the first ("parent") gate.
   */
  typename Gate::GateKind kind;
  /**
   * The time index of the first ("parent") gate.
   */
  unsigned time;
  /**
   * A list of qubits these gates act upon. Control qubits for
   * explicitly-controlled gates are excluded from this list.
   */
  std::vector<unsigned> qubits;
  /**
   * Pointer to the first ("parent") gate.
   */
  const Gate* parent;
  /**
   * Ordered list of component gates.
   */
  std::vector<const Gate*> gates;
};

/**
 * A base class for fuser classes with some common functions.
 */
template <typename IO, typename Gate>
class Fuser {
 protected:
  using RGate = typename std::remove_pointer<Gate>::type;

  static const RGate& GateToConstRef(const RGate& gate) {
    return gate;
  }

  static const RGate& GateToConstRef(const RGate* gate) {
    return *gate;
  }

  static std::vector<unsigned> MergeWithMeasurementTimes(
      typename std::vector<Gate>::const_iterator gfirst,
      typename std::vector<Gate>::const_iterator glast,
      const std::vector<unsigned>& times) {
    std::vector<unsigned> epochs;
    epochs.reserve(glast - gfirst + times.size());

    std::size_t last = 0;
    unsigned max_time = 0;

    for (auto gate_it = gfirst; gate_it < glast; ++gate_it) {
      const auto& gate = GateToConstRef(*gate_it);

      if (gate.time > max_time) {
        max_time = gate.time;
      }

      if (epochs.size() > 0 && gate.time < epochs.back()) {
        IO::errorf("gate crosses the time boundary.\n");
        epochs.resize(0);
        return epochs;
      }

      if (gate.kind == gate::kMeasurement) {
        if (epochs.size() == 0 || epochs.back() < gate.time) {
          if (!AddBoundary(gate.time, max_time, epochs)) {
            epochs.resize(0);
            return epochs;
          }
        }
      }

      while (last < times.size() && times[last] <= gate.time) {
        unsigned prev = times[last++];
        epochs.push_back(prev);
        if (!AddBoundary(prev, max_time, epochs)) {
          epochs.resize(0);
          return epochs;
        }
        while (last < times.size() && times[last] <= prev) ++last;
      }
    }

    if (epochs.size() == 0 || epochs.back() < max_time) {
      epochs.push_back(max_time);
    }

    return epochs;
  }

 private:
  static bool AddBoundary(unsigned time, unsigned max_time,
                          std::vector<unsigned>& boundaries) {
    if (max_time > time) {
      IO::errorf("gate crosses the time boundary.\n");
      return false;
    }

    boundaries.push_back(time);
    return true;
  }
};

/**
 * Multiplies component gate matrices of a fused gate.
 * @param gate Fused gate.
 * @return Matrix product of component matrices.
 */
template <typename fp_type, typename FusedGate>
inline Matrix<fp_type> CalculateFusedMatrix(const FusedGate& gate) {
  Matrix<fp_type> matrix;
  MatrixIdentity(unsigned{1} << gate.qubits.size(), matrix);

  for (auto pgate : gate.gates) {
    if (gate.qubits.size() == pgate->qubits.size()) {
      MatrixMultiply(gate.qubits.size(), pgate->matrix, matrix);
    } else {
      unsigned mask = 0;

      for (auto q : pgate->qubits) {
        for (std::size_t i = 0; i < gate.qubits.size(); ++i) {
          if (q == gate.qubits[i]) {
            mask |= unsigned{1} << i;
            break;
          }
        }
      }

      MatrixMultiply(mask, pgate->qubits.size(), pgate->matrix,
                     gate.qubits.size(), matrix);
    }
  }

  return matrix;
}

}  // namespace qsim

#endif  // FUSER_H_
