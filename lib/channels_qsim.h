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

#ifndef CHANNELS_QSIM_H_
#define CHANNELS_QSIM_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "channel.h"
#include "gates_qsim.h"

namespace qsim {

/**
 * Amplitude damping channel factory.
 */
template <typename fp_type>
struct AmplitudeDampingChannel {
  AmplitudeDampingChannel(double gamma) : gamma(gamma) {}

  static Channel<GateQSim<fp_type>> Create(
      unsigned time, unsigned q, double gamma) {
    double p1 = 1 - gamma;
    double p2 = 0;

    fp_type r = std::sqrt(p1);
    fp_type s = std::sqrt(gamma);

    using M = GateMatrix1<fp_type>;
    auto normal = KrausOperator<GateQSim<fp_type>>::kNormal;

    return {{normal, 0, p1,
             {M::Create(time, q, {1, 0, 0, 0, 0, 0, r, 0})},
             {1, 0, 0, 0, 0, 0, r * r, 0}, {q},
            },
            {normal, 0, p2,
             {M::Create(time, q, {0, 0, s, 0, 0, 0, 0, 0})},
             {0, 0, 0, 0, 0, 0, s * s, 0}, {q},
            },
           };
  }

  Channel<GateQSim<fp_type>> Create(unsigned time, unsigned q) const {
    return Create(time, q, gamma);
  }

  double gamma = 0;
};

/**
 * Returns an amplitude damping channel factory object.
 */
template <typename fp_type>
inline AmplitudeDampingChannel<fp_type> amplitude_damp(double gamma) {
  return AmplitudeDampingChannel<fp_type>(gamma);
}

/**
 *  Phase damping channel factory.
 */
template <typename fp_type>
struct PhaseDampingChannel {
  PhaseDampingChannel(double gamma) : gamma(gamma) {}

  static Channel<GateQSim<fp_type>> Create(
      unsigned time, unsigned q, double gamma) {
    double p1 = 1 - gamma;
    double p2 = 0;

    fp_type r = std::sqrt(p1);
    fp_type s = std::sqrt(gamma);

    using M = GateMatrix1<fp_type>;
    auto normal = KrausOperator<GateQSim<fp_type>>::kNormal;

    return {{normal, 0, p1,
             {M::Create(time, q, {1, 0, 0, 0, 0, 0, r, 0})},
             {1, 0, 0, 0, 0, 0, r * r, 0}, {q},
            },
            {normal, 0, p2,
             {M::Create(time, q, {0, 0, 0, 0, 0, 0, s, 0})},
             {0, 0, 0, 0, 0, 0, s * s, 0}, {q},
            },
           };
  }

  Channel<GateQSim<fp_type>> Create(unsigned time, unsigned q) const {
    return Create(time, q, gamma);
  }

  double gamma = 0;
};

/**
 * Returns a phase damping channel factory object.
 */
template <typename fp_type>
inline PhaseDampingChannel<fp_type> phase_damp(double gamma) {
  return PhaseDampingChannel<fp_type>(gamma);
}

}  // namespace qsim

#endif  // CHANNELS_QSIM_H_
