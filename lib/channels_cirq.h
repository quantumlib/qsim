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

#ifndef CHANNELS_CIRQ_H_
#define CHANNELS_CIRQ_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "channel.h"
#include "gates_cirq.h"

namespace qsim {

namespace Cirq {

template <typename fp_type>
using Channel = qsim::Channel<GateCirq<fp_type>>;

/**
 * Asymmetric depolarizing channel factory.
 */
template <typename fp_type>
struct AsymmetricDepolarizingChannel {
  static constexpr char name[] = "asymmetric_depolarize";

  AsymmetricDepolarizingChannel(double p_x, double p_y, double p_z)
      : p_x(p_x), p_y(p_y), p_z(p_z) {}

  static Channel<fp_type> Create(unsigned time, unsigned q,
                                 double p_x, double p_y, double p_z) {
    double p1 = 1 - p_x - p_y - p_z;

    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    return {{normal, 1, p1, {}},
            {normal, 1, p_x, {X<fp_type>::Create(time, q)}},
            {normal, 1, p_y, {Y<fp_type>::Create(time, q)}},
            {normal, 1, p_z, {Z<fp_type>::Create(time, q)}}};
  }

  static Channel<fp_type> Create(unsigned time,
                                 const std::vector<unsigned>& qubits,
                                 double p_x, double p_y, double p_z) {
    double p1 = 1 - p_x - p_y - p_z;

    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    uint64_t size = uint64_t{1} << (2 * qubits.size());

    Channel<fp_type> channel;
    channel.reserve(size);

    for (uint64_t i = 0; i < size; ++i) {
      channel.push_back({normal, 1, 0, {}});
      auto& kop = channel.back();

      kop.ops.reserve(qubits.size());

      double prob = 1;

      for (unsigned q = 0; q < qubits.size(); ++q) {
        unsigned pauli_index = (i >> (2 * q)) & 3;

        switch (pauli_index) {
        case 0:
          prob *= p1;
          break;
        case 1:
          prob *= p_x;
          kop.ops.push_back(X<fp_type>::Create(time, q));
          break;
        case 2:
          prob *= p_y;
          kop.ops.push_back(Y<fp_type>::Create(time, q));
          break;
        case 3:
          prob *= p_z;
          kop.ops.push_back(Z<fp_type>::Create(time, q));
          break;
        }
      }

      kop.prob = prob;
    }

    return channel;
  }

  Channel<fp_type> Create(unsigned time, unsigned q) const {
    return Create(time, q, p_x, p_y, p_z);
  }

  Channel<fp_type> Create(
      unsigned time, const std::vector<unsigned>& qubits) const {
    return Create(time, qubits, p_x, p_y, p_z);
  }

  double p_x = 0;
  double p_y = 0;
  double p_z = 0;
};

/**
 * Returns an asymmetric depolarizing channel factory object.
 */
template <typename fp_type>
inline AsymmetricDepolarizingChannel<fp_type> asymmetric_depolarize(
    double p_x, double p_y, double p_z) {
  return AsymmetricDepolarizingChannel<fp_type>(p_x, p_y, p_z);
}

/**
 * Depolarizing channel factory.
 */
template <typename fp_type>
struct DepolarizingChannel {
  static constexpr char name[] = "depolarize";

  DepolarizingChannel(double p) : p(p) {}

  static Channel<fp_type> Create(unsigned time, unsigned q, double p) {
    double p1 = 1 - p;
    double p2 = p / 3;

    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    return {{normal, 1, p1, {}},
            {normal, 1, p2, {X<fp_type>::Create(time, q)}},
            {normal, 1, p2, {Y<fp_type>::Create(time, q)}},
            {normal, 1, p2, {Z<fp_type>::Create(time, q)}}};
  }

  static Channel<fp_type> Create(
      unsigned time, const std::vector<unsigned>& qubits, double p) {
    double p1 = 1 - p;
    double p2 = p / 3;

    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    uint64_t size = uint64_t{1} << (2 * qubits.size());

    Channel<fp_type> channel;
    channel.reserve(size);

    for (uint64_t i = 0; i < size; ++i) {
      channel.push_back({normal, 1, 0, {}});
      auto& kop = channel.back();

      kop.ops.reserve(qubits.size());

      double prob = 1;

      for (unsigned q = 0; q < qubits.size(); ++q) {
        unsigned pauli_index = (i >> (2 * q)) & 3;

        switch (pauli_index) {
        case 0:
          prob *= p1;
          break;
        case 1:
          prob *= p2;
          kop.ops.push_back(X<fp_type>::Create(time, q));
          break;
        case 2:
          prob *= p2;
          kop.ops.push_back(Y<fp_type>::Create(time, q));
          break;
        case 3:
          prob *= p2;
          kop.ops.push_back(Z<fp_type>::Create(time, q));
          break;
        }
      }

      kop.prob = prob;
    }

    return channel;
  }

  Channel<fp_type> Create(unsigned time, unsigned q) const {
    return Create(time, q, p);
  }

  Channel<fp_type> Create(
      unsigned time, const std::vector<unsigned>& qubits) const {
    return Create(time, qubits, p);
  }

  double p = 0;
};

/**
 * Returns a depolarizing channel factory object.
 */
template <typename fp_type>
inline DepolarizingChannel<fp_type> depolarize(double p) {
  return DepolarizingChannel<fp_type>(p);
}

/**
 * Generalized amplitude damping channel factory.
 */
template <typename fp_type>
struct GeneralizedAmplitudeDampingChannel {
  static constexpr char name[] = "generalized_amplitude_damp";

  GeneralizedAmplitudeDampingChannel(double p, double gamma)
      : p(p), gamma(gamma) {}

  static Channel<fp_type> Create(
      unsigned time, unsigned q, double p, double gamma) {
    double p1 = p * (1 - gamma);
    double p2 = (1 - p) * (1 - gamma);
    double p3 = 0;

    fp_type t1 = std::sqrt(p);
    fp_type r1 = std::sqrt(p * (1 - gamma));
    fp_type s1 = std::sqrt(p * gamma);
    fp_type t2 = std::sqrt(1 - p);
    fp_type r2 = std::sqrt((1 - p) * (1 - gamma));
    fp_type s2 = std::sqrt((1 - p) * gamma);

    using M = Cirq::MatrixGate1<fp_type>;
    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;


    return {{normal, 0, p1, {M::Create(time, q, {t1, 0,  0, 0,  0, 0, r1, 0})}},
            {normal, 0, p2, {M::Create(time, q, {r2, 0,  0, 0,  0, 0, t2, 0})}},
            {normal, 0, p3, {M::Create(time, q, { 0, 0, s1, 0,  0,  0, 0, 0})}},
            {normal, 0, p3, {M::Create(time, q, { 0, 0,  0, 0, s2, 0,  0, 0})}},
           };
  }

  Channel<fp_type> Create(unsigned time, unsigned q) const {
    return Create(time, q, p, gamma);
  }

  double p = 1;
  double gamma = 0;
};

/**
 * Returns a generalized amplitude damping channel factory object.
 */
template <typename fp_type>
inline GeneralizedAmplitudeDampingChannel<fp_type> generalized_amplitude_damp(
    double p, double gamma) {
  return GeneralizedAmplitudeDampingChannel<fp_type>(p, gamma);
}

/**
 * Amplitude damping channel factory.
 */
template <typename fp_type>
struct AmplitudeDampingChannel {
  static constexpr char name[] = "amplitude_damp";

  AmplitudeDampingChannel(double gamma) : gamma(gamma) {}

  static Channel<fp_type> Create(unsigned time, unsigned q, double gamma) {
    double p1 = 1 - gamma;
    double p2 = 0;

    fp_type r = std::sqrt(p1);
    fp_type s = std::sqrt(gamma);

    using M = Cirq::MatrixGate1<fp_type>;
    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    return {{normal, 0, p1, {M::Create(time, q, {1, 0, 0, 0, 0, 0, r, 0})}},
            {normal, 0, p2, {M::Create(time, q, {0, 0, s, 0, 0, 0, 0, 0})}},
           };
  }

  Channel<fp_type> Create(unsigned time, unsigned q) const {
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
  static constexpr char name[] = "phase_dump";

  PhaseDampingChannel(double gamma) : gamma(gamma) {}

  static Channel<fp_type> Create(unsigned time, unsigned q, double gamma) {
    double p1 = 1 - gamma;
    double p2 = 0;

    fp_type r = std::sqrt(p1);
    fp_type s = std::sqrt(gamma);

    using M = Cirq::MatrixGate1<fp_type>;
    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    return {{normal, 0, p1, {M::Create(time, q, {1, 0, 0, 0, 0, 0, r, 0})}},
            {normal, 0, p2, {M::Create(time, q, {0, 0, 0, 0, 0, 0, s, 0})}},
           };
  }

  Channel<fp_type> Create(unsigned time, unsigned q) const {
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

/**
 *  Reset channel factory.
 */
template <typename fp_type>
struct ResetChannel {
  static constexpr char name[] = "reset";

  static Channel<fp_type> Create(unsigned time, unsigned q) {
    using M = Cirq::MatrixGate1<fp_type>;
    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    return {{normal, 0, 0, {M::Create(time, q, {1, 0, 0, 0, 0, 0, 0, 0})}},
            {normal, 0, 0, {M::Create(time, q, {0, 0, 1, 0, 0, 0, 0, 0})}},
           };
  }
};

/**
 * Returns a reset channel factory object.
 */
template <typename fp_type>
inline ResetChannel<fp_type> reset() {
  return ResetChannel<fp_type>();
}

/**
 *  Phase flip channel factory.
 */
template <typename fp_type>
struct PhaseFlipChannel {
  static constexpr char name[] = "phase_flip";

  PhaseFlipChannel(double p) : p(p) {}

  static Channel<fp_type> Create(unsigned time, unsigned q, double p) {
    double p1 = 1 - p;
    double p2 = p;

    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    return {{normal, 1, p1, {}},
            {normal, 1, p2, {Z<fp_type>::Create(time, q)}}
           };
  }

  Channel<fp_type> Create(unsigned time, unsigned q) const {
    return Create(time, q, p);
  }

  double p = 0;
};

/**
 * Returns a phase flip channel factory object.
 */
template <typename fp_type>
inline PhaseFlipChannel<fp_type> phase_flip(double p) {
  return PhaseFlipChannel<fp_type>(p);
}

/**
 *  Bit flip channel factory.
 */
template <typename fp_type>
struct BitFlipChannel {
  static constexpr char name[] = "bit_flip";

  BitFlipChannel(double p) : p(p) {}

  static Channel<fp_type> Create(unsigned time, unsigned q, double p) {
    double p1 = 1 - p;
    double p2 = p;

    auto normal = KrausOperator<GateCirq<fp_type>>::kNormal;

    return {{normal, 1, p1, {}},
            {normal, 1, p2, {X<fp_type>::Create(time, q)}}
           };
  }

  Channel<fp_type> Create(unsigned time, unsigned q) const {
    return Create(time, q, p);
  }

  double p = 0;
};

/**
 * Returns a bit flip channel factory object.
 */
template <typename fp_type>
inline BitFlipChannel<fp_type> bit_flip(double p) {
  return BitFlipChannel<fp_type>(p);
}

}  // namesapce Cirq

}  // namespace qsim

#endif  // CHANNELS_CIRQ_H_
