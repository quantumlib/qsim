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

#ifndef CHANNEL_H_
#define CHANNEL_H_

#include "gate.h"

namespace qsim {

/**
 * Kraus operator.
 */
template <typename Gate>
struct KrausOperator {
  enum Kind {
    kNormal = 0,
    kMeasurement = gate::kMeasurement,
  };

  /**
   * Kraus operator type;
   */
  Kind kind;

  /**
   * If true, the Kraus operator is a unitary operator times a constant.
   */
  bool unitary;

  /**
   * Lower bound on Kraus operator probability.
   */
  double prob;

  /**
   * Sequence of operations that represent the Kraus operator. This can be just
   * one operation.
   */
  std::vector<Gate> ops;
};

/**
 * Quantum channel.
 */
template <typename Gate>
using Channel = std::vector<KrausOperator<Gate>>;

/**
 * Makes a channel from the gate.
 * @param time The time to place the channel at.
 * @param gate The input gate.
 * @return The output channel.
 */
template <typename Gate>
Channel<Gate> MakeChannelFromGate(unsigned time, const Gate& gate) {
  auto normal = KrausOperator<Gate>::kNormal;
  auto measurement = KrausOperator<Gate>::kMeasurement;

  auto kind = gate.kind == gate::kMeasurement ? measurement : normal;

  Channel<Gate> channel = {{kind, true, 1, {gate}}};
  channel[0].ops[0].time = time;

  return channel;
}

}  // namespace qsim

#endif  // CHANNEL_H_
