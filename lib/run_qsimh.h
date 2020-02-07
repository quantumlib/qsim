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

#ifndef RUN_QSIMH_H_
#define RUN_QSIMH_H_

#include <string>
#include <vector>

#include "hybrid.h"
#include "util.h"

namespace qsim {

// Helper struct to run qsimh.

template <typename IO, typename HybridSimulator>
struct QSimHRunner final {
  using fp_type = typename HybridSimulator::fp_type;
  using Gate = qsim::Gate<fp_type>;

  using Parameter = typename HybridSimulator::Parameter;
  using HybridData = typename HybridSimulator::HybridData;
  using Fuser = typename HybridSimulator::Fuser;

  static bool Run(const Parameter& param, unsigned maxtime,
                  const std::vector<unsigned>& parts,
                  const std::vector<Gate>& gates,
                  const std::vector<uint64_t>& bitstrings,
                  std::vector<std::complex<fp_type>>& results) {
    double t0;

    if (param.verbosity > 0) {
      t0 = GetTime();
    }

    HybridData hd = HybridSimulator::SplitLattice(parts, gates);

    if (hd.num_gatexs < param.num_prefix_gatexs + param.num_root_gatexs) {
      IO::errorf("invalid breakup: %up+%ur > num_gatexs=%u\n",
                 param.num_prefix_gatexs, param.num_root_gatexs,
                 hd.num_gatexs);
      return false;
    }

    if (param.verbosity > 0) {
      PrintInfo(param, hd);
    }

    auto fgates0 = Fuser::FuseGates(hd.num_qubits0, hd.gates0, maxtime);
    auto fgates1 = Fuser::FuseGates(hd.num_qubits1, hd.gates1, maxtime);

    bool rc = HybridSimulator::Run(
        param, hd, parts, fgates0, fgates1, bitstrings, results);

    if (rc && param.verbosity > 0) {
      double t1 = GetTime();
      IO::messagef("time elapsed %g seconds.\n", t1 - t0);
    }

    return rc;
  }

 private:
  static void PrintInfo(const Parameter& param, const HybridData& hd) {
    unsigned num_suffix_gates =
        hd.num_gatexs - param.num_prefix_gatexs - param.num_root_gatexs;

    IO::messagef("part 0: %u, part 1: %u\n", hd.num_qubits0, hd.num_qubits1);
    IO::messagef("%u gates on the cut\n", hd.num_gatexs);
    IO::messagef("breakup: %up+%ur+%us\n", param.num_prefix_gatexs,
                 param.num_root_gatexs, num_suffix_gates);
  }

};

}  // namespace qsim

#endif  // RUN_QSIM_H_
