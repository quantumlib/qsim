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

/**
 * Helper struct for running qsimh.
 */
template <typename IO, typename Fuser, typename HybridSimulator>
struct QSimHRunner final {
  using Parameter = typename HybridSimulator::Parameter<typename Fuser::Parameter>;

  /**
   * Evaluates the amplitudes for a given circuit and set of output states.
   * @param param Options for gate fusion, parallelism and logging. Also
   *   specifies the size of the 'prefix' and 'root' sections of the lattice.
   * @param factory Object to create simulators and state spaces.
   * @param circuit The circuit to be simulated.
   * @param parts Lattice sections to be simulated.
   * @param bitstrings List of output states to simulate, as bitstrings.
   * @param results Output vector of amplitudes. After a successful run, this
   *   will be populated with amplitudes for each state in 'bitstrings'.
   * @return True if the simulation completed successfully; false otherwise.
   */
  template <typename Factory, typename Circuit, typename FP>
  static bool Run(const Parameter& param, const Factory& factory,
                  const Circuit& circuit, const std::vector<unsigned>& parts,
                  const std::vector<uint64_t>& bitstrings,
                  std::vector<std::complex<FP>>& results) {
    if (circuit.num_qubits != parts.size()) {
      IO::errorf("parts size is not equal to the number of qubits.");
      return false;
    }

    double t0 = 0.0;

    if (param.verbosity > 0) {
      t0 = GetTime();
    }

    const auto& ops = Operations<Circuit>::get(circuit);

    using fp_type = OpFpType<decltype(ops[0])>;

    typename HybridSimulator::HybridData<fp_type> hd;
    bool rc = HybridSimulator::SplitLattice(parts, ops, hd);

    if (!rc) {
      return false;
    }

    if (hd.num_gatexs < param.num_prefix_gatexs + param.num_root_gatexs) {
      IO::errorf("error: num_prefix_gates (%u) plus num_root gates (%u) is "
                 "greater than num_gates_on_the_cut (%u).\n",
                 param.num_prefix_gatexs, param.num_root_gatexs,
                 hd.num_gatexs);
      return false;
    }

    if (param.verbosity > 0) {
      PrintInfo(param, hd);
    }

    auto fops0 = Fuser::FuseGates(param, hd.num_qubits0, hd.ops0);
    if (fops0.size() == 0 && hd.ops0.size() > 0) {
      return false;
    }

    auto fops1 = Fuser::FuseGates(param, hd.num_qubits1, hd.ops1);
    if (fops1.size() == 0 && hd.ops1.size() > 0) {
      return false;
    }

    rc = HybridSimulator(param.num_threads).Run(
        param, factory, hd, parts, fops0, fops1, bitstrings, results);

    if (rc && param.verbosity > 0) {
      double t1 = GetTime();
      IO::messagef("time elapsed %g seconds.\n", t1 - t0);
    }

    return rc;
  }

 private:
  template <typename HybridData>
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
