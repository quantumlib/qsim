// Copyright 2026 Google LLC. All Rights Reserved.
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

#ifndef RUN_QSIM_H_
#define RUN_QSIM_H_

#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <vector>

#include "circuit.h"
#include "classical_control_expr.h"
#include "classical_control_symbol.h"
#include "error.h"
#include "gate.h"
#include "gate_appl.h"
#include "operation.h"
#include "operation_base.h"

namespace qsim {

/**
 * Applies the given measurement to the simulator state.
 * @tparam StateSpace State space implementation.
 * @tparam RGen Random number generator type.
 * @tparam SymTable Symbol table container holding classical variables and
 *   measurement identifiers.
 * @tparam Obss Container holding observables (measurement histograms).
 * @param state_space StateSpace object required to manipulate state vector.
 * @param m The measurement to be applied.
 * @param rgen Random number generator to perform measurements.
 * @param state State of the system, to be updated by this method.
 * @param symtab Symbol table containing classical variables and
   *   measurement identifiers.
 * @param obss Output observables (measurement histograms).
 * @return The measurement result.
 */
template <typename StateSpace, typename Rgen, typename SymTable, typename Obss>
inline auto ApplyMeasurementGate(
    const StateSpace& state_space, const Measurement& m, Rgen& rgen,
    typename StateSpace::State& state, SymTable& symtab, Obss& obss) {
  auto mresult = state_space.Measure(m.qubits, rgen, state);

  if (!mresult.valid) {
    Error::Throw("measurement failed");
  } else if (!m.id.empty()) {
    auto* mea_var = symtab.Lookup(m.id);
    if (mea_var == nullptr) {
      Error::Throw("measurement {} not found", m.id);
    }

    auto* obs = obss.Lookup(m.id);
    if (obs != nullptr && obs->Size() != m.qubits.size()) {
      Error::Throw("malformed observable");
    }

    auto& mea = mea_var->GetMea();

    if (mea.num_bits != mresult.bitstring.size()) {
      Error::Throw("bitstring sizes mismatch");
    }

    mea.bits = 0;
    for (unsigned i = 0; i < m.qubits.size(); ++i) {
      mea.bits |= uint64_t{mresult.bitstring[i]} << i;
    }

    if (obs != nullptr) {
      ++obs->cur_count[cc::Symbol::GetMeaInt(mea)];
    }
  }

  return mresult;
}

/**
 * Applies a classically controlled operation.
 * @tparam Parameter Parameters type.
 * @tparam ClassicallyControlledOperation Classically controlled operation type.
 * @tparam Simulator State vector simulator implementation.
 * @tparam RGen Random number generator type.
 * @tparam SymTable Symbol table container holding classical variables and
 *   measurement identifiers.
 * @tparam Obss Container holding observables (measurement histograms).
 * @tparam Nested Callback function type.
 * @param param Parameters for gate fusion and simulation.
 * @param ccop Classically controlled operation to be applied.
 * @param state_space StateSpace object required to manipulate state vector.
 * @param simulator Simulator object for applying gates.
 * @param rgen Random number generator to perform measurements.
 * @param state State of the system, to be updated by this method.
 * @param symtab Symbol table containing classical variables and
 *   measurement identifiers.
 * @param obss Output observables (measurement histograms).
 * @param run_nested Callback function to run nested blocks of classical
 *   control.
 * @return True if the discard statement was triggered; false otherwise.
 */
template <typename Parameter, typename ClassicallyControlledOperation,
          typename Simulator, typename Rgen, typename SymTable, typename Obss,
          typename Nested>
inline bool RunClassicalControl(
    const Parameter& param, const ClassicallyControlledOperation& ccop,
    const typename Simulator::StateSpace& state_space,
    const Simulator& simulator, Rgen& rgen, typename Simulator::State& state,
    SymTable& symtab, Obss& obss, Nested&& run_nested) {
  switch (ccop.kind) {
  case ClassicallyControlledOperation::kIfElse:
    if (ccop.sub_ops.size() > ccop.exprs.size() + 1) {
      Error::Throw("malformed classical control operation");
    }

    for (unsigned i = 0; i < ccop.sub_ops.size(); ++i) {
      if (i < ccop.exprs.size()) {
        if (cc::EvalCondExpr(symtab, ccop.exprs[i])) {
          symtab.EnterScope(ccop.scope_indices[i]);
          bool rc = run_nested(param, ccop.sub_ops[i], state_space,
                               simulator, rgen, state, symtab, obss);
          symtab.ExitScope();
          if (rc == false) {
            return false;
          }

          break;
        }
      } else {
        symtab.EnterScope(ccop.scope_indices[i]);
        bool rc = run_nested(param, ccop.sub_ops[i], state_space,
                             simulator, rgen, state, symtab, obss);
        symtab.ExitScope();
        if (rc == false) {
          return false;
        }
      }
    }

    break;
  case ClassicallyControlledOperation::kDoWhile:
    symtab.EnterScope(ccop.scope_indices[0]);
    do {
      bool rc = run_nested(param, ccop.sub_ops[0], state_space,
                           simulator, rgen, state, symtab, obss);
      if (rc == false) {
        return false;
      }
    } while (cc::EvalCondExpr(symtab, ccop.exprs[0]));
    symtab.ExitScope();

    break;
  case ClassicallyControlledOperation::kRepeat:
    symtab.EnterScope(ccop.scope_indices[0]);
    while (cc::EvalCondExpr(symtab, ccop.exprs[0])) {
      bool rc = run_nested(param, ccop.sub_ops[0], state_space,
                           simulator, rgen, state, symtab, obss);
      if (rc == false) {
        return false;
      }
    };
    symtab.ExitScope();

    break;
  case ClassicallyControlledOperation::kAssign:
    {
      auto* var = symtab.Lookup(ccop.str[0]);

      if (var == nullptr) {
        Error::Throw("variable {} not found", ccop.str[0]);
      }

      if (ccop.indices.empty()) {
        var->Assign(symtab, ccop.exprs);
      } else {
        var->Assign(symtab, ccop.exprs[0], ccop.indices[0]);
      }
    }

    break;
  case ClassicallyControlledOperation::kPrintLn:
    {
      std::string_view str = ccop.str.size() > 0 ? ccop.str[0] : "";
      PrintExpressions(symtab, str, ccop.exprs);
    }

    break;
  case ClassicallyControlledOperation::kDiscard:
    if (cc::EvalCondExpr(symtab, ccop.exprs[0])) {
      return false;
    }

    break;
  }

  return true;
}

/**
 * Helper runner for executing quantum circuits with (optionl) classical
 * control flow. Can run clean and noisy circuits. Manages gate fusion and
 * quantum trajectory execution.
 * @tparam Fuser Fuser type.
 * @tparam RGen Random number generator type (defaults to `std::mt19937`).
 */
template <typename Fuser, typename RGen = std::mt19937>
class QSimRunner {
 private:
  template <typename FP, typename Operation>
  using GateOrOperation = std::variant<const Gate<FP>*, const Operation*>;

 public:
  /**
   * User-specified parameters for gate fusion and simulation.
   */
  struct Parameter : public Fuser::Parameter {
    /**
     * If true, normalize the state vector before performing measurements.
     */
    bool normalize_before_mea_gates = true;
  };

  /**
   * Executes a clean or noisy circuit trajectory with classical control.
   * @tparam Operation Operation variant type.
   * @tparam Simulator State vector simulator implementation.
   * @tparam SymTable Symbol table container holding classical variables and
   *   measurement identifiers.
   * @tparam Obss Container holding observables (measurement histograms).
   * @param param Execution parameters.
   * @param circuit Circuit to be simulated.
   * @param state_space StateSpace object required to manipulate state vector.
   * @param simulator Simulator object for applying gates.
   * @param seed Seed for random number generation.
   * @param state Input initial state vector; updated to final state on success.
   * @param symtab Root scope symbol table containing classical variables and
   *   measurement identifiers.
   * @param obss Output observables (measurement histograms).
   * @return True if trajectory completed successfully; false if discarded
   *   via `kDiscard`.
   */
  template
      <typename Operation, typename Simulator, typename SymTable, typename Obss>
  static bool Run(const Parameter& param, const Circuit<Operation>& circuit,
                  const typename Simulator::StateSpace& state_space,
                  const Simulator& simulator, uint64_t seed,
                  typename Simulator::State& state, SymTable& symtab,
                  Obss& obss) {
    using fp_type = OpFpType<Operation>;

    std::vector<GateOrOperation<fp_type, Operation>> deferred_ops;
    deferred_ops.reserve(4 * circuit.ops.size());

    RGen rgen(seed);

    return Run(param, circuit.ops, state_space, simulator,
               deferred_ops, rgen, state, symtab, obss);
  }

 private:
  template <typename Operation, typename Simulator,
            typename DeferredOps, typename SymTable, typename Obss>
  static bool Run(const Parameter& param, const std::vector<Operation>& ops,
                  const typename Simulator::StateSpace& state_space,
                  const Simulator& simulator, DeferredOps& deferred_ops,
                  RGen& rgen, typename Simulator::State& state,
                  SymTable& symtab, Obss& obss) {
    using fp_type = OpFpType<Operation>;
    using Channel = qsim::Channel<fp_type>;
    using CCOP = qsim::ClassicallyControlledOperation<fp_type>;

    deferred_ops.clear();

    std::uniform_real_distribution<double> distr(0.0, 1.0);

    // Flag to track non-unitary channels after state normalization.
    bool non_unitary = false;

    for (const auto& op : ops) {
      if (const auto* m = OpGetAlternative<Measurement>(op)) {
        // Measurement gate.

        if (!ApplyDeferredOps(param, simulator, deferred_ops, state, symtab)) {
          return false;
        }

        bool normalize = non_unitary && param.normalize_before_mea_gates;
        NormalizeState(normalize, state_space, non_unitary, state);

        ApplyMeasurementGate(state_space, *m, rgen, state, symtab, obss);

        continue;
      } else if (const auto* ccop = OpGetAlternative<CCOP>(op)) {
        // Classicaly controlled operation.

        if (!ApplyDeferredOps(param, simulator, deferred_ops, state, symtab)) {
          return false;
        }

        auto f = [&deferred_ops](
            const auto& param, const auto& ops, const auto& state_space,
            const auto& simulator, auto& rgen, auto& state, auto& symtab,
            auto& obss) {
          return Run(param, ops, state_space, simulator,
                     deferred_ops, rgen, state, symtab, obss);
        };

        bool rc = RunClassicalControl(param, *ccop, state_space,
                                      simulator, rgen, state, symtab, obss, f);

        if (!rc) {
          return false;
        }

        continue;
      }

      const auto* c = OpGetAlternative<Channel>(op);

      if (!c) {
        DeferOp(op, deferred_ops);
        continue;
      }

      // Channel.

      double r = distr(rgen);
      double cp = 0;

      const auto& channel = *c;

      // Perform sampling of Kraus operators using probability bounds.
      for (std::size_t i = 0; i < channel.kops.size(); ++i) {
        const auto& kop = channel.kops[i];

        cp += kop.prob;

        if (r < cp) {
          DeferOps(kop.ops, deferred_ops);
          non_unitary = non_unitary || !kop.unitary;
          break;
        }
      }

      if (r < cp) continue;

      if (!ApplyDeferredOps(param, simulator, deferred_ops, state, symtab)) {
        return false;
      }

      NormalizeState(non_unitary, state_space, non_unitary, state);

      double max_prob = 0;
      std::size_t max_prob_index = 0;

      // Perform sampling of Kraus operators using norms of updated states.
      for (std::size_t i = 0; i < channel.kops.size(); ++i) {
        const auto& kop = channel.kops[i];

        if (kop.unitary) continue;

        double prob = std::real(
            simulator.ExpectationValue(kop.qubits, kop.kd_k.data(), state));

        if (prob > max_prob) {
          max_prob = prob;
          max_prob_index = i;
        }

        cp += prob - kop.prob;

        if (r < cp || i == channel.kops.size() - 1) {
          // Sample ith Kraus operator if r < cp
          // Sample the highest probability Kraus operator if r is greater
          // than the sum of all probabilities due to round-off errors.
          uint64_t k = r < cp ? i : max_prob_index;

          DeferOps(channel.kops[k].ops, deferred_ops);
          non_unitary = true;
          break;
        }
      }
    }

    if (!ApplyDeferredOps(param, simulator, deferred_ops, state, symtab)) {
      return false;
    }

    NormalizeState(non_unitary, state_space, non_unitary, state);

    return true;
  }

  template <typename Simulator, typename Deferred, typename SymTable>
  static bool ApplyDeferredOps(
      const Parameter& param, const Simulator& simulator,
      std::vector<Deferred>& ops, typename Simulator::State& state,
      SymTable& symtab) {
    if (ops.size() == 0) return true;

    auto fused_ops = Fuser::FuseGates(param, state.num_qubits(), ops);
    if (fused_ops.size() == 0) {
      Error::Throw("fuser failed");
    }

    ops.clear();

    // Apply fused operations.
    for (auto& fop : fused_ops) {
      using FusedGate = qsim::FusedGate<OpFpType<decltype(fop)>>;

      if (auto* fg = OpGetAlternative<FusedGate>(fop)) {
        if (fg->defer_matrix_computation) {
          for (auto& g : fg->gates) {
            if (auto* ffg = FusedGate::GetRuntimeResolvedGate(g)) {
              for (unsigned i = 0; i < ffg->params.size(); ++i) {
                ffg->params[i] = cc::EvalExpr(symtab, ffg->param_exprs[i]);
              }
              ffg->matrix_func(ffg->params, ffg->matrix);
            }
          }

          CalculateFusedMatrix(*fg);
        }
      }

      ApplyGate(simulator, fop, state);
    }

    return true;
  }

  template <typename Operation, typename Deferred>
  static void DeferOp(const Operation& op, std::vector<Deferred>& ops) {
    ops.push_back(&op);
  }

  template <typename Gate, typename Deferred>
  static void DeferOps(
      const std::vector<Gate>& gates, std::vector<Deferred>& ops) {
    for (const auto& gate : gates) {
      ops.push_back(&gate);
    }
  }

  template <typename StateSpace>
  static void NormalizeState(bool normalize, const StateSpace& state_space,
                             bool& flag, typename StateSpace::State& state) {
    if (normalize) {
      double a = 1.0 / std::sqrt(state_space.Norm(state));
      state_space.Multiply(a, state);
      flag = false;
    }
  }
};

}  // namespace qsim

#endif  // RUN_QSIM_H_
