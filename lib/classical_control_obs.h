#ifndef CLASSICAL_CONTROL_OBS_H_
#define CLASSICAL_CONTROL_OBS_H_

#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qsim::cc {

/**
 * Tracks raw measurement counts for quantum circuit trajectories.
 * Maintains two count vectors of size 2^num_qubits:
 * - `cur_count`: Counts accumulated during the current active trajectory run.
 * - `total_count`: Aggregate counts across all completed trajectories.
 */
struct MeasurementHistogram {
  /** Constructs an empty, uninitialized MeasurementHistogram. */
  MeasurementHistogram() {}

  /**
   * Constructs a MeasurementHistogram allocated for 2^num_qubits outcomes.
   * @param num_qubits Number of measured qubits determining histogram bit
   *   width.
   */
  MeasurementHistogram(unsigned num_qubits) : num_qubits(num_qubits) {
    uint64_t size = uint64_t{1} << num_qubits;
    cur_count.resize(size, 0);
    total_count.resize(size, 0);
  }

  /**
   * Resets current trajectory counts without accumulating them into
   * total counts. Called when a simulation trajectory is aborted via
   * a `discard` condition.
   */
  void Discard() {
    for (auto& v : cur_count) {
      v = 0;
    }
  }

  /**
   * Flushes `cur_count` into `total_count` and zeroes `cur_count` for
   * the next trajectory.
   */
  void Update() {
    for (std::size_t i = 0; i < cur_count.size(); ++i) {
      total_count[i] += cur_count[i];
      cur_count[i] = 0;
    }
  }

  /**
   * Returns the number of qubits tracked by this histogram.
   * @return Qubit count.
   */
  unsigned Size() const {
    return num_qubits;
  }

  /** Measurement counts for the active trajectory. */
  std::vector<uint64_t> cur_count;
  /** Accumulated measurement counts across all completed trajectories. */
  std::vector<uint64_t> total_count;
  /** Number of qubits tracked by this histogram. */
  unsigned num_qubits = 0;
};

/**
 * Type alias for simulation observables (currently defaults to
 * MeasurementHistogram).
 */
using Observable = MeasurementHistogram;

/**
 * Map container managing named `Observable` instances (measurement histograms).
 */
struct Observables {
  /**
   * Inserts or replaces a named observable.
   * @param name Unique string identifier for the observable.
   * @param obs Rvalue instance of Observable to move into storage.
   * @return Pointer to the inserted Observable in storage.
   */
  Observable* Insert(std::string_view name, Observable&& obs) {
    return &(obss[name] = std::move(obs));
  }

  /**
   * Searches for a named observable.
   * @param name Target identifier name.
   * @return Pointer to the matching Observable or `nullptr` if not found.
   */
  Observable* Lookup(std::string_view name) {
    auto o = obss.find(name);
    return o != obss.end() ? &o->second : nullptr;
  }

  /** Const version of Lookup */
  const Observable* Lookup(std::string_view name) const {
    auto o = obss.find(name);
    return o != obss.end() ? &o->second : nullptr;
  }

  /**
   * Checks if the container holds any observables.
   * @return True if no observables are registered.
   */
  bool Empty() const {
    return obss.empty();
  }

  /**
   * Invokes callable f on each `(name, observable)` pair in storage.
   * @tparam F Callable type with signature
   *   `void(std::string_view, Observable&)`.
   * @param f Function or lambda to execute per element.
   */
  template <typename F>
  void Iterate(F&& f) {
    for (auto& [name, obs] : obss) {
      f(name, obs);
    }
  }

  /** Const version of Iterate. */
  template <typename F>
  void Iterate(F&& f) const {
    for (const auto& [name, obs] : obss) {
      f(name, obs);
    }
  }

 private:
  std::unordered_map<std::string_view, Observable> obss;
};

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_OBS_H_
