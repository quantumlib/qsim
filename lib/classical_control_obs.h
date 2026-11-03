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

#ifndef CLASSICAL_CONTROL_OBS_H_
#define CLASSICAL_CONTROL_OBS_H_

#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "error.h"

namespace qsim::cc {

/**
 * Tracks raw measurement counts for quantum circuit trajectories.
 * Maintains a count vector of size 2^num_qubits:
 * - `count`: Aggregate counts across all completed trajectories.
 * Not recommended for use when the number of measured qubits (num_qubits)
 * is large.
 */
struct MeasurementHistogram {
  /** Constructs an empty, uninitialized MeasurementHistogram. */
  MeasurementHistogram() {}

  /**
   * Constructs a MeasurementHistogram allocated for 2^num_qubits outcomes.
   * @param num_qubits Number of measured qubits determining the histogram bit
   *   width.
   */
  MeasurementHistogram(unsigned num_qubits) : num_qubits(num_qubits) {
    uint64_t size = uint64_t{1} << num_qubits;
    count.resize(size, 0);
  }

  /**
   * Adds a measured bitstring.
   * @param bitstring Measured bitstring to add.
   * @throws std::runtime_error If bitstring exceeds the maximum value or
   *   if called a second time before calling `Discard` or `Update`.
   */
  void AddBitstring(uint64_t bitstring) {
    if (bitstring >= count.size()) {
      Error::Throw(
          "bitstring value {} must be less than 2^{num_measured_qubits} = {} ",
          bitstring, count.size());
    }

    if (have_bitstring) {
      Error::Throw("cannot add second bitstring");
    }

    this->bitstring = bitstring;
    have_bitstring = true;
  }

  /**
   * Invalidates the last measured bitstring. Called when a simulation
   * trajectory is aborted via a `discard` condition.
   */
  void Discard() {
    have_bitstring = false;
  }

  /**
   * Flushes the last measured bitstring into `count` and invalidates it for
   * the next trajectory.
   */
  void Update() {
    if (have_bitstring) {
      ++count[bitstring];
      have_bitstring = false;
    }
  }

  /**
   * Returns the number of qubits tracked by this histogram.
   * @return Qubit count.
   */
  unsigned Size() const {
    return num_qubits;
  }

  /**
   * Returns a const reference to `count`.
   * @return Reference to `count`
   */
  const std::vector<uint64_t>& Count() const {
    return count;
  }

 private:
  /** Accumulated measurement counts across all completed trajectories. */
  std::vector<uint64_t> count;
  /** Last measured bitstring. */
  uint64_t bitstring;
  /** Number of qubits tracked by this histogram. */
  unsigned num_qubits = 0;
  /** Used to prevent double `AddBitstring` calls. */
  bool have_bitstring = false;
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
