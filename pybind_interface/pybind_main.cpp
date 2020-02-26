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

#include "pybind_main.h"

#include <complex>
#include <limits>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "../lib/bitstring.h"
#include "../lib/circuit_reader.h"
#include "../lib/fuser_basic.h"
#include "../lib/gate.h"
#include "../lib/io.h"
#include "../lib/parfor.h"
#include "../lib/run_qsim.h"
#include "../lib/run_qsimh.h"
#include "../lib/simulator_avx.h"
#include "../lib/util.h"

using namespace qsim;

namespace {

template <typename T>
T parseOptions(const py::dict &options, const char *key) {
  if (!options.contains(key)) {
    char msg[100];
    std::sprintf(msg, "Argument %s is not provided.", key);
    throw std::invalid_argument(msg);
  }
  const auto &value = options[key];
  return value.cast<T>();
}

Circuit<Gate<float>> getCircuit(const py::dict &options) {
  Circuit<Gate<float>> circuit;
  std::string circuit_str;
  try {
    circuit_str = parseOptions<std::string>(options, "c\0");
  } catch (const std::invalid_argument &exp) {
    throw;
  }
  std::stringstream ss(circuit_str);
  if (!CircuitReader<IO>::FromStream(std::numeric_limits<unsigned>::max(),
                                     "cirq_circuit_str", ss, circuit)) {
    throw std::invalid_argument("Unable to parse provided circuit.");
  }
  return circuit;
}

std::vector<Bitstring> getBitstrings(const py::dict &options, int num_qubits) {
  std::string bitstrings_str;
  try {
    bitstrings_str = parseOptions<std::string>(options, "i\0");
  } catch (const std::invalid_argument &exp) {
    throw;
  }
  std::stringstream bitstrings_stream(bitstrings_str);
  std::vector<Bitstring> bitstrings;

  if (!BitstringsFromStream<IO>(num_qubits, "bitstrings_str", bitstrings_stream,
                                bitstrings)) {
    throw std::invalid_argument("Unable to parse provided bit strings.");
  }
  return bitstrings;
}

auto reorder_fsv = [](unsigned n, unsigned m, uint64_t i, float *fsv) {
  auto a = fsv + 16 * i;

  auto r1 = a[1];
  auto r2 = a[2];
  auto r3 = a[3];
  auto r4 = a[4];
  auto r5 = a[5];
  auto r6 = a[6];
  auto r7 = a[7];
  auto i0 = a[8];
  auto i1 = a[9];
  auto i2 = a[10];
  auto i3 = a[11];
  auto i4 = a[12];
  auto i5 = a[13];
  auto i6 = a[14];

  a[1] = i0;
  a[2] = r1;
  a[3] = i1;
  a[4] = r2;
  a[5] = i2;
  a[6] = r3;
  a[7] = i3;
  a[8] = r4;
  a[9] = i4;
  a[10] = r5;
  a[11] = i5;
  a[12] = r6;
  a[13] = i6;
  a[14] = r7;
};

}  // namespace

std::vector<std::complex<float>> qsim_simulate(const py::dict &options) {
  Circuit<Gate<float>> circuit;
  std::vector<Bitstring> bitstrings;
  try {
    circuit = getCircuit(options);
    bitstrings = getBitstrings(options, circuit.num_qubits);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;

  // Define container for amplitudes
  std::vector<std::complex<float>> amplitudes;
  amplitudes.reserve(bitstrings.size());

  auto measure = [&bitstrings, &circuit, &amplitudes](
                     unsigned k, const StateSpace &state_space,
                     const State &state) {
    for (const auto &b : bitstrings) {
      amplitudes.push_back(state_space.GetAmpl(state, b));
    }
  };

  using Runner = QSimRunner<IO, BasicGateFuser<Gate<float>>, Simulator>;

  Runner::Parameter param;
  try {
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }
  Runner::Run(param, std::numeric_limits<unsigned>::max(), circuit, measure);
  return amplitudes;
}

py::array_t<float> qsim_simulate_fullstate(const py::dict &options) {
  Circuit<Gate<float>> circuit;
  try {
    circuit = getCircuit(options);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  float *fsv;
  const uint64_t fsv_size = std::pow(2, circuit.num_qubits + 1);
  const uint64_t buff_size = std::max(fsv_size, (uint64_t)16);
  if (posix_memalign((void **)&fsv, 32, buff_size * sizeof(float))) {
    IO::errorf("Memory allocation failed");
    return {};
  }

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;

  auto measure = [](unsigned k, const StateSpace &state_space,
                    const State &state) {};

  using Runner = QSimRunner<IO, BasicGateFuser<Gate<float>>, Simulator>;

  Runner::Parameter param;
  try {
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }
  if (!Runner::Run(param, std::numeric_limits<unsigned>::max(), circuit,
                   measure, fsv)) {
    IO::errorf("Qsim full state simulation of the circuit errored out.");
    return {};
  }
  if (circuit.num_qubits == 1) {
    fsv[2] = fsv[1];
    fsv[1] = fsv[8];
    fsv[3] = fsv[9];
  } else if (circuit.num_qubits == 2) {
    fsv[6] = fsv[3];
    fsv[4] = fsv[2];
    fsv[2] = fsv[1];
    fsv[1] = fsv[8];
    fsv[3] = fsv[9];
    fsv[5] = fsv[10];
    fsv[7] = fsv[11];
  } else {
    ParallelFor::Run(param.num_threads, buff_size / 16, reorder_fsv, fsv);
  }
  auto capsule = py::capsule(
      fsv, [](void *data) { delete reinterpret_cast<float *>(data); });
  return py::array_t<float>(fsv_size, fsv, capsule);
}

std::vector<std::complex<float>> qsimh_simulate(const py::dict &options) {
  using Simulator = SimulatorAVX<ParallelFor>;
  using HybridSimulator =
      HybridSimulator<IO, BasicGateFuser, Simulator, ParallelFor>;
  using Runner = QSimHRunner<IO, HybridSimulator>;

  Circuit<Gate<float>> circuit;
  std::vector<Bitstring> bitstrings;
  Runner::Parameter param;
  py::list dense_parts;

  try {
    circuit = getCircuit(options);
    bitstrings = getBitstrings(options, circuit.num_qubits);
    dense_parts = parseOptions<py::list>(options, "k\0");
    param.prefix = parseOptions<uint64_t>(options, "w\0");
    param.num_prefix_gatexs = parseOptions<unsigned>(options, "p\0");
    param.num_root_gatexs = parseOptions<unsigned>(options, "r\0");
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  std::vector<unsigned> parts(circuit.num_qubits, 0);
  for (auto i : dense_parts) {
    unsigned idx = i.cast<unsigned>();
    if (idx >= circuit.num_qubits) {
      IO::errorf("Invalid arguments are provided for arg k.");
      return {};
    }
    parts[i.cast<unsigned>()] = 1;
  }

  // Define container for amplitudes
  std::vector<std::complex<float>> amplitudes(bitstrings.size(), 0);

  if (Runner::Run(param, std::numeric_limits<unsigned>::max(), parts,
                  circuit.gates, bitstrings, amplitudes)) {
    return amplitudes;
  }
  IO::errorf("Qsimh simulation of the circuit errored out.");
  return {};
}
