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
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "../lib/bitstring.h"
#include "../lib/formux.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/run_qsim.h"
#include "../lib/run_qsimh.h"
#include "../lib/simmux.h"
#include "../lib/util.h"

using namespace qsim;

namespace {

template <typename T>
T parseOptions(const py::dict &options, const char *key) {
  if (!options.contains(key)) {
    char msg[100];
    std::sprintf(msg, "Argument %s is not provided.\n", key);
    throw std::invalid_argument(msg);
  }
  const auto &value = options[key];
  return value.cast<T>();
}

Circuit<Cirq::GateCirq<float>> getCircuit(const py::dict &options) {
  try {
    return options["c\0"].cast<Circuit<Cirq::GateCirq<float>>>();
  } catch (const std::invalid_argument &exp) {
    throw;
  }
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
    throw std::invalid_argument("Unable to parse provided bit strings.\n");
  }
  return bitstrings;
}

}  // namespace

void add_gate(const qsim::Cirq::GateKind gate_kind, const unsigned time,
              const std::vector<unsigned>& qubits,
              const std::map<std::string, float>& params,
              Circuit<Cirq::GateCirq<float>>* circuit) {
  switch (gate_kind) {
    case Cirq::kI:
      circuit->gates.push_back(Cirq::I<float>::Create(time, qubits[0]));
      break;
    case Cirq::kI2:
      circuit->gates.push_back(
        Cirq::I2<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kXPowGate:
      circuit->gates.push_back(
        Cirq::XPowGate<float>::Create(time, qubits[0], params.at("exponent"),
                                      params.at("global_shift")));
      break;
    case Cirq::kYPowGate:
      circuit->gates.push_back(
        Cirq::YPowGate<float>::Create(time, qubits[0], params.at("exponent"),
                                      params.at("global_shift")));
      break;
    case Cirq::kZPowGate:
      circuit->gates.push_back(
        Cirq::ZPowGate<float>::Create(time, qubits[0], params.at("exponent"),
                                      params.at("global_shift")));
      break;
    case Cirq::kHPowGate:
      circuit->gates.push_back(
        Cirq::HPowGate<float>::Create(time, qubits[0], params.at("exponent"),
                                      params.at("global_shift")));
      break;
    case Cirq::kCZPowGate:
      circuit->gates.push_back(
        Cirq::CZPowGate<float>::Create(time, qubits[0], qubits[1],
                                       params.at("exponent"),
                                       params.at("global_shift")));
      break;
    case Cirq::kCXPowGate:
      circuit->gates.push_back(
        Cirq::CXPowGate<float>::Create(time, qubits[0], qubits[1],
                                       params.at("exponent"),
                                       params.at("global_shift")));
      break;
    case Cirq::krx:
      circuit->gates.push_back(
        Cirq::rx<float>::Create(time, qubits[0], params.at("phi")));
      break;
    case Cirq::kry:
      circuit->gates.push_back(
        Cirq::ry<float>::Create(time, qubits[0], params.at("phi")));
      break;
    case Cirq::krz:
      circuit->gates.push_back(
        Cirq::rz<float>::Create(time, qubits[0], params.at("phi")));
      break;
    case Cirq::kH:
      circuit->gates.push_back(Cirq::H<float>::Create(time, qubits[0]));
      break;
    case Cirq::kS:
      circuit->gates.push_back(Cirq::S<float>::Create(time, qubits[0]));
      break;
    case Cirq::kCZ:
      circuit->gates.push_back(
        Cirq::CZ<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kCX:
      circuit->gates.push_back(
        Cirq::CX<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kT:
      circuit->gates.push_back(Cirq::T<float>::Create(time, qubits[0]));
      break;
    case Cirq::kX:
      circuit->gates.push_back(Cirq::X<float>::Create(time, qubits[0]));
      break;
    case Cirq::kY:
      circuit->gates.push_back(Cirq::Y<float>::Create(time, qubits[0]));
      break;
    case Cirq::kZ:
      circuit->gates.push_back(Cirq::Z<float>::Create(time, qubits[0]));
      break;
    case Cirq::kPhasedXPowGate:
      circuit->gates.push_back(
        Cirq::PhasedXPowGate<float>::Create(time, qubits[0],
                                            params.at("phase_exponent"),
                                            params.at("exponent"),
                                            params.at("global_shift")));
      break;
    case Cirq::kPhasedXZGate:
      circuit->gates.push_back(
        Cirq::PhasedXZGate<float>::Create(time, qubits[0],
                                          params.at("x_exponent"),
                                          params.at("z_exponent"),
                                          params.at("axis_phase_exponent")));
      break;
    case Cirq::kXXPowGate:
      circuit->gates.push_back(
        Cirq::XXPowGate<float>::Create(time, qubits[0], qubits[1],
                                       params.at("exponent"),
                                       params.at("global_shift")));
      break;
    case Cirq::kYYPowGate:
      circuit->gates.push_back(
        Cirq::YYPowGate<float>::Create(time, qubits[0], qubits[1],
                                       params.at("exponent"),
                                       params.at("global_shift")));
      break;
    case Cirq::kZZPowGate:
      circuit->gates.push_back(
        Cirq::ZZPowGate<float>::Create(time, qubits[0], qubits[1],
                                       params.at("exponent"),
                                       params.at("global_shift")));
      break;
    case Cirq::kXX:
      circuit->gates.push_back(
        Cirq::XX<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kYY:
      circuit->gates.push_back(
        Cirq::YY<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kZZ:
      circuit->gates.push_back(
        Cirq::ZZ<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kSwapPowGate:
      circuit->gates.push_back(
        Cirq::SwapPowGate<float>::Create(time, qubits[0], qubits[1],
                                         params.at("exponent"),
                                         params.at("global_shift")));
      break;
    case Cirq::kISwapPowGate:
      circuit->gates.push_back(
        Cirq::ISwapPowGate<float>::Create(time, qubits[0], qubits[1],
                                          params.at("exponent"),
                                          params.at("global_shift")));
      break;
    case Cirq::kriswap:
      circuit->gates.push_back(
        Cirq::riswap<float>::Create(time, qubits[0], qubits[1],
                                    params.at("phi")));
      break;
    case Cirq::kSWAP:
      circuit->gates.push_back(
        Cirq::SWAP<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kISWAP:
      circuit->gates.push_back(
        Cirq::ISWAP<float>::Create(time, qubits[0], qubits[1]));
      break;
    case Cirq::kPhasedISwapPowGate:
      circuit->gates.push_back(
        Cirq::PhasedISwapPowGate<float>::Create(time, qubits[0], qubits[1],
                                                params.at("phase_exponent"),
                                                params.at("exponent")));
      break;
    case Cirq::kgivens:
      circuit->gates.push_back(
        Cirq::givens<float>::Create(time, qubits[0], qubits[1],
                                    params.at("phi")));
      break;
    case Cirq::kFSimGate:
      circuit->gates.push_back(
        Cirq::FSimGate<float>::Create(time, qubits[0], qubits[1],
                                      params.at("theta"), params.at("phi")));
      break;
    case Cirq::kMeasurement: {
      std::vector<unsigned> qubits_;
      for (unsigned i = 0; i < qubits.size(); i++){
        qubits_.push_back(qubits[i]);
      }
      circuit->gates.push_back(
        gate::Measurement<Cirq::GateCirq<float>>::Create(time, std::move(qubits_)));
      }
      break;
    // Matrix gates are handled in the add_matrix methods below.
    default:
      throw std::invalid_argument("GateKind not supported.");
  }
}

// The Matrix(1|2)q objects are typedefs for Python-compatible objects.
void add_matrix1(const unsigned time, const std::vector<unsigned>& qubits,
                 const qsim::Cirq::Matrix1q<float>& matrix,
                 Circuit<Cirq::GateCirq<float>>* circuit) {
  circuit->gates.push_back(
    Cirq::MatrixGate1<float>::Create(time, qubits[0], matrix));
}
void add_matrix2(const unsigned time, const std::vector<unsigned>& qubits,
                 const qsim::Cirq::Matrix2q<float>& matrix,
                 Circuit<Cirq::GateCirq<float>>* circuit) {
  circuit->gates.push_back(
    Cirq::MatrixGate2<float>::Create(time, qubits[0], qubits[1], matrix));
}

std::vector<std::complex<float>> qsim_simulate(const py::dict &options) {
  Circuit<Cirq::GateCirq<float>> circuit;
  std::vector<Bitstring> bitstrings;
  try {
    circuit = getCircuit(options);
    bitstrings = getBitstrings(options, circuit.num_qubits);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  using Simulator = qsim::Simulator<For>;
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

  using Runner = QSimRunner<IO, BasicGateFuser<IO, Cirq::GateCirq<float>>,
                            Simulator>;

  Runner::Parameter param;
  try {
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }
  Runner::Run(param, circuit, measure);
  return amplitudes;
}

py::array_t<float> qsim_simulate_fullstate(const py::dict &options) {
  Circuit<Cirq::GateCirq<float>> circuit;
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
    IO::errorf("Memory allocation failed.\n");
    return {};
  }

  using Simulator = qsim::Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<IO, Cirq::GateCirq<float>>,
                            Simulator>;

  Runner::Parameter param;
  try {
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  StateSpace state_space(circuit.num_qubits, param.num_threads);
  State state = state_space.CreateState(fsv);

  state_space.SetStateZero(state);

  if (!Runner::Run(param, circuit, state)) {
    IO::errorf("qsim full state simulation of the circuit errored out.\n");
    return {};
  }

  state_space.InternalToNormalOrder(state);

  auto capsule = py::capsule(
      fsv, [](void *data) { delete reinterpret_cast<float *>(data); });
  return py::array_t<float>(fsv_size, fsv, capsule);
}

std::vector<std::complex<float>> qsimh_simulate(const py::dict &options) {
  using Simulator = qsim::Simulator<For>;
  using HybridSimulator = HybridSimulator<IO, Cirq::GateCirq<float>,
                                          BasicGateFuser, Simulator, For>;
  using Runner = QSimHRunner<IO, HybridSimulator>;

  Circuit<Cirq::GateCirq<float>> circuit;
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
      IO::errorf("Invalid arguments are provided for arg k.\n");
      return {};
    }
    parts[i.cast<unsigned>()] = 1;
  }

  // Define container for amplitudes
  std::vector<std::complex<float>> amplitudes(bitstrings.size(), 0);

  if (Runner::Run(param, circuit, parts, bitstrings, amplitudes)) {
    return amplitudes;
  }
  IO::errorf("qsimh simulation of the circuit errored out.\n");
  return {};
}
