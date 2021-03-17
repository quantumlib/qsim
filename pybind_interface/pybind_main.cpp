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
#include "../lib/expect.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/qtrajectory.h"
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

NoisyCircuit<Cirq::GateCirq<float>> getNoisyCircuit(const py::dict &options) {
  try {
    return options["c\0"].cast<NoisyCircuit<Cirq::GateCirq<float>>>();
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

Cirq::GateCirq<float> create_gate(const qsim::Cirq::GateKind gate_kind,
                                  const unsigned time,
                                  const std::vector<unsigned>& qubits,
                                  const std::map<std::string, float>& params) {
  switch (gate_kind) {
    case Cirq::kI1:
      return Cirq::I1<float>::Create(time, qubits[0]);
    case Cirq::kI2:
      return Cirq::I2<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kI:
      return Cirq::I<float>::Create(time, qubits);
    case Cirq::kXPowGate:
      return Cirq::XPowGate<float>::Create(
        time, qubits[0], params.at("exponent"), params.at("global_shift"));
    case Cirq::kYPowGate:
      return Cirq::YPowGate<float>::Create(
        time, qubits[0], params.at("exponent"), params.at("global_shift"));
    case Cirq::kZPowGate:
      return Cirq::ZPowGate<float>::Create(
        time, qubits[0], params.at("exponent"), params.at("global_shift"));
    case Cirq::kHPowGate:
      return Cirq::HPowGate<float>::Create(
        time, qubits[0], params.at("exponent"), params.at("global_shift"));
    case Cirq::kCZPowGate:
      return Cirq::CZPowGate<float>::Create(
        time, qubits[0], qubits[1],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kCXPowGate:
      return Cirq::CXPowGate<float>::Create(
        time, qubits[0], qubits[1],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::krx:
      return Cirq::rx<float>::Create(time, qubits[0], params.at("phi"));
    case Cirq::kry:
      return Cirq::ry<float>::Create(time, qubits[0], params.at("phi"));
    case Cirq::krz:
      return Cirq::rz<float>::Create(time, qubits[0], params.at("phi"));
    case Cirq::kH:
      return Cirq::H<float>::Create(time, qubits[0]);
    case Cirq::kS:
      return Cirq::S<float>::Create(time, qubits[0]);
    case Cirq::kCZ:
      return Cirq::CZ<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kCX:
      return Cirq::CX<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kT:
      return Cirq::T<float>::Create(time, qubits[0]);
    case Cirq::kX:
      return Cirq::X<float>::Create(time, qubits[0]);
    case Cirq::kY:
      return Cirq::Y<float>::Create(time, qubits[0]);
    case Cirq::kZ:
      return Cirq::Z<float>::Create(time, qubits[0]);
    case Cirq::kPhasedXPowGate:
      return Cirq::PhasedXPowGate<float>::Create(
        time, qubits[0], params.at("phase_exponent"), params.at("exponent"),
        params.at("global_shift"));
    case Cirq::kPhasedXZGate:
      return Cirq::PhasedXZGate<float>::Create(
        time, qubits[0], params.at("x_exponent"), params.at("z_exponent"),
        params.at("axis_phase_exponent"));
    case Cirq::kXXPowGate:
      return Cirq::XXPowGate<float>::Create(
        time, qubits[0], qubits[1], params.at("exponent"),
        params.at("global_shift"));
    case Cirq::kYYPowGate:
      return Cirq::YYPowGate<float>::Create(
        time, qubits[0], qubits[1],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kZZPowGate:
      return Cirq::ZZPowGate<float>::Create(
        time, qubits[0], qubits[1],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kXX:
      return Cirq::XX<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kYY:
      return Cirq::YY<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kZZ:
      return Cirq::ZZ<float>::Create(time, qubits[0], qubits[1]);
      break;
    case Cirq::kSwapPowGate:
      return Cirq::SwapPowGate<float>::Create(
        time, qubits[0], qubits[1],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kISwapPowGate:
      return Cirq::ISwapPowGate<float>::Create(
        time, qubits[0], qubits[1],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kriswap:
      return Cirq::riswap<float>::Create(time, qubits[0], qubits[1],
                                    params.at("phi"));
    case Cirq::kSWAP:
      return Cirq::SWAP<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kISWAP:
      return Cirq::ISWAP<float>::Create(time, qubits[0], qubits[1]);
    case Cirq::kPhasedISwapPowGate:
      return Cirq::PhasedISwapPowGate<float>::Create(
        time, qubits[0], qubits[1],
        params.at("phase_exponent"), params.at("exponent"));
    case Cirq::kgivens:
      return Cirq::givens<float>::Create(
        time, qubits[0], qubits[1], params.at("phi"));
    case Cirq::kFSimGate:
      return Cirq::FSimGate<float>::Create(
        time, qubits[0], qubits[1], params.at("theta"), params.at("phi"));
    case Cirq::kCCZPowGate:
      return Cirq::CCZPowGate<float>::Create(
        time, qubits[0], qubits[1], qubits[2],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kCCXPowGate:
      return Cirq::CCXPowGate<float>::Create(
        time, qubits[0], qubits[1], qubits[2],
        params.at("exponent"), params.at("global_shift"));
    case Cirq::kCSwapGate:
      return Cirq::CSwapGate<float>::Create(
        time, qubits[0], qubits[1], qubits[2]);
    case Cirq::kCCZ:
      return Cirq::CCZ<float>::Create(time, qubits[0], qubits[1], qubits[2]);
    case Cirq::kCCX:
      return Cirq::CCX<float>::Create(time, qubits[0], qubits[1], qubits[2]);
    case Cirq::kMeasurement: {
      std::vector<unsigned> qubits_ = qubits;
      return gate::Measurement<Cirq::GateCirq<float>>::Create(
        time, std::move(qubits_));
      }
    // Matrix gates are handled in the add_matrix methods below.
    default:
      throw std::invalid_argument("GateKind not supported.");
  }
}

Cirq::GateCirq<float> create_diagonal_gate(const unsigned time,
                                           const std::vector<unsigned>& qubits,
                                           const std::vector<float>& angles) {
  switch (qubits.size()) {
  case 2:
    return Cirq::TwoQubitDiagonalGate<float>::Create(
      time, qubits[0], qubits[1], angles);
  case 3:
    return Cirq::ThreeQubitDiagonalGate<float>::Create(
      time, qubits[0], qubits[1], qubits[2], angles);
  default:
    throw std::invalid_argument(
        "Only 2- or 3-qubit diagonal gates sre supported.");
  }
}

Cirq::GateCirq<float> create_matrix_gate(const unsigned time,
                                         const std::vector<unsigned>& qubits,
                                         const std::vector<float>& matrix) {
  switch (qubits.size()) {
  case 1:
    return Cirq::MatrixGate1<float>::Create(time, qubits[0], matrix);
  case 2:
    return Cirq::MatrixGate2<float>::Create(time, qubits[0], qubits[1], matrix);
  case 3:
  case 4:
  case 5:
  case 6:
    return Cirq::MatrixGate<float>::Create(time, qubits, matrix);
  default:
    throw std::invalid_argument(
        "Only up to 6-qubit matrix gates are supported.");
  }
}

void add_gate(const qsim::Cirq::GateKind gate_kind, const unsigned time,
              const std::vector<unsigned>& qubits,
              const std::map<std::string, float>& params,
              Circuit<Cirq::GateCirq<float>>* circuit) {
  circuit->gates.push_back(create_gate(gate_kind, time, qubits, params));
}

void add_diagonal_gate(const unsigned time, const std::vector<unsigned>& qubits,
                       const std::vector<float>& angles,
                       Circuit<Cirq::GateCirq<float>>* circuit) {
  circuit->gates.push_back(create_diagonal_gate(time, qubits, angles));
}

void add_matrix_gate(const unsigned time,
                     const std::vector<unsigned>& qubits,
                     const std::vector<float>& matrix,
                     Circuit<Cirq::GateCirq<float>>* circuit) {
  circuit->gates.push_back(create_matrix_gate(time, qubits, matrix));
}

void control_last_gate(const std::vector<unsigned>& qubits,
                       const std::vector<unsigned>& values,
                       Circuit<Cirq::GateCirq<float>>* circuit) {
  MakeControlledGate(qubits, values, circuit->gates.back());
}

template <typename Gate>
Channel<Gate> create_single_gate_channel(Gate gate) {
  auto gate_kind = KrausOperator<Gate>::kNormal;
  if (gate.kind == gate::kMeasurement) {
    gate_kind = KrausOperator<Gate>::kMeasurement;
  }
  return {{gate_kind, 1, 1.0, {gate}}};
}

void add_gate_channel(const qsim::Cirq::GateKind gate_kind, const unsigned time,
                      const std::vector<unsigned>& qubits,
                      const std::map<std::string, float>& params,
                      NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  ncircuit->channels.push_back(create_single_gate_channel(
    create_gate(gate_kind, time, qubits, params)));
}

void add_diagonal_gate_channel(const unsigned time,
                               const std::vector<unsigned>& qubits,
                               const std::vector<float>& angles,
                               NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  ncircuit->channels.push_back(create_single_gate_channel(
    create_diagonal_gate(time, qubits, angles)));
}

void add_matrix_gate_channel(const unsigned time,
                             const std::vector<unsigned>& qubits,
                             const std::vector<float>& matrix,
                             NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  ncircuit->channels.push_back(create_single_gate_channel(
    create_matrix_gate(time, qubits, matrix)));
}

void control_last_gate_channel(const std::vector<unsigned>& qubits,
                               const std::vector<unsigned>& values,
                               NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  if (ncircuit->channels.back().size() > 1) {
    throw std::invalid_argument(
        "Control cannot be added to noisy channels.");
  }
  for (Cirq::GateCirq<float>& op : ncircuit->channels.back()[0].ops) {
    MakeControlledGate(qubits, values, op);
  }
}

void add_channel(const unsigned time,
                 const std::vector<unsigned>& qubits,
                 const std::vector<std::tuple<float, std::vector<float>, bool>>&
                     prob_matrix_unitary_triples,
                 NoisyCircuit<Cirq::GateCirq<float>>* ncircuit) {
  // Adds a channel to the noisy circuit.
  using Gate = Cirq::GateCirq<float>;
  Channel<Gate> channel;
  // prob_matrix_unitary_triples contains triples with these elements:
  //   0. The lower-bound probability of applying the matrix.
  //   1. The matrix to be applied.
  //   2. Whether the matrix is unitary.
  for (const auto &triple : prob_matrix_unitary_triples) {
    const float prob = std::get<0>(triple);
    const std::vector<float>& mat = std::get<1>(triple);
    bool is_unitary = std::get<2>(triple);
    Gate gate = create_matrix_gate(time, qubits, mat);
    channel.emplace_back(KrausOperator<Gate>{
      KrausOperator<Gate>::kNormal, is_unitary, prob, {gate}
    });
  }
  ncircuit->channels.push_back(channel);
}

void add_gate_to_opstring(const Cirq::GateKind gate_kind,
                          const std::vector<unsigned>& qubits,
                          OpString<Cirq::GateCirq<float>>* opstring) {
  static std::map<std::string, float> params;
  opstring->ops.push_back(create_gate(gate_kind, 0, qubits, params));
}

// Methods for simulating amplitudes.

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

  using Runner = QSimRunner<IO, MultiQubitGateFuser<IO, Cirq::GateCirq<float>>,
                            Simulator>;

  Runner::Parameter param;
  try {
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    param.seed = parseOptions<unsigned>(options, "s\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }
  Runner::Run(param, circuit, measure);
  return amplitudes;
}

std::vector<std::complex<float>> qtrajectory_simulate(const py::dict &options) {
  NoisyCircuit<Cirq::GateCirq<float>> ncircuit;
  unsigned num_qubits;
  std::vector<Bitstring> bitstrings;
  try {
    ncircuit = getNoisyCircuit(options);
    num_qubits = ncircuit.num_qubits;
    bitstrings = getBitstrings(options, num_qubits);
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

  using Runner = qsim::QuantumTrajectorySimulator<IO, Cirq::GateCirq<float>,
                                                  MultiQubitGateFuser,
                                                  Simulator>;

  Runner::Parameter param;
  unsigned num_threads;
  uint64_t seed;

  try {
    num_threads = parseOptions<unsigned>(options, "t\0");
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    seed = parseOptions<unsigned>(options, "s\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  Simulator simulator(num_threads);
  StateSpace state_space(num_threads);

  auto measure = [&bitstrings, &ncircuit, &amplitudes, &state_space](
                  unsigned k, const State &state,
                  std::vector<uint64_t>& stat) {
    for (const auto &b : bitstrings) {
      amplitudes.push_back(state_space.GetAmpl(state, b));
    }
  };

  if (!Runner::RunBatch(param, ncircuit, seed, seed + 1, state_space,
                        simulator, measure)) {
    IO::errorf("qtrajectory simulation of the circuit errored out.\n");
    return {};
  }
  return amplitudes;
}

// Helper class for simulating circuits of all types.
class SimulatorHelper {
 public:
  using Simulator = qsim::Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;

  using Gate = Cirq::GateCirq<float>;
  using Runner = QSimRunner<IO, MultiQubitGateFuser<IO, Gate>, Simulator>;
  using NoisyRunner = qsim::QuantumTrajectorySimulator<
      IO, Gate, MultiQubitGateFuser, Simulator>;

  SimulatorHelper() = delete;

  template <typename StateType>
  static py::array_t<float> simulate_fullstate(
      const py::dict &options, bool is_noisy, const StateType& input_state) {
    auto helper = SimulatorHelper(options, is_noisy);
    if (!helper.is_valid || !helper.simulate(input_state)) {
      return {};
    }
    return helper.release_state_to_python();
  }

  template <typename StateType>
  static std::vector<std::complex<double>> simulate_expectation_values(
      const py::dict &options,
      const std::vector<std::tuple<
                            std::vector<OpString<Cirq::GateCirq<float>>>,
                            unsigned>>& opsums_and_qubit_counts,
      bool is_noisy, const StateType& input_state) {
    auto helper = SimulatorHelper(options, is_noisy);
    if (!helper.is_valid) {
      return {};
    }
    if (!is_noisy) {
      if (!helper.simulate(input_state)) {
        return {};
      }
      return helper.get_expectation_value(opsums_and_qubit_counts);
    }

    // Aggregate expectation values for noisy circuits.
    std::vector<std::complex<double>> results(
      opsums_and_qubit_counts.size(), 0);
    for (unsigned rep = 0; rep < helper.noisy_reps; ++rep) {
      if (!helper.simulate(input_state)) {
        return {};
      }
      auto evs = helper.get_expectation_value(opsums_and_qubit_counts);
      for (unsigned i = 0; i < evs.size(); ++i) {
        results[i] += evs[i];
      }
    }
    double inverse_num_reps = 1.0 / helper.noisy_reps;
    for (unsigned i = 0; i < results.size(); ++i) {
      results[i] *= inverse_num_reps;
    }
    return results;
  }

 private:
  SimulatorHelper(const py::dict &options, bool noisy)
      : state_space(StateSpace(1)), state(StateSpace::Null()),
        scratch(StateSpace::Null()) {
    is_valid = false;
    is_noisy = noisy;
    try {
      if (is_noisy) {
        ncircuit = getNoisyCircuit(options);
        num_qubits = ncircuit.num_qubits;
        noisy_reps = parseOptions<unsigned>(options, "r\0");
      } else {
        circuit = getCircuit(options);
        num_qubits = circuit.num_qubits;
      }
      num_threads = parseOptions<unsigned>(options, "t\0");
      max_fused_size = parseOptions<unsigned>(options, "f\0");
      verbosity = parseOptions<unsigned>(options, "v\0");
      seed = parseOptions<unsigned>(options, "s\0");

      state_space = StateSpace(num_threads);
      state = state_space.Create(num_qubits);
      is_valid = true;
    } catch (const std::invalid_argument &exp) {
      // If this triggers, is_valid is false.
      IO::errorf(exp.what());
    }
  }

  void init_state(uint64_t input_state) {
    state_space.SetAllZeros(state);
    state_space.SetAmpl(state, input_state, 1, 0);
  }

  void init_state(const py::array_t<float> &input_vector) {
    const float* ptr = input_vector.data();
    auto f = [](unsigned n, unsigned m, uint64_t i, const float* ptr,
                float* fsv) {
      fsv[i] = ptr[i];
    };
    For(num_threads).Run(input_vector.size(), f, ptr, state.get());
    state_space.NormalToInternalOrder(state);
  }

  Runner::Parameter get_params() const {
    Runner::Parameter params;
    params.num_threads = num_threads;
    params.max_fused_size = max_fused_size;
    params.verbosity = verbosity;
    params.seed = seed;
    return params;
  }

  NoisyRunner::Parameter get_noisy_params() const {
    NoisyRunner::Parameter params;
    params.max_fused_size = max_fused_size;
    params.verbosity = verbosity;
    return params;
  }

  template <typename StateType>
  bool simulate(const StateType& input_state) {
    init_state(input_state);
    bool result = false;
    if (is_noisy) {
      std::vector<uint64_t> stat;
      auto params = get_noisy_params();

      Simulator simulator(num_threads);
      StateSpace state_space(num_threads);

      result = NoisyRunner::RunOnce(params, ncircuit, seed, state_space,
                                    simulator, scratch, state, stat);
    } else {
      result = Runner::Run(get_params(), circuit, state);
    }
    seed += 1;
    return result;
  }

  py::array_t<float> release_state_to_python() {
    state_space.InternalToNormalOrder(state);
    uint64_t fsv_size = 2 * (uint64_t{1} << num_qubits);
    float* fsv = state.release();
    auto capsule = py::capsule(
        fsv, [](void *data) { delete reinterpret_cast<float *>(data); });
    return py::array_t<float>(fsv_size, fsv, capsule);
  }

  std::vector<std::complex<double>> get_expectation_value(
      const std::vector<std::tuple<std::vector<OpString<Gate>>,
                                   unsigned>>& opsums_and_qubit_counts) {
    Simulator simulator(num_threads);
    using Fuser = MultiQubitGateFuser<IO, Gate>;

    std::vector<std::complex<double>> results;
    results.reserve(opsums_and_qubit_counts.size());
    for (const auto& opsum_qubit_count_pair : opsums_and_qubit_counts) {
      const auto& opsum = std::get<0>(opsum_qubit_count_pair);
      const auto& opsum_qubits = std::get<1>(opsum_qubit_count_pair);
      if (opsum_qubits <= 6) {
        results.push_back(ExpectationValue<IO, Fuser>(opsum, simulator, state));
      } else {
        Fuser::Parameter param;
        results.push_back(ExpectationValue<Fuser>(
            param, opsum, state_space, simulator, state, scratch));
      }
    }
    return results;
  }

  bool is_noisy;
  // Only one of these will be populated, as specified by is_noisy.
  Circuit<Gate> circuit;
  NoisyCircuit<Gate> ncircuit;

  StateSpace state_space;
  State state;
  State scratch;

  unsigned num_qubits;
  unsigned num_threads;
  unsigned noisy_reps;
  unsigned max_fused_size;
  unsigned verbosity;
  unsigned seed;

  // Only set to "true" once initialization is complete.
  bool is_valid;
};

// Methods for simulating full state vectors.

py::array_t<float> qsim_simulate_fullstate(
    const py::dict &options, uint64_t input_state) {
  return SimulatorHelper::simulate_fullstate(options, false, input_state);
}

py::array_t<float> qsim_simulate_fullstate(
    const py::dict &options, const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_fullstate(options, false, input_vector);
}

py::array_t<float> qtrajectory_simulate_fullstate(
    const py::dict &options, uint64_t input_state) {
  return SimulatorHelper::simulate_fullstate(options, true, input_state);
}

py::array_t<float> qtrajectory_simulate_fullstate(
    const py::dict &options, const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_fullstate(options, true, input_vector);
}

// Methods for calculating expectation values.

std::vector<std::complex<double>> qsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<OpString<Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    uint64_t input_state) {
  return SimulatorHelper::simulate_expectation_values(
    options, opsums_and_qubit_counts, false, input_state);
}

std::vector<std::complex<double>> qsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<OpString<Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_expectation_values(
    options, opsums_and_qubit_counts, false, input_vector);
}

std::vector<std::complex<double>> qtrajectory_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<OpString<Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    uint64_t input_state) {
  return SimulatorHelper::simulate_expectation_values(
    options, opsums_and_qubit_counts, true, input_state);
}

std::vector<std::complex<double>> qtrajectory_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<OpString<Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector) {
  return SimulatorHelper::simulate_expectation_values(
    options, opsums_and_qubit_counts, true, input_vector);
}

// Methods for sampling.

std::vector<unsigned> qsim_sample(const py::dict &options) {
  Circuit<Cirq::GateCirq<float>> circuit;
  try {
    circuit = getCircuit(options);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  using Simulator = qsim::Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using MeasurementResult = StateSpace::MeasurementResult;
  using Runner = QSimRunner<IO, MultiQubitGateFuser<IO, Cirq::GateCirq<float>>,
                            Simulator>;

  Runner::Parameter param;
  try {
    param.num_threads = parseOptions<unsigned>(options, "t\0");
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    param.seed = parseOptions<unsigned>(options, "s\0");
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  std::vector<MeasurementResult> results;
  StateSpace state_space(param.num_threads);
  State state = state_space.Create(circuit.num_qubits);
  state_space.SetStateZero(state);

  if (!Runner::Run(param, circuit, state, results)) {
    IO::errorf("qsim sampling of the circuit errored out.\n");
    return {};
  }

  std::vector<unsigned> result_bits;
  for (const auto& result : results) {
    result_bits.insert(result_bits.end(), result.bitstring.begin(),
                       result.bitstring.end());
  }
  return result_bits;
}

std::vector<unsigned> qtrajectory_sample(const py::dict &options) {
  NoisyCircuit<Cirq::GateCirq<float>> ncircuit;
  try {
    ncircuit = getNoisyCircuit(options);
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  using Simulator = qsim::Simulator<For>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = qsim::QuantumTrajectorySimulator<IO, Cirq::GateCirq<float>,
                                                  MultiQubitGateFuser,
                                                  Simulator>;

  Runner::Parameter param;
  unsigned num_threads;
  uint64_t seed;

  try {
    num_threads = parseOptions<unsigned>(options, "t\0");
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
    param.verbosity = parseOptions<unsigned>(options, "v\0");
    seed = parseOptions<unsigned>(options, "s\0");
    param.collect_mea_stat = true;
  } catch (const std::invalid_argument &exp) {
    IO::errorf(exp.what());
    return {};
  }

  Simulator simulator(num_threads);
  StateSpace state_space(num_threads);

  std::vector<std::vector<unsigned>> results;

  auto measure = [&results, &ncircuit, &state_space](
                  unsigned k, const State &state,
                  std::vector<uint64_t>& stat) {
    // Converts stat (which matches the MeasurementResult 'bits' field) into
    // bitstrings matching the MeasurementResult 'bitstring' field.
    unsigned idx = 0;
    for (const auto& channel : ncircuit.channels) {
      if (channel[0].kind != gate::kMeasurement)
        continue;
      for (const auto &op : channel[0].ops) {
        std::vector<unsigned> bitstring;
        uint64_t val = stat[idx];
        for (const auto &q : op.qubits) {
          bitstring.push_back((val >> q) & 1);
        }
        results.push_back(bitstring);

        idx += 1;
        if (idx >= stat.size())
          return;
      }
    }
  };

  if (!Runner::RunBatch(param, ncircuit, seed, seed + 1,
                        state_space, simulator, measure)) {
    IO::errorf("qtrajectory sampling of the circuit errored out.\n");
    return {};
  }

  std::vector<unsigned> result_bits;
  for (const auto& bitstring : results) {
    result_bits.insert(result_bits.end(), bitstring.begin(), bitstring.end());
  }
  return result_bits;
}

// Method for running the hybrid simulator.

std::vector<std::complex<float>> qsimh_simulate(const py::dict &options) {
  using Simulator = qsim::Simulator<For>;
  using HybridSimulator = HybridSimulator<IO, Cirq::GateCirq<float>,
                                          MultiQubitGateFuser, Simulator, For>;
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
    param.max_fused_size = parseOptions<unsigned>(options, "f\0");
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
