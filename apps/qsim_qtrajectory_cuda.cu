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

#include <unistd.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

#include "../lib/channels_qsim.h"
#include "../lib/circuit_qsim_parser.h"
#include "../lib/expect.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_qsim.h"
#include "../lib/io_file.h"
#include "../lib/qtrajectory.h"
#include "../lib/simulator_cuda.h"

struct Options {
  std::string circuit_file;
  std::vector<unsigned> times = {std::numeric_limits<unsigned>::max()};
  double amplitude_damp_const = 0;
  double phase_damp_const = 0;
  unsigned traj0 = 0;
  unsigned num_trajectories = 10;
  unsigned max_fused_size = 2;
  unsigned verbosity = 0;
};

constexpr char usage[] = "usage:\n  ./qsim_qtrajectory_cuda.x "
                         "-c circuit_file -d times_to_calculate_observables "
                         "-a amplitude_damping_const -p phase_damping_const "
                         "-t traj0 -n num_trajectories -f max_fused_size "
                         "-v verbosity\n";

Options GetOptions(int argc, char* argv[]) {
  Options opt;

  int k;

  auto to_int = [](const std::string& word) -> unsigned {
    return std::atoi(word.c_str());
  };

  while ((k = getopt(argc, argv, "c:d:a:p:t:n:f:v:")) != -1) {
    switch (k) {
      case 'c':
        opt.circuit_file = optarg;
        break;
      case 'd':
        qsim::SplitString(optarg, ',', to_int, opt.times);
        break;
      case 'a':
        opt.amplitude_damp_const = std::atof(optarg);
        break;
      case 'p':
        opt.phase_damp_const = std::atof(optarg);
        break;
      case 't':
        opt.traj0 = std::atoi(optarg);
        break;
      case 'n':
        opt.num_trajectories = std::atoi(optarg);
        break;
      case 'f':
        opt.max_fused_size = std::atoi(optarg);
        break;
      case 'v':
        opt.verbosity = std::atoi(optarg);
        break;
        break;
      default:
        qsim::IO::errorf(usage);
        exit(1);
    }
  }

  return opt;
}

bool ValidateOptions(const Options& opt) {
  if (opt.circuit_file.empty()) {
    qsim::IO::errorf("circuit file is not provided.\n");
    qsim::IO::errorf(usage);
    return false;
  }

  if (opt.times.size() == 0) {
    qsim::IO::errorf("times to calculate observables are not provided.\n");
    return false;
  }

  for (std::size_t i = 1; i < opt.times.size(); i++) {
    if (opt.times[i - 1] == opt.times[i]) {
      qsim::IO::errorf("duplicate times to calculate observables.\n");
      return false;
    } else if (opt.times[i - 1] > opt.times[i]) {
      qsim::IO::errorf("times to calculate observables are not sorted.\n");
      return false;
    }
  }

  return true;
}

template <typename Gate, typename Channel1, typename Channel2>
std::vector<qsim::NoisyCircuit<Gate>> AddNoise(
    const qsim::Circuit<Gate>& circuit, const std::vector<unsigned>& times,
    const Channel1& channel1, const Channel2& channel2) {
  std::vector<qsim::NoisyCircuit<Gate>> ncircuits;
  ncircuits.reserve(times.size());

  qsim::NoisyCircuit<Gate> ncircuit;

  ncircuit.num_qubits = circuit.num_qubits;
  ncircuit.channels.reserve(5 * circuit.gates.size());

  unsigned cur_time_index = 0;

  for (std::size_t i = 0; i < circuit.gates.size(); ++i) {
    const auto& gate = circuit.gates[i];

    ncircuit.channels.push_back(qsim::MakeChannelFromGate(3 * gate.time, gate));

    for (auto q : gate.qubits) {
      ncircuit.channels.push_back(channel1.Create(3 * gate.time + 1, q));
    }

    for (auto q : gate.qubits) {
      ncircuit.channels.push_back(channel2.Create(3 * gate.time + 2, q));
    }

    unsigned t = times[cur_time_index];

    if (i == circuit.gates.size() - 1 || t < circuit.gates[i + 1].time) {
      ncircuits.push_back(std::move(ncircuit));

      ncircuit = {};

      if (i < circuit.gates.size() - 1) {
        if (circuit.gates[i + 1].time > times.back()) {
          break;
        }

        ncircuit.num_qubits = circuit.num_qubits;
        ncircuit.channels.reserve(5 * circuit.gates.size());
      }

      ++cur_time_index;
    }
  }

  return ncircuits;
}

template <typename Gate>
std::vector<std::vector<qsim::OpString<Gate>>> GetObservables(
    unsigned num_qubits) {
  std::vector<std::vector<qsim::OpString<Gate>>> observables;
  observables.reserve(num_qubits);

  using X = qsim::GateX<typename Gate::fp_type>;

  for (unsigned q = 0; q < num_qubits; ++q) {
    observables.push_back({{{1.0, 0.0}, {X::Create(0, q)}}});
  }

  return observables;
}

int main(int argc, char* argv[]) {
  using namespace qsim;

  using fp_type = float;

  struct Factory {
    using Simulator = qsim::SimulatorCUDA<fp_type>;
    using StateSpace = Simulator::StateSpace;

    Factory(const StateSpace::Parameter& param1,
            const Simulator::Parameter& param2)
        : param1(param1), param2(param2) {}

    StateSpace CreateStateSpace() const {
      return StateSpace(param1);
    }

    Simulator CreateSimulator() const {
      return Simulator(param2);
    }

    const StateSpace::Parameter& param1;
    const Simulator::Parameter& param2;
  };

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Fuser = MultiQubitGateFuser<IO, GateQSim<fp_type>>;
  using QTSimulator = QuantumTrajectorySimulator<IO, GateQSim<fp_type>,
                                                 MultiQubitGateFuser,
                                                 Simulator>;

  auto opt = GetOptions(argc, argv);
  if (!ValidateOptions(opt)) {
    return 1;
  }

  Circuit<GateQSim<fp_type>> circuit;
  unsigned maxtime = opt.times.back();
  if (!CircuitQsimParser<IOFile>::FromFile(maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  if (opt.times.size() == 1
      && opt.times[0] == std::numeric_limits<unsigned>::max()) {
    opt.times[0] = circuit.gates.back().time;
  }

  StateSpace::Parameter param1;
  Simulator::Parameter param2;
  Factory factory(param1, param2);

  Simulator simulator = factory.CreateSimulator();
  StateSpace state_space = factory.CreateStateSpace();

  State state = state_space.Create(circuit.num_qubits);

  if (state_space.IsNull(state)) {
    IO::errorf("not enough memory: is the number of qubits too large?\n");
    return 1;
  }

  typename QTSimulator::Parameter param3;
  param3.max_fused_size = opt.max_fused_size;
  param3.verbosity = opt.verbosity;
  param3.apply_last_deferred_ops = true;

  auto channel1 = AmplitudeDampingChannel<fp_type>(opt.amplitude_damp_const);
  auto channel2 = PhaseDampingChannel<fp_type>(opt.phase_damp_const);

  auto noisy_circuits = AddNoise(circuit, opt.times, channel1, channel2);

  auto observables = GetObservables<GateQSim<fp_type>>(circuit.num_qubits);

  std::vector<std::vector<std::vector<std::complex<double>>>> results;
  results.reserve(opt.num_trajectories);

  QTSimulator::Stat stat;

  using CleanResults = std::vector<std::vector<std::complex<double>>>;
  CleanResults primary_results(noisy_circuits.size());

  for (unsigned i = 0; i < opt.num_trajectories; ++i) {
    results.push_back({});
    results[i].reserve(noisy_circuits.size());

    state_space.SetStateZero(state);

    auto seed = noisy_circuits.size() * (i + opt.traj0);

    for (unsigned s = 0; s < noisy_circuits.size(); ++s) {
      if (!QTSimulator::RunOnce(param3, noisy_circuits[s], seed++,
                                state_space, simulator, state, stat)) {
        return 1;
      }

      results[i].push_back({});
      results[i][s].reserve(observables.size());

      primary_results[s].reserve(observables.size());

      if (stat.primary && !primary_results[s].empty()) {
        for (std::size_t k = 0; k < observables.size(); ++k) {
          results[i][s].push_back(primary_results[s][k]);
        }
      } else {
        for (const auto& obs : observables) {
          auto result = ExpectationValue<IO, Fuser>(obs, simulator, state);
          results[i][s].push_back(result);

          if (stat.primary) {
            primary_results[s].push_back(result);
            param3.apply_last_deferred_ops = false;
          }
        }
      }
    }
  }

  for (unsigned i = 1; i < opt.num_trajectories; ++i) {
    for (unsigned s = 0; s < noisy_circuits.size(); ++s) {
      for (unsigned k = 0; k < observables.size(); ++k) {
        results[0][s][k] += results[i][s][k];
      }
    }
  }

  double f = 1.0 / opt.num_trajectories;

  for (unsigned s = 0; s < noisy_circuits.size(); ++s) {
    for (unsigned k = 0; k < observables.size(); ++k) {
      results[0][s][k] *= f;
    }
  }

  for (unsigned s = 0; s < noisy_circuits.size(); ++s) {
    IO::messagef("#time=%u\n", opt.times[s]);

    for (unsigned k = 0; k < observables.size(); ++k) {
      IO::messagef("%4u %4u %17.9g %17.9g\n", s, k,
                   std::real(results[0][s][k]), std::imag(results[0][s][k]));
    }
  }

  return 0;
}
