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

// Benchmark app for the proposal-faithful gate-batch runner.
// -f is the per-batch max_fused_size (at least 2; the proposal suggests
// 2-3).

#include <unistd.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/cpu_thread_topology.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_qsim.h"
#include "../lib/io_file.h"
#include "../lib/operation.h"
#include "../lib/run_qsim_gate_batch.h"
#include "../lib/simmux.h"
#include "../lib/util_cpu.h"

struct Options {
  std::string circuit_file;
  unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned seed = 1;
  unsigned num_threads = 1;
  unsigned inner_threads = 1;
  unsigned max_fused_size = 3;
  unsigned block_qubits = 19;
  unsigned min_eviction_floor = 5;
  unsigned max_gate_seeds = 64;
  bool commute_diagonal_gates = false;
  unsigned verbosity = 0;
};

Options GetOptions(int argc, char* argv[]) {
  constexpr char usage[] = "usage:\n  ./qsim_gate_batch -c circuit "
                           "-d maxtime -s seed -t threads "
                           "-f max_fused_size -l block_qubits "
                           "-e min_eviction_floor -x commute_diagonal_gates "
                           "-g max_gate_seeds "
                           "-i inner_threads -v verbosity\n";
  Options opt;
  int k;
  while ((k = getopt(argc, argv, "c:d:s:t:f:l:e:x:g:i:v:")) != -1) {
    switch (k) {
      case 'c': opt.circuit_file = optarg; break;
      case 'd': opt.maxtime = std::atoi(optarg); break;
      case 's': opt.seed = std::atoi(optarg); break;
      case 't': opt.num_threads = std::atoi(optarg); break;
      case 'f': opt.max_fused_size = std::atoi(optarg); break;
      case 'l': opt.block_qubits = std::atoi(optarg); break;
      case 'e': opt.min_eviction_floor = std::atoi(optarg); break;
      case 'x': opt.commute_diagonal_gates = std::atoi(optarg) != 0; break;
      case 'g': opt.max_gate_seeds = std::atoi(optarg); break;
      case 'i': opt.inner_threads = std::atoi(optarg); break;
      case 'v': opt.verbosity = std::atoi(optarg); break;
      default: qsim::IO::errorf(usage); exit(1);
    }
  }
  if (opt.circuit_file.empty()) {
    qsim::IO::errorf(usage);
    exit(1);
  }
  return opt;
}

template <typename StateSpace, typename QubitMappedState>
void PrintAmplitudes(unsigned num_qubits, const StateSpace& state_space,
                     const QubitMappedState& state) {
  static constexpr char const* bits[8] = {
    "000", "001", "010", "011", "100", "101", "110", "111",
  };
  uint64_t size = std::min(uint64_t{8}, uint64_t{1} << num_qubits);
  unsigned s = 3 - std::min(unsigned{3}, num_qubits);
  for (uint64_t i = 0; i < size; ++i) {
    auto a = state.GetAmpl(state_space, i);
    qsim::IO::messagef("%s:%16.8g%16.8g%16.8g\n",
                       bits[i] + s, std::real(a), std::imag(a), std::norm(a));
  }
}

int main(int argc, char* argv[]) {
  using namespace qsim;

  auto opt = GetOptions(argc, argv);
  std::vector<unsigned> team_thread_cpus;
  if (opt.inner_threads > 1) {
    std::string topology_error;
    const auto topology = CpuThreadTopology::Discover();
    if (!topology.BuildTeamCpuOrder(opt.num_threads, opt.inner_threads,
                                    team_thread_cpus, topology_error)) {
      IO::errorf("cannot configure SMT teams: %s.\n",
                 topology_error.c_str());
      return 1;
    }
  }

  Circuit<Operation<float>> circuit;
  if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  struct Factory {
    Factory(unsigned num_threads) : num_threads(num_threads) {}
    using Simulator = qsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;
    StateSpace CreateStateSpace() const { return StateSpace(num_threads); }
    Simulator CreateSimulator() const { return Simulator(num_threads); }
    unsigned num_threads;
  };

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Fuser = MultiQubitGateFuser<IO>;
  using Runner = QSimGateBatchRunner<IO, Fuser, Factory>;

  StateSpace state_space = Factory(opt.num_threads).CreateStateSpace();
  QubitMappedState<State> state(state_space.Create(circuit.num_qubits));
  if (state_space.IsNull(state.state)) {
    IO::errorf("not enough memory: is the number of qubits too large?\n");
    return 1;
  }
  state_space.SetStateZero(state.state);

  Runner::Parameter param;
  param.max_fused_size = opt.max_fused_size;
  param.block_qubits = opt.block_qubits;
  param.min_eviction_floor = opt.min_eviction_floor;
  param.place_hot_qubits = true;
  param.max_gate_seeds = opt.max_gate_seeds;
  param.commute_diagonal_gates = opt.commute_diagonal_gates;
  param.num_threads = opt.num_threads;
  param.inner_threads = opt.inner_threads;
  param.team_thread_cpus = std::move(team_thread_cpus);
  param.verbosity = opt.verbosity;

  if (Runner::Run(param, Factory(opt.num_threads), circuit, state)) {
    PrintAmplitudes(circuit.num_qubits, state_space, state);
  }

  return 0;
}
