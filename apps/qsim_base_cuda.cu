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

#include <algorithm>
#include <complex>
#include <limits>
#include <string>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_qsim.h"
#include "../lib/io_file.h"
#include "../lib/run_qsim.h"
#include "../lib/simulator_cuda.h"

struct Options {
  std::string circuit_file;
  unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned seed = 1;
  unsigned max_fused_size = 2;
  unsigned num_threads = 256;
  unsigned num_dblocks = 16;
  unsigned verbosity = 0;
};

Options GetOptions(int argc, char* argv[]) {
  constexpr char usage[] = "usage:\n  ./qsim_base -c circuit -d maxtime "
                           "-s seed -f max_fused_size -t num_threads"
                           "-n num_dblocks -v verbosity\n";

  Options opt;

  int k;

  while ((k = getopt(argc, argv, "c:d:s:f:t:n:v:")) != -1) {
    switch (k) {
      case 'c':
        opt.circuit_file = optarg;
        break;
      case 'd':
        opt.maxtime = std::atoi(optarg);
        break;
      case 's':
        opt.seed = std::atoi(optarg);
        break;
      case 'f':
        opt.max_fused_size = std::atoi(optarg);
        break;
      case 't':
        opt.num_threads = std::atoi(optarg);
        break;
      case 'n':
        opt.num_dblocks = std::atoi(optarg);
        break;
      case 'v':
        opt.verbosity = std::atoi(optarg);
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
    return false;
  }

  return true;
}

template <typename StateSpace, typename State>
void PrintAmplitudes(
    unsigned num_qubits, const StateSpace& state_space, const State& state) {
  static constexpr char const* bits[8] = {
    "000", "001", "010", "011", "100", "101", "110", "111",
  };

  uint64_t size = std::min(uint64_t{8}, uint64_t{1} << num_qubits);
  unsigned s = 3 - std::min(unsigned{3}, num_qubits);

  for (uint64_t i = 0; i < size; ++i) {
    auto a = state_space.GetAmpl(state, i);
    qsim::IO::messagef("%s:%16.8g%16.8g%16.8g\n",
                       bits[i] + s, std::real(a), std::imag(a), std::norm(a));
  }
}

int main(int argc, char* argv[]) {
  using namespace qsim;

  auto opt = GetOptions(argc, argv);
  if (!ValidateOptions(opt)) {
    return 1;
  }

  Circuit<GateQSim<float>> circuit;
  if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  struct Factory {
    using Simulator = qsim::SimulatorCUDA<float>;
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
  using Fuser = MultiQubitGateFuser<IO, GateQSim<float>>;
  using Runner = QSimRunner<IO, Fuser, Factory>;

  StateSpace::Parameter param1;
  param1.num_threads = opt.num_threads;
  param1.num_dblocks = opt.num_dblocks;

  Simulator::Parameter param2;
  param2.num_threads = opt.num_threads;

  Factory factory(param1, param2);

  StateSpace state_space = factory.CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  if (state_space.IsNull(state)) {
    IO::errorf("not enough memory: is the number of qubits too large?\n");
    return 1;
  }

  state_space.SetStateZero(state);

  Runner::Parameter param3;
  param3.max_fused_size = opt.max_fused_size;
  param3.seed = opt.seed;
  param3.verbosity = opt.verbosity;

  if (Runner::Run(param3, factory, circuit, state)) {
    PrintAmplitudes(circuit.num_qubits, state_space, state);
  }

  return 0;
}
