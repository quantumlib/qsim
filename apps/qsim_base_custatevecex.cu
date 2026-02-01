// Copyright 2025 Google LLC. All Rights Reserved.
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
#include "../lib/gates_qsim.h"
#include "../lib/io_file.h"
#include "../lib/multiprocess_custatevecex.h"
#include "../lib/run_custatevecex.h"
#include "../lib/simulator_custatevecex.h"
#include "../lib/util_custatevec.h"

struct Options {
  std::string circuit_file;
  unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned seed = 1;
  unsigned lbuf = 30;
  unsigned verbosity = 0;
};

Options GetOptions(int argc, char* argv[]) {
  constexpr char usage[] = "usage:\n  ./qsim_base_custatevecex.x -c circuit "
                           "-d maxtime -s seed  -l lbuf -v verbosity\n";

  Options opt;

  int k;

  while ((k = getopt(argc, argv, "c:d:s:l:v:")) != -1) {
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
      case 'l':
        opt.lbuf = std::atoi(optarg);
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

  using fp_type = float;

  Circuit<GateQSim<fp_type>> circuit;
  if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  struct Factory {
    using Simulator = qsim::SimulatorCuStateVecEx<fp_type>;
    using StateSpace = Simulator::StateSpace;

    explicit Factory(uint64_t transfer_buffer_size, unsigned verbosity = 0)
        : verbosity(verbosity) {
      MultiProcessCuStateVecEx::Parameter param = {transfer_buffer_size};
      mp.Initialize(param);

      if (verbosity > 2 && mp.Initialized()) {
        qsim::IO::messagef("# transfer_buf_size=%lu\n", transfer_buffer_size);
      }
    }

    StateSpace CreateStateSpace() const {
      StateSpace::Parameter param;
      param.verbosity = verbosity;

      return StateSpace{mp, param};
    }

    Simulator CreateSimulator() const {
      return Simulator{};
    }

    MultiProcessCuStateVecEx mp;
    unsigned verbosity;
  };

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = CuStateVecExRunner<IO, Factory>;

  Factory factory(uint64_t{1} << opt.lbuf, opt.verbosity);

  StateSpace state_space = factory.CreateStateSpace();
  State state = state_space.Create(circuit.num_qubits);

  if (state_space.IsNull(state)) {
    IO::errorf("not enough memory: is the number of qubits too large?\n");
    return 1;
  }

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.seed = opt.seed;
  param.verbosity = opt.verbosity;

  if (Runner::Run(param, factory, circuit, state)) {
    PrintAmplitudes(circuit.num_qubits, state_space, state);
  }

  return 0;
}
