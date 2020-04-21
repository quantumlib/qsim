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

#include "../lib/circuit_reader.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_def.h"
#include "../lib/io.h"
#include "../lib/parfor.h"
#include "../lib/run_qsim.h"
#include "../lib/simmux.h"

struct Options {
  std::string circuit_file;
  unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned num_threads = 1;
  unsigned verbosity = 0;
};

Options GetOptions(int argc, char* argv[]) {
  constexpr char usage[] = "usage:\n  ./qsim_base -c circuit -d maxtime "
                           "-t threads -v verbosity\n";

  Options opt;

  int k;

  while ((k = getopt(argc, argv, "c:d:t:v:")) != -1) {
    switch (k) {
      case 'c':
        opt.circuit_file = optarg;
        break;
      case 'd':
        opt.maxtime = std::atoi(optarg);
        break;
      case 't':
        opt.num_threads = std::atoi(optarg);
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

  uint64_t size = std::min(uint64_t{8}, state_space.Size(state));
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
  if (!CircuitReader<IO>::FromFile(opt.maxtime, opt.circuit_file, circuit)) {
    return 1;
  }

  using Simulator = qsim::Simulator<ParallelFor>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Runner = QSimRunner<IO, BasicGateFuser<GateQSim<float>>, Simulator>;

  StateSpace state_space(circuit.num_qubits, opt.num_threads);
  State state = state_space.CreateState();

  if (state_space.IsNull(state)) {
    IO::errorf("not enough memory: is the number of qubits too large?\n");
    return 1;
  }

  state_space.SetStateZero(state);

  Runner::Parameter param;
  param.num_threads = opt.num_threads;
  param.verbosity = opt.verbosity;

  if (Runner::Run(param, opt.maxtime, circuit, state)) {
    PrintAmplitudes(circuit.num_qubits, state_space, state);
  }

  return 0;
}
