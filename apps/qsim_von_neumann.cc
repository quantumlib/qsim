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
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <string>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser_basic.h"
#include "../lib/gates_qsim.h"
#include "../lib/io_file.h"
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
  constexpr char usage[] = "usage:\n  ./qsim_von_neumann -c circuit -d maxtime "
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

  using Simulator = qsim::Simulator<ParallelFor>;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;

  auto measure = [&opt, &circuit](
      unsigned k, const StateSpace& state_space, const State& state) {
    using Op = std::plus<double>;

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const StateSpace& state_space, const State& state) -> double {
      auto p = std::norm(state_space.GetAmpl(state, i));
      return p != 0 ? p * std::log(p) : 0;
    };

    auto size = state_space.Size(state);
    double entropy = -ParallelFor::RunReduce(opt.num_threads, size, f, Op(),
                                             state_space, state);
    IO::messagef("entropy=%g\n", entropy);
  };

  using Runner = QSimRunner<IO, BasicGateFuser<GateQSim<float>>, Simulator>;

  Runner::Parameter param;
  param.num_threads = opt.num_threads;
  param.verbosity = opt.verbosity;

  Runner::Run(param, opt.maxtime, circuit, measure);

  return 0;
}
