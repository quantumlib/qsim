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

#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../lib/bitstring.h"
#include "../lib/circuit_qsim_parser.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_qsim.h"
#include "../lib/io_file.h"
#include "../lib/run_qsim.h"
#include "../lib/simmux.h"
#include "../lib/util.h"
#include "../lib/util_cpu.h"

constexpr char usage[] = "usage:\n  ./qsim_amplitudes -c circuit_file "
                         "-d times_to_save_results -i input_files "
                         "-o output_files -s seed -t num_threads "
                         "-f max_fused_size -v verbosity -z\n";

struct Options {
  std::string circuit_file;
  std::vector<unsigned> times = {std::numeric_limits<unsigned>::max()};
  std::vector<std::string> input_files;
  std::vector<std::string> output_files;
  unsigned seed = 1;
  unsigned num_threads = 1;
  unsigned max_fused_size = 2;
  unsigned verbosity = 0;
  bool denormals_are_zeros = false;
};

Options GetOptions(int argc, char* argv[]) {
  Options opt;

  int k;

  auto to_int = [](const std::string& word) -> unsigned {
    return std::atoi(word.c_str());
  };

  while ((k = getopt(argc, argv, "c:d:i:s:o:t:f:v:z")) != -1) {
    switch (k) {
      case 'c':
        opt.circuit_file = optarg;
        break;
      case 'd':
        qsim::SplitString(optarg, ',', to_int, opt.times);
        break;
      case 'i':
        qsim::SplitString(optarg, ',', opt.input_files);
        break;
      case 'o':
        qsim::SplitString(optarg, ',', opt.output_files);
        break;
      case 's':
        opt.seed = std::atoi(optarg);
        break;
      case 't':
        opt.num_threads = std::atoi(optarg);
        break;
      case 'f':
        opt.max_fused_size = std::atoi(optarg);
        break;
      case 'v':
        opt.verbosity = std::atoi(optarg);
        break;
      case 'z':
        opt.denormals_are_zeros = true;
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

  if (opt.input_files.empty()) {
    qsim::IO::errorf("input files are not provided.\n");
    qsim::IO::errorf(usage);
    return false;
  }

  if (opt.output_files.empty()) {
    qsim::IO::errorf("output files are not provided.\n");
    qsim::IO::errorf(usage);
    return false;
  }

  if (opt.times.size() != opt.input_files.size()
      || opt.times.size() != opt.output_files.size()) {
    qsim::IO::errorf("the number of times is not the same as the number of "
                     "input or output files.\n");
    return false;
  }

  for (std::size_t i = 1; i < opt.times.size(); i++) {
    if (opt.times[i - 1] == opt.times[i]) {
      qsim::IO::errorf("duplicate times to save results.\n");
      return false;
    } else if (opt.times[i - 1] > opt.times[i]) {
      qsim::IO::errorf("times to save results are not sorted.\n");
      return false;
    }
  }

  return true;
}

template <typename StateSpace, typename State, typename Bitstring>
bool WriteAmplitudes(const std::string& file,
                    StateSpace& state_space, const State& state,
                    const std::vector<Bitstring>& bitstrings) {
  std::stringstream ss;

  const unsigned width = 2 * sizeof(float) + 1;
  ss << std::setprecision(width);
  for (const auto& bitstring : bitstrings) {
    auto a = state_space.GetAmpl(state, bitstring);
    ss << std::setw(width + 8) << std::real(a)
       << std::setw(width + 8) << std::imag(a) << "\n";
  }

  return qsim::IOFile::WriteToFile(file, ss.str());
}

int main(int argc, char* argv[]) {
  using namespace qsim;

  auto opt = GetOptions(argc, argv);
  if (!ValidateOptions(opt)) {
    return 1;
  }

  Circuit<GateQSim<float>> circuit;
  unsigned maxtime = opt.times.back();
  if (!CircuitQsimParser<IOFile>::FromFile(maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  if (opt.denormals_are_zeros) {
    SetFlushToZeroAndDenormalsAreZeros();
  }

  struct Factory {
    Factory(unsigned num_threads) : num_threads(num_threads) {}

    using Simulator = qsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;

    StateSpace CreateStateSpace() const {
      return StateSpace(num_threads);
    }

    Simulator CreateSimulator() const {
      return Simulator(num_threads);
    }

    unsigned num_threads;
  };

  using Simulator = Factory::Simulator;
  using StateSpace = Simulator::StateSpace;
  using State = StateSpace::State;
  using Fuser = MultiQubitGateFuser<IO, GateQSim<float>>;
  using Runner = QSimRunner<IO, Fuser, Factory>;

  auto measure = [&opt, &circuit](
      unsigned k, const StateSpace& state_space, const State& state) {
    std::vector<Bitstring> bitstrings;
    BitstringsFromFile<IOFile>(
        circuit.num_qubits, opt.input_files[k], bitstrings);
    if (bitstrings.size() > 0) {
      WriteAmplitudes(opt.output_files[k], state_space, state, bitstrings);
    }
  };

  Runner::Parameter param;
  param.max_fused_size = opt.max_fused_size;
  param.seed = opt.seed;
  param.verbosity = opt.verbosity;
  Runner::Run(param, Factory(opt.num_threads), opt.times, circuit, measure);

  IO::messagef("all done.\n");

  return 0;
}
