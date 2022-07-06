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
#include "../lib/fuser_basic.h"
#include "../lib/gates_qsim.h"
#include "../lib/io_file.h"
#include "../lib/run_qsimh.h"
#include "../lib/simmux.h"
#include "../lib/util.h"
#include "../lib/util_cpu.h"

constexpr char usage[] = "usage:\n  ./qsimh_amplitudes -c circuit_file "
                         "-d maxtime -k part1_qubits "
                         "-w prefix -p num_prefix_gates -r num_root_gates "
                         "-i input_file -o output_file -t num_threads "
                         "-v verbosity -z\n";

struct Options {
  std::string circuit_file;
  std::string input_file;
  std::string output_file;
  std::vector<unsigned> part1;
  uint64_t prefix;
  unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned num_prefix_gatexs = 0;
  unsigned num_root_gatexs = 0;
  unsigned num_threads = 1;
  unsigned verbosity = 0;
  bool denormals_are_zeros = false;
};

Options GetOptions(int argc, char* argv[]) {
  Options opt;

  int k;

  auto to_int = [](const std::string& word) -> unsigned {
    return std::atoi(word.c_str());
  };

  while ((k = getopt(argc, argv, "c:d:k:w:p:r:i:o:t:v:z")) != -1) {
    switch (k) {
      case 'c':
        opt.circuit_file = optarg;
        break;
      case 'd':
        opt.maxtime = std::atoi(optarg);
        break;
      case 'k':
        qsim::SplitString(optarg, ',', to_int, opt.part1);
        break;
      case 'w':
        opt.prefix = std::atol(optarg);
        break;
      case 'p':
        opt.num_prefix_gatexs = std::atoi(optarg);
        break;
      case 'r':
        opt.num_root_gatexs = std::atoi(optarg);
        break;
      case 'i':
        opt.input_file = optarg;
        break;
      case 'o':
        opt.output_file = optarg;
        break;
      case 't':
        opt.num_threads = std::atoi(optarg);
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

  if (opt.input_file.empty()) {
    qsim::IO::errorf("input file is not provided.\n");
    qsim::IO::errorf(usage);
    return false;
  }

  if (opt.output_file.empty()) {
    qsim::IO::errorf("output file is not provided.\n");
    qsim::IO::errorf(usage);
    return false;
  }

  return true;
}

bool ValidatePart1(unsigned num_qubits, const std::vector<unsigned>& part1) {
  for (std::size_t i = 0; i < part1.size(); ++i) {
    if (part1[i] >= num_qubits) {
      qsim::IO::errorf("part 1 qubit indices are too large.\n");
      return false;
    }
  }

  return true;
}

std::vector<unsigned> GetParts(
    unsigned num_qubits, const std::vector<unsigned>& part1) {
  std::vector<unsigned> parts(num_qubits, 0);

  for (std::size_t i = 0; i < part1.size(); ++i) {
    parts[part1[i]] = 1;
  }

  return parts;
}

template <typename Bitstring, typename Ctype>
bool WriteAmplitudes(const std::string& file,
                    const std::vector<Bitstring>& bitstrings,
                    const std::vector<Ctype>& results) {
  std::stringstream ss;

  const unsigned width = 2 * sizeof(float) + 1;
  ss << std::setprecision(width);
  for (size_t i = 0; i < bitstrings.size(); ++i) {
    const auto& a = results[i];
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
  if (!CircuitQsimParser<IOFile>::FromFile(opt.maxtime, opt.circuit_file,
                                           circuit)) {
    return 1;
  }

  if (!ValidatePart1(circuit.num_qubits, opt.part1)) {
    return 1;
  }
  auto parts = GetParts(circuit.num_qubits, opt.part1);

  if (opt.denormals_are_zeros) {
    SetFlushToZeroAndDenormalsAreZeros();
  }

  std::vector<Bitstring> bitstrings;
  auto num_qubits = circuit.num_qubits;
  if (!BitstringsFromFile<IOFile>(num_qubits, opt.input_file, bitstrings)) {
    return 1;
  }

  struct Factory {
    Factory(unsigned num_threads) : num_threads(num_threads) {}

    using Simulator = qsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;
    using fp_type = Simulator::fp_type;

    StateSpace CreateStateSpace() const {
      return StateSpace(num_threads);
    }

    Simulator CreateSimulator() const {
      return Simulator(num_threads);
    }

    unsigned num_threads;
  };

  using HybridSimulator = HybridSimulator<IO, GateQSim<float>, BasicGateFuser,
                                          For>;
  using Runner = QSimHRunner<IO, HybridSimulator>;

  Runner::Parameter param;
  param.prefix = opt.prefix;
  param.num_prefix_gatexs = opt.num_prefix_gatexs;
  param.num_root_gatexs = opt.num_root_gatexs;
  param.num_threads = opt.num_threads;
  param.verbosity = opt.verbosity;

  std::vector<std::complex<Factory::fp_type>> results(bitstrings.size(), 0);

  Factory factory(opt.num_threads);

  if (Runner::Run(param, factory, circuit, parts, bitstrings, results)) {
    WriteAmplitudes(opt.output_file, bitstrings, results);
    IO::messagef("all done.\n");
  }

  return 0;
}
