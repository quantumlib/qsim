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

#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <map>
#include <string>
#include <utility>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/classical_control_error.h"
#include "../lib/classical_control_symbol.h"
#include "../lib/classical_control_symtab.h"
#include "../lib/classical_control_util.h"
#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/io.h"
#include "../lib/run_qsim.h"
#include "../lib/simmux.h"
#include "../lib/util.h"
#include "../lib/util_cpu.h"

struct ParallelFor {
  explicit ParallelFor(unsigned num_threads) : num_threads(num_threads) {}

  uint64_t GetIndex0(uint64_t size, unsigned thread_id) const {
    return size * thread_id / num_threads;
  }

  uint64_t GetIndex1(uint64_t size, unsigned thread_id) const {
    return size * (thread_id + 1) / num_threads;
  }

  template <typename Function, typename... Args>
  void Run(uint64_t size, Function&& func, Args&&... args) const {
    if (num_threads == 1) {
      func(1, 0, 0, size, args...);
    } else {
      std::mutex mutex;
      std::exception_ptr exception = nullptr;

      #pragma omp parallel num_threads(num_threads)
      {
        unsigned n = omp_get_num_threads();
        unsigned m = omp_get_thread_num();

        uint64_t i0 = GetIndex0(size, m);
        uint64_t i1 = GetIndex1(size, m);

        try {
          func(n, m, i0, i1, args...);
        } catch (...) {
          std::lock_guard<std::mutex> lock(mutex);
          if (!exception) {
            exception = std::current_exception();
          }
        }
      }

      if (exception) {
        std::rethrow_exception(exception);
      }
    }
  }

  unsigned num_threads;
};

constexpr char usage[] = "usage:\n  ./qsim -c circuit -s symbol_defs "
                         "-0 rep0 -1 rep1 -w num_workers "
                         "-t num_threads_per_worker -f max_fused_size "
                         "-v verbosity -z\n";

struct Options {
  std::string circuit_file;
  std::string symbol_defs;
  unsigned rep0 = 0;
  unsigned rep1 = 1;
  unsigned num_workers = 1;
  unsigned num_threads = 1;
  unsigned max_fused_size = 2;
  unsigned verbosity = 0;
  bool denormals_are_zeros = false;
};

Options GetOptions(int argc, char* argv[]) {
  Options opt;

  int k;

  while ((k = getopt(argc, argv, "c:s:0:1:w:t:f:v:z")) != -1) {
    switch (k) {
      case 'c':
        opt.circuit_file = optarg;
        break;
      case 's':
        opt.symbol_defs = optarg;
        break;
      case '0':
        opt.rep0 = std::atoi(optarg);
        break;
      case '1':
        opt.rep1 = std::atoi(optarg);
        break;
      case 'w':
        opt.num_workers = std::atoi(optarg);
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

  return true;
}

int main(int argc, char* argv[]) {
  using namespace qsim;
  using namespace qsim::cc;

  auto opt = GetOptions(argc, argv);
  if (!ValidateOptions(opt)) {
    return 1;
  }

  if (opt.denormals_are_zeros) {
    SetFlushToZeroAndDenormalsAreZeros();
  }

  omp_set_max_active_levels(2);

  try {
    double t0 = 0;

    if (opt.verbosity > 1) {
      t0 = GetTime();
    }

    uint64_t num_reps = opt.rep1 - opt.rep0;
    unsigned num_workers = std::min(uint64_t{opt.num_workers}, num_reps);

    SymTable symtab;
    symtab.EnterScope(symtab.AddScope());

    symtab.Insert("pi", Symbol::Float(M_PI));
    symtab.Insert("nw", Symbol::Int(num_workers));
    symtab.Insert("wid", {Symbol::Int(0), /*read_only=*/false});
    symtab.Insert("rid", {Symbol::Int(0), /*read_only=*/false});
    symtab.Insert("rep0", Symbol::Int(opt.rep0));
    symtab.Insert("rep1", Symbol::Int(opt.rep1));

    ParseSymbols<ParserError, RuntimeError>(opt.symbol_defs, symtab);

    auto cstr = ReadFile(opt.circuit_file);
    unsigned maxtime = std::numeric_limits<unsigned>::max();
    auto [circuit, obss] = CircuitQsimParser<float>::Run(cstr, maxtime, symtab);

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
    using Fuser = MultiQubitGateFuser<IO>;
    using Runner = qsim::QSimRunner<Fuser>;

    Factory factory(opt.num_threads);

    Runner::Parameter param;
    param.max_fused_size = opt.max_fused_size;
    param.verbosity = opt.verbosity;

    std::vector<decltype(obss)> obsss;
    obsss.resize(opt.num_workers);

    if (param.verbosity > 1) {
      double t1 = GetTime();
      IO::messagef("# initialization runtime is %g seconds.\n", t1 - t0);
    }

    auto f = [&opt, &circuit, &symtab, &obss, &factory, &param, &obsss](
        unsigned, unsigned m, uint64_t r0, uint64_t r1) {
      auto circuit_m = circuit;
      auto symtab_m = symtab;
      auto obss_m = obss;

      Simulator simulator = factory.CreateSimulator();
      StateSpace state_space = factory.CreateStateSpace();

      State state = state_space.Create(circuit.num_qubits);

      if (state_space.IsNull(state)) {
        Error::Throw("not enough memory: is the number of qubits too large?\n");
      }

      Symbol* sym = symtab_m.Lookup("wid");
      sym->Assign(Symbol::Int(m));

      r0 += opt.rep0;
      r1 += opt.rep0;

      for (uint64_t r = r0; r < r1; r++) {
        Symbol* sym = symtab_m.Lookup("rid");
        sym->Assign(Symbol::Int(r));

        state_space.SetStateZero(state);

        uint64_t seed = 2 * r + 1;

        bool rc = Runner::Run(param, circuit_m, state_space, simulator,
                              seed, state, symtab_m, obss_m);

        if (rc) {
          obss_m.Iterate([](auto, auto& obs) { obs.Update(); });
        } else {
          obss_m.Iterate([](auto, auto& obs) { obs.Discard(); });
        }
      }

      obsss[m] = std::move(obss_m);
    };

    if (opt.verbosity > 0) {
      t0 = GetTime();
    }

    ::ParallelFor{num_workers}.Run(opt.rep1 - opt.rep0, f);

    if (param.verbosity > 0) {
      double t1 = GetTime();
      IO::messagef("# simulation runtime is %g seconds.\n", t1 - t0);
    }

    if (param.verbosity > 1) {
      t0 = GetTime();
    }

    std::map<std::string_view, std::vector<uint64_t>> hists;

    auto h = [&hists](auto name, const auto& obs) {
      auto& hist = hists[name];

      if (hist.empty()) {
        hist.reserve(obs.total_count.size());

        for (auto val : obs.total_count) {
          hist.emplace_back(val);
        }
      } else {
        for (unsigned i = 0; i < obs.total_count.size(); ++i) {
          hist[i] += obs.total_count[i];
        }
      }
    };

    for (const auto& obss : obsss) {
      obss.Iterate(h);
    }

    if (param.verbosity > 1) {
      double t1 = GetTime();
      IO::messagef("# postprocessing runtime is %g seconds.\n", t1 - t0);
    }

    for (auto& [name, hist] : hists) {
      IO::messagef("%6.*s [", (int) name.size(), name.data());

      for (std::size_t i = 0; i < hist.size(); ++i) {
        if (i > 0) {
          IO::messagef(", ");
        }
        IO::messagef("%lu: %lu", i, hist[i]);
      }

      puts("]");
    };
  } catch (std::exception& e) {
    IO::errorf("%s\n", e.what());
  }

  return 0;
}
