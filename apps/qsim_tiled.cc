#include <getopt.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <complex>
#include <limits>
#include <string>

#include "../lib/circuit_qsim_parser.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/formux.h"
#include "../lib/io_file.h"
#include "../lib/run_qsim_tiled.h"
#include "../lib/simmux.h"
#include "../lib/util_cpu.h"

struct Options {
  std::string circuit; unsigned maxtime = std::numeric_limits<unsigned>::max();
  unsigned threads = 1, inner = 1, tile_qubits = 0, fused = 2, verbosity = 0;
  std::string swizzle = "linear"; bool numa = true;
};

Options Parse(int argc, char** argv) {
  Options o; int c; int option_index = 0;
  static const option long_options[] = {
      {"circuit", required_argument, nullptr, 'c'},
      {"max-time", required_argument, nullptr, 'd'},
      {"outer-threads", required_argument, nullptr, 't'},
      {"inner-threads", required_argument, nullptr, 'i'},
      {"tile-qubits", required_argument, nullptr, 'l'},
      {"max-fused-size", required_argument, nullptr, 'f'},
      {"swizzle", required_argument, nullptr, 's'},
      {"numa", required_argument, nullptr, 'n'},
      {"verbosity", required_argument, nullptr, 'v'},
      {nullptr, 0, nullptr, 0}};
  while ((c = getopt_long(argc, argv, "c:d:t:i:l:f:s:n:v:", long_options,
                          &option_index)) != -1) switch (c) {
    case 'c': o.circuit = optarg; break; case 'd': o.maxtime = std::atoi(optarg); break;
    case 't': o.threads = std::atoi(optarg); break; case 'i': o.inner = std::atoi(optarg); break;
    case 'l': o.tile_qubits = std::atoi(optarg); break; case 'f': o.fused = std::atoi(optarg); break;
    case 's': o.swizzle = optarg; break; case 'n': o.numa = std::string(optarg) != "off"; break;
    case 'v': o.verbosity = std::atoi(optarg); break; default: break;
  }
  return o;
}

int main(int argc, char** argv) {
  using namespace qsim;
  auto o = Parse(argc, argv);
  if (o.circuit.empty()) { IO::errorf("usage: qsim_tiled -c circuit [--outer-threads N] [--inner-threads N] [--tile-qubits N] [--max-fused-size N] [--swizzle MODE] [--numa on|off]\n"); return 1; }
  Circuit<Operation<float>> circuit;
  if (!CircuitQsimParser<IOFile>::FromFile(o.maxtime, o.circuit, circuit)) return 1;
  if (o.inner > 1 && o.verbosity) IO::messagef("cooperative inner threads: %u\n", o.inner);
  struct Factory {
    explicit Factory(unsigned n) : n(n) {} using Simulator = qsim::Simulator<For>;
    using StateSpace = Simulator::StateSpace;
    StateSpace CreateStateSpace() const { return StateSpace(n); }
    Simulator CreateSimulator() const { return Simulator(n); }
    unsigned n;
  } factory(o.threads);
  TiledOptions config; config.tile_qubits = o.tile_qubits; config.outer_threads = o.threads;
  config.inner_threads = o.inner; config.max_fused_size = o.fused;
  config.schedule = ParseTileSchedule(o.swizzle); config.numa = o.numa;
  TiledStats stats;
  using Fuser = MultiQubitGateFuser<IO>;
  bool tiled_ok = RunQSimTiledSimdRemapped<IO, Fuser>(config, factory, circuit, &stats);
  if (!tiled_ok) tiled_ok = RunQSimTiledNormal<IO, Fuser>(config, circuit, &stats);
  if (!tiled_ok && o.verbosity > 1) IO::messagef("normal tile path fallback: %s\n", stats.fallback_reason.c_str());
  if (!tiled_ok && !RunQSimTiled<IO, Fuser>(config, factory, circuit, &stats)) return 1;
  if (o.verbosity) IO::messagef("backend=%s tiles=%llu batches=%llu total=%g planning=%g remap=%g fusion=%g gates=%g scheduling=%g allocation=%g logical-cpus=%u physical-cores=%u numa-nodes=%u smt=%s inner-threads=%u\n",
      stats.backend.c_str(),
      static_cast<unsigned long long>(stats.tiles), static_cast<unsigned long long>(stats.batches),
      stats.total_seconds, stats.planning_seconds, stats.remap_seconds,
      stats.fusion_seconds, stats.gate_seconds, stats.scheduling_seconds,
      stats.allocation_seconds,
      stats.topology.logical_cpus, stats.topology.physical_cores, stats.topology.numa_nodes,
      stats.topology.smt ? "on" : "off", o.inner);
  if (o.verbosity > 1) {
    unsigned cache_groups = 0;
    for (const auto& node : stats.topology.nodes) cache_groups += node.caches.size();
    IO::messagef("cache-groups=%u affinity-cpus=%llu\n", cache_groups,
                 static_cast<unsigned long long>(stats.topology.affinity.size()));
  }
  if (o.verbosity > 1) {
    for (std::size_t i = 0; i < stats.first_amplitudes.size(); ++i)
      IO::messagef("%llu:%16.8g%16.8g%16.8g\n",
                   static_cast<unsigned long long>(i), stats.first_amplitudes[i].real(),
                   stats.first_amplitudes[i].imag(), std::norm(stats.first_amplitudes[i]));
    IO::messagef("norm=%16.8g\n", stats.norm);
  }
  return 0;
}
