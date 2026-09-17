#ifndef QSIM_CPU_TOPOLOGY_H_
#define QSIM_CPU_TOPOLOGY_H_

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

namespace qsim {

struct CpuTopology {
  struct Core { unsigned id = 0; std::vector<unsigned> siblings; };
  struct CacheGroup { unsigned level = 0; unsigned id = 0; std::vector<unsigned> cpus; uint64_t bytes = 0; };
  struct Node { unsigned id = 0; std::vector<Core> cores; std::vector<CacheGroup> caches; };
  std::vector<Node> nodes;
  std::vector<unsigned> affinity;
  unsigned logical_cpus = 0;
  unsigned physical_cores = 0;
  unsigned packages = 0;
  unsigned numa_nodes = 0;
  bool smt = false;
  std::string process_status;

  static std::vector<unsigned> ParseList(const std::string& text) {
    std::vector<unsigned> result;
    std::stringstream ss(text); std::string part;
    while (std::getline(ss, part, ',')) {
      auto dash = part.find('-');
      unsigned first = std::stoul(part.substr(0, dash));
      unsigned last = dash == std::string::npos ? first : std::stoul(part.substr(dash + 1));
      for (unsigned i = first; i <= last; ++i) result.push_back(i);
    }
    return result;
  }

  static std::string Read(const std::string& path) {
    std::ifstream in(path); std::string value; std::getline(in, value); return value;
  }
  static std::string ReadAll(const std::string& path) {
    std::ifstream in(path); std::ostringstream value; value << in.rdbuf(); return value.str();
  }
  static uint64_t ParseSize(const std::string& text) {
    if (text.empty()) return 0;
    double value = std::stod(text);
    if (text.back() == 'K' || text.back() == 'k') value *= 1024;
    if (text.back() == 'M' || text.back() == 'm') value *= 1024 * 1024;
    if (text.back() == 'G' || text.back() == 'g') value *= 1024 * 1024 * 1024;
    return static_cast<uint64_t>(value);
  }

  static CpuTopology Detect() {
    CpuTopology t;
#if defined(__linux__)
    t.process_status = ReadAll("/proc/self/status");
    std::set<unsigned> packages, cores;
    for (unsigned cpu = 0; cpu < 4096; ++cpu) {
      std::string root = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
      std::string online = Read(root + "/online");
      if (cpu && online.empty() && access(root.c_str(), F_OK) != 0) break;
      if (cpu && online == "0") continue;
      ++t.logical_cpus;
      unsigned pkg = 0, core = cpu;
      try { pkg = std::stoul(Read(root + "/topology/physical_package_id")); } catch (...) {}
      try { core = std::stoul(Read(root + "/topology/core_id")); } catch (...) {}
      packages.insert(pkg); cores.insert(pkg * 100000 + core);
      auto siblings = ParseList(Read(root + "/topology/thread_siblings_list"));
      unsigned node_id = 0;
      for (unsigned n = 0; n < 256; ++n) {
        auto cpulist = Read("/sys/devices/system/node/node" + std::to_string(n) + "/cpulist");
        if (cpulist.empty()) break;
        auto node_cpus = ParseList(cpulist);
        if (std::find(node_cpus.begin(), node_cpus.end(), cpu) != node_cpus.end()) { node_id = n; break; }
      }
      if (t.nodes.size() <= node_id) t.nodes.resize(node_id + 1);
      t.nodes[node_id].id = node_id;
      bool found = false;
      for (auto& c : t.nodes[node_id].cores) if (c.id == core) found = true;
      if (!found) t.nodes[node_id].cores.push_back({core, siblings});
      for (unsigned cache = 0; cache < 8; ++cache) {
        std::string cache_root = root + "/cache/index" + std::to_string(cache);
        auto level_text = Read(cache_root + "/level");
        auto shared_text = Read(cache_root + "/shared_cpu_list");
        if (level_text.empty() || shared_text.empty()) break;
        unsigned level = std::stoul(level_text);
        unsigned cache_id = cache;
        try { cache_id = std::stoul(Read(cache_root + "/id")); } catch (...) {}
        auto shared = ParseList(shared_text);
        auto& groups = t.nodes[node_id].caches;
        auto it = std::find_if(groups.begin(), groups.end(), [&](const CacheGroup& group) {
          return group.level == level && group.id == cache_id;
        });
        uint64_t bytes = 0;
        try { bytes = ParseSize(Read(cache_root + "/size")); } catch (...) {}
        if (it == groups.end()) groups.push_back({level, cache_id, shared, bytes});
      }
    }
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
      for (unsigned i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &set)) t.affinity.push_back(i);
    t.packages = packages.size(); t.physical_cores = cores.size();
    t.numa_nodes = t.nodes.size(); t.smt = t.logical_cpus > t.physical_cores;
#else
    t.logical_cpus = 1; t.physical_cores = 1; t.packages = 1; t.numa_nodes = 1;
    t.nodes.push_back({0, {{0, {0}}}, {}}); t.affinity.push_back(0);
#endif
    return t;
  }

  uint64_t TargetTileCacheBytes() const {
    uint64_t smallest_l2 = 0;
    for (const auto& node : nodes) for (const auto& cache : node.caches) {
      if (cache.level == 2 && cache.bytes != 0 &&
          (smallest_l2 == 0 || cache.bytes < smallest_l2)) smallest_l2 = cache.bytes;
    }
    if (smallest_l2 == 0) return 512 * 1024;
    return std::max<uint64_t>(256 * 1024, std::min<uint64_t>(2 * 1024 * 1024, smallest_l2));
  }
};

}  // namespace qsim
#endif
