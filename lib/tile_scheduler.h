#ifndef QSIM_TILE_SCHEDULER_H_
#define QSIM_TILE_SCHEDULER_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include "cpu_topology.h"
#if defined(__linux__)
#include <sched.h>
#endif

namespace qsim {
enum class TileSchedule { kLinear, kRoundRobin, kCacheGroup, kNumaGroup, kXorSwizzle, kDynamic };
inline TileSchedule ParseTileSchedule(const std::string& s) {
  if (s == "round_robin") return TileSchedule::kRoundRobin;
  if (s == "cache_group") return TileSchedule::kCacheGroup;
  if (s == "numa_group") return TileSchedule::kNumaGroup;
  if (s == "xor_swizzle") return TileSchedule::kXorSwizzle;
  if (s == "dynamic") return TileSchedule::kDynamic;
  return TileSchedule::kLinear;
}
class TileScheduler {
 public:
  TileScheduler(uint64_t tiles, unsigned workers, TileSchedule mode)
      : tiles_(tiles), workers_(std::max(1U, workers)), mode_(mode) {}
  unsigned WorkerFor(uint64_t tile) const {
    switch (mode_) {
      case TileSchedule::kXorSwizzle: return (tile ^ (tile >> 1)) % workers_;
      case TileSchedule::kCacheGroup:
      case TileSchedule::kNumaGroup:
      case TileSchedule::kRoundRobin: return tile % workers_;
      case TileSchedule::kDynamic: return tile % workers_;
      case TileSchedule::kLinear: return std::min<uint64_t>(workers_ - 1, tile * workers_ / std::max<uint64_t>(1, tiles_));
    }
    return 0;
  }
  std::vector<uint64_t> StaticOrder() const {
    std::vector<uint64_t> order(tiles_); for (uint64_t i = 0; i < tiles_; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [this](uint64_t a, uint64_t b) { return WorkerFor(a) < WorkerFor(b); });
    return order;
  }
  bool dynamic() const { return mode_ == TileSchedule::kDynamic; }

  // Select one logical CPU per worker, preferring one CPU from each cache or
  // NUMA locality group before filling the remaining worker slots.  The
  // caller may still pass an externally restricted affinity mask; CPUs not in
  // that mask are never returned.
  std::vector<unsigned> WorkerCpus(const CpuTopology& topology) const {
    std::vector<unsigned> allowed = topology.affinity;
    if (allowed.empty()) {
      for (unsigned cpu = 0; cpu < topology.logical_cpus; ++cpu)
        allowed.push_back(cpu);
    }
    std::vector<std::vector<unsigned>> groups;
    if (mode_ == TileSchedule::kCacheGroup) {
      for (const auto& node : topology.nodes) {
        for (const auto& cache : node.caches) {
          if (cache.level != 2) continue;
          std::vector<unsigned> group;
          for (auto cpu : cache.cpus)
            if (std::find(allowed.begin(), allowed.end(), cpu) != allowed.end())
              group.push_back(cpu);
          if (!group.empty()) groups.push_back(std::move(group));
        }
      }
    } else if (mode_ == TileSchedule::kNumaGroup) {
      for (const auto& node : topology.nodes) {
        std::vector<unsigned> group;
        for (const auto& core : node.cores)
          for (auto cpu : core.siblings)
            if (std::find(allowed.begin(), allowed.end(), cpu) != allowed.end())
              group.push_back(cpu);
        if (!group.empty()) groups.push_back(std::move(group));
      }
    }
    std::vector<unsigned> result;
    result.reserve(workers_);
    std::vector<bool> used(allowed.size(), false);
    auto add = [&](unsigned cpu) {
      if (result.size() >= workers_) return;
      auto it = std::find(allowed.begin(), allowed.end(), cpu);
      if (it == allowed.end()) return;
      auto index = static_cast<std::size_t>(it - allowed.begin());
      if (!used[index]) { used[index] = true; result.push_back(cpu); }
    };
    for (unsigned round = 0; result.size() < workers_ && round < groups.size(); ++round)
      add(groups[round][0]);
    for (unsigned offset = 0; result.size() < workers_ && offset < allowed.size(); ++offset)
      add(allowed[offset]);
    if (result.empty()) result.push_back(0);
    while (result.size() < workers_) result.push_back(result[result.size() % result.size()]);
    return result;
  }

  std::vector<std::vector<unsigned>> WorkerTeams(
      const CpuTopology& topology, unsigned lanes) const {
    lanes = std::max(1U, lanes);
    std::vector<unsigned> allowed = topology.affinity;
    if (allowed.empty()) {
      for (unsigned cpu = 0; cpu < topology.logical_cpus; ++cpu)
        allowed.push_back(cpu);
    }
    std::vector<std::vector<unsigned>> cores;
    for (const auto& node : topology.nodes) {
      for (const auto& core : node.cores) {
        std::vector<unsigned> siblings;
        for (auto cpu : core.siblings)
          if (std::find(allowed.begin(), allowed.end(), cpu) != allowed.end())
            siblings.push_back(cpu);
        if (!siblings.empty()) cores.push_back(std::move(siblings));
      }
    }
    if (cores.empty()) {
      for (auto cpu : allowed) cores.push_back({cpu});
    }
    std::vector<std::vector<unsigned>> result;
    result.reserve(workers_);
    for (unsigned worker = 0; worker < workers_; ++worker) {
      const auto& core = cores[worker % cores.size()];
      std::vector<unsigned> team;
      for (unsigned lane = 0; lane < lanes && lane < core.size(); ++lane)
        team.push_back(core[lane]);
      if (team.empty()) team.push_back(allowed[worker % allowed.size()]);
      result.push_back(std::move(team));
    }
    return result;
  }
#if defined(__linux__)
  static bool PinCurrentThread(const std::vector<unsigned>& cpus, unsigned worker) {
    if (cpus.empty()) return false;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpus[worker % cpus.size()], &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
  }
  static bool PinCurrentThread(const std::vector<std::vector<unsigned>>& teams,
                               unsigned worker) {
    if (teams.empty() || teams[worker % teams.size()].empty()) return false;
    cpu_set_t set; CPU_ZERO(&set);
    for (auto cpu : teams[worker % teams.size()]) CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
  }
#else
  static bool PinCurrentThread(const std::vector<unsigned>&, unsigned) { return false; }
  static bool PinCurrentThread(const std::vector<std::vector<unsigned>>&, unsigned) { return false; }
#endif
 private:
  uint64_t tiles_; unsigned workers_; TileSchedule mode_;
};

class DynamicTileQueue {
 public:
  explicit DynamicTileQueue(uint64_t tiles) : tiles_(tiles) {}
  bool Next(uint64_t* tile) {
    auto i = next_.fetch_add(1, std::memory_order_relaxed);
    if (i >= tiles_) return false;
    *tile = i; return true;
  }
 private:
  uint64_t tiles_; std::atomic<uint64_t> next_{0};
};
}
#endif
