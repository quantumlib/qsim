// Copyright 2026 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CPU_THREAD_TOPOLOGY_H_
#define CPU_THREAD_TOPOLOGY_H_

#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace qsim {

struct CpuCore {
  unsigned package_id;
  unsigned core_id;
  std::vector<unsigned> logical_cpus;
};

class CpuThreadTopology {
 public:
  static CpuThreadTopology Discover() {
    CpuThreadTopology topology;

#ifdef __linux__
    cpu_set_t allowed_cpus;
    CPU_ZERO(&allowed_cpus);
    if (sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) != 0) {
      topology.error_ = "cannot read the process CPU affinity";
      return topology;
    }

    std::map<std::pair<unsigned, unsigned>, std::vector<unsigned>> cores;
    for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
      if (!CPU_ISSET(cpu, &allowed_cpus)) continue;

      unsigned package_id;
      unsigned core_id;
      if (!ReadTopologyValue(cpu, "physical_package_id", package_id) ||
          !ReadTopologyValue(cpu, "core_id", core_id)) {
        topology.error_ = "cannot read Linux CPU topology from sysfs";
        return topology;
      }
      cores[{package_id, core_id}].push_back(cpu);
    }

    for (auto& [core, logical_cpus] : cores) {
      std::sort(logical_cpus.begin(), logical_cpus.end());
      topology.cores_.push_back(
          CpuCore{core.first, core.second, std::move(logical_cpus)});
    }
#else
    topology.error_ = "automatic CPU topology discovery requires Linux";
#endif

    return topology;
  }

  bool BuildTeamCpuOrder(unsigned num_threads, unsigned threads_per_team,
                         std::vector<unsigned>& thread_cpus,
                         std::string& error) const {
    thread_cpus.clear();
    error.clear();
    if (!error_.empty()) {
      error = error_;
      return false;
    }
    if (threads_per_team == 0 || num_threads % threads_per_team != 0) {
      error = "thread count must be divisible by threads per team";
      return false;
    }

    const auto num_teams = num_threads / threads_per_team;
    for (const auto& core : cores_) {
      if (core.logical_cpus.size() < threads_per_team) continue;
      for (unsigned lane = 0; lane < threads_per_team; ++lane) {
        thread_cpus.push_back(core.logical_cpus[lane]);
      }
      if (thread_cpus.size() == num_threads) return true;
    }

    error = "not enough physical cores with the requested SMT siblings";
    thread_cpus.clear();
    return false;
  }

 private:
#ifdef __linux__
  static bool ReadTopologyValue(unsigned cpu, const char* name,
                                unsigned& value) {
    const auto path = "/sys/devices/system/cpu/cpu" +
                      std::to_string(cpu) + "/topology/" + name;
    std::ifstream input(path);
    return bool(input >> value);
  }
#endif

  std::vector<CpuCore> cores_;
  std::string error_;
};

inline bool PinCurrentThreadToCpu(unsigned cpu) {
#ifdef __linux__
  if (cpu >= CPU_SETSIZE) return false;
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  CPU_SET(cpu, &affinity);
  return sched_setaffinity(0, sizeof(affinity), &affinity) == 0;
#else
  (void) cpu;
  return false;
#endif
}

}  // namespace qsim

#endif  // CPU_THREAD_TOPOLOGY_H_
