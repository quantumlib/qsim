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

#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace qsim {

namespace cpu_topology_internal {

#ifdef __linux__
using PhysicalCoreKey = std::pair<unsigned, unsigned>;
using CoreSiblingsMap = std::map<PhysicalCoreKey, std::vector<unsigned>>;

inline bool ReadSysfsTopologyId(unsigned cpu, const char* field,
                                unsigned& value) {
  std::ifstream input("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                      "/topology/" + field);
  return bool(input >> value);
}

inline bool ReadPhysicalCoreKey(unsigned cpu, PhysicalCoreKey& core_key) {
  return ReadSysfsTopologyId(cpu, "physical_package_id", core_key.first) &&
         ReadSysfsTopologyId(cpu, "core_id", core_key.second);
}

inline bool GroupAllowedCpusByPhysicalCore(CoreSiblingsMap& cores,
                                           std::string& error) {
  cpu_set_t allowed_cpus;
  CPU_ZERO(&allowed_cpus);
  if (sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) != 0) {
    error = "cannot read the process CPU affinity";
    return false;
  }

  for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed_cpus)) continue;
    PhysicalCoreKey core_key;
    if (!ReadPhysicalCoreKey(cpu, core_key)) {
      error = "cannot read Linux CPU topology from sysfs";
      return false;
    }
    cores[core_key].push_back(cpu);
  }
  return true;
}

inline bool SelectSmtTeamsFromCores(const CoreSiblingsMap& cores,
                                    unsigned num_threads,
                                    unsigned threads_per_team,
                                    std::vector<unsigned>& thread_cpus,
                                    std::string& error) {
  for (const auto& entry : cores) {
    const auto& siblings = entry.second;
    if (siblings.size() < threads_per_team) continue;
    for (unsigned lane = 0; lane < threads_per_team; ++lane) {
      thread_cpus.push_back(siblings[lane]);
    }
    if (thread_cpus.size() == num_threads) return true;
  }

  error = "not enough physical cores with the requested SMT siblings";
  thread_cpus.clear();
  return false;
}
#endif

}  // namespace cpu_topology_internal

inline bool BuildSmtTeamCpuOrder(unsigned num_threads,
                                 unsigned threads_per_team,
                                 std::vector<unsigned>& thread_cpus,
                                 std::string& error) {
  thread_cpus.clear();
  error.clear();
  if (threads_per_team == 0 || num_threads % threads_per_team != 0) {
    error = "thread count must be divisible by threads per team";
    return false;
  }

#ifdef __linux__
  cpu_topology_internal::CoreSiblingsMap cores;
  if (!cpu_topology_internal::GroupAllowedCpusByPhysicalCore(cores, error)) {
    return false;
  }
  return cpu_topology_internal::SelectSmtTeamsFromCores(
      cores, num_threads, threads_per_team, thread_cpus, error);
#else
  error = "automatic CPU topology discovery requires Linux";
  return false;
#endif
}

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
