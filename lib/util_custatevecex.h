// Copyright 2025 Google LLC. All Rights Reserved.
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

#ifndef UTIL_CUSTATEVECEX_H_
#define UTIL_CUSTATEVECEX_H_

#include <custatevecEx.h>
#include <custatevecEx_ext.h>

#include "io.h"
#include "util_cuda.h"

namespace qsim {

inline void ErrorAssert(
    custatevecExCommunicatorStatus_t code, const char* file, unsigned line) {
  if (code != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS) {
    IO::errorf(
        "cuStateVecEx communicator error %d at %s %d\n", code, file, line);
    exit(code);
  }
}

inline unsigned get_num_global_qubits(unsigned num_devices) {
  unsigned num_global_qubits = 0;
  while ((num_devices >>= 1) > 0) {
    ++num_global_qubits;
  }

  return num_global_qubits;
}

}  // namespace qsim

#endif  // UTIL_CUSTATEVECEX_H_
