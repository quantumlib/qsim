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

#ifndef UTIL_CUSTATEVEC_H_
#define UTIL_CUSTATEVEC_H_

#include <cublas_v2.h>
#include <custatevec.h>

#include "io.h"
#include "util_cuda.h"

namespace qsim {

inline void ErrorAssert(cublasStatus_t code, const char* file, unsigned line) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    IO::errorf("cuBLAS error %i: %s %d\n", code, file, line);
    exit(code);
  }
}

inline void ErrorAssert(
    custatevecStatus_t code, const char* file, unsigned line) {
  if (code != CUSTATEVEC_STATUS_SUCCESS) {
    IO::errorf("custatevec error: %s %s %d\n",
                custatevecGetErrorString(code), file, line);
    exit(code);
  }
}

}  // namespace qsim

#endif  // UTIL_CUSTATEVEC_H_
