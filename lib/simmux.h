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

#ifndef SIMMUX_H_
#define SIMMUX_H_

#ifdef __AVX2__
# include "simulator_avx.h"
  namespace qsim {
    template <typename For>
    using Simulator = SimulatorAVX<For>;
  }
#elif __SSE4_1__
# include "simulator_sse.h"
  namespace qsim {
    template <typename For>
    using Simulator = SimulatorSSE<For>;
  }
#else
# include "simulator_basic.h"
  namespace qsim {
    template <typename For>
    using Simulator = SimulatorBasic<For>;
  }
#endif

#endif  // SIMMUX_H_
