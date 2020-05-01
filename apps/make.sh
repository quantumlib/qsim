# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# This file provides an alternate method for building apps in this directory.
# Prefer using the Makefile (e.g. `make -C apps/`) if possible.

g++ -O3 -march=native -fopenmp -o qsim_base.x qsim_base.cc
g++ -O3 -march=native -fopenmp -o qsim_von_neumann.x qsim_von_neumann.cc
g++ -O3 -march=native -fopenmp -o qsim_amplitudes.x qsim_amplitudes.cc
g++ -O3 -march=native -fopenmp -o qsimh_base.x qsimh_base.cc
g++ -O3 -march=native -fopenmp -o qsimh_amplitudes.x qsimh_amplitudes.cc
