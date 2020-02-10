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

# This file provides an alternate method for building tests in this directory.
# Prefer using the Makefile (e.g. `make run-all`) if possible.

path_to_include=googletest/googletest/include
path_to_lib=googletest/lib

g++ -O3 -I$path_to_include -L$path_to_lib -o bitstring_test bitstring_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -o circuit_reader_test circuit_reader_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -o fuser_basic_test fuser_basic_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -o gates_def_test gates_def_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -o matrix_test matrix_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -lgomp -o hybrid_test hybrid_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -lgomp -o run_qsim_test run_qsim_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -lgomp -o run_qsimh_test run_qsimh_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -lgomp -o simulator_avx_test simulator_avx_test.cc -lpthread -lgtest
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -lgomp -o simulator_basic_test simulator_basic_test.cc -lpthread -lgtest
