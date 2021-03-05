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
# Prefer using the Makefile (e.g. `make -C tests/`) if possible.

path_to_include=googletest/googletest/include
path_to_lib=googletest/lib

g++ -O3 -I$path_to_include -L$path_to_lib -o bitstring_test.x bitstring_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o channels_cirq_test.x channels_cirq_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -o circuit_qsim_parser_test.x circuit_qsim_parser_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o expect_test.x expect_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -o fuser_basic_test.x fuser_basic_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -o fuser_mqubit_test.x fuser_mqubit_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -o gates_qsim_test.x gates_qsim_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o hybrid_test.x hybrid_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -o matrix_test.x matrix_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o qtrajectory_test.x qtrajectory_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o run_qsim_test.x run_qsim_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o run_qsimh_test.x run_qsimh_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o simulator_avx_test.x simulator_avx_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -fopenmp -o simulator_basic_test.x simulator_basic_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -msse4 -fopenmp -o simulator_sse_test.x simulator_sse_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o statespace_avx_test.x statespace_avx_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -fopenmp -o statespace_basic_test.x statespace_basic_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -msse4 -fopenmp -o statespace_sse_test.x statespace_sse_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o unitary_calculator_avx_test.x unitary_calculator_avx_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -fopenmp -o unitary_calculator_basic_test.x unitary_calculator_basic_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -msse4 -fopenmp -o unitary_calculator_sse_test.x unitary_calculator_sse_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -mavx2 -mfma -fopenmp -o unitaryspace_avx_test.x unitaryspace_avx_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -fopenmp -o unitaryspace_basic_test.x unitaryspace_basic_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -msse4 -fopenmp -o unitaryspace_sse_test.x unitaryspace_sse_test.cc -lgtest -lpthread
g++ -O3 -I$path_to_include -L$path_to_lib -o vectorspace_test.x vectorspace_test.cc -lgtest -lpthread
