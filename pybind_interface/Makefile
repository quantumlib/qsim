# The names of the shared libraries that result after compiling qsim for Pybind11
QSIMLIB_BASIC = ../qsimcirq/qsim_basic`python3-config --extension-suffix`
QSIMLIB_SSE = ../qsimcirq/qsim_sse`python3-config --extension-suffix`
QSIMLIB_AVX2 = ../qsimcirq/qsim_avx2`python3-config --extension-suffix`
QSIMLIB_AVX512 = ../qsimcirq/qsim_avx512`python3-config --extension-suffix`
QSIMLIB_DECIDE = ../qsimcirq/qsim_decide`python3-config --extension-suffix`


# The flags for the compilation of the simd-specific Pybind11 interfaces
PYBINDFLAGS_BASIC = -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`
PYBINDFLAGS_SSE = -msse4.1 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`
PYBINDFLAGS_AVX2 = -mavx2 -mfma -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`
PYBINDFLAGS_AVX512 = -mavx512f -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes`

.PHONY: pybind
pybind:
	$(CXX) basic/pybind_main_basic.cpp -o $(QSIMLIB_BASIC) $(CXXFLAGS) $(PYBINDFLAGS_BASIC)
	$(CXX) sse/pybind_main_sse.cpp -o $(QSIMLIB_SSE) $(CXXFLAGS) $(PYBINDFLAGS_SSE)
	$(CXX) avx2/pybind_main_avx2.cpp -o $(QSIMLIB_AVX2) $(CXXFLAGS) $(PYBINDFLAGS_AVX2)
	$(CXX) avx512/pybind_main_avx512.cpp -o $(QSIMLIB_AVX512) $(CXXFLAGS) $(PYBINDFLAGS_AVX512)
	$(CXX) decide/decide.cpp -o $(QSIMLIB_DECIDE) $(CXXFLAGS) $(PYBINDFLAGS_BASIC)

.PHONY: clean
clean:
	-rm -f ./basic/*.x ./basic/*.a ./basic/*.so ./basic/*.mod $(QSIMLIB_BASIC)
	-rm -f ./sse/*.x ./sse/*.a ./sse/*.so ./sse/*.mod $(QSIMLIB_SSE)
	-rm -f ./avx2/*.x ./avx2/*.a ./avx2/*.so ./avx2/*.mod $(QSIMLIB_AVX2)
	-rm -f ./avx512/*.x ./avx512/*.a ./avx512/*.so ./avx512/*.mod $(QSIMLIB_AVX512)
	-rm -f ./decide/*.x ./decide/*.a ./decide/*.so ./decide/*.mod $(QSIMLIB_DECIDE)
