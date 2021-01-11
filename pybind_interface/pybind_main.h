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

#ifndef __PYBIND_MAIN
#define __PYBIND_MAIN

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <vector>

#include "../lib/circuit.h"
#include "../lib/gates_cirq.h"

void add_gate(const qsim::Cirq::GateKind gate_kind, const unsigned time,
              const std::vector<unsigned>& qubits,
              const std::map<std::string, float>& params,
              qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit);

void add_diagonal_gate(const unsigned time, const std::vector<unsigned>& qubits,
                       const std::vector<float>& angles,
                       qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit);

void add_matrix_gate(const unsigned time, const std::vector<unsigned>& qubits,
                     const std::vector<float>& matrix,
                     qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit);

void control_last_gate(const std::vector<unsigned>& qubits,
                       const std::vector<unsigned>& values,
                       qsim::Circuit<qsim::Cirq::GateCirq<float>>* circuit);

std::vector<std::complex<float>> qsim_simulate(const py::dict &options);

py::array_t<float> qsim_simulate_fullstate(
      const py::dict &options, uint64_t input_state);
py::array_t<float> qsim_simulate_fullstate(
      const py::dict &options, const py::array_t<float> &input_vector);

std::vector<unsigned> qsim_sample(const py::dict &options);

std::vector<std::complex<float>> qsimh_simulate(const py::dict &options);

PYBIND11_MODULE(qsim, m) {
  m.doc() = "pybind11 plugin";  // optional module docstring

  m.def("qsim_simulate", &qsim_simulate, "Call the qsim simulator");
  m.def("qsim_simulate_fullstate",
        static_cast<py::array_t<float>(*)(const py::dict&, uint64_t)>(
            &qsim_simulate_fullstate),
        "Call the qsim simulator for full state vector simulation");
  m.def("qsim_simulate_fullstate",
        static_cast<py::array_t<float>(*)(const py::dict&,
                                          const py::array_t<float>&)>(
            &qsim_simulate_fullstate),
        "Call the qsim simulator for full state vector simulation");
  m.def("qsim_sample", &qsim_sample, "Call the qsim sampler");
  m.def("qsimh_simulate", &qsimh_simulate, "Call the qsimh simulator");

  using GateCirq = qsim::Cirq::GateCirq<float>;
  using GateKind = qsim::Cirq::GateKind;
  using Circuit = qsim::Circuit<GateCirq>;

  py::class_<Circuit>(m, "Circuit")
    .def(py::init<>())
    .def_readwrite("num_qubits", &Circuit::num_qubits)
    .def_readwrite("gates", &Circuit::gates);

  py::enum_<GateKind>(m, "GateKind")
    .value("kI1", GateKind::kI1)
    .value("kI2", GateKind::kI2)
    .value("kI", GateKind::kI)
    .value("kXPowGate", GateKind::kXPowGate)
    .value("kYPowGate", GateKind::kYPowGate)
    .value("kZPowGate", GateKind::kZPowGate)
    .value("kHPowGate", GateKind::kHPowGate)
    .value("kCZPowGate", GateKind::kCZPowGate)
    .value("kCXPowGate", GateKind::kCXPowGate)
    .value("krx", GateKind::krx)
    .value("kry", GateKind::kry)
    .value("krz", GateKind::krz)
    .value("kH", GateKind::kH)
    .value("kS", GateKind::kS)
    .value("kCZ", GateKind::kCZ)
    .value("kCX", GateKind::kCX)
    .value("kT", GateKind::kT)
    .value("kX", GateKind::kX)
    .value("kY", GateKind::kY)
    .value("kZ", GateKind::kZ)
    .value("kPhasedXPowGate", GateKind::kPhasedXPowGate)
    .value("kPhasedXZGate", GateKind::kPhasedXZGate)
    .value("kXXPowGate", GateKind::kXXPowGate)
    .value("kYYPowGate", GateKind::kYYPowGate)
    .value("kZZPowGate", GateKind::kZZPowGate)
    .value("kXX", GateKind::kXX)
    .value("kYY", GateKind::kYY)
    .value("kZZ", GateKind::kZZ)
    .value("kSwapPowGate", GateKind::kSwapPowGate)
    .value("kISwapPowGate", GateKind::kISwapPowGate)
    .value("kriswap", GateKind::kriswap)
    .value("kSWAP", GateKind::kSWAP)
    .value("kISWAP", GateKind::kISWAP)
    .value("kPhasedISwapPowGate", GateKind::kPhasedISwapPowGate)
    .value("kgivens", GateKind::kgivens)
    .value("kFSimGate", GateKind::kFSimGate)
    .value("kTwoQubitDiagonalGate", GateKind::kTwoQubitDiagonalGate)
    .value("kThreeQubitDiagonalGate", GateKind::kThreeQubitDiagonalGate)
    .value("kCCZPowGate", GateKind::kCCZPowGate)
    .value("kCCXPowGate", GateKind::kCCXPowGate)
    .value("kCSwapGate", GateKind::kCSwapGate)
    .value("kCCZ", GateKind::kCCZ)
    .value("kCCX", GateKind::kCCX)
    .value("kMatrixGate", GateKind::kMatrixGate)
    .value("kMeasurement", GateKind::kMeasurement)
    .export_values();

  m.def("add_gate", &add_gate, "Adds a gate to the given circuit.");
  m.def("add_diagonal_gate", &add_diagonal_gate,
        "Adds a two- or three-qubit diagonal gate to the given circuit.");
  m.def("add_matrix_gate", &add_matrix_gate,
        "Adds a matrix-defined gate to the given circuit.");
  m.def("control_last_gate", &control_last_gate,
        "Applies controls to the final gate of a circuit.");
}

#endif
