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
#include <pybind11/stl_bind.h>
namespace py = pybind11;

#include <vector>

#include "../lib/circuit.h"
#include "../lib/expect.h"
#include "../lib/gates_cirq.h"
#include "../lib/qtrajectory.h"

// Methods for mutating noiseless circuits.
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

// Methods for mutating noisy circuits.
void add_gate_channel(
    const qsim::Cirq::GateKind gate_kind,
    const unsigned time,
    const std::vector<unsigned>& qubits,
    const std::map<std::string, float>& params,
    qsim::NoisyCircuit<qsim::Cirq::GateCirq<float>>* ncircuit);

void add_diagonal_gate_channel(
    const unsigned time, const std::vector<unsigned>& qubits,
    const std::vector<float>& angles,
    qsim::NoisyCircuit<qsim::Cirq::GateCirq<float>>* ncircuit);

void add_matrix_gate_channel(
    const unsigned time, const std::vector<unsigned>& qubits,
    const std::vector<float>& matrix,
    qsim::NoisyCircuit<qsim::Cirq::GateCirq<float>>* ncircuit);

void control_last_gate_channel(
    const std::vector<unsigned>& qubits, const std::vector<unsigned>& values,
    qsim::NoisyCircuit<qsim::Cirq::GateCirq<float>>* ncircuit);

void add_channel(const unsigned time,
                 const std::vector<unsigned>& qubits,
                 const std::vector<std::tuple<float, std::vector<float>, bool>>&
                     prob_matrix_unitary_triples,
                 qsim::NoisyCircuit<qsim::Cirq::GateCirq<float>>* ncircuit);

// Method for populating opstrings.
void add_gate_to_opstring(
    const qsim::Cirq::GateKind gate_kind,
    const std::vector<unsigned>& qubits,
    qsim::OpString<qsim::Cirq::GateCirq<float>>* opstring);

// Methods for simulating noiseless circuits.
std::vector<std::complex<float>> qsim_simulate(const py::dict &options);

py::array_t<float> qsim_simulate_fullstate(
      const py::dict &options, uint64_t input_state);
py::array_t<float> qsim_simulate_fullstate(
      const py::dict &options, const py::array_t<float> &input_vector);

std::vector<unsigned> qsim_sample(const py::dict &options);

// Methods for simulating noisy circuits.
std::vector<std::complex<float>> qtrajectory_simulate(const py::dict &options);

py::array_t<float> qtrajectory_simulate_fullstate(
      const py::dict &options, uint64_t input_state);
py::array_t<float> qtrajectory_simulate_fullstate(
      const py::dict &options, const py::array_t<float> &input_vector);

std::vector<unsigned> qtrajectory_sample(const py::dict &options);

// As above, but returning expectation values instead.
std::vector<std::complex<double>> qsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<qsim::OpString<
                              qsim::Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    uint64_t input_state);
std::vector<std::complex<double>> qsim_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<qsim::OpString<
                              qsim::Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector);
std::vector<std::complex<double>> qtrajectory_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<qsim::OpString<
                              qsim::Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    uint64_t input_state);
std::vector<std::complex<double>> qtrajectory_simulate_expectation_values(
    const py::dict &options,
    const std::vector<std::tuple<
                          std::vector<qsim::OpString<
                              qsim::Cirq::GateCirq<float>>>,
                          unsigned>>& opsums_and_qubit_counts,
    const py::array_t<float> &input_vector);

// Hybrid simulator.
std::vector<std::complex<float>> qsimh_simulate(const py::dict &options);

PYBIND11_MODULE(qsim, m) {
  m.doc() = "pybind11 plugin";  // optional module docstring

  // Methods for returning amplitudes
  m.def("qsim_simulate", &qsim_simulate, "Call the qsim simulator");
  m.def("qtrajectory_simulate", &qtrajectory_simulate,
        "Call the qtrajectory simulator");

  // Methods for returning full state
  m.def("qsim_simulate_fullstate",
        static_cast<py::array_t<float>(*)(const py::dict&, uint64_t)>(
            &qsim_simulate_fullstate),
        "Call the qsim simulator for full state vector simulation");
  m.def("qsim_simulate_fullstate",
        static_cast<py::array_t<float>(*)(const py::dict&,
                                          const py::array_t<float>&)>(
            &qsim_simulate_fullstate),
        "Call the qsim simulator for full state vector simulation");

  m.def("qtrajectory_simulate_fullstate",
        static_cast<py::array_t<float>(*)(const py::dict&, uint64_t)>(
            &qtrajectory_simulate_fullstate),
        "Call the qtrajectory simulator for full state vector simulation");
  m.def("qtrajectory_simulate_fullstate",
        static_cast<py::array_t<float>(*)(const py::dict&,
                                          const py::array_t<float>&)>(
            &qtrajectory_simulate_fullstate),
        "Call the qtrajectory simulator for full state vector simulation");

  // Methods for returning samples
  m.def("qsim_sample", &qsim_sample, "Call the qsim sampler");
  m.def("qtrajectory_sample", &qtrajectory_sample,
        "Call the qtrajectory sampler");

  using GateCirq = qsim::Cirq::GateCirq<float>;
  using OpString = qsim::OpString<GateCirq>;

  // Methods for returning expectation values
  m.def("qsim_simulate_expectation_values",
        static_cast<std::vector<std::complex<double>>(*)(
            const py::dict&,
            const std::vector<std::tuple<std::vector<OpString>, unsigned>>&,
            uint64_t)>(
          &qsim_simulate_expectation_values),
        "Call the qsim simulator for expectation value simulation");
  m.def("qsim_simulate_expectation_values",
        static_cast<std::vector<std::complex<double>>(*)(
            const py::dict&,
            const std::vector<std::tuple<std::vector<OpString>, unsigned>>&,
            const py::array_t<float>&)>(
          &qsim_simulate_expectation_values),
        "Call the qsim simulator for expectation value simulation");

  m.def("qtrajectory_simulate_expectation_values",
        static_cast<std::vector<std::complex<double>>(*)(
            const py::dict&,
            const std::vector<std::tuple<std::vector<OpString>, unsigned>>&,
            uint64_t)>(
          &qtrajectory_simulate_expectation_values),
        "Call the qtrajectory simulator for expectation value simulation");
  m.def("qtrajectory_simulate_expectation_values",
        static_cast<std::vector<std::complex<double>>(*)(
            const py::dict&,
            const std::vector<std::tuple<std::vector<OpString>, unsigned>>&,
            const py::array_t<float>&)>(
          &qtrajectory_simulate_expectation_values),
        "Call the qtrajectory simulator for expectation value simulation");

  // Method for hybrid simulation
  m.def("qsimh_simulate", &qsimh_simulate, "Call the qsimh simulator");

  using GateKind = qsim::Cirq::GateKind;
  using Circuit = qsim::Circuit<GateCirq>;
  using NoisyCircuit = qsim::NoisyCircuit<GateCirq>;

  py::class_<Circuit>(m, "Circuit")
    .def(py::init<>())
    .def_readwrite("num_qubits", &Circuit::num_qubits)
    .def_readwrite("gates", &Circuit::gates);

  py::class_<NoisyCircuit>(m, "NoisyCircuit")
    .def(py::init<>())
    .def_readwrite("num_qubits", &NoisyCircuit::num_qubits)
    .def_readwrite("channels", &NoisyCircuit::channels);

  py::class_<OpString>(m, "OpString")
    .def(py::init<>())
    .def_readwrite("weight", &OpString::weight)
    .def_readwrite("ops", &OpString::ops);

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

  m.def("add_gate_channel", &add_gate_channel,
        "Adds a gate to the given noisy circuit.");
  m.def("add_diagonal_gate_channel", &add_diagonal_gate_channel,
        "Adds a two- or three-qubit diagonal gate to the given noisy circuit.");
  m.def("add_matrix_gate_channel", &add_matrix_gate_channel,
        "Adds a matrix-defined gate to the given noisy circuit.");
  m.def("control_last_gate_channel", &control_last_gate_channel,
        "Applies controls to the final channel of a noisy circuit.");

  m.def("add_channel", &add_channel,
        "Adds a channel to the given noisy circuit.");

  m.def("add_gate_to_opstring", &add_gate_to_opstring,
        "Adds a gate to the given opstring.");
}

#endif
