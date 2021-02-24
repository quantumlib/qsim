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

#ifndef CIRCUIT_QSIM_PARSER_H_
#define CIRCUIT_QSIM_PARSER_H_

#include <algorithm>
#include <cctype>
#include <string>
#include <sstream>
#include <vector>

#include "circuit.h"
#include "gates_qsim.h"

namespace qsim {

/**
 * Parser for the (deprecated) qsim <a href="https://github.com/quantumlib/qsim/blob/master/docs/input_format.md">file input format</a>.
 * The primary supported interface for designing circuits to simulate with qsim
 * is <a href="https://github.com/quantumlib/Cirq">Cirq</a>, which relies on
 * the Python-based qsimcirq interface. For C++ applications, Cirq gates can be
 * explicitly constructed in code.
 */
template <typename IO>
class CircuitQsimParser final {
 public:
  /**
   * Parses the given input stream into a Circuit object, following the rules
   * defined in "docs/input_format.md".
   * @param maxtime Maximum gate "time" to read operations for (inclusive).
   * @param provider Circuit source; only used for error reporting.
   * @param fs The stream to read the circuit from.
   * @param circuit Output circuit object. If parsing is successful, this will
   *   contain the circuit defined in 'fs'.
   * @return True if parsing succeeds; false otherwise.
   */
  template <typename Stream, typename fp_type>
  static bool FromStream(unsigned maxtime, const std::string& provider,
                         Stream& fs, Circuit<GateQSim<fp_type>>& circuit) {
    circuit.num_qubits = 0;

    circuit.gates.resize(0);
    circuit.gates.reserve(1024);

    unsigned k = 0;

    std::string line;
    line.reserve(128);

    unsigned time;
    std::string gate_name;
    gate_name.reserve(16);

    unsigned max_time = 0;
    unsigned prev_mea_time = 0;

    std::vector<unsigned> last_times;

    while (std::getline(fs, line)) {
      ++k;

      if (line.size() == 0 || line[0] == '#') continue;

      std::stringstream ss(line);

      if (circuit.num_qubits == 0) {
        ss >> circuit.num_qubits;
        if (circuit.num_qubits == 0) {
          IO::errorf("invalid number of qubits in %s in line %u.\n",
                     provider.c_str(), k);
          return false;
        }

        last_times.resize(circuit.num_qubits, -1);

        continue;
      }

      ss >> time >> gate_name;

      if (!ss) {
        InvalidGateError(provider, k);
        return false;
      }

      if (time > maxtime) {
        break;
      }

      if (gate_name == "c") {
        if (!ParseControlledGate<fp_type>(ss, time,
                                          circuit.num_qubits, circuit.gates)) {
          InvalidGateError(provider, k);
          return false;
        }
      } else if (!ParseGate<fp_type>(ss, time, circuit.num_qubits,
                                     gate_name, circuit.gates)) {
        InvalidGateError(provider, k);
        return false;
      }

      const auto& gate = circuit.gates.back();

      if (time < prev_mea_time
          || (gate.kind == gate::kMeasurement && time < max_time)) {
        IO::errorf("gate crosses the time boundary set by measurement "
                   "gates in line %u in %s.\n", k, provider.c_str());
        return false;
      }

      if (gate.kind == gate::kMeasurement) {
        prev_mea_time = time;
      }

      if (GateIsOutOfOrder(time, gate.qubits, last_times)
          || GateIsOutOfOrder(time, gate.controlled_by, last_times)) {
        IO::errorf("gate is out of time order in line %u in %s.\n",
                   k, provider.c_str());
        return false;
      }

      if (time > max_time) {
        max_time = time;
      }
    }

    return true;
  }

  /**
   * Parses the given file into a Circuit object, following the rules defined
   * in "docs/input_format.md".
   * @param maxtime Maximum gate "time" to read operations for (inclusive).
   * @param file The name of the file to read the circuit from.
   * @param circuit Output circuit object. If parsing is successful, this will
   *   contain the circuit defined in 'file'.
   * @return True if parsing succeeds; false otherwise.
   */
  template <typename fp_type>
  static bool FromFile(unsigned maxtime, const std::string& file,
                       Circuit<GateQSim<fp_type>>& circuit) {
    auto fs = IO::StreamFromFile(file);

    if (!fs) {
      return false;
    } else {
      bool rc = FromStream(maxtime, file, fs, circuit);
      IO::CloseStream(fs);
      return rc;
    }
  }

 private:
  static void InvalidGateError(const std::string& provider, unsigned line) {
    IO::errorf("invalid gate in %s in line %u.\n", provider.c_str(), line);
  }

  /**
   * Checks formatting for a single-qubit gate parsed from 'ss'.
   * @param ss Input stream containing the gate specification.
   * @param num_qubits Number of qubits, as defined at the start of the file.
   * @param q0 Index of the affected qubit.
   * @param provider Circuit source; only used for error reporting.
   * @param line Line number of the parsed gate; only used for error reporting.
   */
  static bool ValidateGate(std::stringstream& ss,
                           unsigned num_qubits, unsigned q0) {
    return ss && ss.peek() == std::stringstream::traits_type::eof()
        && q0 < num_qubits;
  }

  /**
   * Checks formatting for a two-qubit gate parsed from 'ss'.
   * @param ss Input stream containing the gate specification.
   * @param num_qubits Number of qubits, as defined at the start of the file.
   * @param q0 Index of the first affected qubit.
   * @param q1 Index of the second affected qubit.
   */
  static bool ValidateGate(std::stringstream& ss,
                           unsigned num_qubits, unsigned q0, unsigned q1) {
    return ss && ss.peek() == std::stringstream::traits_type::eof()
        && q0 < num_qubits && q1 < num_qubits && q0 != q1;
  }

  /**
   * Checks formatting for a multiqubit gate parsed from 'ss'.
   * @param ss Input stream containing the gate specification.
   * @param num_qubits Number of qubits, as defined at the start of the file.
   * @param qubits Indices of affected qubits.
   */
  static bool ValidateGate(std::stringstream& ss, unsigned num_qubits,
                           const std::vector<unsigned>& qubits) {
    return ss && ValidateQubits(num_qubits, qubits);
  }

  static bool ValidateControlledGate(
      unsigned num_qubits, const std::vector<unsigned>& qubits,
      const std::vector<unsigned>& controlled_by) {
    if (!ValidateQubits(num_qubits, controlled_by)) return false;

    std::size_t i = 0, j = 0;

    while (i < qubits.size() && j < controlled_by.size()) {
      if (qubits[i] == controlled_by[j]) {
        return false;
      } else if (qubits[i] < controlled_by[j]) {
        ++i;
      } else {
        ++j;
      }
    }

    return true;
  }

  static bool ValidateQubits(unsigned num_qubits,
                             const std::vector<unsigned>& qubits) {
    if (qubits.size() == 0 || qubits[0] >= num_qubits) return false;

    // qubits should be sorted.

    for (std::size_t i = 1; i < qubits.size(); ++i) {
      if (qubits[i] >= num_qubits || qubits[i] == qubits[i - 1]) {
        return false;
      }
    }

    return true;
  }

  static bool GateIsOutOfOrder(unsigned time,
                               const std::vector<unsigned>& qubits,
                               std::vector<unsigned>& last_times) {
    for (auto q : qubits) {
      if (last_times[q] != unsigned(-1) && time <= last_times[q]) {
        return true;
      }

      last_times[q] = time;
    }

    return false;
  }

  template <typename fp_type, typename Stream, typename Gate>
  static bool ParseGate(Stream& ss, unsigned time, unsigned num_qubits,
                        const std::string& gate_name,
                        std::vector<Gate>& gates) {
    unsigned q0, q1;
    fp_type phi, theta;

    if (gate_name == "id1") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateId1<fp_type>::Create(time, q0));
    } else if (gate_name == "h") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateHd<fp_type>::Create(time, q0));
    } else if (gate_name == "t") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateT<fp_type>::Create(time, q0));
    } else if (gate_name == "x") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateX<fp_type>::Create(time, q0));
    } else if (gate_name == "y") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateY<fp_type>::Create(time, q0));
    } else if (gate_name == "z") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateZ<fp_type>::Create(time, q0));
    } else if (gate_name == "x_1_2") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateX2<fp_type>::Create(time, q0));
    } else if (gate_name == "y_1_2") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateY2<fp_type>::Create(time, q0));
    } else if (gate_name == "rx") {
      ss >> q0 >> phi;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateRX<fp_type>::Create(time, q0, phi));
    } else if (gate_name == "ry") {
      ss >> q0 >> phi;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateRY<fp_type>::Create(time, q0, phi));
    } else if (gate_name == "rz") {
      ss >> q0 >> phi;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateRZ<fp_type>::Create(time, q0, phi));
    } else if (gate_name == "rxy") {
      ss >> q0 >> theta >> phi;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateRXY<fp_type>::Create(time, q0, theta, phi));
    } else if (gate_name == "hz_1_2") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateHZ2<fp_type>::Create(time, q0));
    } else if (gate_name == "s") {
      ss >> q0;
      if (!ValidateGate(ss, num_qubits, q0)) return false;
      gates.push_back(GateS<fp_type>::Create(time, q0));
    } else if (gate_name == "id2") {
      ss >> q0 >> q1;
      if (!ValidateGate(ss, num_qubits, q0, q1)) return false;
      gates.push_back(GateId2<fp_type>::Create(time, q0, q1));
    } else if (gate_name == "cz") {
      ss >> q0 >> q1;
      if (!ValidateGate(ss, num_qubits, q0, q1)) return false;
      gates.push_back(GateCZ<fp_type>::Create(time, q0, q1));
    } else if (gate_name == "cnot" || gate_name == "cx") {
      ss >> q0 >> q1;
      if (!ValidateGate(ss, num_qubits, q0, q1)) return false;
      gates.push_back(GateCNot<fp_type>::Create(time, q0, q1));
    } else if (gate_name == "sw") {
      ss >> q0 >> q1;
      if (!ValidateGate(ss, num_qubits, q0, q1)) return false;
      gates.push_back(GateSwap<fp_type>::Create(time, q0, q1));
    } else if (gate_name == "is") {
      ss >> q0 >> q1;
      if (!ValidateGate(ss, num_qubits, q0, q1)) return false;
      gates.push_back(GateIS<fp_type>::Create(time, q0, q1));
    } else if (gate_name == "fs") {
      ss >> q0 >> q1 >> theta >> phi;
      if (!ValidateGate(ss, num_qubits, q0, q1)) return false;
      gates.push_back(GateFS<fp_type>::Create(time, q0, q1, theta, phi));
    } else if (gate_name == "cp") {
      ss >> q0 >> q1 >> phi;
      if (!ValidateGate(ss, num_qubits, q0, q1)) return false;
      gates.push_back(GateCP<fp_type>::Create(time, q0, q1, phi));
    } else if (gate_name == "m") {
      std::vector<unsigned> qubits;
      qubits.reserve(num_qubits);

      while (ss.good()) {
        ss >> q0;
        if (ss) {
          qubits.push_back(q0);
        } else {
          return false;
        }
      }

      gates.push_back(gate::Measurement<GateQSim<fp_type>>::Create(
          time, std::move(qubits)));

      if (!ValidateQubits(num_qubits, gates.back().qubits)) return false;
    } else {
      return false;
    }

    return true;
  }

  template <typename fp_type, typename Stream, typename Gate>
  static bool ParseControlledGate(Stream& ss, unsigned time,
                                  unsigned num_qubits,
                                  std::vector<Gate>& gates) {
    std::vector<unsigned> controlled_by;
    controlled_by.reserve(64);

    std::string gate_name;
    gate_name.reserve(16);

    while (1) {
      while (ss.good()) {
        if (!std::isblank(ss.get())) {
          ss.unget();
          break;
        }
      }

      if (!ss.good()) {
        return false;
      }

      if (!std::isdigit(ss.peek())) {
        break;
      } else {
        unsigned q;
        ss >> q;

        if (!ss.good() || !std::isblank(ss.get())) {
          return false;
        }

        controlled_by.push_back(q);
      }
    }

    if (controlled_by.size() == 0) {
      return false;
    }

    ss >> gate_name;

    if (!ss.good() || !ParseGate<fp_type>(ss, time,
                                          num_qubits, gate_name, gates)) {
      return false;
    }

    gates.back().ControlledBy(std::move(controlled_by));

    if (!ValidateControlledGate(num_qubits, gates.back().qubits,
                                gates.back().controlled_by)) {
      return false;
    }

    return true;
  }
};

}  // namespace qsim

#endif  // CIRCUIT_QSIM_PARSER_H_
