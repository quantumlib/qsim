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
#include <string>
#include <sstream>
#include <vector>

#include "circuit.h"
#include "gates_qsim.h"

namespace qsim {

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

    unsigned time, q0, q1;
    std::string gate_name;
    gate_name.reserve(16);
    fp_type phi, theta;

    unsigned prev_time = 0;

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

        continue;
      }

      ss >> time >> gate_name;

      if (!ss) {
        InvalidGateError(provider, k);
        return false;
      }

      if (time < prev_time) {
        IO::errorf("gate time is out of order in %s in line %u.\n",
                   provider.c_str(), k);
        return false;
      }

      prev_time = time;

      unsigned num_qubits = circuit.num_qubits;
      auto& gates = circuit.gates;

      if (time <= maxtime) {
        if (gate_name == "id1") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateId1<fp_type>::Create(time, q0));
        } else if (gate_name == "h") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateHd<fp_type>::Create(time, q0));
        } else if (gate_name == "t") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateT<fp_type>::Create(time, q0));
        } else if (gate_name == "x") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateX<fp_type>::Create(time, q0));
        } else if (gate_name == "y") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateY<fp_type>::Create(time, q0));
        } else if (gate_name == "z") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateZ<fp_type>::Create(time, q0));
        } else if (gate_name == "x_1_2") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateX2<fp_type>::Create(time, q0));
        } else if (gate_name == "y_1_2") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateY2<fp_type>::Create(time, q0));
        } else if (gate_name == "rx") {
          ss >> q0 >> phi;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateRX<fp_type>::Create(time, q0, phi));
        } else if (gate_name == "ry") {
          ss >> q0 >> phi;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateRY<fp_type>::Create(time, q0, phi));
        } else if (gate_name == "rz") {
          ss >> q0 >> phi;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateRZ<fp_type>::Create(time, q0, phi));
        } else if (gate_name == "rxy") {
          ss >> q0 >> theta >> phi;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateRXY<fp_type>::Create(time, q0, theta, phi));
        } else if (gate_name == "hz_1_2") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateHZ2<fp_type>::Create(time, q0));
        } else if (gate_name == "s") {
          ss >> q0;
          if (!ValidateGate(ss, num_qubits, q0, provider, k)) return false;
          gates.push_back(GateS<fp_type>::Create(time, q0));
        } else if (gate_name == "id2") {
          ss >> q0 >> q1;
          if (!ValidateGate(ss, num_qubits, q0, q1, provider, k)) return false;
          gates.push_back(GateId2<fp_type>::Create(time, q0, q1));
        } else if (gate_name == "cz") {
          ss >> q0 >> q1;
          if (!ValidateGate(ss, num_qubits, q0, q1, provider, k)) return false;
          gates.push_back(GateCZ<fp_type>::Create(time, q0, q1));
        } else if (gate_name == "cnot" || gate_name == "cx") {
          ss >> q0 >> q1;
          if (!ValidateGate(ss, num_qubits, q0, q1, provider, k)) return false;
          gates.push_back(GateCNot<fp_type>::Create(time, q0, q1));
        } else if (gate_name == "sw") {
          ss >> q0 >> q1;
          if (!ValidateGate(ss, num_qubits, q0, q1, provider, k)) return false;
          gates.push_back(GateSwap<fp_type>::Create(time, q0, q1));
        } else if (gate_name == "is") {
          ss >> q0 >> q1;
          if (!ValidateGate(ss, num_qubits, q0, q1, provider, k)) return false;
          gates.push_back(GateIS<fp_type>::Create(time, q0, q1));
        } else if (gate_name == "fs") {
          ss >> q0 >> q1 >> theta >> phi;
          if (!ValidateGate(ss, num_qubits, q0, q1, provider, k)) return false;
          gates.push_back(GateFS<fp_type>::Create(time, q0, q1, theta, phi));
        } else if (gate_name == "cp") {
          ss >> q0 >> q1 >> phi;
          if (!ValidateGate(ss, num_qubits, q0, q1, provider, k)) return false;
          gates.push_back(GateCP<fp_type>::Create(time, q0, q1, phi));
        } else if (gate_name == "m") {
          std::vector<unsigned> qubits;
          qubits.reserve(num_qubits);
          do {
            ss >> q0;
            if (ss) {
              qubits.push_back(q0);
            } else {
              InvalidGateError(provider, k);
              return false;
            }
          } while (ss.peek() != std::stringstream::traits_type::eof());
          if (!ValidateGate(ss, num_qubits, qubits, provider, k)) return false;
          gates.push_back(gate::Measurement<GateQSim<fp_type>>::Create(
              time, std::move(qubits)));
        } else {
          InvalidGateError(provider, k);
          return false;
        }
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
                           unsigned num_qubits, unsigned q0,
                           const std::string& provider, unsigned line) {
    if (!ss || ss.peek() != std::stringstream::traits_type::eof()
        || q0 >= num_qubits) {
      InvalidGateError(provider, line);
      return false;
    }

    return true;
  }

  /**
   * Checks formatting for a two-qubit gate parsed from 'ss'.
   * @param ss Input stream containing the gate specification.
   * @param num_qubits Number of qubits, as defined at the start of the file.
   * @param q0 Index of the first affected qubit.
   * @param q1 Index of the second affected qubit.
   * @param provider Circuit source; only used for error reporting.
   * @param line Line number of the parsed gate; only used for error reporting.
   */
  static bool ValidateGate(std::stringstream& ss,
                           unsigned num_qubits, unsigned q0, unsigned q1,
                           const std::string& provider, unsigned line) {
    if (!ss || ss.peek() != std::stringstream::traits_type::eof()
        || q0 >= num_qubits || q1 >= num_qubits || q0 == q1) {
      InvalidGateError(provider, line);
      return false;
    }

    return true;
  }

  /**
   * Checks formatting for a multiqubit gate parsed from 'ss'.
   * @param ss Input stream containing the gate specification.
   * @param num_qubits Number of qubits, as defined at the start of the file.
   * @param qubits Indices of affected qubits.
   * @param provider Circuit source; only used for error reporting.
   * @param line Line number of the parsed gate; only used for error reporting.
   */
  static bool ValidateGate(std::stringstream& ss, unsigned num_qubits,
                           const std::vector<unsigned>& qubits,
                           const std::string& provider, unsigned line) {
    for (auto q : qubits) {
      if (q >= num_qubits) {
        InvalidGateError(provider, line);
        return false;
      }
    }

    return true;
  }
};

}  // namespace qsim

#endif  // CIRCUIT_QSIM_PARSER_H_
