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

#ifndef BITSTRING_H_
#define BITSTRING_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qsim {

using Bitstring = uint64_t;

/**
 * Reads bitstrings (representing initialized or measured states of qubits)
 * from a provided stream object and stores them in a vector.
 * @param num_qubits Number of qubits represented in each bitstring.
 * @param provider Source of bitstrings; only used for error reporting.
 * @param fs The stream to read bitstrings from.
 * @param bitstrings Output vector of bitstrings. On success, this will contain
 *   all bitstrings read in from 'fs'.
 * @return True if reading succeeded; false otherwise.
 */
template <typename IO, typename Stream>
bool BitstringsFromStream(unsigned num_qubits, const std::string& provider,
                          Stream& fs, std::vector<Bitstring>& bitstrings) {
  bitstrings.resize(0);
  bitstrings.reserve(100000);

  // Bitstrings are in text format. One bitstring per line.

  do {
    char buf[128];
    fs.getline(buf, 128);

    if (fs) {
      Bitstring b{0};

      unsigned p = 0;
      while (p < 128 && (buf[p] == '0' || buf[p] == '1')) {
        b |= uint64_t(buf[p] - '0') << p;
        ++p;
      }

      if (p != num_qubits) {
        IO::errorf("wrong bitstring length in %s: "
                   "got %u; should be %u.\n", provider.c_str(), p, num_qubits);
        bitstrings.resize(0);
        return false;
      }

      bitstrings.push_back(b);
    }
  } while (fs);

  return true;
}

/**
 * Reads bitstrings (representing initialized or measured states of qubits)
 * from the given file and stores them in a vector.
 * @param num_qubits Number of qubits represented in each bitstring.
 * @param file The name of the file to read bitstrings from.
 * @param bitstrings Output vector of bitstrings. On success, this will contain
 *   all bitstrings read in from 'file'.
 * @return True if reading succeeded; false otherwise.
 */
template <typename IO>
inline bool BitstringsFromFile(unsigned num_qubits, const std::string& file,
                               std::vector<Bitstring>& bitstrings) {
  auto fs = IO::StreamFromFile(file);

  if (!fs) {
    return false;
  } else {
    bool rc = BitstringsFromStream<IO>(num_qubits, file, fs, bitstrings);
    IO::CloseStream(fs);
    return rc;
  }
}

}  // namespace qsim

#endif  // BITSTRING_H_
