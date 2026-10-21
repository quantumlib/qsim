// Copyright 2026 Google LLC. All Rights Reserved.
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

#ifndef ERROR_H_
#define ERROR_H_

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace qsim {

/** Utilities for reporting generic runtime errors. */
struct Error {
  /**
   * Formats an error message and throws a std::runtime_error. Applies variadic
   * formatting arguments to the message string using std::vformat, and throws
   * a std::runtime_error.
   * @tparam Args Variadic types of the format arguments.
   * @param message Format string explaining the error.
   * @param args Formatting arguments to substitute into the message.
   * @throws std::runtime_error Always throws with the formatted error string.
   */
  template <typename... Args>
  static void Throw(
      const std::string_view message, Args&&... args) {
    auto error_message = std::vformat(
        std::string("error: ") + std::string(message),
        std::make_format_args(args...));

    throw std::runtime_error(error_message);
  }
};

}  // namespace qsim

#endif  // ERROR_H_
