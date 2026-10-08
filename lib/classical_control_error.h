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

#ifndef CLASSICAL_CONTROL_ERROR_H_
#define CLASSICAL_CONTROL_ERROR_H_

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include "classical_control_tokenizer.h"

namespace qsim::cc {

/** Utilities and custom exception types for expression parsing errors. */
struct ParserError {
  /** Custom exception thrown when a syntax error is detected during parsing. */
  class syntax_error : public std::exception {
   private:
    std::string message;

   public:
    /**
     * Constructs a syntax_error with the provided explanatory string.
     * @param message Explanatory string describing the syntax error (moved).
     */
    explicit syntax_error(std::string&& message)
        : message(std::move(message)) {}

    /**
     * Returns a pointer to the explanatory string.
     * @return Null-terminated C string describing the syntax error.
     */
    const char* what() const noexcept override {
      return message.c_str();
    }
  };

  /**
   * Unpacks source coordinates, formats an error message, and throws
   * a syntax_error. Converts the packed line and column bits into 1-indexed
   * numbers, applies variadic formatting arguments to the message string
   * using std::vformat, and throws a syntax_error.
   * @tparam Args Variadic types of the format arguments.
   * @param message Format string explaining the error.
   * @param lc Packed line and column value.
   * @param args Formatting arguments to substitute into the message.
   * @throws syntax_error Always throws with the formatted error string and
   *   location.
   */
  template <typename... Args>
  static void Throw(
      std::string_view message, unsigned lc, Args&&... args) {
    auto [line, col] = Tokenizer::UnpackLineAndColumn(lc);

    ++line; ++col;

    auto error_message = std::vformat(
        std::string("expression syntax error: ") + std::string(message) +
        " at {}:{}", std::make_format_args(args..., line, col));

    throw syntax_error(std::move(error_message));
  }
};

/**
 * Utilities for reporting expression runtime errors.
 */
struct RuntimeError {
  /**
   * Unpacks source coordinates, formats an error message, and throws
   * a std::runtime_error. Converts the packed line and column bits into
   * 1-indexed numbers, applies variadic formatting arguments to the message
   * string using std::vformat, and throws a std::runtime_error.
   * @tparam Args Variadic types of the format arguments.
   * @param message Format string explaining the error.
   * @param lc Packed line and column value.
   * @param args Formatting arguments to substitute into the message.
   * @throws std::runtime_error Always throws with the formatted error string
   *   and location.
   */
  template <typename... Args>
  static void Throw(
      const std::string_view message, unsigned lc, Args&&... args) {
    auto [line, col] = Tokenizer::UnpackLineAndColumn(lc);

    ++line; ++col;

    auto error_message = std::vformat(
        std::string("expression runtime error: ") + std::string(message) +
        " at {}:{}", std::make_format_args(args..., line, col));

    throw std::runtime_error(error_message);
  }
};

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_ERROR_H_
