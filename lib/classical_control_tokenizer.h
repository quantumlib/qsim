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

#ifndef CLASSICAL_CONTROL_TOKENIZER_H_
#define CLASSICAL_CONTROL_TOKENIZER_H_

#include <array>
#include <cctype>
#include <deque>
#include <string_view>
#include <utility>

#include "classical_control_util.h"

namespace qsim::cc {

/** Represents a token produced by the \c Tokenizer. */
struct Token {
  /** Defines the syntactic classification of a token. */
  enum TokenKind {
    /** Alphanumeric identifiers (e.g., `var_1`). */
    kIdentifier,
    /** Decimal (`42`), hexadecimal (`0x1a`), or binary (`0b01`) integers. */
    kInteger,
    /** Floating-point numbers (e.g., `301.34`, `1.2e-5`). */
    kFloat,
    /** Single (`'...'`) or double (`"..."`) quoted string literals. */
    kString,
    /** Single and two-character operators (e.g., `+`, `==`, `<<`). */
    kOperator,
    /** Parentheses `()` and square brackets `[]`. */
    kBracket,
    /** Semicolons `;` or newline characters `\n`. */
    kDelimiter,
    /** Indicates end-of-string or file termination. */
    kEndOfFile,
    /** Malformed token or unrecognized character stream. */
    kInvalid,
  };

  TokenKind kind;
  unsigned lc;           // Line and column.
  std::string_view val;
};

/** A stateful lexical analyzer for tokenizing source text. */
struct Tokenizer {
  /**
   * Constructs a tokenizer for the provided string view.
   * @param content The source code buffer to tokenize.
   */
  explicit Tokenizer(std::string_view content) : content(content) {}

  /** Resets the tokenizer back to the start of the source buffer. */
  void Restart() {
    buffer.clear();
    last = Token{};
    pos = 0;
    pos0 = 0;
    line = 0;
  }

  /**
   * Inspects upcoming tokens without advancing the current reader position.
   * @param distance Zero-based index of lookahead (0 = next token to be
   *   consumed).
   * @return The token at the lookahead position.
   */
  Token Peek(std::size_t distance = 0) {
    while (buffer.size() <= distance) {
      Token t = Next();
      buffer.push_back(t);
      if (t.kind == Token::kEndOfFile) {
        break;
      }
    }

    if (distance >= buffer.size()) {
      return {Token::kEndOfFile, {}};
    }

    return buffer[distance];
  }

  /**
   * Advances the stream and returns the next consumed token.
   * @return The next \c Token in the stream.
   */
  Token operator()() {
    if (!buffer.empty()) {
      Token t = buffer.front();
      buffer.pop_front();
      last = t;
      return t;
    }

    Token t = Next();
    last = t;
    return t;
  }

  /**
   * Retrieves the most recently consumed token.
   * @return Reference to the current token.
   */
  const Token& Current() {
    return last;
  }

  /**
   * Packs zero-based line and column numbers into a single 32-bit integer.
   * @param line Zero-based line index (upper 20 bits).
   * @param col Zero-based column index (lower 12 bits, max 4095).
   * @return The packed 32-bit unsigned integer.
   */
  static unsigned PackLineAndColumn(unsigned line, unsigned col) {
    return (line << 12) | (col & ((1u << 12) - 1));
  }

  /**
   * Decodes a packed line-and-column integer into a pair.
   * @param lc Packed integer containing line and column information.
   * @return A pair formatted as `{line, column}`.
   */
  static std::pair<unsigned, unsigned> UnpackLineAndColumn(unsigned lc) {
    return {lc >> 12, lc & ((1u << 12) - 1)};
  }

private:
  std::string_view content;
  std::deque<Token> buffer;
  Token last;
  std::size_t pos = 0;
  std::size_t pos0 = 0;
  unsigned line = 0;

  static bool IsIdentifierStart(char c) {
    return std::isalpha(c) || c == '_';
  }

  static bool IsIdentifierBody(char c) {
    return std::isalnum(c) || c == '_';
  }

  char PeekChar(std::size_t offset = 0) const {
    return pos + offset < content.size() ? content[pos + offset] : '\0';
  }

  void ConsumeDigits() {
    while (pos < content.size() && std::isdigit(content[pos])) {
      ++pos;
    }
  }

  void ConsumeHexDigits() {
    while (pos < content.size() && std::isxdigit(content[pos])) {
      ++pos;
    }
  }

  void ConsumeBinDigits() {
    while (pos < content.size() && isbdigit(content[pos])) {
      ++pos;
    }
  }

  static bool isbdigit(char c) {
    return c == '0' || c == '1';
  }

  Token Next() {
    while (pos < content.size()) {
      char ch = content[pos];

      if (ch != ' ' && ch != '\t' && ch != '\r') {
        break;
      }

      ++pos;
    }

    if (pos == content.size()) {
      unsigned lc = PackLineAndColumn(line, content.size() - pos0);
      return {Token::kEndOfFile, lc, {}};
    }

    if (content[pos] == '#') {
      while (pos < content.size() && content[pos] != '\n') {
        ++pos;
      }

      if (pos == content.size()) {
        unsigned lc = PackLineAndColumn(line, content.size() - pos0);
        return {Token::kEndOfFile, lc, {}};
      }
    }

    char cur = content[pos];
    std::size_t start = pos;

    unsigned lc = PackLineAndColumn(line, pos - pos0);

    if (cur == ';') {
      ++pos;
      return {Token::kDelimiter, lc, content.substr(pos - 1, 1)};
    }

    if (cur == '\n') {
      ++line;
      ++pos;
      pos0 = pos;
      return {Token::kDelimiter, lc, content.substr(pos - 1, 1)};
    }

    if (cur == '\'') {
      while (++pos < content.size() && content[pos] != '\n'
             && content[pos] != '\'');

      if (pos == content.size() || content[pos] == '\n') {
        return {Token::kInvalid, lc, content.substr(start, 1)};
      }

      ++pos;

      return
          {Token::kString, lc, content.substr(start + 1, pos - start - 2)};
    }

    if (cur == '\"') {
      while (++pos < content.size() && content[pos] != '\n'
             && content[pos] != '\"');

      if (pos == content.size() || content[pos] == '\n') {
        return {Token::kInvalid, lc, content.substr(start, 1)};
      }

      ++pos;

      return
          {Token::kString, lc, content.substr(start + 1, pos - start - 2)};
    }

    if (IsIdentifierStart(cur)) {
      while (pos < content.size() && IsIdentifierBody(content[pos])) {
        ++pos;
      }
      return {Token::kIdentifier, lc, content.substr(start, pos - start)};
    }

    if (std::isdigit(cur)) {
      char next = PeekChar(1);

      if (cur == '0' && (next == 'x' || next == 'X')) {
        pos += 2;
        bool is_valid = std::isxdigit(PeekChar(0));

        ConsumeHexDigits();
        std::string_view val = content.substr(start, pos - start);

        return {is_valid ? Token::kInteger : Token::kInvalid, lc, val};
      } else if (cur == '0' && (next == 'b' || next == 'B')) {
        pos += 2;
        bool is_valid = isbdigit(PeekChar(0));

        ConsumeBinDigits();
        std::string_view val = content.substr(start, pos - start);

        return {is_valid ? Token::kInteger : Token::kInvalid, lc, val};
      } else {
        bool is_float = false;

        ConsumeDigits();

        if (PeekChar(0) == '.' && std::isdigit(PeekChar(1))) {
          ++pos;
          is_float = true;
          ConsumeDigits();
        }

        char next = PeekChar(0);
        if (next == 'e' || next == 'E') {
          ++pos;
          next = PeekChar(0);

          if (next == '+' || next == '-') {
            ++pos;
            next = PeekChar(0);
          }

          if (std::isdigit(next)) {
            is_float = true;
            ConsumeDigits();
          } else {
            std::string_view val = content.substr(start, pos - start);
            return {Token::kInvalid, lc, val};
          }
        }

        std::string_view val = content.substr(start, pos - start);
        return {is_float ? Token::kFloat : Token::kInteger, lc, val};
      }
    }

    if (pos + 1 < content.size()) {
      std::string_view val = content.substr(pos, 2);

      switch (OpHash(val)) {
      case "||"_ophash:
      case "^^"_ophash:
      case "&&"_ophash:
      case "=="_ophash:
      case "!="_ophash:
      case "<="_ophash:
      case ">="_ophash:
      case "**"_ophash:
      case "<<"_ophash:
      case ">>"_ophash:
        pos += 2;
        return {Token::kOperator, lc, val};
      }
    }

    static constexpr std::array<bool, 256> is_operator = []() {
      std::array<bool, 256> array = {false};
      for (char c : "-+*/%=<>!~|^&") {
        array[static_cast<unsigned char>(c)] = true;
      }
      return array;
    }();

    if (is_operator[cur]) {
      return {Token::kOperator, lc, content.substr(pos++, 1)};
    }

    static constexpr std::array<bool, 256> is_bracket = []() {
      std::array<bool, 256> array = {false};
      for (char c : "()[]") {
        array[static_cast<unsigned char>(c)] = true;
      }
      return array;
    }();

    if (is_bracket[cur]) {
      return {Token::kBracket, lc, content.substr(pos++, 1)};
    }

    return {Token::kInvalid, lc, content.substr(pos++, 1)};
  }
};

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_TOKENIZER_H_
