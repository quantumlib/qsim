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

#ifndef CLASSICAL_CONTROL_UTIL_H_
#define CLASSICAL_CONTROL_UTIL_H_

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "classical_control_expr.h"
#include "error.h"

namespace qsim::cc {

/**
 * Parses a string representation of an integer in decimal, hex, or binary
 * format. Supports decimal (e.g., `"123"`), hexadecimal (`"0x1a"` or
 * `"0X1a"`), and binary (`"0b0101"` or `"0B0101"`) prefixes.
 * @param s String view containing the numerical text to convert.
 * @return Converted integer as an `int64_t`.
 * @throws std::runtime_error If parsing fails or trailing non-numeric
 *   characters exist.
 */
inline int64_t ToInt(std::string_view s) {
  int val;
  unsigned base = 10;

  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    base = 16;
  } else if (s.size() > 2 && s[0] == '0' && (s[1] == 'b' || s[1] == 'B')) {
    base = 2;
  }

  auto start = base == 10 ? s.data() : s.data() + 2;

  auto [p, ec] = std::from_chars(start, s.data() + s.size(), val, base);

  if (ec != std::errc{} || p != s.data() + s.size()) {
    Error::Throw("cannot convert string to int.");
  }

  return val;
}

/**
 * Parses a string representation of a floating-point number.
 * @param s String view containing standard decimal or exponential
 *   floating-point text.
 * @return Converted floating-point value as a `double`.
 * @throws std::runtime_error If parsing fails or trailing non-numeric
 *   characters exist.
 */
inline double ToFloat(std::string_view s) {
  double val;

#if defined(_MSC_VER) || (defined(__GLIBCXX__) && __GLIBCXX__ >= 20210427) || \
    (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 170000)
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), val);

  if (ec != std::errc{} || p != s.data() + s.size()) {
    Error::Throw("cannot convert string to float.");
  }
#else
  char* end;
  std::string str(s);
  val = std::strtod(str.c_str(), &end);
  if (end == str.c_str() || errno == ERANGE) {
    Error::Throw("cannot convert string to float.");
  }
#endif

  return val;
}

namespace detail {

/**
 * Computes a compile-time 64-bit FNV-1a hash over a byte string.
 * @param s Pointer to char array.
 * @param len Length of char sequence.
 * @return 64-bit FNV-1a unsigned hash value.
 */
inline constexpr unsigned fnv1a_hash(const char* s, std::size_t len) {
  constexpr uint64_t offset_basis = 0xcbf29ce484222325;
  constexpr uint64_t prime = 0x100000001b3;

  uint64_t hash = offset_basis;
  for (std::size_t i = 0; i < len; ++i) {
    hash ^= static_cast<uint8_t>(s[i]);
    hash *= prime;
  }
  return hash;
}

}  // namespace detail

/**
 * Computes a compile-time FNV-1a string hash.
 * @param s Input string view.
 * @return Calculated hash value.
 */
inline constexpr uint64_t Hash(std::string_view s) {
  return detail::fnv1a_hash(s.data(), s.size());
}

/**
 * User-defined literal operator for compile-time string hashing.
 * Example usage: `"my_symbol"_hash`.
 */
inline constexpr uint64_t operator""_hash(const char* s, std::size_t len) {
  return detail::fnv1a_hash(s, len);
}

/**
 * Packs a 1- or 2-character operator string into a 16-bit integer
 * representation. Allows efficient switch-case matching over single- or
 * double-character operators (e.g., `"+"`, `"=="`, `">="`).
 * @param s Operator string view (must be length 1 or 2).
 * @return Packed unsigned integer hash or `0` for invalid lengths.
 */
inline constexpr unsigned short OpHash(std::string_view s) {
  switch (s.size()) {
  case 1:
    return s[0];
  case 2:
    return (s[1] << 8) | s[0];
  default:
    return 0;
  }
}

/**
 * User-defined literal operator for packing operator strings into
 * 16-bit hashes. Example usage: `"=="__ophash`.
 */
inline constexpr unsigned short operator""_ophash(
    const char* s, std::size_t len) {
  switch (len) {
  case 1:
    return s[0];
  case 2:
    return (s[1] << 8) | s[0];
  default:
    return 0;
  }
}

/**
 * Reads the entire contents of a file on disk into an in-memory std::string
 * buffer.
 * @param path File system path string.
 * @return File content string.
 * @throws std::runtime_error If the file cannot be opened or read.
 */
inline std::string ReadFile(std::string_view path) {
  std::ifstream file(std::filesystem::path(path),
                     std::ios::in | std::ios::binary | std::ios::ate);

  if (!file) {
    Error::Throw("failed to open {}", path);
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer;
  buffer.resize(static_cast<size_t>(size));

  if (!file.read(buffer.data(), size)) {
    Error::Throw("error reading data from {}", path);
  }

  return buffer;
}

/**
 * Recursive variadic helper template for formatting expressions into a string.
 * @tparam Count Remaining number of expressions to evaluate.
 * @tparam Vals Accumulated evaluated expression types (`TInt` or `TFloat`).
 */
template <unsigned Count, typename... Vals>
struct ExprToStr {
  /** Formats expressions using an explicit C++ std::format string. */
  template <typename SymTable, typename Expr>
  static std::string Run1(const SymTable& symtab, std::string_view fmt,
                          const std::vector<Expr>& es, Vals&&... vs) {
    if constexpr (Count == 0) {
      return std::vformat(fmt, std::make_format_args(vs...));
    } else {
      auto k = es.size() - Count;

      if (IsConvertibleToInt(symtab, es[k])) {
        return ExprToStr<Count - 1, Vals..., TInt>::Run1(
            symtab, fmt, es, std::move(vs)..., EvalIntExpr(symtab, es[k]));
      } else {
        return ExprToStr<Count - 1, Vals..., TFloat>::Run1(
            symtab, fmt, es, std::move(vs)..., EvalExpr(symtab, es[k]));
      }
    }
  }

  /**
   * Formats expressions sequentially with auto-generated space delimiters
   * (`"{}"`).
   */
  template <typename SymTable, typename Expr>
  static std::string Run2(const SymTable& symtab, std::string&& fmt,
                          const std::vector<Expr>& es, Vals&&... vs) {
    if constexpr (Count == 0) {
      return std::vformat(fmt, std::make_format_args(vs...));
    } else {
      auto k = es.size() - Count;
      if constexpr (Count > 1) {
        fmt += "{} ";
      } else {
        fmt += "{}";
      }

      if (IsConvertibleToInt(symtab, es[k])) {
        return ExprToStr<Count - 1, Vals..., TInt>::Run2(
            symtab, std::move(fmt), es, std::move(vs)...,
            EvalIntExpr(symtab, es[k]));
      } else {
        return ExprToStr<Count - 1, Vals..., TFloat>::Run2(
            symtab, std::move(fmt), es, std::move(vs)...,
            EvalExpr(symtab, es[k]));
      }
    }
  }
};

/**
 * Evaluates up to 4 expressions and prints the formatted string to standard
 * stdout (`puts`).
 * @tparam SymTable Active symbol table type.
 * @tparam Expr Expression variant type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param fmt Optional format string. If empty, expressions are printed
 *   space-separated.
 * @param es Vector of expressions (up to 4 items).
  */
template <typename SymTable, typename Expr>
inline void PrintExpressions(
    const SymTable& symtab, std::string_view fmt, const std::vector<Expr>& es) {
  if (!fmt.empty()) {
    switch (es.size()) {
    case 0:
      puts(ExprToStr<0>::Run1(symtab, fmt, es).c_str());
      break;
    case 1:
      puts(ExprToStr<1>::Run1(symtab, fmt, es).c_str());
      break;
    case 2:
      puts(ExprToStr<2>::Run1(symtab, fmt, es).c_str());
      break;
    case 3:
      puts(ExprToStr<3>::Run1(symtab, fmt, es).c_str());
      break;
    case 4:
      puts(ExprToStr<4>::Run1(symtab, fmt, es).c_str());
      break;
    }
  } else {
    switch (es.size()) {
    case 0:
      puts(ExprToStr<0>::Run2(symtab, std::string{}, es).c_str());
      break;
    case 1:
      puts(ExprToStr<1>::Run2(symtab, std::string{}, es).c_str());
      break;
    case 2:
      puts(ExprToStr<2>::Run2(symtab, std::string{}, es).c_str());
      break;
    case 3:
      puts(ExprToStr<3>::Run2(symtab, std::string{}, es).c_str());
      break;
    case 4:
      puts(ExprToStr<4>::Run2(symtab, std::string{}, es).c_str());
      break;
    }
  }
}

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_UTIL_H_
