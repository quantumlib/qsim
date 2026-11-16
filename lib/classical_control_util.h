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
  int64_t val;
  unsigned base = 10;

  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    base = 16;
  } else if (s.size() > 2 && s[0] == '0' && (s[1] == 'b' || s[1] == 'B')) {
    base = 2;
  }

  auto start = base == 10 ? s.data() : s.data() + 2;

  auto [p, ec] = std::from_chars(start, s.data() + s.size(), val, base);

  if (ec != std::errc{} || p != s.data() + s.size()) {
    Error::Throw("cannot convert string to int: {}", s);
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
    (defined(_LIBCPP_VERSION) && _LIBCPP_VERSION >= 200000)
  auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), val);

  if (ec != std::errc{} || p != s.data() + s.size()) {
    Error::Throw("cannot convert string to float: {}", s);
  }
#else
  char* end;
  std::string str(s);
  val = std::strtod(str.c_str(), &end);
  if (end == str.c_str() || errno == ERANGE) {
    Error::Throw("cannot convert string to float: {}", s);
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
inline constexpr uint64_t fnv1a_hash(const char* s, std::size_t len) {
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
  return OpHash({s, len});
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
 * Formats evaluated DSL arithmetic expressions into strings using C++20
 * format specifiers. `ExprToStr` evaluates expressions sequentially against
 * a symbol table and formats each result via `std::vformat_to`.
 *
 * ### Supported Format Specifiers
 * Standard C++20 single-argument replacement fields (`{}` and `{:spec}`) are
 * supported, including fill/alignment (`{:*^10}`), signs (`{:+d}`), alternate
 * forms (`{:#x}`, `{:#b}`), zero-padding (`{:08d}`), static width and
 * precision (`{:15.8g}`, `{:.2f}`), and escaped literal braces (`{{` and `}}`).
 *
 * ### Format Specifier Limitations
 * Because `fmt` is sliced sequentially one placeholder at a time and each
 * expression is formatted in isolation:
 *   1. **No positional argument indices:** Explicit indices such as `{0}` or
 *      `{1:6x}` are rejected with `std::format_error`. Placeholders are
 *      matched strictly in 1-to-1 sequential order with `es`.
 *   2. **No nested `{}` (dynamic width or precision):** Specifiers containing
 *      nested replacement fields such as `{:>{}}` or `{:{}.{}f}` are not
 *      supported and throw `std::format_error`.
 *   3. **Exact placeholder-to-expression count:** Supplying more or fewer
 *      expressions than unescaped `{...}` placeholders throws
 *      `std::format_error`. Note that passing `fmt = ""` with a non-empty
 *      `es` to `Format(symtab, fmt, es)` throws; use the two-argument overload
 *      `Format(symtab, es)` (or `PrintExpressions`) for default formatting.
 *   4. **No implicit int-to-float specifier promotion:** Expressions for which
 *      `IsConvertibleToInt` returns `true` are formatted strictly as integers
 *      (floating-point specifiers like `{:f}`, `{:g}`, `{:e}`, or `{:.2}`
 *      throw `std::format_error`). Conversely, non-integer expressions reject
 *      integer-only specifiers (`{:d}`, `{:x}`, `{:b}`, `{:o}`, `{:c}`).
 */
struct ExprToStr {
  /**
   * Evaluates and formats `es` according to the format string `fmt`.
   * @tparam SymTable Active symbol table type.
   * @tparam Expr Expression variant type.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param fmt C++20 format string containing one unescaped `{...}`
   *   replacement field per element in `es`.
   * @param es Vector of expressions to evaluate and format.
   * @return The formatted output string with `{{` and `}}` unescaped.
   * @throws std::format_error If `fmt` has unmatched braces, positional
   *         indices, nested `{}` specifiers, mismatched placeholder count, or
   *         a specifier incompatible with the evaluated expression's type.
   */
  template <typename SymTable, typename Expr>
  static std::string Format(const SymTable& symtab,
                            std::string_view fmt, const std::vector<Expr>& es) {
    std::string s;
    s.reserve(fmt.size() + 32 * es.size());

    for (std::size_t i = 0; i < es.size(); ++i) {
      std::string_view sub_fmt = NextSubFormat(fmt);

      if (IsConvertibleToInt(symtab, es[i])) {
        auto val = EvalIntExpr(symtab, es[i]);
        std::vformat_to(
            std::back_inserter(s), sub_fmt, std::make_format_args(val));
      } else {
        auto val = EvalFloatExpr(symtab, es[i]);
        std::vformat_to(
            std::back_inserter(s), sub_fmt, std::make_format_args(val));
      }
    }

    std::vformat_to(std::back_inserter(s), fmt, std::make_format_args());

    return s;
  }

  /**
   * Evaluates and formats `es` using default `{}` formatting separated
   * by single spaces. Produces `""` when `es` is empty, and avoids a trailing
   * space after the final element.
   * @tparam SymTable Active symbol table type.
   * @tparam Expr Expression variant type.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param es Vector of expressions to evaluate and format.
   * @return Space-delimited string of formatted expression values.
   */
  template <typename SymTable, typename Expr>
  static std::string Format(
      const SymTable& symtab, const std::vector<Expr>& es) {
    std::string s;
    s.reserve(35 * es.size());

    for (std::size_t i = 0; i < es.size(); ++i) {
      std::string_view fmt = i < es.size() - 1 ? "{} " : "{}";

      if (IsConvertibleToInt(symtab, es[i])) {
        auto val = EvalIntExpr(symtab, es[i]);
        std::vformat_to(std::back_inserter(s), fmt, std::make_format_args(val));
      } else {
        auto val = EvalFloatExpr(symtab, es[i]);
        std::vformat_to(std::back_inserter(s), fmt, std::make_format_args(val));
      }
    }

    return s;
  }

 protected:
  /**
   * Extracts the prefix of `fmt` through the closing `}` of the next
   * unescaped replacement field, advancing `fmt` past that slice. Escaped
   * braces (`{{` and `}}`) preceding the placeholder are preserved in
   * the returned slice so `std::vformat_to` can unescape them.
   *
   * @param[in,out] fmt Remaining format string view; mutated in-place to remove
   *                    the returned prefix on success.
   * @return            View covering any leading literal text up to and
   *                    including the closing `}` of the first active `{...}`.
   * @throws std::format_error If an unmatched `{` or `}` is encountered, if the
   *         placeholder starts with a positional digit (`{0}`..`{9}`), or if no
   *         unescaped placeholder remains in `fmt`.
   */
  static std::string_view NextSubFormat(std::string_view& fmt) {
    std::size_t n = fmt.size();

    for (std::size_t pos = 0; pos < n; ++pos) {
      if (fmt[pos] == '{') {
        if (pos + 1 < n && fmt[pos + 1] == '{') {
          ++pos; // Skip escaped '{{'
          continue;
        }

        if (pos + 1 < n && fmt[pos + 1] >= '0' && fmt[pos + 1] <= '9') {
          FormatError::Throw("positional argument indices are not supported");
        }

        std::size_t close = fmt.find('}', pos + 1);
        if (close == std::string_view::npos) {
          FormatError::Throw("unmatched '{' in format string");
        }

        std::string_view sub_fmt = fmt.substr(0, close + 1);
        fmt.remove_prefix(close + 1);
        return sub_fmt;
      }

      if (fmt[pos] == '}') {
        if (pos + 1 < n && fmt[pos + 1] == '}') {
          ++pos; // Skip escaped '}}'
          continue;
        }
        FormatError::Throw("unmatched '}' in format string");
      }
    }

    FormatError::Throw("too many expressions for format string (missing '{}')");
  }
};

/**
 * @brief Formats `es` and writes the result followed by a newline to `stdout`.
 *
 * If `fmt` is non-empty, formats according to `ExprToStr::Format(symtab, fmt, es)`.
 * If `fmt` is empty (`""`), falls back to default space-separated formatting
 * via `ExprToStr::Format(symtab, es)`.
 * @tparam SymTable Active symbol table type.
 * @tparam Expr Expression variant type.
 * @param symtab Symbol table instance used for symbol lookups.
 * @param fmt Optional C++20 format string. If empty, expressions are printed
 *   space-separated.
 * @param es Vector of expressions to print.
 * @throws std::format_error Propagated from `ExprToStr::Format` on invalid
 *         format strings or specifier/type mismatches when `fmt` is non-empty.
 */
template <typename SymTable, typename Expr>
inline void PrintExpressions(
    const SymTable& symtab, std::string_view fmt, const std::vector<Expr>& es) {
  if (!fmt.empty()) {
    puts(ExprToStr::Format(symtab, fmt, es).c_str());
  } else {
    puts(ExprToStr::Format(symtab, es).c_str());
  }
}

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_UTIL_H_
