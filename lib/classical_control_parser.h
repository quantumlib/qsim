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

#ifndef CLASSICAL_CONTROL_PARSER_H_
#define CLASSICAL_CONTROL_PARSER_H_

#include <functional>
#include <string_view>
#include <utility>

#include "classical_control_expr.h"
#include "classical_control_symbol.h"
#include "classical_control_symtab.h"
#include "classical_control_tokenizer.h"
#include "classical_control_util.h"
#include "error.h"

namespace qsim::cc {

/**
 * Parses numerical and conditional expressions from a Tokenizer stream.
 * @tparam IndexParser Parser struct used to parse vector/measurement bracket
 *   indices.
 * @tparam ParserError Error policy for syntax errors.
 * @tparam RuntimeError Error policy for runtime errors. Used in closures
 *   when division by zero is detected.
 * @tparam int_only If true, restricts parsing strictly to integer/boolean
 *   expressions. Encountering floating-point literals, float symbols, or `**`
 *   throws an error.
 */
template <typename IndexParser, typename ParserError, typename RuntimeError,
          bool int_only = false>
struct ExprParser {
 private:
  using PE = ParserError;
  using RE = RuntimeError;

  static constexpr unsigned kOrPrec = 1;
  static constexpr unsigned kXorPrec = 2;
  static constexpr unsigned kAndPrec = 3;
  static constexpr unsigned kBitOrPrec = 4;
  static constexpr unsigned kBitXorPrec = 5;
  static constexpr unsigned kBitAndPrec = 6;
  static constexpr unsigned kEqPrec = 7;
  static constexpr unsigned kNotEqPrec = 7;
  static constexpr unsigned kLessPrec = 8;
  static constexpr unsigned kLessEqPrec = 8;
  static constexpr unsigned kGreaterPrec = 8;
  static constexpr unsigned kGreaterEqPrec = 8;
  static constexpr unsigned kBitLeftShiftPrec = 9;
  static constexpr unsigned kBitRightShiftPrec = 9;
  static constexpr unsigned kAddPrec = 10;
  static constexpr unsigned kSubPrec = 10;
  static constexpr unsigned kMulPrec = 11;
  static constexpr unsigned kDivPrec = 11;
  static constexpr unsigned kModPrec = 11;
  static constexpr unsigned kPowPrec = 12;

  static constexpr auto empty = [](auto&&...) {};

 public:
  /**
   * Parses an expression starting from the current tokenizer position.
   * Stops automatically before the first non-operator lookahead token (such
   * as a delimiter, an unmatched `)`, a literal, or the `(` starting a
   * subsequent expression), leaving that token unconsumed.
   * If the expression involves only compile-time constants, the expression
   * is folded directly into a scalar literal (`VInt`, `VFloat`, or `VBool`).
   * @tparam F Callback function type with signature
   *   `void(const Symbol::Mea&, std::string_view)`.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param tok Lexical analyzer stream positioned at the start of
   *   the expression.
   * @param f Optional callback invoked whenever a measurement (`Mea`) symbol
   *   is parsed.
   * @return Parsed `Expr` variant containing a constant scalar value or
   *   an executable function closure.
   * @throws syntax_error on syntax errors, unexpected tokens, or type
   *   mismatches.
   */
  template <typename F = decltype(empty)>
  static Expr Run(
      const SymTable& symtab, Tokenizer& tok, F&& f = std::forward<F>(empty)) {
    return RunWithPrecedence(symtab, tok, 0, f);
  }

 private:
  static bool IsInt(const Expr& e) {
    return std::holds_alternative<VInt>(e);
  }

  static bool IsSymbol(const Expr& e) {
    return std::holds_alternative<SymbolKey>(e);
  }

  static bool IsSymbolInd(const Expr& e) {
    return std::holds_alternative<SymbolKeyInd>(e);
  }

  static unsigned GetPrecedence(const Token& t) {
    switch (OpHash(t.val)) {
    case "||"_ophash:
      return kOrPrec;
    case "^^"_ophash:
      return kXorPrec;
    case "&&"_ophash:
      return kAndPrec;
    case "|"_ophash:
      return kBitOrPrec;
    case "^"_ophash:
      return kBitXorPrec;
    case "&"_ophash:
      return kBitAndPrec;
    case "=="_ophash:
      return kEqPrec;
    case "!="_ophash:
      return kNotEqPrec;
    case "<"_ophash:
      return kLessPrec;
    case "<="_ophash:
      return kLessEqPrec;
    case ">"_ophash:
      return kGreaterPrec;
    case ">="_ophash:
      return kGreaterEqPrec;
    case "<<"_ophash:
      return kBitLeftShiftPrec;
    case ">>"_ophash:
      return kBitRightShiftPrec;
    case "+"_ophash:
      return kAddPrec;
    case "-"_ophash:
      return kSubPrec;
    case "*"_ophash:
      return kMulPrec;
    case "/"_ophash:
      return kDivPrec;
    case "%"_ophash:
      return kModPrec;
    case "**"_ophash:
      return kPowPrec;
    default:
      PE::Throw("unknown operator", t.lc);
      return 0;
    };
  }

  /**
   * Parses a sub-expression starting from the current tokenizer position
   * using operator-precedence (Pratt) parsing.
   * Consumes operators and their right-hand operands as long as the lookahead
   * token is an operator with precedence strictly greater than `precedence`.
   * If the sub-expression involves only compile-time constants, it is folded
   * directly into a scalar literal (`VInt`, `VFloat`, or `VBool`).
   * @tparam F Callback function type with signature
   *   `void(const Symbol::Mea&, std::string_view)`.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param tok Lexical analyzer stream positioned at the start of
   *   the sub-expression.
   * @param precedence Minimum precedence floor; parsing stops before any
   *   operator whose precedence is less than or equal to this value.
   * @param f Callback invoked whenever a measurement (`Mea`) symbol
   *   is parsed.
   * @return Parsed `Expr` variant containing a constant scalar value or
   *   an executable function closure.
   * @throws syntax_error on syntax errors, unexpected tokens, or type
   *   mismatches.
   */
  template <typename F>
  static Expr RunWithPrecedence(
      const SymTable& symtab, Tokenizer& tok, unsigned precedence, F&& f) {
    Expr l = GetLeft(symtab, tok, f);

    while (1) {
      Token n = tok.Peek();

      if (n.kind != Token::kOperator) {
        break;
      }

      if (precedence >= GetPrecedence(n)) {
        break;
      }

      l = GetRight(symtab, tok, l, f);
    }

    return l;
  }

  /**
   * Parses a primary or prefix expression starting from the current tokenizer
   * position, such as a literal, symbol reference, unary prefix operation,
   * or parenthesized sub-expression.
   * If the parsed construct involves only compile-time constants, it is
   * folded directly into a scalar literal (`VInt`, `VFloat`, or `VBool`).
   * @tparam F Callback function type with signature
   *   `void(const Symbol::Mea&, std::string_view)`.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param tok Lexical analyzer stream positioned at the start of
   *   the primary or prefix token.
   * @param f Callback invoked whenever a measurement (`Mea`) symbol
   *   is parsed.
   * @return Parsed `Expr` variant containing a constant scalar value or
   *   an executable function closure.
   * @throws syntax_error on syntax errors, unexpected tokens, or type
   *   mismatches.
   */
  template <typename F>
  static Expr GetLeft(const SymTable& symtab, Tokenizer& tok, F&& f) {
    Token t = tok();

    switch (OpHash(t.val)) {
    case "!"_ophash:
      {
        Token n = tok.Peek();

        Expr l = GetLeft(symtab, tok, f);

        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expression is not convertible to 'bool'", n.lc);
        }

        if (IsConstExpr(symtab, l)) {
          return VBool{!EvalCondExpr(symtab, l)};
        } else {
          return EvalBoolFn{[l](const SymTable& symtab) {
            return !EvalCondExpr(symtab, l);
          }};
        }
      }
    case "~"_ophash:
      {
        Expr l = GetLeft(symtab, tok, f);

        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected an integer expression", t.lc);
        }

        if (IsConstExpr(symtab, l)) {
          return VInt{~EvalIntExpr(symtab, l)};
        } else {
          return EvalIntFn{[l](const SymTable& symtab) {
            auto val = ~EvalIntExpr(symtab, l);
            val = TruncateMea(symtab, l, val);
            return VInt{val};
          }};
        }
      }
    case "-"_ophash:
      {
        Expr l = GetLeft(symtab, tok, f);

        if (IsConstExpr(symtab, l)) {
          if (IsConvertibleToInt(symtab, l)) {
            return VInt{-EvalIntExpr(symtab, l)};
          } else {
            return VFloat{-EvalFloatExpr(symtab, l)};
          }
        } else {
          if (IsConvertibleToInt(symtab, l)) {
            return EvalIntFn{[l](const SymTable& symtab) {
              return -EvalIntExpr(symtab, l);
            }};
          } else {
            return EvalFloatFn{[l](const SymTable& symtab) {
              return -EvalFloatExpr(symtab, l);
            }};
          }
        }
      }
    case "("_ophash:
      {
        Expr l = RunWithPrecedence(symtab, tok, 0, f);

        Token t = tok();
        if (t.val != ")") {
          PE::Throw("expected ')'", t.lc);
        }

        return l;
      }
    }

    switch (t.kind) {
    case Token::kInteger:
      return VInt{ToInt(t.val)};
    case Token::kFloat:
      if constexpr (int_only) {
        PE::Throw("float value in an integer expression", t.lc);
      }

      return VFloat{ToFloat(t.val)};
    case Token::kIdentifier:
      {
        const auto* sym = symtab.Lookup(t.val);

        if (sym == nullptr) {
          PE::Throw("identifier '{}' is not defined", t.lc, t.val);
        }

        bool convertible_to_int = sym->IsConvertibleToInt();

        if (int_only && !convertible_to_int) {
          PE::Throw("float identifier in an integer expression", t.lc);
        }

        Token n = tok.Peek();

        if (n.val != "[") {
          if (sym->IsReadOnly()) {
            if (convertible_to_int) {
              return VInt{sym->GetInt()};
            } else {
              return VFloat{sym->GetFloat()};
            }
          } else {
            if (sym->IsMea()) {
              f(sym->GetMea(), t.val);
            }
            return SymbolKey{t.val};
          }
        } else {
          if (!sym->IsMea() && !sym->IsVector()) {
            PE::Throw("scalar identifier cannot be indexed", t.lc);
          }

          Token n = tok.Peek(1);

          auto index = IndexParser::Run(symtab, tok);
          bool const_index = IsConstIndex(index);
          VInt i = const_index ? EvalConstIndex(symtab, index) : 0;

          if (const_index) {
            if (i < 0) {
              PE::Throw("negative index {}", n.lc, i);
            } else if (static_cast<std::size_t>(i) >= sym->Size()) {
              PE::Throw("index {} is out of range", n.lc, i);
            }
          }

          if (sym->IsReadOnly() && const_index) {
            if (convertible_to_int) {
              return VInt{sym->GetInt(i)};
            } else {
              return VFloat{sym->GetFloat(i)};
            }
          } else {
            if (sym->IsMea()) {
              f(sym->GetMea(), t.val);
            }

            if (const_index) {
              return SymbolKeyInd{t.val, i};
            } else {
              return SymbolKeyInd{t.val, std::move(index)};
            }
          }
        }
      }
    default:
      PE::Throw("unexpected token", t.lc);
      return 0.0;
    }
  }

  /**
   * Parses an infix or postfix operation starting at the current operator
   * token and combines it with the left-hand operand `l`.
   * Consumes the operator token and any corresponding right-hand expression
   * according to the operator's binding power.
   * If both operands are compile-time constants, the result is folded
   * directly into a scalar literal (`VInt`, `VFloat`, or `VBool`).
   * @tparam F Callback function type with signature
   *   `void(const Symbol::Mea&, std::string_view)`.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param tok Lexical analyzer stream positioned at the operator token.
   * @param l Already-parsed left-hand `Expr` operand to bind to the operator.
   * @param f Callback invoked whenever a measurement (`Mea`) symbol
   *   is parsed.
   * @return Combined `Expr` variant containing a constant scalar value or
   *   an executable function closure.
   * @throws syntax_error on syntax errors, unexpected tokens, or type
   *   mismatches.
   */
  template <typename F>
  static Expr GetRight(
      const SymTable& symtab, Tokenizer& tok, const Expr& l, F&& f) {
    Token t = tok();

    auto empty = [](unsigned lc, auto r) {};

    auto div_by_zero = [](unsigned lc, auto r) {
      if (r == decltype(r){0}) {
        PE::Throw("division by zero", lc);
      }
    };

    switch (OpHash(t.val)) {
    case "||"_ophash:
      return LogicalExpr(symtab, tok, l, kOrPrec, t.lc, f,
                         [](auto l, auto r)  { return l || r; });
    case "^^"_ophash:
      return LogicalExpr(symtab, tok, l, kXorPrec, t.lc, f,
                         [](auto l, auto r)  { return l != r; });
    case "&&"_ophash:
      return LogicalExpr(symtab, tok, l, kAndPrec, t.lc, f,
                         [](auto l, auto r)  { return l && r; });
    case "|"_ophash:
      return ArithIntExpr(symtab, tok, l, kBitOrPrec, t.lc, f,
                          [](auto l, auto r)  { return l | r; }, empty);
    case "^"_ophash:
      return ArithIntExpr(symtab, tok, l, kBitXorPrec, t.lc, f,
                          [](auto l, auto r)  { return l ^ r; }, empty);
    case "&"_ophash:
      return ArithIntExpr(symtab, tok, l, kBitAndPrec, t.lc, f,
                          [](auto l, auto r)  { return l & r; }, empty);
    case "=="_ophash:
      return ComparisonExpr(symtab, tok, l, kEqPrec, f,
                            [](auto l, auto r)  { return l == r; });
    case "!="_ophash:
      return ComparisonExpr(symtab, tok, l, kNotEqPrec, f,
                            [](auto l, auto r)  { return l != r; });
    case "<"_ophash:
      return ComparisonExpr(symtab, tok, l, kLessPrec, f,
                            [](auto l, auto r)  { return l < r; });
    case "<="_ophash:
      return ComparisonExpr(symtab, tok, l, kLessEqPrec, f,
                            [](auto l, auto r)  { return l <= r; });
    case ">"_ophash:
      return ComparisonExpr(symtab, tok, l, kGreaterPrec, f,
                            [](auto l, auto r)  { return l > r; });
    case ">="_ophash:
      return ComparisonExpr(symtab, tok, l, kGreaterEqPrec, f,
                            [](auto l, auto r)  { return l >= r; });
    case "<<"_ophash:
      return ShiftExpr(symtab, tok, l, kBitLeftShiftPrec, t.lc, f,
                       [](auto l, auto r)  { return l << r; });
    case ">>"_ophash:
      return ShiftExpr(symtab, tok, l, kBitRightShiftPrec, t.lc, f,
                       [](auto l, auto r)  { return l >> r; });
    case "+"_ophash:
      return ArithIntOrFloatExpr(symtab, tok, l, kAddPrec, t.lc, f,
                                 [](auto l, auto r)  { return l + r; }, empty);
    case "-"_ophash:
      return ArithIntOrFloatExpr(symtab, tok, l, kSubPrec, t.lc, f,
                                 [](auto l, auto r)  { return l - r; }, empty);
    case "*"_ophash:
      return ArithIntOrFloatExpr(symtab, tok, l, kMulPrec, t.lc, f,
                                 [](auto l, auto r)  { return l * r; }, empty);
    case "/"_ophash:
      return ArithIntOrFloatExpr(symtab, tok, l, kDivPrec, t.lc, f,
                                 [](auto l, auto r)  { return l / r; },
                                 div_by_zero);
    case "%"_ophash:
      return ArithIntExpr(symtab, tok, l, kModPrec, t.lc, f,
                          [](auto l, auto r)  { return l % r; }, div_by_zero);
    case "**"_ophash:
      {
        if constexpr (int_only) {
          PE::Throw(
              "the operator ** is defined only for float expressions", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kPowPrec - 1, f);

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return VFloat{std::pow(EvalFloatExpr(symtab, l),
                                 EvalFloatExpr(symtab, r))};
        } else {
          return EvalFloatFn{[l, r](const SymTable& symtab) {
            return std::pow(EvalFloatExpr(symtab, l), EvalFloatExpr(symtab, r));
          }};
        }
      }
    default:
      PE::Throw("unexpected operator", t.lc);
      return 0.0;
    }
  }

  template <typename F>
  static Expr GetIntOrError(const SymTable& symtab, Tokenizer& tok,
                            const Expr& l, unsigned precedence, unsigned lc,
                            F&& f) {
    if (!IsConvertibleToInt(symtab, l)) {
      PE::Throw("expected a left operand of type 'bool' or 'int'", lc);
    }

    Expr r = RunWithPrecedence(symtab, tok, precedence, f);

    if (!IsConvertibleToInt(symtab, r)) {
      PE::Throw("expected a right operand of type 'bool' or 'int'", lc);
    }

    return r;
  }

  template <typename F, typename G>
  static Expr LogicalExpr(const SymTable& symtab, Tokenizer& tok,
                          const Expr& l, unsigned lc, unsigned precedence,
                          F&& f, G&& g) {
    Expr r = GetIntOrError(symtab, tok, l, precedence, lc, f);

    if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
      return VBool{g(EvalCondExpr(symtab, l), EvalCondExpr(symtab, r))};
    } else {
      return EvalBoolFn{[l, r, g](const SymTable& symtab) {
        return VBool{g(EvalCondExpr(symtab, l), EvalCondExpr(symtab, r))};
      }};
    }
  }

  template <typename F, typename G>
  static Expr ShiftExpr(const SymTable& symtab, Tokenizer& tok,
                        const Expr& l, unsigned precedence, unsigned lc,
                        F&& f, G&& g) {
    Expr r = GetIntOrError(symtab, tok, l, precedence, lc, f);

    if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
      return VInt{g(EvalIntExpr(symtab, l), EvalIntExpr(symtab, r))};
    } else {
      return EvalIntFn{[l, r, g](const SymTable& symtab) {
      auto val = EvalIntExpr(symtab, l);
        return VInt{g(TruncateMea(symtab, l, val), EvalIntExpr(symtab, r))};
      }};
    }
  }

  template <typename F, typename G>
  static Expr ComparisonExpr(const SymTable& symtab, Tokenizer& tok,
                             const Expr& l, unsigned precedence, F&& f, G&& g) {
    Expr r = RunWithPrecedence(symtab, tok, precedence, f);

    if (IsConvertibleToInt(symtab, l) && IsConvertibleToInt(symtab, r)) {
      if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
        return VBool{g(EvalIntExpr(symtab, l), EvalIntExpr(symtab, r))};
      } else {
        return EvalBoolFn{[l, r, g](const SymTable& symtab) {
          return VBool{g(EvalIntExpr(symtab, l), EvalIntExpr(symtab, r))};
        }};
      }
    } else {
      if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
        return VBool{g(EvalFloatExpr(symtab, l), EvalFloatExpr(symtab, r))};
      } else {
        return EvalBoolFn{[l, r, g](const SymTable& symtab) {
          return VBool{g(EvalFloatExpr(symtab, l), EvalFloatExpr(symtab, r))};
        }};
      }
    }
  }

  template <typename F, typename G, typename VR>
  static Expr ArithIntExpr(const SymTable& symtab, Tokenizer& tok,
                           const Expr& l, unsigned precedence, unsigned lc,
                           F&& f, G&& g, VR&& validate_right) {
    Expr r = GetIntOrError(symtab, tok, l, precedence, lc, f);

    if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
      auto val_r = EvalIntExpr(symtab, r);
      validate_right(lc, val_r);
      return VInt{g(EvalIntExpr(symtab, l), val_r)};
    } else {
      return EvalIntFn{[l, r, lc, g, validate_right](const SymTable& symtab) {
        auto val_r = EvalIntExpr(symtab, r);
        validate_right(lc, val_r);
        return VInt{g(EvalIntExpr(symtab, l), EvalIntExpr(symtab, r))};
      }};
    }
  }

  template <typename F, typename G, typename VR>
  static Expr ArithIntOrFloatExpr(
      const SymTable& symtab, Tokenizer& tok, const Expr& l,
      unsigned precedence, unsigned lc, F&& f, G&& g, VR&& validate_right) {
    Expr r = RunWithPrecedence(symtab, tok, kAddPrec, f);

    if (IsConvertibleToInt(symtab, l) && IsConvertibleToInt(symtab, r)) {
      if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
        auto val_r = EvalIntExpr(symtab, r);
        validate_right(lc, val_r);
        return VInt{g(EvalIntExpr(symtab, l), val_r)};
      } else {
        return EvalIntFn{[l, r, lc, g, validate_right](const SymTable& symtab) {
          auto val_r = EvalIntExpr(symtab, r);
          validate_right(lc, val_r);
          return VInt{g(EvalIntExpr(symtab, l), val_r)};
        }};
      }
    } else {
      if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
        auto val_r = EvalFloatExpr(symtab, r);
        validate_right(lc, val_r);
        return VFloat{g(EvalFloatExpr(symtab, l), val_r)};
      } else {
        return EvalFloatFn{[l, r, lc, g, validate_right](
            const SymTable& symtab) {
          auto val_r = EvalFloatExpr(symtab, r);
          validate_right(lc, val_r);
          return VFloat{g(EvalFloatExpr(symtab, l), val_r)};
        }};
      }
    }
  }

  static VInt TruncateMea(const SymTable& symtab, const Expr& l, VInt val) {
    if (IsSymbol(l)) {
      auto name = std::get<SymbolKey>(l);
      const auto* sym = symtab.Lookup(name);

      if (sym == nullptr) {
        Error::Throw("identifier '{}' is not defined", name);
      }

      if (sym->IsMea()) {
        // sym->Size() should should not exceed 63.
        val &= (Symbol::Int{1} << sym->Size()) - 1;
      }
    } else if (IsSymbolInd(l)) {
      const auto& p = std::get<SymbolKeyInd>(l);
      const auto* sym = symtab.Lookup(p.first);

      if (sym == nullptr) {
        Error::Throw("identifier '{}' is not defined", p.first);
      }

      if (sym->IsMea()) {
        val &= (Symbol::Int{1} << IndexSize(p.second)) - 1;
      }
    }

    return val;
  }
};

/**
 * Parses bracketed index expressions (`[ <int_expr> ]`).
 * @tparam ParserError Error policy for syntax errors.
 * @tparam RuntimeError Error policy for runtime errors.
 */
template <typename ParserError, typename RuntimeError>
struct IndexParser {
 private:
  using PE = ParserError;
  using RE = RuntimeError;

  using IntExprParser = ExprParser<IndexParser, PE, RE, true>;

 public:
  /**
   * Consumes a opening bracket '[', parses an integer expression, and
   *   consumes ']'.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param tok Tokenizer stream positioned at '['.
   * @return Index variant containing a constant scalar or integer evaluation
   *   closure (`EvalIntFn`/`EvalBoolFn`).
   * @throws syntax_error If the bracket syntax is malformed or
   *   if the expression is non-integer.
   */
  static Index Run(const SymTable& symtab, Tokenizer& tok) {
    Token n = tok();  // Consume '['.
    n = tok.Peek();

    auto e = IntExprParser::Run(symtab, tok);

    auto f = [&symtab, &n](auto&& e) -> Index {
      using E = std::decay_t<decltype(e)>;

      if constexpr (std::is_same_v<E, VInt>) {
        return e;
      } else if constexpr (std::is_same_v<E, VBool>) {
        return e;
      } else if constexpr (std::is_same_v<E, EvalIntFn>) {
        return e;
      } else if constexpr (std::is_same_v<E, EvalBoolFn>) {
        return e;
      } else if constexpr (std::is_same_v<E, SymbolKey>) {
        const auto* sym = symtab.LookupOrError(
            e, "identifier '{}' is not defined", e);
        if (!sym->IsConvertibleToInt()) {
          PE::Throw("index must be an integer expression", n.lc);
        }
        return EvalIntFn{[e](const SymTable& symtab) {
          return EvalIntExpr(symtab, e);
        }};
      } else if constexpr (std::is_same_v<E, SymbolKeyInd>) {
        const auto* sym = symtab.LookupOrError(
            e.first, "identifier '{}' is not defined", e.first);
        if (!sym->IsConvertibleToInt()) {
          PE::Throw("index must be an integer expression", n.lc);
        }
        return EvalIntFn{[e](const SymTable& symtab) {
          return EvalIntExpr(symtab, e);
        }};
      } else {
        PE::Throw("index must be an integer expression", n.lc);
        return VInt{0};
      }
    };

    n = tok();
    if (n.val != "]") {
      PE::Throw("expected ']'", n.lc);
    }

    return std::visit(f, e);
  }
};

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_PARSER_H_
