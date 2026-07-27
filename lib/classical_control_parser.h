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
   * Stops automatically when operators are exhausted or when an opening
   * (left) parenthesis is reached.
   * If the expression involves only compile-time constants, the expression
   * is folded directly into a scalar literal (`TInt`, `TFloat`, or `TBool`).
   * @tparam F Callback function type with signature
   *   `void(const Symbol::Mea&, std::string_view)`.
   * @param symtab Symbol table instance used for symbol lookups.
   * @param tok Lexical analyzer stream positioned at the start of
   *   the expression.
   * @param f Optional callback invoked whenever a measurement (`Mea`) symbol
   *   is parsed.
   * @return Parsed `Expr` variant containing a constant scalar value or
   *   an executable function closure.
   * @throws syntax_erorr on syntax errors, unexpected tokens, or type
   *   mismatches.
   */
  template <typename F = decltype(empty)>
  static Expr Run(
      const SymTable& symtab, Tokenizer& tok, F&& f = std::forward<F>(empty)) {
    return RunWithPrecedence(symtab, tok, 0, f);
  }

 private:
  static bool IsInt(const Expr& e) {
    return std::holds_alternative<TInt>(e);
  }

  static bool IsSymbol(const Expr& e) {
    return std::holds_alternative<TSymbol>(e);
  }

  static bool IsSymbolInd(const Expr& e) {
    return std::holds_alternative<TSymbolInd>(e);
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

  template <typename F>
  static Expr RunWithPrecedence(
      const SymTable& symtab, Tokenizer& tok, unsigned precedence, F&& f) {
    Expr l = GetLeft(symtab, tok, f);

    while (1) {
      Token n = tok.Peek();

      if (n.kind != Token::kOperator ||
          n.val == "(" || n.val == ")" || n.val == "]") {
        break;
      }

      if (precedence >= GetPrecedence(n)) {
        break;
      }

      l = GetRight(symtab, tok, l, f);
    }

    return l;
  }

  template <typename F>
  static Expr GetLeft(const SymTable& symtab, Tokenizer& tok, F&& f) {
    Token t = tok();

    switch (OpHash(t.val)) {
    case "!"_ophash:
      {
        if (t.val == "!") {
          Token n = tok.Peek();

          Expr l = GetLeft(symtab, tok, f);

          if (!IsConvertibleToInt(symtab, l)) {
            PE::Throw("expression is not convertible to 'bool'", n.lc);
          }

          if (IsConstExpr(symtab, l)) {
            return TBool{!EvalCondExpr(symtab, l)};
          } else {
            return TFuncB{[l](const SymTable& symtab) {
              return !EvalCondExpr(symtab, l);
            }};
          }
        }
      }
    case "~"_ophash:
      {
        Token n = tok.Peek();
        unsigned lc = n.lc;

        Expr l = GetLeft(symtab, tok, f);

        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected an integer expression", t.lc);
        }

        if (IsConstExpr(symtab, l)) {
          return TInt{~EvalIntExpr(symtab, l)};
        } else {
          return TFuncI{[l, lc](const SymTable& symtab) {
            auto val = ~EvalIntExpr(symtab, l);
            val = TruncateMea(symtab, l, val);
            return TInt{val};
          }};
        }
      }
    case "-"_ophash:
      {
        Expr l = GetLeft(symtab, tok, f);

        if (IsConstExpr(symtab, l)) {
          if (IsConvertibleToInt(symtab, l)) {
            return TInt{-EvalIntExpr(symtab, l)};
          } else {
            return TFloat{-EvalExpr(symtab, l)};
          }
        } else {
          if (IsConvertibleToInt(symtab, l)) {
            return TFuncI{[l](const SymTable& symtab) {
              return -EvalIntExpr(symtab, l);
            }};
          } else {
            return TFuncF{[l](const SymTable& symtab) {
              return -EvalExpr(symtab, l);
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
      return TInt{ToInt(t.val)};
    case Token::kFloat:
      if constexpr (int_only) {
        PE::Throw("float value in an integer expression", t.lc);
      }

      return TFloat{ToFloat(t.val)};
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
              return TInt{sym->GetInt()};
            } else {
              return TFloat{sym->GetFloat()};
            }
          } else {
            if (sym->HoldsMea()) {
              f(sym->GetMea(), t.val);
            }
            return TSymbol{t.val};
          }
        } else {
          if (!sym->HoldsMea() && !sym->HoldsVector()) {
            PE::Throw("scalar identifier cannot be indexed", t.lc);
          }

          Token n = tok.Peek(1);

          auto index = IndexParser::Run(symtab, tok);
          uint64_t i = EvalIndex(symtab, index);

          if (i >= sym->Size()) {
            PE::Throw("index is out of range", n.lc);
          }

          if (sym->IsReadOnly() && IsConstIndex(symtab, index)) {
            if (convertible_to_int) {
              return TInt{sym->GetInt(i)};
            } else {
              return TFloat{sym->GetFloat(i)};
            }
          } else {
            if (sym->HoldsMea()) {
              f(sym->GetMea(), t.val);
            }
            return TSymbolInd{t.val, std::move(index)};
          }
        }
      }
    default:
      PE::Throw("unexpected token", t.lc);
      return 0.0;
    }
  }

  template <typename F>
  static Expr GetRight(
      const SymTable& symtab, Tokenizer& tok, const Expr& l, F&& f) {
    Token t = tok();

    switch (OpHash(t.val)) {
    case "||"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kOrPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TBool{EvalCondExpr(symtab, l) || EvalCondExpr(symtab, r)};
        } else {
          return TFuncB{[l, r](const SymTable& symtab) {
            return EvalCondExpr(symtab, l) || EvalCondExpr(symtab, r);
          }};
        }
      }
    case "^^"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kXorPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TBool{EvalCondExpr(symtab, l) != EvalCondExpr(symtab, r)};
        } else {
          return TFuncB{[l, r](const SymTable& symtab) {
            return EvalCondExpr(symtab, l) != EvalCondExpr(symtab, r);
          }};
        }
      }
    case "&&"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kAndPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TBool{EvalCondExpr(symtab, l) && EvalCondExpr(symtab, r)};
        } else {
          return TFuncB{[l, r](const SymTable& symtab) {
            return EvalCondExpr(symtab, l) && EvalCondExpr(symtab, r);
          }};
        }
      }
    case "|"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kBitOrPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TInt{EvalIntExpr(symtab, l) | EvalIntExpr(symtab, r)};
        } else {
          return TFuncI{[l, r](const SymTable& symtab) {
            return EvalIntExpr(symtab, l) | EvalIntExpr(symtab, r);
          }};
        }
      }
    case "^"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kBitXorPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TInt{EvalIntExpr(symtab, l) ^ EvalIntExpr(symtab, r)};
        } else {
          return TFuncI{[l, r](const SymTable& symtab) {
            return EvalIntExpr(symtab, l) ^ EvalIntExpr(symtab, r);
          }};
        }
      }
    case "&"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kBitAndPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TInt{EvalIntExpr(symtab, l) & EvalIntExpr(symtab, r)};
        } else {
          return TFuncI{[l, r](const SymTable& symtab) {
            return EvalIntExpr(symtab, l) & EvalIntExpr(symtab, r);
          }};
        }
      }
    case "=="_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kEqPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalIntExpr(symtab, l) == EvalIntExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) == EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalExpr(symtab, l) == EvalExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) == EvalExpr(symtab, r);
            }};
          }
        }
      }
    case "!="_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kNotEqPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalIntExpr(symtab, l) != EvalIntExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) != EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalExpr(symtab, l) != EvalExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) != EvalExpr(symtab, r);
            }};
          }
        }
      }
    case "<"_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kLessPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalIntExpr(symtab, l) < EvalIntExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) < EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalExpr(symtab, l) < EvalExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) < EvalExpr(symtab, r);
            }};
          }
        }
      }
    case "<="_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kLessEqPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalIntExpr(symtab, l) <= EvalIntExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) <= EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalExpr(symtab, l) <= EvalExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) <= EvalExpr(symtab, r);
            }};
          }
        }
      }
    case ">"_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kGreaterPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalIntExpr(symtab, l) > EvalIntExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) > EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalExpr(symtab, l) > EvalExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) > EvalExpr(symtab, r);
            }};
          }
        }
      }
    case ">="_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kGreaterEqPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalIntExpr(symtab, l) >= EvalIntExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) >= EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TBool{EvalExpr(symtab, l) >= EvalExpr(symtab, r)};
          } else {
            return TFuncB{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) >= EvalExpr(symtab, r);
            }};
          }
        }
      }
    case "<<"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kBitLeftShiftPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TInt{EvalIntExpr(symtab, l) << EvalIntExpr(symtab, r)};
        } else {
          return TFuncI{[l, r](const SymTable& symtab) {
            auto val = EvalIntExpr(symtab, l);
            return TruncateMea(symtab, l, val) << EvalIntExpr(symtab, r);
          }};
        }
      }
    case ">>"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kBitRightShiftPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TInt{EvalIntExpr(symtab, l) >> EvalIntExpr(symtab, r)};
        } else {
          return TFuncI{[l, r](const SymTable& symtab) {
            auto val = EvalIntExpr(symtab, l);
            return TruncateMea(symtab, l, val) >> EvalIntExpr(symtab, r);
          }};
        }
      }
    case "+"_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kAddPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TInt{EvalIntExpr(symtab, l) + EvalIntExpr(symtab, r)};
          } else {
            return TFuncI{[l, r](const SymTable& symtab) -> TInt {
              return EvalIntExpr(symtab, l) + EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TFloat{EvalExpr(symtab, l) + EvalExpr(symtab, r)};
          } else {
            return TFuncF{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) + EvalExpr(symtab, r);
            }};
          }
        }
      }
    case "-"_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kSubPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TInt{EvalIntExpr(symtab, l) - EvalIntExpr(symtab, r)};
          } else {
            return TFuncI{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) - EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TFloat{EvalExpr(symtab, l) - EvalExpr(symtab, r)};
          } else {
            return TFuncF{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) - EvalExpr(symtab, r);
            }};
          }
        }
      }
    case "*"_ophash:
      {
        Expr r = RunWithPrecedence(symtab, tok, kMulPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TInt{EvalIntExpr(symtab, l) * EvalIntExpr(symtab, r)};
          } else {
            return TFuncI{[l, r](const SymTable& symtab) {
              return EvalIntExpr(symtab, l) * EvalIntExpr(symtab, r);
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            return TFloat{EvalExpr(symtab, l) * EvalExpr(symtab, r)};
          } else {
            return TFuncF{[l, r](const SymTable& symtab) {
              return EvalExpr(symtab, l) * EvalExpr(symtab, r);
            }};
          }
        }
      }
    case "/"_ophash:
      {
        auto lc = t.lc;
        Expr r = RunWithPrecedence(symtab, tok, kDivPrec, f);

        if (IsConvertibleToInt(symtab, l)
            && IsConvertibleToInt(symtab, r)) {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            auto divisor = EvalIntExpr(symtab, r);
            if (divisor == 0) {
              PE::Throw("division by zero", lc);
            }
            return TInt{EvalIntExpr(symtab, l) / divisor};
          } else {
            return TFuncI{[l, r, lc](const SymTable& symtab) {
              auto divisor = EvalIntExpr(symtab, r);
              if (divisor == 0) {
                RE::Throw("division by zero", lc);
              }
              return EvalIntExpr(symtab, l) / divisor;
            }};
          }
        } else {
          if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
            auto divisor = EvalExpr(symtab, r);
            if (divisor == 0.0) {
              PE::Throw("division by zero", lc);
            }
            return TFloat{EvalExpr(symtab, l) / divisor};
          } else {
            return TFuncF{[l, r, lc](const SymTable& symtab) {
              auto divisor = EvalExpr(symtab, r);
              if (divisor == 0.0) {
                RE::Throw("division by zero", lc);
              }
              return EvalExpr(symtab, l) / divisor;
            }};
          }
        }
      }
    case "%"_ophash:
      {
        if (!IsConvertibleToInt(symtab, l)) {
          PE::Throw("expected a left operand of type 'bool' or 'int'", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kModPrec, f);

        if (!IsConvertibleToInt(symtab, r)) {
          PE::Throw("expected a right operand of type 'bool' or 'int'", t.lc);
        }

        auto lc = t.lc;

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          auto divisor = EvalIntExpr(symtab, r);
          if (divisor == 0) {
            PE::Throw("division by zero", lc);
          }
          return TInt{EvalIntExpr(symtab, l) % divisor};
        } else {
          return TFuncI{[l, r, lc](const SymTable& symtab) {
            auto divisor = EvalIntExpr(symtab, r);
            if (divisor == 0) {
              RE::Throw("division by zero", lc);
            }
            return EvalIntExpr(symtab, l) % divisor;
          }};
        }
      }
    case "**"_ophash:
      {
        if constexpr (int_only) {
          PE::Throw(
              "the operator ** is defined only for float expressions", t.lc);
        }

        Expr r = RunWithPrecedence(symtab, tok, kPowPrec, f);

        if (IsConstExpr(symtab, l) && IsConstExpr(symtab, r)) {
          return TFloat{std::pow(EvalExpr(symtab, l), EvalExpr(symtab, r))};
        } else {
          return TFuncF{[l, r](const SymTable& symtab) {
            return std::pow(EvalExpr(symtab, l), EvalExpr(symtab, r));
          }};
        }
      }
    default:
      PE::Throw("unexpected operator", t.lc);
      return 0.0;
    }
  }

  static TInt TruncateMea(const SymTable& symtab, const Expr& l, TInt val) {
    if (IsSymbol(l)) {
      auto name = std::get<TSymbol>(l);
      const auto* sym = symtab.Lookup(name);

      if (sym == nullptr) {
        Error::Throw("identifier '{}' is not defined", name);
      }

      if (sym->HoldsMea()) {
        // sym->Size() should should not exceed 63.
        val &= (Symbol::Int{1} << sym->Size()) - 1;
      }
    } else if (IsSymbolInd(l)) {
      const auto& p = std::get<TSymbolInd>(l);
      const auto* sym = symtab.Lookup(p.first);

      if (sym == nullptr) {
        Error::Throw("identifier '{}' is not defined", p.first);
      }

      if (sym->HoldsMea()) {
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
   *   closure (`TFuncI`/`TFuncB`).
   * @throws syntax_error If the bracket syntax is malformed or
   *   if the expression is non-integer.
   */
  static Index Run(const SymTable& symtab, Tokenizer& tok) {
    Token n = tok();  // Consume '['.
    n = tok.Peek();

    auto e = IntExprParser::Run(symtab, tok);

    auto f = [&symtab, &n](auto&& e) -> Index {
      using E = std::decay_t<decltype(e)>;

      if constexpr (std::is_same_v<E, TInt>) {
        return e;
      } else if constexpr (std::is_same_v<E, TBool>) {
        return e;
      } else if constexpr (std::is_same_v<E, TFuncI>) {
        return e;
      } else if constexpr (std::is_same_v<E, TFuncB>) {
        return e;
      } else if constexpr (std::is_same_v<E, TSymbol>) {
        const auto* sym = symtab.LookupOrError(
            e, "identifier '{}' is not defined", e);
        if (!sym->IsConvertibleToInt()) {
          PE::Throw("index must be an integer expression", n.lc);
        }
        return TFuncI{[sym](const SymTable& symtab) {
          return sym->GetInt();
        }};
      } else if constexpr (std::is_same_v<E, TSymbolInd>) {
        const auto* sym = symtab.LookupOrError(
            e.first, "identifier '{}' is not defined", e.first);
        if (!sym->IsConvertibleToInt()) {
          PE::Throw("index must be an integer expression", n.lc);
        }
        return TFuncI{[sym, e](const SymTable& symtab) {
          return sym->GetInt(EvalIndex(symtab, e.second));
        }};
      } else {
        PE::Throw("index must be an integer expression", n.lc);
        return TInt{0};
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
