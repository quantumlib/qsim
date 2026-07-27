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

#ifndef PARSER_H_
#define PARSER_H_

#include <cmath>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "channels_cirq.h"
#include "circuit.h"
#include "classical_control_error.h"
#include "classical_control_expr.h"
#include "classical_control_obs.h"
#include "classical_control_parser.h"
#include "classical_control_symbol.h"
#include "classical_control_symtab.h"
#include "classical_control_tokenizer.h"
#include "classical_control_util.h"
#include "error.h"
#include "gates_qsim.h"
#include "matrix.h"
#include "operation.h"
#include "operation_base.h"

namespace qsim {

namespace cc {

/**
 * Parses space-, semicolon-, or line-delimited constant definitions from
 * a string buffer.
 *
 * Populates the provided symbol table symtab with evaluation results from
 * statements like `"nq = 5 c1 = 1 c2 = c1 * 2"`. Constant definitions do not
 * use `const` or type keywords in this mode. Evaluated expressions convert to
 * `TInt` if `IsConvertibleToInt` returns true, otherwise to `TFloat`.
 *
 * @tparam ParserError Error policy for syntax errors.
 * @tparam RuntimeError Error policy for runtime errors.
 * @tparam SymTable Symbol table type.
 * @param sym_defs String view containing symbol assignment definitions.
 * @param symtab Symbol table instance to populate with parsed symbols.
 * @throws syntax_error on syntax or indexing errors during symbol parsing.
 */
template <typename ParserError, typename RuntimeError, typename SymTable>
inline void ParseSymbols(std::string_view sym_defs, SymTable& symtab) {
  using PE = ParserError;
  using RE = RuntimeError;

  // Ensure we are operating in a active scope
  if (symtab.Empty()) {
    auto scope = symtab.AddScope();
    symtab.EnterScope(scope);
  }

  Tokenizer tok(sym_defs);
  Token t = tok();

  // Skip optional semicolons or delimiters
  while (t.kind == Token::kDelimiter) {
    t = tok();
  }

  while (t.kind != Token::kEndOfFile) {
    if (t.kind != Token::kIdentifier) {
      PE::Throw("expected an identifier", t.lc);
    }

    if (auto* sym = symtab.LookupInCurrentScope(t.val)) {
      PE::Throw("identifier '{}' is already defined", t.lc, t.val);
    }

    Token n = tok();

    if (n.val != "=") {
      PE::Throw("expected '='", n.lc);
    }

    Expr expr = ExprParser<IndexParser<PE, RE>, PE, RE>::Run(symtab, tok);

    if (IsConvertibleToInt(symtab, expr)) {
      symtab.Insert(t.val,
                    Symbol{Symbol::Int{EvalIntExpr(symtab, expr)}, true});
    } else {
      symtab.Insert(t.val, Symbol{Symbol::Float{EvalExpr(symtab, expr)}, true});
    }

    t = tok();

    // Skip optional semicolons or delimiters
    while (t.kind == Token::kDelimiter) {
      t = tok();
    }
  }
}

/**
 * Parser for the qsim circuit definition language.
 * @tparam FP Floating-point precision type (`float` or `double`).
 * @tparam ParserError Error policy for syntax errors.
 * @tparam RuntimeError Error policy for runtime errors.
 */
template <typename FP, typename ParserError, typename RuntimeError>
struct CircuitQsimParser final {
 private:
  using fp_type = FP;
  using Operation = qsim::Operation<fp_type>;
  using Operations = std::vector<Operation>;
  using ClassicallyControlledOperation =
      qsim::ClassicallyControlledOperation<fp_type>;
  using Circuit = qsim::Circuit<Operation>;
  using Channel = qsim::Channel<fp_type>;

  using RuntimeResolvedGate = qsim::RuntimeResolvedGate<fp_type>;

  using PE = ParserError;
  using RE = RuntimeError;

  using IndexParser = cc::IndexParser<PE, RE>;
  using ExprParser = cc::ExprParser<IndexParser, PE, RE>;
  using IntExprParser = cc::ExprParser<IndexParser, PE, RE, true>;
  using CondExprParser = IntExprParser;

 public:
  /**
   * Parses a string representation of a circuit into a Circuit object.
   * @param circuit_str Input string containing the qsim DSL circuit program.
   * @param max_depth Maximum gate depth/time (inclusive) up to which
   *   operations will be parsed.
   * @return The parsed `Circuit` object.
   * @throws syntax_error if syntax errors occur during parsing.
   */
  static auto Run(std::string_view circuit_str, unsigned max_depth) {
    auto [circuit, _] = Run(circuit_str, max_depth, SymTable{});
    return std::move(circuit);
  }

  /**
   * Parses a string representation of a circuit and populates an existing
   * symbol table.
   * @tparam SymTable Symbols table container type.
   * @param circuit_str Input string containing the qsim DSL circuit program.
   * @param max_depth Maximum gate depth/time (inclusive) up to which
   *   operations will be parsed.
   * @param symtab Symbol table instance populated during parsing.
   * @return Pair containing the parsed `Circuit` and `Observables` objects.
   * @throws syntax_error if syntax errors occur during parsing.
   */
  template <typename SymTable>
  static auto Run(
      std::string_view circuit_str, unsigned max_depth, SymTable&& symtab) {
    Circuit circuit;
    Observables obss;

    if (symtab.Empty()) {
      symtab.EnterScope(symtab.AddScope());
    }

    Tokenizer tok(circuit_str);

    Token t = tok.Peek();

    while (t.kind == Token::kDelimiter) {
      tok();
      t = tok.Peek();
    }

    try {
      Expr e = IntExprParser::Run(symtab, tok);

      t = tok.Peek();
      if (t.kind != Token::kDelimiter) {
        Error::Throw("");
      }

      circuit.num_qubits = EvalIntExpr(symtab, e);
    } catch (std::runtime_error&) {
      PE::Throw("expected a delimiter", t.lc);
    } catch (...) {
      tok.Restart();

      Symbol* sym = symtab.Lookup("nq");
      if (sym == nullptr) {
        circuit.num_qubits = 1;
      } else {
        if (sym->IsConvertibleToInt()) {
          circuit.num_qubits = sym->GetInt();
        } else {
          Error::Throw("'nq' is not integer");
        }
      }
    }

    Symbol* sym = symtab.Lookup("nq");
    if (sym != nullptr) {
      sym->Assign(cc::Symbol::Int(circuit.num_qubits));
    } else {
      symtab.Insert("nq", cc::Symbol::Int(circuit.num_qubits));
    }

    t = tok.Peek();

    while (t.kind == Token::kDelimiter) {
      tok();
      t = tok.Peek();
    }

    std::optional<Channel> channel;

    if (t.val == "noise") {
      tok();
      channel = GetChannel(symtab, tok);

      t = tok.Peek();
      if (t.kind != Token::kDelimiter) {
        PE::Throw("expected a delimiter", t.lc);
      }
    }

    std::vector<bool> acts_on(circuit.num_qubits, false);

    circuit.ops = ParseOps(0, circuit.num_qubits, max_depth,
                           symtab, obss, channel, acts_on, tok);

    return std::make_pair(circuit, obss);
  }

 private:
  static Operations ParseOps(
      unsigned level, unsigned num_qubits,
      unsigned max_depth, SymTable& symtab, Observables& obss,
      const std::optional<Channel>& channel, std::vector<bool>& acts_on,
      Tokenizer& tok) {
    Operations ops;
    ops.reserve(2048);  // Okay for not very long circuits.

    Token t = tok();

    unsigned cur_time = 0;
    std::vector<int> qubit_times(num_qubits, -1);

    bool is_add_noise = IsAddNoise(channel.has_value(), level);

    while (t.kind != Token::kEndOfFile) {
      while (t.kind == Token::kDelimiter) {
        t = tok();
      }

      if (t.kind == Token::kEndOfFile) {
        break;
      }

      int time = -1;
      unsigned prev_time = cur_time;

      if (t.kind == Token::kInteger) {
        time = ToInt(t.val);
        if (static_cast<unsigned>(time) > max_depth) {
          break;
        }

        t = tok();
      }

      int max_mea_time = -2;
      auto f = [&symtab, &max_mea_time](
          const Symbol::Mea& mea, std::string_view name) {
        if (symtab.LookupInCurrentScope(name) != nullptr) {
          int time = mea.time;
          if (time > max_mea_time) {
            max_mea_time = time;
          }
        }
      };

      if (auto gate = GetGate(t.val, num_qubits, symtab, tok, f)) {
        // Matrix gate.

        auto& bop = OpBaseOperation(*gate);

        cur_time = GetTime(t, time, cur_time, bop.qubits, qubit_times,
                           max_mea_time, is_add_noise);

        if (cur_time > max_depth) {
          break;
        }

        if (is_add_noise && cur_time != prev_time) {
          AddNoise(num_qubits, prev_time, *channel, ops);
        }

        if (level > 0) {
          ActsOn(bop.qubits, acts_on);
        }

        bop.time = TimeToNoisyTime(cur_time, is_add_noise);
        ops.push_back(std::move(*gate));
      } else if (auto* sym = symtab.Lookup(t.val)) {
        if (sym->IsReadOnly() || t.val == "wid" || t.val == "rid") {
          PE::Throw("unexpected const identifier '{}'", t.lc, t.val);
        }

        // Assignment.

        ClassicallyControlledOperation cop;
        cop.time = TimeToNoisyTime(cur_time, is_add_noise);
        cop.str.push_back(t.val);
        cop.kind = ClassicallyControlledOperation::kAssign;

        Token n = tok.Peek();

        if (n.val != "[") {
          if (n.val != "=") {
            PE::Throw("expected '='", n.lc);
          }

          tok();  // Consume '='.

          Assign(&cop, t, symtab, sym, tok);
        } else {
          n = tok.Peek(1);

          Index index = IndexParser::Run(symtab, tok);
          std::size_t i = EvalIndex(symtab, index);

          if (i >= sym->Size()) {
            PE::Throw("index {} is out of range", n.lc, i);
          }

          n = tok.Peek();

          if (n.val != "=") {
            PE::Throw("expected '='", n.lc);
          }

          tok();  // Consume '='.

          Assign(cop, t, symtab, sym, std::move(index), tok);
        }

        ops.push_back(std::move(cop));
      } else {
        auto hash = Hash(t.val);

        switch (hash) {
        case "c"_hash:
          {
            if (t.val != "c") {
              PE::Throw("unexpected token", t.lc);
            }

            // Controlled gate.

            std::set<unsigned> visited;
            auto controlled_by = GetNQubits(num_qubits, symtab, visited, tok);

            if (controlled_by.size() == 0) {
              PE::Throw("controlled gate should be controlled "
                        "by at least one qubit", t.lc);
            }

            cur_time = GetTime(t, time, cur_time, controlled_by, qubit_times);

            if (cur_time > max_depth) {
              return ops;
            }

            if (level > 0) {
              ActsOn(controlled_by, acts_on);
            }

            Token n = tok();

            if (auto gate = GetGate(n.val, num_qubits, symtab, tok)) {
              if (auto* g = OpGetAlternative<Gate<fp_type>>(*gate)) {
                for (unsigned q : g->qubits) {
                  if (visited.contains(q)) {
                    PE::Throw("control and gate qubit indices overlap", t.lc);
                  }
                }

                cur_time = GetTime(t, time, cur_time, g->qubits, qubit_times);

                if (cur_time > max_depth) {
                  return ops;
                }

                if (is_add_noise && cur_time != prev_time) {
                  AddNoise(num_qubits, prev_time, *channel, ops);
                }

                if (level > 0) {
                  ActsOn(g->qubits, acts_on);
                }

                g->time = TimeToNoisyTime(cur_time, is_add_noise);

                ops.push_back(
                    ControlledGate<fp_type>(*g, std::move(controlled_by)));
              } else {
                PE::Throw("runtime resolved gate cannot be controlled", t.lc);
              }
            } else {
              PE::Throw("expected a gate to be controlled", t.lc);
            }
          }

          break;
        case "m"_hash:
          {
            if (t.val != "m") {
              PE::Throw("unexpected token", t.lc);
            }

            // Measurement gate.

            std::set<unsigned> visited;
            auto qubits = GetNQubits(num_qubits, symtab, visited, tok);

            if (qubits.size() == 0) {
              PE::Throw(
                  "measurement gate should have at least one qubit", t.lc);
            }

            cur_time = GetTime(t, time, cur_time, qubits, qubit_times);
            unsigned mtime = TimeToNoisyTime(cur_time, is_add_noise);

            if (cur_time > max_depth) {
              return ops;
            }

            if (is_add_noise && cur_time != prev_time) {
              AddNoise(num_qubits, prev_time, *channel, ops);
            }

            if (level > 0) {
              ActsOn(qubits, acts_on);
            }

            if (tok.Peek().kind != Token::kIdentifier) {
              ops.push_back(CreateMeasurement(mtime, std::move(qubits)));
            } else {
              Token n = tok();

              if (IsKeyWord(n.val)) {
                PE::Throw("identifier '{}' is reserved", n.lc, n.val);
              }

              unsigned num_bits = qubits.size();
              symtab.Insert(n.val, Symbol{Symbol::Mea{0, num_bits, mtime}});

              ops.push_back(
                  CreateMeasurement(mtime, std::move(qubits), n.val));
            }
          }

          break;
        case "if"_hash:
        case "repeat"_hash:
          {
            if (t.val != "if" && t.val != "repeat") {
              PE::Throw("unexpected token", t.lc);
            }

            // Classical control if_else or repeat.

            ExpectCondition(tok);

            std::vector<bool> acts_on2(num_qubits, false);
            unsigned max_depth = std::numeric_limits<unsigned>::max();

            auto if_else = ClassicallyControlledOperation::kIfElse;
            auto repeat = ClassicallyControlledOperation::kRepeat;

            ClassicallyControlledOperation cop;

            cop.qubits.reserve(num_qubits);
            cop.kind = hash == "if"_hash ? if_else : repeat;
            cop.exprs.push_back(CondExprParser::Run(symtab, tok, f));

            enum State {
              kStart = 1,
              kElse,
            };

            State state = kStart;

            while (1) {
              unsigned scope_index = symtab.AddScope();
              symtab.EnterScope(scope_index);

              cop.sub_ops.push_back(ParseOps(level + 1, num_qubits, max_depth,
                                    symtab, obss, channel, acts_on2, tok));
              cop.scope_indices.push_back(scope_index);

              symtab.ExitScope();

              Token c = tok.Current();

              if (c.kind == Token::kEndOfFile) {
                PE::Throw("unexpected end of file", c.lc);
              }

              if (c.val == "end") {
                break;
              }

              if (cop.kind == repeat) {
                PE::Throw("expected 'end'", c.lc);
              }

              if (c.val == "else") {
                if (state == kElse) {
                  PE::Throw("too many 'else' clauses", c.lc);
                }

                if (tok.Peek().kind != Token::kDelimiter) {
                  PE::Throw("'else' should be followed be a delimiter", c.lc);
                }

                state = kElse;
                continue;
              }

              if (c.val == "elsif") {
                if (state == kElse) {
                  PE::Throw("'elsif' cannot follow 'else'", c.lc);
                }

                ExpectCondition(tok);
                cop.exprs.push_back(CondExprParser::Run(symtab, tok, f));
                continue;
              }

              if (state == kElse) {
                PE::Throw("expected 'end'", c.lc);
              } else {
                PE::Throw("expected 'else', 'elsif' or 'end'", c.lc);
              }
            }

            for (unsigned i = 0; i < num_qubits; ++i) {
              if (acts_on2[i]) {
                cop.qubits.push_back(i);
                if (level > 0) {
                  acts_on[i] = true;
                }
              }
            }

            cur_time = GetTime(t, time, cur_time, cop.qubits, qubit_times,
                               max_mea_time, is_add_noise);

            if (cur_time > max_depth) {
              return ops;
            }

            if (is_add_noise && cur_time != prev_time) {
              AddNoise(num_qubits, prev_time, *channel, ops);
            }

            cop.time = TimeToNoisyTime(cur_time, is_add_noise);
            ops.push_back(std::move(cop));
          }

          break;
        case "do"_hash:
          {
            if (t.val != "do") {
              PE::Throw("unexpected token", t.lc);
            }

            // Classical control do_while.

            std::vector<bool> acts_on2(num_qubits, false);
            unsigned max_depth = std::numeric_limits<unsigned>::max();

            auto f = [&symtab, &max_mea_time](
                const Symbol::Mea& mea, std::string_view name) {
              if (symtab.LookupInCurrentScope(name) == nullptr &&
                  symtab.LookupInPreviousScope(name) != nullptr) {
                int time = mea.time;
                if (time > max_mea_time) {
                  max_mea_time = time;
                }
              }
            };

            ClassicallyControlledOperation cop;

            unsigned scope_index = symtab.AddScope();
            symtab.EnterScope(scope_index);

            cop.sub_ops.push_back(ParseOps(level + 1, num_qubits, max_depth,
                                           symtab, obss, channel, acts_on2,
                                           tok));

            Token n = tok.Current();

            if (n.val != "while") {
              PE::Throw("expected 'while'", n.lc);
            }

            ExpectCondition(tok);

            cop.qubits.reserve(num_qubits);
            cop.exprs.push_back(CondExprParser::Run(symtab, tok, f));
            cop.kind = ClassicallyControlledOperation::kDoWhile;
            cop.scope_indices.push_back(scope_index);

            symtab.ExitScope();

            for (unsigned i = 0; i < num_qubits; ++i) {
              if (acts_on2[i]) {
                cop.qubits.push_back(i);
                if (level > 0) {
                  acts_on[i] = true;
                }
              }
            }

            cur_time = GetTime(t, time, cur_time, cop.qubits, qubit_times,
                               max_mea_time, is_add_noise);

            if (cur_time > max_depth) {
              return ops;
            }

            if (is_add_noise && cur_time != prev_time) {
              AddNoise(num_qubits, prev_time, *channel, ops);
            }

            cop.time = TimeToNoisyTime(cur_time, is_add_noise);
            ops.push_back(std::move(cop));
          }

          break;
        case "int"_hash:
        case "float"_hash:
          {
            Token n = tok();

            if (t.val == "int") {
              sym = DeclareSymbol<false, true>(n, cur_time, symtab, ops, tok);
            } else if (t.val == "float") {
              sym = DeclareSymbol<false, false>(n, cur_time, symtab, ops, tok);
            } else {
              PE::Throw("unexpected token", t.lc);
            }

            if (tok.Peek().val == "=") {
              tok();  // Consume '='.

              ClassicallyControlledOperation cop;
              cop.time = TimeToNoisyTime(cur_time, is_add_noise);
              cop.str.push_back(n.val);
              cop.kind = ClassicallyControlledOperation::kAssign;

              Assign(&cop, n, symtab, sym, tok);

              ops.push_back(std::move(cop));
            }
          }

          break;
        case "const"_hash:
          {
            if (t.val != "const") {
              PE::Throw("unexpected token", t.lc);
            }

            Symbol* sym = nullptr;
            Token n = tok();

            if (n.val == "int") {
              n = tok();
              sym = DeclareSymbol<true, true>(n, cur_time, symtab, ops, tok);
            } else if (n.val == "float") {
              n = tok();
              sym = DeclareSymbol<true, false>(n, cur_time, symtab, ops, tok);
            } else {
              PE::Throw("type '{}' after 'const' is unknown", n.lc, n.val);
            }

            if (tok.Peek().val == "=") {
              tok();  // Consume '='.
              Assign(nullptr, n, symtab, sym, tok);
            }
          }

          break;
        case "println"_hash:
          {
            if (t.val != "println") {
              PE::Throw("unexpected token", t.lc);
            }

            ClassicallyControlledOperation cop;

            Token n = tok.Peek();
            if (n.kind == Token::kString) {
              cop.str.push_back(n.val);
              tok();
            }

            cop.time = TimeToNoisyTime(cur_time, is_add_noise);
            cop.exprs.reserve(4);
            cop.kind = ClassicallyControlledOperation::kPrintLn;

            while (tok.Peek().kind != Token::kDelimiter &&
                   tok.Peek().kind != Token::kEndOfFile) {
              if (cop.exprs.size() == 4) {
                PE::Throw("'println' supports only up to "
                          "four expression arguments", t.lc);
              }

              cop.exprs.push_back(ExprParser::Run(symtab, tok));
            }

            ops.push_back(std::move(cop));
          }

          break;
        case "discard"_hash:
          {
            if (t.val != "discard") {
              PE::Throw("unexpected token", t.lc);
            }

            ExpectCondition(tok);

            ClassicallyControlledOperation cop;
            cop.time = TimeToNoisyTime(cur_time, is_add_noise);
            cop.exprs.push_back(CondExprParser::Run(symtab, tok));
            cop.kind = ClassicallyControlledOperation::kDiscard;

            ops.push_back(std::move(cop));
          }

          break;
        case "histogram"_hash:
          {
            if (t.val != "histogram") {
              PE::Throw("unexpected token", t.lc);
            }

            Token n = tok.Peek();

            if (n.kind != Token::kIdentifier) {
              PE::Throw("expected a measurement tag", n.lc);
            }

            auto* sym = symtab.Lookup(n.val);

            if (sym == nullptr) {
              PE::Throw("identifier '{}' is not defined", n.lc, n.val);
            }

            if (!sym->HoldsMea()) {
              PE::Throw("identifier '{}' is not a measurement", n.lc, n.val);
            }

            if (obss.Lookup(n.val) != nullptr) {
              PE::Throw("histogram for '{}' is already defined", n.lc, n.val);
            }

            obss.Insert(n.val, MeasurementHistogram(sym->Size()));

            tok();  // Consume idetifier.
          }

          break;
        case "else"_hash:
          if (t.val != "else") {
            PE::Throw("unexpected token", t.lc);
          }

          return ops;
        case "elsif"_hash:
          if (t.val != "elsif") {
            PE::Throw("unexpected token", t.lc);
          }

          return ops;
        case "end"_hash:
          if (t.val != "end") {
            PE::Throw("unexpected token", t.lc);
          }

          return ops;
        case "while"_hash:
          if (t.val != "while") {
            PE::Throw("unexpected token", t.lc);
          }

          return ops;
        default:
          PE::Throw("unexpected token", t.lc);
        }
      }

      t = tok();

      if (t.kind != Token::kEndOfFile && t.kind != Token::kDelimiter) {
        PE::Throw("expected a delimiter or end of file", t.lc);
      }
    }

    return ops;
  }

  static bool IsKeyWord(std::string_view s) {
    static std::unordered_set<std::string_view> keywords = {
      "p", "id1", "h", "t", "x", "y", "z", "x_1_2", "y_1_2", "rx", "ry", "rz",
      "rxy", "hz_1_2", "s", "id2", "cz", "cx", "cnot", "sw", "is", "fs", "cp",
      "c", "m", "if", "elsif", "else", "end", "repeat", "do", "while", "int",
      "float", "const", "println", "discard", "histogram",
    };

    return s.size() <= 8 && keywords.contains(s);
  }

  static void ExpectCondition(Tokenizer& tok) {
    Token n = tok.Peek();
    if (n.kind == Token::kDelimiter || n.kind == Token::kEndOfFile) {
      PE::Throw("expected a condition", n.lc);
    }
  }

  template <bool read_only, bool int_symbol>
  static Symbol* DeclareSymbol(Token t, unsigned cur_time, SymTable& symtab,
                               Operations& ops, Tokenizer& tok) {
    if (t.kind != Token::kIdentifier) {
      PE::Throw("expected an identifier", t.lc);
    }

    if (auto* sym = symtab.LookupInCurrentScope(t.val)) {
      PE::Throw("identifier '{}' is already defined", t.lc, t.val);
    }

    if (IsKeyWord(t.val)) {
      PE::Throw("identifier '{}' is reserved", t.lc, t.val);
    }

    Token n = tok.Peek();

    if (n.val != "(") {
      if constexpr (int_symbol) {
        return symtab.Insert(t.val, Symbol{Symbol::Int{0}, read_only});
      } else {
        return symtab.Insert(t.val, Symbol{Symbol::Float{0.0}, read_only});
      }
    } else {
      // Vector.

      tok();  // Consume '('.

      auto e = IntExprParser::Run(symtab, tok);

      tok();  // Consume ')'.

      if (!IsConstExpr(symtab, e)) {
        PE::Throw("unknown vector size", n.lc);
      }

      unsigned size = EvalIntExpr(symtab, e);
      auto v = std::views::iota(unsigned{0}, size);

      if constexpr (int_symbol) {
        return symtab.Insert(
            t.val, Symbol{Symbol::IntVector(v.begin(), v.end()), read_only});
      } else {
        return symtab.Insert(
            t.val, Symbol{Symbol::FloatVector(v.begin(), v.end()), read_only});
      }
    }
  }

  static auto Assign(ClassicallyControlledOperation* cop, Token t,
                     const SymTable& symtab, Symbol* sym, Tokenizer& tok) {
    if (cop != nullptr && sym->IsReadOnly()) {
      PE::Throw("cannot assign to the const identifier '{}'", t.lc, t.val);
    }

    if (sym->HoldsMea()) {
      PE::Throw("cannot assign to the measurement identifier '{}'",
                t.lc, t.val);
    }

    Token n = tok.Peek();
    bool is_float = sym->IsFloat();

    if (!sym->HoldsVector()) {
      auto e = ExprParser::Run(symtab, tok);

      if (cop == nullptr && !IsConstExpr(symtab, e)) {
        PE::Throw("cannot assign the const identifier '{}'", n.lc, t.val);
      }

      if (is_float) {
        sym->Assign(EvalExpr(symtab, e));
      } else {
        if (!IsConvertibleToInt(symtab, e)) {
          PE::Throw("'float' cannot be assigned to the integer identifer '{}'",
                    n.lc, t.val);
        }

        sym->Assign(EvalIntExpr(symtab, e));
      }

      if (cop != nullptr) {
        cop->exprs.push_back(std::move(e));
      }
    } else {
      unsigned count = 0;
      unsigned size = sym->Size();

      if (cop != nullptr) {
        cop->exprs.reserve(size);
      }

      while (n.kind != Token::kDelimiter && n.kind != Token::kEndOfFile) {
        if (count >= size) {
          PE::Throw("too many expressions in vector assignment", n.lc);
        }

        auto e = ExprParser::Run(symtab, tok);

        if (cop == nullptr && !IsConstExpr(symtab, e)) {
          PE::Throw("cannot assign to the const identifier '{}'", n.lc, t.val);
        }

        if (is_float) {
          sym->Assign(EvalExpr(symtab, e), count);
        } else {
          if (!IsConvertibleToInt(symtab, e)) {
            PE::Throw(
                "'float' cannot be assigned to the integer identifer '{}'",
                n.lc, t.val);
          }

          sym->Assign(EvalIntExpr(symtab, e), count);
        }

        if (cop != nullptr) {
          cop->exprs.push_back(std::move(e));
        }

        ++count;
        n = tok.Peek();
      }
    }

    return cop;
  }

  static void Assign(ClassicallyControlledOperation& cop, Token t,
                     const SymTable& symtab, Symbol* sym, Index&& index,
                     Tokenizer& tok) {
    if (sym->IsReadOnly()) {
      PE::Throw("cannot assign to the const identifier '{}'", t.lc, t.val);
    }

    if (sym->HoldsMea()) {
      PE::Throw("cannot assign to the measurement identifier '{}'",
                t.lc, t.val);
    }

    Token n = tok.Peek();

    auto e = ExprParser::Run(symtab, tok);

    if (sym->IsFloat()) {
      sym->Assign(EvalExpr(symtab, e), EvalIndex(symtab, index));
    } else {
      if (!IsConvertibleToInt(symtab, e)) {
        PE::Throw("'float' cannot be assigned to the integer identifer '{}'",
                  n.lc, t.val);
      }

      sym->Assign(EvalIntExpr(symtab, e), EvalIndex(symtab, index));
    }

    cop.exprs.push_back(std::move(e));
    cop.indices.push_back(std::move(index));
  }

  static void ActsOn(
      const std::vector<unsigned>& qubits, std::vector<bool>& acts_on) {
    for (auto q : qubits) {
      acts_on[q] = true;
    }
  }

  static unsigned GetTime(const Token& t, int time, unsigned cur_time,
                          const std::vector<unsigned>& qubits,
                          std::vector<int>& qubit_times,
                          int max_mea_time = -2, bool is_add_noise = false) {
    if (time > -1) {
      if (static_cast<unsigned>(time) < cur_time) {
        PE::Throw("gate time is not in order", t.lc);
      }

      for (unsigned q : qubits) {
        if (qubit_times[q] == time) {
          PE::Throw("qubit indices overlap at time {}", t.lc, time);
        } else if (qubit_times[q] > time) {
          PE::Throw("gate time is not in order", t.lc);
        }

        qubit_times[q] = time;
      }

      cur_time = time;
    } else {
      int time = -1;

      for (unsigned q : qubits) {
        if (qubit_times[q] > time) {
          time = qubit_times[q];
        }
      }

      max_mea_time = NoisyTimeToTime(max_mea_time, is_add_noise);

      if (static_cast<unsigned>(time) == cur_time ||
          static_cast<unsigned>(max_mea_time) == cur_time) {
        ++cur_time;
      }

      for (unsigned q : qubits) {
        qubit_times[q] = cur_time;
      }
    }

    return cur_time;
  };

  static bool IsAddNoise(bool have_noise, unsigned level) {
    return have_noise && level == 0;
  }

  static unsigned TimeToNoisyTime(unsigned time, bool noisy) {
    return noisy ? 2 * time : time;
  }

  static unsigned NoisyTimeToTime(unsigned time, bool noisy) {
    return noisy ? time / 2 : time;
  }

  static void AddNoise(
      unsigned num_qubits, unsigned time, Channel ch, Operations& ops) {
    time = TimeToNoisyTime(time, true) + 1;

    for (unsigned q = 0; q < num_qubits; ++q) {
      ch.time = time;
      ch.qubits[0] = q;
      for (auto& kop : ch.kops) {
        if (!kop.qubits.empty()) {
          kop.qubits[0] = q;
        }

        for (auto& op : kop.ops) {
          op.time = time;
          op.qubits[0] = q;
        }
      }
      ops.push_back(ch);
    }
  }

  static auto GetOneQubit(
      unsigned num_qubits, const SymTable& symtab, Tokenizer& tok) {
    Token n = tok.Peek();

     if (n.kind == Token::kDelimiter || n.kind == Token::kEndOfFile) {
      PE::Throw("expected a qubit index", n.lc);
    }

    Expr e = IntExprParser::Run(symtab, tok);
    if (!IsConstExpr(symtab, e)) {
      PE::Throw("qubit index should be const", n.lc);
    }

    unsigned q = EvalIntExpr(symtab, e);
    if (q >= num_qubits) {
      PE::Throw("qubit index {} is out of range", n.lc, q);
    }

    return q;
  }

  static auto GetTwoQubits(
      unsigned num_qubits, const SymTable& symtab, Tokenizer& tok) {
    Token n0 = tok.Peek();

    if (n0.kind == Token::kDelimiter || n0.kind == Token::kEndOfFile) {
      PE::Throw("expected a qubit index", n0.lc);
    }

    Expr e0 = IntExprParser::Run(symtab, tok);
    if (!IsConstExpr(symtab, e0)) {
      PE::Throw("qubit index must be const", n0.lc);
    }

    unsigned q0 = EvalIntExpr(symtab, e0);
    if (q0 >= num_qubits) {
      PE::Throw("qubit index {} is out of range", n0.lc, q0);
    }

    Token n1 = tok.Peek();

    if (n1.kind == Token::kDelimiter || n1.kind == Token::kEndOfFile) {
      PE::Throw("expected a qubit index", n1.lc);
    }

    Expr e1 = IntExprParser::Run(symtab, tok);
    if (!IsConstExpr(symtab, e1)) {
      PE::Throw("qubit index must be const", n1.lc);
    }

    unsigned q1 = EvalIntExpr(symtab, e1);
    if (q1 >= num_qubits) {
      PE::Throw("qubit index {} is out of range", n1.lc, q1);
    }

    if (q0 == q1) {
      PE::Throw("the second qubit index is the same as the first one", n1.lc);
    }

    return std::make_pair(q0, q1);
  }

  static auto GetNQubits(unsigned num_qubits, const SymTable& symtab,
                         std::set<unsigned>& visited, Tokenizer& tok) {
    std::vector<unsigned> qubits;
    qubits.reserve(num_qubits);

    Token n = tok.Peek();

    while (n.kind != Token::kDelimiter) {
      if (n.kind == Token::kIdentifier &&
          symtab.LookupInCurrentScope(n.val) == nullptr) {
        break;
      }

      Expr e = IntExprParser::Run(symtab, tok);
      if (!IsConstExpr(symtab, e)) {
        PE::Throw("qubit index must be const", n.lc);
      }

      unsigned q = EvalIntExpr(symtab, e);
      if (q >= num_qubits) {
        PE::Throw("qubit index {} is out of range", n.lc, q);
      } else if (visited.contains(q)) {
        PE::Throw("repeated qubit indices", n.lc);
      }

      qubits.push_back(q);
      visited.insert(q);

      n = tok.Peek();
    }

    return qubits;
  }

  template <typename... Args>
  static std::optional<Operation> GetGate(
      std::string_view name, unsigned num_qubits,
      const SymTable& symtab, Tokenizer& tok, Args&&... args) {
    if (name.size() > 6) {
      return std::nullopt;
    }

    switch (Hash(name)) {
    case "p"_hash:
      {
        if (name != "p") {
          break;
        }

        auto e = ExprParser::Run(symtab, tok, args...);

        if (IsConstExpr(symtab, e)) {
          double phi = EvalExpr(symtab, e);
          return GateGPh<fp_type>::Create(0, phi);
        } else {
          RuntimeResolvedGate g{GateGPh<fp_type>::Create(0, 0.0)};
          g.param_exprs = {std::move(e)};
          g.matrix_func = [](const auto& p, Matrix<fp_type>& m) {
            return GateGPh<fp_type>::UpdateMatrix(p[0], m);
          };

          return g;
        }
      }
    case "id1"_hash:
      {
        if (name != "id1") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateId1<fp_type>::Create(0, q0);
      }
    case "h"_hash:
      {
        if (name != "h") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateHd<fp_type>::Create(0, q0);
      }
    case "t"_hash:
      {
        if (name != "t") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateT<fp_type>::Create(0, q0);
      }
    case "x"_hash:
      {
        if (name != "x") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateX<fp_type>::Create(0, q0);
      }
    case "y"_hash:
      {
        if (name != "y") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateY<fp_type>::Create(0, q0);
      }
    case "z"_hash:
      {
        if (name != "z") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateZ<fp_type>::Create(0, q0);
      }
    case "x_1_2"_hash:
      {
        if (name != "x_1_2") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateX2<fp_type>::Create(0, q0);
      }
    case "y_1_2"_hash:
      {
        if (name != "y_1_2") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateY2<fp_type>::Create(0, q0);
      }
    case "rx"_hash:
      {
        if (name != "rx") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        auto e = ExprParser::Run(symtab, tok, args...);

        if (IsConstExpr(symtab, e)) {
          double phi = EvalExpr(symtab, e);
          return GateRX<fp_type>::Create(0, q0, phi);
        } else {
          RuntimeResolvedGate g{GateRX<fp_type>::Create(0, q0, 0.0)};
          g.param_exprs = {std::move(e)};
          g.matrix_func = [](const auto& p, Matrix<fp_type>& m) {
            return GateRX<fp_type>::UpdateMatrix(p[0], m);
          };

          return g;
        }
      }
    case "ry"_hash:
      {
        if (name != "ry") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        auto e = ExprParser::Run(symtab, tok, args...);

        if (IsConstExpr(symtab, e)) {
          double phi = EvalExpr(symtab, e);
          return GateRY<fp_type>::Create(0, q0, phi);
        } else {
          RuntimeResolvedGate g{GateRY<fp_type>::Create(0, q0, 0.0)};
          g.param_exprs = {std::move(e)};
          g.matrix_func = [](const auto& p, Matrix<fp_type>& m) {
            return GateRY<fp_type>::UpdateMatrix(p[0], m);
          };

          return g;
        }
      }
    case "rz"_hash:
      {
        if (name != "rz") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        auto e = ExprParser::Run(symtab, tok, args...);

        if (IsConstExpr(symtab, e)) {
          double phi = EvalExpr(symtab, e);
          return GateRZ<fp_type>::Create(0, q0, phi);
        } else {
          RuntimeResolvedGate g{GateRZ<fp_type>::Create(0, q0, 0.0)};
          g.param_exprs = {std::move(e)};
          g.matrix_func = [](const auto& p, Matrix<fp_type>& m) {
            return GateRZ<fp_type>::UpdateMatrix(p[0], m);
          };

          return g;
        }
      }
    case "rxy"_hash:
      {
        if (name != "rxy") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        auto e1 = ExprParser::Run(symtab, tok, args...);

        Token t = tok.Peek();
        if (t.kind == Token::kDelimiter) {
          PE::Throw("expected the second gate parameter", t.lc);
        }

        auto e2 = ExprParser::Run(symtab, tok, args...);

        if (IsConstExpr(symtab, e1) && IsConstExpr(symtab, e2)) {
          double theta = EvalExpr(symtab, e1);
          double phi = EvalExpr(symtab, e2);
          return GateRXY<fp_type>::Create(0, q0, theta, phi);
        } else {
          RuntimeResolvedGate g{GateRXY<fp_type>::Create(0, q0, 0.0, 0.0)};
          g.param_exprs = {std::move(e1), std::move(e2)};
          g.matrix_func = [](const auto& p, Matrix<fp_type>& m) {
            return GateRXY<fp_type>::UpdateMatrix(p[0], p[1], m);
          };

          return g;
        }
      }
    case "hz_1_2"_hash:
      {
        if (name != "hz_1_2") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateHZ2<fp_type>::Create(0, q0);
      }
    case "s"_hash:
      {
        if (name != "s") {
          break;
        }

        auto q0 = GetOneQubit(num_qubits, symtab, tok);
        return GateS<fp_type>::Create(0, q0);
      }
    case "id2"_hash:
      {
        if (name != "id2") {
          break;
        }

        auto [q0, q1] = GetTwoQubits(num_qubits, symtab, tok);
        return GateId2<fp_type>::Create(0, q0, q1);
      }
    case "cz"_hash:
      {
        if (name != "cz") {
          break;
        }

        auto [q0, q1] = GetTwoQubits(num_qubits, symtab, tok);
        return GateCZ<fp_type>::Create(0, q0, q1);
      };
    case "cx"_hash:
    case "cnot"_hash:
      {
        if (name != "cx" && name != "cnot") {
          break;
        }

        auto [q0, q1] = GetTwoQubits(num_qubits, symtab, tok);
        return GateCNot<fp_type>::Create(0, q0, q1);
      }
    case "sw"_hash:
      {
        if (name != "sw") {
          break;
        }

        auto [q0, q1] = GetTwoQubits(num_qubits, symtab, tok);
        return GateSwap<fp_type>::Create(0, q0, q1);
      }
    case "is"_hash:
      {
        if (name != "is") {
          break;
        }

        auto [q0, q1] = GetTwoQubits(num_qubits, symtab, tok);
        return GateIS<fp_type>::Create(0, q0, q1);
      }
    case "fs"_hash:
      {
        if (name != "fs") {
          break;
        }

        auto [q0, q1] = GetTwoQubits(num_qubits, symtab, tok);
        auto e1 = ExprParser::Run(symtab, tok, args...);

        Token t = tok.Peek();
        if (t.kind == Token::kDelimiter) {
          PE::Throw("expected the second gate parameter", t.lc);
        }

        auto e2 = ExprParser::Run(symtab, tok, args...);

        if (IsConstExpr(symtab, e1) && IsConstExpr(symtab, e2)) {
          double theta = EvalExpr(symtab, e1);
          double phi = EvalExpr(symtab, e2);
          return GateFS<fp_type>::Create(0, q0, q1, theta, phi);
        } else {
          RuntimeResolvedGate g{GateFS<fp_type>::Create(0, q0, q1, 0.0, 0.0)};
          g.param_exprs = {std::move(e1), std::move(e2)};
          g.matrix_func = [](const auto& p, Matrix<fp_type>& m) {
            return GateFS<fp_type>::UpdateMatrix(p[0], p[1], m);
          };

          return g;
        }
      }
    case "cp"_hash:
      {
        if (name != "cp") {
          break;
        }

        auto [q0, q1] = GetTwoQubits(num_qubits, symtab, tok);
        auto e = ExprParser::Run(symtab, tok, args...);

        if (IsConstExpr(symtab, e)) {
          double phi = EvalExpr(symtab, e);
          return GateCP<fp_type>::Create(0, q0, q1, phi);
        } else {
          RuntimeResolvedGate g{GateCP<fp_type>::Create(0, q0, q1, 0.0)};
          g.param_exprs = {std::move(e)};
          g.matrix_func = [](const auto& p, Matrix<fp_type>& m) {
            return GateRX<fp_type>::UpdateMatrix(p[0], m);
          };

          return g;
        }
      }
    }

    return std::nullopt;
  }

  static Channel GetChannel(const SymTable& symtab, Tokenizer& tok) {
    Token t = tok();

    if (t.kind != Token::kIdentifier) {
      PE::Throw("expected a noise channel", t.lc);
    }

    auto hash = Hash(t.val);

    if (hash == "reset"_hash) {
      return Cirq::ResetChannel<fp_type>::Create(0, 0);
    }

    Token n = tok.Peek();

    Expr e1 = ExprParser::Run(symtab, tok);
    if (!IsConstExpr(symtab, e1)) {
      PE::Throw("expected a const expression", n.lc);
    }

    TFloat p1 = EvalExpr(symtab, e1);

    switch (hash) {
    case "amplitude_damp"_hash:
      return Cirq::AmplitudeDampingChannel<fp_type>::Create(0, 0, p1);
    case "phase_damp"_hash:
      return Cirq::PhaseDampingChannel<fp_type>::Create(0, 0, p1);
    case "phase_flip"_hash:
      return Cirq::PhaseFlipChannel<fp_type>::Create(0, 0, p1);
    case "bit_flip"_hash:
      return Cirq::BitFlipChannel<fp_type>::Create(0, 0, p1);
    case "depolarize"_hash:
      return Cirq::DepolarizingChannel<fp_type>::Create(0, 0, p1);
    case "generalized_amplitude_damp"_hash:
      {
        n = tok.Peek();

        Expr e2 = ExprParser::Run(symtab, tok);
        if (!IsConstExpr(symtab, e2)) {
          PE::Throw("expected a const expression", n.lc);
        }

        TFloat p2 = EvalExpr(symtab, e2);

        return Cirq::GeneralizedAmplitudeDampingChannel<fp_type>::Create(
            0, 0, p1, p2);
      }
    case "asymmetric_depolarize"_hash:
      {
        n = tok.Peek();

        Expr e2 = ExprParser::Run(symtab, tok);
        if (!IsConstExpr(symtab, e2)) {
          PE::Throw("expected a const expression", n.lc);
        }

        TFloat p2 = EvalExpr(symtab, e2);

        n = tok.Peek();

        Expr e3 = ExprParser::Run(symtab, tok);
        if (!IsConstExpr(symtab, e3)) {
          PE::Throw("expected a const expression", n.lc);
        }

        TFloat p3 = EvalExpr(symtab, e3);

        return Cirq::AsymmetricDepolarizingChannel<fp_type>::Create(
            0, 0, p1, p2, p3);
      }
    default:
      PE::Throw("unknown channel '{}'", t.lc, t.val);
    }

    return {};
  }
};

}  // namespace cc

template <typename FP, typename ParserError = cc::ParserError,
          typename RuntimeError = cc::RuntimeError>
using CircuitQsimParser = cc::CircuitQsimParser<FP, ParserError, RuntimeError>;

}  // namespace qsim

#endif  // PARSER_H_
