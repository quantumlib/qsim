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

#ifndef CLASSICAL_CONTROL_SYMTAB_H_
#define CLASSICAL_CONTROL_SYMTAB_H_

#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "classical_control_expr.h"
#include "error.h"

namespace qsim::cc {

/**
 * Manages scoped symbols for the quantum circuit parser and interpreter.
 * Maintains a collection of scope tables (`symtab`) and an active call stack
 * (`path`). Scopes are created during parsing and later re-entered during
 * interpretation in the same topological sequence.
 */
struct SymTable {
  /** Constructs an empty symbol table with pre-allocated storage. */
  SymTable() {
    symtab.reserve(1024);
    path.reserve(32);
  }

  /**
   * Creates a new scope and returns its global index.
   * @return The 0-based index of the newly created scope.
   */
  std::size_t AddScope() {
    symtab.emplace_back();
    return symtab.size() - 1;
  }

  /**
   * Enters the scope by pushing a scope index onto the active scope path.
   * @param k The scope index returned previously by AddScope().
   * @throws std::runtime_error If k is out of range.
   */
  void EnterScope(std::size_t k) {
    if (k >= symtab.size()) {
      Error::Throw("the scope index {} is too large", k);
      return;
    }

    path.push_back(k);
  }

  /** Exits the current active scope by popping it from the scope path. */
  void ExitScope() {
    if (path.size() > 0) {
      path.pop_back();
    }
  }

  /**
   * Checks whether any scopes have been created.
   * @return True if no scopes exist in the table.
   */
  bool Empty() const {
    return symtab.empty();
  }

  /**
   * Inserts or retrieves a symbol in the current active scope.
   * If a symbol with name already exists in the current scope, the existing
   * symbol pointer is returned. Otherwise, sym is moved into place.
   * @param name Name identifier for the symbol.
   * @param sym Symbol rvalue instance to insert.
   * @return Pointer to the inserted or existing Symbol in the current scope.
   */
  Symbol* Insert(std::string_view name, Symbol&& sym) {
    auto& cur = symtab[path.back()];

    if (auto s = cur.find(name); s != cur.end()) {
      return &s->second;
    }

    return &(cur[name] = std::move(sym));
  }

  /**
   * Searches for a symbol up the scope chain.
   * @param name Symbol name to search for.
   * @return Pointer to matching Symbol, or nullptr if not found.
   */
  Symbol* Lookup(std::string_view name) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      auto& symbols = symtab[*it];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    return nullptr;
  }

  /** Const version of Lookup. */
  const Symbol* Lookup(std::string_view name) const {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      const auto& symbols = symtab[*it];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    return nullptr;
  }

  /**
   * Searches for a symbol exclusively in the current active scope.
   * @param name Symbol name to search for.
   * @return Pointer to matching Symbol in current scope, or nullptr if
   *   not found.
   */
  Symbol* LookupInCurrentScope(std::string_view name) {
    if (!path.empty()) {
      auto& symbols = symtab[path.back()];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    return nullptr;
  }

  /** Const version of LookupInCurrentScope. */
  const Symbol* LookupInCurrentScope(std::string_view name) const {
    if (!path.empty()) {
      const auto& symbols = symtab[path.back()];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    return nullptr;
  }

  /**
   * Searches for a symbol in the parent/enclosing scope (one level up path).
   * @param name Symbol name to search for.
   * @return Pointer to matching Symbol in previous scope, or nullptr
   *   if not found.
   */
  Symbol* LookupInPreviousScope(std::string_view name) {
    if (path.size() > 1) {
      auto& symbols = symtab[path[path.size() - 2]];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    return nullptr;
  }

  /** Const version of LookupInPreviousScope. */
  const Symbol* LookupInPreviousScope(std::string_view name) const {
    if (path.size() > 1) {
      const auto& symbols = symtab[path[path.size() - 2]];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    return nullptr;
  }

  /**
   * Searches for a symbol up the scope chain or throw an error if not found.
   * @tparam Args Format string argument types.
   * @param name Symbol name to search for.
   * @param message Exception message (format) string.
   * @param args Arguments passed to Error::Throw.
   * @return Pointer to matching Symbol.
   * @throws std::runtime_error If symbol is not found.
   */
  template <typename... Args>
  Symbol* LookupOrError(
      std::string_view name, std::string_view message, Args&&... args) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      auto& symbols = symtab[*it];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    Error::Throw(message, args...);
    return nullptr;
  }

  /** Const version of LookupOrError. */
  template <typename... Args>
  const Symbol* LookupOrError(
      std::string_view name, std::string_view message, Args&&... args) const {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      const auto& symbols = symtab[*it];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    Error::Throw(message, args...);
    return nullptr;
  }

  /**
   * Searches for a symbol in the current scope or throw an error if not found.
   * @tparam Args Format string argument types.
   * @param name Symbol name to search for.
   * @param message Exception message (format) string.
   * @param args Arguments passed to Error::Throw.
   * @return Pointer to matching Symbol in current scope.
   * @throws std::runtime_error If symbol is not found.
   */
  template <typename... Args>
  Symbol* LookupInCurrentScopeOrError(
      std::string_view name, std::string_view message, Args&&... args) {
    if (!path.empty()) {
      auto& symbols = symtab[path.back()];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    Error::Throw(message, args...);
    return nullptr;
  }

  /** Const version of LookupInCurrentScopeOrError. */
  template <typename... Args>
  const Symbol* LookupInCurrentScopeOrError(
      std::string_view name, std::string_view message, Args&&... args) const {
    if (!path.empty()) {
      const auto& symbols = symtab[path.back()];
      if (auto s = symbols.find(name); s != symbols.end()) {
        return &s->second;
      }
    }

    Error::Throw(message, args...);
    return nullptr;
  }

 private:
  std::vector<std::unordered_map<std::string_view, Symbol>> symtab;
  std::vector<unsigned> path;
};

}  // namespace qsim::cc

#endif  // CLASSICAL_CONTROL_SYMTAB_H_
