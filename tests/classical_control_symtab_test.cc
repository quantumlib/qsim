#include <gtest/gtest.h>

#include <stdexcept>

#include "../lib/classical_control_symbol.h"
#include "../lib/classical_control_symtab.h"

namespace qsim::cc {
namespace {

TEST(SymTableTest, InitialStateIsEmpty) {
  SymTable table;
  EXPECT_TRUE(table.Empty());
}

TEST(SymTableTest, AddAndEnterScope) {
  SymTable table;
  std::size_t s0 = table.AddScope();
  EXPECT_EQ(s0, 0u);
  EXPECT_FALSE(table.Empty());

  table.EnterScope(s0);

  // Verify insertion in current active scope
  Symbol* inserted = table.Insert("x", Symbol(Symbol::Int{10}));
  ASSERT_NE(inserted, nullptr);
  EXPECT_EQ(inserted->GetInt(), 10);

  table.ExitScope();
}

TEST(SymTableTest, OutOfBoundsScopeIndexThrows) {
  SymTable table;
  EXPECT_THROW(table.EnterScope(99), std::runtime_error);
}

TEST(SymTableTest, LexicalScopingAndShadowing) {
  SymTable table;

  // Root Scope
  auto s0 = table.AddScope();
  table.EnterScope(s0);
  table.Insert("x", Symbol(Symbol::Int{1}));
  table.Insert("y", Symbol(Symbol::Int{2}));

  // Child Scope
  auto s1 = table.AddScope();
  table.EnterScope(s1);
  table.Insert("x", Symbol(Symbol::Int{100}));  // Shadows root 'x'

  // Lookups in Child Scope
  EXPECT_EQ(table.Lookup("x")->GetInt(), 100);  // Inner 'x'
  EXPECT_EQ(table.Lookup("y")->GetInt(), 2);    // Inherited 'y' from root

  // Lookup in specific scopes
  EXPECT_EQ(table.LookupInCurrentScope("x")->GetInt(), 100);
  EXPECT_EQ(table.LookupInPreviousScope("x")->GetInt(), 1);
  EXPECT_EQ(table.LookupInCurrentScope("y"), nullptr);  // 'y' is in prev scope

  table.ExitScope();

  // Back in Root Scope
  EXPECT_EQ(table.Lookup("x")->GetInt(), 1);
}

TEST(SymTableTest, ReEnteringScopesSimulatesInterpreterPhase) {
  SymTable table;

  // Parser phase: create scope graph
  auto root = table.AddScope();
  table.EnterScope(root);
  table.Insert("a", Symbol(Symbol::Float{1.5}));

  auto inner = table.AddScope();
  table.EnterScope(inner);
  table.Insert("b", Symbol(Symbol::Float{2.5}));

  table.ExitScope();  // Exit inner
  table.ExitScope();  // Exit root

  // Interpreter phase: re-enter recorded scope indices
  table.EnterScope(root);
  EXPECT_NE(table.Lookup("a"), nullptr);
  EXPECT_EQ(table.Lookup("b"), nullptr);

  table.EnterScope(inner);
  EXPECT_NE(table.Lookup("a"), nullptr);
  EXPECT_NE(table.Lookup("b"), nullptr);

  table.ExitScope();
  table.ExitScope();
}

TEST(SymTableTest, LookupOrErrorMethods) {
  SymTable table;
  auto s0 = table.AddScope();
  table.EnterScope(s0);

  table.Insert("valid_var", Symbol(Symbol::Int{42}));

  // Success path
  EXPECT_NO_THROW({
    Symbol* s = table.LookupOrError(
        "valid_var", "Variable '{}' not found", "valid_var");
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->GetInt(), 42);
  });

  // Failure paths
  EXPECT_THROW(
      table.LookupOrError("missing_var", "Variable not declared"),
      std::runtime_error);

  EXPECT_THROW(
      table.LookupInCurrentScopeOrError("missing_var", "Not in current scope"),
      std::runtime_error);
}

TEST(SymTableTest, ConstLookupMethods) {
  SymTable table;
  auto s0 = table.AddScope();
  table.EnterScope(s0);
  table.Insert("const_var", Symbol(Symbol::Int{7}));

  const SymTable& const_table = table;

  const Symbol* s1 = const_table.Lookup("const_var");
  ASSERT_NE(s1, nullptr);
  EXPECT_EQ(s1->GetInt(), 7);

  const Symbol* s2 = const_table.LookupInCurrentScope("const_var");
  ASSERT_NE(s2, nullptr);

  EXPECT_EQ(const_table.Lookup("nonexistent"), nullptr);
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
