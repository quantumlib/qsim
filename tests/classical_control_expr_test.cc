#include <gtest/gtest.h>

#include <stdexcept>

#include "../lib/classical_control_expr.h"
#include "../lib/classical_control_symbol.h"
#include "../lib/classical_control_symtab.h"

namespace qsim::cc {
namespace {

// Helper fixture to set up a populated SymTable
class ExprTest : public ::testing::Test {
 protected:
  SymTable symtab;

  void SetUp() override {
    auto scope = symtab.AddScope();
    symtab.EnterScope(scope);

    // Insert constants and variables
    symtab.Insert("const_int", Symbol(VInt{42}, /*read_only=*/true));
    symtab.Insert("const_float", Symbol(VFloat{3.14}, /*read_only=*/true));
    symtab.Insert("var_int", Symbol(VInt{10}, /*read_only=*/false));

    // Vector symbols
    const Symbol::IntVector const_int_vec = {100, 200, 300};
    symtab.Insert("const_int_vec", Symbol(const_int_vec, /*read_only=*/true));
    Symbol::IntVector var_int_vec = {100, 200, 300};
    symtab.Insert("var_int_vec", Symbol(var_int_vec, /*read_only=*/false));
    Symbol::FloatVector const_float_vec = {100.0, 200.0, 300.0};
    symtab.Insert("float_vec", Symbol(const_float_vec, /*read_only=*/true));

    // Quantum Measurement symbol (0b1011 = bit0:1, bit1:1, bit2:0, bit3:1)
    Symbol::Mea mea{.bits = 0b1011, .num_bits = 4, .time = 0};
    symtab.Insert("m_out", Symbol(mea)); // Mea is always read_only = false
  }
};

TEST_F(ExprTest, IndexEvaluation) {
  Index i_int = VInt{2};
  Index i_int_negative = VInt{-1};
  Index i_bool = VBool{true};
  Index i_func = EvalIntFn{[](const SymTable&) { return VInt{1}; }};

  EXPECT_TRUE(IsConstIndex(i_int));
  EXPECT_TRUE(IsConstIndex(i_bool));
  EXPECT_FALSE(IsConstIndex(i_func));

  EXPECT_EQ(EvalIndex(symtab, i_int), 2u);
  EXPECT_EQ(EvalIndex(symtab, i_bool), 1u);
  EXPECT_EQ(EvalIndex(symtab, i_func), 1u);
  EXPECT_THROW(EvalIndex(symtab, i_int_negative), std::runtime_error);
}

TEST_F(ExprTest, EvalLiterals) {
  Expr e_int = VInt{5};
  Expr e_float = VFloat{2.5};
  Expr e_bool = VBool{true};

  EXPECT_DOUBLE_EQ(EvalFloatExpr(symtab, e_int), 5.0);
  EXPECT_DOUBLE_EQ(EvalFloatExpr(symtab, e_float), 2.5);
  EXPECT_DOUBLE_EQ(EvalFloatExpr(symtab, e_bool), 1.0);

  EXPECT_EQ(EvalIntExpr(symtab, e_int), 5);
  EXPECT_EQ(EvalIntExpr(symtab, e_bool), 1);

  EXPECT_TRUE(EvalCondExpr(symtab, e_int));
  EXPECT_TRUE(EvalCondExpr(symtab, e_bool));
}

TEST_F(ExprTest, EvalNestedFunctions) {
  Expr l = VInt{10};
  Expr r = VInt{20};

  // Addition function closure: l + r
  Expr add_func = EvalIntFn{[l, r](const SymTable& st) -> VInt {
    return EvalIntExpr(st, l) + EvalIntExpr(st, r);
  }};

  // Nested multiplication closure: (l + r) * 2
  Expr nested_func = EvalIntFn{[add_func](const SymTable& st) -> VInt {
    return EvalIntExpr(st, add_func) * 2;
  }};

  EXPECT_EQ(EvalIntExpr(symtab, nested_func), 60);
  EXPECT_DOUBLE_EQ(EvalFloatExpr(symtab, nested_func), 60.0);
  EXPECT_TRUE(EvalCondExpr(symtab, nested_func));

  // Nested multiplication closure: (l + r) * symtab->var_int
  Expr nested_func2 = EvalIntFn{[add_func](const SymTable& st) -> VInt {
    return EvalIntExpr(st, add_func) * EvalIntExpr(st, SymbolKey{"var_int"});
  }};

  EXPECT_EQ(EvalIntExpr(symtab, nested_func2), 300);
}

TEST_F(ExprTest, EvalSymbolLookups) {
  Expr e_sym = SymbolKey{"const_int"};
  EXPECT_EQ(EvalIntExpr(symtab, e_sym), 42);
  EXPECT_DOUBLE_EQ(EvalFloatExpr(symtab, e_sym), 42.0);

  // Indexing vector: const_int_vec[1]
  Expr e_vec_ind = SymbolKeyInd{"const_int_vec", VInt{1}};
  EXPECT_EQ(EvalIntExpr(symtab, e_vec_ind), 200);

  // Indexing measurement bits: m_out[3] -> 1
  Expr e_mea_ind = SymbolKeyInd{"m_out", VInt{3}};
  EXPECT_EQ(EvalIntExpr(symtab, e_mea_ind), 1);
}

TEST_F(ExprTest, ConvertibleToIntChecks) {
  Expr e_int = VInt{1};
  Expr e_float = VFloat{1.5};
  Expr e_func_f = EvalFloatFn{[](const SymTable&) { return 1.5; }};
  Expr e_sym_int = SymbolKey{"const_int"};
  Expr e_sym_float = SymbolKey{"const_float"};
  Expr e_sym_int_indexed = SymbolKeyInd{SymbolKey{"const_int_vec"}, Index{1}};
  Expr e_sym_float_indexed = SymbolKeyInd{SymbolKey{"float_vec"}, Index{1}};

  EXPECT_TRUE(IsConvertibleToInt(symtab, e_int));
  EXPECT_FALSE(IsConvertibleToInt(symtab, e_float));
  EXPECT_FALSE(IsConvertibleToInt(symtab, e_func_f));
  EXPECT_TRUE(IsConvertibleToInt(symtab, e_sym_int));
  EXPECT_FALSE(IsConvertibleToInt(symtab, e_sym_float));
  EXPECT_TRUE(IsConvertibleToInt(symtab, e_sym_int_indexed));
  EXPECT_FALSE(IsConvertibleToInt(symtab, e_sym_float_indexed));
}

TEST_F(ExprTest, ConstExprChecks) {
  Expr e_literal = VFloat{3.14};
  Expr e_func = EvalIntFn{[](const SymTable&) { return 1; }};
  Expr e_const_sym = SymbolKey{"const_int"};
  Expr e_var_sym = SymbolKey{"var_int"};
  Expr e_const_vec_const_index =
      SymbolKeyInd{SymbolKey{"const_int_vec"}, Index{1}};
  Expr e_var_vec_const_index = SymbolKeyInd{SymbolKey{"var_int_vec"}, Index{1}};
  Expr e_const_vec_var_index = SymbolKeyInd{SymbolKey{"const_int_vec"},
                                          EvalIntFn{
                                            [](const SymTable&) { return 1; }
                                          }};
  Expr e_var_vec_var_index = SymbolKeyInd{SymbolKey{"var_int_vec"},
                                        EvalIntFn{
                                          [](const SymTable&) { return 1; }
                                        }};

  EXPECT_TRUE(IsConstExpr(symtab, e_literal));
  EXPECT_FALSE(IsConstExpr(symtab, e_func));
  EXPECT_TRUE(IsConstExpr(symtab, e_const_sym));
  EXPECT_FALSE(IsConstExpr(symtab, e_var_sym));
  EXPECT_TRUE(IsConstExpr(symtab, e_const_vec_const_index));
  EXPECT_FALSE(IsConstExpr(symtab, e_const_vec_var_index));
  EXPECT_FALSE(IsConstExpr(symtab, e_var_vec_const_index));
  EXPECT_FALSE(IsConstExpr(symtab, e_var_vec_var_index));
}

TEST_F(ExprTest, FloatToIntConversionThrows) {
  Expr e_float = VFloat{3.14};
  Expr e_float_indexed = SymbolKeyInd{"float_vec", VInt{0}};
  EXPECT_THROW(EvalIntExpr(symtab, e_float), std::runtime_error);
  EXPECT_THROW(EvalCondExpr(symtab, e_float), std::runtime_error);
  EXPECT_THROW(EvalIntExpr(symtab, e_float_indexed), std::runtime_error);
  EXPECT_THROW(EvalCondExpr(symtab, e_float_indexed), std::runtime_error);
}

TEST_F(ExprTest, MissingSymbolLookupThrows) {
  Expr e_missing = SymbolKey{"nonexistent_var"};
  EXPECT_THROW(EvalFloatExpr(symtab, e_missing), std::runtime_error);
  EXPECT_THROW(EvalIntExpr(symtab, e_missing), std::runtime_error);
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
