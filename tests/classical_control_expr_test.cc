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
    symtab.Insert("const_int", Symbol(TInt{42}, /*read_only=*/true));
    symtab.Insert("const_float", Symbol(TFloat{3.14}, /*read_only=*/true));
    symtab.Insert("var_int", Symbol(TInt{10}, /*read_only=*/false));

    // Vector symbol
    Symbol::IntVector vec = {100, 200, 300};
    symtab.Insert("int_vec", Symbol(vec, /*read_only=*/true));

    // Quantum Measurement symbol (0b1011 = bit0:1, bit1:1, bit2:0, bit3:1)
    Symbol::Mea mea{.bits = 0b1011, .num_bits = 4, .time = 0};
    symtab.Insert("m_out", Symbol(mea)); // Mea is always read_only = false
  }
};

TEST_F(ExprTest, IndexEvaluation) {
  Index i_int = TInt{2};
  Index i_bool = TBool{true};
  Index i_func = TFuncI{[](const SymTable&) { return TInt{1}; }};

  EXPECT_TRUE(IsConstIndex(symtab, i_int));
  EXPECT_TRUE(IsConstIndex(symtab, i_bool));
  EXPECT_FALSE(IsConstIndex(symtab, i_func));

  EXPECT_EQ(EvalIndex(symtab, i_int), 2u);
  EXPECT_EQ(EvalIndex(symtab, i_bool), 1u);
  EXPECT_EQ(EvalIndex(symtab, i_func), 1u);
}

TEST_F(ExprTest, EvalLiterals) {
  Expr e_int = TInt{5};
  Expr e_float = TFloat{2.5};
  Expr e_bool = TBool{true};

  EXPECT_DOUBLE_EQ(EvalExpr(symtab, e_int), 5.0);
  EXPECT_DOUBLE_EQ(EvalExpr(symtab, e_float), 2.5);
  EXPECT_DOUBLE_EQ(EvalExpr(symtab, e_bool), 1.0);

  EXPECT_EQ(EvalIntExpr(symtab, e_int), 5);
  EXPECT_EQ(EvalIntExpr(symtab, e_bool), 1);

  EXPECT_TRUE(EvalCondExpr(symtab, e_int));
  EXPECT_TRUE(EvalCondExpr(symtab, e_bool));
}

TEST_F(ExprTest, EvalNestedFunctions) {
  Expr l = TInt{10};
  Expr r = TInt{20};

  // Addition function closure: l + r
  Expr add_func = TFuncI{[l, r](const SymTable& st) -> TInt {
    return EvalIntExpr(st, l) + EvalIntExpr(st, r);
  }};

  // Nested multiplication closure: (l + r) * 2
  Expr nested_func = TFuncI{[add_func](const SymTable& st) -> TInt {
    return EvalIntExpr(st, add_func) * 2;
  }};

  EXPECT_EQ(EvalIntExpr(symtab, nested_func), 60);
  EXPECT_DOUBLE_EQ(EvalExpr(symtab, nested_func), 60.0);
  EXPECT_TRUE(EvalCondExpr(symtab, nested_func));
}

TEST_F(ExprTest, EvalSymbolLookups) {
  Expr e_sym = TSymbol{"const_int"};
  EXPECT_EQ(EvalIntExpr(symtab, e_sym), 42);
  EXPECT_DOUBLE_EQ(EvalExpr(symtab, e_sym), 42.0);

  // Indexing vector: int_vec[1]
  Expr e_vec_ind = TSymbolInd{"int_vec", TInt{1}};
  EXPECT_EQ(EvalIntExpr(symtab, e_vec_ind), 200);

  // Indexing measurement bits: m_out[3] -> 1
  Expr e_mea_ind = TSymbolInd{"m_out", TInt{3}};
  EXPECT_EQ(EvalIntExpr(symtab, e_mea_ind), 1);
}

TEST_F(ExprTest, ConvertibleToIntChecks) {
  Expr e_int = TInt{1};
  Expr e_float = TFloat{1.5};
  Expr e_func_f = TFuncF{[](const SymTable&) { return 1.5; }};
  Expr e_sym_int = TSymbol{"const_int"};
  Expr e_sym_float = TSymbol{"const_float"};

  EXPECT_TRUE(IsConvertibleToInt(symtab, e_int));
  EXPECT_FALSE(IsConvertibleToInt(symtab, e_float));
  EXPECT_FALSE(IsConvertibleToInt(symtab, e_func_f));
  EXPECT_TRUE(IsConvertibleToInt(symtab, e_sym_int));
  EXPECT_FALSE(IsConvertibleToInt(symtab, e_sym_float));
}

TEST_F(ExprTest, ConstExprChecks) {
  Expr e_literal = TFloat{3.14};
  Expr e_func = TFuncI{[](const SymTable&) { return 1; }};
  Expr e_const_sym = TSymbol{"const_int"};
  Expr e_var_sym = TSymbol{"var_int"};

  EXPECT_TRUE(IsConstExpr(symtab, e_literal));
  EXPECT_FALSE(IsConstExpr(symtab, e_func));
  EXPECT_TRUE(IsConstExpr(symtab, e_const_sym));
  EXPECT_FALSE(IsConstExpr(symtab, e_var_sym));
}

TEST_F(ExprTest, FloatToIntConversionThrows) {
  Expr e_float = TFloat{3.14};
  EXPECT_THROW(EvalIntExpr(symtab, e_float), std::runtime_error);
  EXPECT_THROW(EvalCondExpr(symtab, e_float), std::runtime_error);
}

TEST_F(ExprTest, MissingSymbolLookupThrows) {
  Expr e_missing = TSymbol{"nonexistent_var"};
  EXPECT_THROW(EvalExpr(symtab, e_missing), std::runtime_error);
  EXPECT_THROW(EvalIntExpr(symtab, e_missing), std::runtime_error);
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
