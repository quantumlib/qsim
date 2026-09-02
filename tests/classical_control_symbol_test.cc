#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "../lib/classical_control_expr.h"
#include "../lib/classical_control_symbol.h"
#include "../lib/classical_control_symtab.h"

namespace qsim::cc {
namespace {

TEST(SymbolTest, DefaultConstructor) {
  Symbol s;
  EXPECT_TRUE(s.IsReadOnly());
}

TEST(SymbolTest, ConstructsIntAndDefaultReadOnly) {
  Symbol s(Symbol::Int{42});
  EXPECT_FALSE(s.HoldsMea());
  EXPECT_FALSE(s.HoldsVector());
  EXPECT_FALSE(s.IsFloat());
  EXPECT_TRUE(s.IsConvertibleToInt());
  EXPECT_TRUE(s.IsReadOnly());
  EXPECT_EQ(s.Size(), 1u);
  EXPECT_EQ(s.GetInt(), 42);
}

TEST(SymbolTest, ConstructsFloatAndExplicitReadOnly) {
  Symbol s(Symbol::Float{3.14}, /*read_only=*/false);
  EXPECT_FALSE(s.HoldsMea());
  EXPECT_FALSE(s.HoldsVector());
  EXPECT_TRUE(s.IsFloat());
  EXPECT_FALSE(s.IsConvertibleToInt());
  EXPECT_FALSE(s.IsReadOnly());
  EXPECT_EQ(s.Size(), 1u);
  EXPECT_DOUBLE_EQ(s.GetFloat(), 3.14);
}

TEST(SymbolTest, ConstructsMeaDefaultsToNotReadOnly) {
  Symbol::Mea mea{.bits = 0b1011, .num_bits = 4, .time = 10};
  Symbol s(mea);

  EXPECT_TRUE(s.HoldsMea());
  EXPECT_FALSE(s.HoldsVector());
  EXPECT_FALSE(s.IsReadOnly());
  EXPECT_EQ(s.Size(), 4u);
  EXPECT_EQ(s.GetMea().time, 10u);
}

TEST(SymbolTest, ConstructsVectorsCorrectly) {
  Symbol::IntVector iv = {1, 2, 3, 4};
  Symbol s_int_vec(iv);

  EXPECT_TRUE(s_int_vec.HoldsVector());
  EXPECT_FALSE(s_int_vec.IsFloat());
  EXPECT_TRUE(s_int_vec.IsConvertibleToInt());
  EXPECT_EQ(s_int_vec.Size(), 4u);

  Symbol::FloatVector fv = {1.1, 2.2};
  Symbol s_float_vec(fv);

  EXPECT_TRUE(s_float_vec.HoldsVector());
  EXPECT_TRUE(s_float_vec.IsFloat());
  EXPECT_FALSE(s_float_vec.IsConvertibleToInt());
  EXPECT_EQ(s_float_vec.Size(), 2u);
}

TEST(SymbolTest, GetIntAndGetFloatFromMea) {
  Symbol::Mea mea{.bits = 0b1101, .num_bits = 4, .time = 0};
  Symbol s(mea);

  EXPECT_EQ(s.GetInt(), 0b1101);
  EXPECT_DOUBLE_EQ(s.GetFloat(), 13.0);
}

TEST(SymbolTest, InvalidScalarConversionsThrow) {
  Symbol s_float(Symbol::Float{5.5});
  EXPECT_THROW(s_float.GetInt(), std::runtime_error);

  Symbol s_vec(Symbol::IntVector{1, 2});
  EXPECT_THROW(s_vec.GetInt(), std::runtime_error);
  EXPECT_THROW(s_vec.GetFloat(), std::runtime_error);
}

TEST(SymbolTest, GetIntIndexedFromIntVector) {
  Symbol s(Symbol::IntVector{10, 20, 30});

  EXPECT_EQ(s.GetInt(0), 10);
  EXPECT_EQ(s.GetInt(1), 20);
  EXPECT_EQ(s.GetInt(2), 30);
  EXPECT_THROW(s.GetInt(3), std::runtime_error);
}

TEST(SymbolTest, GetFloatIndexedFromVectors) {
  Symbol s_int(Symbol::IntVector{10, 20});
  EXPECT_DOUBLE_EQ(s_int.GetFloat(1), 20.0);

  Symbol s_float(Symbol::FloatVector{1.5, 2.5});
  EXPECT_DOUBLE_EQ(s_float.GetFloat(0), 1.5);
  EXPECT_THROW(s_float.GetFloat(2), std::runtime_error);
}

TEST(SymbolTest, GetIntFromFloatVectorThrows) {
  Symbol s(Symbol::FloatVector{1.0, 2.0});
  EXPECT_THROW(s.GetInt(0), std::runtime_error);
}

TEST(SymbolTest, IndexingScalarsThrows) {
  Symbol s(Symbol::Int{100});
  EXPECT_THROW(s.GetInt(0), std::runtime_error);
  EXPECT_THROW(s.GetFloat(0), std::runtime_error);
}

TEST(SymbolTest, GetMeaIntBits) {
  // 0b1010 -> Bit 0: 0, Bit 1: 1, Bit 2: 0, Bit 3: 1
  Symbol::Mea mea{.bits = 0b1010, .num_bits = 4, .time = 1};
  Symbol s(mea);

  EXPECT_EQ(s.GetInt(0), 0);
  EXPECT_EQ(s.GetInt(1), 1);
  EXPECT_EQ(s.GetInt(2), 0);
  EXPECT_EQ(s.GetInt(3), 1);

  EXPECT_THROW(s.GetInt(4), std::runtime_error);
}

TEST(SymbolTest, AssignScalarIntAndFloat) {
  Symbol s_int(Symbol::Int{10}, /*read_only=*/false);
  s_int.Assign(Symbol::Int{99});
  EXPECT_EQ(s_int.GetInt(), 99);

  Symbol s_float(Symbol::Float{1.0}, /*read_only=*/false);
  s_float.Assign(Symbol::Float{4.5});
  EXPECT_DOUBLE_EQ(s_float.GetFloat(), 4.5);
}

TEST(SymbolTest, AssignVectorElements) {
  Symbol s_int_vec(Symbol::IntVector{1, 2, 3}, /*read_only=*/false);
  s_int_vec.Assign(Symbol::Int{100}, 1);
  EXPECT_EQ(s_int_vec.GetInt(1), 100);

  Symbol s_float_vec(Symbol::FloatVector{0.1, 0.2}, /*read_only=*/false);
  s_float_vec.Assign(Symbol::Float{9.9}, 0);
  EXPECT_DOUBLE_EQ(s_float_vec.GetFloat(0), 9.9);
}

class SymbolAssignTest : public ::testing::Test {
 protected:
  SymTable symtab;

  void SetUp() override {
    auto scope = symtab.AddScope();
    symtab.EnterScope(scope);

    symtab.Insert("rhs_var", Symbol(Symbol::Int{10}));
  }
};

TEST_F(SymbolAssignTest, AssignsScalarInt) {
  Symbol s(Symbol::Int{0}, /*read_only=*/false);

  // Assign with expression referencing a symbol table variable: rhs_var + 5
  Expr expr = TFuncI{[](const SymTable& st) -> TInt {
    return EvalIntExpr(st, TSymbol{"rhs_var"}) + 5;
  }};

  s.Assign(symtab, std::vector<Expr>{expr});
  EXPECT_EQ(s.GetInt(), 15);
}

TEST_F(SymbolAssignTest, AssignsScalarFloatAndConvertsIntToFloat) {
  Symbol s(Symbol::Float{0.0}, /*read_only=*/false);

  s.Assign(symtab, std::vector<Expr>{TInt{42}});
  EXPECT_DOUBLE_EQ(s.GetFloat(), 42.0);
}

TEST_F(SymbolAssignTest, AssigningFloatToScalarIntThrows) {
  Symbol s(Symbol::Int{0}, /*read_only=*/false);

  EXPECT_THROW(
      s.Assign(symtab, std::vector<Expr>{TFloat{3.14}}), std::runtime_error);
}

TEST_F(SymbolAssignTest, AssignsIntVectorElementWise) {
  Symbol s(Symbol::IntVector{0, 0, 0}, /*read_only=*/false);
  std::vector<Expr> exprs = {TInt{10}, TInt{20}, TInt{30}};

  s.Assign(symtab, exprs);
  EXPECT_EQ(s.GetInt(0), 10);
  EXPECT_EQ(s.GetInt(1), 20);
  EXPECT_EQ(s.GetInt(2), 30);
}

TEST_F(SymbolAssignTest, AssignsFloatVectorElementWise) {
  Symbol s(Symbol::FloatVector{0.0, 0.0}, /*read_only=*/false);
  std::vector<Expr> exprs = {TFloat{1.1}, TInt{2}};

  s.Assign(symtab, exprs);
  EXPECT_DOUBLE_EQ(s.GetFloat(0), 1.1);
  EXPECT_DOUBLE_EQ(s.GetFloat(1), 2.0);
}

TEST_F(SymbolAssignTest, AssignsVectorUpToContainerSize) {
  Symbol s(Symbol::IntVector{0, 0}, /*read_only=*/false);
  std::vector<Expr> exprs = {TInt{10}, TInt{20}, TInt{30}, TInt{40}};

  s.Assign(symtab, exprs);
  EXPECT_EQ(s.Size(), 2u);
  EXPECT_EQ(s.GetInt(0), 10);
  EXPECT_EQ(s.GetInt(1), 20);
}

TEST_F(SymbolAssignTest, AssigningFloatInIntVectorThrows) {
  Symbol s(Symbol::IntVector{0, 0}, /*read_only=*/false);
  std::vector<Expr> exprs = {TInt{10}, TFloat{2.5}};

  EXPECT_THROW(s.Assign(symtab, exprs), std::runtime_error);
}

TEST_F(SymbolAssignTest, AssignsIndexedVectorElement) {
  Symbol s(Symbol::IntVector{100, 200, 300}, /*read_only=*/false);
  Index idx = TInt{1};
  Expr expr = TInt{999};

  s.Assign(symtab, expr, idx);
  EXPECT_EQ(s.GetInt(1), 999);
}

TEST_F(SymbolAssignTest, IndexedAssignmentOutOfBoundsThrows) {
  Symbol s(Symbol::FloatVector{1.0, 2.0}, /*read_only=*/false);
  Index idx = TInt{5};
  Expr expr = TFloat{9.9};

  EXPECT_THROW(s.Assign(symtab, expr, idx), std::runtime_error);
}

TEST_F(SymbolAssignTest, IndexedAssignmentTypeMismatchThrows) {
  Symbol s(Symbol::IntVector{1, 2, 3}, /*read_only=*/false);
  Index idx = TInt{0};
  Expr expr = TFloat{1.23};

  EXPECT_THROW(s.Assign(symtab, expr, idx), std::runtime_error);
}

TEST_F(SymbolAssignTest, AssigningToMeasurementThrows) {
  Symbol::Mea mea{.bits = 0, .num_bits = 2, .time = 0};
  Symbol s(mea);

  EXPECT_THROW(
      s.Assign(symtab, std::vector<Expr>{TInt{1}}), std::runtime_error);
  EXPECT_THROW(s.Assign(symtab, TInt{1}, TInt{0}), std::runtime_error);
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
