#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

#include "../lib/classical_control_expr.h"
#include "../lib/classical_control_parser.h"
#include "../lib/classical_control_symbol.h"
#include "../lib/classical_control_symtab.h"
#include "../lib/classical_control_tokenizer.h"

namespace qsim::cc {
namespace {

struct TestParserError {
  template <typename... Args>
  [[noreturn]] static void Throw(
      std::string_view msg, unsigned lc, Args&&... args) {
    throw std::runtime_error("syntax error");
  }
};

struct TestRuntimeError {
  template <typename... Args>
  [[noreturn]] static void Throw(
      std::string_view msg, unsigned lc, Args&&... args) {
    throw std::runtime_error("runtime error");
  }
};

using TestIndexParser = IndexParser<TestParserError, TestRuntimeError>;

using TestExprParser = ExprParser<TestIndexParser,
                                  TestParserError, TestRuntimeError, false>;

using TestIntExprParser = ExprParser<TestIndexParser,
                                     TestParserError, TestRuntimeError, true>;

class ParserTest : public ::testing::Test {
 protected:
  SymTable symtab;

  void SetUp() override {
    auto scope = symtab.AddScope();
    symtab.EnterScope(scope);

    // Constant symbols (must fold to literals at compile-time)
    symtab.Insert("c_int", Symbol(TInt{10}, /*read_only=*/true));
    symtab.Insert("c_float", Symbol(TFloat{2.5}, /*read_only=*/true));

    // Variable symbols (must generate runtime closures)
    symtab.Insert("v_int", Symbol(TInt{5}, /*read_only=*/false));
    symtab.Insert("v_float", Symbol(TFloat{1.5}, /*read_only=*/false));

    // Vector and measurement symbols
    symtab.Insert(
        "vec", Symbol(Symbol::IntVector{10, 20, 30}, /*read_only=*/true));

    // Mea bits: b0=1, b1=0, b2=1 (val = 5)
    Symbol::Mea mea{.bits = 0b101, .num_bits = 3, .time = 0};
    symtab.Insert("m0", Symbol(mea));
  }
};

TEST_F(ParserTest, ConstantExpressionsFoldToScalarLiterals) {
  Tokenizer tok("2 * c_int + 4");
  Expr expr = TestExprParser::Run(symtab, tok);

  // Must hold TInt directly (folded at compile-time)
  ASSERT_TRUE(std::holds_alternative<TInt>(expr));
  EXPECT_EQ(std::get<TInt>(expr), 24);
}

TEST_F(ParserTest, ConstantFloatExpressionsFoldToFloatLiterals) {
  Tokenizer tok("c_float * 2.0");
  Expr expr = TestExprParser::Run(symtab, tok);

  // Must hold TFloat directly
  ASSERT_TRUE(std::holds_alternative<TFloat>(expr));
  EXPECT_DOUBLE_EQ(std::get<TFloat>(expr), 5.0);
}

TEST_F(ParserTest, ConstantComparisonFoldsToBoolLiteral) {
  Tokenizer tok("c_int == 10");
  Expr expr = TestExprParser::Run(symtab, tok);

  // Must hold TBool directly
  ASSERT_TRUE(std::holds_alternative<TBool>(expr));
  EXPECT_TRUE(std::get<TBool>(expr));
}

TEST_F(ParserTest, VariableExpressionProducesRuntimeClosure) {
  Tokenizer tok("v_int * 2 + 1");
  Expr expr = TestExprParser::Run(symtab, tok);

  // Must NOT be a scalar literal; must produce an executable TFuncI closure
  EXPECT_FALSE(std::holds_alternative<TInt>(expr));
  EXPECT_TRUE(std::holds_alternative<TFuncI>(expr));

  // Evaluate closure at runtime
  EXPECT_EQ(EvalIntExpr(symtab, expr), 11);
}

TEST_F(ParserTest, ParserStopsWhenOperatorsExhausted) {
  Tokenizer tok("2 * c_int 3 * v_int");

  // First run consumes "2 * c_int"
  Expr e1 = TestExprParser::Run(symtab, tok);
  ASSERT_TRUE(std::holds_alternative<TInt>(e1));
  EXPECT_EQ(std::get<TInt>(e1), 20);

  // Next token remaining in stream must be "3"
  EXPECT_EQ(tok.Peek().val, "3");

  // Second run consumes "3 * v_int"
  Expr e2 = TestExprParser::Run(symtab, tok);
  EXPECT_EQ(EvalIntExpr(symtab, e2), 15);
}

TEST_F(ParserTest, ParserStopsAtParenthesis) {
  Tokenizer tok("2 * c_int (-3 * v_int)");

  // Should parse as single expression: "2 * c_int" = 20
  Expr expr = TestExprParser::Run(symtab, tok);
  EXPECT_EQ(EvalIntExpr(symtab, expr), 20);
}

TEST_F(ParserTest, IndexParserEvaluatesBracketedIndex) {
  Tokenizer tok("[1 + 1]");
  Index idx = TestIndexParser::Run(symtab, tok);

  // Constant index folds to TInt
  ASSERT_TRUE(std::holds_alternative<TInt>(idx));
  EXPECT_EQ(EvalIndex(symtab, idx), 2u);
}

TEST_F(ParserTest, VectorAndMeasurementIndexing) {
  Tokenizer tok("vec[1] + m0[0]"); // vec[1] = 20, m0[0] = 1 -> 21
  Expr expr = TestExprParser::Run(symtab, tok);

  EXPECT_EQ(EvalIntExpr(symtab, expr), 21);
}

TEST_F(ParserTest, IntOnlyParserRejectsFloatsAndPowerOperator) {
  Tokenizer tok_float("2.5 + 1");
  EXPECT_THROW(TestIntExprParser::Run(symtab, tok_float), std::runtime_error);

  Tokenizer tok_pow("2 ** 3");
  EXPECT_THROW(TestIntExprParser::Run(symtab, tok_pow), std::runtime_error);
}

TEST_F(ParserTest, TriggersMeasurementCallback) {
  Tokenizer tok("m0 + 1");
  bool callback_called = false;

  auto callback = [&callback_called](
      const Symbol::Mea& mea, std::string_view name) {
    callback_called = true;
    EXPECT_EQ(name, "m0");
    EXPECT_EQ(mea.num_bits, 3u);
  };

  Expr expr = TestExprParser::Run(symtab, tok, callback);
  EXPECT_TRUE(callback_called);
  EXPECT_EQ(EvalIntExpr(symtab, expr), 6); // m0 bitstring = 5; 5 + 1 = 6
}

class ComplexParserTest : public ::testing::Test {
 protected:
  SymTable symtab;

  void SetUp() override {
    auto scope = symtab.AddScope();
    symtab.EnterScope(scope);

    // Integer symbols and vector
    symtab.Insert("a", Symbol(TInt{3}, /*read_only=*/true));
    symtab.Insert("b", Symbol(TInt{7}, /*read_only=*/true));
    symtab.Insert("c", Symbol(TInt{2}, /*read_only=*/true));
    symtab.Insert(
        "ivec", Symbol(Symbol::IntVector{5, 12, 18, 42}, /*read_only=*/true));

    // Float symbols and vector
    symtab.Insert("x", Symbol(TFloat{2.5}, /*read_only=*/true));
    symtab.Insert("y", Symbol(TFloat{0.5}, /*read_only=*/true));
    symtab.Insert("z", Symbol(TFloat{4.0}, /*read_only=*/true));
    symtab.Insert("fvec", Symbol(
        Symbol::FloatVector{1.5, 3.25, 8.0}, /*read_only=*/true));

    // Mea bits: 0b1011 -> m[0]=1, m[1]=1, m[2]=0, m[3]=1
    Symbol::Mea mea{.bits = 0b1011, .num_bits = 4, .time = 0};
    symtab.Insert("m", Symbol(mea));
  }
};

TEST_F(ComplexParserTest, RandomIntExpressions) {
  // Case 1: Precedence of bitwise, shift, and arithmetic operators
  {
    Tokenizer tok("(a << 2) + (b & 0x05) * ~c + (ivec[1] % 5)");
    TInt expected = (3 << 2) + (7 & 0x05) * ~2 + (12 % 5);
    Expr expr = TestIntExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalIntExpr(symtab, expr), expected);
  }

  // Case 2: Deep parenthetical nesting with comparisons and logical conversions
  {
    Tokenizer tok("((ivec[a - 1] + 2) * (b > a)) - (ivec[0] << (c + 1))");
    // (20 * 1) - 40 = -20
    TInt expected = ((18 + 2) * (7 > 3)) - (5 << (2 + 1));
    Expr expr = TestIntExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalIntExpr(symtab, expr), expected);
  }

  // Case 3: Measurement bit indexing combined with bitwise XOR and binary
  // literals
  {
    Tokenizer tok("(m[0] ^ m[2]) + (m[3] << 3) * (ivec[m[1]] - 0b0010)");
    // m[0]=1, m[2]=0, m[3]=1, m[1]=1 -> ivec[1]=12
    TInt expected = (1 ^ 0) + (1 << 3) * (12 - 2); // 1 + 8 * 10 = 81
    Expr expr = TestIntExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalIntExpr(symtab, expr), expected);
  }

  // Case 4: Multi-operator arithmetic with hexadecimal constants
  {
    Tokenizer tok("0x1F & (ivec[3] / (a + 1)) + (0x0A * (b - c))");
    // 31 & (10 + 50) = 31 & 60 = 28
    TInt expected = 0x1F & ((42 / (3 + 1)) + (0x0A * (7 - 2)));
    Expr expr = TestIntExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalIntExpr(symtab, expr), expected);
  }

  // Case 5: Complex chain with relational booleans coerced to integers
  {
    Tokenizer tok("(a + b * c > 10) + 2 * (ivec[2] <= 20) + 3 * (b != 7)");
    // (3 + 14 > 10) -> 1
    // (18 <= 20) -> 1
    // (7 != 7) -> 0
    TInt expected = 1 + 2 * 1 + 3 * 0; // 3
    Expr expr = TestIntExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalIntExpr(symtab, expr), expected);
  }
}

TEST_F(ComplexParserTest, RandomFloatExpressions) {
  // Case 1: Mixed float/int arithmetic and power (**) operator
  {
    Tokenizer tok("x ** 2.0 + fvec[0] * (z - y) / 0.5");
     // 6.25 + 10.5 = 16.75
    TFloat expected = std::pow(2.5, 2.0) + 1.5 * (4.0 - 0.5) / 0.5;
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_DOUBLE_EQ(EvalExpr(symtab, expr), expected);
  }

  // Case 2: Deeply nested float expressions with vector indexing via dynamic
  // int expressions
  {
    Tokenizer tok("(fvec[a - 2] * (z ** (y * 2.0))) - (x / (fvec[1] - 0.25))");
    // fvec[1] * (4.0 ** 1.0) - (2.5 / (3.25 - 0.25)) -> 3.25 * 4 - (2.5 / 3.0)
    TFloat expected = (3.25 * 4.0) - (2.5 / 3.0);
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_DOUBLE_EQ(EvalExpr(symtab, expr), expected);
  }

  // Case 3: Float expression with promoted integer symbols and float division
  {
    Tokenizer tok("(a + 1.5) * (ivec[0] ** 2) / (z + y)");
    // (4.5 * 25.0) / 4.5 = 25.0
    TFloat expected = (3.0 + 1.5) * std::pow(5.0, 2.0) / (4.0 + 0.5);
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_DOUBLE_EQ(EvalExpr(symtab, expr), expected);
  }

  // Case 4: Precedence test with negation, addition, and power
  {
    Tokenizer tok("-x + (y + fvec[2]) ** 0.5 * z");
    // -2.5 + 3.0 * 4.0 = 9.5
    TFloat expected = -2.5 + std::pow(0.5 + 8.0, 0.5) * 4.0;
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_DOUBLE_EQ(EvalExpr(symtab, expr), expected);
  }
}

TEST_F(ComplexParserTest, RandomBoolExpressions) {
  // Case 1: Logical AND, OR, NOT, and XOR (^^) chains over comparisons
  {
    Tokenizer tok("(a < b) && (ivec[1] == 12) ^^ !(x >= z) || (c == 0)");
    // (3 < 7)[1] && (12 == 12)[1] ^^ !(2.5 >= 4.0)[1] || (2 == 0)[0]
    // (1 && 1) ^^ 1 || 0 -> 1 ^^ 1 || 0 -> 0 || 0 -> false
    TBool expected = false;
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalCondExpr(symtab, expr), expected);
  }

  // Case 2: Mixed measurement bit conditions and boolean logical operators
  {
    Tokenizer tok("!(m[2] == 1) && ((m[0] ^^ m[1]) || (ivec[a] > 30))");
    // !(0 == 1)[1] && ((1 ^^ 1)[0] || (ivec[3]=42 > 30)[1])
    // 1 && (0 || 1) -> true
    TBool expected = true;
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalCondExpr(symtab, expr), expected);
  }

  // Case 3: Compound conditional mixing floating-point and integer
  // subexpressions
  {
    Tokenizer tok("(x * 2.0 == ivec[0]) && (fvec[1] > y) ^^ (b - a != 4)");
    // (5.0 == 5)[1] && (3.25 > 0.5)[1] ^^ (7 - 3 != 4)[0]
    // (1 && 1) ^^ 0 -> 1 ^^ 0 -> true
    TBool expected = true;
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalCondExpr(symtab, expr), expected);
  }

  // Case 4: Deep logical expression evaluating integer zero/non-zero values
  // implicitly as booleans
  {
    Tokenizer tok("!((a & b) ^^ (ivec[0] % 2)) || ((z > 0.0) && (m[3]))");
    // (3 & 7) = 3 (true); (5 % 2) = 1 (true) -> (3 ^^ 1) => 1 ^^ 1 = 0 (false)
    // !0 -> 1 (true)
    // true || ... -> true
    TBool expected = true;
    Expr expr = TestExprParser::Run(symtab, tok);
    EXPECT_EQ(EvalCondExpr(symtab, expr), expected);
  }
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
