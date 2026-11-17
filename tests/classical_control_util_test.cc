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

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../lib/classical_control_expr.h"
#include "../lib/classical_control_symbol.h"
#include "../lib/classical_control_symtab.h"
#include "../lib/classical_control_util.h"

namespace qsim::cc {
namespace {

TEST(UtilsTest, ToIntDecimal) {
  EXPECT_EQ(ToInt("0"), 0u);
  EXPECT_EQ(ToInt("42"), 42u);
  EXPECT_EQ(ToInt("1000"), 1000u);
}

TEST(UtilsTest, ToIntHexadecimal) {
  EXPECT_EQ(ToInt("0x1a"), 26u);
  EXPECT_EQ(ToInt("0X1A"), 26u);
  EXPECT_EQ(ToInt("0xFF"), 255u);
}

TEST(UtilsTest, ToIntBinary) {
  EXPECT_EQ(ToInt("0b1010"), 10u);
  EXPECT_EQ(ToInt("0B1111"), 15u);
  EXPECT_EQ(ToInt("0b0"), 0u);
}

TEST(UtilsTest, ToIntInvalidThrows) {
  EXPECT_THROW(ToInt("abc"), std::runtime_error);      // Non-numeric
  EXPECT_THROW(ToInt("12abc"), std::runtime_error);    // Trailing non-numeric
  EXPECT_THROW(ToInt("0xG1"), std::runtime_error);     // Invalid hex
  EXPECT_THROW(ToInt("0b1020"), std::runtime_error);   // Invalid binary
}

TEST(UtilsTest, ToFloatStandardAndExponent) {
  EXPECT_DOUBLE_EQ(ToFloat("0.0"), 0.0);
  EXPECT_DOUBLE_EQ(ToFloat("3.14159"), 3.14159);
  EXPECT_DOUBLE_EQ(ToFloat("1.2e-3"), 0.0012);
  EXPECT_DOUBLE_EQ(ToFloat("2.5E2"), 250.0);
}

TEST(UtilsTest, ToFloatInvalidThrows) {
  EXPECT_THROW(ToFloat("not_a_float"), std::runtime_error);
  EXPECT_THROW(ToFloat("3.14e-1000"), std::runtime_error);
  EXPECT_THROW(ToFloat("3.14e1000"), std::runtime_error);
}

TEST(UtilsTest, Fnv1aHashAndLiteralOperator) {
  constexpr std::size_t h1 = Hash("my_variable");
  constexpr std::size_t h2 = "my_variable"_hash;

  EXPECT_EQ(h1, h2);
  EXPECT_NE("var1"_hash, "var2"_hash);
}

TEST(UtilsTest, OpHashPacking) {
  // Single char op
  constexpr unsigned short op1 = OpHash("+");
  constexpr unsigned short op1_lit = "+"_ophash;
  EXPECT_EQ(op1, op1_lit);

  // Two char op
  constexpr unsigned short op2 = OpHash("==");
  constexpr unsigned short op2_lit = "=="_ophash;
  EXPECT_EQ(op2, op2_lit);

  // Over length op returns 0
  EXPECT_EQ(OpHash("+++"), 0);

  const std::vector<std::string_view> ops = {
      "||", "^^", "&&", "==", "!=", "<=", ">=", "**", "<<", ">>", "-", "+",
      "*", "/", "=", "<", ">", "!", "~", "|", "^", "&"};

  std::set<unsigned short> set;

  for (const auto& op : ops) {
    unsigned short op_hash = OpHash(op);
    EXPECT_FALSE(set.contains(op_hash));
    set.insert(op_hash);
  }
}

TEST(UtilsTest, ReadFileSuccessAndFailure) {
  // Create a temporary file
  std::filesystem::path temp_path =
      std::filesystem::temp_directory_path() / "qsim_test_file.txt";
  {
    std::ofstream ofs(temp_path);
    ofs << "hello qsim\nsecond line";
  }

  std::string content = ReadFile(temp_path.string());
  EXPECT_EQ(content, "hello qsim\nsecond line");

  // Cleanup
  std::filesystem::remove(temp_path);

  // Missing file read throws
  EXPECT_THROW(ReadFile("nonexistent_file_path_12345.txt"), std::runtime_error);
}

class ExprToStrTest : public ::testing::Test {
 protected:
  SymTable symtab;

  void SetUp() override {
    auto scope = symtab.AddScope();
    symtab.EnterScope(scope);
  }
};

TEST_F(ExprToStrTest, DefaultFormatEmptyExpressionsReturnsEmptyString) {
  const std::vector<Expr> es;
  EXPECT_EQ(ExprToStr::Format(symtab, es), "");
}

TEST_F(ExprToStrTest, DefaultFormatSingleIntNoTrailingSpace) {
  const std::vector<Expr> es = {VInt(42)};
  EXPECT_EQ(ExprToStr::Format(symtab, es), "42");
}

TEST_F(ExprToStrTest, DefaultFormatSingleFloatNoTrailingSpace) {
  const std::vector<Expr> es = {VFloat(2.5)};
  EXPECT_EQ(ExprToStr::Format(symtab, es), "2.5");
}

TEST_F(ExprToStrTest, DefaultFormatMixedMultipleExpressions) {
  const std::vector<Expr> es = {VInt(-42), VFloat(1.5), VInt(0), VFloat(-0.25)};
  EXPECT_EQ(ExprToStr::Format(symtab, es), "-42 1.5 0 -0.25");
}

TEST_F(ExprToStrTest, FormatLargeExpressionCount) {
  std::vector<Expr> es;
  std::string expected;
  std::string fmt;
  for (int i = 0; i < 100; ++i) {
    es.push_back(VInt(i));
    if (i > 0) {
      expected += ' ';
      fmt += ' ';
    }
    expected += std::to_string(i);
    fmt += "{}";
  }
  EXPECT_EQ(ExprToStr::Format(symtab, es), expected);
  EXPECT_EQ(ExprToStr::Format(symtab, fmt, es), expected);
}

TEST_F(ExprToStrTest, ExplicitFormatEmptyStringAndEmptyExpressions) {
  const std::vector<Expr> es;
  EXPECT_EQ(ExprToStr::Format(symtab, "", es), "");
}

TEST_F(ExprToStrTest, ExplicitFormatLiteralOnlyWithZeroExpressions) {
  const std::vector<Expr> es;
  EXPECT_EQ(ExprToStr::Format(symtab, "Hello, World!", es), "Hello, World!");
}

TEST_F(ExprToStrTest, ExplicitFormatDslPrintlnExample) {
  const std::vector<Expr> es = {
      VInt(2 * 21),
      VFloat(1.0 / 3.0),
  };
  EXPECT_EQ(ExprToStr::Format(symtab, "{:6x} {:15.8g}", es),
            "    2a      0.33333333");
}

TEST_F(ExprToStrTest, ExplicitFormatRichIntSpecifiersAndInt64Bounds) {
  const std::vector<Expr> es = {
    VInt(255), VInt(10), VInt(7),
    VInt(std::numeric_limits<std::int64_t>::min()),
    VInt(std::numeric_limits<std::int64_t>::max()),
  };
  EXPECT_EQ(
      ExprToStr::Format(symtab, "{:#06x}|{:+#06b}|{:*^5d}|{}|{}", es),
      "0x00ff|+0b1010|**7**|-9223372036854775808|9223372036854775807");
}

TEST_F(ExprToStrTest, ExplicitFormatRichFloatSpecifiersAndSpecialValues) {
  const double pos_nan =
      std::copysign(std::numeric_limits<double>::quiet_NaN(), 1.0);
  const std::vector<Expr> es = {
    VFloat(3.14159265), VFloat(-0.0),
    VFloat(std::numeric_limits<double>::infinity()),
    VFloat(-std::numeric_limits<double>::infinity()),
    VFloat(pos_nan),
  };
  EXPECT_EQ(
      ExprToStr::Format(symtab, "{:+.3e}|{:.1f}|{:F}|{:f}|{:F}", es),
      "+3.142e+00|-0.0|INF|-inf|NAN");
}

TEST_F(ExprToStrTest, EscapedBracesWithZeroExpressions) {
  const std::vector<Expr> es;
  EXPECT_EQ(ExprToStr::Format(symtab, "{{}}", es), "{}");
  EXPECT_EQ(ExprToStr::Format(symtab, "{{{{}}}}", es), "{{}}");
  EXPECT_EQ(ExprToStr::Format(symtab, "a {{ b }} c", es), "a { b } c");
}

TEST_F(ExprToStrTest, EscapedBracesAdjacentToActivePlaceholders) {
  const std::vector<Expr> es1 = {VInt(42)};
  //EXPECT_EQ(ExprToStr::Format(symtab, "{{{}}", es1), "{42");
  EXPECT_EQ(ExprToStr::Format(symtab, "{}}}", es1), "42}");
  EXPECT_EQ(ExprToStr::Format(symtab, "{{{}}}", es1), "{42}");

  const std::vector<Expr> es2 = {VInt(1), VInt(2)};
  EXPECT_EQ(ExprToStr::Format(symtab, "{{start}} {} {{mid}} {} {{end}}", es2),
            "{start} 1 {mid} 2 {end}");
}

TEST_F(ExprToStrTest, ThrowsWhenEmptyFormatStringHasNonEmptyExpressions) {
  // Note: 3-arg Format("", es) requires explicit placeholders; 2-arg Format(es)
  // or PrintExpressions("", es) handles automatic formatting.
  const std::vector<Expr> es = {VInt(42)};
  EXPECT_THROW(ExprToStr::Format(symtab, "", es), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsWhenEscapedBracesHaveNoActivePlaceholder) {
  // "{{}}" has 0 active placeholders, so passing 1 expression must fail.
  const std::vector<Expr> es = {VInt(42)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{{}}", es), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsWhenTooManyExpressionsProvided) {
  const std::vector<Expr> es = {VInt(1), VInt(2), VInt(3)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{} {}", es), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsWhenTooFewExpressionsProvided) {
  EXPECT_THROW(
      ExprToStr::Format(symtab, "{}", std::vector<Expr>{}), std::format_error);

  const std::vector<Expr> es = {VInt(42)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{} {}", es), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsOnPositionalArgumentIndices) {
  const std::vector<Expr> es1 = {VInt(42)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{0}", es1), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{0:6x}", es1), std::format_error);

  const std::vector<Expr> es2 = {VInt(1), VInt(2)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{} {1}", es2), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsOnNestedBracesDynamicWidthOrPrecision) {
  // Rejected whether 1 expression or 2/3 expressions are passed
  const std::vector<Expr> es1 = {VInt(42)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{:>{}}", es1), std::format_error);

  const std::vector<Expr> es2 = {VInt(1), VInt(2)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{:>{}}", es2), std::format_error);

  const std::vector<Expr> es3 = {VFloat(1.5), VInt(8), VInt(2)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{:{}.{}f}", es3), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsOnUnmatchedBracesInPrefixMiddleOrTail) {
  const std::vector<Expr> es0;
  const std::vector<Expr> es1 = {VInt(42)};

  // Unmatched '{'
  EXPECT_THROW(ExprToStr::Format(symtab, "{", es0), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{:6x", es1), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{} {", es1), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{{{", es1), std::format_error);

  // Unmatched '}'
  EXPECT_THROW(ExprToStr::Format(symtab, "}", es0), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "} {}", es1), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{} }", es1), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{}}}}", es1), std::format_error);

  // '}' used illegally as a fill character slices early and throws
  EXPECT_THROW(ExprToStr::Format(symtab, "{:}>6}", es1), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsWhenIntExpressionUsesFloatSpecifier) {
  const std::vector<Expr> es = {VInt(42)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{:f}", es), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{:15.8g}", es), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{:e}", es), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{:.2}", es), std::format_error);
}

TEST_F(ExprToStrTest, ThrowsWhenFloatExpressionUsesIntSpecifier) {
  const std::vector<Expr> es = {VFloat(3.14)};
  EXPECT_THROW(ExprToStr::Format(symtab, "{:d}", es), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{:6x}", es), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{:#b}", es), std::format_error);
  EXPECT_THROW(ExprToStr::Format(symtab, "{:o}", es), std::format_error);
}

TEST_F(ExprToStrTest, NextSubFormatSlicesSequentiallyAndLeavesTail) {
  struct TestableExprToStr : public ExprToStr {
   public:
    using ExprToStr::NextSubFormat;
  };

  std::string_view fmt = "prefix {{ {:04d} mid }} {:.2f} suffix";

  EXPECT_EQ(TestableExprToStr::NextSubFormat(fmt), "prefix {{ {:04d}");
  EXPECT_EQ(fmt, " mid }} {:.2f} suffix");

  EXPECT_EQ(TestableExprToStr::NextSubFormat(fmt), " mid }} {:.2f}");
  EXPECT_EQ(fmt, " suffix");

  EXPECT_THROW(TestableExprToStr::NextSubFormat(fmt), std::format_error);
}

TEST_F(ExprToStrTest, PrintExpressionsUsesDefaultFormatWhenFmtIsEmpty) {
  const std::vector<Expr> es = {VInt(7), VFloat(2.5)};

  ::testing::internal::CaptureStdout();
  PrintExpressions(symtab, "", es);
  const std::string output = ::testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "7 2.5\n");
}

TEST_F(ExprToStrTest, PrintExpressionsUsesExplicitFormatWhenFmtIsNonEmpty) {
  const std::vector<Expr> es = {VInt(255)};

  ::testing::internal::CaptureStdout();
  PrintExpressions(symtab, "val={:#x}", es);
  const std::string output = ::testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "val=0xff\n");
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
