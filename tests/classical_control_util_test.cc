#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
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
}

/*
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
*/

TEST(UtilsTest, ExprToStrRun1ExplicitFormat) {
  SymTable symtab;
  auto scope = symtab.AddScope();
  symtab.EnterScope(scope);

  symtab.Insert("x", Symbol(TInt{10}));
  symtab.Insert("y", Symbol(TFloat{2.5}));

  std::vector<Expr> es = {TSymbol{"x"}, TSymbol{"y"}};

  // Run1 uses std::vformat with user string "x = {}, y = {}"
  std::string res = ExprToStr<2>::Run1(symtab, "x = {}, y = {}", es);
  EXPECT_EQ(res, "x = 10, y = 2.5");
}

TEST(UtilsTest, ExprToStrRun2AutoFormat) {
  SymTable symtab;
  auto scope = symtab.AddScope();
  symtab.EnterScope(scope);

  std::vector<Expr> es = {TInt{100}, TFloat{3.14}};

  // Run2 auto-builds space-separated "{}" format specifier string
  std::string res = ExprToStr<2>::Run2(symtab, std::string{}, es);
  EXPECT_EQ(res, "100 3.14");
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
