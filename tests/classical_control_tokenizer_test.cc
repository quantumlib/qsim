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

#include "../lib/classical_control_tokenizer.h"

namespace qsim::cc {
namespace {

void AssertToken(const Token& t, Token::TokenKind expected_kind,
                 std::string_view expected_val) {
  EXPECT_EQ(t.kind, expected_kind);
  EXPECT_EQ(t.val, expected_val);
}

TEST(TokenizerUtilsTest, PackAndUnpackLineColumn) {
  unsigned line = 42;
  unsigned col = 128;

  unsigned packed = Tokenizer::PackLineAndColumn(line, col);
  auto [unpacked_line, unpacked_col] = Tokenizer::UnpackLineAndColumn(packed);

  EXPECT_EQ(unpacked_line, line);
  EXPECT_EQ(unpacked_col, col);
}

TEST(TokenizerUtilsTest, PackAndUnpackBoundaryValues) {
  unsigned line = 1048575; // 20-bit max
  unsigned col = 4095;     // 12-bit max

  unsigned packed = Tokenizer::PackLineAndColumn(line, col);
  auto [unpacked_line, unpacked_col] = Tokenizer::UnpackLineAndColumn(packed);

  EXPECT_EQ(unpacked_line, line);
  EXPECT_EQ(unpacked_col, col);
}

TEST(TokenizerTest, Identifiers) {
  Tokenizer tok("foo _bar123 VAR_NAME_");

  AssertToken(tok(), Token::kIdentifier, "foo");
  AssertToken(tok(), Token::kIdentifier, "_bar123");
  AssertToken(tok(), Token::kIdentifier, "VAR_NAME_");
  AssertToken(tok(), Token::kEndOfFile, "");
}

TEST(TokenizerTest, Integers) {
  Tokenizer tok("123 0x1a 0X1A 0b01 0B10");

  AssertToken(tok(), Token::kInteger, "123");
  AssertToken(tok(), Token::kInteger, "0x1a");
  AssertToken(tok(), Token::kInteger, "0X1A");
  AssertToken(tok(), Token::kInteger, "0b01");
  AssertToken(tok(), Token::kInteger, "0B10");
  AssertToken(tok(), Token::kEndOfFile, "");
}

TEST(TokenizerTest, FloatingPointNumbers) {
  Tokenizer tok("301.34 1.2e-5 0.0 5.0e+10 1e9");

  AssertToken(tok(), Token::kFloat, "301.34");
  AssertToken(tok(), Token::kFloat, "1.2e-5");
  AssertToken(tok(), Token::kFloat, "0.0");
  AssertToken(tok(), Token::kFloat, "5.0e+10");
  AssertToken(tok(), Token::kFloat, "1e9");
  AssertToken(tok(), Token::kEndOfFile, "");
}

TEST(TokenizerTest, StringLiterals) {
  Tokenizer tok("\"hello world\" 'single quoted'");

  AssertToken(tok(), Token::kString, "hello world");
  AssertToken(tok(), Token::kString, "single quoted");
  AssertToken(tok(), Token::kEndOfFile, "");
}

TEST(TokenizerTest, Operators) {
  // Test both multi-character and single-character operators
  Tokenizer tok("|| ^^ && == != <= >= ** << >> - + * / = < > ! ~ | ^ &");

  const std::vector<std::string_view> expected_ops = {
      "||", "^^", "&&", "==", "!=", "<=", ">=", "**", "<<", ">>", "-", "+",
      "*", "/", "=", "<", ">", "!", "~", "|", "^", "&"};

  for (std::string_view op : expected_ops) {
    AssertToken(tok(), Token::kOperator, op);
  }

  AssertToken(tok(), Token::kEndOfFile, "");
}

TEST(TokenizerTest, BracketsAndDelimiters) {
  Tokenizer tok("( ) [ ] ; \n \r\n");

  AssertToken(tok(), Token::kBracket, "(");
  AssertToken(tok(), Token::kBracket, ")");
  AssertToken(tok(), Token::kBracket, "[");
  AssertToken(tok(), Token::kBracket, "]");
  AssertToken(tok(), Token::kDelimiter, ";");
  AssertToken(tok(), Token::kDelimiter, "\n");
  AssertToken(tok(), Token::kDelimiter, "\n");
  AssertToken(tok(), Token::kEndOfFile, "");
}

TEST(TokenizerTest, InvalidTokens) {
  const std::vector<std::string_view> contains_invalid_tokens = {
    "\"unclosed string", "0b", "0b2", "0x", "1e", "1ek", "`",
  };

  for (auto s : contains_invalid_tokens) {
    Tokenizer tok(s);

    Token t = tok();
    EXPECT_EQ(t.kind, Token::kInvalid);
  }
}

TEST(TokenizerTest, LineAndColumn) {
  Tokenizer tok("a b \n c d\n  e\n\t f");

  {
    Token t = tok();
    auto [line, col] = Tokenizer::UnpackLineAndColumn(t.lc);
    EXPECT_EQ(line, 0);
    EXPECT_EQ(col, 0);
  }

  {
    Token t = tok();
    auto [line, col] = Tokenizer::UnpackLineAndColumn(t.lc);
    EXPECT_EQ(line, 0);
    EXPECT_EQ(col, 2);
  }

  tok();

  {
    Token t = tok();
    auto [line, col] = Tokenizer::UnpackLineAndColumn(t.lc);
    EXPECT_EQ(line, 1);
    EXPECT_EQ(col, 1);
  }

  {
    Token t = tok();
    auto [line, col] = Tokenizer::UnpackLineAndColumn(t.lc);
    EXPECT_EQ(line, 1);
    EXPECT_EQ(col, 3);
  }

  tok();

  {
    Token t = tok();
    auto [line, col] = Tokenizer::UnpackLineAndColumn(t.lc);
    EXPECT_EQ(line, 2);
    EXPECT_EQ(col, 2);
  }

  tok();

  {
    Token t = tok();
    auto [line, col] = Tokenizer::UnpackLineAndColumn(t.lc);
    EXPECT_EQ(line, 3);
    EXPECT_EQ(col, 2);  // '\t' is currently treated as a single character
  }
}

TEST(TokenizerStateTest, CurrentTokenTracking) {
  Tokenizer tok("x + 1");

  tok();
  EXPECT_EQ(tok.Current().kind, Token::kIdentifier);
  EXPECT_EQ(tok.Current().val, "x");

  tok();
  EXPECT_EQ(tok.Current().kind, Token::kOperator);
  EXPECT_EQ(tok.Current().val, "+");
}

TEST(TokenizerStateTest, PeekLookahead) {
  Tokenizer tok("a + b");

  // Peek(0) should preview 'a' without advancing
  Token peek0 = tok.Peek(0);
  AssertToken(peek0, Token::kIdentifier, "a");

  // Peek(1) previews '+'
  Token peek1 = tok.Peek(1);
  AssertToken(peek1, Token::kOperator, "+");

  // Peek(2) previews 'b'
  Token peek2 = tok.Peek(2);
  AssertToken(peek2, Token::kIdentifier, "b");

  // Peek past the end of the input
  Token peek3 = tok.Peek(3);
  AssertToken(peek3, Token::kEndOfFile, "");

  // Peek past the end of the input
  Token peek4 = tok.Peek(4);
  AssertToken(peek4, Token::kEndOfFile, "");

  // Calling operator() should still retrieve 'a'
  Token consumed = tok();
  AssertToken(consumed, Token::kIdentifier, "a");
}

TEST(TokenizerStateTest, Restart) {
  Tokenizer tok("foo 123");

  AssertToken(tok(), Token::kIdentifier, "foo");
  AssertToken(tok(), Token::kInteger, "123");

  // Restart back to beginning
  tok.Restart();

  AssertToken(tok(), Token::kIdentifier, "foo");
  AssertToken(tok(), Token::kInteger, "123");
  AssertToken(tok(), Token::kEndOfFile, "");
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
