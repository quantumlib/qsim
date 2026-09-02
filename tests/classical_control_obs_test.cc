#include <gtest/gtest.h>

#include <string_view>
#include <unordered_map>
#include <utility>

#include "../lib/classical_control_obs.h"

namespace qsim::cc {
namespace {

TEST(MeasurementHistogramTest, DefaultConstructor) {
  MeasurementHistogram hist;
  EXPECT_EQ(hist.Size(), 0u);
  EXPECT_TRUE(hist.cur_count.empty());
  EXPECT_TRUE(hist.total_count.empty());
}

TEST(MeasurementHistogramTest, ParameterizedConstructorAllocatesCorrectSize) {
  unsigned num_qubits = 3;
  MeasurementHistogram hist(num_qubits);

  EXPECT_EQ(hist.Size(), 3u);
  // 2^3 = 8 outcome entries
  EXPECT_EQ(hist.cur_count.size(), 8u);
  EXPECT_EQ(hist.total_count.size(), 8u);

  for (uint64_t val : hist.cur_count) {
    EXPECT_EQ(val, 0u);
  }
  for (uint64_t val : hist.total_count) {
    EXPECT_EQ(val, 0u);
  }
}

TEST(MeasurementHistogramTest, UpdateAccumulatesAndResetsCurrentCounts) {
  MeasurementHistogram hist(2); // 4 outcomes: 00, 01, 10, 11

  // First Trajectory
  hist.cur_count[0b00] = 5;
  hist.cur_count[0b11] = 2;
  hist.Update();

  // Verify total count updated and cur_count reset
  EXPECT_EQ(hist.total_count[0b00], 5u);
  EXPECT_EQ(hist.total_count[0b11], 2u);
  EXPECT_EQ(hist.cur_count[0b00], 0u);
  EXPECT_EQ(hist.cur_count[0b11], 0u);

  // Second Trajectory
  hist.cur_count[0b00] = 3;
  hist.cur_count[0b01] = 4;
  hist.Update();

  EXPECT_EQ(hist.total_count[0b00], 8u); // 5 + 3
  EXPECT_EQ(hist.total_count[0b01], 4u);
  EXPECT_EQ(hist.total_count[0b11], 2u);
  EXPECT_EQ(hist.cur_count[0b00], 0u);
}

TEST(MeasurementHistogramTest, DiscardClearsCurrentCountsWithoutUpdatingTotal) {
  MeasurementHistogram hist(2);

  hist.total_count[0b00] = 10;
  hist.cur_count[0b00] = 5;
  hist.cur_count[0b10] = 3;

  hist.Discard();

  // Verify cur_count cleared and total_count unchanged
  EXPECT_EQ(hist.cur_count[0b00], 0u);
  EXPECT_EQ(hist.cur_count[0b10], 0u);
  EXPECT_EQ(hist.total_count[0b00], 10u);
  EXPECT_EQ(hist.total_count[0b10], 0u);
}

TEST(ObservablesTest, InitialStateIsEmpty) {
  Observables obss;
  EXPECT_TRUE(obss.Empty());
  EXPECT_EQ(obss.Lookup("nonexistent"), nullptr);
}

TEST(ObservablesTest, InsertAndLookup) {
  Observables obss;

  Observable* inserted = obss.Insert("m0", MeasurementHistogram(2));
  ASSERT_NE(inserted, nullptr);
  EXPECT_FALSE(obss.Empty());
  EXPECT_EQ(inserted->Size(), 2u);

  // Lookup non-const
  Observable* found = obss.Lookup("m0");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found, inserted);

  // Const Lookup
  const Observables& const_obss = obss;
  const Observable* const_found = const_obss.Lookup("m0");
  ASSERT_NE(const_found, nullptr);
  EXPECT_EQ(const_found->Size(), 2u);
}

TEST(ObservablesTest, IteratesOverRegisteredObservables) {
  Observables obss;
  obss.Insert("h1", MeasurementHistogram(1));
  obss.Insert("h2", MeasurementHistogram(2));

  std::unordered_map<std::string_view, unsigned> visited;

  obss.Iterate([&visited](std::string_view name, Observable& obs) {
    visited[name] = obs.Size();
  });

  EXPECT_EQ(visited.size(), 2u);
  EXPECT_EQ(visited["h1"], 1u);
  EXPECT_EQ(visited["h2"], 2u);

  // Const iteration check
  const Observables& const_obss = obss;
  std::size_t const_count = 0;
  const_obss.Iterate([&const_count](std::string_view, const Observable&) {
    ++const_count;
  });
  EXPECT_EQ(const_count, 2u);
}

}  // namespace
}  // namespace qsim::cc

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
