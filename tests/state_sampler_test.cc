#include <complex>
#include <memory>

#include "gtest/gtest.h"

#include "../lib/gates_appl.h"
#include "../lib/gates_def.h"
#include "../lib/parfor.h"
#include "../lib/simulator_avx.h"
#include "../lib/state_sampler.h"
#include "../lib/statespace.h"

namespace qsim {

TEST(SampleStatesTest, ZeroState) {
  unsigned num_qubits = 2;
  unsigned num_threads = 1;

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = typename Simulator::StateSpace;
  using StateSampler = StateSampler<StateSpace>;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);
  StateSampler sampler;

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  unsigned num_samples = 100000;
  unsigned seed = 0;

  // Prior to applying gates, only the zero state should be observed.
  std::vector<float> expected_prob = {1, 0, 0, 0};
  std::vector<float> counts = {0, 0, 0, 0};
  std::vector<uint64_t> measurements;
  sampler.SampleState(state_space, state, num_samples, seed, &measurements);
  ASSERT_EQ(measurements.size(), num_samples);
  for (unsigned i = 0; i < measurements.size(); ++i) {
    int val = measurements[i];
    counts[val] += 1.0;
  }
  for (unsigned i = 0; i < counts.size(); ++i) {
    EXPECT_NEAR(counts[i] / num_samples, expected_prob[i], 1e-2);
  }
}

TEST(SampleStatesTest, BellState) {
  unsigned num_qubits = 2;
  unsigned num_threads = 1;

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = typename Simulator::StateSpace;
  using StateSampler = StateSampler<StateSpace>;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);
  StateSampler sampler;

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  unsigned num_samples = 100000;
  unsigned seed = 0;

  // Construct a Bell state of |01> and |10>.
  auto h_0_gate = GateHd<float>::Create(0, 0);
  auto x_1_gate = GateX<float>::Create(0, 1);
  auto cx_0_1_gate = GateCNot<float>::Create(1, 0, 1);
  ApplyGate(simulator, h_0_gate, state);
  ApplyGate(simulator, x_1_gate, state);
  ApplyGate(simulator, cx_0_1_gate, state);

  // Only |01> and |10> (i.e. 1 and 2) should be observed.
  std::vector<float> expected_prob = {0, 0.5, 0.5, 0};
  std::vector<float> counts = {0, 0, 0, 0};
  std::vector<uint64_t> measurements;
  sampler.SampleState(state_space, state, num_samples, seed, &measurements);
  ASSERT_EQ(measurements.size(), num_samples);
  for (unsigned i = 0; i < measurements.size(); ++i) {
    int val = measurements[i];
    counts[val] += 1.0;
  }
  for (unsigned i = 0; i < counts.size(); ++i) {
    EXPECT_NEAR(counts[i] / num_samples, expected_prob[i], 1e-2);
  }
}

TEST(SampleStatesTest, ArbitraryAmplitudes) {
  unsigned num_qubits = 3;
  unsigned num_threads = 1;

  using Simulator = SimulatorAVX<ParallelFor>;
  using StateSpace = typename Simulator::StateSpace;
  using StateSampler = StateSampler<StateSpace>;

  StateSpace state_space(num_qubits, num_threads);
  Simulator simulator(num_qubits, num_threads);
  StateSampler sampler;

  auto state = state_space.CreateState();
  state_space.SetStateZero(state);

  unsigned num_samples = 100000;
  unsigned seed = 0;

  // Assign amplitudes for each state with a rotation in the complex plane.
  std::vector<float> expected_prob = {0.0,  0.05, 0.07, 0.1,
                                      0.25, 0.2,  0.18, 0.15};
  float pi = 3.14159265358979323846;
  for (unsigned i = 0; i < expected_prob.size(); ++i) {
    float ampl = std::sqrt(expected_prob[i]);
    float real = std::cos(i * pi / 4.0) * ampl;
    float imag = std::sin(i * pi / 4.0) * ampl;
    state_space.SetAmpl(state, i, std::complex<float>(real, imag));
  }

  std::vector<float> counts = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> measurements;
  sampler.SampleState(state_space, state, num_samples, seed, &measurements);
  ASSERT_EQ(measurements.size(), num_samples);
  for (unsigned i = 0; i < measurements.size(); ++i) {
    int val = measurements[i];
    counts[val] += 1.0;
  }
  for (unsigned i = 0; i < counts.size(); ++i) {
    EXPECT_NEAR(counts[i] / num_samples, expected_prob[i], 1e-2);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
