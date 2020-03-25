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

TEST(SampleStatesTest, BasicStateSampling) {
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

  unsigned num_samples = 100;
  unsigned seed = 0;
  std::vector<uint64_t> measurements;
  sampler.SampleState(state_space, state, num_samples, seed, &measurements);
  // Prior to applying gates, only the zero state should be observed.
  for (const auto& measured_state : measurements) {
    ASSERT_EQ(measured_state, 0);
  }

  // Construct a Bell state of |01> and |10>.
  auto h_0_gate = GateHd<float>::Create(0, 0);
  auto x_1_gate = GateX<float>::Create(0, 1);
  auto cx_0_1_gate = GateCNot<float>::Create(1, 0, 1);
  ApplyGate(simulator, h_0_gate, state);
  ApplyGate(simulator, x_1_gate, state);
  ApplyGate(simulator, cx_0_1_gate, state);

  // Only |01> and |10> (i.e. 1 and 2) should be observed.
  std::set<uint64_t> allowed_results = {1, 2};
  std::vector<uint64_t> bell_measurements;
  sampler.SampleState(state_space, state, num_samples, seed,
                      &bell_measurements);
  for (const auto& measured_state : bell_measurements) {
    ASSERT_TRUE(allowed_results.find(measured_state) != allowed_results.end());
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
