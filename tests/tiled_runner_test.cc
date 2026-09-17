#include <cmath>
#include <complex>

#include "gtest/gtest.h"

#include "../lib/formux.h"
#include "../lib/fuser_mqubit.h"
#include "../lib/gates_qsim.h"
#include "../lib/io.h"
#include "../lib/run_qsim.h"
#include "../lib/run_qsim_tiled.h"
#include "../lib/simmux.h"

namespace {

using Operation = qsim::Operation<float>;
using Circuit = qsim::Circuit<Operation>;
using Fuser = qsim::MultiQubitGateFuser<qsim::IO>;

struct Factory {
  using Simulator = qsim::Simulator<qsim::For>;
  using StateSpace = typename Simulator::StateSpace;
  StateSpace CreateStateSpace() const { return StateSpace(1); }
  Simulator CreateSimulator() const { return Simulator(1); }
};

Circuit MakeCircuit(bool measurement) {
  Circuit circuit;
  circuit.num_qubits = 4;
  circuit.ops.push_back(qsim::GateHd<float>::Create(0, 3));
  circuit.ops.push_back(qsim::GateHd<float>::Create(1, 0));
  circuit.ops.push_back(qsim::GateCNot<float>::Create(2, 0, 3));
  circuit.ops.push_back(qsim::GateX<float>::Create(3, 1));
  circuit.ops.push_back(
      qsim::GateX<float>::Create(4, 2).ControlledBy(qsim::Qubits{3, 1}));
  if (measurement) {
    circuit.ops.push_back(qsim::CreateMeasurement(5, qsim::Qubits{3}));
    circuit.ops.push_back(qsim::GateHd<float>::Create(6, 2));
  }
  return circuit;
}

Circuit MakeRandomCircuit() {
  Circuit circuit;
  circuit.num_qubits = 6;
  circuit.ops = {
      qsim::GateHd<float>::Create(0, 0),
      qsim::GateX<float>::Create(1, 4),
      qsim::GateY<float>::Create(2, 2),
      qsim::GateCNot<float>::Create(3, 0, 5),
      qsim::GateCNot<float>::Create(4, 4, 1),
      qsim::GateZ<float>::Create(5, 3),
      qsim::GateX<float>::Create(6, 2).ControlledBy(qsim::Qubits{5}),
      qsim::GateHd<float>::Create(7, 5),
      qsim::GateCNot<float>::Create(8, 1, 3),
  };
  return circuit;
}

template <typename State>
void ExpectFirstAmplitudes(const typename Factory::StateSpace& space,
                           const State& baseline,
                           const qsim::TiledStats& tiled) {
  ASSERT_GE(tiled.first_amplitudes.size(), 8U);
  for (unsigned i = 0; i < 8; ++i) {
    auto expected = Factory::StateSpace::GetAmpl(baseline, i);
    EXPECT_NEAR(std::real(expected), tiled.first_amplitudes[i].real(), 2e-5);
    EXPECT_NEAR(std::imag(expected), tiled.first_amplitudes[i].imag(), 2e-5);
  }
  EXPECT_NEAR(space.Norm(baseline), tiled.norm, 2e-5);
}

void CheckCircuit(const Circuit& circuit, unsigned tile_qubits = 3,
                  qsim::TileSchedule schedule = qsim::TileSchedule::kXorSwizzle,
                  unsigned inner_threads = 1) {
  Factory factory;
  auto state_space = factory.CreateStateSpace();
  auto baseline = state_space.Create(circuit.num_qubits);
  ASSERT_FALSE(state_space.IsNull(baseline));
  state_space.SetStateZero(baseline);
  qsim::QSimRunner<qsim::IO, Fuser, Factory>::Parameter param;
  param.max_fused_size = 3;
  param.seed = 1;
  ASSERT_TRUE((qsim::QSimRunner<qsim::IO, Fuser, Factory>::Run(
      param, circuit, state_space, factory.CreateSimulator(), baseline)));

  qsim::TiledOptions options;
  options.tile_qubits = tile_qubits;
  options.outer_threads = 2;
  options.inner_threads = inner_threads;
  options.max_fused_size = 3;
  options.numa = false;
  options.schedule = qsim::TileSchedule::kXorSwizzle;
  qsim::TiledStats tiled;
  ASSERT_TRUE((qsim::RunQSimTiledSimdRemapped<qsim::IO, Fuser>(
      options, factory, circuit, &tiled)));
  ExpectFirstAmplitudes(state_space, baseline, tiled);
}

TEST(TiledRunnerTest, RemappingMatchesBaseline) {
  CheckCircuit(MakeCircuit(false));
}

TEST(TiledRunnerTest, MeasurementMatchesBaseline) {
  CheckCircuit(MakeCircuit(true));
}

TEST(TiledRunnerTest, TileSizesAndSchedulesMatchBaseline) {
  const Circuit circuit = MakeRandomCircuit();
  for (unsigned tile_qubits : {2U, 3U, 4U}) {
    for (auto schedule : {qsim::TileSchedule::kLinear,
                          qsim::TileSchedule::kRoundRobin,
                          qsim::TileSchedule::kCacheGroup,
                          qsim::TileSchedule::kNumaGroup,
                          qsim::TileSchedule::kXorSwizzle,
                          qsim::TileSchedule::kDynamic}) {
      CheckCircuit(circuit, tile_qubits, schedule,
                   tile_qubits == 2 ? 2U : 1U);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
