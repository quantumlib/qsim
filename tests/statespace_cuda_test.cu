// Copyright 2019 Google LLC. All Rights Reserved.
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

#include "statespace_testfixture.h"

#include "gtest/gtest.h"

#include "../lib/simulator_cuda.h"
#include "../lib/statespace_cuda.h"

namespace qsim {

template <class T>
class StateSpaceCUDATest : public testing::Test {};

using fp_impl = ::testing::Types<float, double>;

TYPED_TEST_SUITE(StateSpaceCUDATest, fp_impl);

template <typename fp_type>
struct Factory {
  using Simulator = qsim::SimulatorCUDA<fp_type>;
  using StateSpace = typename Simulator::StateSpace;

  Factory(const typename StateSpace::Parameter& param1,
          const typename Simulator::Parameter& param2)
      : param1(param1), param2(param2) {}

  StateSpace CreateStateSpace() const {
    return StateSpace(param1);
  }

  Simulator CreateSimulator() const {
    return Simulator(param2);
  }

  typename StateSpace::Parameter param1;
  typename Simulator::Parameter param2;
};

TYPED_TEST(StateSpaceCUDATest, Add) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestAdd(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, NormSmall) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestNormSmall(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, NormAndInnerProductSmall) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestNormAndInnerProductSmall(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, NormAndInnerProduct) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestNormAndInnerProduct(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, SamplingSmall) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestSamplingSmall(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, SamplingCrossEntropyDifference) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {16}) {
    for (unsigned num_threads : {256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestSamplingCrossEntropyDifference(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, Ordering) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestOrdering(factory);
    }
  }
}

TEST(StateSpaceCUDATest, MeasurementSmall) {
  using Factory = qsim::Factory<float>;
  Factory::StateSpace::Parameter param1;
  Factory::Simulator::Parameter param2;
  Factory factory(param1, param2);
  TestMeasurementSmall(factory, true);
}

TYPED_TEST(StateSpaceCUDATest, MeasurementLarge) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestMeasurementLarge(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, Collapse) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestCollapse(factory);
    }
  }
}

TEST(StateSpaceCUDATest, InvalidStateSize) {
  using Factory = qsim::Factory<float>;
  Factory::StateSpace::Parameter param1;
  Factory::Simulator::Parameter param2;
  Factory factory(param1, param2);
  TestInvalidStateSize(factory);
}

TYPED_TEST(StateSpaceCUDATest, BulkSetAmpl) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestBulkSetAmplitude(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, BulkSetAmplExclusion) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestBulkSetAmplitudeExclusion(factory);
    }
  }
}

TYPED_TEST(StateSpaceCUDATest, BulkSetAmplDefault) {
  using Factory = qsim::Factory<TypeParam>;

  for (unsigned num_dblocks : {2, 16}) {
    for (unsigned num_threads : {64, 256, 1024}) {
      typename Factory::StateSpace::Parameter param;
      param.num_threads = num_threads;

      Factory factory(param, typename Factory::Simulator::Parameter());

      TestBulkSetAmplitudeDefault(factory);
    }
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
