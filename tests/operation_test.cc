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

#include <variant>

#include "gtest/gtest.h"

#include "../lib/gate.h"
#include "../lib/operation.h"
#include "../lib/operation_base.h"

namespace qsim {

TEST(OperationTest, Test1) {
  using Gate = qsim::Gate<float>;
  using ControlledGate = qsim::ControlledGate<float>;
  using Operation = qsim::Operation<float>;

  {
    Gate op1 = {0, 1, {2}};
    const Gate op2 = {0, 1, {2}};

    Gate* pg1 = OpGetAlternative<Gate>(op1);
    ASSERT_NE(pg1, nullptr);
    EXPECT_EQ(pg1->time, 1);
    EXPECT_EQ(pg1->qubits.size(), 1);

    const Gate* pg2 = OpGetAlternative<Gate>(&op2);
    ASSERT_NE(pg2, nullptr);
    EXPECT_EQ(pg2->time, 1);
    EXPECT_EQ(pg2->qubits.size(), 1);

    unsigned time1 = OpTime(op1);
    EXPECT_EQ(time1, 1);

    unsigned time2 = OpTime(op2);
    EXPECT_EQ(time2, 1);

    const Qubits& qubits1 = OpQubits(op1);
    EXPECT_EQ(qubits1.size(), 1);

    const Qubits& qubits2 = OpQubits(&op2);
    EXPECT_EQ(qubits2.size(), 1);

    {
      const BaseOperation& bop1 = OpBaseOperation(&op1);
      EXPECT_EQ(bop1.time, 1);
      EXPECT_EQ(bop1.qubits.size(), 1);

      const BaseOperation& bop2 = OpBaseOperation(&op2);
      EXPECT_EQ(bop2.time, 1);
      EXPECT_EQ(bop2.qubits.size(), 1);
    }

    {
      Gate* po1 = &op1;
      const Gate* const po2 = &op2;

      BaseOperation& bop1 = OpBaseOperation(po1);
      EXPECT_EQ(bop1.time, 1);
      EXPECT_EQ(bop1.qubits.size(), 1);

      EXPECT_EQ(op1.time, 1);
      bop1.time = 2;
      EXPECT_EQ(op1.time, 2);

      const BaseOperation& bop2 = OpBaseOperation(po2);
      EXPECT_EQ(bop2.time, 1);
      EXPECT_EQ(bop2.qubits.size(), 1);
    }

    ControlledGate* pg3 = OpGetAlternative<ControlledGate>(op1);
    EXPECT_EQ(pg3, nullptr);
  }

  {
    Operation op1 = Gate{0, 1, {2}};
    const Operation op2 = Gate{0, 1, {2}};

    Gate* pg1 = OpGetAlternative<Gate>(op1);
    ASSERT_NE(pg1, nullptr);
    EXPECT_EQ(pg1->time, 1);
    EXPECT_EQ(pg1->qubits.size(), 1);

    const Gate* pg2 = OpGetAlternative<Gate>(op2);
    ASSERT_NE(pg2, nullptr);
    EXPECT_EQ(pg2->time, 1);
    EXPECT_EQ(pg2->qubits.size(), 1);

    unsigned time1 = OpTime(op1);
    EXPECT_EQ(time1, 1);

    unsigned time2 = OpTime(op2);
    EXPECT_EQ(time2, 1);

    const Qubits& qubits1 = OpQubits(op1);
    EXPECT_EQ(qubits1.size(), 1);

    const Qubits& qubits2 = OpQubits(op2);
    EXPECT_EQ(qubits2.size(), 1);

    {
      BaseOperation& bop1 = OpBaseOperation(op1);
      EXPECT_EQ(bop1.time, 1);
      EXPECT_EQ(bop1.qubits.size(), 1);

      EXPECT_EQ(std::get<Gate>(op1).time, 1);
      bop1.time = 2;
      EXPECT_EQ(std::get<Gate>(op1).time, 2);

      const BaseOperation& bop2 = OpBaseOperation(op2);
      EXPECT_EQ(bop2.time, 1);
      EXPECT_EQ(bop2.qubits.size(), 1);
    }

    {
      Operation* const po1 = &op1;
      const Operation* const po2 = &op2;

      const BaseOperation& bop1 = OpBaseOperation(po1);
      EXPECT_EQ(bop1.time, 2);
      EXPECT_EQ(bop1.qubits.size(), 1);

      const BaseOperation& bop2 = OpBaseOperation(po2);
      EXPECT_EQ(bop2.time, 1);
      EXPECT_EQ(bop2.qubits.size(), 1);
    }

    ControlledGate* pg3 = OpGetAlternative<ControlledGate>(op1);
    EXPECT_EQ(pg3, nullptr);
  }

  {
    Operation op1 = ControlledGate{Gate{0, 1, {2}}, {0, 1}};
    const Operation op2 = ControlledGate{Gate{0, 1, {2}}, {0, 1}};
    Operation* po1 = &op1;

    ControlledGate* pg1 = OpGetAlternative<ControlledGate>(po1);
    ASSERT_NE(pg1, nullptr);
    EXPECT_EQ(pg1->time, 1);
    EXPECT_EQ(pg1->qubits.size(), 1);
    EXPECT_EQ(pg1->controlled_by.size(), 2);

    const ControlledGate* pg2 = OpGetAlternative<ControlledGate>(&op2);
    ASSERT_NE(pg2, nullptr);
    EXPECT_EQ(pg2->time, 1);
    EXPECT_EQ(pg2->qubits.size(), 1);
    EXPECT_EQ(pg2->controlled_by.size(), 2);

    unsigned time1 = OpTime(&op1);
    EXPECT_EQ(time1, 1);

    unsigned time2 = OpTime(&op2);
    EXPECT_EQ(time2, 1);

    const Qubits& qubits1 = OpQubits(op1);
    EXPECT_EQ(qubits1.size(), 1);

    const Qubits& qubits2 = OpQubits(&op2);
    EXPECT_EQ(qubits2.size(), 1);

    BaseOperation& bop1 = OpBaseOperation(op1);
    EXPECT_EQ(bop1.time, 1);
    EXPECT_EQ(bop1.qubits.size(), 1);

    const BaseOperation& bop2 = OpBaseOperation(&op2);
    EXPECT_EQ(bop2.time, 1);
    EXPECT_EQ(bop2.qubits.size(), 1);

    const Gate* pg3 = OpGetAlternative<Gate>(&op1);
    EXPECT_EQ(pg3, nullptr);
  }
}

TEST(OperationTest, Test2) {
  using Gate = qsim::Gate<float>;
  using FusedGate = qsim::FusedGate<float>;
  using Operation = qsim::Operation<float>;
  using OperationF = std::variant<FusedGate, Measurement, const Operation*>;

  {
    Operation op1 = Gate{0, 1, {2}};
    const OperationF opf1 = &op1;

    const Gate* pg1 = OpGetAlternative<Gate>(opf1);
    ASSERT_NE(pg1, nullptr);
    EXPECT_EQ(pg1->time, 1);
    EXPECT_EQ(pg1->qubits.size(), 1);

    unsigned time1 = OpTime(&opf1);
    EXPECT_EQ(time1, 1);

    const Qubits& qubits1 = OpQubits(&opf1);
    EXPECT_EQ(qubits1.size(), 1);

    const BaseOperation& bop1 = OpBaseOperation(opf1);
    EXPECT_EQ(bop1.time, 1);
    EXPECT_EQ(bop1.qubits.size(), 1);

    const Measurement* pg3 = OpGetAlternative<Measurement>(&opf1);
    EXPECT_EQ(pg3, nullptr);
  }

  {
    OperationF opf1 = CreateMeasurement(1, {1});
    const OperationF opf2 = CreateMeasurement(1, {1});

    EXPECT_EQ(std::get<Measurement>(opf1).time, 1);

    Measurement* pm1 = OpGetAlternative<Measurement>(opf1);
    ASSERT_NE(pm1, nullptr);
    EXPECT_EQ(pm1->time, 1);
    EXPECT_EQ(pm1->qubits.size(), 1);

    pm1->time = 2;
    EXPECT_EQ(std::get<Measurement>(opf1).time, 2);

    const Measurement* pm2 = OpGetAlternative<Measurement>(opf2);
    ASSERT_NE(pm2, nullptr);
    EXPECT_EQ(pm2->time, 1);
    EXPECT_EQ(pm2->qubits.size(), 1);

    unsigned time1 = OpTime(&opf1);
    EXPECT_EQ(time1, 2);

    unsigned time2 = OpTime(opf2);
    EXPECT_EQ(time2, 1);

    const Qubits& qubits1 = OpQubits(opf1);
    EXPECT_EQ(qubits1.size(), 1);

    const Qubits& qubits2 = OpQubits(&opf2);
    EXPECT_EQ(qubits2.size(), 1);

    const BaseOperation& bop2 = OpBaseOperation(&opf2);
    EXPECT_EQ(bop2.time, 1);
    EXPECT_EQ(bop2.qubits.size(), 1);

    const Gate* pg3 = OpGetAlternative<Gate>(opf2);
    EXPECT_EQ(pg3, nullptr);
  }
}

TEST(OperationTest, Test3) {
  using Gate = qsim::Gate<float>;
  using FusedGate = qsim::FusedGate<float>;
  using Operation = qsim::Operation<float>;
  using OperationF = std::variant<FusedGate, Measurement, Operation*>;

  {
    Operation op1 = Gate{0, 1, {2}};
    OperationF opf1 = &op1;

    const Gate* pg1 = OpGetAlternative<Gate>(opf1);
    ASSERT_NE(pg1, nullptr);
    EXPECT_EQ(pg1->time, 1);
    EXPECT_EQ(pg1->qubits.size(), 1);

    unsigned time1 = OpTime(&opf1);
    EXPECT_EQ(time1, 1);

    const Qubits& qubits1 = OpQubits(&opf1);
    EXPECT_EQ(qubits1.size(), 1);

    BaseOperation& bop1 = OpBaseOperation(opf1);
    EXPECT_EQ(bop1.time, 1);
    EXPECT_EQ(bop1.qubits.size(), 1);

    const Measurement* pg3 = OpGetAlternative<Measurement>(&opf1);
    EXPECT_EQ(pg3, nullptr);
  }

  {
    OperationF opf1 = CreateMeasurement(1, {1});
    const OperationF opf2 = CreateMeasurement(1, {1});

    EXPECT_EQ(std::get<Measurement>(opf1).time, 1);

    Measurement* pm1 = OpGetAlternative<Measurement>(opf1);
    ASSERT_NE(pm1, nullptr);
    EXPECT_EQ(pm1->time, 1);
    EXPECT_EQ(pm1->qubits.size(), 1);

    pm1->time = 2;
    EXPECT_EQ(std::get<Measurement>(opf1).time, 2);

    const Measurement* pm2 = OpGetAlternative<Measurement>(opf2);
    ASSERT_NE(pm2, nullptr);
    EXPECT_EQ(pm2->time, 1);
    EXPECT_EQ(pm2->qubits.size(), 1);

    unsigned time1 = OpTime(&opf1);
    EXPECT_EQ(time1, 2);

    unsigned time2 = OpTime(opf2);
    EXPECT_EQ(time2, 1);

    const Qubits& qubits1 = OpQubits(opf1);
    EXPECT_EQ(qubits1.size(), 1);

    const Qubits& qubits2 = OpQubits(&opf2);
    EXPECT_EQ(qubits2.size(), 1);

    OperationF* pof1 = &opf1;
    BaseOperation& bop1 = OpBaseOperation(pof1);
    EXPECT_EQ(bop1.time, 2);
    EXPECT_EQ(bop1.qubits.size(), 1);

    const BaseOperation& bop2 = OpBaseOperation(&opf2);
    EXPECT_EQ(bop2.time, 1);
    EXPECT_EQ(bop2.qubits.size(), 1);

    const Gate* pg3 = OpGetAlternative<Gate>(opf2);
    EXPECT_EQ(pg3, nullptr);
  }
}

}  // namespace qsim

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
