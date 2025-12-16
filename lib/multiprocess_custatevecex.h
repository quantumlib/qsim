// Copyright 2025 Google LLC. All Rights Reserved.
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

#ifndef MULTIPROCESS_CUSTATEVECEX_H_
#define MULTIPROCESS_CUSTATEVECEX_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <custatevecEx.h>
#include <custatevecEx_ext.h>

#include <cstdint>
#include <vector>

#include "io.h"
#include "util_custatevec.h"
#include "util_custatevecex.h"

namespace qsim {

struct MultiProcessCuStateVecEx {
  enum NetworkType {
    kSuperPod = 0,
    kGB200NVL = 1,
    kSwitchTree = 2,
    kCommunicator = 3,
  };

  struct Parameter {
    uint64_t transfer_buffer_size = 16777216;
    NetworkType network_type = kSuperPod;
  };

  MultiProcessCuStateVecEx(Parameter param = Parameter{16777216, kSuperPod})
      : param_(param), communicator_(nullptr), initialized_(false) {}

  ~MultiProcessCuStateVecEx() {
    if (communicator_) {
      custatevecExCommunicatorDestroy(communicator_);
    }

    custatevecExCommunicatorStatus_t status;
    custatevecExCommunicatorFinalize(&status);
  }

  custatevecExCommunicatorDescriptor_t communicator() const {
    return communicator_;
  }

  unsigned num_processes() const {
    return num_processes_;
  }

  unsigned rank() const {
    return rank_;
  }

  bool initialized() const {
    return initialized_;
  }

  void initialize() {
    int argc = 0;
    char** argv = nullptr;

    auto comm_type = CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI;

    custatevecExCommunicatorStatus_t comm_status;
    auto status = custatevecExCommunicatorInitialize(
        comm_type, nullptr, &argc, &argv, &comm_status);

    if (status != CUSTATEVEC_STATUS_SUCCESS ||
        comm_status != CUSTATEVEC_EX_COMMUNICATOR_STATUS_SUCCESS) {
      return;
    }

    communicator_ = nullptr;
    status = custatevecExCommunicatorCreate(&communicator_);

    if (status != CUSTATEVEC_STATUS_SUCCESS) {
      return;
    }

    int num_processes, rank;
    ErrorCheck(communicator_->intf->getSize(communicator_, &num_processes));
    ErrorCheck(communicator_->intf->getRank(communicator_, &rank));

    ErrorCheck(communicator_->intf->getRank(communicator_, &rank));
    if (rank != 0) {
      output::enabled = false;
    }

    if (num_processes < 2 || (num_processes & (num_processes - 1)) != 0) {
      return;
    }

    num_global_qubits_ = get_num_global_qubits(num_processes);

    unsigned num_acc_global_qubits = 0;
    auto network_layers = get_network_layers(param_.network_type);

    num_global_qubits_per_layer_.reserve(2);
    global_index_bit_classes_.reserve(2);

    for (const auto& layer : network_layers) {
      auto k = num_global_qubits_ - num_acc_global_qubits;
      global_index_bit_classes_.push_back(layer.global_index_bit_class);

      if (layer.num_global_qubits == 0 || k <= layer.num_global_qubits) {
        num_global_qubits_per_layer_.push_back(k);
        num_acc_global_qubits = num_global_qubits_;
        break;
      }

      num_global_qubits_per_layer_.push_back(layer.num_global_qubits);
      num_acc_global_qubits += layer.num_global_qubits;
    }

    if (num_acc_global_qubits < num_global_qubits_) {
      IO::errorf("erorr: too few network layers at %s %d.\n",
                 __FILE__, __LINE__);
      exit(1);
    }

    memory_sharing_method_ = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE;

    for (const auto& layer : network_layers) {
      if (layer.global_index_bit_class ==
          CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P) {
        memory_sharing_method_ = CUSTATEVEC_EX_MEMORY_SHARING_METHOD_AUTODETECT;
        break;
      }
    }

    num_processes_ = num_processes;
    rank_ = rank;
    initialized_ = true;
  }

  auto create_sv_config(unsigned num_qubits, cudaDataType_t data_type) const {
    custatevecExDictionaryDescriptor_t sv_config = nullptr;

    if (!initialized_ ||
        num_qubits < 3 || num_global_qubits_ + 2 > num_qubits) {
      return sv_config;
    }

    unsigned num_local_qubits = num_qubits - num_global_qubits_;

    ErrorCheck(custatevecExConfigureStateVectorMultiProcess(
        &sv_config, data_type, num_qubits, num_local_qubits, -1,
        memory_sharing_method_, global_index_bit_classes_.data(),
        (int32_t*) num_global_qubits_per_layer_.data(),
        (int32_t) global_index_bit_classes_.size(),
        param_.transfer_buffer_size, nullptr, 0));

    return sv_config;
  }

 private:
  Parameter param_;
  custatevecExCommunicatorDescriptor_t communicator_;
  std::vector<unsigned> num_global_qubits_per_layer_;
  std::vector<custatevecExGlobalIndexBitClass_t> global_index_bit_classes_;
  custatevecExMemorySharingMethod_t memory_sharing_method_;
  unsigned num_processes_;
  unsigned num_global_qubits_;
  unsigned rank_;
  bool initialized_;

  struct NetworkLayer {
    custatevecExGlobalIndexBitClass_t global_index_bit_class;
    unsigned num_global_qubits;
  };

  using NetworkLayers = std::vector<NetworkLayer>;

  static NetworkLayers get_network_layers(NetworkType id) {
    switch (id) {
    case kSuperPod:
      return {{CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, 3},
             {CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR, 0}};
    case kGB200NVL:
      return {{CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, 0}};
      break;
    case kSwitchTree:
      return {{CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, 2},
             {CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P, 1}};
      break;
    case kCommunicator:
      return {{CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR, 0}};
      break;
    }

    return NetworkLayers{};
  }
};

}  // namespace qsim

#endif  // MULTIPROCESS_CUSTATEVECEX_H_
