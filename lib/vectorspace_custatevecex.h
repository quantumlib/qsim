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

#ifndef VECTORSPACE_CUSTATEVECEX_H_
#define VECTORSPACE_CUSTATEVECEX_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <custatevecEx.h>

#include <algorithm>
#include <complex>
#include <type_traits>
#include <vector>

#include "io.h"
#include "multiprocess_custatevecex.h"
#include "util_cuda.h"
#include "util_custatevec.h"
#include "util_custatevecex.h"

namespace qsim {

namespace detail {

inline void free(void* ptr) {}

}  // namespace detail

// Routines for vector manipulations.
template <typename Impl, typename FP>
class VectorSpaceCuStateVecEx {
 public:
  using fp_type = FP;

  static constexpr auto is_float = std::is_same<fp_type, float>::value;
  static constexpr auto kStateDataType = is_float ? CUDA_C_32F : CUDA_C_64F;

  enum DistributionType {
    kNoDistr,
    kSingleDevice,
    kMultiDevice,
    kMultiProcess,
  };

  enum DeviceNetworkType {
    kSwitch = 0,
    kFullMesh = 1,
  };

  struct Parameter {
    unsigned num_devices = 0;
    DeviceNetworkType device_network_type = kSwitch;
    unsigned verbosity = 0;
  };

  class Vector {
   public:
    struct CuStateVecResources {
      int32_t device_id = -1;
      void* device_ptr = nullptr;
      cudaStream_t stream = nullptr;
      custatevecHandle_t custatevec_handle = nullptr;
    };

    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    Vector() : mp_(nullptr), ptr_(nullptr),
        num_qubits_(0), num_substates_(0), distr_type_(kNoDistr) {}

    Vector(const MultiProcessCuStateVecEx* mp,
           custatevecExStateVectorDescriptor_t ptr, unsigned num_qubits,
           unsigned num_substates, DistributionType distr_type)
        : mp_(mp), ptr_(ptr), wire_ordering_(num_qubits),
          num_qubits_(num_qubits), num_substates_(num_substates),
          distr_type_(distr_type) {}

    Vector(Vector&& r) : mp_(r.mp_), ptr_(r.ptr_),
        wire_ordering_(std::move(r.wire_ordering_)),
        num_qubits_(r.num_qubits_), num_substates_(r.num_substates_),
        distr_type_(r.distr_type_) {
      r.mp_ = nullptr;
      r.ptr_ = nullptr;
      r.num_qubits_ = 0;
      r.num_substates_ = 0;
      r.distr_type_ = kNoDistr;
    }

    ~Vector() {
      if (ptr_ != nullptr) {
        ErrorCheck(custatevecExStateVectorDestroy(ptr_));
      }
    }

    Vector& operator=(Vector&& r) {
      if (this != &r) {
        mp_ = r.mp_;
        ptr_ = r.ptr_;
        wire_ordering_ = std::move(r.wire_ordering_);
        num_qubits_ = r.num_qubits_;
        num_substates_ = r.num_substates_;
        distr_type_ = r.distr_type_;

        r.mp_ = nullptr;
        r.ptr_ = nullptr;
        r.num_qubits_ = 0;
        r.num_substates_ = 0;
        r.distr_type_ = kNoDistr;
      }

      return *this;
    }

    auto get() {
      return ptr_;
    }

    const auto get() const {
      return ptr_;
    }

    custatevecExStateVectorDescriptor_t release() {
      auto ptr = ptr_;

      mp_ = nullptr;
      ptr_ = nullptr;
      num_qubits_ = 0;
      num_substates_ = 0;
      distr_type_ = kNoDistr;

      return ptr;
    }

    unsigned num_qubits() const {
      return num_qubits_;
    }

    unsigned num_substates() const {
      return num_substates_;
    }

    DistributionType distr_type() const {
      return distr_type_;
    }

    static constexpr bool requires_copy_to_host() {
      return true;
    }

    const auto& get_wire_ordering() const {
      ErrorCheck(custatevecExStateVectorGetProperty(
          ptr_, CUSTATEVEC_EX_SV_PROP_WIRE_ORDERING,
          const_cast<int32_t*>(wire_ordering_.data()),
          sizeof(int32_t) * num_qubits_));

      return wire_ordering_;
    }

    void to_normal_order() const {
      const auto& wire_ordering = get_wire_ordering();

      ErrorCheck(custatevecExStateVectorPermuteIndexBits(
          ptr_, wire_ordering.data(), num_qubits_,
          CUSTATEVEC_EX_PERMUTATION_SCATTER));
    }

    CuStateVecResources get_resources(unsigned substate_index) const {
      CuStateVecResources r;

      ErrorCheck(custatevecExStateVectorGetResourcesFromDeviceSubSV(
          ptr_, substate_index, &r.device_id, &r.device_ptr, &r.stream,
          &r.custatevec_handle));

      return r;
    }

    template <typename Callback>
    void assign(Callback&& callback) const {
      if (distr_type_ == kMultiProcess) {
        unsigned num_devices = 1;
        std::vector<int32_t> substate_indices(num_devices);

        ErrorCheck(custatevecExStateVectorGetProperty(
            ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
            substate_indices.data(), num_devices * sizeof(int32_t)));

        unsigned k = substate_indices[0];
        auto res = get_resources(k);

        ErrorCheck(cudaSetDevice(res.device_id));

        callback(k, res);
      } else {
        if (num_substates_ == 1) {
          callback(0, get_resources(0));
        } else {
          std::vector<int32_t> substate_indices(num_substates_);
          ErrorCheck(custatevecExStateVectorGetProperty(
              ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
              substate_indices.data(), num_substates_ * sizeof(int32_t)));

          for (unsigned i = 0; i < num_substates_; ++i) {
            unsigned k = substate_indices[i];
            auto res = get_resources(k);

            ErrorCheck(cudaSetDevice(res.device_id));

            callback(k, res);
          }
        }
      }
    }

    template <typename Callback>
    auto reduce(Callback&& callback) const {
      using ResultType = std::invoke_result_t<Callback, unsigned,
                                              CuStateVecResources>;

      if (distr_type_ == kMultiProcess) {
        unsigned num_devices = 1;
        std::vector<int32_t> substate_indices(num_devices);

        ErrorCheck(custatevecExStateVectorGetProperty(
            ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
            substate_indices.data(), num_devices * sizeof(int32_t)));

        unsigned k = substate_indices[0];
        auto res = get_resources(k);

        ErrorCheck(cudaSetDevice(res.device_id));

        ResultType r;
        ResultType local_r = callback(k, res);

        auto cuda_type = GetCudaType<ResultType>();
        auto comm = mp_->communicator();
        ErrorCheck(comm->intf->allreduce(comm, &local_r, &r, 1, cuda_type));

        return r;
      } else {
        if (num_substates_ == 1) {
          return callback(0, get_resources(0));
        } else {
          std::vector<int32_t> substate_indices(num_substates_);
          ErrorCheck(custatevecExStateVectorGetProperty(
              ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
              substate_indices.data(), num_substates_ * sizeof(int32_t)));

          ResultType r = 0;

          for (unsigned i = 0; i < num_substates_; ++i) {
            unsigned k = substate_indices[i];
            auto res = get_resources(k);

            ErrorCheck(cudaSetDevice(res.device_id));

            r += callback(k, res);
          }

          return r;
        }
      }
    }

    template <typename Callback>
    void assign(const Vector& vec, Callback&& callback) const {
      if (distr_type_ == kMultiProcess) {
        unsigned num_devices = 1;
        std::vector<int32_t> substate_indices(num_devices);

        ErrorCheck(custatevecExStateVectorGetProperty(
            ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
            substate_indices.data(), num_devices * sizeof(int32_t)));

        unsigned k = substate_indices[0];
        auto res1 = get_resources(k);
        auto res2 = vec.get_resources(k);

        ErrorCheck(cudaSetDevice(res1.device_id));

        callback(k, res1, res2);
      } else {
        if (num_substates_ == 1) {
          callback(0, get_resources(0), vec.get_resources(0));
        } else {
          std::vector<int32_t> substate_indices(num_substates_);
          ErrorCheck(custatevecExStateVectorGetProperty(
              ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
              substate_indices.data(), num_substates_ * sizeof(int32_t)));

          for (unsigned i = 0; i < num_substates_; ++i) {
            unsigned k = substate_indices[i];
            auto res1 = get_resources(k);
            auto res2 = vec.get_resources(k);

            ErrorCheck(cudaSetDevice(res1.device_id));

            callback(k, res1, res2);
          }
        }
      }
    }

    template <typename Callback>
    auto reduce(const Vector& vec, Callback&& callback) const {
      using ResultType = std::invoke_result_t<Callback, unsigned,
                                              CuStateVecResources,
                                              CuStateVecResources>;

      if (distr_type_ == kMultiProcess) {
        unsigned num_devices = 1;
        std::vector<int32_t> substate_indices(num_devices);

        ErrorCheck(custatevecExStateVectorGetProperty(
            ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
            substate_indices.data(), num_devices * sizeof(int32_t)));

        unsigned k = substate_indices[0];
        auto res1 = get_resources(k);
        auto res2 = vec.get_resources(k);

        ErrorCheck(cudaSetDevice(res2.device_id));
        ErrorCheck(cudaStreamSynchronize(res2.stream));

        ResultType r;
        ResultType local_r = callback(k, res1, res2);

        auto cuda_type = GetCudaType<ResultType>();
        auto comm = mp_->communicator();
        ErrorCheck(comm->intf->allreduce(comm, &local_r, &r, 1, cuda_type));

        return r;
      } else {
        if (num_substates_ == 1) {
          return callback(0, get_resources(0), vec.get_resources(0));
        } else {
          std::vector<int32_t> substate_indices(num_substates_);
          ErrorCheck(custatevecExStateVectorGetProperty(
              ptr_, CUSTATEVEC_EX_SV_PROP_DEVICE_SUBSV_INDICES,
              substate_indices.data(), num_substates_ * sizeof(int32_t)));

          ResultType r = 0;

          for (unsigned i = 0; i < num_substates_; ++i) {
            unsigned k = substate_indices[i];
            auto res1 = get_resources(k);
            auto res2 = vec.get_resources(k);

            ErrorCheck(cudaSetDevice(res2.device_id));
            ErrorCheck(cudaStreamSynchronize(res2.stream));

            r += callback(k, res1, res2);
          }

          return r;
        }
      }
    }

   private:
    const MultiProcessCuStateVecEx* mp_;
    custatevecExStateVectorDescriptor_t ptr_;
    std::vector<int32_t> wire_ordering_;
    unsigned num_qubits_;
    unsigned num_substates_;
    DistributionType distr_type_;
  };

  VectorSpaceCuStateVecEx(const Parameter& param,
                          const MultiProcessCuStateVecEx& mp)
      : param(param), mp(mp) {}

  Vector Create(unsigned num_qubits) const {
    custatevecExStateVectorDescriptor_t state_vec;
    custatevecExDictionaryDescriptor_t sv_config
        = mp.create_sv_config(num_qubits, kStateDataType);

    unsigned num_substates = 1;
    DistributionType distr_type = kNoDistr;

    if (sv_config != nullptr) {
      ErrorCheck(custatevecExStateVectorCreateMultiProcess(
          &state_vec, sv_config, nullptr, mp.communicator(), nullptr));

      num_substates = mp.num_processes();
      distr_type = kMultiProcess;

      if (param.verbosity > 2) {
        unsigned num_global_qubits = get_num_global_qubits(num_substates);
        IO::messagef("multi-process mode: %u %u.\n",
                     num_qubits, num_global_qubits);
      }
    } else {
      num_substates = param.num_devices;

      if (num_qubits < 3) {
        num_substates = 1;
      } else if (num_substates == 0) {
        int count = 0;
        ErrorCheck(cudaGetDeviceCount(&count));
        num_substates = count;
      }

      if (num_substates == 1) {
        ErrorCheck(custatevecExConfigureStateVectorSingleDevice(
            &sv_config, kStateDataType, num_qubits, num_qubits, -1, 0));

        distr_type = kSingleDevice;

        if (param.verbosity > 2) {
          IO::messagef("single device mode.\n");
        }
      } else {
        unsigned num_global_qubits = get_num_global_qubits(num_substates);

        while (num_global_qubits + 2 > num_qubits && num_substates > 1) {
          num_substates /= 2;
          --num_global_qubits;
        }

        if (num_substates == 1) {
          ErrorCheck(custatevecExConfigureStateVectorSingleDevice(
              &sv_config, kStateDataType, num_qubits, num_qubits, -1, 0));

          distr_type = kSingleDevice;

          if (param.verbosity > 2) {
            IO::messagef("single-device mode (too few qubits).\n");
          }
        } else {
          std::vector<int32_t> device_ids(num_substates);
          for (unsigned i = 0; i < num_substates; ++i) {
            device_ids[i] = i;
          }

          unsigned num_local_qubits = num_qubits - num_global_qubits;

          auto device_network_type =
            get_device_network_type(param.device_network_type);

          ErrorCheck(custatevecExConfigureStateVectorMultiDevice(
              &sv_config, kStateDataType, num_qubits, num_local_qubits,
              device_ids.data(), num_substates, device_network_type, 0));

          distr_type = kMultiDevice;

          if (param.verbosity > 2) {
            IO::messagef("multi-device mode: %u %u.\n",
                         num_qubits, num_global_qubits);
          }
        }
      }

      ErrorCheck(custatevecExStateVectorCreateSingleProcess(
        &state_vec, sv_config, nullptr, 0, nullptr));
    }

    ErrorCheck(custatevecExDictionaryDestroy(sv_config));

    return Vector{&mp, state_vec, num_qubits, num_substates, distr_type};
  }

  static Vector Null() {
    return Vector{nullptr, nullptr, 0, 0, kNoDistr};
  }

  static bool IsNull(const Vector& vector) {
    return vector.get() == nullptr;
  }

  bool Copy(const Vector& src, Vector& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    uint64_t size = (uint64_t{1} << src.num_qubits()) / src.num_substates();

    auto f = [&size](unsigned i, const auto& rd, const auto& rs) {
      ErrorCheck(cudaMemcpy(
          rd.device_ptr, rs.device_ptr, 2 * sizeof(fp_type) * size,
          cudaMemcpyDeviceToDevice));
    };

    dest.assign(src, f);

    return true;
  }

  // It is the client's responsibility to make sure that dest has at least
  // 2^src.num_qubits() elements.
  bool Copy(const Vector& src, fp_type* dest) const {
    if (src.distr_type() == kMultiProcess) {
      uint64_t size = (uint64_t{1} << src.num_qubits()) / src.num_substates();
      uint64_t offset = size * mp.rank();

      ErrorCheck(custatevecExStateVectorGetState(
          src.get(), dest + 2 * offset, kStateDataType,
          offset, offset + size, 1));
      ErrorCheck(custatevecExStateVectorSynchronize(src.get()));

      auto cuda_type = GetCudaType<std::complex<fp_type>>();
      auto comm = mp.communicator();
      ErrorCheck(comm->intf->allgather(
          comm, dest + 2 * offset, dest, size, cuda_type));
    } else {
      uint64_t size = uint64_t{1} << src.num_qubits();
      ErrorCheck(custatevecExStateVectorGetState(
          src.get(), dest, kStateDataType, 0, size, 1));
      ErrorCheck(custatevecExStateVectorSynchronize(src.get()));
    }

    return true;
  }

  // It is the client's responsibility to make sure that src has at least
  // 2^dest.num_qubits() elements.
  bool Copy(const fp_type* src, Vector& dest) const {
    if (dest.distr_type() == kMultiProcess) {
      uint64_t size = (uint64_t{1} << dest.num_qubits()) / dest.num_substates();
      uint64_t offset = size * mp.rank();

      ErrorCheck(custatevecExStateVectorSetState(
          dest.get(), src + 2 * offset, kStateDataType,
          offset, offset + size, 1));
    } else {
      uint64_t size = uint64_t{1} << dest.num_qubits();
      ErrorCheck(custatevecExStateVectorSetState(
          dest.get(), src, kStateDataType, 0, size, 1));
    }

    ErrorCheck(custatevecExStateVectorSynchronize(dest.get()));

    // TODO: do we need that?
    dest.to_normal_order();

    return true;
  }

  // It is the client's responsibility to make sure that src has at least
  // 2^dest.num_qubits() elements.
  bool Copy(const fp_type* src, uint64_t size, Vector& dest) const {
    size = size / 2;

    if (size != (uint64_t{1} << dest.num_qubits())) {
      IO::errorf("wrong size in VectorSpaceCuStateVecEx::Copy.\n");
      return false;
    }

    if (dest.distr_type() == kMultiProcess) {
      size /= dest.num_substates();
      uint64_t offset = size * mp.rank();

      ErrorCheck(custatevecExStateVectorSetState(
          dest.get(), src + 2 * offset, kStateDataType,
          offset, offset + size, 1));
    } else {
      ErrorCheck(custatevecExStateVectorSetState(
          dest.get(), src, kStateDataType, 0, size, 1));
    }

    ErrorCheck(custatevecExStateVectorSynchronize(dest.get()));

    // TODO: do we need that?
    dest.to_normal_order();

    return true;
  }

  static void DeviceSync() {
    ErrorCheck(cudaDeviceSynchronize());
  }

 protected:
  Parameter param;
  const MultiProcessCuStateVecEx& mp;

 private:
  static custatevecDeviceNetworkType_t get_device_network_type(
      DeviceNetworkType id) {
    custatevecDeviceNetworkType_t device_network_type =
        CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;

    switch (id) {
    case kSwitch:
      device_network_type = CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH;
      break;
    case kFullMesh:
      device_network_type = CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH;
      break;
    }

    return device_network_type;
  }
};

}  // namespace qsim

#endif  // VECTORSPACE_CUSTATEVECEX_H_
