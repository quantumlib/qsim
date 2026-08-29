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

#ifndef VECTORSPACE_H_
#define VECTORSPACE_H_

#ifdef _WIN32
  #include <malloc.h>
#elif defined(__linux__)
  #include <sys/syscall.h>
  #include <unistd.h>
  #include <fstream>
  #include <sstream>
  #include <string>
#endif

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

namespace qsim {

namespace detail {

inline void do_not_free(void*) {}

inline void free(void* ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

#if defined(__linux__) && defined(SYS_mbind)
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif

inline bool ParseNodeRange(const std::string& str, unsigned long* nodemask,
                           unsigned max_bits, unsigned& num_nodes_found) {
  std::stringstream ss(str);
  std::string item;
  num_nodes_found = 0;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) continue;
    size_t dash = item.find('-');
    if (dash == std::string::npos) {
      unsigned node = std::stoul(item);
      if (node < max_bits) {
        nodemask[node / (8 * sizeof(unsigned long))] |=
            (1UL << (node % (8 * sizeof(unsigned long))));
        num_nodes_found++;
      }
    } else {
      unsigned start = std::stoul(item.substr(0, dash));
      unsigned end = std::stoul(item.substr(dash + 1));
      for (unsigned node = start; node <= end && node < max_bits; ++node) {
        nodemask[node / (8 * sizeof(unsigned long))] |=
            (1UL << (node % (8 * sizeof(unsigned long))));
        num_nodes_found++;
      }
    }
  }
  return num_nodes_found > 0;
}

inline void ApplyNumaInterleave(void* ptr, std::size_t size) {
  if (ptr == nullptr || size < 2 * 1024 * 1024) return;

  unsigned long nodemask[16] = {0};
  constexpr unsigned max_bits = sizeof(nodemask) * 8;
  unsigned num_nodes = 0;

  for (const char* path : {"/sys/devices/system/node/has_cpu",
                           "/sys/devices/system/node/has_memory",
                           "/sys/devices/system/node/online"}) {
    std::ifstream file(path);
    if (file.is_open()) {
      std::string content;
      if (file >> content &&
          ParseNodeRange(content, nodemask, max_bits, num_nodes)) {
        break;
      }
    }
  }

  if (num_nodes >= 2) {
    syscall(SYS_mbind, ptr, size, MPOL_INTERLEAVE, nodemask, max_bits, 0);
  }
}
#endif

}  // namespace detail

// Routines for vector manipulations.
template <typename Impl, typename For, typename FP>
class VectorSpace {
 public:
  using fp_type = FP;

 private:
  using Pointer = std::unique_ptr<fp_type, decltype(&detail::free)>;

 public:
  class Vector {
   public:
    Vector() = delete;

    Vector(Pointer&& ptr, unsigned num_qubits)
        : ptr_(std::move(ptr)), num_qubits_(num_qubits) {}

    fp_type* get() {
      return ptr_.get();
    }

    const fp_type* get() const {
      return ptr_.get();
    }

    fp_type* release() {
      num_qubits_ = 0;
      return ptr_.release();
    }

    unsigned num_qubits() const {
      return num_qubits_;
    }

    static constexpr bool requires_copy_to_host() {
      return false;
    }

   private:
    Pointer ptr_;
    unsigned num_qubits_;
  };

  template <typename... ForArgs>
  VectorSpace(ForArgs&&... args) : for_(args...) {}

  static Vector Create(unsigned num_qubits) {
    auto size = sizeof(fp_type) * Impl::MinSize(num_qubits);
    #ifdef _WIN32
      Pointer ptr{(fp_type*) _aligned_malloc(size, 64), &detail::free};
      return Vector{std::move(ptr), ptr.get() != nullptr ? num_qubits : 0};
    #else
      void* p = nullptr;
      if (posix_memalign(&p, 64, size) == 0) {
        #if defined(__linux__) && defined(SYS_mbind)
        detail::ApplyNumaInterleave(p, size);
        #endif
        return Vector{Pointer{(fp_type*) p, &detail::free}, num_qubits};
      } else {
        return Null();
      }
    #endif
  }

  // It is the client's responsibility to make sure that p has at least
  // Impl::MinSize(num_qubits) elements.
  static Vector Create(fp_type* p, unsigned num_qubits) {
    return Vector{Pointer{p, &detail::do_not_free}, num_qubits};
  }

  static Vector Null() {
    return Vector{Pointer{nullptr, &detail::free}, 0};
  }

  static bool IsNull(const Vector& vec) {
    return vec.get() == nullptr;
  }

  static void Free(fp_type* ptr) {
    detail::free(ptr);
  }

  bool Copy(const Vector& src, Vector& dest) const {
    if (src.num_qubits() != dest.num_qubits()) {
      return false;
    }

    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* src, fp_type* dest) {
      dest[i] = src[i];
    };

    for_.Run(Impl::MinSize(src.num_qubits()), f, src.get(), dest.get());

    return true;
  }

  // It is the client's responsibility to make sure that dest has at least
  // Impl::MinSize(src.num_qubits()) elements.
  bool Copy(const Vector& src, fp_type* dest) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* src, fp_type* dest) {
      dest[i] = src[i];
    };

    for_.Run(Impl::MinSize(src.num_qubits()), f, src.get(), dest);

    return true;
  }

  // It is the client's responsibility to make sure that src has at least
  // Impl::MinSize(dest.num_qubits()) elements.
  bool Copy(const fp_type* src, Vector& dest) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* src, fp_type* dest) {
      dest[i] = src[i];
    };

    for_.Run(Impl::MinSize(dest.num_qubits()), f, src, dest.get());

    return true;
  }

  // It is the client's responsibility to make sure that src has at least
  // min(size, Impl::MinSize(dest.num_qubits())) elements.
  bool Copy(const fp_type* src, uint64_t size, Vector& dest) const {
    auto f = [](unsigned n, unsigned m, uint64_t i,
                const fp_type* src, fp_type* dest) {
      dest[i] = src[i];
    };

    size = std::min(size, Impl::MinSize(dest.num_qubits()));
    for_.Run(size, f, src, dest.get());

    return true;
  }

  static void DeviceSync() {}

 protected:
  For for_;
};

}  // namespace qsim

#endif  // VECTORSPACE_H_
