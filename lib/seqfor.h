#ifndef SEQFOR_H_
#define SEQFOR_H_

#include <cstdint>

namespace qsim {

struct SequentialFor {
  template <typename Function, typename... Args>
  static void Run(
    unsigned num_threads, uint64_t size, Function&& func, Args&&... args) {
    for (uint64_t i = 0; i < size; ++i) {
      func(1, 0, i, args...);
    }
  }

  template <typename Function, typename Op, typename... Args>
  static typename Op::result_type RunReduce(unsigned num_threads,
                                            uint64_t size, Function&& func,
                                            Op&& op, Args&&... args) {
    typename Op::result_type result = 0;

    for (uint64_t i = 0; i < size; ++i) {
      result = op(result, func(1, 0, i, args...));
    }

    return result;
  }
};

}  // namespace qsim

#endif  // SEQFOR_H_
