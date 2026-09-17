#ifndef QSIM_STATE_TILE_H_
#define QSIM_STATE_TILE_H_

#include <cstdint>

namespace qsim {

struct StateTile {
  uint64_t id = 0;
  uint64_t first_amplitude = 0;
  uint64_t amplitudes = 0;
  unsigned numa_node = 0;
  unsigned cache_group = 0;
};

}  // namespace qsim
#endif
