#ifndef QSIM_TILE_PARTITION_H_
#define QSIM_TILE_PARTITION_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "state_tile.h"

namespace qsim {

class TilePartition {
 public:
  TilePartition() = default;
  TilePartition(unsigned qubits, unsigned tile_qubits, unsigned local_qubits = 0) {
    Reset(qubits, tile_qubits, local_qubits);
  }
  void Reset(unsigned qubits, unsigned tile_qubits, unsigned local_qubits = 0) {
    qubits_ = qubits; tile_qubits_ = std::min(tile_qubits, qubits);
    local_qubits_ = local_qubits;
    num_tiles_ = uint64_t{1} << (qubits_ - tile_qubits_);
  }
  unsigned tile_qubits() const { return tile_qubits_; }
  uint64_t num_tiles() const { return num_tiles_; }
  uint64_t amplitudes_per_tile() const { return uint64_t{1} << tile_qubits_; }
  uint64_t BytesPerTile(uint64_t bytes_per_amplitude) const {
    return amplitudes_per_tile() * bytes_per_amplitude;
  }
  std::vector<StateTile> Tiles(unsigned numa_nodes = 1) const {
    std::vector<StateTile> result; result.reserve(num_tiles_);
    for (uint64_t i = 0; i < num_tiles_; ++i)
      result.push_back({i, i * amplitudes_per_tile(), amplitudes_per_tile(),
                        static_cast<unsigned>(i % std::max(1U, numa_nodes)), 0});
    return result;
  }
 private:
  unsigned qubits_ = 0, tile_qubits_ = 0, local_qubits_ = 0;
  uint64_t num_tiles_ = 1;
};
}
#endif
