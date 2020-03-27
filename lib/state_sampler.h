#ifndef STATE_SAMPLER_H_
#define STATE_SAMPLER_H_

#include <complex>
#include <memory>
#include <random>

namespace qsim {

template <typename StateSpace>
class StateSampler {
 public:
  using State = typename StateSpace::State;

  void SampleState(const StateSpace& state_space, const State& state,
                   const unsigned num_samples, unsigned seed,
                   std::vector<uint64_t>* samples) {
    if (num_samples <= 0) {
      return;
    }
    samples->reserve(num_samples);

    double norm = 0;
    for (uint64_t k = 0; k < state_space.Size(state); ++k) {
      norm += std::norm(state_space.GetAmpl(state, k));
    }

    std::mt19937_64 rgen(seed);
    double rgen_norm = norm / std::mt19937_64::max();

    std::vector<double> rs;
    rs.reserve(num_samples);

    for (unsigned i = 0; i < num_samples; ++i) {
      rs.emplace_back(rgen() * rgen_norm);
    }

    std::sort(rs.begin(), rs.end());

    unsigned m = 0;
    double csum = 0;

    for (uint64_t k = 0; k < state_space.Size(state); ++k) {
      csum += std::norm(state_space.GetAmpl(state, k));

      while (m < num_samples && rs[m] <= csum) {
        samples->emplace_back(k);
        ++m;
      }
    }
  }
};

}  // namespace qsim

#endif  // STATE_SAMPLER_H_
