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

  // Sample using a random seed.
  void SampleState(const StateSpace statespace, const State& state,
                   const int m, std::vector<uint64_t>* samples) {
    std::random_device device;
    SampleState(statespace, state, m, device(), samples);
  }

  // Function to draw m samples from a StateSpace Object in
  // O(2 ** num_qubits + m * log(m)) time.
  // Samples are stored as bit encoded integers.
  //
  // This method takes a seed for the RNG. To choose a random seed,
  // Use the other version of this method.
  void SampleState(const StateSpace statespace, const State& state,
                   const int m, unsigned seed,
                   std::vector<uint64_t>* samples) {
    if (m == 0) {
      return;
    }
    std::mt19937 mt(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    double cdf_so_far = 0.0;
    std::vector<float> random_vals(m, 0.0);
    samples->reserve(m);
    for (int i = 0; i < m; i++) {
      random_vals[i] = dist(mt);
    }
    std::sort(random_vals.begin(), random_vals.end());

    int j = 0;
    for (uint64_t i = 0; i < statespace.Size(state); i++) {
      const std::complex<float> f_amp = statespace.GetAmpl(state, i);
      const std::complex<double> d_amp = std::complex<double>(
        static_cast<double>(f_amp.real()),
        static_cast<double>(f_amp.imag()));
      cdf_so_far += std::norm(d_amp);
      while (random_vals[j] < cdf_so_far && j < m) {
        samples->push_back(i);
        j++;
      }
    }

    // Safety measure in case of state norm underflow.
    // Likely to not have huge impact.
    while (j < m) {
      samples->push_back(samples->at(samples->size() - 1));
      j++;
    }
  }
};

}  // namespace qsim

#endif  // STATE_SAMPLER_H_
