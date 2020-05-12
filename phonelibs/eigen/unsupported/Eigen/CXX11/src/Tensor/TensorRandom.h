// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_RANDOM_H
#define EIGEN_CXX11_TENSOR_TENSOR_RANDOM_H

namespace Eigen {
namespace internal {

namespace {

EIGEN_DEVICE_FUNC uint64_t get_random_seed() {
#ifdef __CUDA_ARCH__
  // We don't support 3d kernels since we currently only use 1 and
  // 2d kernels.
  assert(threadIdx.z == 0);
  return clock64() +
      blockIdx.x * blockDim.x + threadIdx.x +
      gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + threadIdx.y);

#elif defined _WIN32
  // Use the current time as a baseline.
  SYSTEMTIME st;
  GetSystemTime(&st);
  int time = st.wSecond + 1000 * st.wMilliseconds;
  // Mix in a random number to make sure that we get different seeds if
  // we try to generate seeds faster than the clock resolution.
  // We need 2 random values since the generator only generate 16 bits at
  // a time (https://msdn.microsoft.com/en-us/library/398ax69y.aspx)
  int rnd1 = ::rand();
  int rnd2 = ::rand();
  uint64_t rnd = (rnd1 | rnd2 << 16) ^ time;
  return rnd;

#elif defined __APPLE__
  // Same approach as for win32, except that the random number generator
  // is better (// https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man3/random.3.html#//apple_ref/doc/man/3/random).
  uint64_t rnd = ::random() ^ mach_absolute_time();
  return rnd;

#else
  // Augment the current time with pseudo random number generation
  // to ensure that we get different seeds if we try to generate seeds
  // faster than the clock resolution.
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t rnd = ::random() ^ ts.tv_nsec;
  return rnd;
#endif
}

static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE unsigned PCG_XSH_RS_generator(uint64_t* state) {
  // TODO: Unify with the implementation in the non blocking thread pool.
  uint64_t current = *state;
  // Update the internal state
  *state = current * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
  // Generate the random output (using the PCG-XSH-RS scheme)
  return static_cast<unsigned>((current ^ (current >> 22)) >> (22 + (current >> 61)));
}

static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE uint64_t PCG_XSH_RS_state(uint64_t seed) {
  seed = seed ? seed : get_random_seed();
  return seed * 6364136223846793005ULL + 0xda3e39cb94b95bdbULL;
}

}  // namespace


template <typename T> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
T RandomToTypeUniform(uint64_t* state) {
  unsigned rnd = PCG_XSH_RS_generator(state);
  return static_cast<T>(rnd);
}


template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Eigen::half RandomToTypeUniform<Eigen::half>(uint64_t* state) {
  Eigen::half result;
  // Generate 10 random bits for the mantissa
  unsigned rnd = PCG_XSH_RS_generator(state);
  result.x = static_cast<uint16_t>(rnd & 0x3ffu);
  // Set the exponent
  result.x |= (static_cast<uint16_t>(15) << 10);
  // Return the final result
  return result - Eigen::half(1.0f);
}


template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
float RandomToTypeUniform<float>(uint64_t* state) {
  typedef union {
    uint32_t raw;
    float fp;
  } internal;
  internal result;
  // Generate 23 random bits for the mantissa mantissa
  const unsigned rnd = PCG_XSH_RS_generator(state);
  result.raw = rnd & 0x7fffffu;
  // Set the exponent
  result.raw |= (static_cast<uint32_t>(127) << 23);
  // Return the final result
  return result.fp - 1.0f;
}

template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
double RandomToTypeUniform<double>(uint64_t* state) {
  typedef union {
    uint64_t raw;
    double dp;
  } internal;
  internal result;
  result.raw = 0;
  // Generate 52 random bits for the mantissa
  // First generate the upper 20 bits
  unsigned rnd1 = PCG_XSH_RS_generator(state) & 0xfffffu;
  // The generate the lower 32 bits
  unsigned rnd2 = PCG_XSH_RS_generator(state);
  result.raw = (static_cast<uint64_t>(rnd1) << 32) | rnd2;
  // Set the exponent
  result.raw |= (static_cast<uint64_t>(1023) << 52);
  // Return the final result
  return result.dp - 1.0;
}

template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
std::complex<float> RandomToTypeUniform<std::complex<float> >(uint64_t* state) {
  return std::complex<float>(RandomToTypeUniform<float>(state),
                             RandomToTypeUniform<float>(state));
}
template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
std::complex<double> RandomToTypeUniform<std::complex<double> >(uint64_t* state) {
  return std::complex<double>(RandomToTypeUniform<double>(state),
                              RandomToTypeUniform<double>(state));
}

template <typename T> class UniformRandomGenerator {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE UniformRandomGenerator(
      uint64_t seed = 0) {
    m_state = PCG_XSH_RS_state(seed);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE UniformRandomGenerator(
      const UniformRandomGenerator& other) {
    m_state = other.m_state;
  }

  template<typename Index> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  T operator()(Index i) const {
    uint64_t local_state = m_state + i;
    T result = RandomToTypeUniform<T>(&local_state);
    m_state = local_state;
    return result;
  }

  template<typename Packet, typename Index> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Packet packetOp(Index i) const {
    const int packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_ALIGN_MAX T values[packetSize];
    uint64_t local_state = m_state + i;
    for (int j = 0; j < packetSize; ++j) {
      values[j] = RandomToTypeUniform<T>(&local_state);
    }
    m_state = local_state;
    return internal::pload<Packet>(values);
  }

 private:
  mutable uint64_t m_state;
};

template <typename Scalar>
struct functor_traits<UniformRandomGenerator<Scalar> > {
  enum {
    // Rough estimate for floating point, multiplied by ceil(sizeof(T) / sizeof(float)).
    Cost = 12 * NumTraits<Scalar>::AddCost *
           ((sizeof(Scalar) + sizeof(float) - 1) / sizeof(float)),
    PacketAccess = UniformRandomGenerator<Scalar>::PacketAccess
  };
};



template <typename T> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
T RandomToTypeNormal(uint64_t* state) {
  // Use the ratio of uniform method to generate numbers following a normal
  // distribution. See for example Numerical Recipes chapter 7.3.9 for the
  // details.
  T u, v, q;
  do {
    u = RandomToTypeUniform<T>(state);
    v = T(1.7156) * (RandomToTypeUniform<T>(state) - T(0.5));
    const T x = u - T(0.449871);
    const T y = numext::abs(v) + T(0.386595);
    q = x*x + y * (T(0.196)*y - T(0.25472)*x);
  } while (q > T(0.27597) &&
           (q > T(0.27846) || v*v > T(-4) * numext::log(u) * u*u));

  return v/u;
}

template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
std::complex<float> RandomToTypeNormal<std::complex<float> >(uint64_t* state) {
  return std::complex<float>(RandomToTypeNormal<float>(state),
                             RandomToTypeNormal<float>(state));
}
template <> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
std::complex<double> RandomToTypeNormal<std::complex<double> >(uint64_t* state) {
  return std::complex<double>(RandomToTypeNormal<double>(state),
                              RandomToTypeNormal<double>(state));
}


template <typename T> class NormalRandomGenerator {
 public:
  static const bool PacketAccess = true;

  // Uses the given "seed" if non-zero, otherwise uses a random seed.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE NormalRandomGenerator(uint64_t seed = 0) {
    m_state = PCG_XSH_RS_state(seed);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE NormalRandomGenerator(
      const NormalRandomGenerator& other) {
    m_state = other.m_state;
  }

 template<typename Index> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  T operator()(Index i) const {
    uint64_t local_state = m_state + i;
    T result = RandomToTypeNormal<T>(&local_state);
    m_state = local_state;
    return result;
  }

  template<typename Packet, typename Index> EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Packet packetOp(Index i) const {
    const int packetSize = internal::unpacket_traits<Packet>::size;
    EIGEN_ALIGN_MAX T values[packetSize];
    uint64_t local_state = m_state + i;
    for (int j = 0; j < packetSize; ++j) {
      values[j] = RandomToTypeNormal<T>(&local_state);
    }
    m_state = local_state;
    return internal::pload<Packet>(values);
  }

 private:
  mutable uint64_t m_state;
};


template <typename Scalar>
struct functor_traits<NormalRandomGenerator<Scalar> > {
  enum {
    // On average, we need to generate about 3 random numbers
    // 15 mul, 8 add, 1.5 logs
    Cost = 3 * functor_traits<UniformRandomGenerator<Scalar> >::Cost +
           15 * NumTraits<Scalar>::AddCost + 8 * NumTraits<Scalar>::AddCost +
           3 * functor_traits<scalar_log_op<Scalar> >::Cost / 2,
    PacketAccess = NormalRandomGenerator<Scalar>::PacketAccess
  };
};


} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_RANDOM_H
