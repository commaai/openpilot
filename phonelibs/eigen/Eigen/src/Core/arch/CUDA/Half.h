// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// The conversion routines are Copyright (c) Fabian Giesen, 2016.
// The original license follows:
//
// Copyright (c) Fabian Giesen, 2016
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


// Standard 16-bit float type, mostly useful for GPUs. Defines a new
// type Eigen::half (inheriting from CUDA's __half struct) with
// operator overloads such that it behaves basically as an arithmetic
// type. It will be quite slow on CPUs (so it is recommended to stay
// in fp32 for CPUs, except for simple parameter conversions, I/O
// to disk and the likes), but fast on GPUs.


#ifndef EIGEN_HALF_CUDA_H
#define EIGEN_HALF_CUDA_H

#if __cplusplus > 199711L
#define EIGEN_EXPLICIT_CAST(tgt_type) explicit operator tgt_type()
#else
#define EIGEN_EXPLICIT_CAST(tgt_type) operator tgt_type()
#endif


namespace Eigen {

struct half;

namespace half_impl {

#if !defined(EIGEN_HAS_CUDA_FP16)

// Make our own __half definition that is similar to CUDA's.
struct __half {
  EIGEN_DEVICE_FUNC __half() {}
  explicit EIGEN_DEVICE_FUNC __half(unsigned short raw) : x(raw) {}
  unsigned short x;
};

#endif

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x);
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half float_to_half_rtne(float ff);
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC float half_to_float(__half h);

struct half_base : public __half {
  EIGEN_DEVICE_FUNC half_base() {}
  EIGEN_DEVICE_FUNC half_base(const half_base& h) : __half(h) {}
  EIGEN_DEVICE_FUNC half_base(const __half& h) : __half(h) {}
};

} // namespace half_impl

// Class definition.
struct half : public half_impl::half_base {
  #if !defined(EIGEN_HAS_CUDA_FP16)
    typedef half_impl::__half __half;
  #endif

  EIGEN_DEVICE_FUNC half() {}

  EIGEN_DEVICE_FUNC half(const __half& h) : half_impl::half_base(h) {}
  EIGEN_DEVICE_FUNC half(const half& h) : half_impl::half_base(h) {}

  explicit EIGEN_DEVICE_FUNC half(bool b)
      : half_impl::half_base(half_impl::raw_uint16_to_half(b ? 0x3c00 : 0)) {}
  template<class T>
  explicit EIGEN_DEVICE_FUNC half(const T& val)
      : half_impl::half_base(half_impl::float_to_half_rtne(static_cast<float>(val))) {}
  explicit EIGEN_DEVICE_FUNC half(float f)
      : half_impl::half_base(half_impl::float_to_half_rtne(f)) {}

  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(bool) const {
    // +0.0 and -0.0 become false, everything else becomes true.
    return (x & 0x7fff) != 0;
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(signed char) const {
    return static_cast<signed char>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned char) const {
    return static_cast<unsigned char>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(short) const {
    return static_cast<short>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned short) const {
    return static_cast<unsigned short>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(int) const {
    return static_cast<int>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned int) const {
    return static_cast<unsigned int>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(long) const {
    return static_cast<long>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned long) const {
    return static_cast<unsigned long>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(long long) const {
    return static_cast<long long>(half_impl::half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(unsigned long long) const {
    return static_cast<unsigned long long>(half_to_float(*this));
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(float) const {
    return half_impl::half_to_float(*this);
  }
  EIGEN_DEVICE_FUNC EIGEN_EXPLICIT_CAST(double) const {
    return static_cast<double>(half_impl::half_to_float(*this));
  }

  EIGEN_DEVICE_FUNC half& operator=(const half& other) {
    x = other.x;
    return *this;
  }
};

namespace half_impl {

#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530

// Intrinsics for native fp16 support. Note that on current hardware,
// these are no faster than fp32 arithmetic (you need to use the half2
// versions to get the ALU speed increased), but you do save the
// conversion steps back and forth.

__device__ half operator + (const half& a, const half& b) {
  return __hadd(a, b);
}
__device__ half operator * (const half& a, const half& b) {
  return __hmul(a, b);
}
__device__ half operator - (const half& a, const half& b) {
  return __hsub(a, b);
}
__device__ half operator / (const half& a, const half& b) {
  float num = __half2float(a);
  float denom = __half2float(b);
  return __float2half(num / denom);
}
__device__ half operator - (const half& a) {
  return __hneg(a);
}
__device__ half& operator += (half& a, const half& b) {
  a = a + b;
  return a;
}
__device__ half& operator *= (half& a, const half& b) {
  a = a * b;
  return a;
}
__device__ half& operator -= (half& a, const half& b) {
  a = a - b;
  return a;
}
__device__ half& operator /= (half& a, const half& b) {
  a = a / b;
  return a;
}
__device__ bool operator == (const half& a, const half& b) {
  return __heq(a, b);
}
__device__ bool operator != (const half& a, const half& b) {
  return __hne(a, b);
}
__device__ bool operator < (const half& a, const half& b) {
  return __hlt(a, b);
}
__device__ bool operator <= (const half& a, const half& b) {
  return __hle(a, b);
}
__device__ bool operator > (const half& a, const half& b) {
  return __hgt(a, b);
}
__device__ bool operator >= (const half& a, const half& b) {
  return __hge(a, b);
}

#else  // Emulate support for half floats

// Definitions for CPUs and older CUDA, mostly working through conversion
// to/from fp32.

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator + (const half& a, const half& b) {
  return half(float(a) + float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator * (const half& a, const half& b) {
  return half(float(a) * float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator - (const half& a, const half& b) {
  return half(float(a) - float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator / (const half& a, const half& b) {
  return half(float(a) / float(b));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator - (const half& a) {
  half result;
  result.x = a.x ^ 0x8000;
  return result;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator += (half& a, const half& b) {
  a = half(float(a) + float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator *= (half& a, const half& b) {
  a = half(float(a) * float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator -= (half& a, const half& b) {
  a = half(float(a) - float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half& operator /= (half& a, const half& b) {
  a = half(float(a) / float(b));
  return a;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator == (const half& a, const half& b) {
  return float(a) == float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator != (const half& a, const half& b) {
  return float(a) != float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator < (const half& a, const half& b) {
  return float(a) < float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator <= (const half& a, const half& b) {
  return float(a) <= float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator > (const half& a, const half& b) {
  return float(a) > float(b);
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool operator >= (const half& a, const half& b) {
  return float(a) >= float(b);
}

#endif  // Emulate support for half floats

// Division by an index. Do it in full float precision to avoid accuracy
// issues in converting the denominator to half.
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half operator / (const half& a, Index b) {
  return half(static_cast<float>(a) / static_cast<float>(b));
}

// Conversion routines, including fallbacks for the host or older CUDA.
// Note that newer Intel CPUs (Haswell or newer) have vectorized versions of
// these in hardware. If we need more performance on older/other CPUs, they are
// also possible to vectorize directly.

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half raw_uint16_to_half(unsigned short x) {
  __half h;
  h.x = x;
  return h;
}

union FP32 {
  unsigned int u;
  float f;
};

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC __half float_to_half_rtne(float ff) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  return __float2half(ff);

#elif defined(EIGEN_HAS_FP16_C)
  __half h;
  h.x = _cvtss_sh(ff, 0);
  return h;

#else
  FP32 f; f.f = ff;

  const FP32 f32infty = { 255 << 23 };
  const FP32 f16max = { (127 + 16) << 23 };
  const FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
  unsigned int sign_mask = 0x80000000u;
  __half o;
  o.x = static_cast<unsigned short>(0x0u);

  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f16max.u) {  // result is Inf or NaN (all exponent bits set)
    o.x = (f.u > f32infty.u) ? 0x7e00 : 0x7c00; // NaN->qNaN and Inf->Inf
  } else {  // (De)normalized number or zero
    if (f.u < (113 << 23)) {  // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;

      // and one integer subtract of the bias later, we have our final float!
      o.x = static_cast<unsigned short>(f.u - denorm_magic.u);
    } else {
      unsigned int mant_odd = (f.u >> 13) & 1; // resulting mantissa is odd

      // update exponent, rounding bias part 1
      f.u += ((unsigned int)(15 - 127) << 23) + 0xfff;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      o.x = static_cast<unsigned short>(f.u >> 13);
    }
  }

  o.x |= static_cast<unsigned short>(sign >> 16);
  return o;
#endif
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC float half_to_float(__half h) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
  return __half2float(h);

#elif defined(EIGEN_HAS_FP16_C)
  return _cvtsh_ss(h.x);

#else
  const FP32 magic = { 113 << 23 };
  const unsigned int shifted_exp = 0x7c00 << 13; // exponent mask after shift
  FP32 o;

  o.u = (h.x & 0x7fff) << 13;             // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.u;   // just the exponent
  o.u += (127 - 15) << 23;                // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {     // Inf/NaN?
    o.u += (128 - 16) << 23;    // extra exp adjust
  } else if (exp == 0) {        // Zero/Denormal?
    o.u += 1 << 23;             // extra exp adjust
    o.f -= magic.f;             // renormalize
  }

  o.u |= (h.x & 0x8000) << 16;    // sign bit
  return o.f;
#endif
}

// --- standard functions ---

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isinf)(const half& a) {
  return (a.x & 0x7fff) == 0x7c00;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isnan)(const half& a) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hisnan(a);
#else
  return (a.x & 0x7fff) > 0x7c00;
#endif
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC bool (isfinite)(const half& a) {
  return !(isinf EIGEN_NOT_A_MACRO (a)) && !(isnan EIGEN_NOT_A_MACRO (a));
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half abs(const half& a) {
  half result;
  result.x = a.x & 0x7FFF;
  return result;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half exp(const half& a) {
  return half(::expf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half log(const half& a) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return Eigen::half(::hlog(a));
#else
  return half(::logf(float(a)));
#endif
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half log1p(const half& a) {
  return half(numext::log1p(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half log10(const half& a) {
  return half(::log10f(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half sqrt(const half& a) {
  return half(::sqrtf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half pow(const half& a, const half& b) {
  return half(::powf(float(a), float(b)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half sin(const half& a) {
  return half(::sinf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half cos(const half& a) {
  return half(::cosf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half tan(const half& a) {
  return half(::tanf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half tanh(const half& a) {
  return half(::tanhf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half floor(const half& a) {
  return half(::floorf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half ceil(const half& a) {
  return half(::ceilf(float(a)));
}

EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half (min)(const half& a, const half& b) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(b, a) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f2 < f1 ? b : a;
#endif
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC half (max)(const half& a, const half& b) {
#if defined(EIGEN_HAS_CUDA_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return __hlt(a, b) ? b : a;
#else
  const float f1 = static_cast<float>(a);
  const float f2 = static_cast<float>(b);
  return f1 < f2 ? b : a;
#endif
}

EIGEN_ALWAYS_INLINE std::ostream& operator << (std::ostream& os, const half& v) {
  os << static_cast<float>(v);
  return os;
}

} // end namespace half_impl

// import Eigen::half_impl::half into Eigen namespace
// using half_impl::half;

namespace internal {

template<>
struct random_default_impl<half, false, false>
{
  static inline half run(const half& x, const half& y)
  {
    return x + (y-x) * half(float(std::rand()) / float(RAND_MAX));
  }
  static inline half run()
  {
    return run(half(-1.f), half(1.f));
  }
};

template<> struct is_arithmetic<half> { enum { value = true }; };

} // end namespace internal

template<> struct NumTraits<Eigen::half>
    : GenericNumTraits<Eigen::half>
{
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half epsilon() {
    return half_impl::raw_uint16_to_half(0x0800);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half dummy_precision() { return Eigen::half(1e-2f); }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half highest() {
    return half_impl::raw_uint16_to_half(0x7bff);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half lowest() {
    return half_impl::raw_uint16_to_half(0xfbff);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half infinity() {
    return half_impl::raw_uint16_to_half(0x7c00);
  }
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE Eigen::half quiet_NaN() {
    return half_impl::raw_uint16_to_half(0x7c01);
  }
};

} // end namespace Eigen

// C-like standard mathematical functions and trancendentals.
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half fabsh(const Eigen::half& a) {
  Eigen::half result;
  result.x = a.x & 0x7FFF;
  return result;
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half exph(const Eigen::half& a) {
  return Eigen::half(::expf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half logh(const Eigen::half& a) {
#if defined __CUDACC_VER__ && __CUDACC_VER__ >= 80000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530
  return Eigen::half(::hlog(a));
#else
  return Eigen::half(::logf(float(a)));
#endif
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half sqrth(const Eigen::half& a) {
  return Eigen::half(::sqrtf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half powh(const Eigen::half& a, const Eigen::half& b) {
  return Eigen::half(::powf(float(a), float(b)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half floorh(const Eigen::half& a) {
  return Eigen::half(::floorf(float(a)));
}
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half ceilh(const Eigen::half& a) {
  return Eigen::half(::ceilf(float(a)));
}

namespace std {

#if __cplusplus > 199711L
template <>
struct hash<Eigen::half> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE std::size_t operator()(const Eigen::half& a) const {
    return static_cast<std::size_t>(a.x);
  }
};
#endif

} // end namespace std


// Add the missing shfl_xor intrinsic
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
__device__ EIGEN_STRONG_INLINE Eigen::half __shfl_xor(Eigen::half var, int laneMask, int width=warpSize) {
  return static_cast<Eigen::half>(__shfl_xor(static_cast<float>(var), laneMask, width));
}
#endif

// ldg() has an overload for __half, but we also need one for Eigen::half.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC Eigen::half __ldg(const Eigen::half* ptr) {
  return Eigen::half_impl::raw_uint16_to_half(
      __ldg(reinterpret_cast<const unsigned short*>(ptr)));
}
#endif


#if defined(__CUDA_ARCH__)
namespace Eigen {
namespace numext {

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
bool (isnan)(const Eigen::half& h) {
  return (half_impl::isnan)(h);
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
bool (isinf)(const Eigen::half& h) {
  return (half_impl::isinf)(h);
}

template<>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
bool (isfinite)(const Eigen::half& h) {
  return (half_impl::isfinite)(h);
}

} // namespace Eigen
}  // namespace numext
#endif

#endif // EIGEN_HALF_CUDA_H
