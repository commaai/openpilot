// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_NEON_H
#define EIGEN_COMPLEX_NEON_H

namespace Eigen {

namespace internal {

static uint32x4_t p4ui_CONJ_XOR = EIGEN_INIT_NEON_PACKET4(0x00000000, 0x80000000, 0x00000000, 0x80000000);
static uint32x2_t p2ui_CONJ_XOR = EIGEN_INIT_NEON_PACKET2(0x00000000, 0x80000000);

//---------- float ----------
struct Packet2cf
{
  EIGEN_STRONG_INLINE Packet2cf() {}
  EIGEN_STRONG_INLINE explicit Packet2cf(const Packet4f& a) : v(a) {}
  Packet4f  v;
};

template<> struct packet_traits<std::complex<float> >  : default_packet_traits
{
  typedef Packet2cf type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet2cf> { typedef std::complex<float> type; enum {size=2}; };

template<> EIGEN_STRONG_INLINE Packet2cf pset1<Packet2cf>(const std::complex<float>&  from)
{
  float32x2_t r64;
  r64 = vld1_f32((float *)&from);

  return Packet2cf(vcombine_f32(r64, r64));
}

template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(padd<Packet4f>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(psub<Packet4f>(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a) { return Packet2cf(pnegate<Packet4f>(a.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a)
{
  Packet4ui b = vreinterpretq_u32_f32(a.v);
  return Packet2cf(vreinterpretq_f32_u32(veorq_u32(b, p4ui_CONJ_XOR)));
}

template<> EIGEN_STRONG_INLINE Packet2cf pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet4f v1, v2;

  // Get the real values of a | a1_re | a1_re | a2_re | a2_re |
  v1 = vcombine_f32(vdup_lane_f32(vget_low_f32(a.v), 0), vdup_lane_f32(vget_high_f32(a.v), 0));
  // Get the real values of a | a1_im | a1_im | a2_im | a2_im |
  v2 = vcombine_f32(vdup_lane_f32(vget_low_f32(a.v), 1), vdup_lane_f32(vget_high_f32(a.v), 1));
  // Multiply the real a with b
  v1 = vmulq_f32(v1, b.v);
  // Multiply the imag a with b
  v2 = vmulq_f32(v2, b.v);
  // Conjugate v2 
  v2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v2), p4ui_CONJ_XOR));
  // Swap real/imag elements in v2.
  v2 = vrev64q_f32(v2);
  // Add and return the result
  return Packet2cf(vaddq_f32(v1, v2));
}

template<> EIGEN_STRONG_INLINE Packet2cf pand   <Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet2cf por    <Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet2cf pxor   <Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}
template<> EIGEN_STRONG_INLINE Packet2cf pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return Packet2cf(vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(a.v),vreinterpretq_u32_f32(b.v))));
}

template<> EIGEN_STRONG_INLINE Packet2cf pload<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet2cf(pload<Packet4f>((const float*)from)); }
template<> EIGEN_STRONG_INLINE Packet2cf ploadu<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cf(ploadu<Packet4f>((const float*)from)); }

template<> EIGEN_STRONG_INLINE Packet2cf ploaddup<Packet2cf>(const std::complex<float>* from) { return pset1<Packet2cf>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((float*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((float*)to, from.v); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float> *   addr) { __pld((float *)addr); }

template<> EIGEN_STRONG_INLINE std::complex<float>  pfirst<Packet2cf>(const Packet2cf& a)
{
  std::complex<float> EIGEN_ALIGN16 x[2];
  vst1q_f32((float *)x, a.v);
  return x[0];
}

template<> EIGEN_STRONG_INLINE Packet2cf preverse(const Packet2cf& a)
{
  float32x2_t a_lo, a_hi;
  Packet4f a_r128;

  a_lo = vget_low_f32(a.v);
  a_hi = vget_high_f32(a.v);
  a_r128 = vcombine_f32(a_hi, a_lo);

  return Packet2cf(a_r128);
}

template<> EIGEN_STRONG_INLINE Packet2cf pcplxflip<Packet2cf>(const Packet2cf& a)
{
  return Packet2cf(vrev64q_f32(a.v));
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet2cf>(const Packet2cf& a)
{
  float32x2_t a1, a2;
  std::complex<float> s;

  a1 = vget_low_f32(a.v);
  a2 = vget_high_f32(a.v);
  a2 = vadd_f32(a1, a2);
  vst1_f32((float *)&s, a2);

  return s;
}

template<> EIGEN_STRONG_INLINE Packet2cf preduxp<Packet2cf>(const Packet2cf* vecs)
{
  Packet4f sum1, sum2, sum;

  // Add the first two 64-bit float32x2_t of vecs[0]
  sum1 = vcombine_f32(vget_low_f32(vecs[0].v), vget_low_f32(vecs[1].v));
  sum2 = vcombine_f32(vget_high_f32(vecs[0].v), vget_high_f32(vecs[1].v));
  sum = vaddq_f32(sum1, sum2);

  return Packet2cf(sum);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet2cf>(const Packet2cf& a)
{
  float32x2_t a1, a2, v1, v2, prod;
  std::complex<float> s;

  a1 = vget_low_f32(a.v);
  a2 = vget_high_f32(a.v);
   // Get the real values of a | a1_re | a1_re | a2_re | a2_re |
  v1 = vdup_lane_f32(a1, 0);
  // Get the real values of a | a1_im | a1_im | a2_im | a2_im |
  v2 = vdup_lane_f32(a1, 1);
  // Multiply the real a with b
  v1 = vmul_f32(v1, a2);
  // Multiply the imag a with b
  v2 = vmul_f32(v2, a2);
  // Conjugate v2 
  v2 = vreinterpret_f32_u32(veor_u32(vreinterpret_u32_f32(v2), p2ui_CONJ_XOR));
  // Swap real/imag elements in v2.
  v2 = vrev64_f32(v2);
  // Add v1, v2
  prod = vadd_f32(v1, v2);

  vst1_f32((float *)&s, prod);

  return s;
}

template<int Offset>
struct palign_impl<Offset,Packet2cf>
{
  EIGEN_STRONG_INLINE static void run(Packet2cf& first, const Packet2cf& second)
  {
    if (Offset==1)
    {
      first.v = vextq_f32(first.v, second.v, 2);
    }
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, false,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return internal::pmul(a, pconj(b));
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,false>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return internal::pmul(pconj(a), b);
  }
};

template<> struct conj_helper<Packet2cf, Packet2cf, true,true>
{
  EIGEN_STRONG_INLINE Packet2cf pmadd(const Packet2cf& x, const Packet2cf& y, const Packet2cf& c) const
  { return padd(pmul(x,y),c); }

  EIGEN_STRONG_INLINE Packet2cf pmul(const Packet2cf& a, const Packet2cf& b) const
  {
    return pconj(internal::pmul(a, b));
  }
};

template<> EIGEN_STRONG_INLINE Packet2cf pdiv<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  // TODO optimize it for AltiVec
  Packet2cf res = conj_helper<Packet2cf,Packet2cf,false,true>().pmul(a,b);
  Packet4f s, rev_s;

  // this computes the norm
  s = vmulq_f32(b.v, b.v);
  rev_s = vrev64q_f32(s);

  return Packet2cf(pdiv(res.v, vaddq_f32(s,rev_s)));
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_NEON_H
