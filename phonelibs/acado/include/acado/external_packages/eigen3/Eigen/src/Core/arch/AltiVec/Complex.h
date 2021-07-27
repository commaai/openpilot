// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_ALTIVEC_H
#define EIGEN_COMPLEX_ALTIVEC_H

namespace Eigen {

namespace internal {

static Packet4ui  p4ui_CONJ_XOR = vec_mergeh((Packet4ui)p4i_ZERO, (Packet4ui)p4f_ZERO_);//{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 };
static Packet16uc p16uc_COMPLEX_RE   = vec_sld((Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 0), (Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 2), 8);//{ 0,1,2,3, 0,1,2,3, 8,9,10,11, 8,9,10,11 };
static Packet16uc p16uc_COMPLEX_IM   = vec_sld((Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 1), (Packet16uc) vec_splat((Packet4ui)p16uc_FORWARD, 3), 8);//{ 4,5,6,7, 4,5,6,7, 12,13,14,15, 12,13,14,15 };
static Packet16uc p16uc_COMPLEX_REV  = vec_sld(p16uc_REVERSE, p16uc_REVERSE, 8);//{ 4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11 };
static Packet16uc p16uc_COMPLEX_REV2 = vec_sld(p16uc_FORWARD, p16uc_FORWARD, 8);//{ 8,9,10,11, 12,13,14,15, 0,1,2,3, 4,5,6,7 };
static Packet16uc p16uc_PSET_HI = (Packet16uc) vec_mergeh((Packet4ui) vec_splat((Packet4ui)p16uc_FORWARD, 0), (Packet4ui) vec_splat((Packet4ui)p16uc_FORWARD, 1));//{ 0,1,2,3, 4,5,6,7, 0,1,2,3, 4,5,6,7 };
static Packet16uc p16uc_PSET_LO = (Packet16uc) vec_mergeh((Packet4ui) vec_splat((Packet4ui)p16uc_FORWARD, 2), (Packet4ui) vec_splat((Packet4ui)p16uc_FORWARD, 3));//{ 8,9,10,11, 12,13,14,15, 8,9,10,11, 12,13,14,15 };

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
  Packet2cf res;
  /* On AltiVec we cannot load 64-bit registers, so wa have to take care of alignment */
  if((ptrdiff_t(&from) % 16) == 0)
    res.v = pload<Packet4f>((const float *)&from);
  else
    res.v = ploadu<Packet4f>((const float *)&from);
  res.v = vec_perm(res.v, res.v, p16uc_PSET_HI);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_add(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_sub(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a) { return Packet2cf(pnegate(a.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a) { return Packet2cf((Packet4f)vec_xor((Packet4ui)a.v, p4ui_CONJ_XOR)); }

template<> EIGEN_STRONG_INLINE Packet2cf pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet4f v1, v2;

  // Permute and multiply the real parts of a and b
  v1 = vec_perm(a.v, a.v, p16uc_COMPLEX_RE);
  // Get the imaginary parts of a
  v2 = vec_perm(a.v, a.v, p16uc_COMPLEX_IM);
  // multiply a_re * b 
  v1 = vec_madd(v1, b.v, p4f_ZERO);
  // multiply a_im * b and get the conjugate result
  v2 = vec_madd(v2, b.v, p4f_ZERO);
  v2 = (Packet4f) vec_xor((Packet4ui)v2, p4ui_CONJ_XOR);
  // permute back to a proper order
  v2 = vec_perm(v2, v2, p16uc_COMPLEX_REV);
  
  return Packet2cf(vec_add(v1, v2));
}

template<> EIGEN_STRONG_INLINE Packet2cf pand   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_and(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf por    <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_or(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pxor   <Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_xor(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(vec_and(a.v, vec_nor(b.v,b.v))); }

template<> EIGEN_STRONG_INLINE Packet2cf pload <Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet2cf(pload<Packet4f>((const float*)from)); }
template<> EIGEN_STRONG_INLINE Packet2cf ploadu<Packet2cf>(const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cf(ploadu<Packet4f>((const float*)from)); }

template<> EIGEN_STRONG_INLINE Packet2cf ploaddup<Packet2cf>(const std::complex<float>*     from)
{
  return pset1<Packet2cf>(*from);
}

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((float*)to, from.v); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((float*)to, from.v); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float> *   addr) { vec_dstt((float *)addr, DST_CTRL(2,2,32), DST_CHAN); }

template<> EIGEN_STRONG_INLINE std::complex<float>  pfirst<Packet2cf>(const Packet2cf& a)
{
  std::complex<float> EIGEN_ALIGN16 res[2];
  pstore((float *)&res, a.v);

  return res[0];
}

template<> EIGEN_STRONG_INLINE Packet2cf preverse(const Packet2cf& a)
{
  Packet4f rev_a;
  rev_a = vec_perm(a.v, a.v, p16uc_COMPLEX_REV2);
  return Packet2cf(rev_a);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet2cf>(const Packet2cf& a)
{
  Packet4f b;
  b = (Packet4f) vec_sld(a.v, a.v, 8);
  b = padd(a.v, b);
  return pfirst(Packet2cf(b));
}

template<> EIGEN_STRONG_INLINE Packet2cf preduxp<Packet2cf>(const Packet2cf* vecs)
{
  Packet4f b1, b2;
  
  b1 = (Packet4f) vec_sld(vecs[0].v, vecs[1].v, 8);
  b2 = (Packet4f) vec_sld(vecs[1].v, vecs[0].v, 8);
  b2 = (Packet4f) vec_sld(b2, b2, 8);
  b2 = padd(b1, b2);

  return Packet2cf(b2);
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet2cf>(const Packet2cf& a)
{
  Packet4f b;
  Packet2cf prod;
  b = (Packet4f) vec_sld(a.v, a.v, 8);
  prod = pmul(a, Packet2cf(b));

  return pfirst(prod);
}

template<int Offset>
struct palign_impl<Offset,Packet2cf>
{
  static EIGEN_STRONG_INLINE void run(Packet2cf& first, const Packet2cf& second)
  {
    if (Offset==1)
    {
      first.v = vec_sld(first.v, second.v, 8);
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
  Packet4f s = vec_madd(b.v, b.v, p4f_ZERO);
  return Packet2cf(pdiv(res.v, vec_add(s,vec_perm(s, s, p16uc_COMPLEX_REV))));
}

template<> EIGEN_STRONG_INLINE Packet2cf pcplxflip<Packet2cf>(const Packet2cf& x)
{
  return Packet2cf(vec_perm(x.v, x.v, p16uc_COMPLEX_REV));
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_COMPLEX_ALTIVEC_H
