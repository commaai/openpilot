// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Konstantinos Margaritis <markos@codex.gr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PACKET_MATH_ALTIVEC_H
#define EIGEN_PACKET_MATH_ALTIVEC_H

namespace Eigen {

namespace internal {

#ifndef EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD
#define EIGEN_CACHEFRIENDLY_PRODUCT_THRESHOLD 4
#endif

#ifndef EIGEN_HAS_FUSE_CJMADD
#define EIGEN_HAS_FUSE_CJMADD 1
#endif

// NOTE Altivec has 32 registers, but Eigen only accepts a value of 8 or 16
#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 16
#endif

typedef __vector float          Packet4f;
typedef __vector int            Packet4i;
typedef __vector unsigned int   Packet4ui;
typedef __vector __bool int     Packet4bi;
typedef __vector short int      Packet8i;
typedef __vector unsigned char  Packet16uc;

// We don't want to write the same code all the time, but we need to reuse the constants
// and it doesn't really work to declare them global, so we define macros instead

#define _EIGEN_DECLARE_CONST_FAST_Packet4f(NAME,X) \
  Packet4f p4f_##NAME = (Packet4f) vec_splat_s32(X)

#define _EIGEN_DECLARE_CONST_FAST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = vec_splat_s32(X)

#define _EIGEN_DECLARE_CONST_Packet4f(NAME,X) \
  Packet4f p4f_##NAME = pset1<Packet4f>(X)

#define _EIGEN_DECLARE_CONST_Packet4f_FROM_INT(NAME,X) \
  Packet4f p4f_##NAME = vreinterpretq_f32_u32(pset1<int>(X))

#define _EIGEN_DECLARE_CONST_Packet4i(NAME,X) \
  Packet4i p4i_##NAME = pset1<Packet4i>(X)

#define DST_CHAN 1
#define DST_CTRL(size, count, stride) (((size) << 24) | ((count) << 16) | (stride))

// Define global static constants:
static Packet4f p4f_COUNTDOWN = { 3.0, 2.0, 1.0, 0.0 };
static Packet4i p4i_COUNTDOWN = { 3, 2, 1, 0 };
static Packet16uc p16uc_REVERSE = {12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3};
static Packet16uc p16uc_FORWARD = vec_lvsl(0, (float*)0);
static Packet16uc p16uc_DUPLICATE = {0,1,2,3, 0,1,2,3, 4,5,6,7, 4,5,6,7};

static _EIGEN_DECLARE_CONST_FAST_Packet4f(ZERO, 0);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(ZERO, 0);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(ONE,1);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(MINUS16,-16);
static _EIGEN_DECLARE_CONST_FAST_Packet4i(MINUS1,-1);
static Packet4f p4f_ONE = vec_ctf(p4i_ONE, 0);
static Packet4f p4f_ZERO_ = (Packet4f) vec_sl((Packet4ui)p4i_MINUS1, (Packet4ui)p4i_MINUS1);

template<> struct packet_traits<float>  : default_packet_traits
{
  typedef Packet4f type;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4,

    // FIXME check the Has*
    HasSin  = 0,
    HasCos  = 0,
    HasLog  = 0,
    HasExp  = 0,
    HasSqrt = 0
  };
};
template<> struct packet_traits<int>    : default_packet_traits
{
  typedef Packet4i type;
  enum {
    // FIXME check the Has*
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size=4
  };
};

template<> struct unpacket_traits<Packet4f> { typedef float  type; enum {size=4}; };
template<> struct unpacket_traits<Packet4i> { typedef int    type; enum {size=4}; };
/*
inline std::ostream & operator <<(std::ostream & s, const Packet4f & v)
{
  union {
    Packet4f   v;
    float n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet4i & v)
{
  union {
    Packet4i   v;
    int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packet4ui & v)
{
  union {
    Packet4ui   v;
    unsigned int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}

inline std::ostream & operator <<(std::ostream & s, const Packetbi & v)
{
  union {
    Packet4bi v;
    unsigned int n[4];
  } vt;
  vt.v = v;
  s << vt.n[0] << ", " << vt.n[1] << ", " << vt.n[2] << ", " << vt.n[3];
  return s;
}
*/
template<> EIGEN_STRONG_INLINE Packet4f pset1<Packet4f>(const float&  from) {
  // Taken from http://developer.apple.com/hardwaredrivers/ve/alignment.html
  float EIGEN_ALIGN16 af[4];
  af[0] = from;
  Packet4f vc = vec_ld(0, af);
  vc = vec_splat(vc, 0);
  return vc;
}

template<> EIGEN_STRONG_INLINE Packet4i pset1<Packet4i>(const int&    from)   {
  int EIGEN_ALIGN16 ai[4];
  ai[0] = from;
  Packet4i vc = vec_ld(0, ai);
  vc = vec_splat(vc, 0);
  return vc;
}

template<> EIGEN_STRONG_INLINE Packet4f plset<float>(const float& a) { return vec_add(pset1<Packet4f>(a), p4f_COUNTDOWN); }
template<> EIGEN_STRONG_INLINE Packet4i plset<int>(const int& a)     { return vec_add(pset1<Packet4i>(a), p4i_COUNTDOWN); }

template<> EIGEN_STRONG_INLINE Packet4f padd<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_add(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i padd<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_add(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f psub<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_sub(a,b); }
template<> EIGEN_STRONG_INLINE Packet4i psub<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_sub(a,b); }

template<> EIGEN_STRONG_INLINE Packet4f pnegate(const Packet4f& a) { return psub<Packet4f>(p4f_ZERO, a); }
template<> EIGEN_STRONG_INLINE Packet4i pnegate(const Packet4i& a) { return psub<Packet4i>(p4i_ZERO, a); }

template<> EIGEN_STRONG_INLINE Packet4f pconj(const Packet4f& a) { return a; }
template<> EIGEN_STRONG_INLINE Packet4i pconj(const Packet4i& a) { return a; }

template<> EIGEN_STRONG_INLINE Packet4f pmul<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_madd(a,b,p4f_ZERO); }
/* Commented out: it's actually slower than processing it scalar
 *
template<> EIGEN_STRONG_INLINE Packet4i pmul<Packet4i>(const Packet4i& a, const Packet4i& b)
{
  // Detailed in: http://freevec.org/content/32bit_signed_integer_multiplication_altivec
  //Set up constants, variables
  Packet4i a1, b1, bswap, low_prod, high_prod, prod, prod_, v1sel;

  // Get the absolute values
  a1  = vec_abs(a);
  b1  = vec_abs(b);

  // Get the signs using xor
  Packet4bi sgn = (Packet4bi) vec_cmplt(vec_xor(a, b), p4i_ZERO);

  // Do the multiplication for the asbolute values.
  bswap = (Packet4i) vec_rl((Packet4ui) b1, (Packet4ui) p4i_MINUS16 );
  low_prod = vec_mulo((Packet8i) a1, (Packet8i)b1);
  high_prod = vec_msum((Packet8i) a1, (Packet8i) bswap, p4i_ZERO);
  high_prod = (Packet4i) vec_sl((Packet4ui) high_prod, (Packet4ui) p4i_MINUS16);
  prod = vec_add( low_prod, high_prod );

  // NOR the product and select only the negative elements according to the sign mask
  prod_ = vec_nor(prod, prod);
  prod_ = vec_sel(p4i_ZERO, prod_, sgn);

  // Add 1 to the result to get the negative numbers
  v1sel = vec_sel(p4i_ZERO, p4i_ONE, sgn);
  prod_ = vec_add(prod_, v1sel);

  // Merge the results back to the final vector.
  prod = vec_sel(prod, prod_, sgn);

  return prod;
}
*/
template<> EIGEN_STRONG_INLINE Packet4f pdiv<Packet4f>(const Packet4f& a, const Packet4f& b)
{
  Packet4f t, y_0, y_1, res;

  // Altivec does not offer a divide instruction, we have to do a reciprocal approximation
  y_0 = vec_re(b);

  // Do one Newton-Raphson iteration to get the needed accuracy
  t   = vec_nmsub(y_0, b, p4f_ONE);
  y_1 = vec_madd(y_0, t, y_0);

  res = vec_madd(a, y_1, p4f_ZERO);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet4i pdiv<Packet4i>(const Packet4i& /*a*/, const Packet4i& /*b*/)
{ eigen_assert(false && "packet integer division are not supported by AltiVec");
  return pset1<Packet4i>(0);
}

// for some weird raisons, it has to be overloaded for packet of integers
template<> EIGEN_STRONG_INLINE Packet4f pmadd(const Packet4f& a, const Packet4f& b, const Packet4f& c) { return vec_madd(a, b, c); }
template<> EIGEN_STRONG_INLINE Packet4i pmadd(const Packet4i& a, const Packet4i& b, const Packet4i& c) { return padd(pmul(a,b), c); }

template<> EIGEN_STRONG_INLINE Packet4f pmin<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_min(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pmin<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_min(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pmax<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_max(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pmax<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_max(a, b); }

// Logical Operations are not supported for float, so we have to reinterpret casts using NEON intrinsics
template<> EIGEN_STRONG_INLINE Packet4f pand<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_and(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pand<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_and(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f por<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_or(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i por<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_or(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pxor<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_xor(a, b); }
template<> EIGEN_STRONG_INLINE Packet4i pxor<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_xor(a, b); }

template<> EIGEN_STRONG_INLINE Packet4f pandnot<Packet4f>(const Packet4f& a, const Packet4f& b) { return vec_and(a, vec_nor(b, b)); }
template<> EIGEN_STRONG_INLINE Packet4i pandnot<Packet4i>(const Packet4i& a, const Packet4i& b) { return vec_and(a, vec_nor(b, b)); }

template<> EIGEN_STRONG_INLINE Packet4f pload<Packet4f>(const float* from) { EIGEN_DEBUG_ALIGNED_LOAD return vec_ld(0, from); }
template<> EIGEN_STRONG_INLINE Packet4i pload<Packet4i>(const int*     from) { EIGEN_DEBUG_ALIGNED_LOAD return vec_ld(0, from); }

template<> EIGEN_STRONG_INLINE Packet4f ploadu<Packet4f>(const float* from)
{
  EIGEN_DEBUG_ALIGNED_LOAD
  // Taken from http://developer.apple.com/hardwaredrivers/ve/alignment.html
  Packet16uc MSQ, LSQ;
  Packet16uc mask;
  MSQ = vec_ld(0, (unsigned char *)from);          // most significant quadword
  LSQ = vec_ld(15, (unsigned char *)from);         // least significant quadword
  mask = vec_lvsl(0, from);                        // create the permute mask
  return (Packet4f) vec_perm(MSQ, LSQ, mask);           // align the data

}
template<> EIGEN_STRONG_INLINE Packet4i ploadu<Packet4i>(const int* from)
{
  EIGEN_DEBUG_ALIGNED_LOAD
  // Taken from http://developer.apple.com/hardwaredrivers/ve/alignment.html
  Packet16uc MSQ, LSQ;
  Packet16uc mask;
  MSQ = vec_ld(0, (unsigned char *)from);          // most significant quadword
  LSQ = vec_ld(15, (unsigned char *)from);         // least significant quadword
  mask = vec_lvsl(0, from);                        // create the permute mask
  return (Packet4i) vec_perm(MSQ, LSQ, mask);    // align the data
}

template<> EIGEN_STRONG_INLINE Packet4f ploaddup<Packet4f>(const float*   from)
{
  Packet4f p;
  if((ptrdiff_t(&from) % 16) == 0)  p = pload<Packet4f>(from);
  else                              p = ploadu<Packet4f>(from);
  return vec_perm(p, p, p16uc_DUPLICATE);
}
template<> EIGEN_STRONG_INLINE Packet4i ploaddup<Packet4i>(const int*     from)
{
  Packet4i p;
  if((ptrdiff_t(&from) % 16) == 0)  p = pload<Packet4i>(from);
  else                              p = ploadu<Packet4i>(from);
  return vec_perm(p, p, p16uc_DUPLICATE);
}

template<> EIGEN_STRONG_INLINE void pstore<float>(float*   to, const Packet4f& from) { EIGEN_DEBUG_ALIGNED_STORE vec_st(from, 0, to); }
template<> EIGEN_STRONG_INLINE void pstore<int>(int*       to, const Packet4i& from) { EIGEN_DEBUG_ALIGNED_STORE vec_st(from, 0, to); }

template<> EIGEN_STRONG_INLINE void pstoreu<float>(float*  to, const Packet4f& from)
{
  EIGEN_DEBUG_UNALIGNED_STORE
  // Taken from http://developer.apple.com/hardwaredrivers/ve/alignment.html
  // Warning: not thread safe!
  Packet16uc MSQ, LSQ, edges;
  Packet16uc edgeAlign, align;

  MSQ = vec_ld(0, (unsigned char *)to);                     // most significant quadword
  LSQ = vec_ld(15, (unsigned char *)to);                    // least significant quadword
  edgeAlign = vec_lvsl(0, to);                              // permute map to extract edges
  edges=vec_perm(LSQ,MSQ,edgeAlign);                        // extract the edges
  align = vec_lvsr( 0, to );                                // permute map to misalign data
  MSQ = vec_perm(edges,(Packet16uc)from,align);             // misalign the data (MSQ)
  LSQ = vec_perm((Packet16uc)from,edges,align);             // misalign the data (LSQ)
  vec_st( LSQ, 15, (unsigned char *)to );                   // Store the LSQ part first
  vec_st( MSQ, 0, (unsigned char *)to );                    // Store the MSQ part
}
template<> EIGEN_STRONG_INLINE void pstoreu<int>(int*      to, const Packet4i& from)
{
  EIGEN_DEBUG_UNALIGNED_STORE
  // Taken from http://developer.apple.com/hardwaredrivers/ve/alignment.html
  // Warning: not thread safe!
  Packet16uc MSQ, LSQ, edges;
  Packet16uc edgeAlign, align;

  MSQ = vec_ld(0, (unsigned char *)to);                     // most significant quadword
  LSQ = vec_ld(15, (unsigned char *)to);                    // least significant quadword
  edgeAlign = vec_lvsl(0, to);                              // permute map to extract edges
  edges=vec_perm(LSQ, MSQ, edgeAlign);                      // extract the edges
  align = vec_lvsr( 0, to );                                // permute map to misalign data
  MSQ = vec_perm(edges, (Packet16uc) from, align);          // misalign the data (MSQ)
  LSQ = vec_perm((Packet16uc) from, edges, align);          // misalign the data (LSQ)
  vec_st( LSQ, 15, (unsigned char *)to );                   // Store the LSQ part first
  vec_st( MSQ, 0, (unsigned char *)to );                    // Store the MSQ part
}

template<> EIGEN_STRONG_INLINE void prefetch<float>(const float* addr) { vec_dstt(addr, DST_CTRL(2,2,32), DST_CHAN); }
template<> EIGEN_STRONG_INLINE void prefetch<int>(const int*     addr) { vec_dstt(addr, DST_CTRL(2,2,32), DST_CHAN); }

template<> EIGEN_STRONG_INLINE float  pfirst<Packet4f>(const Packet4f& a) { float EIGEN_ALIGN16 x[4]; vec_st(a, 0, x); return x[0]; }
template<> EIGEN_STRONG_INLINE int    pfirst<Packet4i>(const Packet4i& a) { int   EIGEN_ALIGN16 x[4]; vec_st(a, 0, x); return x[0]; }

template<> EIGEN_STRONG_INLINE Packet4f preverse(const Packet4f& a) { return (Packet4f)vec_perm((Packet16uc)a,(Packet16uc)a, p16uc_REVERSE); }
template<> EIGEN_STRONG_INLINE Packet4i preverse(const Packet4i& a) { return (Packet4i)vec_perm((Packet16uc)a,(Packet16uc)a, p16uc_REVERSE); }

template<> EIGEN_STRONG_INLINE Packet4f pabs(const Packet4f& a) { return vec_abs(a); }
template<> EIGEN_STRONG_INLINE Packet4i pabs(const Packet4i& a) { return vec_abs(a); }

template<> EIGEN_STRONG_INLINE float predux<Packet4f>(const Packet4f& a)
{
  Packet4f b, sum;
  b   = (Packet4f) vec_sld(a, a, 8);
  sum = vec_add(a, b);
  b   = (Packet4f) vec_sld(sum, sum, 4);
  sum = vec_add(sum, b);
  return pfirst(sum);
}

template<> EIGEN_STRONG_INLINE Packet4f preduxp<Packet4f>(const Packet4f* vecs)
{
  Packet4f v[4], sum[4];

  // It's easier and faster to transpose then add as columns
  // Check: http://www.freevec.org/function/matrix_4x4_transpose_floats for explanation
  // Do the transpose, first set of moves
  v[0] = vec_mergeh(vecs[0], vecs[2]);
  v[1] = vec_mergel(vecs[0], vecs[2]);
  v[2] = vec_mergeh(vecs[1], vecs[3]);
  v[3] = vec_mergel(vecs[1], vecs[3]);
  // Get the resulting vectors
  sum[0] = vec_mergeh(v[0], v[2]);
  sum[1] = vec_mergel(v[0], v[2]);
  sum[2] = vec_mergeh(v[1], v[3]);
  sum[3] = vec_mergel(v[1], v[3]);

  // Now do the summation:
  // Lines 0+1
  sum[0] = vec_add(sum[0], sum[1]);
  // Lines 2+3
  sum[1] = vec_add(sum[2], sum[3]);
  // Add the results
  sum[0] = vec_add(sum[0], sum[1]);

  return sum[0];
}

template<> EIGEN_STRONG_INLINE int predux<Packet4i>(const Packet4i& a)
{
  Packet4i sum;
  sum = vec_sums(a, p4i_ZERO);
  sum = vec_sld(sum, p4i_ZERO, 12);
  return pfirst(sum);
}

template<> EIGEN_STRONG_INLINE Packet4i preduxp<Packet4i>(const Packet4i* vecs)
{
  Packet4i v[4], sum[4];

  // It's easier and faster to transpose then add as columns
  // Check: http://www.freevec.org/function/matrix_4x4_transpose_floats for explanation
  // Do the transpose, first set of moves
  v[0] = vec_mergeh(vecs[0], vecs[2]);
  v[1] = vec_mergel(vecs[0], vecs[2]);
  v[2] = vec_mergeh(vecs[1], vecs[3]);
  v[3] = vec_mergel(vecs[1], vecs[3]);
  // Get the resulting vectors
  sum[0] = vec_mergeh(v[0], v[2]);
  sum[1] = vec_mergel(v[0], v[2]);
  sum[2] = vec_mergeh(v[1], v[3]);
  sum[3] = vec_mergel(v[1], v[3]);

  // Now do the summation:
  // Lines 0+1
  sum[0] = vec_add(sum[0], sum[1]);
  // Lines 2+3
  sum[1] = vec_add(sum[2], sum[3]);
  // Add the results
  sum[0] = vec_add(sum[0], sum[1]);

  return sum[0];
}

// Other reduction functions:
// mul
template<> EIGEN_STRONG_INLINE float predux_mul<Packet4f>(const Packet4f& a)
{
  Packet4f prod;
  prod = pmul(a, (Packet4f)vec_sld(a, a, 8));
  return pfirst(pmul(prod, (Packet4f)vec_sld(prod, prod, 4)));
}

template<> EIGEN_STRONG_INLINE int predux_mul<Packet4i>(const Packet4i& a)
{
  EIGEN_ALIGN16 int aux[4];
  pstore(aux, a);
  return aux[0] * aux[1] * aux[2] * aux[3];
}

// min
template<> EIGEN_STRONG_INLINE float predux_min<Packet4f>(const Packet4f& a)
{
  Packet4f b, res;
  b = vec_min(a, vec_sld(a, a, 8));
  res = vec_min(b, vec_sld(b, b, 4));
  return pfirst(res);
}

template<> EIGEN_STRONG_INLINE int predux_min<Packet4i>(const Packet4i& a)
{
  Packet4i b, res;
  b = vec_min(a, vec_sld(a, a, 8));
  res = vec_min(b, vec_sld(b, b, 4));
  return pfirst(res);
}

// max
template<> EIGEN_STRONG_INLINE float predux_max<Packet4f>(const Packet4f& a)
{
  Packet4f b, res;
  b = vec_max(a, vec_sld(a, a, 8));
  res = vec_max(b, vec_sld(b, b, 4));
  return pfirst(res);
}

template<> EIGEN_STRONG_INLINE int predux_max<Packet4i>(const Packet4i& a)
{
  Packet4i b, res;
  b = vec_max(a, vec_sld(a, a, 8));
  res = vec_max(b, vec_sld(b, b, 4));
  return pfirst(res);
}

template<int Offset>
struct palign_impl<Offset,Packet4f>
{
  static EIGEN_STRONG_INLINE void run(Packet4f& first, const Packet4f& second)
  {
    if (Offset!=0)
      first = vec_sld(first, second, Offset*4);
  }
};

template<int Offset>
struct palign_impl<Offset,Packet4i>
{
  static EIGEN_STRONG_INLINE void run(Packet4i& first, const Packet4i& second)
  {
    if (Offset!=0)
      first = vec_sld(first, second, Offset*4);
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PACKET_MATH_ALTIVEC_H
