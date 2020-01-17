// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GENERIC_PACKET_MATH_H
#define EIGEN_GENERIC_PACKET_MATH_H

namespace Eigen {

namespace internal {

/** \internal
  * \file GenericPacketMath.h
  *
  * Default implementation for types not supported by the vectorization.
  * In practice these functions are provided to make easier the writing
  * of generic vectorized code.
  */

#ifndef EIGEN_DEBUG_ALIGNED_LOAD
#define EIGEN_DEBUG_ALIGNED_LOAD
#endif

#ifndef EIGEN_DEBUG_UNALIGNED_LOAD
#define EIGEN_DEBUG_UNALIGNED_LOAD
#endif

#ifndef EIGEN_DEBUG_ALIGNED_STORE
#define EIGEN_DEBUG_ALIGNED_STORE
#endif

#ifndef EIGEN_DEBUG_UNALIGNED_STORE
#define EIGEN_DEBUG_UNALIGNED_STORE
#endif

struct default_packet_traits
{
  enum {
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasNegate = 1,
    HasAbs    = 1,
    HasAbs2   = 1,
    HasMin    = 1,
    HasMax    = 1,
    HasConj   = 1,
    HasSetLinear = 1,

    HasDiv    = 0,
    HasSqrt   = 0,
    HasExp    = 0,
    HasLog    = 0,
    HasPow    = 0,

    HasSin    = 0,
    HasCos    = 0,
    HasTan    = 0,
    HasASin   = 0,
    HasACos   = 0,
    HasATan   = 0
  };
};

template<typename T> struct packet_traits : default_packet_traits
{
  typedef T type;
  enum {
    Vectorizable = 0,
    size = 1,
    AlignedOnScalar = 0
  };
  enum {
    HasAdd    = 0,
    HasSub    = 0,
    HasMul    = 0,
    HasNegate = 0,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0
  };
};

/** \internal \returns a + b (coeff-wise) */
template<typename Packet> inline Packet
padd(const Packet& a,
        const Packet& b) { return a+b; }

/** \internal \returns a - b (coeff-wise) */
template<typename Packet> inline Packet
psub(const Packet& a,
        const Packet& b) { return a-b; }

/** \internal \returns -a (coeff-wise) */
template<typename Packet> inline Packet
pnegate(const Packet& a) { return -a; }

/** \internal \returns conj(a) (coeff-wise) */
template<typename Packet> inline Packet
pconj(const Packet& a) { return numext::conj(a); }

/** \internal \returns a * b (coeff-wise) */
template<typename Packet> inline Packet
pmul(const Packet& a,
        const Packet& b) { return a*b; }

/** \internal \returns a / b (coeff-wise) */
template<typename Packet> inline Packet
pdiv(const Packet& a,
        const Packet& b) { return a/b; }

/** \internal \returns the min of \a a and \a b  (coeff-wise) */
template<typename Packet> inline Packet
pmin(const Packet& a,
        const Packet& b) { using std::min; return (min)(a, b); }

/** \internal \returns the max of \a a and \a b  (coeff-wise) */
template<typename Packet> inline Packet
pmax(const Packet& a,
        const Packet& b) { using std::max; return (max)(a, b); }

/** \internal \returns the absolute value of \a a */
template<typename Packet> inline Packet
pabs(const Packet& a) { using std::abs; return abs(a); }

/** \internal \returns the bitwise and of \a a and \a b */
template<typename Packet> inline Packet
pand(const Packet& a, const Packet& b) { return a & b; }

/** \internal \returns the bitwise or of \a a and \a b */
template<typename Packet> inline Packet
por(const Packet& a, const Packet& b) { return a | b; }

/** \internal \returns the bitwise xor of \a a and \a b */
template<typename Packet> inline Packet
pxor(const Packet& a, const Packet& b) { return a ^ b; }

/** \internal \returns the bitwise andnot of \a a and \a b */
template<typename Packet> inline Packet
pandnot(const Packet& a, const Packet& b) { return a & (!b); }

/** \internal \returns a packet version of \a *from, from must be 16 bytes aligned */
template<typename Packet> inline Packet
pload(const typename unpacket_traits<Packet>::type* from) { return *from; }

/** \internal \returns a packet version of \a *from, (un-aligned load) */
template<typename Packet> inline Packet
ploadu(const typename unpacket_traits<Packet>::type* from) { return *from; }

/** \internal \returns a packet with elements of \a *from duplicated.
  * For instance, for a packet of 8 elements, 4 scalar will be read from \a *from and
  * duplicated to form: {from[0],from[0],from[1],from[1],,from[2],from[2],,from[3],from[3]}
  * Currently, this function is only used for scalar * complex products.
 */
template<typename Packet> inline Packet
ploaddup(const typename unpacket_traits<Packet>::type* from) { return *from; }

/** \internal \returns a packet with constant coefficients \a a, e.g.: (a,a,a,a) */
template<typename Packet> inline Packet
pset1(const typename unpacket_traits<Packet>::type& a) { return a; }

/** \internal \brief Returns a packet with coefficients (a,a+1,...,a+packet_size-1). */
template<typename Scalar> inline typename packet_traits<Scalar>::type
plset(const Scalar& a) { return a; }

/** \internal copy the packet \a from to \a *to, \a to must be 16 bytes aligned */
template<typename Scalar, typename Packet> inline void pstore(Scalar* to, const Packet& from)
{ (*to) = from; }

/** \internal copy the packet \a from to \a *to, (un-aligned store) */
template<typename Scalar, typename Packet> inline void pstoreu(Scalar* to, const Packet& from)
{ (*to) = from; }

/** \internal tries to do cache prefetching of \a addr */
template<typename Scalar> inline void prefetch(const Scalar* addr)
{
#if !defined(_MSC_VER)
__builtin_prefetch(addr);
#endif
}

/** \internal \returns the first element of a packet */
template<typename Packet> inline typename unpacket_traits<Packet>::type pfirst(const Packet& a)
{ return a; }

/** \internal \returns a packet where the element i contains the sum of the packet of \a vec[i] */
template<typename Packet> inline Packet
preduxp(const Packet* vecs) { return vecs[0]; }

/** \internal \returns the sum of the elements of \a a*/
template<typename Packet> inline typename unpacket_traits<Packet>::type predux(const Packet& a)
{ return a; }

/** \internal \returns the product of the elements of \a a*/
template<typename Packet> inline typename unpacket_traits<Packet>::type predux_mul(const Packet& a)
{ return a; }

/** \internal \returns the min of the elements of \a a*/
template<typename Packet> inline typename unpacket_traits<Packet>::type predux_min(const Packet& a)
{ return a; }

/** \internal \returns the max of the elements of \a a*/
template<typename Packet> inline typename unpacket_traits<Packet>::type predux_max(const Packet& a)
{ return a; }

/** \internal \returns the reversed elements of \a a*/
template<typename Packet> inline Packet preverse(const Packet& a)
{ return a; }


/** \internal \returns \a a with real and imaginary part flipped (for complex type only) */
template<typename Packet> inline Packet pcplxflip(const Packet& a)
{
  // FIXME: uncomment the following in case we drop the internal imag and real functions.
//   using std::imag;
//   using std::real;
  return Packet(imag(a),real(a));
}

/**************************
* Special math functions
***************************/

/** \internal \returns the sine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet psin(const Packet& a) { using std::sin; return sin(a); }

/** \internal \returns the cosine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pcos(const Packet& a) { using std::cos; return cos(a); }

/** \internal \returns the tan of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet ptan(const Packet& a) { using std::tan; return tan(a); }

/** \internal \returns the arc sine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pasin(const Packet& a) { using std::asin; return asin(a); }

/** \internal \returns the arc cosine of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pacos(const Packet& a) { using std::acos; return acos(a); }

/** \internal \returns the exp of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet pexp(const Packet& a) { using std::exp; return exp(a); }

/** \internal \returns the log of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet plog(const Packet& a) { using std::log; return log(a); }

/** \internal \returns the square-root of \a a (coeff-wise) */
template<typename Packet> EIGEN_DECLARE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS
Packet psqrt(const Packet& a) { using std::sqrt; return sqrt(a); }

/***************************************************************************
* The following functions might not have to be overwritten for vectorized types
***************************************************************************/

/** \internal copy a packet with constant coeficient \a a (e.g., [a,a,a,a]) to \a *to. \a to must be 16 bytes aligned */
// NOTE: this function must really be templated on the packet type (think about different packet types for the same scalar type)
template<typename Packet>
inline void pstore1(typename unpacket_traits<Packet>::type* to, const typename unpacket_traits<Packet>::type& a)
{
  pstore(to, pset1<Packet>(a));
}

/** \internal \returns a * b + c (coeff-wise) */
template<typename Packet> inline Packet
pmadd(const Packet&  a,
         const Packet&  b,
         const Packet&  c)
{ return padd(pmul(a, b),c); }

/** \internal \returns a packet version of \a *from.
  * If LoadMode equals #Aligned, \a from must be 16 bytes aligned */
template<typename Packet, int LoadMode>
inline Packet ploadt(const typename unpacket_traits<Packet>::type* from)
{
  if(LoadMode == Aligned)
    return pload<Packet>(from);
  else
    return ploadu<Packet>(from);
}

/** \internal copy the packet \a from to \a *to.
  * If StoreMode equals #Aligned, \a to must be 16 bytes aligned */
template<typename Scalar, typename Packet, int LoadMode>
inline void pstoret(Scalar* to, const Packet& from)
{
  if(LoadMode == Aligned)
    pstore(to, from);
  else
    pstoreu(to, from);
}

/** \internal default implementation of palign() allowing partial specialization */
template<int Offset,typename PacketType>
struct palign_impl
{
  // by default data are aligned, so there is nothing to be done :)
  static inline void run(PacketType&, const PacketType&) {}
};

/** \internal update \a first using the concatenation of the packet_size minus \a Offset last elements
  * of \a first and \a Offset first elements of \a second.
  * 
  * This function is currently only used to optimize matrix-vector products on unligned matrices.
  * It takes 2 packets that represent a contiguous memory array, and returns a packet starting
  * at the position \a Offset. For instance, for packets of 4 elements, we have:
  *  Input:
  *  - first = {f0,f1,f2,f3}
  *  - second = {s0,s1,s2,s3}
  * Output: 
  *   - if Offset==0 then {f0,f1,f2,f3}
  *   - if Offset==1 then {f1,f2,f3,s0}
  *   - if Offset==2 then {f2,f3,s0,s1}
  *   - if Offset==3 then {f3,s0,s1,s3}
  */
template<int Offset,typename PacketType>
inline void palign(PacketType& first, const PacketType& second)
{
  palign_impl<Offset,PacketType>::run(first,second);
}

/***************************************************************************
* Fast complex products (GCC generates a function call which is very slow)
***************************************************************************/

template<> inline std::complex<float> pmul(const std::complex<float>& a, const std::complex<float>& b)
{ return std::complex<float>(real(a)*real(b) - imag(a)*imag(b), imag(a)*real(b) + real(a)*imag(b)); }

template<> inline std::complex<double> pmul(const std::complex<double>& a, const std::complex<double>& b)
{ return std::complex<double>(real(a)*real(b) - imag(a)*imag(b), imag(a)*real(b) + real(a)*imag(b)); }

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GENERIC_PACKET_MATH_H

