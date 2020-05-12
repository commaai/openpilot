// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Eugene Brevdo <ebrevdo@gmail.com>
// Copyright (C) 2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPECIALFUNCTIONS_FUNCTORS_H
#define EIGEN_SPECIALFUNCTIONS_FUNCTORS_H

namespace Eigen {

namespace internal {


/** \internal
  * \brief Template functor to compute the incomplete gamma function igamma(a, x)
  *
  * \sa class CwiseBinaryOp, Cwise::igamma
  */
template<typename Scalar> struct scalar_igamma_op : binary_op_base<Scalar,Scalar>
{
  EIGEN_EMPTY_STRUCT_CTOR(scalar_igamma_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& x) const {
    using numext::igamma; return igamma(a, x);
  }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& x) const {
    return internal::pigamma(a, x);
  }
};
template<typename Scalar>
struct functor_traits<scalar_igamma_op<Scalar> > {
  enum {
    // Guesstimate
    Cost = 20 * NumTraits<Scalar>::MulCost + 10 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasIGamma
  };
};


/** \internal
  * \brief Template functor to compute the complementary incomplete gamma function igammac(a, x)
  *
  * \sa class CwiseBinaryOp, Cwise::igammac
  */
template<typename Scalar> struct scalar_igammac_op : binary_op_base<Scalar,Scalar>
{
  EIGEN_EMPTY_STRUCT_CTOR(scalar_igammac_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& a, const Scalar& x) const {
    using numext::igammac; return igammac(a, x);
  }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& x) const
  {
    return internal::pigammac(a, x);
  }
};
template<typename Scalar>
struct functor_traits<scalar_igammac_op<Scalar> > {
  enum {
    // Guesstimate
    Cost = 20 * NumTraits<Scalar>::MulCost + 10 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasIGammac
  };
};


/** \internal
  * \brief Template functor to compute the incomplete beta integral betainc(a, b, x)
  *
  */
template<typename Scalar> struct scalar_betainc_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_betainc_op)
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& x, const Scalar& a, const Scalar& b) const {
    using numext::betainc; return betainc(x, a, b);
  }
  template<typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& x, const Packet& a, const Packet& b) const
  {
    return internal::pbetainc(x, a, b);
  }
};
template<typename Scalar>
struct functor_traits<scalar_betainc_op<Scalar> > {
  enum {
    // Guesstimate
    Cost = 400 * NumTraits<Scalar>::MulCost + 400 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasBetaInc
  };
};


/** \internal
 * \brief Template functor to compute the natural log of the absolute
 * value of Gamma of a scalar
 * \sa class CwiseUnaryOp, Cwise::lgamma()
 */
template<typename Scalar> struct scalar_lgamma_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_lgamma_op)
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const {
    using numext::lgamma; return lgamma(a);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const { return internal::plgamma(a); }
};
template<typename Scalar>
struct functor_traits<scalar_lgamma_op<Scalar> >
{
  enum {
    // Guesstimate
    Cost = 10 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasLGamma
  };
};

/** \internal
 * \brief Template functor to compute psi, the derivative of lgamma of a scalar.
 * \sa class CwiseUnaryOp, Cwise::digamma()
 */
template<typename Scalar> struct scalar_digamma_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_digamma_op)
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const {
    using numext::digamma; return digamma(a);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const { return internal::pdigamma(a); }
};
template<typename Scalar>
struct functor_traits<scalar_digamma_op<Scalar> >
{
  enum {
    // Guesstimate
    Cost = 10 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasDiGamma
  };
};

/** \internal
 * \brief Template functor to compute the Riemann Zeta function of two arguments.
 * \sa class CwiseUnaryOp, Cwise::zeta()
 */
template<typename Scalar> struct scalar_zeta_op {
    EIGEN_EMPTY_STRUCT_CTOR(scalar_zeta_op)
    EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& x, const Scalar& q) const {
        using numext::zeta; return zeta(x, q);
    }
    typedef typename packet_traits<Scalar>::type Packet;
    EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& x, const Packet& q) const { return internal::pzeta(x, q); }
};
template<typename Scalar>
struct functor_traits<scalar_zeta_op<Scalar> >
{
    enum {
        // Guesstimate
        Cost = 10 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost,
        PacketAccess = packet_traits<Scalar>::HasZeta
    };
};

/** \internal
 * \brief Template functor to compute the polygamma function.
 * \sa class CwiseUnaryOp, Cwise::polygamma()
 */
template<typename Scalar> struct scalar_polygamma_op {
    EIGEN_EMPTY_STRUCT_CTOR(scalar_polygamma_op)
    EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& n, const Scalar& x) const {
        using numext::polygamma; return polygamma(n, x);
    }
    typedef typename packet_traits<Scalar>::type Packet;
    EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& n, const Packet& x) const { return internal::ppolygamma(n, x); }
};
template<typename Scalar>
struct functor_traits<scalar_polygamma_op<Scalar> >
{
    enum {
        // Guesstimate
        Cost = 10 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost,
        PacketAccess = packet_traits<Scalar>::HasPolygamma
    };
};

/** \internal
 * \brief Template functor to compute the Gauss error function of a
 * scalar
 * \sa class CwiseUnaryOp, Cwise::erf()
 */
template<typename Scalar> struct scalar_erf_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_erf_op)
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const {
    using numext::erf; return erf(a);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const { return internal::perf(a); }
};
template<typename Scalar>
struct functor_traits<scalar_erf_op<Scalar> >
{
  enum {
    // Guesstimate
    Cost = 10 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasErf
  };
};

/** \internal
 * \brief Template functor to compute the Complementary Error Function
 * of a scalar
 * \sa class CwiseUnaryOp, Cwise::erfc()
 */
template<typename Scalar> struct scalar_erfc_op {
  EIGEN_EMPTY_STRUCT_CTOR(scalar_erfc_op)
  EIGEN_DEVICE_FUNC inline const Scalar operator() (const Scalar& a) const {
    using numext::erfc; return erfc(a);
  }
  typedef typename packet_traits<Scalar>::type Packet;
  EIGEN_DEVICE_FUNC inline Packet packetOp(const Packet& a) const { return internal::perfc(a); }
};
template<typename Scalar>
struct functor_traits<scalar_erfc_op<Scalar> >
{
  enum {
    // Guesstimate
    Cost = 10 * NumTraits<Scalar>::MulCost + 5 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasErfc
  };
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_SPECIALFUNCTIONS_FUNCTORS_H
