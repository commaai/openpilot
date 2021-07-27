// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATHFUNCTIONS_H
#define EIGEN_MATHFUNCTIONS_H

namespace Eigen {

namespace internal {

/** \internal \struct global_math_functions_filtering_base
  *
  * What it does:
  * Defines a typedef 'type' as follows:
  * - if type T has a member typedef Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl, then
  *   global_math_functions_filtering_base<T>::type is a typedef for it.
  * - otherwise, global_math_functions_filtering_base<T>::type is a typedef for T.
  *
  * How it's used:
  * To allow to defined the global math functions (like sin...) in certain cases, like the Array expressions.
  * When you do sin(array1+array2), the object array1+array2 has a complicated expression type, all what you want to know
  * is that it inherits ArrayBase. So we implement a partial specialization of sin_impl for ArrayBase<Derived>.
  * So we must make sure to use sin_impl<ArrayBase<Derived> > and not sin_impl<Derived>, otherwise our partial specialization
  * won't be used. How does sin know that? That's exactly what global_math_functions_filtering_base tells it.
  *
  * How it's implemented:
  * SFINAE in the style of enable_if. Highly susceptible of breaking compilers. With GCC, it sure does work, but if you replace
  * the typename dummy by an integer template parameter, it doesn't work anymore!
  */

template<typename T, typename dummy = void>
struct global_math_functions_filtering_base
{
  typedef T type;
};

template<typename T> struct always_void { typedef void type; };

template<typename T>
struct global_math_functions_filtering_base
  <T,
   typename always_void<typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl>::type
  >
{
  typedef typename T::Eigen_BaseClassForSpecializationOfGlobalMathFuncImpl type;
};

#define EIGEN_MATHFUNC_IMPL(func, scalar) Eigen::internal::func##_impl<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>
#define EIGEN_MATHFUNC_RETVAL(func, scalar) typename Eigen::internal::func##_retval<typename Eigen::internal::global_math_functions_filtering_base<scalar>::type>::type

/****************************************************************************
* Implementation of real                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct real_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct real_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x)
  {
    using std::real;
    return real(x);
  }
};

template<typename Scalar> struct real_impl : real_default_impl<Scalar> {};

template<typename Scalar>
struct real_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};


/****************************************************************************
* Implementation of imag                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct imag_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar&)
  {
    return RealScalar(0);
  }
};

template<typename Scalar>
struct imag_default_impl<Scalar,true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x)
  {
    using std::imag;
    return imag(x);
  }
};

template<typename Scalar> struct imag_impl : imag_default_impl<Scalar> {};

template<typename Scalar>
struct imag_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of real_ref                                             *
****************************************************************************/

template<typename Scalar>
struct real_ref_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[0];
  }
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<const RealScalar*>(&x)[0];
  }
};

template<typename Scalar>
struct real_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of imag_ref                                             *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct imag_ref_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar& run(Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
  static inline const RealScalar& run(const Scalar& x)
  {
    return reinterpret_cast<RealScalar*>(&x)[1];
  }
};

template<typename Scalar>
struct imag_ref_default_impl<Scalar, false>
{
  static inline Scalar run(Scalar&)
  {
    return Scalar(0);
  }
  static inline const Scalar run(const Scalar&)
  {
    return Scalar(0);
  }
};

template<typename Scalar>
struct imag_ref_impl : imag_ref_default_impl<Scalar, NumTraits<Scalar>::IsComplex> {};

template<typename Scalar>
struct imag_ref_retval
{
  typedef typename NumTraits<Scalar>::Real & type;
};

/****************************************************************************
* Implementation of conj                                                 *
****************************************************************************/

template<typename Scalar, bool IsComplex = NumTraits<Scalar>::IsComplex>
struct conj_impl
{
  static inline Scalar run(const Scalar& x)
  {
    return x;
  }
};

template<typename Scalar>
struct conj_impl<Scalar,true>
{
  static inline Scalar run(const Scalar& x)
  {
    using std::conj;
    return conj(x);
  }
};

template<typename Scalar>
struct conj_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of abs2                                                 *
****************************************************************************/

template<typename Scalar>
struct abs2_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x)
  {
    return x*x;
  }
};

template<typename RealScalar>
struct abs2_impl<std::complex<RealScalar> >
{
  static inline RealScalar run(const std::complex<RealScalar>& x)
  {
    return real(x)*real(x) + imag(x)*imag(x);
  }
};

template<typename Scalar>
struct abs2_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of norm1                                                *
****************************************************************************/

template<typename Scalar, bool IsComplex>
struct norm1_default_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x)
  {
    using std::abs;
    return abs(real(x)) + abs(imag(x));
  }
};

template<typename Scalar>
struct norm1_default_impl<Scalar, false>
{
  static inline Scalar run(const Scalar& x)
  {
    using std::abs;
    return abs(x);
  }
};

template<typename Scalar>
struct norm1_impl : norm1_default_impl<Scalar, NumTraits<Scalar>::IsComplex> {};

template<typename Scalar>
struct norm1_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of hypot                                                *
****************************************************************************/

template<typename Scalar>
struct hypot_impl
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline RealScalar run(const Scalar& x, const Scalar& y)
  {
    using std::max;
    using std::min;
    using std::abs;
    using std::sqrt;
    RealScalar _x = abs(x);
    RealScalar _y = abs(y);
    RealScalar p = (max)(_x, _y);
    if(p==RealScalar(0)) return 0;
    RealScalar q = (min)(_x, _y);
    RealScalar qp = q/p;
    return p * sqrt(RealScalar(1) + qp*qp);
  }
};

template<typename Scalar>
struct hypot_retval
{
  typedef typename NumTraits<Scalar>::Real type;
};

/****************************************************************************
* Implementation of cast                                                 *
****************************************************************************/

template<typename OldType, typename NewType>
struct cast_impl
{
  static inline NewType run(const OldType& x)
  {
    return static_cast<NewType>(x);
  }
};

// here, for once, we're plainly returning NewType: we don't want cast to do weird things.

template<typename OldType, typename NewType>
inline NewType cast(const OldType& x)
{
  return cast_impl<OldType, NewType>::run(x);
}

/****************************************************************************
* Implementation of atanh2                                                *
****************************************************************************/

template<typename Scalar, bool IsInteger>
struct atanh2_default_impl
{
  typedef Scalar retval;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    using std::abs;
    using std::log;
    using std::sqrt;
    Scalar z = x / y;
    if (y == Scalar(0) || abs(z) > sqrt(NumTraits<RealScalar>::epsilon()))
      return RealScalar(0.5) * log((y + x) / (y - x));
    else
      return z + z*z*z / RealScalar(3);
  }
};

template<typename Scalar>
struct atanh2_default_impl<Scalar, true>
{
  static inline Scalar run(const Scalar&, const Scalar&)
  {
    EIGEN_STATIC_ASSERT_NON_INTEGER(Scalar)
    return Scalar(0);
  }
};

template<typename Scalar>
struct atanh2_impl : atanh2_default_impl<Scalar, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar>
struct atanh2_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of pow                                                  *
****************************************************************************/

template<typename Scalar, bool IsInteger>
struct pow_default_impl
{
  typedef Scalar retval;
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    using std::pow;
    return pow(x, y);
  }
};

template<typename Scalar>
struct pow_default_impl<Scalar, true>
{
  static inline Scalar run(Scalar x, Scalar y)
  {
    Scalar res(1);
    eigen_assert(!NumTraits<Scalar>::IsSigned || y >= 0);
    if(y & 1) res *= x;
    y >>= 1;
    while(y)
    {
      x *= x;
      if(y&1) res *= x;
      y >>= 1;
    }
    return res;
  }
};

template<typename Scalar>
struct pow_impl : pow_default_impl<Scalar, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar>
struct pow_retval
{
  typedef Scalar type;
};

/****************************************************************************
* Implementation of random                                               *
****************************************************************************/

template<typename Scalar,
         bool IsComplex,
         bool IsInteger>
struct random_default_impl {};

template<typename Scalar>
struct random_impl : random_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar>
struct random_retval
{
  typedef Scalar type;
};

template<typename Scalar> inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y);
template<typename Scalar> inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random();

template<typename Scalar>
struct random_default_impl<Scalar, false, false>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    return x + (y-x) * Scalar(std::rand()) / Scalar(RAND_MAX);
  }
  static inline Scalar run()
  {
    return run(Scalar(NumTraits<Scalar>::IsSigned ? -1 : 0), Scalar(1));
  }
};

enum {
  floor_log2_terminate,
  floor_log2_move_up,
  floor_log2_move_down,
  floor_log2_bogus
};

template<unsigned int n, int lower, int upper> struct floor_log2_selector
{
  enum { middle = (lower + upper) / 2,
         value = (upper <= lower + 1) ? int(floor_log2_terminate)
               : (n < (1 << middle)) ? int(floor_log2_move_down)
               : (n==0) ? int(floor_log2_bogus)
               : int(floor_log2_move_up)
  };
};

template<unsigned int n,
         int lower = 0,
         int upper = sizeof(unsigned int) * CHAR_BIT - 1,
         int selector = floor_log2_selector<n, lower, upper>::value>
struct floor_log2 {};

template<unsigned int n, int lower, int upper>
struct floor_log2<n, lower, upper, floor_log2_move_down>
{
  enum { value = floor_log2<n, lower, floor_log2_selector<n, lower, upper>::middle>::value };
};

template<unsigned int n, int lower, int upper>
struct floor_log2<n, lower, upper, floor_log2_move_up>
{
  enum { value = floor_log2<n, floor_log2_selector<n, lower, upper>::middle, upper>::value };
};

template<unsigned int n, int lower, int upper>
struct floor_log2<n, lower, upper, floor_log2_terminate>
{
  enum { value = (n >= ((unsigned int)(1) << (lower+1))) ? lower+1 : lower };
};

template<unsigned int n, int lower, int upper>
struct floor_log2<n, lower, upper, floor_log2_bogus>
{
  // no value, error at compile time
};

template<typename Scalar>
struct random_default_impl<Scalar, false, true>
{
  typedef typename NumTraits<Scalar>::NonInteger NonInteger;

  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    return x + Scalar((NonInteger(y)-x+1) * std::rand() / (RAND_MAX + NonInteger(1)));
  }

  static inline Scalar run()
  {
#ifdef EIGEN_MAKING_DOCS
    return run(Scalar(NumTraits<Scalar>::IsSigned ? -10 : 0), Scalar(10));
#else
    enum { rand_bits = floor_log2<(unsigned int)(RAND_MAX)+1>::value,
           scalar_bits = sizeof(Scalar) * CHAR_BIT,
           shift = EIGEN_PLAIN_ENUM_MAX(0, int(rand_bits) - int(scalar_bits)),
           offset = NumTraits<Scalar>::IsSigned ? (1 << (EIGEN_PLAIN_ENUM_MIN(rand_bits,scalar_bits)-1)) : 0
    };
    return Scalar((std::rand() >> shift) - offset);
#endif
  }
};

template<typename Scalar>
struct random_default_impl<Scalar, true, false>
{
  static inline Scalar run(const Scalar& x, const Scalar& y)
  {
    return Scalar(random(real(x), real(y)),
                  random(imag(x), imag(y)));
  }
  static inline Scalar run()
  {
    typedef typename NumTraits<Scalar>::Real RealScalar;
    return Scalar(random<RealScalar>(), random<RealScalar>());
  }
};

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run(x, y);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(random, Scalar) random()
{
  return EIGEN_MATHFUNC_IMPL(random, Scalar)::run();
}

} // end namespace internal

/****************************************************************************
* Generic math function                                                    *
****************************************************************************/

namespace numext {

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(real, Scalar) real(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(real, Scalar)::run(x);
}  

template<typename Scalar>
inline typename internal::add_const_on_value_type< EIGEN_MATHFUNC_RETVAL(real_ref, Scalar) >::type real_ref(const Scalar& x)
{
  return internal::real_ref_impl<Scalar>::run(x);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(real_ref, Scalar) real_ref(Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(real_ref, Scalar)::run(x);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(imag, Scalar) imag(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(imag, Scalar)::run(x);
}

template<typename Scalar>
inline typename internal::add_const_on_value_type< EIGEN_MATHFUNC_RETVAL(imag_ref, Scalar) >::type imag_ref(const Scalar& x)
{
  return internal::imag_ref_impl<Scalar>::run(x);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(imag_ref, Scalar) imag_ref(Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(imag_ref, Scalar)::run(x);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(conj, Scalar) conj(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(conj, Scalar)::run(x);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(abs2, Scalar) abs2(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(abs2, Scalar)::run(x);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(norm1, Scalar) norm1(const Scalar& x)
{
  return EIGEN_MATHFUNC_IMPL(norm1, Scalar)::run(x);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(hypot, Scalar) hypot(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(hypot, Scalar)::run(x, y);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(atanh2, Scalar) atanh2(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(atanh2, Scalar)::run(x, y);
}

template<typename Scalar>
inline EIGEN_MATHFUNC_RETVAL(pow, Scalar) pow(const Scalar& x, const Scalar& y)
{
  return EIGEN_MATHFUNC_IMPL(pow, Scalar)::run(x, y);
}

// std::isfinite is non standard, so let's define our own version,
// even though it is not very efficient.
template<typename T> bool (isfinite)(const T& x)
{
  return x<NumTraits<T>::highest() && x>NumTraits<T>::lowest();
}

} // end namespace numext

namespace internal {

/****************************************************************************
* Implementation of fuzzy comparisons                                       *
****************************************************************************/

template<typename Scalar,
         bool IsComplex,
         bool IsInteger>
struct scalar_fuzzy_default_impl {};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, false, false>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar>
  static inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y, const RealScalar& prec)
  {
    using std::abs;
    return abs(x) <= abs(y) * prec;
  }
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    using std::min;
    using std::abs;
    return abs(x - y) <= (min)(abs(x), abs(y)) * prec;
  }
  static inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    return x <= y || isApprox(x, y, prec);
  }
};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, false, true>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar>
  static inline bool isMuchSmallerThan(const Scalar& x, const Scalar&, const RealScalar&)
  {
    return x == Scalar(0);
  }
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar&)
  {
    return x == y;
  }
  static inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y, const RealScalar&)
  {
    return x <= y;
  }
};

template<typename Scalar>
struct scalar_fuzzy_default_impl<Scalar, true, false>
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  template<typename OtherScalar>
  static inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y, const RealScalar& prec)
  {
    return numext::abs2(x) <= numext::abs2(y) * prec * prec;
  }
  static inline bool isApprox(const Scalar& x, const Scalar& y, const RealScalar& prec)
  {
    using std::min;
    return numext::abs2(x - y) <= (min)(numext::abs2(x), numext::abs2(y)) * prec * prec;
  }
};

template<typename Scalar>
struct scalar_fuzzy_impl : scalar_fuzzy_default_impl<Scalar, NumTraits<Scalar>::IsComplex, NumTraits<Scalar>::IsInteger> {};

template<typename Scalar, typename OtherScalar>
inline bool isMuchSmallerThan(const Scalar& x, const OtherScalar& y,
                                   typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::template isMuchSmallerThan<OtherScalar>(x, y, precision);
}

template<typename Scalar>
inline bool isApprox(const Scalar& x, const Scalar& y,
                          typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::isApprox(x, y, precision);
}

template<typename Scalar>
inline bool isApproxOrLessThan(const Scalar& x, const Scalar& y,
                                    typename NumTraits<Scalar>::Real precision = NumTraits<Scalar>::dummy_precision())
{
  return scalar_fuzzy_impl<Scalar>::isApproxOrLessThan(x, y, precision);
}

/******************************************
***  The special case of the  bool type ***
******************************************/

template<> struct random_impl<bool>
{
  static inline bool run()
  {
    return random<int>(0,1)==0 ? false : true;
  }
};

template<> struct scalar_fuzzy_impl<bool>
{
  typedef bool RealScalar;
  
  template<typename OtherScalar>
  static inline bool isMuchSmallerThan(const bool& x, const bool&, const bool&)
  {
    return !x;
  }
  
  static inline bool isApprox(bool x, bool y, bool)
  {
    return x == y;
  }

  static inline bool isApproxOrLessThan(const bool& x, const bool& y, const bool&)
  {
    return (!x) || y;
  }
  
};

  
} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_MATHFUNCTIONS_H
