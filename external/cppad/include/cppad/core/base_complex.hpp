# ifndef CPPAD_CORE_BASE_COMPLEX_HPP
# define CPPAD_CORE_BASE_COMPLEX_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/configure.hpp>
# include <limits>
# include <complex>

// needed before one can use CPPAD_ASSERT_FIRST_CALL_NOT_PARALLEL
# include <cppad/utility/thread_alloc.hpp>

/*
$begin base_complex.hpp$$
$spell
	azmul
	expm1
	atanh
	acosh
	asinh
	endif
	eps
	abs_geq
	Rel
	Lt Le Eq Ge Gt
	imag
	gcc
	isnan
	cppad.hpp
	sqrt
	exp
	cos
	std
	const
	CppAD
	Op
	inline
	enum
	undef
	acos
	asin
	atan
	erf
	Cond
	namespace
	bool
$$


$section Enable use of AD<Base> where Base is std::complex<double>$$

$children%example/general/complex_poly.cpp
%$$
$head Example$$
The file $cref complex_poly.cpp$$ contains an example use of
$code std::complex<double>$$ type for a CppAD $icode Base$$ type.
It returns true if it succeeds and false otherwise.

$head Include Order$$
This file is included before $code <cppad/cppad.hpp>$$
so it is necessary to define the error handler
in addition to including
$cref/base_require.hpp/base_require/Include Order/$$
$srccode%cpp% */
# include <limits>
# include <complex>
# include <cppad/base_require.hpp>
# include <cppad/core/cppad_assert.hpp>

/* %$$

$head CondExpOp$$
The type $code std::complex<double>$$ does not supports the
$code <$$, $code <=$$, $code ==$$, $code >=$$, and $code >$$ operators; see
$cref/not ordered/base_cond_exp/CondExpTemplate/Not Ordered/$$.
Hence its $code CondExpOp$$ function is defined by
$srccode%cpp% */
namespace CppAD {
	inline std::complex<double> CondExpOp(
		enum CppAD::CompareOp      cop        ,
		const std::complex<double> &left      ,
		const std::complex<double> &right     ,
		const std::complex<double> &trueCase  ,
		const std::complex<double> &falseCase )
	{	CppAD::ErrorHandler::Call(
			true     , __LINE__ , __FILE__ ,
			"std::complex<float> CondExpOp(...)",
			"Error: cannot use CondExp with a complex type"
		);
		return std::complex<double>(0);
	}
}
/* %$$

$head CondExpRel$$
The $cref/CPPAD_COND_EXP_REL/base_cond_exp/CondExpRel/$$ macro invocation
$srccode%cpp% */
namespace CppAD {
	CPPAD_COND_EXP_REL( std::complex<double> )
}
/* %$$
used $code CondExpOp$$ above to
define $codei%CondExp%Rel%$$ for $code std::complex<double>$$ arguments
and $icode%Rel%$$ equal to
$code Lt$$, $code Le$$, $code Eq$$, $code Ge$$, and $code Gt$$.

$head EqualOpSeq$$
Complex numbers do not carry operation sequence information.
Thus they are equal in this sense if and only if there values are equal.
$srccode%cpp% */
namespace CppAD {
	inline bool EqualOpSeq(
		const std::complex<double> &x ,
		const std::complex<double> &y )
	{	return x == y;
	}
}
/* %$$

$head Identical$$
Complex numbers do not carry operation sequence information.
Thus they are all parameters so the identical functions just check values.
$srccode%cpp% */
namespace CppAD {
	inline bool IdenticalPar(const std::complex<double> &x)
	{	return true; }
	inline bool IdenticalZero(const std::complex<double> &x)
	{	return (x == std::complex<double>(0., 0.) ); }
	inline bool IdenticalOne(const std::complex<double> &x)
	{	return (x == std::complex<double>(1., 0.) ); }
	inline bool IdenticalEqualPar(
		const std::complex<double> &x, const std::complex<double> &y)
	{	return (x == y); }
}
/* %$$

$head Ordered$$
Complex types do not support comparison operators,
$srccode%cpp% */
# undef  CPPAD_USER_MACRO
# define CPPAD_USER_MACRO(Fun)                                     \
inline bool Fun(const std::complex<double>& x)                     \
{      CppAD::ErrorHandler::Call(                                  \
               true     , __LINE__ , __FILE__ ,                    \
               #Fun"(x)",                                          \
               "Error: cannot use " #Fun " with x complex<double> " \
       );                                                          \
       return false;                                               \
}
namespace CppAD {
	CPPAD_USER_MACRO(LessThanZero)
	CPPAD_USER_MACRO(LessThanOrZero)
	CPPAD_USER_MACRO(GreaterThanOrZero)
	CPPAD_USER_MACRO(GreaterThanZero)
	inline bool abs_geq(
		const std::complex<double>& x ,
		const std::complex<double>& y )
	{	return std::abs(x) >= std::abs(y); }
}
/* %$$

$head Integer$$
The implementation of this function must agree
with the CppAD user specifications for complex arguments to the
$cref/Integer/Integer/x/Complex Types/$$ function:
$srccode%cpp% */
namespace CppAD {
	inline int Integer(const std::complex<double> &x)
	{	return static_cast<int>( x.real() ); }
}
/* %$$

$head azmul$$
$srccode%cpp% */
namespace CppAD {
	CPPAD_AZMUL( std::complex<double> )
}
/* %$$

$head isnan$$
The gcc 4.1.1 complier defines the function
$codei%
	int std::complex<double>::isnan( std::complex<double> %z% )
%$$
(which is not specified in the C++ 1998 standard ISO/IEC 14882).
This causes an ambiguity between the function above and the CppAD
$cref/isnan/nan/$$ template function.
We avoid this ambiguity by defining a non-template version of
this function in the CppAD namespace.
$srccode%cpp% */
namespace CppAD {
	inline bool isnan(const std::complex<double>& z)
	{	return (z != z);
	}
}
/* %$$

$head Valid Unary Math$$
The following macro invocations define the standard unary
math functions that are valid with complex arguments and are
required to use $code AD< std::complex<double> >$$.
$srccode%cpp% */
namespace CppAD {
	CPPAD_STANDARD_MATH_UNARY(std::complex<double>, cos)
	CPPAD_STANDARD_MATH_UNARY(std::complex<double>, cosh)
	CPPAD_STANDARD_MATH_UNARY(std::complex<double>, exp)
	CPPAD_STANDARD_MATH_UNARY(std::complex<double>, log)
	CPPAD_STANDARD_MATH_UNARY(std::complex<double>, sin)
	CPPAD_STANDARD_MATH_UNARY(std::complex<double>, sinh)
	CPPAD_STANDARD_MATH_UNARY(std::complex<double>, sqrt)
}
/* %$$

$head Invalid Unary Math$$
The following macro definition and invocations define the standard unary
math functions that are invalid with complex arguments and are
required to use $code AD< std::complex<double> >$$.
$srccode%cpp% */
# undef  CPPAD_USER_MACRO
# define CPPAD_USER_MACRO(Fun)                                     \
inline std::complex<double> Fun(const std::complex<double>& x)     \
{      CppAD::ErrorHandler::Call(                                  \
               true     , __LINE__ , __FILE__ ,                    \
               #Fun"(x)",                                          \
               "Error: cannot use " #Fun " with x complex<double> " \
       );                                                          \
       return std::complex<double>(0);                             \
}
namespace CppAD {
	CPPAD_USER_MACRO(abs)
	CPPAD_USER_MACRO(fabs)
	CPPAD_USER_MACRO(acos)
	CPPAD_USER_MACRO(asin)
	CPPAD_USER_MACRO(atan)
	CPPAD_USER_MACRO(sign)
# if CPPAD_USE_CPLUSPLUS_2011
	CPPAD_USER_MACRO(erf)
	CPPAD_USER_MACRO(asinh)
	CPPAD_USER_MACRO(acosh)
	CPPAD_USER_MACRO(atanh)
	CPPAD_USER_MACRO(expm1)
	CPPAD_USER_MACRO(log1p)
# endif
}
/* %$$

$head pow $$
The following defines a $code CppAD::pow$$ function that
is required to use $code AD< std::complex<double> >$$:
$srccode%cpp% */
namespace CppAD {
	inline std::complex<double> pow(
		const std::complex<double> &x ,
		const std::complex<double> &y )
	{	return std::pow(x, y); }
}
/* %$$

$head numeric_limits$$
The following defines the CppAD $cref numeric_limits$$
for the type $code std::complex<double>$$:
$srccode%cpp% */
namespace CppAD {
	CPPAD_NUMERIC_LIMITS(double, std::complex<double>)
}
/* %$$

$head to_string$$
The following defines the function CppAD $cref to_string$$
for the type $code std::complex<double>$$:
$srccode%cpp% */
namespace CppAD {
	CPPAD_TO_STRING(std::complex<double>)
}
/* %$$
$end
*/
# undef  CPPAD_USER_MACRO_ONE
# define CPPAD_USER_MACRO_ONE(Fun)                                 \
inline bool Fun(const std::complex<float>& x)                      \
{      CppAD::ErrorHandler::Call(                                  \
               true     , __LINE__ , __FILE__ ,                    \
               #Fun"(x)",                                          \
               "Error: cannot use " #Fun " with x complex<float> " \
       );                                                          \
       return false;                                               \
}
# undef  CPPAD_USER_MACRO_TWO
# define CPPAD_USER_MACRO_TWO(Fun)                                 \
inline std::complex<float> Fun(const std::complex<float>& x)       \
{      CppAD::ErrorHandler::Call(                                  \
               true     , __LINE__ , __FILE__ ,                    \
               #Fun"(x)",                                          \
               "Error: cannot use " #Fun " with x complex<float> " \
       );                                                          \
       return std::complex<float>(0);                              \
}
namespace CppAD {
	// CondExpOp ------------------------------------------------------
	inline std::complex<float> CondExpOp(
		enum CppAD::CompareOp      cop       ,
		const std::complex<float> &left      ,
		const std::complex<float> &right     ,
		const std::complex<float> &trueCase  ,
		const std::complex<float> &falseCase )
	{	CppAD::ErrorHandler::Call(
			true     , __LINE__ , __FILE__ ,
			"std::complex<float> CondExpOp(...)",
			"Error: cannot use CondExp with a complex type"
		);
		return std::complex<float>(0);
	}
	// CondExpRel --------------------------------------------------------
	CPPAD_COND_EXP_REL( std::complex<float> )
	// EqualOpSeq -----------------------------------------------------
	inline bool EqualOpSeq(
		const std::complex<float> &x ,
		const std::complex<float> &y )
	{	return x == y;
	}
	// Identical ------------------------------------------------------
	inline bool IdenticalPar(const std::complex<float> &x)
	{	return true; }
	inline bool IdenticalZero(const std::complex<float> &x)
	{	return (x == std::complex<float>(0., 0.) ); }
	inline bool IdenticalOne(const std::complex<float> &x)
	{	return (x == std::complex<float>(1., 0.) ); }
	inline bool IdenticalEqualPar(
		const std::complex<float> &x, const std::complex<float> &y)
	{	return (x == y); }
	// Ordered --------------------------------------------------------
	CPPAD_USER_MACRO_ONE(LessThanZero)
	CPPAD_USER_MACRO_ONE(LessThanOrZero)
	CPPAD_USER_MACRO_ONE(GreaterThanOrZero)
	CPPAD_USER_MACRO_ONE(GreaterThanZero)
	inline bool abs_geq(
		const std::complex<float>& x ,
		const std::complex<float>& y )
	{	return std::abs(x) >= std::abs(y); }
	// Integer ------------------------------------------------------
	inline int Integer(const std::complex<float> &x)
	{	return static_cast<int>( x.real() ); }
	// isnan -------------------------------------------------------------
	inline bool isnan(const std::complex<float>& z)
	{	return (z != z);
	}
	// Valid standard math functions --------------------------------
	CPPAD_STANDARD_MATH_UNARY(std::complex<float>, cos)
	CPPAD_STANDARD_MATH_UNARY(std::complex<float>, cosh)
	CPPAD_STANDARD_MATH_UNARY(std::complex<float>, exp)
	CPPAD_STANDARD_MATH_UNARY(std::complex<float>, log)
	CPPAD_STANDARD_MATH_UNARY(std::complex<float>, sin)
	CPPAD_STANDARD_MATH_UNARY(std::complex<float>, sinh)
	CPPAD_STANDARD_MATH_UNARY(std::complex<float>, sqrt)
	// Invalid standrd math functions -------------------------------
	CPPAD_USER_MACRO_TWO(abs)
	CPPAD_USER_MACRO_TWO(acos)
	CPPAD_USER_MACRO_TWO(asin)
	CPPAD_USER_MACRO_TWO(atan)
	CPPAD_USER_MACRO_TWO(sign)
	// The pow function
	inline std::complex<float> pow(
		const std::complex<float> &x ,
		const std::complex<float> &y )
	{	return std::pow(x, y); }
	// numeric_limits -------------------------------------------------
	CPPAD_NUMERIC_LIMITS(float, std::complex<float>)
	// to_string -------------------------------------------------
	CPPAD_TO_STRING(std::complex<float>)
}

// undefine macros only used by this file
# undef CPPAD_USER_MACRO
# undef CPPAD_USER_MACRO_ONE
# undef CPPAD_USER_MACRO_TWO

# endif
