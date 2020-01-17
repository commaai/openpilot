// $Id$
# ifndef CPPAD_CORE_BASE_DOUBLE_HPP
# define CPPAD_CORE_BASE_DOUBLE_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
# include <cppad/configure.hpp>
# include <limits>

/*
$begin base_double.hpp$$
$spell
	namespaces
	cppad
	hpp
	azmul
	expm1
	atanh
	acosh
	asinh
	erf
	endif
	abs_geq
	acos
	asin
	atan
	cos
	sqrt
	tanh
	std
	fabs
	bool
	Lt Le Eq Ge Gt
	Rel
	CppAD
	CondExpOp
	namespace
	inline
	enum
	const
	exp
	const
$$


$section Enable use of AD<Base> where Base is double$$

$head CondExpOp$$
The type $code double$$ is a relatively simple type that supports
$code <$$, $code <=$$, $code ==$$, $code >=$$, and $code >$$ operators; see
$cref/ordered type/base_cond_exp/CondExpTemplate/Ordered Type/$$.
Hence its $code CondExpOp$$ function is defined by
$srccode%cpp% */
namespace CppAD {
	inline double CondExpOp(
		enum CompareOp     cop          ,
		const double&       left         ,
		const double&       right        ,
		const double&       exp_if_true  ,
		const double&       exp_if_false )
	{	return CondExpTemplate(cop, left, right, exp_if_true, exp_if_false);
	}
}
/* %$$

$head CondExpRel$$
The $cref/CPPAD_COND_EXP_REL/base_cond_exp/CondExpRel/$$ macro invocation
$srccode%cpp% */
namespace CppAD {
	CPPAD_COND_EXP_REL(double)
}
/* %$$
uses $code CondExpOp$$ above to
define $codei%CondExp%Rel%$$ for $code double$$ arguments
and $icode%Rel%$$ equal to
$code Lt$$, $code Le$$, $code Eq$$, $code Ge$$, and $code Gt$$.

$head EqualOpSeq$$
The type $code double$$ is simple (in this respect) and so we define
$srccode%cpp% */
namespace CppAD {
	inline bool EqualOpSeq(const double& x, const double& y)
	{	return x == y; }
}
/* %$$

$head Identical$$
The type $code double$$ is simple (in this respect) and so we define
$srccode%cpp% */
namespace CppAD {
	inline bool IdenticalPar(const double& x)
	{	return true; }
	inline bool IdenticalZero(const double& x)
	{	return (x == 0.); }
	inline bool IdenticalOne(const double& x)
	{	return (x == 1.); }
	inline bool IdenticalEqualPar(const double& x, const double& y)
	{	return (x == y); }
}
/* %$$

$head Integer$$
$srccode%cpp% */
namespace CppAD {
	inline int Integer(const double& x)
	{	return static_cast<int>(x); }
}
/* %$$

$head azmul$$
$srccode%cpp% */
namespace CppAD {
	CPPAD_AZMUL( double )
}
/* %$$

$head Ordered$$
The $code double$$ type supports ordered comparisons
$srccode%cpp% */
namespace CppAD {
	inline bool GreaterThanZero(const double& x)
	{	return x > 0.; }
	inline bool GreaterThanOrZero(const double& x)
	{	return x >= 0.; }
	inline bool LessThanZero(const double& x)
	{	return x < 0.; }
	inline bool LessThanOrZero(const double& x)
	{	return x <= 0.; }
	inline bool abs_geq(const double& x, const double& y)
	{	return std::fabs(x) >= std::fabs(y); }
}
/* %$$

$head Unary Standard Math$$
The following macro invocations import the $code double$$ versions of
the unary standard math functions into the $code CppAD$$ namespace.
Importing avoids ambiguity errors when using both the
$code CppAD$$ and $code std$$ namespaces.
Note this also defines the $cref/float/base_float.hpp/Unary Standard Math/$$
versions of these functions.
$srccode%cpp% */
namespace CppAD {
	using std::acos;
	using std::asin;
	using std::atan;
	using std::cos;
	using std::cosh;
	using std::exp;
	using std::fabs;
	using std::log;
	using std::log10;
	using std::sin;
	using std::sinh;
	using std::sqrt;
	using std::tan;
	using std::tanh;
# if CPPAD_USE_CPLUSPLUS_2011
	using std::erf;
	using std::asinh;
	using std::acosh;
	using std::atanh;
	using std::expm1;
	using std::log1p;
# endif
}
/* %$$
The absolute value function is special because its $code std$$ name is
$code fabs$$
$srccode%cpp% */
namespace CppAD {
	inline double abs(const double& x)
	{	return std::fabs(x); }
}
/* %$$

$head sign$$
The following defines the $code CppAD::sign$$ function that
is required to use $code AD<double>$$:
$srccode%cpp% */
namespace CppAD {
	inline double sign(const double& x)
	{	if( x > 0. )
			return 1.;
		if( x == 0. )
			return 0.;
		return -1.;
	}
}
/* %$$

$head pow$$
The following defines a $code CppAD::pow$$ function that
is required to use $code AD<double>$$.
As with the unary standard math functions,
this has the exact same signature as $code std::pow$$,
so use it instead of defining another function.
$srccode%cpp% */
namespace CppAD {
	using std::pow;
}
/* %$$

$head numeric_limits$$
The following defines the CppAD $cref numeric_limits$$
for the type $code double$$:
$srccode%cpp% */
namespace CppAD {
	CPPAD_NUMERIC_LIMITS(double, double)
}
/* %$$

$head to_string$$
There is no need to define $code to_string$$ for $code double$$
because it is defined by including $code cppad/utility/to_string.hpp$$;
see $cref to_string$$.
See $cref/base_complex.hpp/base_complex.hpp/to_string/$$ for an example where
it is necessary to define $code to_string$$ for a $icode Base$$ type.

$end
*/

# endif
