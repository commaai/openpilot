# ifndef CPPAD_UTILITY_NEAR_EQUAL_HPP
# define CPPAD_UTILITY_NEAR_EQUAL_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
$begin NearEqual$$
$spell
	cppad.hpp
	sqrt
	cout
	endl
	Microsoft
	std
	Cpp
	namespace
	const
	bool
$$

$section Determine if Two Values Are Nearly Equal$$
$mindex NearEqual near absolute difference relative$$


$head Syntax$$

$codei%# include <cppad/utility/near_equal.hpp>
%$$
$icode%b% = NearEqual(%x%, %y%, %r%, %a%)%$$


$head Purpose$$
Returns true,
if $icode x$$ and $icode y$$ are nearly equal,
and false otherwise.

$head x$$
The argument $icode x$$
has one of the following possible prototypes
$codei%
	const %Type%               &%x%,
	const std::complex<%Type%> &%x%,
%$$

$head y$$
The argument $icode y$$
has one of the following possible prototypes
$codei%
	const %Type%               &%y%,
	const std::complex<%Type%> &%y%,
%$$

$head r$$
The relative error criteria $icode r$$ has prototype
$codei%
	const %Type% &%r%
%$$
It must be greater than or equal to zero.
The relative error condition is defined as:
$latex \[
	| x - y | \leq r ( |x| + |y| )
\] $$

$head a$$
The absolute error criteria $icode a$$ has prototype
$codei%
	const %Type% &%a%
%$$
It must be greater than or equal to zero.
The absolute error condition is defined as:
$latex \[
	| x - y | \leq a
\] $$

$head b$$
The return value $icode b$$ has prototype
$codei%
	bool %b%
%$$
If either $icode x$$ or $icode y$$ is infinite or not a number,
the return value is false.
Otherwise, if either the relative or absolute error
condition (defined above) is satisfied, the return value is true.
Otherwise, the return value is false.

$head Type$$
The type $icode Type$$ must be a
$cref NumericType$$.
The routine $cref CheckNumericType$$ will generate
an error message if this is not the case.
In addition, the following operations must be defined objects
$icode a$$ and $icode b$$ of type $icode Type$$:
$table
$bold Operation$$     $cnext
	$bold Description$$ $rnext
$icode%a% <= %b%$$  $cnext
	less that or equal operator (returns a $code bool$$ object)
$tend

$head Include Files$$
The file $code cppad/near_equal.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.

$head Example$$
$children%
	example/utility/near_equal.cpp
%$$
The file $cref near_equal.cpp$$ contains an example
and test of $code NearEqual$$.
It return true if it succeeds and false otherwise.

$head Exercise$$
Create and run a program that contains the following code:
$codep
	using std::complex;
	using std::cout;
	using std::endl;

	complex<double> one(1., 0), i(0., 1);
	complex<double> x = one / i;
	complex<double> y = - i;
	double          r = 1e-12;
	double          a = 0;
	bool           ok = CppAD::NearEqual(x, y, r, a);
	if( ok )
		cout << "Ok"    << endl;
	else	cout << "Error" << endl;
$$

$end

*/

# include <limits>
# include <complex>
# include <cppad/core/cppad_assert.hpp>
# include <cppad/utility/check_numeric_type.hpp>

namespace CppAD { // Begin CppAD namespace

// determine if both x and y are finite values
template <class Type>
bool near_equal_isfinite(const Type &x , const Type &y)
{	Type infinity = Type( std::numeric_limits<double>::infinity() );

	// handle bug where some compilers return true for nan == nan
	bool xNan = x != x;
	bool yNan = y != y;

	// infinite cases
	bool xInf = (x == infinity   || x == - infinity);
	bool yInf = (x == infinity   || x == - infinity);

	return ! (xNan | yNan | xInf | yInf);
}

template <class Type>
bool NearEqual(const Type &x, const Type &y, const Type &r, const Type &a)
{
	CheckNumericType<Type>();
	Type zero(0);

	CPPAD_ASSERT_KNOWN(
		zero <= r,
		"Error in NearEqual: relative error is less than zero"
	);
	CPPAD_ASSERT_KNOWN(
		zero <= a,
		"Error in NearEqual: absolute error is less than zero"
	);

	// check for special cases
	if( ! CppAD::near_equal_isfinite(x, y) )
		return false;

	Type ax = x;
	if( ax <= zero )
		ax = - ax;

	Type ay = y;
	if( ay <= zero )
		ay = - ay;

	Type ad = x - y;
	if( ad <= zero )
		ad = - ad;

	if( ad <= a )
		return true;

	if( ad <= r * (ax + ay) )
		return true;

	return false;
}

template <class Type>
bool NearEqual(
	const std::complex<Type> &x ,
	const std::complex<Type> &y ,
	const              Type  &r ,
	const              Type  & a )
{
	CheckNumericType<Type>();
# ifndef NDEBUG
	Type zero(0);
# endif

	CPPAD_ASSERT_KNOWN(
		zero <= r,
		"Error in NearEqual: relative error is less than zero"
	);
	CPPAD_ASSERT_KNOWN(
		zero <= a,
		"Error in NearEqual: absolute error is less than zero"
	);

	// check for special cases
	if( ! CppAD::near_equal_isfinite(x.real(), x.imag()) )
		return false;
	if( ! CppAD::near_equal_isfinite(y.real(), y.imag()) )
		return false;

	std::complex<Type> d = x - y;

	Type ad = std::abs(d);
	if( ad <= a )
		return true;

	Type ax = std::abs(x);
	Type ay = std::abs(y);
	if( ad <= r * (ax + ay) )
		return true;

	return false;
}

template <class Type>
bool NearEqual(
	const std::complex<Type> &x ,
	const              Type  &y ,
	const              Type  &r ,
	const              Type  & a )
{
	return NearEqual(x, std::complex<Type>(y, Type(0)), r, a);
}

template <class Type>
bool NearEqual(
	const              Type  &x ,
	const std::complex<Type> &y ,
	const              Type  &r ,
	const              Type  & a )
{
	return NearEqual(std::complex<Type>(x, Type(0)), y, r, a);
}

} // END CppAD namespace

# endif
