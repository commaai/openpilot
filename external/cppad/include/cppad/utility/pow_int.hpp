# ifndef CPPAD_UTILITY_POW_INT_HPP
# define CPPAD_UTILITY_POW_INT_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */

/*
-------------------------------------------------------------------------------
$begin pow_int$$
$spell
	cppad.hpp
	CppAD
	namespace
	const
$$


$section The Integer Power Function$$
$mindex pow exponent$$

$head Syntax$$
$codei%# include <cppad/utility/pow_int.hpp>
%$$
$icode%z% = pow(%x%, %y%)%$$

$head See Also$$
$cref pow$$

$head Purpose$$
Determines the value of the power function
$latex \[
	{\rm pow} (x, y) = x^y
\] $$
for integer exponents $icode n$$
using multiplication and possibly division to compute the value.
The other CppAD $cref pow$$ function may use logarithms and exponentiation
to compute derivatives of the same value
(which will not work if $icode x$$ is less than or equal zero).

$head Include$$
The file $code cppad/pow_int.h$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.
Including this file defines
this version of the $code pow$$ within the $code CppAD$$ namespace.

$head x$$
The argument $icode x$$ has prototype
$codei%
	const %Type%& %x%
%$$

$head y$$
The argument $icode y$$ has prototype
$codei%
	const int& %y%
%$$

$head z$$
The result $icode z$$ has prototype
$codei%
	%Type% %z%
%$$

$head Type$$
The type $icode Type$$ must support the following operations
where $icode a$$ and $icode b$$ are $icode Type$$ objects
and $icode i$$ is an $code int$$:
$table
$bold Operation$$  $pre  $$
	$cnext $bold Description$$
	$cnext $bold Result Type$$
$rnext
$icode%Type% %a%(%i%)%$$
	$cnext construction of a $icode Type$$ object from an $code int$$
	$cnext $icode Type$$
$rnext
$icode%a% * %b%$$
	$cnext binary multiplication of $icode Type$$ objects
	$cnext $icode Type$$
$rnext
$icode%a% / %b%$$
	$cnext binary division of $icode Type$$ objects
	$cnext $icode Type$$
$tend

$head Operation Sequence$$
The $icode Type$$ operation sequence used to calculate $icode z$$ is
$cref/independent/glossary/Operation/Independent/$$
of $icode x$$.

$head Example$$
$children%
	example/general/pow_int.cpp
%$$
The file $cref pow_int.cpp$$
is an example and test of this function.
It returns true if it succeeds and false otherwise.


$end
-------------------------------------------------------------------------------
*/

namespace CppAD {

	template <class Type>
	inline Type pow (const Type& x, const int& n)
	{
		Type p(1);
		int n2 = n / 2;

		if( n == 0 )
			return p;
		if( n < 0 )
			return p / pow(x, -n);
		if( n == 1 )
			return x;

		// p = (x^2)^(n/2)
		p = pow( x * x , n2 );

		// n is even case
		if( n % 2 == 0 )
			return p;

		// n is odd case
		return p * x;
	}

}

# endif
