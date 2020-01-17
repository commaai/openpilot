# ifndef CPPAD_UTILITY_POLY_HPP
# define CPPAD_UTILITY_POLY_HPP

/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-17 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*
$begin Poly$$
$spell
	cppad.hpp
	CppAD
	namespace
	cstddef
	ifndef
	endif
	deg
	const
	std
	da
$$


$section Evaluate a Polynomial or its Derivative$$
$mindex Poly template$$

$head Syntax$$
$codei%# include <cppad/utility/poly.hpp>
%$$
$icode%p% = Poly(%k%, %a%, %z%)%$$


$head Description$$
Computes the $th k$$ derivative of the polynomial
$latex \[
	P(z) = a_0 + a_1 z^1 + \cdots + a_d z^d
\] $$
If $icode k$$ is equal to zero, the return value is $latex P(z)$$.

$head Include$$
The file $code cppad/poly.hpp$$ is included by $code cppad/cppad.hpp$$
but it can also be included separately with out the rest of
the $code CppAD$$ routines.
Including this file defines
$code Poly$$ within the $code CppAD$$ namespace.

$head k$$
The argument $icode k$$ has prototype
$codei%
	size_t %k%
%$$
It specifies the order of the derivative to calculate.

$head a$$
The argument $icode a$$ has prototype
$codei%
	const %Vector% &%a%
%$$
(see $cref/Vector/Poly/Vector/$$ below).
It specifies the vector corresponding to the polynomial $latex P(z)$$.

$head z$$
The argument $icode z$$ has prototype
$codei%
	const %Type% &%z%
%$$
(see $icode Type$$ below).
It specifies the point at which to evaluate the polynomial

$head p$$
The result $icode p$$  has prototype
$codei%
	%Type% %p%
%$$
(see $cref/Type/Poly/Type/$$ below)
and it is equal to the $th k$$ derivative of $latex P(z)$$; i.e.,
$latex \[
p = \frac{k !}{0 !} a_k
  + \frac{(k+1) !}{1 !} a_{k+1} z^1
  + \ldots
  + \frac{d !}{(d - k) !} a_d z^{d - k}
\]
$$
If $latex k > d$$, $icode%p% = %Type%(0)%$$.

$head Type$$
The type $icode Type$$ is determined by the argument $icode z$$.
It is assumed that
multiplication and addition of $icode Type$$ objects
are commutative.

$subhead Operations$$
The following operations must be supported where
$icode x$$ and $icode y$$ are objects of type $icode Type$$
and $icode i$$ is an $code int$$:
$table
$icode%x%  = %i%$$   $cnext assignment     $rnext
$icode%x%  = %y%$$   $cnext assignment     $rnext
$icode%x% *= %y%$$   $cnext multiplication compound assignment $rnext
$icode%x% += %y%$$   $cnext addition compound assignment

$tend


$head Vector$$
The type $icode Vector$$ must be a $cref SimpleVector$$ class with
$cref/elements of type/SimpleVector/Elements of Specified Type/$$
$icode Type$$.
The routine $cref CheckSimpleVector$$ will generate an error message
if this is not the case.

$head Operation Sequence$$
The $icode Type$$ operation sequence used to calculate $icode p$$ is
$cref/independent/glossary/Operation/Independent/$$
of $icode z$$ and the elements of $icode a$$
(it does depend on the size of the vector $icode a$$).


$children%
	example/general/poly.cpp%
	omh/poly_hpp.omh
%$$

$head Example$$
The file
$cref poly.cpp$$
contains an example and test of this routine.
It returns true if it succeeds and false otherwise.

$head Source$$
The file $cref poly.hpp$$ contains the
current source code that implements these specifications.

$end
------------------------------------------------------------------------------
*/
// BEGIN C++
# include <cstddef>  // used to defined size_t
# include <cppad/utility/check_simple_vector.hpp>

namespace CppAD {    // BEGIN CppAD namespace

template <class Type, class Vector>
Type Poly(size_t k, const Vector &a, const Type &z)
{	size_t i;
	size_t d = a.size() - 1;

	Type tmp;

	// check Vector is Simple Vector class with Type elements
	CheckSimpleVector<Type, Vector>();

	// case where derivative order greater than degree of polynomial
	if( k > d )
	{	tmp = 0;
		return tmp;
	}
	// case where we are evaluating a derivative
	if( k > 0 )
	{	// initialize factor as (k-1) !
		size_t factor = 1;
		for(i = 2; i < k; i++)
			factor *= i;

		// set b to coefficient vector corresponding to derivative
		Vector b(d - k + 1);
		for(i = k; i <= d; i++)
		{	factor   *= i;
			tmp       = double( factor );
			b[i - k]  = a[i] * tmp;
			factor   /= (i - k + 1);
		}
		// value of derivative polynomial
		return Poly(0, b, z);
	}
	// case where we are evaluating the original polynomial
	Type sum = a[d];
	i        = d;
	while(i > 0)
	{	sum *= z;
		sum += a[--i];
	}
	return sum;
}
} // END CppAD namespace
// END C++
# endif
