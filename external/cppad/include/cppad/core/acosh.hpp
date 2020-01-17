# ifndef CPPAD_CORE_ACOSH_HPP
# define CPPAD_CORE_ACOSH_HPP

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
$begin acosh$$
$spell
	acosh
	const
	Vec
	std
	cmath
	CppAD
$$
$section The Inverse Hyperbolic Cosine Function: acosh$$

$head Syntax$$
$icode%y% = acosh(%x%)%$$

$head Description$$
The inverse hyperbolic cosine function is defined by
$icode%x% == cosh(%y%)%$$.

$head x, y$$
See the $cref/possible types/unary_standard_math/Possible Types/$$
for a unary standard math function.

$head CPPAD_USE_CPLUSPLUS_2011$$

$subhead true$$
If this preprocessor symbol is true ($code 1$$),
and $icode x$$ is an AD type,
this is an $cref/atomic operation/glossary/Operation/Atomic/$$.

$subhead false$$
If this preprocessor symbol is false ($code 0$$),
CppAD uses the representation
$latex \[
\R{acosh} (x) = \log \left( x + \sqrt{ x^2 - 1 } \right)
\] $$
to compute this function.

$head Example$$
$children%
	example/general/acosh.cpp
%$$
The file
$cref acosh.cpp$$
contains an example and test of this function.
It returns true if it succeeds and false otherwise.

$end
-------------------------------------------------------------------------------
*/
# include <cppad/configure.hpp>
# if ! CPPAD_USE_CPLUSPLUS_2011

// BEGIN CppAD namespace
namespace CppAD {

template <class Type>
Type acosh_template(const Type &x)
{	return CppAD::log( x + CppAD::sqrt( x * x - Type(1) ) );
}

inline float acosh(const float &x)
{	return acosh_template(x); }

inline double acosh(const double &x)
{	return acosh_template(x); }

template <class Base>
inline AD<Base> acosh(const AD<Base> &x)
{	return acosh_template(x); }

template <class Base>
inline AD<Base> acosh(const VecAD_reference<Base> &x)
{	return acosh_template( x.ADBase() ); }


} // END CppAD namespace

# endif // CPPAD_USE_CPLUSPLUS_2011
# endif // CPPAD_ACOSH_INCLUDED
