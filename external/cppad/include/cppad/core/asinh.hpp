# ifndef CPPAD_CORE_ASINH_HPP
# define CPPAD_CORE_ASINH_HPP

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

$begin asinh$$
$spell
	asinh
	const
	Vec
	std
	cmath
	CppAD
$$
$section The Inverse Hyperbolic Sine Function: asinh$$

$head Syntax$$
$icode%y% = asinh(%x%)%$$

$head Description$$
The inverse hyperbolic sine function is defined by
$icode%x% == sinh(%y%)%$$.

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
\R{asinh} (x) = \log \left( x + \sqrt{ 1 + x^2 } \right)
\] $$
to compute this function.

$head Example$$
$children%
	example/general/asinh.cpp
%$$
The file
$cref asinh.cpp$$
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
Type asinh_template(const Type &x)
{	return CppAD::log( x + CppAD::sqrt( Type(1) + x * x ) );
}

inline float asinh(const float &x)
{	return asinh_template(x); }

inline double asinh(const double &x)
{	return asinh_template(x); }

template <class Base>
inline AD<Base> asinh(const AD<Base> &x)
{	return asinh_template(x); }

template <class Base>
inline AD<Base> asinh(const VecAD_reference<Base> &x)
{	return asinh_template( x.ADBase() ); }


} // END CppAD namespace

# endif // CPPAD_USE_CPLUSPLUS_2011
# endif // CPPAD_ASINH_INCLUDED
